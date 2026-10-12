"""Training losses, EMA, flow freezing, and complete checkpoints."""

from collections import OrderedDict

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import ModalContentPoleWaveletDWT3
from basicsr.archs.spynet_arch import SpyNet
from basicsr.models.lr_scheduler import CosineSchedule
from basicsr.utils.checkpoint import atomic_save, gather_rng, restore_rng
from basicsr.utils.dist_util import is_distributed, rank_and_size, require_finite


class Trainer:
    _LEVELS = ("w3", "w2", "w1")

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.rank, self.world_size = rank_and_size()
        options = config["train"]
        network = ModalContentPoleWaveletDWT3(**config["model"]).to(device)
        self.net_g = (
            DistributedDataParallel(network, device_ids=[device.index])
            if is_distributed()
            else network
        )
        self.net_g.train()
        self.net_g_ema = ModalContentPoleWaveletDWT3(**config["model"]).to(device)
        self.net_g_ema.load_state_dict(network.state_dict(), strict=True)
        self.net_g_ema.eval().requires_grad_(False)
        self.ema_decay = float(options["ema_decay"])
        for key in (
            "lambda_rec",
            "lambda_ssim",
            "lambda_wave_res",
            "lambda_preview",
            "lambda_flow",
            "energy_ema_decay",
            "energy_floor",
            "charbonnier_eps",
            "gradient_clip_norm",
        ):
            setattr(self, key, float(options[key]))
        self.fix_flow_iter = int(options["fix_flow"])
        self.flow_teacher_schedule = tuple(options["flow_teacher_schedule"])
        self.trainable_spynet_levels = (4, 5)
        self.energy_weight_bounds = tuple(options["energy_weight_bounds"])
        self.aux_loss_weights = dict(options["aux_loss_weights"])
        self.gt_spynet_teacher = SpyNet(config["model"]["spynet_pretrained"]).to(device)
        self.gt_spynet_teacher.eval().requires_grad_(False)
        normal, flow = [], []
        for name, parameter in network.named_parameters():
            if parameter.requires_grad:
                target = (
                    flow
                    if any(
                        f"spynet.basic_module.{level}." in name
                        for level in self.trainable_spynet_levels
                    )
                    else normal
                )
                target.append(parameter)
        lr = float(options["optim_g"]["lr"])
        self.optimizer_g = torch.optim.Adam(
            [
                {"params": normal, "lr": lr},
                {"params": flow, "lr": lr * float(options["flow_lr_mul"])},
            ],
            **options["optim_g"],
        )
        self.scheduler = CosineSchedule(self.optimizer_g, options["total_iter"], options["eta_min"])

    @staticmethod
    def get_bare_model(network):
        return network.module if isinstance(network, DistributedDataParallel) else network

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)

    @torch.no_grad()
    def model_ema(self):
        source = dict(self.get_bare_model(self.net_g).named_parameters())
        for name, parameter in self.net_g_ema.named_parameters():
            parameter.mul_(self.ema_decay).add_(source[name], alpha=1.0 - self.ema_decay)
        self._copy_generator_buffers_to_ema()

    def _copy_generator_buffers_to_ema(self):
        source = dict(self.get_bare_model(self.net_g).named_buffers())
        target = dict(self.net_g_ema.named_buffers())
        for name in target:
            target[name].copy_(source[name])

    @staticmethod
    def _ramp(current_iter, schedule, maximum):
        start, end = schedule
        if current_iter <= start:
            return 0.0
        if current_iter >= end:
            return float(maximum)
        return float(maximum) * (current_iter - start) / (end - start)

    @staticmethod
    def _weighted_mean(value, weight):
        weight = weight.to(dtype=value.dtype)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _local_average_2d(value, window):
        padding = window // 2
        value = F.pad(value, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(value, window, stride=1)

    @classmethod
    def _ssim_loss(cls, prediction, target):
        batch, frames, channels, height, width = prediction.shape
        prediction = prediction.reshape(batch * frames, channels, height, width)
        target = target.reshape(batch * frames, channels, height, width)
        window = 11
        mu_prediction = cls._local_average_2d(prediction, window)
        mu_target = cls._local_average_2d(target, window)
        sigma_prediction = (
            cls._local_average_2d(prediction.square(), window) - mu_prediction.square()
        ).clamp_min(0.0)
        sigma_target = (
            cls._local_average_2d(target.square(), window) - mu_target.square()
        ).clamp_min(0.0)
        covariance = cls._local_average_2d(prediction * target, window) - mu_prediction * mu_target
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
        denominator = (mu_prediction.square() + mu_target.square() + c1) * (
            sigma_prediction + sigma_target + c2
        )
        return (1.0 - (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0)).mean()

    def _charbonnier(self, prediction, target):
        return torch.sqrt((prediction - target).square() + self.charbonnier_eps**2).mean()

    def _teacher_validity(self, flow, reverse_flow):
        batch, frames, _, height, width = flow.shape
        flat_flow = flow.reshape(batch * frames, 2, height, width)
        flat_reverse = reverse_flow.reshape(batch * frames, 2, height, width)
        warped_reverse = flow_warp(
            flat_reverse, flat_flow.permute(0, 2, 3, 1), padding_mode="zeros"
        )
        valid = flow_warp(
            torch.ones_like(flat_flow[:, :1]), flat_flow.permute(0, 2, 3, 1), padding_mode="zeros"
        ).clamp(0.0, 1.0)
        residual = flat_flow + warped_reverse
        denominator = (
            0.01
            * (
                flat_flow.square().sum(dim=1, keepdim=True)
                + warped_reverse.square().sum(dim=1, keepdim=True)
            )
            + 0.5
        )
        consistency = torch.exp(
            -residual.square().sum(dim=1, keepdim=True)
            / denominator.clamp_min(self.charbonnier_eps)
        )
        return (valid * consistency).reshape(batch, frames, 1, height, width).detach()

    def _gt_spynet_teacher_flows(self):
        batch, frames, channels, height, width = self.gt.shape
        gt_quarter = F.interpolate(
            self.gt.reshape(batch * frames, channels, height, width),
            size=(height // 4, width // 4),
            mode="bicubic",
            align_corners=False,
        ).reshape(batch, frames, channels, height // 4, width // 4)
        first = gt_quarter[:, :-1].reshape(batch * (frames - 1), channels, height // 4, width // 4)
        second = gt_quarter[:, 1:].reshape_as(first)
        with torch.no_grad():
            teacher_next = self.gt_spynet_teacher(first, second).reshape(
                batch, frames - 1, 2, height // 4, width // 4
            )
            teacher_previous = self.gt_spynet_teacher(second, first).reshape_as(teacher_next)
        return teacher_next, teacher_previous

    def _flow_teacher_loss(self, outputs, teacher_next, teacher_previous):
        student_next = outputs["student_flow_to_next"]
        student_previous = outputs["student_flow_to_previous"]
        valid_next = self._teacher_validity(teacher_next, teacher_previous)
        valid_previous = self._teacher_validity(teacher_previous, teacher_next)
        height, width = student_next.shape[-2:]
        scale = student_next.new_tensor([max(width - 1, 1), max(height - 1, 1)]).view(1, 1, 2, 1, 1)

        def error(student, teacher):
            difference = (student - teacher) / scale
            return torch.sqrt(
                difference.square().sum(dim=2, keepdim=True) + self.charbonnier_eps**2
            )

        loss = 0.5 * (
            self._weighted_mean(error(student_next, teacher_next), valid_next)
            + self._weighted_mean(error(student_previous, teacher_previous), valid_previous)
        )
        return loss

    def _synchronized_target_energy(self, outputs):
        energies = []
        for level in self._LEVELS:
            target_residual = outputs[f"{level}_target"] - outputs[f"{level}_base_anchor"]
            energies.append(
                torch.stack(
                    [band.detach().abs().mean() for band in torch.chunk(target_residual, 3, dim=2)]
                )
            )
        energies = torch.stack(energies)
        if is_distributed():
            dist.all_reduce(energies, op=dist.ReduceOp.SUM)
            energies /= dist.get_world_size()
        return energies

    def _update_energy_ema(self, outputs):
        energy = self._synchronized_target_energy(outputs)
        generator = self.get_bare_model(self.net_g)
        with torch.no_grad():
            if not bool(generator.wavelet_residual_energy_initialized.item()):
                generator.wavelet_residual_energy_ema.copy_(energy)
                generator.wavelet_residual_energy_initialized.fill_(True)
            else:
                generator.wavelet_residual_energy_ema.mul_(self.energy_ema_decay).add_(
                    energy, alpha=1.0 - self.energy_ema_decay
                )
        return generator.wavelet_residual_energy_ema.detach().clone()

    def _wavelet_residual_loss(self, outputs, energy_ema):
        reference = energy_ema.mean()
        weights = (reference / energy_ema.clamp_min(self.energy_floor)).clamp(
            *self.energy_weight_bounds
        )
        losses = []
        for level_index, level in enumerate(self._LEVELS):
            target = outputs[f"{level}_target"] - outputs[f"{level}_base_anchor"]
            prediction = outputs[f"{level}_residual"]
            for orientation, (prediction_band, target_band) in enumerate(
                zip(torch.chunk(prediction, 3, dim=2), torch.chunk(target, 3, dim=2))
            ):
                losses.append(
                    weights[level_index, orientation]
                    * self._charbonnier(prediction_band, target_band)
                )
        return torch.stack(losses).mean(), weights

    def step(self, current_iter):
        if current_iter > 1:
            self.scheduler.step()
        self.optimizer_g.zero_grad()
        outputs = self.net_g(self.lq, gt=self.gt)
        self.output = outputs["restored"]
        losses = OrderedDict(
            l_final_rec=self._charbonnier(self.output, self.gt),
            l_final_ssim=self._ssim_loss(self.output, self.gt),
            l_preview=self._charbonnier(outputs["stage1_preview"], self.gt),
        )
        energy_ema = self._update_energy_ema(outputs)
        losses["l_wave_res"], _ = self._wavelet_residual_loss(outputs, energy_ema)
        flow_weight = self._ramp(current_iter, self.flow_teacher_schedule, self.lambda_flow)
        flow_loss = self.output.new_zeros(())
        if flow_weight > 0:
            teacher_next, teacher_previous = self._gt_spynet_teacher_flows()
            flow_loss = self._flow_teacher_loss(outputs, teacher_next, teacher_previous)
        losses["l_flow_teacher"] = flow_loss
        total = (
            self.lambda_rec * losses["l_final_rec"]
            + self.lambda_ssim * losses["l_final_ssim"]
            + self.lambda_wave_res * losses["l_wave_res"]
            + self.lambda_preview * losses["l_preview"]
            + flow_weight * flow_loss
        )
        for name, value in outputs["aux_losses"].items():
            total = total + self.aux_loss_weights[name] * value
            losses[f"l_aux_{name}"] = value
        losses["l_total"] = total
        require_finite(losses, iteration=current_iter, stage="loss")
        total.backward()
        if current_iter <= self.fix_flow_iter:
            for name, parameter in self.net_g.named_parameters():
                if "spynet" in name:
                    parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.gradient_clip_norm)
        require_finite({"gradient_norm": norm}, iteration=current_iter, stage="gradient")
        self.optimizer_g.step()
        self.model_ema()
        self.get_bare_model(self.net_g).clear_inference_cache()
        with torch.no_grad():
            values = torch.stack([value.detach() for value in losses.values()])
            if is_distributed():
                dist.all_reduce(values)
                values /= self.world_size
        return {name: value.item() for name, value in zip(losses, values)}

    def save(self, path, *, step, epoch, batch_in_epoch, epoch_rng):
        rng = gather_rng(epoch_rng)
        if self.rank == 0:
            bare = self.get_bare_model(self.net_g)
            atomic_save(
                {
                    "params": {
                        name: value.detach().cpu() for name, value in bare.state_dict().items()
                    },
                    "params_ema": {
                        name: value.detach().cpu()
                        for name, value in self.net_g_ema.state_dict().items()
                    },
                    "optimizer": self.optimizer_g.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "rng": rng,
                    "world_size": self.world_size,
                    "config": self.config,
                },
                path,
            )

    def resume(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["world_size"] != self.world_size:
            raise ValueError("Resume requires the same number of distributed ranks.")
        if checkpoint["config"] != self.config:
            raise ValueError(
                "Resume requires the same configuration; use generator weights for a new experiment."
            )
        self.get_bare_model(self.net_g).load_state_dict(checkpoint["params"], strict=True)
        self.net_g_ema.load_state_dict(checkpoint["params_ema"], strict=True)
        self.optimizer_g.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        rng = checkpoint["rng"][self.rank]
        restore_rng(rng["current"])
        return checkpoint["step"], checkpoint["epoch"], checkpoint["batch_in_epoch"], rng
