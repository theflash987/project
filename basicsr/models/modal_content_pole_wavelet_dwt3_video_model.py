"""Training wrapper for isolated direct DWT-3 ModalContent restoration."""

from collections import Counter, OrderedDict

import torch
from pyiqa import create_metric
from torch import distributed as dist
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import get_dist_info


class ModalContentPoleWaveletDWT3VideoModel(BaseModel):
    """Train final/base reconstruction, wavelet, preview, flow and pole losses."""

    _LEVELS = ('w3', 'w2', 'w1')

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(build_network(opt['network_g']))
        self.print_network(self.net_g)
        self.init_training_settings()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = float(train_opt['ema_decay'])
        self.net_g_ema = build_network(
            self.opt['network_g']).to(self.device)
        self.model_ema(0)
        self._copy_generator_buffers_to_ema()
        self.net_g_ema.eval()

        self.lambda_rec = float(train_opt['lambda_rec'])
        self.lambda_ssim = float(train_opt['lambda_ssim'])
        self.lambda_base_rec = float(train_opt['lambda_base_rec'])
        self.lambda_base_ssim = float(train_opt['lambda_base_ssim'])
        self.lambda_wave_res = float(train_opt['lambda_wave_res'])
        self.lambda_preview = float(train_opt['lambda_preview'])
        self.lambda_flow = float(train_opt['lambda_flow'])
        self.fix_flow_iter = int(train_opt['fix_flow'])
        self.flow_teacher_schedule = tuple(
            int(value) for value in train_opt['flow_teacher_schedule'])
        self.trainable_spynet_levels = tuple(
            int(value) for value in train_opt['trainable_spynet_levels'])
        self.gt_spynet_teacher_pretrained = str(
            self.opt['network_g']['spynet_pretrained'])
        self.energy_ema_decay = float(train_opt['energy_ema_decay'])
        self.energy_floor = float(train_opt['energy_floor'])
        self.energy_weight_bounds = tuple(
            float(value) for value in train_opt['energy_weight_bounds'])
        self.charbonnier_eps = float(train_opt['charbonnier_eps'])
        self.gradient_clip_norm = float(train_opt['gradient_clip_norm'])
        self.aux_loss_weights = {
            str(name): float(weight)
            for name, weight in train_opt['aux_loss_weights'].items()
        }
        self.gt_spynet_teacher = SpyNet(
            load_path=self.gt_spynet_teacher_pretrained).to(self.device)
        self.gt_spynet_teacher.requires_grad_(False)
        self.gt_spynet_teacher.eval()
        self.setup_optimizers()
        self.setup_schedulers()
        get_root_logger().info(
            'Use a frozen clean-GT SpyNet teacher.  The teacher is outside '
            'the generator, EMA, optimizer and inference graph.')

    def _copy_generator_buffers_to_ema(self):
        source = dict(self.get_bare_model(self.net_g).named_buffers())
        target = dict(self.net_g_ema.named_buffers())
        for name in target:
            target[name].copy_(source[name])

    def setup_optimizers(self):
        train_opt = self.opt['train']
        base_lr = float(train_opt['optim_g']['lr'])
        flow_lr_mul = float(train_opt['flow_lr_mul'])
        normal_parameters = []
        flow_parameters = []
        for name, parameter in self.net_g.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(
                    f'spynet.basic_module.{level}.' in name
                    for level in self.trainable_spynet_levels):
                flow_parameters.append(parameter)
            else:
                normal_parameters.append(parameter)
        optimizer_parameters = [
            {'params': normal_parameters, 'lr': base_lr},
            {'params': flow_parameters, 'lr': base_lr * flow_lr_mul},
        ]
        optimizer_options = dict(train_opt['optim_g'])
        self.optimizer_g = torch.optim.Adam(
            optimizer_parameters, **optimizer_options)
        self.optimizers.append(self.optimizer_g)
        get_root_logger().info(
            'SpyNet levels 0-3 are permanently frozen; levels 4-5 use '
            f'{flow_lr_mul}x LR after iteration {self.fix_flow_iter}.')

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
        value = F.pad(
            value,
            (padding, padding, padding, padding),
            mode='reflect')
        return F.avg_pool2d(value, window, stride=1)

    @classmethod
    def _ssim_loss(cls, prediction, target):
        batch, frames, channels, height, width = prediction.shape
        prediction = prediction.reshape(
            batch * frames, channels, height, width)
        target = target.reshape(batch * frames, channels, height, width)
        window = 11
        mu_prediction = cls._local_average_2d(prediction, window)
        mu_target = cls._local_average_2d(target, window)
        sigma_prediction = (
            cls._local_average_2d(prediction.square(), window)
            - mu_prediction.square()).clamp_min(0.0)
        sigma_target = (
            cls._local_average_2d(target.square(), window)
            - mu_target.square()).clamp_min(0.0)
        covariance = (
            cls._local_average_2d(prediction * target, window)
            - mu_prediction * mu_target)
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (
            (2 * mu_prediction * mu_target + c1)
            * (2 * covariance + c2))
        denominator = (
            (mu_prediction.square() + mu_target.square() + c1)
            * (sigma_prediction + sigma_target + c2))
        return (
            1.0
            - (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0)
        ).mean()

    def _charbonnier(self, prediction, target):
        return torch.sqrt(
            (prediction - target).square() + self.charbonnier_eps**2).mean()

    def _teacher_validity(self, flow, reverse_flow):
        batch, frames, _, height, width = flow.shape
        flat_flow = flow.reshape(batch * frames, 2, height, width)
        flat_reverse = reverse_flow.reshape(batch * frames, 2, height, width)
        warped_reverse = flow_warp(
            flat_reverse,
            flat_flow.permute(0, 2, 3, 1),
            padding_mode='zeros')
        valid = flow_warp(
            torch.ones_like(flat_flow[:, :1]),
            flat_flow.permute(0, 2, 3, 1),
            padding_mode='zeros').clamp(0.0, 1.0)
        residual = flat_flow + warped_reverse
        denominator = (
            0.01 * (
                flat_flow.square().sum(dim=1, keepdim=True)
                + warped_reverse.square().sum(dim=1, keepdim=True))
            + 0.5)
        consistency = torch.exp(
            -residual.square().sum(dim=1, keepdim=True)
            / denominator.clamp_min(self.charbonnier_eps))
        return (valid * consistency).reshape(
            batch, frames, 1, height, width).detach()

    def _gt_spynet_teacher_flows(self):
        batch, frames, channels, height, width = self.gt.shape
        gt_quarter = F.interpolate(
            self.gt.reshape(batch * frames, channels, height, width),
            size=(height // 4, width // 4),
            mode='bicubic',
            align_corners=False).reshape(
                batch, frames, channels, height // 4, width // 4)
        first = gt_quarter[:, :-1].reshape(
            batch * (frames - 1), channels, height // 4, width // 4)
        second = gt_quarter[:, 1:].reshape_as(first)
        with torch.no_grad():
            teacher_next = self.gt_spynet_teacher(first, second).reshape(
                batch, frames - 1, 2, height // 4, width // 4)
            teacher_previous = self.gt_spynet_teacher(second, first).reshape_as(
                teacher_next)
        return teacher_next, teacher_previous

    def _flow_teacher_loss(
            self, outputs, teacher_next, teacher_previous):
        student_next = outputs['student_flow_to_next']
        student_previous = outputs['student_flow_to_previous']
        valid_next = self._teacher_validity(teacher_next, teacher_previous)
        valid_previous = self._teacher_validity(
            teacher_previous, teacher_next)
        height, width = student_next.shape[-2:]
        scale = student_next.new_tensor([
            max(width - 1, 1), max(height - 1, 1)
        ]).view(1, 1, 2, 1, 1)

        def error(student, teacher):
            difference = (student - teacher) / scale
            return torch.sqrt(
                difference.square().sum(dim=2, keepdim=True)
                + self.charbonnier_eps**2)

        loss = 0.5 * (
            self._weighted_mean(error(student_next, teacher_next), valid_next)
            + self._weighted_mean(
                error(student_previous, teacher_previous), valid_previous))
        epe = 0.5 * (
            self._weighted_mean(
                torch.sqrt((student_next - teacher_next).square().sum(
                    dim=2, keepdim=True)), valid_next)
            + self._weighted_mean(
                torch.sqrt((student_previous - teacher_previous).square().sum(
                    dim=2, keepdim=True)), valid_previous))
        return loss, {
            'gt_spynet_teacher_epe_feature_pixels': epe.detach(),
            'gt_spynet_teacher_valid_ratio': 0.5 * (
                valid_next.mean() + valid_previous.mean()).detach(),
        }

    def _synchronized_target_energy(self, outputs):
        energies = []
        for level in self._LEVELS:
            target_residual = (
                outputs[f'{level}_target']
                - outputs[f'{level}_base_anchor'])
            energies.append(torch.stack([
                band.detach().abs().mean()
                for band in torch.chunk(target_residual, 3, dim=2)
            ]))
        energies = torch.stack(energies)
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
                generator.wavelet_residual_energy_ema.mul_(
                    self.energy_ema_decay).add_(
                        energy, alpha=1.0 - self.energy_ema_decay)
        return generator.wavelet_residual_energy_ema.detach().clone()

    def _wavelet_residual_loss(self, outputs, energy_ema):
        reference = energy_ema.mean()
        weights = (
            reference / energy_ema.clamp_min(self.energy_floor)
        ).clamp(*self.energy_weight_bounds)
        losses = []
        for level_index, level in enumerate(self._LEVELS):
            target = (
                outputs[f'{level}_target']
                - outputs[f'{level}_base_anchor'])
            prediction = outputs[f'{level}_residual']
            for orientation, (prediction_band, target_band) in enumerate(zip(
                    torch.chunk(prediction, 3, dim=2),
                    torch.chunk(target, 3, dim=2))):
                losses.append(
                    weights[level_index, orientation]
                    * self._charbonnier(prediction_band, target_band))
        return torch.stack(losses).mean(), weights

    def _gradient_norm(self, predicate):
        gradients = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in self.get_bare_model(
                self.net_g).named_parameters()
            if predicate(name) and parameter.grad is not None
        ]
        if not gradients:
            return self.output.new_zeros(())
        return torch.sqrt(torch.stack(gradients).sum())

    def optimize_parameters(self, current_iter):
        if current_iter == 1:
            get_root_logger().info(
                f'Freeze all SpyNet updates through iter {self.fix_flow_iter}.')
        elif current_iter == self.fix_flow_iter + 1:
            get_root_logger().warning('Enable only SpyNet levels 4/5.')
        self.optimizer_g.zero_grad()
        teacher_next, teacher_previous = self._gt_spynet_teacher_flows()
        outputs = self.net_g(self.lq, gt=self.gt)
        self.output = outputs['restored']
        loss_dict = OrderedDict()

        loss_final_rec = self._charbonnier(self.output, self.gt)
        loss_final_ssim = self._ssim_loss(self.output, self.gt)
        loss_base_rec = self._charbonnier(
            outputs['base_restored'], self.gt)
        loss_base_ssim = self._ssim_loss(
            outputs['base_restored'], self.gt)
        energy_ema = self._update_energy_ema(outputs)
        loss_wave_res, energy_weights = self._wavelet_residual_loss(
            outputs, energy_ema)
        loss_preview = self._charbonnier(outputs['stage1_preview'], self.gt)
        flow_loss, flow_stats = self._flow_teacher_loss(
            outputs, teacher_next, teacher_previous)
        flow_weight = self._ramp(
            current_iter, self.flow_teacher_schedule, self.lambda_flow)
        total = (
            self.lambda_rec * loss_final_rec
            + self.lambda_ssim * loss_final_ssim
            + self.lambda_base_rec * loss_base_rec
            + self.lambda_base_ssim * loss_base_ssim
            + self.lambda_wave_res * loss_wave_res
            + self.lambda_preview * loss_preview
            + flow_weight * flow_loss)
        loss_dict.update({
            'l_final_rec': loss_final_rec,
            'l_final_ssim': loss_final_ssim,
            'l_base_rec': loss_base_rec,
            'l_base_ssim': loss_base_ssim,
            'l_wave_res': loss_wave_res,
            'l_preview': loss_preview,
            'l_flow_teacher': flow_loss,
            'lambda_flow_current': self.output.new_tensor(flow_weight),
            'wave_res_energy_min': energy_ema.min(),
            'wave_res_energy_max': energy_ema.max(),
            'wave_res_weight_min': energy_weights.min(),
            'wave_res_weight_max': energy_weights.max(),
            **flow_stats,
        })
        for name, value in outputs['aux_losses'].items():
            weight = self.aux_loss_weights[name]
            total = total + weight * value
            loss_dict[f'l_aux_{name}'] = value
            loss_dict[f'lambda_aux_{name}'] = self.output.new_tensor(weight)
        for name, value in outputs['log_vars'].items():
            loss_dict[name] = value.detach()
        loss_dict.update({
            'wave_res_w3_abs_mean': outputs['w3_residual'].detach().abs().mean(),
            'wave_res_w2_abs_mean': outputs['w2_residual'].detach().abs().mean(),
            'wave_res_w1_abs_mean': outputs['w1_residual'].detach().abs().mean(),
            'base_to_final_residual_abs_mean': (
                outputs['restored'].detach()
                - outputs['base_restored'].detach()).abs().mean(),
            'teacher_loss_fraction': (
                flow_weight * flow_loss.detach()
                / total.detach().abs().clamp_min(1e-12)),
            'l_total': total.detach(),
        })

        total.backward()
        if current_iter <= self.fix_flow_iter:
            for name, parameter in self.net_g.named_parameters():
                if 'spynet' in name:
                    parameter.grad = None
        spynet_grad_norm = self._gradient_norm(
            lambda name: 'spynet.basic_module.4.' in name
            or 'spynet.basic_module.5.' in name)
        wavelet_branch_grad_norm = self._gradient_norm(
            lambda name: name.startswith((
                'post_degradation_context.',
                'w1_local_guide.',
                'wavelet_experts.')))
        direction_grad_norm = self._gradient_norm(
            lambda name: 'direction_attention.stage_1.' in name)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net_g.parameters(), self.gradient_clip_norm)
        self.optimizer_g.step()
        loss_dict.update({
            'grad_norm': grad_norm.detach(),
            'spynet_grad_norm': spynet_grad_norm.detach(),
            'wavelet_branch_grad_norm': wavelet_branch_grad_norm.detach(),
            'stage1_direction_grad_norm': direction_grad_norm.detach(),
        })
        loss_dict['cuda_peak_allocated_gib'] = self.output.new_tensor(
            torch.cuda.max_memory_allocated(self.device) / 2**30)
        loss_dict['cuda_peak_reserved_gib'] = self.output.new_tensor(
            torch.cuda.max_memory_reserved(self.device) / 2**30)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.model_ema(decay=self.ema_decay)
        self._copy_generator_buffers_to_ema()

    def test(self):
        network = self.net_g_ema
        with torch.no_grad():
            outputs = network(self.lq)
        self.output = outputs['restored']
        self.base_output = outputs['base_restored']
        self.validation_stats = {
            'w1_local_guide_abs_mean': (
                outputs['log_vars']['w1_local_guide_abs_mean']),
            'base_to_final_residual_abs_mean': (
                self.output - self.base_output).abs().mean(),
        }

    def get_current_visuals(self):
        return OrderedDict(
            lq=self.lq.detach().cpu(),
            result=self.output.detach().cpu(),
            base_restored=self.base_output.detach().cpu(),
            gt=self.gt.detach().cpu())

    @staticmethod
    def metric_cal_pre(value):
        value = value.squeeze(0).float().clamp_(0.0, 1.0)
        value = (value * 255).round().to(torch.uint8)
        return value.to(torch.float32).div_(255).unsqueeze(0)

    def dist_validation(self, dataloader, current_iter, tb_logger):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        metric_options = self.opt['val']['metrics']
        metric_names = tuple(metric_options)
        num_frame_each_folder = Counter(dataset.data_info['folder'])
        result_sets = {
            kind: {
                folder: torch.zeros(
                    frames,
                    len(metric_names),
                    dtype=torch.float32,
                    device=self.device)
                for folder, frames in num_frame_each_folder.items()
            }
            for kind in ('final', 'base')
        }
        metrics = {
            name: create_metric(option['type'], metric_mode='FR')
            for name, option in metric_options.items()
        }
        rank, world_size = get_dist_info()
        num_folders = len(dataset)
        num_pad = (world_size - num_folders % world_size) % world_size
        pbar = tqdm(total=num_folders, unit='folder') if rank == 0 else None
        diagnostic_names = (
            'base_to_final_residual_abs_mean',
            'w1_local_guide_abs_mean',
        )
        diagnostic_sums = {name: 0.0 for name in diagnostic_names}
        diagnostic_count = 0
        for data_index in range(rank, num_folders + num_pad, world_size):
            index = min(data_index, num_folders - 1)
            val_data = dataset[index]
            folder = val_data['folder']
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            self.test()
            visuals = self.get_current_visuals()
            if data_index < num_folders:
                for frame in range(visuals['result'].shape[1]):
                    gt = visuals['gt'][0, frame].unsqueeze(0)
                    for kind, key in (
                            ('final', 'result'),
                            ('base', 'base_restored')):
                        prediction = self.metric_cal_pre(
                            visuals[key][0, frame])
                        for metric_index, name in enumerate(metric_names):
                            result_sets[kind][folder][frame, metric_index] = (
                                metrics[name](prediction, gt).cpu().item())
                for name, value in self.validation_stats.items():
                    diagnostic_sums[name] += float(value.detach().cpu())
                diagnostic_count += 1
            del self.lq, self.output, self.base_output
            del self.gt
            torch.cuda.empty_cache()
            if pbar is not None:
                for _ in range(min(world_size, num_folders - pbar.n)):
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
        for kind in result_sets.values():
            for tensor in kind.values():
                dist.reduce(tensor, 0)
        diagnostic_totals = torch.tensor(
            [*(diagnostic_sums[name] for name in diagnostic_names),
             diagnostic_count],
            dtype=torch.float64,
            device=self.device)
        dist.reduce(diagnostic_totals, 0)
        dist.barrier()
        if rank == 0:
            log_lines = [f'Validation {dataset_name} @ {current_iter}']
            for kind, folders in result_sets.items():
                macro = torch.stack([
                    values.mean(dim=0) for values in folders.values()
                ]).mean(dim=0)
                for metric_index, name in enumerate(metric_names):
                    value = macro[metric_index].item()
                    log_lines.append(f'\t{kind}_{name}: {value:.6f}')
                    if tb_logger:
                        tb_logger.add_scalar(
                            f'metrics/{dataset_name}/{kind}_{name}',
                            value,
                            current_iter)
            count = diagnostic_totals[-1].item()
            for index, name in enumerate(diagnostic_names):
                value = diagnostic_totals[index].item() / count
                log_lines.append(f'\t{name}: {value:.6f}')
                if tb_logger:
                    tb_logger.add_scalar(
                        f'diagnostics/{name}', value, current_iter)
            get_root_logger().info('\n'.join(log_lines))

    def save(self, epoch, current_iter):
        self.save_network(
            [self.net_g, self.net_g_ema],
            'net_g',
            current_iter,
            param_key=['params', 'params_ema'])
        self.save_training_state(epoch, current_iter)
