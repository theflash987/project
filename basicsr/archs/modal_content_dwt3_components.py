"""Content-adaptive modal Pole memory and shared DWT-3 refinement blocks."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.official_dense_components import ResidualUnit
from mmcv.ops.modulated_deform_conv import (
    ModulatedDeformConv2d,
    modulated_deform_conv2d,
)


def _flatten_video(value):
    batch, frames, channels, height, width = value.shape
    return value.reshape(batch * frames, channels, height, width), (batch, frames)


def _restore_video(value, batch, frames):
    return value.reshape(
        batch,
        frames,
        value.shape[1],
        value.shape[2],
        value.shape[3])


class VideoPreContext(nn.Module):
    """Extract pre-restoration clip and frame degradation/content tokens."""

    def __init__(
            self,
            branch_channels=16,
            context_channels=32,
            token_channels=64):
        super().__init__()
        self.appearance_encoder = nn.Sequential(
            nn.Conv2d(3, branch_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(branch_channels))
        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(3, branch_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(branch_channels))
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(6, branch_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(branch_channels))
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * branch_channels, context_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(context_channels))
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(
                context_channels,
                context_channels,
                3,
                1,
                1,
                groups=context_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(context_channels, token_channels, 1),
            nn.SiLU(inplace=True))
        self.clip_projection = nn.Sequential(
            nn.Linear(token_channels, token_channels),
            nn.SiLU(inplace=True),
            nn.Linear(token_channels, token_channels))

    @staticmethod
    def _temporal_evidence(lq):
        previous = torch.zeros_like(lq)
        following = torch.zeros_like(lq)
        difference = (lq[:, 1:] - lq[:, :-1]).abs()
        previous[:, 1:] = difference
        following[:, :-1] = difference
        return torch.cat([previous, following], dim=2)

    def forward(self, lq):
        lq_flat, (batch, frames) = _flatten_video(lq)
        highpass = lq_flat - F.avg_pool2d(
            lq_flat, kernel_size=3, stride=1, padding=1)
        temporal, _ = _flatten_video(self._temporal_evidence(lq))
        fused = self.fusion(torch.cat([
            self.appearance_encoder(lq_flat),
            self.frequency_encoder(highpass),
            self.temporal_encoder(temporal),
        ], dim=1))
        spatial_context = _restore_video(fused, batch, frames)
        pooled = spatial_context.mean(dim=(-2, -1)).transpose(1, 2)
        frame_tokens = self.temporal_mixer(pooled).transpose(1, 2)
        clip_token = self.clip_projection(frame_tokens.mean(dim=1))
        return {
            'spatial': spatial_context,
            'frame_tokens': frame_tokens,
            'clip_token': clip_token,
        }


class ModalConditionProjector(nn.Module):
    """Project shared ModalContent features for deformable alignment."""

    def __init__(
            self,
            spatial_channels=32,
            token_channels=64,
            output_channels=96):
        super().__init__()
        self.spatial_projection = nn.Conv2d(
            spatial_channels, output_channels, 3, 1, 1)
        self.token_projection = nn.Linear(
            2 * token_channels, output_channels)
        self.fusion = nn.Sequential(
            nn.SiLU(inplace=True),
            ResidualUnit(output_channels))
        self.alignment_projection = nn.Conv2d(
            output_channels, output_channels, 1)

    def forward(self, spatial_context, frame_tokens, clip_token):
        batch, frames, _, height, width = spatial_context.shape
        spatial = self.spatial_projection(spatial_context.reshape(
            batch * frames,
            spatial_context.shape[2],
            height,
            width))
        clip = clip_token.unsqueeze(1).expand(-1, frames, -1)
        token = self.token_projection(torch.cat([
            frame_tokens,
            clip,
        ], dim=-1)).reshape(batch * frames, -1, 1, 1)
        shared = self.fusion(spatial + token)
        alignment = self.alignment_projection(shared).reshape(
            batch, frames, -1, height, width)
        return alignment


class ModalConditionedAlignment(ModulatedDeformConv2d):
    """Flow-guided DCN whose residual offset and mask use ModalContent."""

    def __init__(
            self,
            *args,
            modal_channels=96,
            max_residue_magnitude=10,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.max_residue_magnitude = float(max_residue_magnitude)
        self.offset_input = nn.Sequential(
            nn.Conv2d(
                2 * self.out_channels + 2,
                self.out_channels,
                3,
                1,
                1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.modal_projection = nn.Conv2d(
            modal_channels, self.out_channels, 1)
        self.offset_body = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                3,
                1,
                1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                3,
                1,
                1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                27 * self.deform_groups,
                3,
                1,
                1))
        nn.init.constant_(self.offset_body[-1].weight, 0.0)
        nn.init.constant_(self.offset_body[-1].bias, 0.0)

    def forward(self, feature, condition, flow, modal_condition):
        offset_feature = self.offset_input(torch.cat([
            condition,
            flow,
        ], dim=1))
        offset_feature = (
            offset_feature
            + self.modal_projection(modal_condition))
        offset_y, offset_x, mask = torch.chunk(
            self.offset_body(offset_feature), 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat([offset_y, offset_x], dim=1))
        offset = (
            offset
            + flow.flip(1).repeat(
                1,
                offset.shape[1] // 2,
                1,
                1))
        return modulated_deform_conv2d(
            feature,
            offset,
            torch.sigmoid(mask),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups)


class Stage1PreviewHead(nn.Module):
    """Decode a lightweight full-resolution preview from Stage-1 features."""

    def __init__(self, channels=96):
        super().__init__()
        self.residual = nn.Conv2d(2 * channels, 3 * 4 * 4, 3, 1, 1)
        nn.init.constant_(self.residual.weight, 0.0)
        nn.init.constant_(self.residual.bias, 0.0)

    def forward(self, lq, spatial_feature, stage1_feature):
        residual = F.pixel_shuffle(
            self.residual(torch.cat([spatial_feature, stage1_feature], dim=1)),
            4)
        return lq + residual


class FixedStablePoles(nn.Module):
    """Globally learned poles shared by every frame and video."""

    def __init__(
            self,
            num_poles=16,
            delta_t=1.0,
            decay_eps=1e-6,
            omega_max=math.pi,
            separation_bandwidth=0.01):
        super().__init__()
        self.num_poles = int(num_poles)
        self.delta_t = float(delta_t)
        self.decay_eps = float(decay_eps)
        self.omega_max = float(omega_max)
        self.separation_bandwidth = float(separation_bandwidth)
        rho = torch.logspace(math.log10(0.02), math.log10(1.0), num_poles)
        omega_fraction = torch.linspace(0.03, 0.97, num_poles)
        self.raw_rho = nn.Parameter(torch.log(torch.expm1(rho)))
        self.raw_omega = nn.Parameter(torch.logit(omega_fraction))

    def _discretize(self, rho, omega):
        phase = omega * self.delta_t
        decay = torch.exp(-rho * self.delta_t)
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        a_real = decay * cosine
        a_imag = decay * sine
        expm1_real = (
            torch.expm1(-rho * self.delta_t) * cosine
            - 2.0 * torch.sin(0.5 * phase).square())
        denominator = (rho.square() + omega.square()).clamp_min(
            torch.finfo(rho.dtype).tiny)
        return {
            'lambdas': rho,
            'omegas': omega,
            'a_real': a_real,
            'a_imag': a_imag,
            'phi_real': (
                -rho * expm1_real + omega * a_imag) / denominator,
            'phi_imag': (
                -rho * a_imag - omega * expm1_real) / denominator,
        }

    def _separation(self, rho, omega):
        log_rho = torch.log(rho)
        coordinates = torch.stack([
            (log_rho - log_rho.min())
            / (log_rho.max() - log_rho.min()).clamp_min(self.decay_eps),
            omega / self.omega_max,
        ], dim=-1)
        distance_square = torch.cdist(coordinates, coordinates).square()
        upper = torch.triu(torch.ones(
            self.num_poles,
            self.num_poles,
            dtype=torch.bool,
            device=rho.device), diagonal=1)
        pairwise = distance_square[upper]
        return (
            torch.exp(-pairwise / self.separation_bandwidth).mean(),
            pairwise.min().sqrt())

    def forward(self, frame_tokens, _clip_token):
        compute_dtype = frame_tokens.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        base_rho = F.softplus(self.raw_rho.to(
            device=frame_tokens.device, dtype=compute_dtype)) + self.decay_eps
        base_omega = self.omega_max * torch.sigmoid(self.raw_omega.to(
            device=frame_tokens.device, dtype=compute_dtype))
        batch, frames = frame_tokens.shape[:2]
        rho = base_rho.view(1, 1, -1).expand(batch, frames, -1)
        omega = base_omega.view(1, 1, -1).expand(batch, frames, -1)
        poles = self._discretize(rho, omega)
        separation, minimum_distance = self._separation(
            base_rho, base_omega)
        poles.update({
            'aux_losses': {
                'pole_separation': separation,
            },
            'stats': {
                'pole_min_distance': minimum_distance.detach(),
            },
        })
        for name in (
                'lambdas',
                'omegas',
                'a_real',
                'a_imag',
                'phi_real',
                'phi_imag'):
            poles[name] = poles[name].to(frame_tokens.dtype)
        return poles


class ModalContentPoleMixer(nn.Module):
    """Predict independent local mode gates with bounded video-token biases."""

    def __init__(
            self,
            channels,
            num_poles,
            token_channels=64,
            hidden_channels=32):
        super().__init__()
        self.num_poles = int(num_poles)
        output_channels = 2 * self.num_poles + 1
        self.net = nn.Sequential(
            nn.Conv2d(
                4 * int(channels) + 2 * self.num_poles + 3,
                hidden_channels,
                3,
                1,
                1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, 1))
        self.token_bias = nn.Sequential(
            nn.Linear(token_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, output_channels))
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, 0.0)
        nn.init.normal_(self.token_bias[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.token_bias[-1].bias, 0.0)
        read_bias = math.log(0.25 / 0.75)
        write_bias = math.log(0.25 / 0.75)
        self.net[-1].bias.data[:self.num_poles] = read_bias
        self.net[-1].bias.data[
            self.num_poles:2 * self.num_poles] = write_bias
        self.net[-1].bias.data[-1] = math.log(0.90 / 0.10)

    def forward(
            self,
            feat_current,
            feat_aligned,
            spatial_aligned,
            state_evidence,
            reliability_maps,
            frame_token):
        flow_reliability, spatial_reliability, valid = reliability_maps
        evidence = torch.cat([
            feat_current,
            feat_aligned,
            spatial_aligned,
            (feat_current - spatial_aligned).abs(),
            state_evidence,
            flow_reliability,
            spatial_reliability,
            valid,
        ], dim=1)
        logits = (
            self.net(evidence)
            + self.token_bias(frame_token).unsqueeze(-1).unsqueeze(-1))
        read_logits, write_logits, alignment_logit = torch.split(
            logits,
            [self.num_poles, self.num_poles, 1],
            dim=1)
        return {
            'read_logits': read_logits,
            'write_gate': torch.sigmoid(write_logits),
            'alignment_logit': alignment_logit,
        }


class ModalContentPoleHistoryCell(nn.Module):
    """Stable recurrent modes with per-mode writes and non-exclusive readout."""

    def __init__(
            self,
            channels,
            num_poles=16,
            state_channels=2,
            token_channels=64,
            mixer_channels=32,
            decay_eps=1e-6,
            flow_alpha=0.01,
            flow_beta=0.5,
            spatial_temperature=0.5):
        super().__init__()
        self.num_poles = int(num_poles)
        self.state_channels = int(state_channels)
        self.decay_eps = float(decay_eps)
        self.flow_alpha = float(flow_alpha)
        self.flow_beta = float(flow_beta)
        self.spatial_temperature = float(spatial_temperature)
        modal_channels = self.num_poles * self.state_channels
        self.input_proj = nn.Conv2d(
            3 * int(channels),
            modal_channels,
            1)
        self.read_norm = nn.GroupNorm(1, modal_channels)
        self.output_proj = nn.Conv2d(
            modal_channels,
            int(channels),
            1,
            bias=False)
        self.write_real = nn.Parameter(
            torch.ones(self.num_poles, self.state_channels))
        self.write_imag = nn.Parameter(
            torch.zeros(self.num_poles, self.state_channels))
        self.read_real = nn.Parameter(
            torch.ones(self.num_poles, self.state_channels))
        self.read_imag = nn.Parameter(
            torch.zeros(self.num_poles, self.state_channels))
        self.mixer = ModalContentPoleMixer(
            channels,
            self.num_poles,
            token_channels=token_channels,
            hidden_channels=mixer_channels)
        self.last_debug_stats = {}
        self.last_occupancy = None

    def _zero_state(self, reference):
        shape = (
            reference.shape[0],
            self.num_poles,
            self.state_channels,
            reference.shape[-2],
            reference.shape[-1])
        return reference.new_zeros(shape), reference.new_zeros(shape)

    @staticmethod
    def _warp_state(state, flow):
        real, imag = state
        batch, poles, channels, height, width = real.shape
        grid_flow = flow.permute(0, 2, 3, 1)
        real = flow_warp(
            real.reshape(batch, poles * channels, height, width),
            grid_flow,
            padding_mode='zeros').reshape_as(real)
        imag = flow_warp(
            imag.reshape(batch, poles * channels, height, width),
            grid_flow,
            padding_mode='zeros').reshape_as(imag)
        return real, imag

    def _reliability(
            self,
            feat_current,
            spatial_aligned,
            flow,
            reverse_flow):
        shape = (
            feat_current.shape[0],
            1,
            feat_current.shape[-2],
            feat_current.shape[-1])
        if flow is None or reverse_flow is None:
            ones = feat_current.new_ones(shape)
            zeros = feat_current.new_zeros(shape)
            return zeros, (ones, ones, zeros)
        with torch.no_grad():
            flow_detached = flow.detach()
            reverse_detached = reverse_flow.detach()
            warped_reverse = flow_warp(
                reverse_detached,
                flow_detached.permute(0, 2, 3, 1),
                padding_mode='zeros')
            valid = flow_warp(
                torch.ones_like(flow_detached[:, :1]),
                flow_detached.permute(0, 2, 3, 1),
                padding_mode='zeros').clamp_(0.0, 1.0)
            fb_residual = flow_detached + warped_reverse
            fb_square = fb_residual.square().sum(dim=1, keepdim=True)
            fb_denom = (
                self.flow_alpha
                * (
                    flow_detached.square().sum(dim=1, keepdim=True)
                    + warped_reverse.square().sum(dim=1, keepdim=True))
                + self.flow_beta)
            flow_reliability = torch.exp(-fb_square / fb_denom)
            current_unit = F.normalize(
                feat_current.detach(),
                dim=1,
                eps=self.decay_eps)
            aligned_unit = F.normalize(
                spatial_aligned.detach(),
                dim=1,
                eps=self.decay_eps)
            spatial_square = (
                current_unit - aligned_unit).square().sum(
                    dim=1,
                    keepdim=True)
            spatial_reliability = torch.exp(
                -spatial_square / self.spatial_temperature)
            geometry_support = (valid * flow_reliability).clamp_(0.0, 1.0)
        return geometry_support, (
            flow_reliability,
            spatial_reliability,
            valid,
        )

    def forward(
            self,
            feat_current,
            feat_aligned,
            spatial_aligned,
            state,
            flow,
            reverse_flow,
            poles,
            frame_token):
        history_available = state is not None and flow is not None
        if history_available:
            warped_real, warped_imag = self._warp_state(state, flow)
        else:
            warped_real, warped_imag = self._zero_state(feat_current)
            spatial_aligned = feat_current
        geometry_support, reliability_maps = self._reliability(
            feat_current,
            spatial_aligned,
            flow,
            reverse_flow)

        batch, _, height, width = feat_current.shape
        write_feature = self.input_proj(torch.cat([
            feat_current,
            feat_aligned,
            feat_current - spatial_aligned,
        ], dim=1)).reshape(
            batch,
            self.num_poles,
            self.state_channels,
            height,
            width)
        state_amplitude = torch.log1p(
            (
                warped_real.square()
                + warped_imag.square()
            ).mean(dim=2).clamp_min(self.decay_eps).sqrt())
        state_similarity = (
            F.normalize(
                warped_real,
                dim=2,
                eps=self.decay_eps)
            * F.normalize(
                write_feature,
                dim=2,
                eps=self.decay_eps)
        ).sum(dim=2)
        state_evidence = torch.cat([
            state_amplitude,
            state_similarity,
        ], dim=1)
        mixer = self.mixer(
            feat_current,
            feat_aligned,
            spatial_aligned,
            state_evidence,
            reliability_maps,
            frame_token)

        a_real = poles['a_real'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        a_imag = poles['a_imag'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        support = geometry_support.unsqueeze(1)
        history_real = support * (
            a_real * warped_real - a_imag * warped_imag)
        history_imag = support * (
            a_imag * warped_real + a_real * warped_imag)
        read_gate = (
            geometry_support
            * torch.sigmoid(mixer['read_logits'])
        ).unsqueeze(2)
        read_real = self.read_real.view(
            1,
            self.num_poles,
            self.state_channels,
            1,
            1)
        read_imag = self.read_imag.view(
            1,
            self.num_poles,
            self.state_channels,
            1,
            1)
        pole_read = (
            read_real * history_real
            - read_imag * history_imag)
        memory_state = (
            read_gate * pole_read).reshape(
                batch,
                self.num_poles * self.state_channels,
                height,
                width)
        memory = self.output_proj(self.read_norm(memory_state))
        alignment_gate = (
            geometry_support
            * torch.sigmoid(mixer['alignment_logit']))

        write_gate = mixer['write_gate'].unsqueeze(2)
        write_real = self.write_real.view(
            1,
            self.num_poles,
            self.state_channels,
            1,
            1)
        write_imag = self.write_imag.view(
            1,
            self.num_poles,
            self.state_channels,
            1,
            1)
        source_real = write_gate * write_real * write_feature
        source_imag = write_gate * write_imag * write_feature
        phi_real = poles['phi_real'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        phi_imag = poles['phi_imag'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        input_real = phi_real * source_real - phi_imag * source_imag
        input_imag = phi_real * source_imag + phi_imag * source_real
        real = history_real + input_real
        imag = history_imag + input_imag

        amplitude_power = (
            history_real.square()
            + history_imag.square()
        ).mean(dim=2)
        amplitude = (
            (amplitude_power + self.decay_eps).sqrt()
            - math.sqrt(self.decay_eps)).clamp_min(0.0)
        normalized_amplitude = (
            amplitude.detach()
            / amplitude.detach().mean(
                dim=1,
                keepdim=True).clamp_min(self.decay_eps)
        ).clamp_max(2.0)
        self.last_occupancy = (
            read_gate.squeeze(2) * normalized_amplitude).mean(dim=(-2, -1))
        gate_summary = read_gate.squeeze(2).mean(dim=(-2, -1))
        effective_modes = (
            gate_summary.sum(dim=1).square()
            / gate_summary.square().sum(dim=1).clamp_min(
                self.decay_eps))
        transition = torch.sqrt(
            poles['a_real'].square() + poles['a_imag'].square())
        self.last_debug_stats = {
            'lambda_min': poles['lambdas'].detach().min(),
            'lambda_max': poles['lambdas'].detach().max(),
            'lambda_mean': poles['lambdas'].detach().mean(),
            'omega_mean': poles['omegas'].detach().mean(),
            'transition_abs_max': transition.detach().max(),
            'history_available': feat_current.new_tensor(
                float(history_available)),
            'geometry_support_mean': geometry_support.detach().mean(),
            'read_gate_mean': read_gate.detach().mean(),
            'effective_active_poles': effective_modes.detach().mean(),
            'write_gate_mean': mixer['write_gate'].detach().mean(),
            'alignment_gate_mean': alignment_gate.detach().mean(),
            'state_amplitude_mean': state_amplitude.detach().mean(),
            'state_abs_mean': (
                real.detach().abs().mean()
                + imag.detach().abs().mean()) * 0.5,
            'memory_abs_mean': memory.detach().abs().mean(),
        }
        return memory, (real, imag), alignment_gate


class CompactPostDegradationContext(nn.Module):
    """Compact static/temporal context computed at the W2 resolution."""

    def __init__(self, channels=16, branch_channels=8, flow_eps=1e-6):
        super().__init__()
        self.flow_eps = float(flow_eps)
        self.static_encoder = nn.Sequential(
            nn.Conv2d(18, branch_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(branch_channels))
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(14, branch_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(branch_channels))
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * branch_channels, channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(channels))

    @staticmethod
    def _resize_video(value, size):
        flattened, (batch, frames) = _flatten_video(value)
        flattened = F.interpolate(
            flattened,
            size=tuple(size),
            mode='area')
        return _restore_video(flattened, batch, frames)

    @staticmethod
    def _resize_flow(flow, size):
        batch, frames, _, height, width = flow.shape
        target_height, target_width = tuple(size)
        resized = F.interpolate(
            flow.reshape(batch * frames, 2, height, width),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False)
        resized[:, 0] *= target_width / width
        resized[:, 1] *= target_height / height
        return resized.reshape(
            batch,
            frames,
            2,
            target_height,
            target_width)

    @staticmethod
    def _warp_pairs(source, flow):
        batch, frames, channels, height, width = source.shape
        warped = flow_warp(
            source.reshape(batch * frames, channels, height, width),
            flow.reshape(batch * frames, 2, height, width).permute(
                0, 2, 3, 1),
            padding_mode='zeros')
        return warped.reshape(batch, frames, channels, height, width)

    def _flow_reliability(self, flow, reverse_flow):
        batch, frames, _, height, width = flow.shape
        reverse_warped = self._warp_pairs(reverse_flow, flow)
        valid = self._warp_pairs(
            flow.new_ones(batch, frames, 1, height, width),
            flow).clamp(0.0, 1.0)
        residual = flow + reverse_warped
        denominator = (
            0.01 * (
                flow.square().sum(dim=2, keepdim=True)
                + reverse_warped.square().sum(dim=2, keepdim=True))
            + 0.5)
        reliability = torch.exp(
            -residual.square().sum(dim=2, keepdim=True)
            / denominator.clamp_min(self.flow_eps))
        return (valid * reliability).clamp(0.0, 1.0)

    def _temporal_evidence(
            self,
            lq,
            base,
            flows_forward,
            flows_backward):
        batch, frames, _, height, width = lq.shape
        previous_lq = torch.zeros_like(lq)
        following_lq = torch.zeros_like(lq)
        previous_base = torch.zeros_like(base)
        following_base = torch.zeros_like(base)
        previous_reliability = lq.new_zeros(batch, frames, 1, height, width)
        following_reliability = lq.new_zeros(batch, frames, 1, height, width)
        flows_forward = self._resize_flow(
            flows_forward.detach(), (height, width))
        flows_backward = self._resize_flow(
            flows_backward.detach(), (height, width))
        aligned_previous_lq = self._warp_pairs(lq[:, :-1], flows_forward)
        aligned_following_lq = self._warp_pairs(lq[:, 1:], flows_backward)
        aligned_previous_base = self._warp_pairs(
            base[:, :-1], flows_forward)
        aligned_following_base = self._warp_pairs(
            base[:, 1:], flows_backward)
        previous_lq[:, 1:] = (lq[:, 1:] - aligned_previous_lq).abs()
        following_lq[:, :-1] = (lq[:, :-1] - aligned_following_lq).abs()
        previous_base[:, 1:] = (base[:, 1:] - aligned_previous_base).abs()
        following_base[:, :-1] = (base[:, :-1] - aligned_following_base).abs()
        previous_reliability[:, 1:] = self._flow_reliability(
            flows_forward, flows_backward)
        following_reliability[:, :-1] = self._flow_reliability(
            flows_backward, flows_forward)
        evidence = torch.cat([
            previous_lq,
            following_lq,
            previous_base,
            following_base,
            previous_reliability,
            following_reliability,
        ], dim=2)
        return evidence

    def forward(
            self,
            lq,
            base,
            spatial_sizes,
            flows_forward,
            flows_backward):
        reference_size = tuple(spatial_sizes['w2'])
        lq_w2 = self._resize_video(lq, reference_size)
        base_w2 = self._resize_video(base, reference_size)
        appearance = torch.cat([
            lq_w2,
            base_w2,
            (lq_w2 - base_w2).abs(),
        ], dim=2)
        lq_flat, (batch, frames) = _flatten_video(lq_w2)
        base_flat, _ = _flatten_video(base_w2)
        lq_highpass = lq_flat - F.avg_pool2d(
            lq_flat, 3, 1, 1)
        base_highpass = base_flat - F.avg_pool2d(
            base_flat, 3, 1, 1)
        frequency = torch.cat([
            lq_highpass,
            base_highpass,
            (lq_highpass - base_highpass).abs(),
        ], dim=1)
        temporal = self._temporal_evidence(
            lq_w2,
            base_w2,
            flows_forward,
            flows_backward)
        appearance_flat, _ = _flatten_video(appearance)
        temporal_flat, _ = _flatten_video(temporal)
        context_w2 = self.fusion(torch.cat([
            self.static_encoder(torch.cat([
                appearance_flat,
                frequency,
            ], dim=1)),
            self.temporal_encoder(temporal_flat),
        ], dim=1))
        contexts = {}
        for level, size in spatial_sizes.items():
            if tuple(size) == reference_size:
                context = context_w2
            else:
                context = F.interpolate(
                    context_w2,
                    size=tuple(size),
                    mode='bilinear',
                    align_corners=False)
            contexts[level] = _restore_video(context, batch, frames)
        return {'contexts': contexts}
