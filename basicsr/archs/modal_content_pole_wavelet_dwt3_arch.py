"""Official dense propagation with detached pole evidence for direct DWT-3."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.arch_util import (
    PixelShufflePack,
    ResidualBlocksWithInputConv,
    flow_warp,
)
from basicsr.archs.official_dense_components import (
    LocalBidirectionalAttention,
    ResidualUnit,
    dwt3_bior44,
    idwt_level_bior44,
)
from basicsr.archs.modal_content_dwt3_components import (
    CompactPostDegradationContext,
    FixedStablePoles,
    InterStageDWTReanchor,
    ModalConditionedAlignment,
    ModalConditionProjector,
    ModalContentPoleHistoryCell,
    ModalResidueRefiner,
    Stage1PreviewHead,
    VideoPreContext,
)
from basicsr.archs.spynet_arch import SpyNet


def _pad_video_to_multiple(video, multiple):
    batch, frames, channels, height, width = video.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return video, (height, width)
    flattened = video.reshape(batch * frames, channels, height, width)
    padded = F.pad(
        flattened,
        (0, pad_width, 0, pad_height),
        mode='reflect')
    return padded.reshape(
        batch,
        frames,
        channels,
        height + pad_height,
        width + pad_width), (height, width)




class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, 1, dilation, dilation=dilation),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels, channels, 3, 1, dilation, dilation=dilation))

    def forward(self, value):
        return value + self.body(value)


class BlurPoolDownsample(nn.Module):
    """Fixed anti-aliasing blur followed by learned stride-two projection."""

    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1., 4., 6., 4., 1.])
        kernel = (kernel[:, None] * kernel[None, :]) / 256.0
        self.register_buffer(
            'blur_kernel',
            kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1))
        self.channels = int(channels)
        self.projection = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, value):
        value = F.conv2d(
            value,
            self.blur_kernel.to(dtype=value.dtype),
            padding=2,
            groups=self.channels)
        return self.projection(value)


class PoleWaveletRouter(nn.Module):
    """Route detached pole evidence mixed by trainable direction weights."""

    _LEVELS = ('w3', 'w2', 'w1')
    _ORIENTATIONS = ('lh', 'hl', 'hh')

    def __init__(
            self,
            num_poles=16,
            state_channels=2,
            context_channels=16,
            router_channels=32,
            eps=1e-6):
        super().__init__()
        self.num_poles = int(num_poles)
        self.state_channels = int(state_channels)
        self.router_channels = int(router_channels)
        self.eps = float(eps)
        self.state_projection = nn.Conv2d(
            2 * state_channels, 2 * router_channels, 1)
        orientation_evidence_channels = 4 * 3 + context_channels
        self.query = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(
                    orientation_evidence_channels,
                    router_channels,
                    3,
                    1,
                    1),
                nn.SiLU(inplace=True),
                nn.Conv2d(router_channels, router_channels, 1))
            for level in self._LEVELS
        })
        self.w3_adapter = BlurPoolDownsample(router_channels)
        self.w1_band_guide = nn.Sequential(
            nn.Conv2d(
                orientation_evidence_channels,
                router_channels,
                3,
                1,
                1),
            nn.SiLU(inplace=True),
            ResidualUnit(router_channels))
        self.w1_fusion = nn.Sequential(
            nn.Conv2d(2 * router_channels, router_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(router_channels))

    @staticmethod
    def _orientation_evidence(base, lq, context):
        evidence = []
        for base_band, lq_band in zip(
                torch.chunk(base, 3, dim=2),
                torch.chunk(lq, 3, dim=2)):
            difference = base_band - lq_band
            evidence.append(torch.cat([
                base_band,
                lq_band,
                difference,
                difference.abs(),
                context,
            ], dim=2))
        return evidence

    @staticmethod
    def _flatten_video(value):
        batch, frames, channels, height, width = value.shape
        return value.reshape(batch * frames, channels, height, width)

    def _stack_states(self, router_frames, device, dtype):
        # Detach raw evidence at collection, not this weighted mixture: its
        # only upstream trainable path is the isolated direction scorer.
        real = torch.stack([
            frame['real'] for frame in router_frames
        ], dim=1).to(device=device, dtype=dtype)
        imag = torch.stack([
            frame['imag'] for frame in router_frames
        ], dim=1).to(device=device, dtype=dtype)
        reliability = torch.stack([
            frame['reliability'] for frame in router_frames
        ], dim=1).to(device=device, dtype=dtype)
        batch, frames, poles, channels, height, width = real.shape
        projected = self.state_projection(torch.cat([
            real,
            imag,
        ], dim=3).reshape(
            batch * frames * poles,
            2 * channels,
            height,
            width)).reshape(
                batch * frames,
                poles,
                2 * self.router_channels,
                height,
                width)
        keys, values = torch.chunk(projected, 2, dim=2)
        reliability = reliability.reshape(
            batch * frames, poles, height, width)
        return keys, values, reliability, (height, width)

    def forward(
            self,
            level,
            base_band,
            lq_band,
            post_context,
            router_frames):
        evidence = self._orientation_evidence(
            base_band, lq_band, post_context)
        flat_evidence = [self._flatten_video(value) for value in evidence]
        keys, values, reliability, master_size = self._stack_states(
            router_frames,
            base_band.device,
            base_band.dtype)
        queries = []
        for value in flat_evidence:
            query = self.query[level](value)
            if query.shape[-2:] != master_size:
                query = F.interpolate(
                    query,
                    size=master_size,
                    mode='bilinear',
                    align_corners=False)
            queries.append(query)
        query = torch.stack(queries, dim=1)
        logits = torch.einsum(
            'bodhw,bkdhw->bokhw', query, keys) / math.sqrt(
                self.router_channels)
        logits = logits + torch.log(
            reliability.clamp_min(self.eps)).unsqueeze(1)
        alpha = torch.softmax(logits, dim=2)
        context = torch.einsum(
            'bokhw,bkhw,bkdhw->bodhw',
            alpha,
            reliability,
            values)

        batch, frames = base_band.shape[:2]
        native_size = base_band.shape[-2:]
        adapted = []
        for orientation in range(3):
            value = context[:, orientation]
            if level == 'w3':
                value = self.w3_adapter(value)
            elif level == 'w1':
                value = F.interpolate(
                    value,
                    size=native_size,
                    mode='bilinear',
                    align_corners=False)
                guide = self.w1_band_guide(flat_evidence[orientation])
                value = self.w1_fusion(torch.cat([value, guide], dim=1))
            adapted.append(value)
        adapted = torch.stack(adapted, dim=1).reshape(
            batch,
            frames,
            3,
            self.router_channels,
            native_size[0],
            native_size[1])

        stats = {}
        flat_alpha = alpha.detach()
        top_indices = flat_alpha.argmax(dim=2)
        for orientation, name in enumerate(self._ORIENTATIONS):
            orientation_alpha = flat_alpha[:, orientation]
            entropy = -(
                orientation_alpha
                * torch.log(orientation_alpha.clamp_min(self.eps))
            ).sum(dim=1).mean()
            histogram = F.one_hot(
                top_indices[:, orientation],
                num_classes=self.num_poles).float().mean(dim=(0, 1, 2))
            effective = 1.0 / histogram.square().sum().clamp_min(self.eps)
            stats.update({
                f'router_{level}_{name}_entropy': entropy,
                f'router_{level}_{name}_top1_occupancy': histogram.max(),
                f'router_{level}_{name}_effective_modes': effective,
                f'router_{level}_{name}_context_abs_mean': (
                    adapted[:, :, orientation].detach().abs().mean()),
            })
        return adapted, stats, alpha


class DirectWaveletExpert(nn.Module):
    """One scale-specific expert with independent orientation RGB heads."""

    def __init__(
            self,
            input_channels=52,
            hidden_channels=96,
            context_channels=32,
            token_channels=64,
            dilations=(1, 1)):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True))
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(hidden_channels, dilation)
            for dilation in dilations
        ])
        self.token_film = nn.Linear(
            2 * token_channels,
            3 * 2 * hidden_channels)
        self.spatial_film = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(context_channels, hidden_channels, 3, 1, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 2 * hidden_channels, 1))
            for _ in range(3)
        ])
        self.residual_heads = nn.ModuleList([
            nn.Conv2d(hidden_channels, 3, 3, 1, 1)
            for _ in range(3)
        ])
        nn.init.constant_(self.token_film.weight, 0.0)
        nn.init.constant_(self.token_film.bias, 0.0)
        for film in self.spatial_film:
            nn.init.normal_(film[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(film[-1].bias, 0.0)
        for head in self.residual_heads:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, evidence, pole_context, frame_token, clip_token):
        batch, frames, _, height, width = evidence.shape
        feature = self.blocks(self.input_encoder(evidence.reshape(
            batch * frames,
            evidence.shape[2],
            height,
            width)))
        clip = clip_token.unsqueeze(1).expand(-1, frames, -1)
        token_film = self.token_film(torch.cat([
            frame_token,
            clip,
        ], dim=-1)).reshape(
            batch * frames,
            3,
            2,
            self.hidden_channels,
            1,
            1)
        contexts = pole_context.reshape(
            batch * frames,
            3,
            pole_context.shape[3],
            height,
            width)
        residuals = []
        for orientation in range(3):
            spatial_gamma, spatial_beta = torch.chunk(
                self.spatial_film[orientation](
                    contexts[:, orientation]),
                2,
                dim=1)
            token_gamma = token_film[:, orientation, 0]
            token_beta = token_film[:, orientation, 1]
            conditioned = (
                feature * (1.0 + token_gamma + spatial_gamma)
                + token_beta
                + spatial_beta)
            residuals.append(
                self.residual_heads[orientation](conditioned))
        return torch.cat(residuals, dim=1).reshape(
            batch, frames, 9, height, width)


class AverNetModalContentPoleWaveletDWT3OfficialDense(nn.Module):
    """Generator trained by job 6220553."""

    _BRANCHES = ('backward_1', 'forward_1', 'backward_2', 'forward_2')
    _STAGE_BRANCHES = (
        ('backward_1', 'forward_1'),
        ('backward_2', 'forward_2'),
    )
    _FUSION_KEYS = ('stage_1', 'stage_2')

    def __init__(
            self,
            mid_channels,
            num_blocks,
            max_residue_magnitude,
            spynet_pretrained,
            deform_groups,
            num_poles,
            pole_state_channels,
            pole_mixer_channels,
            pole_delta_t,
            pole_decay_eps,
            pole_flow_alpha,
            pole_flow_beta,
            pole_spatial_temperature,
            pole_scene_reset_threshold,
            pole_scene_reset_temperature,
            pole_omega_max,
            pole_separation_bandwidth,
            pole_dead_occupancy_floor,
            memory_fusion_init,
            direction_attention_channels,
            direction_attention_eps,
            pre_context_branch_channels,
            pre_context_channels,
            context_token_channels,
            modal_condition_channels,
            modal_model_channels,
            modal_num_heads,
            modal_num_blocks,
            post_context_channels,
            post_context_branch_channels,
            wavelet_hidden_channels,
            router_channels,
            inter_stage_demand_scale,
            wavelet_infer_chunk):
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.spynet = SpyNet(load_path=spynet_pretrained)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, self.mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(
                self.mid_channels, self.mid_channels, 5))
        self.backbone = nn.ModuleDict({
            branch: ResidualBlocksWithInputConv(
                (2 + index) * self.mid_channels,
                self.mid_channels,
                num_blocks)
            for index, branch in enumerate(self._BRANCHES)
        })
        self.direction_attention = nn.ModuleDict({
            stage: LocalBidirectionalAttention(
                channels=self.mid_channels,
                hidden_channels=direction_attention_channels,
                eps=direction_attention_eps)
            for stage in self._FUSION_KEYS
        })
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * self.mid_channels, self.mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            self.mid_channels, self.mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            self.mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self._attention_debug_stats = {}
        self._last_direction_weights = {}
        self.num_poles = int(num_poles)
        self.pole_state_channels = int(pole_state_channels)
        self.deform_groups = int(deform_groups)
        self.pole_dead_occupancy_floor = float(pole_dead_occupancy_floor)
        self.wavelet_infer_chunk = int(wavelet_infer_chunk)
        for module in self.spynet.basic_module[:4]:
            module.requires_grad_(False)

        self.pre_context = VideoPreContext(
            branch_channels=pre_context_branch_channels,
            context_channels=pre_context_channels,
            token_channels=context_token_channels)
        self.modal_condition_projector = ModalConditionProjector(
            spatial_channels=pre_context_channels,
            token_channels=context_token_channels,
            output_channels=modal_condition_channels)
        self.deform_align = nn.ModuleDict({
            branch: ModalConditionedAlignment(
                self.mid_channels,
                self.mid_channels,
                3,
                padding=1,
                deform_groups=self.deform_groups,
                modal_channels=modal_condition_channels,
                max_residue_magnitude=max_residue_magnitude)
            for branch in self._BRANCHES
        })
        self.stage1_preview = Stage1PreviewHead(channels=mid_channels)
        self.inter_stage_reanchor = InterStageDWTReanchor(
            channels=mid_channels,
            detail_channels=9,
            demand_scale=inter_stage_demand_scale)
        self.shared_poles = FixedStablePoles(
            num_poles=num_poles,
            delta_t=pole_delta_t,
            decay_eps=pole_decay_eps,
            omega_max=pole_omega_max,
            separation_bandwidth=pole_separation_bandwidth)
        self.modal_refiner = ModalResidueRefiner(
            num_poles=num_poles,
            state_channels=pole_state_channels,
            context_channels=context_token_channels,
            model_channels=modal_model_channels,
            num_heads=modal_num_heads,
            num_blocks=modal_num_blocks,
            num_branches=len(self._BRANCHES))
        self.history_cells = nn.ModuleDict({
            branch: ModalContentPoleHistoryCell(
                channels=mid_channels,
                num_poles=num_poles,
                state_channels=pole_state_channels,
                token_channels=context_token_channels,
                mixer_channels=pole_mixer_channels,
                decay_eps=pole_decay_eps,
                flow_alpha=pole_flow_alpha,
                flow_beta=pole_flow_beta,
                spatial_temperature=pole_spatial_temperature,
                scene_reset_threshold=pole_scene_reset_threshold,
                scene_reset_temperature=pole_scene_reset_temperature)
            for branch in self._BRANCHES
        })
        self.memory_fusion = nn.ParameterDict({
            branch: nn.Parameter(torch.atanh(torch.tensor(
                float(memory_fusion_init))))
            for branch in self._BRANCHES
        })
        self.post_degradation_context = CompactPostDegradationContext(
            channels=post_context_channels,
            branch_channels=post_context_branch_channels,
            flow_eps=pole_decay_eps)
        self.pole_wavelet_router = PoleWaveletRouter(
            num_poles=num_poles,
            state_channels=pole_state_channels,
            context_channels=post_context_channels,
            router_channels=router_channels,
            eps=pole_decay_eps)
        expert_input_channels = 4 * 9 + post_context_channels
        self.wavelet_experts = nn.ModuleDict({
            'w3': DirectWaveletExpert(
                input_channels=expert_input_channels,
                hidden_channels=wavelet_hidden_channels,
                context_channels=router_channels,
                token_channels=context_token_channels,
                dilations=(1, 2)),
            'w2': DirectWaveletExpert(
                input_channels=expert_input_channels,
                hidden_channels=wavelet_hidden_channels,
                context_channels=router_channels,
                token_channels=context_token_channels,
                dilations=(1, 1, 1, 1)),
            'w1': DirectWaveletExpert(
                input_channels=expert_input_channels,
                hidden_channels=wavelet_hidden_channels,
                context_channels=router_channels,
                token_channels=context_token_channels,
                dilations=(1, 1)),
        })
        # Wavelet-target EMA is model state, not wrapper-only state, so strict
        # generator checkpoint resume restores normalization exactly.
        self.register_buffer(
            'wavelet_residual_energy_ema',
            torch.zeros(3, 3))
        self.register_buffer(
            'wavelet_residual_energy_initialized',
            torch.tensor(False, dtype=torch.bool))

        self._branch_debug_stats = {}
        self._branch_occupancies = {}
        self._stage2_router_branches = {}
        self._last_pole_router_frames = None
        self._last_pole_sequence = None
        self._last_frame_tokens = None
        self._last_clip_token = None
        self._last_summary_tokens = None
        self._last_stage1_preview = None
        self._context_flows_forward = None
        self._context_flows_backward = None
        self.last_aux_losses = {}
        self.last_temporal_stats = {}
        self.last_wavelet_stats = {}

    def compute_flow(self, lqs):
        batch, frames, channels, height, width = lqs.shape
        first = lqs[:, :-1].reshape(-1, channels, height, width)
        second = lqs[:, 1:].reshape(-1, channels, height, width)
        flows_backward = self.spynet(first, second).reshape(
            batch, frames - 1, 2, height, width)
        flows_forward = self.spynet(second, first).reshape(
            batch, frames - 1, 2, height, width)
        return flows_forward, flows_backward

    def upsample(self, lqs, features):
        outputs = []
        for frame in range(lqs.shape[1]):
            feature = torch.cat([
                features['spatial'][frame],
                *(features[branch][frame] for branch in self._BRANCHES),
            ], dim=1)
            feature = self.reconstruction(feature)
            feature = self.lrelu(self.upsample1(feature))
            feature = self.lrelu(self.upsample2(feature))
            feature = self.lrelu(self.conv_hr(feature))
            output = self.conv_last(feature) + lqs[:, frame]
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def _mean_stats(stats):
        return {
            name: torch.stack([item[name] for item in stats]).mean()
            for name in stats[0]
        }

    @staticmethod
    def _retention_to_tau(retention):
        retention = retention.clamp(0.0, 1.0 - 1e-7)
        return torch.where(
            retention > 0,
            -1.0 / torch.log(retention.clamp_min(1e-12)),
            torch.zeros_like(retention))

    def _fuse_bidirectional_stage(
            self,
            stage_key,
            reference_features,
            backward_features,
            forward_features,
            backward_trust,
            forward_trust):
        fused_features = []
        direction_weights = []
        frame_stats = []
        attention = self.direction_attention[stage_key]
        for frame in range(len(reference_features)):
            reference = reference_features[frame]
            backward = backward_features[frame]
            forward = forward_features[frame]
            backward_reliability = backward_trust[frame]
            forward_reliability = forward_trust[frame]
            attention_inputs = (
                reference, backward, forward,
                backward_reliability, forward_reliability)
            if stage_key == 'stage_2':
                # Dense RGB reconstruction bypasses this fusion. Train its
                # direction scorer through the router, never through recurrence.
                attention_inputs = tuple(value.detach() for value in attention_inputs)
            output = attention(*attention_inputs)
            weights = output['weights']
            frame_stats.append({
                'current_weight_mean': weights[:, 0].detach().mean(),
                'backward_weight_mean': weights[:, 1].detach().mean(),
                'forward_weight_mean': weights[:, 2].detach().mean(),
                'weight_entropy': (
                    -weights.detach()
                    * torch.log(weights.detach().clamp_min(1e-8))
                ).sum(dim=1).mean(),
                'direction_margin_abs_mean': (
                    weights[:, 1].detach()
                    - weights[:, 2].detach()).abs().mean(),
                'backward_reliability_mean': (
                    backward_reliability.detach().mean()),
                'forward_reliability_mean': (
                    forward_reliability.detach().mean()),
            })
            fused = output['fused']
            saved_weights = weights if stage_key == 'stage_2' else weights.detach()
            fused_features.append(fused)
            direction_weights.append(saved_weights)
        self._attention_debug_stats[stage_key] = self._mean_stats(frame_stats)
        self._last_direction_weights[stage_key] = direction_weights
        return fused_features

    @staticmethod
    def _frame_poles(pole_sequence, frame):
        return {
            name: pole_sequence[name][:, frame]
            for name in (
                'lambdas',
                'omegas',
                'a_real',
                'a_imag',
                'phi_real',
                'phi_imag')
        }

    @staticmethod
    def _detach_router_state(state):
        return {
            name: value.detach()
            for name, value in state.items()
        }

    def _propagate_branch(
            self,
            branch,
            spatial_features,
            completed_branches,
            stage_condition,
            modal_alignment,
            inter_stage_demands,
            flows_forward,
            flows_backward,
            pole_sequence,
            frame_tokens,
            summary_tokens):
        direction = branch.split('_', 1)[0]
        if direction == 'backward':
            flows = flows_backward
            reverse_flows = flows_forward
        else:
            flows = flows_forward
            reverse_flows = flows_backward
        batch, flow_frames, _, feature_height, feature_width = flows.shape
        frame_indices = list(range(flow_frames + 1))
        flow_indices = list(range(-1, flow_frames))
        if direction == 'backward':
            frame_indices = frame_indices[::-1]
            flow_indices = frame_indices

        propagated = flows.new_zeros(
            batch, self.mid_channels, feature_height, feature_width)
        state = None
        branch_features = []
        branch_trust = []
        router_states = []
        frame_stats = []
        frame_occupancies = []
        branch_index = self._BRANCHES.index(branch)
        for index, frame in enumerate(frame_indices):
            current = spatial_features[frame]
            alignment_condition = modal_alignment[frame]
            demand = (
                None
                if inter_stage_demands is None
                else inter_stage_demands[frame])
            condition = (
                None
                if stage_condition is None
                else stage_condition[frame])
            previous = propagated.clone()
            if index > 0:
                pair_index = flow_indices[index]
                flow = flows[:, pair_index]
                reverse_flow = reverse_flows[:, pair_index]
                spatial_previous = spatial_features[frame_indices[index - 1]]
                grid_flow = flow.permute(0, 2, 3, 1)
                spatial_aligned = flow_warp(
                    spatial_previous, grid_flow, padding_mode='zeros')
                recurrent_aligned = flow_warp(
                    propagated, grid_flow, padding_mode='zeros')
                deform_aligned = self.deform_align[branch](
                    previous,
                    torch.cat([recurrent_aligned, current], dim=1),
                    flow,
                    alignment_condition)
            else:
                flow = None
                reverse_flow = None
                spatial_aligned = current
                deform_aligned = propagated

            current_poles = self._frame_poles(pole_sequence, frame)
            memory, state, propagation_trust = self.history_cells[branch](
                current,
                deform_aligned,
                spatial_aligned,
                state,
                flow,
                reverse_flow,
                current_poles,
                frame_tokens[:, frame],
                summary_tokens,
                self.modal_refiner,
                branch_index)
            temporal_context = (
                propagation_trust * deform_aligned
                + torch.tanh(self.memory_fusion[branch]) * memory)
            reanchor_delta = temporal_context.new_zeros(())
            if demand is not None:
                reanchor_delta = self.inter_stage_reanchor(
                    temporal_context, condition, demand)
                temporal_context = temporal_context + reanchor_delta
            stats = dict(self.history_cells[branch].last_debug_stats)
            stats.update({
                'modal_alignment_abs_mean': (
                    alignment_condition.detach().abs().mean()),
                'inter_stage_reanchor_abs_mean': (
                    reanchor_delta.detach().abs().mean()),
            })
            frame_stats.append(stats)
            frame_occupancies.append(
                self.history_cells[branch].last_occupancy)
            if branch.endswith('_2'):
                router_states.append(self._detach_router_state(
                    self.history_cells[branch].last_router_state))

            backbone_inputs = [current]
            # Same order as official AverNet: current, all earlier passes,
            # then this branch's aligned recurrent/pole context. No detach.
            backbone_inputs.extend(
                completed_branches[previous_branch][frame]
                for previous_branch in self._BRANCHES[:branch_index])
            backbone_inputs.append(temporal_context)
            propagated = temporal_context + self.backbone[branch](
                torch.cat(backbone_inputs, dim=1))
            branch_features.append(propagated)
            branch_trust.append(propagation_trust)

        if direction == 'backward':
            branch_features = branch_features[::-1]
            branch_trust = branch_trust[::-1]
            router_states = router_states[::-1]
        self._branch_debug_stats[branch] = self._mean_stats(frame_stats)
        self._branch_occupancies[branch] = torch.stack(
            frame_occupancies, dim=1).mean(dim=(0, 1))
        if branch.endswith('_2'):
            self._stage2_router_branches[branch] = router_states
        return branch_features, branch_trust

    def _build_inter_stage_demand(
            self, lqs, spatial_features, stage1_features):
        previews = []
        demands = []
        for frame, (spatial_feature, stage1_feature) in enumerate(zip(
                spatial_features, stage1_features)):
            lq_frame = lqs[:, frame:frame + 1]
            preview = self.stage1_preview(
                lq_frame[:, 0], spatial_feature, stage1_feature).unsqueeze(1)
            lq_pyramid = dwt3_bior44(lq_frame)
            preview_pyramid = dwt3_bior44(preview)
            signed_details = {
                level: preview_pyramid[level][:, 0] - lq_pyramid[level][:, 0]
                for level in ('w3', 'w2', 'w1')
            }
            demand = self.inter_stage_reanchor.encode_demand(
                signed_details, stage1_feature.shape[-2:])
            previews.append(preview[:, 0])
            demands.append(demand)
        return torch.stack(previews, dim=1), demands

    def _combine_stage2_router_states(self):
        backward = self._stage2_router_branches['backward_2']
        forward = self._stage2_router_branches['forward_2']
        weights = self._last_direction_weights['stage_2']
        combined = []
        for backward_frame, forward_frame, frame_weights in zip(
                backward, forward, weights):
            weight_backward = frame_weights[:, 1:2]
            weight_forward = frame_weights[:, 2:3]
            state_weight_backward = weight_backward.unsqueeze(1)
            state_weight_forward = weight_forward.unsqueeze(1)
            real = (
                state_weight_backward * backward_frame['real']
                + state_weight_forward * forward_frame['real'])
            imag = (
                state_weight_backward * backward_frame['imag']
                + state_weight_forward * forward_frame['imag'])
            reliability_backward = (
                weight_backward
                * backward_frame['confidence']
                * backward_frame['history_strength'])
            reliability_forward = (
                weight_forward
                * forward_frame['confidence']
                * forward_frame['history_strength'])
            reliability = (
                reliability_backward
                * backward_frame['read_gate']
                + reliability_forward
                * forward_frame['read_gate']).clamp(0.0, 1.0)
            combined.append({
                # States/gates are already detached when collected. Keep only
                # the Stage-2 direction scorer's gradient through this mixture.
                'real': real,
                'imag': imag,
                'reliability': reliability,
            })
        self._last_pole_router_frames = combined
        self._stage2_router_branches = {}

    def _collect_temporal_stats(self, reference):
        averaged = self._mean_stats(list(self._branch_debug_stats.values()))
        stats = {
            f'pole_mixer_{name}': value
            for name, value in averaged.items()
        }
        for stage, stage_stats in self._attention_debug_stats.items():
            stats.update({
                f'bidirectional_attention_{stage}_{name}': value
                for name, value in stage_stats.items()
            })
        transition = torch.sqrt(
            self._last_pole_sequence['a_real'].square()
            + self._last_pole_sequence['a_imag'].square()).detach()
        occupancy = torch.stack(
            list(self._branch_occupancies.values()), dim=0).detach().mean(dim=0)
        stats.update({
            'pole_mixer_enabled': reference.new_tensor(1.0),
            'pole_mixer_num_poles': reference.new_tensor(float(self.num_poles)),
            'pole_mixer_state_channels': reference.new_tensor(
                float(self.pole_state_channels)),
            'pole_mixer_tau_frames_mean': self._retention_to_tau(
                transition).mean(),
            'pole_mixer_occupancy_mean': occupancy.mean(),
            'pole_mixer_occupancy_min': occupancy.min(),
            'pole_mixer_content_adaptation_enabled': reference.new_tensor(0.0),
            'stage2_pole_states_detached': reference.new_tensor(1.0),
            'stage2_router_direction_weights_trainable': reference.new_tensor(1.0),
        })
        stats.update({
            f'pole_mixer_{name}': value
            for name, value in self._last_pole_sequence['stats'].items()
        })
        return stats

    def prompt_free_backbone(self, lqs):
        self._branch_debug_stats = {}
        self._branch_occupancies = {}
        self._attention_debug_stats = {}
        self._last_direction_weights = {}
        self._stage2_router_branches = {}
        batch, frames, channels, height, width = lqs.shape
        lqs_downsample = F.interpolate(
            lqs.reshape(-1, channels, height, width),
            scale_factor=0.25,
            mode='bicubic',
            align_corners=False).reshape(
                batch, frames, channels, height // 4, width // 4)
        spatial = self.feat_extract(
            lqs.reshape(-1, channels, height, width)).reshape(
                batch,
                frames,
                self.mid_channels,
                height // 4,
                width // 4)
        features = {
            'spatial': [spatial[:, frame] for frame in range(frames)]
        }

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        self._context_flows_forward = flows_forward
        self._context_flows_backward = flows_backward

        pre_context = self.pre_context(lqs_downsample)
        frame_tokens = pre_context['frame_tokens']
        clip_token = pre_context['clip_token']
        summary_tokens = pre_context['summary_tokens']
        modal_alignment_tensor = self.modal_condition_projector(
            pre_context['spatial'], frame_tokens, clip_token)
        modal_alignment = [
            modal_alignment_tensor[:, frame] for frame in range(frames)
        ]
        pole_sequence = self.shared_poles(frame_tokens, clip_token)
        self._last_pole_sequence = pole_sequence
        self._last_frame_tokens = frame_tokens
        self._last_clip_token = clip_token
        self._last_summary_tokens = summary_tokens

        branch_features = {}
        branch_trust = {}
        fused_stages = {}
        inter_stage_demands = None
        for stage_index, stage_branches in enumerate(
                self._STAGE_BRANCHES, start=1):
            stage_condition = (
                None if stage_index == 1 else fused_stages['stage_1'])
            for branch in stage_branches:
                branch_features[branch], branch_trust[branch] = (
                    self._propagate_branch(
                        branch,
                        features['spatial'],
                        branch_features,
                        stage_condition,
                        modal_alignment,
                        inter_stage_demands,
                        flows_forward,
                        flows_backward,
                        pole_sequence,
                        frame_tokens,
                        summary_tokens))
            stage_key = f'stage_{stage_index}'
            reference_features = (
                features['spatial']
                if stage_index == 1
                else fused_stages['stage_1'])
            backward_branch, forward_branch = stage_branches
            fused_stages[stage_key] = self._fuse_bidirectional_stage(
                stage_key,
                reference_features,
                branch_features[backward_branch],
                branch_features[forward_branch],
                branch_trust[backward_branch],
                branch_trust[forward_branch])
            if stage_index == 1:
                self._last_stage1_preview, inter_stage_demands = (
                    self._build_inter_stage_demand(
                        lqs,
                        features['spatial'],
                        fused_stages['stage_1']))
        self._combine_stage2_router_states()

        occupancy = torch.stack(
            list(self._branch_occupancies.values()), dim=0).mean(dim=0)
        pole_dead = F.relu(
            1.0
            - occupancy / occupancy.new_tensor(
                self.pole_dead_occupancy_floor)).square().mean()
        self.last_aux_losses = {
            **pole_sequence['aux_losses'],
            'pole_dead_occupancy': pole_dead,
        }
        features.update(fused_stages)
        features.update(branch_features)
        restored = self.upsample(lqs, features)
        self.last_temporal_stats = self._collect_temporal_stats(restored)
        return restored

    def _tokens_for_chunk(self, lqs, frame_start):
        end = frame_start + lqs.shape[1]
        return (
            self._last_frame_tokens[:, frame_start:end],
            self._last_clip_token)

    def _router_frames_for_chunk(self, frame_start, frames):
        return self._last_pole_router_frames[frame_start:frame_start + frames]

    def _flows_for_chunk(self, frame_start, frames):
        end = frame_start + frames - 1
        return (
            self._context_flows_forward[:, frame_start:end],
            self._context_flows_backward[:, frame_start:end])

    @staticmethod
    def _expert_evidence(base_band, lq_band, context):
        difference = base_band - lq_band
        return torch.cat([
            base_band,
            lq_band,
            difference,
            difference.abs(),
            context,
        ], dim=2)

    def _wavelet_refine_chunk(
            self, lqs, base_restored, gt=None, frame_start=0):
        lq_pyramid = dwt3_bior44(lqs)
        base_pyramid = dwt3_bior44(base_restored)
        spatial_sizes = {
            level: base_pyramid[level].shape[-2:]
            for level in ('w3', 'w2', 'w1')
        }
        flows_forward, flows_backward = self._flows_for_chunk(
            frame_start, lqs.shape[1])
        degradation = self.post_degradation_context(
            lqs,
            base_restored.detach(),
            spatial_sizes,
            flows_forward=flows_forward,
            flows_backward=flows_backward)
        frame_tokens, clip_token = self._tokens_for_chunk(lqs, frame_start)
        router_frames = self._router_frames_for_chunk(
            frame_start, lqs.shape[1])

        predictions = {}
        residuals = {}
        router_stats = {}
        router_alphas = {}
        for level in ('w3', 'w2', 'w1'):
            pole_context, stats, alpha = self.pole_wavelet_router(
                level,
                base_pyramid[level],
                lq_pyramid[level],
                degradation['contexts'][level],
                router_frames)
            evidence = self._expert_evidence(
                base_pyramid[level],
                lq_pyramid[level],
                degradation['contexts'][level])
            residual = self.wavelet_experts[level](
                evidence,
                pole_context,
                frame_tokens,
                clip_token)
            residuals[level] = residual
            predictions[level] = base_pyramid[level] + residual
            router_stats.update(stats)
            router_alphas[level] = alpha

        a2_pred = idwt_level_bior44(
            base_pyramid['a3'], predictions['w3'])
        a1_pred = idwt_level_bior44(a2_pred, predictions['w2'])
        restored = idwt_level_bior44(a1_pred, predictions['w1'])
        outputs = {
            'restored': restored,
            'base_restored': base_restored,
            'a2_pred': a2_pred,
            'a1_pred': a1_pred,
            'aux_losses': dict(self.last_aux_losses),
            'router_stats': router_stats,
            'router_alphas': router_alphas,
        }
        for level in ('w3', 'w2', 'w1'):
            outputs[f'{level}_pred'] = predictions[level]
            outputs[f'{level}_base'] = base_pyramid[level]
            outputs[f'{level}_residual'] = residuals[level]
        if gt is not None:
            target_pyramid = dwt3_bior44(gt)
            for level in ('w3', 'w2', 'w1'):
                outputs[f'{level}_target'] = target_pyramid[level]
        return outputs

    def _wavelet_refine(self, lqs, base_restored, gt=None):
        if gt is not None:
            return self._wavelet_refine_chunk(
                lqs, base_restored, gt=gt, frame_start=0)
        restored_chunks = []
        stat_chunks = []
        for start in range(0, lqs.shape[1], self.wavelet_infer_chunk):
            end = min(start + self.wavelet_infer_chunk, lqs.shape[1])
            extended_start = max(0, start - 1)
            extended_end = min(lqs.shape[1], end + 1)
            chunk_lqs = lqs[:, extended_start:extended_end]
            chunk_base = base_restored[:, extended_start:extended_end]
            chunk = self._wavelet_refine_chunk(
                chunk_lqs,
                chunk_base,
                frame_start=extended_start)
            local_start = start - extended_start
            local_end = local_start + end - start
            restored_chunks.append(
                chunk['restored'][:, local_start:local_end])
            stat_chunks.append(chunk['router_stats'])
        stats = {
            name: torch.stack([chunk[name] for chunk in stat_chunks]).mean()
            for name in stat_chunks[0]
        }
        return {
            'restored': torch.cat(restored_chunks, dim=1),
            'base_restored': base_restored,
            'router_stats': stats,
        }

    def forward(self, lqs, gt=None):
        padded_lqs, original_size = _pad_video_to_multiple(lqs, 8)
        padded_gt = None
        if gt is not None:
            padded_gt, _ = _pad_video_to_multiple(gt, 8)
        base_restored = self.prompt_free_backbone(padded_lqs)
        wavelet_outputs = self._wavelet_refine(
            padded_lqs, base_restored, gt=padded_gt)
        restored = wavelet_outputs['restored'][
            ..., :original_size[0], :original_size[1]]
        base_cropped = base_restored[
            ..., :original_size[0], :original_size[1]]
        preview = self._last_stage1_preview[
            ..., :original_size[0], :original_size[1]]
        self.last_wavelet_stats = {
            'dwt3_direct_residual_enabled': restored.new_tensor(1.0),
            'dwt3_pole_router_detached': restored.new_tensor(0.0),
            **wavelet_outputs['router_stats'],
        }
        outputs = dict(wavelet_outputs)
        outputs.update({
            'restored': restored,
            'base_restored': base_cropped,
            'stage1_preview': preview,
            'student_flow_to_previous': self._context_flows_forward,
            'student_flow_to_next': self._context_flows_backward,
            'log_vars': {
                **self.last_temporal_stats,
                **self.last_wavelet_stats,
            },
        })
        return outputs
