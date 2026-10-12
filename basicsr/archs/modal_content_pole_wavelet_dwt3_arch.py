"""Two-round dense video restoration with additive pole-conditioned DWT refinement."""

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.arch_util import (
    PixelShufflePack,
    ResidualBlocksWithInputConv,
    flow_warp,
)
from basicsr.archs.inference_frame_cache import CpuFrameCache
from basicsr.archs.pole_conditioning import (
    AdditivePoleConditioning,
    LocalPoleFusion,
    PreviewConditioning,
    spatial_film,
)
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.temporal_components import (
    CompactPostDegradationContext,
    FixedStablePoles,
    ModalConditionedAlignment,
    ModalConditionProjector,
    ModalContentPoleHistoryCell,
    Stage1PreviewHead,
    VideoPreContext,
)
from basicsr.archs.wavelet_components import (
    LocalBidirectionalAttention,
    ResidualUnit,
    dwt3_bior44,
    idwt_level_bior44,
)


def _pad_video_to_multiple(video, multiple):
    batch, frames, channels, height, width = video.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return video, (height, width)
    flattened = video.reshape(batch * frames, channels, height, width)
    padded = F.pad(flattened, (0, pad_width, 0, pad_height), mode="reflect")
    return padded.reshape(batch, frames, channels, height + pad_height, width + pad_width), (
        height,
        width,
    )


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
        )

    def forward(self, value):
        return value + self.body(value)


class LocalWaveletGuide(nn.Module):
    """Frame-local, orientation-specific guide for a wavelet expert."""

    def __init__(self, post_context_channels=16, guide_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4 * 3 + post_context_channels, guide_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(guide_channels),
        )

    def forward(self, base_band, lq_band, post_context):
        batch, frames = base_band.shape[:2]
        contexts = []
        for base_rgb, lq_rgb in zip(
            torch.chunk(base_band, 3, dim=2), torch.chunk(lq_band, 3, dim=2)
        ):
            difference = base_rgb - lq_rgb
            evidence = torch.cat(
                [
                    base_rgb,
                    lq_rgb,
                    difference,
                    difference.abs(),
                    post_context,
                ],
                dim=2,
            )
            height, width = evidence.shape[-2:]
            contexts.append(
                self.encoder(evidence.reshape(batch * frames, evidence.shape[2], height, width))
            )
        return torch.stack(contexts, dim=1).reshape(
            batch, frames, 3, -1, base_band.shape[-2], base_band.shape[-1]
        )


class DirectWaveletExpert(nn.Module):
    """One scale-specific expert with independent orientation RGB heads."""

    def __init__(
        self,
        input_channels=52,
        hidden_channels=96,
        spatial_context_channels=None,
        token_channels=64,
        dilations=(1, 1),
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 1), nn.SiLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            *[DilatedResidualBlock(hidden_channels, dilation) for dilation in dilations]
        )
        self.token_film = nn.Linear(2 * token_channels, 3 * 2 * hidden_channels)
        self.spatial_film = None
        if spatial_context_channels is not None:
            self.spatial_film = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(spatial_context_channels, hidden_channels, 3, 1, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(hidden_channels, 2 * hidden_channels, 1),
                    )
                    for _ in range(3)
                ]
            )
        self.residual_heads = nn.ModuleList(
            [nn.Conv2d(hidden_channels, 3, 3, 1, 1) for _ in range(3)]
        )
        nn.init.constant_(self.token_film.weight, 0.0)
        nn.init.constant_(self.token_film.bias, 0.0)
        if self.spatial_film is not None:
            for film in self.spatial_film:
                nn.init.normal_(film[-1].weight, mean=0.0, std=1e-3)
                nn.init.constant_(film[-1].bias, 0.0)
        for head in self.residual_heads:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, evidence, frame_token, clip_token, spatial_context=None):
        batch, frames, _, height, width = evidence.shape
        feature = self.blocks(
            self.input_encoder(evidence.reshape(batch * frames, evidence.shape[2], height, width))
        )
        clip = clip_token.unsqueeze(1).expand(-1, frames, -1)
        token_film = self.token_film(
            torch.cat(
                [
                    frame_token,
                    clip,
                ],
                dim=-1,
            )
        ).reshape(batch * frames, 3, 2, self.hidden_channels, 1, 1)
        contexts = None
        if spatial_context is not None:
            contexts = spatial_context.reshape(
                batch * frames, 3, spatial_context.shape[3], height, width
            )
        residuals = []
        for orientation in range(3):
            token_gamma = token_film[:, orientation, 0]
            token_beta = token_film[:, orientation, 1]
            conditioned = feature * (1.0 + token_gamma) + token_beta
            if self.spatial_film is not None:
                if spatial_context is None:
                    raise ValueError("Spatial context is required for the wavelet expert.")
                spatial_gamma, spatial_beta = torch.chunk(
                    self.spatial_film[orientation](contexts[:, orientation]), 2, dim=1
                )
                conditioned = conditioned + feature * spatial_gamma + spatial_beta
            residuals.append(self.residual_heads[orientation](conditioned))
        return torch.cat(residuals, dim=1).reshape(batch, frames, 9, height, width)


class ModalContentPoleWaveletDWT3(nn.Module):
    """Four dense propagation passes, stable pole memory, and three wavelet experts."""

    _BRANCHES = ("backward_1", "forward_1", "backward_2", "forward_2")
    _STAGE_BRANCHES = (
        ("backward_1", "forward_1"),
        ("backward_2", "forward_2"),
    )
    _FUSION_KEYS = ("stage_1",)

    def __init__(
        self,
        mid_channels=96,
        num_blocks=7,
        max_residue_magnitude=10,
        spynet_pretrained=None,
        deform_groups=24,
        num_poles=16,
        pole_state_channels=2,
        pole_mixer_channels=32,
        pole_delta_t=1.0,
        pole_decay_eps=1e-06,
        pole_flow_alpha=0.01,
        pole_flow_beta=0.5,
        pole_spatial_temperature=0.5,
        pole_omega_max=3.141592653589793,
        pole_separation_bandwidth=0.01,
        memory_fusion_init=0.1,
        direction_attention_channels=32,
        direction_attention_eps=1e-06,
        pre_context_branch_channels=16,
        pre_context_channels=32,
        context_token_channels=64,
        modal_condition_channels=96,
        post_context_channels=16,
        post_context_branch_channels=8,
        wavelet_hidden_channels=96,
        w1_guide_channels=32,
        wavelet_infer_chunk=4,
        pole_context_channels=32,
    ):
        super().__init__()
        if int(num_poles) < 2:
            raise ValueError("Pole separation requires at least two poles.")
        if int(mid_channels) % int(deform_groups):
            raise ValueError("mid_channels must be divisible by deform_groups.")
        if int(w1_guide_channels) != int(pole_context_channels):
            raise ValueError("Guide and additive pole context channels must match.")
        self.mid_channels = int(mid_channels)
        self.spynet = SpyNet(load_path=spynet_pretrained)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, self.mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(self.mid_channels, self.mid_channels, 5),
        )
        self.backbone = nn.ModuleDict(
            {
                branch: ResidualBlocksWithInputConv(
                    (2 + index) * self.mid_channels, self.mid_channels, num_blocks
                )
                for index, branch in enumerate(self._BRANCHES)
            }
        )
        self.direction_attention = nn.ModuleDict(
            {
                stage: LocalBidirectionalAttention(
                    channels=self.mid_channels,
                    hidden_channels=direction_attention_channels,
                    eps=direction_attention_eps,
                )
                for stage in self._FUSION_KEYS
            }
        )
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * self.mid_channels, self.mid_channels, 5
        )
        self.upsample1 = PixelShufflePack(
            self.mid_channels, self.mid_channels, 2, upsample_kernel=3
        )
        self.upsample2 = PixelShufflePack(self.mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.num_poles = int(num_poles)
        self.pole_state_channels = int(pole_state_channels)
        self.deform_groups = int(deform_groups)
        self.wavelet_infer_chunk = int(wavelet_infer_chunk)
        if self.wavelet_infer_chunk < 1:
            raise ValueError("wavelet_infer_chunk must be positive.")
        self.inference_cpu_offload = False
        for module in self.spynet.basic_module[:4]:
            module.requires_grad_(False)

        self.pre_context = VideoPreContext(
            branch_channels=pre_context_branch_channels,
            context_channels=pre_context_channels,
            token_channels=context_token_channels,
        )
        self.modal_condition_projector = ModalConditionProjector(
            spatial_channels=pre_context_channels,
            token_channels=context_token_channels,
            output_channels=modal_condition_channels,
        )
        self.deform_align = nn.ModuleDict(
            {
                branch: ModalConditionedAlignment(
                    self.mid_channels,
                    self.mid_channels,
                    3,
                    padding=1,
                    deform_groups=self.deform_groups,
                    modal_channels=modal_condition_channels,
                    max_residue_magnitude=max_residue_magnitude,
                )
                for branch in self._BRANCHES
            }
        )
        self.stage1_preview = Stage1PreviewHead(channels=mid_channels)
        self.shared_poles = FixedStablePoles(
            num_poles=num_poles,
            delta_t=pole_delta_t,
            decay_eps=pole_decay_eps,
            omega_max=pole_omega_max,
            separation_bandwidth=pole_separation_bandwidth,
        )
        self.history_cells = nn.ModuleDict(
            {
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
                )
                for branch in self._BRANCHES
            }
        )
        self.memory_fusion = nn.ParameterDict(
            {
                branch: nn.Parameter(torch.atanh(torch.tensor(float(memory_fusion_init))))
                for branch in self._BRANCHES
            }
        )
        self.post_degradation_context = CompactPostDegradationContext(
            channels=post_context_channels,
            branch_channels=post_context_branch_channels,
            flow_eps=pole_decay_eps,
        )
        self.w1_local_guide = LocalWaveletGuide(
            post_context_channels=post_context_channels, guide_channels=w1_guide_channels
        )
        expert_input_channels = 4 * 9 + post_context_channels
        self.wavelet_experts = nn.ModuleDict(
            {
                "w3": DirectWaveletExpert(
                    input_channels=expert_input_channels,
                    hidden_channels=wavelet_hidden_channels,
                    spatial_context_channels=None,
                    token_channels=context_token_channels,
                    dilations=(1, 2),
                ),
                "w2": DirectWaveletExpert(
                    input_channels=expert_input_channels,
                    hidden_channels=wavelet_hidden_channels,
                    spatial_context_channels=None,
                    token_channels=context_token_channels,
                    dilations=(1, 1, 1, 1),
                ),
                "w1": DirectWaveletExpert(
                    input_channels=expert_input_channels,
                    hidden_channels=wavelet_hidden_channels,
                    spatial_context_channels=w1_guide_channels,
                    token_channels=context_token_channels,
                    dilations=(1, 1),
                ),
            }
        )
        # Wavelet-target EMA is model state, not wrapper-only state, so strict
        # generator checkpoint resume restores normalization exactly.
        self.register_buffer("wavelet_residual_energy_ema", torch.zeros(3, 3))
        self.register_buffer(
            "wavelet_residual_energy_initialized", torch.tensor(False, dtype=torch.bool)
        )

        self._last_pole_sequence = None
        self._last_frame_tokens = None
        self._last_clip_token = None
        self._last_stage1_preview = None
        self._context_flows_forward = None
        self._context_flows_backward = None
        self.last_aux_losses = {}
        self._stage2_pole_states = {}
        self._last_pole_context_frames = None
        with torch.random.fork_rng(devices=[]):
            self.time_pole_context = AdditivePoleConditioning(
                num_poles=num_poles,
                state_channels=pole_state_channels,
                context_channels=pole_context_channels,
            )
            self.direction_attention["stage_2"] = LocalBidirectionalAttention(
                channels=mid_channels,
                hidden_channels=direction_attention_channels,
                eps=direction_attention_eps,
            )
            for level in ("w3", "w2"):
                self.wavelet_experts[level].spatial_film = spatial_film(
                    pole_context_channels, wavelet_hidden_channels
                )

        with torch.random.fork_rng(devices=[]):
            self.local_guides = nn.ModuleDict(
                {
                    level: LocalWaveletGuide(post_context_channels, w1_guide_channels)
                    for level in ("w3", "w2")
                }
            )

            self.pole_fusions = nn.ModuleDict(
                {
                    level: LocalPoleFusion(w1_guide_channels, pole_context_channels)
                    for level in ("w3", "w2", "w1")
                }
            )

        with torch.random.fork_rng(devices=[]):
            self.preview_condition = PreviewConditioning(w1_guide_channels)

    @property
    def _offload_inference(self):
        return (
            getattr(self, "inference_cpu_offload", False)
            and not self.training
            and not torch.is_grad_enabled()
        )

    def clear_inference_cache(self):
        self._last_pole_sequence = None
        self._last_frame_tokens = None
        self._last_clip_token = None
        self._stage2_pole_states = {}
        self._stage2_direction_weights = None
        self._last_pole_context_frames = None
        self._last_stage1_preview = None
        self._context_flows_forward = None
        self._context_flows_backward = None
        for cell in self.history_cells.values():
            cell.last_conditioning_state = None

    def compute_flow(self, lqs):
        batch, frames, channels, height, width = lqs.shape
        first = lqs[:, :-1].reshape(-1, channels, height, width)
        second = lqs[:, 1:].reshape(-1, channels, height, width)
        flows_backward = self.spynet(first, second).reshape(batch, frames - 1, 2, height, width)
        flows_forward = self.spynet(second, first).reshape(batch, frames - 1, 2, height, width)
        return flows_forward, flows_backward

    def upsample(self, lqs, features):
        outputs = torch.empty_like(lqs) if self._offload_inference else []
        for frame in range(lqs.shape[1]):
            feature = torch.cat(
                [
                    features["spatial"][frame],
                    *(features[branch][frame] for branch in self._BRANCHES),
                ],
                dim=1,
            )
            feature = self.reconstruction(feature)
            feature = self.lrelu(self.upsample1(feature))
            feature = self.lrelu(self.upsample2(feature))
            feature = self.lrelu(self.conv_hr(feature))
            output = self.conv_last(feature) + lqs[:, frame]
            if self._offload_inference:
                outputs[:, frame].copy_(output)
                for values in features.values():
                    values.release(frame)
                del feature, output
            else:
                outputs.append(output)
        if self._offload_inference:
            return outputs
        return torch.stack(outputs, dim=1)

    def _fuse_bidirectional_stage(
        self,
        stage_key,
        reference_features,
        backward_features,
        forward_features,
        backward_trust,
        forward_trust,
    ):
        device = next(self.parameters()).device
        stage2 = stage_key == "stage_2"
        result = CpuFrameCache(device) if self._offload_inference else []
        for frame, reference in enumerate(reference_features):
            inputs = (
                reference,
                backward_features[frame],
                forward_features[frame],
                backward_trust[frame],
                forward_trust[frame],
            )
            if stage2:
                inputs = tuple(value.detach() for value in inputs)
            output = self.direction_attention[stage_key](*inputs)
            result.append(output["weights"] if stage2 else output["fused"])
        if stage2:
            self._stage2_direction_weights = result
            return None
        return result

    @staticmethod
    def _frame_poles(pole_sequence, frame):
        return {
            name: pole_sequence[name][:, frame]
            for name in ("lambdas", "omegas", "a_real", "a_imag", "phi_real", "phi_imag")
        }

    def _propagate_branch(
        self,
        branch,
        spatial_features,
        completed_branches,
        modal_alignment,
        flows_forward,
        flows_backward,
        pole_sequence,
        frame_tokens,
    ):
        direction = branch.split("_", 1)[0]
        if direction == "backward":
            flows = flows_backward
            reverse_flows = flows_forward
        else:
            flows = flows_forward
            reverse_flows = flows_backward
        batch, flow_frames, _, feature_height, feature_width = flows.shape
        frame_indices = list(range(flow_frames + 1))
        flow_indices = list(range(-1, flow_frames))
        if direction == "backward":
            frame_indices = frame_indices[::-1]
            flow_indices = frame_indices

        propagated = flows.new_zeros(batch, self.mid_channels, feature_height, feature_width)
        state = None
        branch_features = CpuFrameCache(flows.device) if self._offload_inference else []
        branch_trust = CpuFrameCache(flows.device) if self._offload_inference else []
        pole_states = CpuFrameCache(flows.device) if self._offload_inference else []
        branch_index = self._BRANCHES.index(branch)
        for index, frame in enumerate(frame_indices):
            current = spatial_features[frame]
            alignment_condition = modal_alignment[frame]
            previous = propagated.clone()
            if index > 0:
                pair_index = flow_indices[index]
                flow = flows[:, pair_index]
                reverse_flow = reverse_flows[:, pair_index]
                spatial_previous = spatial_features[frame_indices[index - 1]]
                grid_flow = flow.permute(0, 2, 3, 1)
                spatial_aligned = flow_warp(spatial_previous, grid_flow, padding_mode="zeros")
                recurrent_aligned = flow_warp(propagated, grid_flow, padding_mode="zeros")
                deform_aligned = self.deform_align[branch](
                    previous,
                    torch.cat([recurrent_aligned, current], dim=1),
                    flow,
                    alignment_condition,
                )
            else:
                flow = None
                reverse_flow = None
                spatial_aligned = current
                deform_aligned = propagated

            current_poles = self._frame_poles(pole_sequence, frame)
            memory, state, alignment_gate = self.history_cells[branch](
                current,
                deform_aligned,
                spatial_aligned,
                state,
                flow,
                reverse_flow,
                current_poles,
                frame_tokens[:, frame],
            )
            temporal_context = (
                alignment_gate * deform_aligned + torch.tanh(self.memory_fusion[branch]) * memory
            )
            if branch.endswith("_2"):
                pole_states.append(self.history_cells[branch].last_conditioning_state)
            backbone_inputs = [current]
            # Same order as the reference backbone: current, all earlier passes,
            # then this branch's aligned recurrent/pole context. No detach.
            backbone_inputs.extend(
                completed_branches[previous_branch][frame]
                for previous_branch in self._BRANCHES[:branch_index]
            )
            backbone_inputs.append(temporal_context)
            propagated = temporal_context + self.backbone[branch](torch.cat(backbone_inputs, dim=1))
            branch_features.append(propagated)
            branch_trust.append(alignment_gate)

        if direction == "backward":
            branch_features = branch_features[::-1]
            branch_trust = branch_trust[::-1]
            pole_states = pole_states[::-1]
        if branch.endswith("_2"):
            self._stage2_pole_states[branch] = pole_states
        if self._offload_inference:
            self.history_cells[branch].last_conditioning_state = None
        return branch_features, branch_trust

    def _build_stage1_preview(self, lqs, spatial_features, stage1_features):
        previews = torch.empty_like(lqs, device="cpu") if self._offload_inference else []
        for frame, (spatial_feature, stage1_feature) in enumerate(
            zip(spatial_features, stage1_features)
        ):
            preview = self.stage1_preview(lqs[:, frame], spatial_feature, stage1_feature)
            if self._offload_inference:
                previews[:, frame].copy_(preview)
            else:
                previews.append(preview)
        if self._offload_inference:
            return previews
        return torch.stack(previews, dim=1)

    def _combine_stage2_pole_states(self):
        combined = CpuFrameCache(next(self.parameters()).device) if self._offload_inference else []
        for backward, forward, weights in zip(
            self._stage2_pole_states["backward_2"],
            self._stage2_pole_states["forward_2"],
            self._stage2_direction_weights,
        ):
            wb, wf = weights[:, 1:2], weights[:, 2:3]
            combined.append(
                {
                    "real": wb.unsqueeze(1) * backward["real"] + wf.unsqueeze(1) * forward["real"],
                    "imag": wb.unsqueeze(1) * backward["imag"] + wf.unsqueeze(1) * forward["imag"],
                    "reliability": wb * backward["reliability"] + wf * forward["reliability"],
                }
            )
        self._last_pole_context_frames = combined
        self._stage2_pole_states = {}
        if self._offload_inference:
            self._stage2_direction_weights = None

    def prompt_free_backbone(self, lqs):
        batch, frames, channels, height, width = lqs.shape
        lqs_downsample = F.interpolate(
            lqs.reshape(-1, channels, height, width),
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
        ).reshape(batch, frames, channels, height // 4, width // 4)
        spatial = self.feat_extract(lqs.reshape(-1, channels, height, width)).reshape(
            batch, frames, self.mid_channels, height // 4, width // 4
        )
        features = {"spatial": [spatial[:, frame] for frame in range(frames)]}
        if self._offload_inference:
            features["spatial"] = CpuFrameCache(lqs.device, features["spatial"])
            del spatial

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        self._context_flows_forward = flows_forward
        self._context_flows_backward = flows_backward

        pre_context = self.pre_context(lqs_downsample)
        frame_tokens = pre_context["frame_tokens"]
        clip_token = pre_context["clip_token"]
        modal_alignment_tensor = self.modal_condition_projector(
            pre_context["spatial"], frame_tokens, clip_token
        )
        modal_alignment = [modal_alignment_tensor[:, frame] for frame in range(frames)]
        if self._offload_inference:
            modal_alignment = CpuFrameCache(lqs.device, modal_alignment)
            del modal_alignment_tensor, pre_context, lqs_downsample
        pole_sequence = self.shared_poles(frame_tokens, clip_token)
        self._last_pole_sequence = pole_sequence
        self._last_frame_tokens = frame_tokens
        self._last_clip_token = clip_token

        branch_features = {}
        branch_trust = {}
        for stage_index, stage_branches in enumerate(self._STAGE_BRANCHES, start=1):
            for branch in stage_branches:
                branch_features[branch], branch_trust[branch] = self._propagate_branch(
                    branch,
                    features["spatial"],
                    branch_features,
                    modal_alignment,
                    flows_forward,
                    flows_backward,
                    pole_sequence,
                    frame_tokens,
                )
            if stage_index == 1:
                backward_branch, forward_branch = stage_branches
                stage1_features = self._fuse_bidirectional_stage(
                    "stage_1",
                    features["spatial"],
                    branch_features[backward_branch],
                    branch_features[forward_branch],
                    branch_trust[backward_branch],
                    branch_trust[forward_branch],
                )
                self._last_stage1_preview = self._build_stage1_preview(
                    lqs, features["spatial"], stage1_features
                )

        self._fuse_bidirectional_stage(
            "stage_2",
            stage1_features,
            branch_features["backward_2"],
            branch_features["forward_2"],
            branch_trust["backward_2"],
            branch_trust["forward_2"],
        )
        self._combine_stage2_pole_states()
        self.last_aux_losses = dict(pole_sequence["aux_losses"])
        if self._offload_inference:
            del stage1_features, modal_alignment, branch_trust
        features.update(branch_features)
        restored = self.upsample(lqs, features)
        return restored

    def _tokens_for_chunk(self, lqs, frame_start):
        end = frame_start + lqs.shape[1]
        return (self._last_frame_tokens[:, frame_start:end], self._last_clip_token)

    def _flows_for_chunk(self, frame_start, frames):
        end = frame_start + frames - 1
        return (
            self._context_flows_forward[:, frame_start:end],
            self._context_flows_backward[:, frame_start:end],
        )

    @staticmethod
    def _expert_evidence(base_band, lq_band, context):
        difference = base_band - lq_band
        return torch.cat(
            [
                base_band,
                lq_band,
                difference,
                difference.abs(),
                context,
            ],
            dim=2,
        )

    def _wavelet_refine_chunk(self, lqs, base_restored, gt=None, frame_start=0):
        lq_pyramid = dwt3_bior44(lqs)
        base_live = dwt3_bior44(base_restored)
        preview = self._last_stage1_preview[:, frame_start : frame_start + lqs.shape[1]]
        # CPU storage preserves the full preview; only this halo interval is needed.
        preview = preview.to(device=lqs.device, dtype=lqs.dtype)
        preview_pyramid = dwt3_bior44(preview)
        base_anchor = {name: value.detach() for name, value in base_live.items()}
        spatial_sizes = {level: base_live[level].shape[-2:] for level in ("w3", "w2", "w1")}
        flows_forward, flows_backward = self._flows_for_chunk(frame_start, lqs.shape[1])
        degradation = self.post_degradation_context(
            lqs,
            base_restored.detach(),
            spatial_sizes,
            flows_forward=flows_forward,
            flows_backward=flows_backward,
        )
        frame_tokens, clip_token = self._tokens_for_chunk(lqs, frame_start)
        frame_tokens = frame_tokens.detach()
        clip_token = clip_token.detach()

        predictions = {}
        residuals = {}
        for level in ("w3", "w2", "w1"):
            evidence = self._expert_evidence(
                base_anchor[level], lq_pyramid[level], degradation["contexts"][level]
            )
            guide_encoder = self.w1_local_guide if level == "w1" else self.local_guides[level]
            guide = guide_encoder(
                base_anchor[level], lq_pyramid[level], degradation["contexts"][level]
            )
            preview_context = self.preview_condition(
                level, preview_pyramid[level], lq_pyramid[level]
            )
            guide = guide + preview_context
            pole_context = self.time_pole_context(
                level,
                self._last_pole_context_frames[frame_start : frame_start + lqs.shape[1]],
                guide,
                {
                    name: self._last_pole_sequence[name][
                        :, frame_start : frame_start + lqs.shape[1]
                    ].detach()
                    for name in ("lambdas", "omegas")
                },
            )
            spatial_context = self.pole_fusions[level](guide, pole_context)
            residual = self.wavelet_experts[level](
                evidence, frame_tokens, clip_token, spatial_context=spatial_context
            )
            residuals[level] = residual
            predictions[level] = base_live[level] + residual

        a2_pred = idwt_level_bior44(base_live["a3"], predictions["w3"])
        a1_pred = idwt_level_bior44(a2_pred, predictions["w2"])
        restored = idwt_level_bior44(a1_pred, predictions["w1"])
        outputs = {
            "restored": restored,
            "base_restored": base_restored,
            "a2_pred": a2_pred,
            "a1_pred": a1_pred,
            "aux_losses": dict(self.last_aux_losses),
        }
        for level in ("w3", "w2", "w1"):
            outputs[f"{level}_pred"] = predictions[level]
            outputs[f"{level}_base_anchor"] = base_anchor[level]
            outputs[f"{level}_residual"] = residuals[level]
        if gt is not None:
            target_pyramid = dwt3_bior44(gt)
            for level in ("w3", "w2", "w1"):
                outputs[f"{level}_target"] = target_pyramid[level]
        return outputs

    def _wavelet_refine(self, lqs, base_restored, gt=None):
        if gt is not None:
            return self._wavelet_refine_chunk(lqs, base_restored, gt=gt, frame_start=0)
        restored_chunks = torch.empty_like(lqs) if self._offload_inference else []
        for start in range(0, lqs.shape[1], self.wavelet_infer_chunk):
            end = min(start + self.wavelet_infer_chunk, lqs.shape[1])
            extended_start = max(0, start - 1)
            extended_end = min(lqs.shape[1], end + 1)
            chunk_lqs = lqs[:, extended_start:extended_end]
            chunk_base = base_restored[:, extended_start:extended_end]
            chunk = self._wavelet_refine_chunk(chunk_lqs, chunk_base, frame_start=extended_start)
            local_start = start - extended_start
            local_end = local_start + end - start
            if self._offload_inference:
                restored_chunks[:, start:end].copy_(chunk["restored"][:, local_start:local_end])
            else:
                restored_chunks.append(chunk["restored"][:, local_start:local_end])
            if self._offload_inference:
                del chunk
        return {
            "restored": (
                restored_chunks if self._offload_inference else torch.cat(restored_chunks, dim=1)
            ),
            "base_restored": base_restored,
        }

    def forward(self, lqs, gt=None):
        if lqs.ndim != 5 or lqs.shape[2] != 3 or lqs.shape[1] < 2:
            raise ValueError("Expected RGB video [B,T,3,H,W] with at least two frames.")
        if gt is not None and gt.shape != lqs.shape:
            raise ValueError("LQ and GT must have identical shapes.")
        padded_lqs, original_size = _pad_video_to_multiple(lqs, 8)
        if min(padded_lqs.shape[-2:]) <= 128:
            raise ValueError("SpyNet needs padded image dimensions above 128; use 256 or larger.")
        padded_gt = None
        if gt is not None:
            padded_gt, _ = _pad_video_to_multiple(gt, 8)
        base_restored = self.prompt_free_backbone(padded_lqs)
        wavelet_outputs = self._wavelet_refine(padded_lqs, base_restored, gt=padded_gt)
        restored = wavelet_outputs["restored"][..., : original_size[0], : original_size[1]]
        base_cropped = base_restored[..., : original_size[0], : original_size[1]]
        preview = self._last_stage1_preview[..., : original_size[0], : original_size[1]]
        outputs = dict(wavelet_outputs)
        outputs.update(
            {
                "restored": restored,
                "base_restored": base_cropped,
                "stage1_preview": preview,
                "student_flow_to_previous": self._context_flows_forward,
                "student_flow_to_next": self._context_flows_backward,
            }
        )
        if self._offload_inference:
            self.clear_inference_cache()
        return outputs
