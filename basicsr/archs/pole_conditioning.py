"""Additive pole mixing, preview conditions, and wavelet guides."""

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.wavelet_components import ResidualUnit


class BlurPoolDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = (kernel[:, None] * kernel[None, :]) / 256.0
        self.register_buffer("blur_kernel", kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1))
        self.channels = channels
        self.projection = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, value):
        value = F.conv2d(
            value, self.blur_kernel.to(dtype=value.dtype), padding=2, groups=self.channels
        )
        return self.projection(value)


def spatial_film(context_channels, hidden_channels):
    films = nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(context_channels, hidden_channels, 3, 1, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 2 * hidden_channels, 1),
            )
            for _ in range(3)
        ]
    )
    for film in films:
        nn.init.normal_(film[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(film[-1].bias, 0.0)
    return films


class LocalPoleFusion(nn.Module):
    """Fuse local evidence and pole context independently per orientation."""

    def __init__(self, guide_channels=32, context_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(guide_channels + context_channels, context_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            ResidualUnit(context_channels),
        )

    def forward(self, guide, context):
        batch, frames, orientations, _, height, width = guide.shape
        fused = self.net(
            torch.cat([guide, context], dim=3).reshape(
                batch * frames * orientations, -1, height, width
            )
        )
        return fused.reshape(batch, frames, orientations, -1, height, width)


class PreviewConditioning(nn.Module):
    """Encode live preview subbands into the existing per-orientation guides.

    The preview is a condition only: it never replaces the base coefficient
    anchor or becomes a reconstruction target. Nonzero small output weights
    allow the preview head to learn once the original zero-initialized wavelet
    output heads have received their first update.
    """

    def __init__(self, channels=32):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.Conv2d(6, channels, 1), nn.SiLU(), nn.Conv2d(channels, channels, 1)
                )
                for level in ("w3", "w2", "w1")
            }
        )
        for encoder in self.encoders.values():
            nn.init.normal_(encoder[-1].weight, std=1e-3)
            nn.init.zeros_(encoder[-1].bias)

    def forward(self, level, preview_band, lq_band):
        batch, frames, channels, height, width = preview_band.shape
        assert channels == 9 and preview_band.shape == lq_band.shape
        preview = preview_band.reshape(batch, frames, 3, 3, height, width)
        difference = (preview_band - lq_band).reshape(batch, frames, 3, 3, height, width)
        evidence = torch.cat([preview, difference], dim=3)
        context = self.encoders[level](evidence.reshape(batch * frames * 3, 6, height, width))
        return context.reshape(batch, frames, 3, -1, height, width)


class AdditivePoleConditioning(nn.Module):
    """Frame/branch-conditioned pole representation mixing, without QK attention.

    This changes readout representations only; the original recurrent poles and
    their decay/frequency are untouched. No temporal pooling crosses chunks.
    """

    def __init__(self, state_channels=2, context_channels=32, num_poles=16):
        super().__init__()
        self.context_channels = context_channels
        self.state_projection = nn.Conv2d(2 * state_channels, context_channels, 1)
        self.w3_adapter = BlurPoolDownsample(context_channels)
        self.num_poles = num_poles
        self.time_encoding = nn.Sequential(
            nn.Linear(2, context_channels), nn.SiLU(), nn.Linear(context_channels, context_channels)
        )
        self.branch_encoding = nn.Embedding(9, context_channels)
        self.guide_projection = nn.Conv2d(context_channels, context_channels, 1)
        self.mixing_mlp = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.SiLU(),
            nn.Linear(context_channels, num_poles * num_poles),
        )
        self.pole_projection = nn.Conv2d(num_poles * context_channels, context_channels, 1)
        self.conditioning_mlp = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, 1),
            nn.SiLU(),
            nn.Conv2d(context_channels, context_channels, 1),
        )

    def forward(self, level, frames, guide, poles):
        real = torch.stack([frame["real"] for frame in frames], dim=1)
        imag = torch.stack([frame["imag"] for frame in frames], dim=1)
        reliability = torch.stack([frame["reliability"] for frame in frames], dim=1)
        batch, length, count, channels, height, width = real.shape
        assert count == self.num_poles
        bt, c = batch * length, self.context_channels
        values = self.state_projection(
            torch.cat([real, imag], dim=3).reshape(bt * count, 2 * channels, height, width)
        ).reshape(bt, count, c, height, width)
        nominal = torch.stack([-poles["lambdas"].log(), poles["omegas"] / torch.pi], dim=-1)
        values = values + self.time_encoding(nominal).reshape(bt, count, c, 1, 1)
        values = values * reliability.reshape(bt, count, 1, height, width)
        local = self.guide_projection(guide.reshape(bt * 3, c, *guide.shape[-2:]))
        local = F.interpolate(local, size=(height, width), mode="bilinear", align_corners=False)
        local = local.reshape(bt, 3, c, height, width)
        branch_ids = torch.arange(3, device=real.device) + 3 * ("w3", "w2", "w1").index(level)
        descriptor = (
            local.mean(dim=(-2, -1))
            + values.mean(dim=(1, 3, 4)).unsqueeze(1)
            + self.branch_encoding(branch_ids).unsqueeze(0)
        )
        weights = self.mixing_mlp(descriptor).reshape(bt, 3, count, count).softmax(dim=-1)
        mixed = torch.einsum("bojk,bkchw->bojchw", weights, values)
        mixed = self.pole_projection(mixed.reshape(bt * 3, count * c, height, width))
        context = self.conditioning_mlp(mixed + local.reshape(bt * 3, c, height, width))
        if level == "w3":
            context = self.w3_adapter(context)
        elif level == "w1":
            context = F.interpolate(
                context, size=guide.shape[-2:], mode="bilinear", align_corners=False
            )
        return context.reshape(batch, length, 3, c, *context.shape[-2:])
