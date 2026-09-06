"""Uniform pole values with the original W3 scale adapter; no learned router."""

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.official_dense_components import ResidualUnit


class BlurPoolDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1., 4., 6., 4., 1.])
        kernel = (kernel[:, None] * kernel[None, :]) / 256.0
        self.register_buffer(
            'blur_kernel', kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1))
        self.channels = channels
        self.projection = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, value):
        value = F.conv2d(
            value, self.blur_kernel.to(dtype=value.dtype),
            padding=2, groups=self.channels)
        return self.projection(value)


class UniformPoleConditioning(nn.Module):
    def __init__(self, state_channels=2, context_channels=32):
        super().__init__()
        self.context_channels = context_channels
        self.state_projection = nn.Conv2d(
            2 * state_channels, context_channels, 1)
        self.w3_adapter = BlurPoolDownsample(context_channels)

    def forward(self, level, frames):
        real = torch.stack([frame['real'] for frame in frames], dim=1)
        imag = torch.stack([frame['imag'] for frame in frames], dim=1)
        reliability = torch.stack(
            [frame['reliability'] for frame in frames], dim=1)
        batch, length, poles, channels, height, width = real.shape
        values = self.state_projection(torch.cat([real, imag], dim=3).reshape(
            batch * length * poles, 2 * channels, height, width)).reshape(
                batch * length, poles, self.context_channels, height, width)
        reliability = reliability.reshape(
            batch * length, poles, 1, height, width)
        context = (reliability * values).mean(dim=1)
        if level == 'w3':
            context = self.w3_adapter(context)
        elif level == 'w1':
            context = F.interpolate(
                context, scale_factor=2, mode='bilinear', align_corners=False)
        context = context.reshape(
            batch, length, 1, self.context_channels, *context.shape[-2:])
        return context.expand(-1, -1, 3, -1, -1, -1)


def spatial_film(context_channels, hidden_channels):
    films = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 * hidden_channels, 1))
        for _ in range(3)
    ])
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
            ResidualUnit(context_channels))

    def forward(self, guide, context):
        batch, frames, orientations, _, height, width = guide.shape
        fused = self.net(torch.cat([guide, context], dim=3).reshape(
            batch * frames * orientations, -1, height, width))
        return fused.reshape(batch, frames, orientations, -1, height, width)


class TimePoleConditioning(UniformPoleConditioning):
    """Local queries select pole values using content and nominal time scale."""

    def __init__(self, state_channels=2, context_channels=32):
        super().__init__(state_channels, context_channels)
        with torch.random.fork_rng(devices=[]):
            self.query = nn.ModuleDict({
                level: nn.Conv2d(context_channels, context_channels, 1)
                for level in ('w3', 'w2', 'w1')
            })
            self.key = nn.Conv2d(context_channels, context_channels, 1, bias=False)
            self.time_encoding = nn.Sequential(
                nn.Linear(2, context_channels),
                nn.SiLU(),
                nn.Linear(context_channels, context_channels, bias=False))

    def forward(self, level, frames, guide, poles):
        real = torch.stack([frame['real'] for frame in frames], dim=1)
        imag = torch.stack([frame['imag'] for frame in frames], dim=1)
        reliability = torch.stack([frame['reliability'] for frame in frames], dim=1)
        batch, length, count, channels, height, width = real.shape
        values = self.state_projection(torch.cat([real, imag], dim=3).reshape(
            batch * length * count, 2 * channels, height, width))
        keys = self.key(values).reshape(
            batch * length, count, self.context_channels, height, width)
        time = torch.stack([
            -poles['lambdas'].log(), poles['omegas'] / torch.pi,
        ], dim=-1)
        keys = keys + self.time_encoding(time).reshape(
            batch * length, count, self.context_channels, 1, 1)
        query = self.query[level](guide.reshape(
            batch * length * 3, self.context_channels, *guide.shape[-2:]))
        query = F.interpolate(
            query, size=(height, width), mode='bilinear', align_corners=False)
        query = query.reshape(
            batch * length, 3, self.context_channels, height, width)
        logits = torch.einsum('bochw,bkchw->bokhw', query, keys)
        weights = torch.softmax(logits / self.context_channels**0.5, dim=2)
        values = values.reshape(
            batch * length, count, self.context_channels, height, width)
        reliability = reliability.reshape(batch * length, count, height, width)
        context = torch.einsum(
            'bokhw,bkhw,bkchw->bochw', weights, reliability, values).reshape(
                batch * length * 3, self.context_channels, height, width)
        if level == 'w3':
            context = self.w3_adapter(context)
        elif level == 'w1':
            context = F.interpolate(
                context, size=guide.shape[-2:], mode='bilinear', align_corners=False)
        return context.reshape(
            batch, length, 3, self.context_channels, *context.shape[-2:])
