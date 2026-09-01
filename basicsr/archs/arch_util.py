"""Convolution and flow primitives used by experiment 6220553."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def _initialize(module, scale=1.0):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            layer.weight.mul_(scale)
            if layer.bias is not None:
                layer.bias.zero_()


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        _initialize(self, 0.1)

    def forward(self, value):
        return value + self.conv2(self.relu(self.conv1(value)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Sequential(*(
                ResidualBlockNoBN(out_channels)
                for _ in range(num_blocks))),
        )

    def forward(self, value):
        return self.main(value)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2)
        _initialize(self)

    def forward(self, value):
        return F.pixel_shuffle(
            self.upsample_conv(value), self.scale_factor)


def flow_warp(
        value,
        flow,
        interp_mode='bilinear',
        padding_mode='zeros',
        align_corners=True):
    height, width = value.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=value.dtype),
        torch.arange(width, device=value.device, dtype=value.dtype),
        indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped_grid = grid + flow
    warped_grid = torch.stack((
        2.0 * warped_grid[..., 0] / max(width - 1, 1) - 1.0,
        2.0 * warped_grid[..., 1] / max(height - 1, 1) - 1.0,
    ), dim=-1)
    return F.grid_sample(
        value,
        warped_grid,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)
