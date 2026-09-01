"""Deterministic component checks for Pole-routed direct DWT-3."""

import io

import torch

from basicsr.archs.official_dense_components import (
    dwt3_bior44,
    idwt_level_bior44,
)
from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import (
    DirectWaveletExpert,
    PoleWaveletRouter,
)


def make_router_frames(batch, frames, height, width, device):
    values = []
    for _ in range(frames):
        values.append({
            'real': torch.randn(batch, 16, 2, height, width, device=device),
            'imag': torch.randn(batch, 16, 2, height, width, device=device),
            'reliability': torch.rand(
                batch, 16, height, width, device=device).clamp_min(0.05),
        })
    return values


def swap_orientations(value, first=0, second=1):
    bands = list(torch.chunk(value, 3, dim=2))
    bands[first], bands[second] = bands[second], bands[first]
    return torch.cat(bands, dim=2)


def main():
    torch.manual_seed(11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    video = torch.randn(1, 2, 3, 256, 256, device=device)
    pyramid = dwt3_bior44(video)
    a2 = idwt_level_bior44(pyramid['a3'], pyramid['w3'])
    a1 = idwt_level_bior44(a2, pyramid['w2'])
    reconstructed = idwt_level_bior44(a1, pyramid['w1'])
    assert torch.allclose(reconstructed, video, atol=2e-5, rtol=2e-5)

    router = PoleWaveletRouter().to(device)
    router_frames = make_router_frames(1, 2, 64, 64, device)
    context_shapes = {'w3': 32, 'w2': 64, 'w1': 128}
    cached = {}
    for level, size in context_shapes.items():
        base = torch.randn(1, 2, 9, size, size, device=device)
        lq = torch.randn_like(base)
        post = torch.randn(1, 2, 16, size, size, device=device)
        context, stats, alpha = router(
            level, base, lq, post, router_frames)
        assert context.shape == (1, 2, 3, 32, size, size)
        assert torch.allclose(
            alpha.sum(dim=2),
            torch.ones_like(alpha[:, :, 0]),
            atol=1e-6,
            rtol=1e-6)
        assert len(stats) == 12
        cached[level] = (base, lq, post, context, alpha)

    base, lq, post, context, alpha = cached['w2']
    swapped_context, _, swapped_alpha = router(
        'w2',
        swap_orientations(base),
        swap_orientations(lq),
        post,
        router_frames)
    assert torch.allclose(
        swapped_alpha[:, 0], alpha[:, 1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        swapped_alpha[:, 1], alpha[:, 0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        swapped_context[:, :, 0], context[:, :, 1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        swapped_context[:, :, 1], context[:, :, 0], atol=1e-5, rtol=1e-5)

    expert = DirectWaveletExpert(dilations=(1, 2)).to(device)
    evidence = torch.randn(1, 2, 52, 32, 32, device=device)
    pole_context = torch.randn(1, 2, 3, 32, 32, 32, device=device)
    frame_token = torch.randn(1, 2, 64, device=device)
    clip_token = torch.randn(1, 64, device=device)
    residual = expert(evidence, pole_context, frame_token, clip_token)
    assert torch.count_nonzero(residual).item() == 0
    target = torch.randn_like(residual)
    (residual - target).square().mean().backward()
    for orientation, head in enumerate(expert.residual_heads):
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0, orientation

    checkpoint = io.BytesIO()
    torch.save(router.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = PoleWaveletRouter().to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    print('OFFICIAL_DENSE_COMPONENTS_OK')


if __name__ == '__main__':
    main()
