"""Deterministic checks for the isolated direct DWT-3 branch."""

import io

import torch

from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import (
    DirectWaveletExpert,
    W1LocalGuide,
)
from basicsr.archs.official_dense_components import (
    dwt3_bior44,
    idwt_level_bior44,
)


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

    guide = W1LocalGuide().to(device)
    base = torch.randn(1, 2, 9, 128, 128, device=device)
    lq = torch.randn_like(base)
    post = torch.randn(1, 2, 16, 128, 128, device=device)
    local = guide(base, lq, post)
    assert local.shape == (1, 2, 3, 32, 128, 128)
    swapped = guide(
        swap_orientations(base),
        swap_orientations(lq),
        post)
    assert torch.allclose(swapped[:, :, 0], local[:, :, 1])
    assert torch.allclose(swapped[:, :, 1], local[:, :, 0])
    assert torch.allclose(swapped[:, :, 2], local[:, :, 2])

    evidence = torch.randn(1, 2, 52, 32, 32, device=device)
    frame_token = torch.randn(1, 2, 64, device=device)
    clip_token = torch.randn(1, 64, device=device)
    for dilations in ((1, 2), (1, 1, 1, 1)):
        expert = DirectWaveletExpert(dilations=dilations).to(device)
        residual = expert(evidence, frame_token, clip_token)
        assert torch.count_nonzero(residual).item() == 0

    w1_expert = DirectWaveletExpert(
        spatial_context_channels=32,
        dilations=(1, 1)).to(device)
    try:
        w1_expert(evidence, frame_token, clip_token)
    except ValueError as error:
        assert str(error) == 'W1 spatial context is required.'
    else:
        raise AssertionError('W1 accepted a missing spatial context.')
    residual = w1_expert(
        evidence,
        frame_token,
        clip_token,
        spatial_context=torch.randn(
            1, 2, 3, 32, 32, 32, device=device))
    assert torch.count_nonzero(residual).item() == 0
    target = torch.randn_like(residual)
    (residual - target).square().mean().backward()
    for orientation, head in enumerate(w1_expert.residual_heads):
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0, orientation

    checkpoint = io.BytesIO()
    torch.save(guide.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = W1LocalGuide().to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True)
    print('NOROUTER_DWT3_COMPONENTS_OK')


if __name__ == '__main__':
    main()
