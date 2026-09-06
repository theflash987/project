"""Small CUDA checks for fusion, aggregation and time-aware pole selection."""

import os

import torch
from torch.nn import functional as F

from basicsr.archs import addback_components as components


torch.manual_seed(23)
variant = os.environ['ABLATION_VARIANT']
is_time = variant == 'time_pole_fusion'
context_type = components.TimePoleConditioning if is_time else components.UniformPoleConditioning
module = context_type().cuda()
real = torch.randn(1, 16, 2, 8, 8, device='cuda')
imag = torch.randn_like(real)
reliability = torch.rand(1, 16, 8, 8, device='cuda')
frames = [{'real': real, 'imag': imag, 'reliability': reliability}]
poles = {
    'lambdas': torch.logspace(-1.7, 0, 16, device='cuda').reshape(1, 1, 16),
    'omegas': torch.linspace(0.03, 0.97, 16, device='cuda').reshape(1, 1, 16) * torch.pi,
}
permutation = torch.randperm(16, device='cuda')
shuffled = [{key: value[:, permutation] for key, value in frames[0].items()}]
permuted_poles = {key: value[:, :, permutation] for key, value in poles.items()}

for level, size in (('w3', 4), ('w2', 8), ('w1', 16)):
    guide = torch.randn(1, 1, 3, 32, size, size, device='cuda', requires_grad=True)
    if is_time:
        output = module(level, frames, guide, poles)
        permuted = module(level, shuffled, guide, permuted_poles)
    else:
        output = module(level, frames)
        permuted = module(level, shuffled)
    assert output.shape == guide.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, permuted, rtol=1e-4, atol=1e-5)

    if not is_time:
        values = module.state_projection(torch.cat([real, imag], dim=2).reshape(
            16, 4, 8, 8)).reshape(1, 16, 32, 8, 8)
        expected = (values * reliability.unsqueeze(2)).mean(dim=1)
        if level == 'w3':
            expected = module.w3_adapter(expected)
        elif level == 'w1':
            expected = F.interpolate(
                expected, scale_factor=2, mode='bilinear', align_corners=False)
        for orientation in range(3):
            torch.testing.assert_close(output[:, 0, orientation], expected)
    else:
        handle = module.time_encoding.register_forward_hook(
            lambda _module, _args, result: torch.zeros_like(result))
        without_time = module(level, frames, guide, poles)
        handle.remove()
        time_delta = float((output - without_time).abs().max())
        assert time_delta > 1e-6, (level, time_delta)
        changed_guide = module(level, frames, -guide, poles)
        assert float((output - changed_guide).abs().max()) > 1e-6
        print('TIME_AND_LOCAL_QUERY_AFFECT_CONTEXT', level, time_delta, flush=True)

    fusion = components.LocalPoleFusion().cuda()
    result = fusion(guide, output)
    assert result.shape == guide.shape
    result.square().mean().backward()
    assert guide.grad is not None and guide.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in fusion.parameters())
    module.zero_grad(set_to_none=True)

if is_time:
    guide = torch.randn(1, 1, 3, 32, 8, 8, device='cuda')
    handles = [query.register_forward_hook(
        lambda _module, _args, result: torch.zeros_like(result))
        for query in module.query.values()]
    actual = module('w2', frames, guide, poles)
    values = module.state_projection(torch.cat([real, imag], dim=2).reshape(
        16, 4, 8, 8)).reshape(1, 16, 32, 8, 8)
    expected = (values * reliability.unsqueeze(2)).mean(dim=1)
    for orientation in range(3):
        torch.testing.assert_close(actual[:, 0, orientation], expected)
    for handle in handles:
        handle.remove()
    module.zero_grad(set_to_none=True)
    module('w2', frames, guide, poles).square().mean().backward()
    for name in ('query', 'key', 'time_encoding'):
        gradients = [p.grad for p in getattr(module, name).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    print('TIME_ROUTER_ZERO_QUERY_EQUALS_UNIFORM_AND_GRADIENTS_OK', flush=True)

print('FUSION_COMPONENTS_ALL_SCALES_OK', variant, flush=True)
