"""CUDA checks for dense dependencies, gradient isolation and strict reload."""

import io
import os
from copy import deepcopy

import torch

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


LEVELS = ('w3', 'w2', 'w1')
WAVELET_PREFIXES = (
    'post_degradation_context.',
    'w1_local_guide.',
    'wavelet_experts.',
)


def norm(parameters):
    return sum(float(parameter.grad.detach().abs().sum())
               for parameter in parameters if parameter.grad is not None)


def wavelet_loss(outputs):
    losses = []
    for level in LEVELS:
        target = (
            outputs[f'{level}_target']
            - outputs[f'{level}_base_anchor'])
        losses.append(
            (outputs[f'{level}_residual'] - target).square().mean())
    return torch.stack(losses).sum()


def total_loss(outputs, gt):
    result = (outputs['restored'] - gt).square().mean()
    result = result + (outputs['base_restored'] - gt).square().mean()
    result = result + 0.05 * (
        outputs['stage1_preview'] - gt).square().mean()
    result = result + 0.1 * wavelet_loss(outputs)
    return result + sum(outputs['aux_losses'].values())


def check_geometry_support(network):
    cell = network.history_cells['forward_1']
    batch, channels, height, width = 1, network.mid_channels, 8, 8
    current = torch.randn(batch, channels, height, width, device='cuda')
    aligned = torch.randn_like(current)
    spatial = torch.randn_like(current)
    token = torch.randn(batch, 64, device='cuda')
    poles = {
        'lambdas': torch.ones(batch, network.num_poles, device='cuda'),
        'omegas': torch.zeros(batch, network.num_poles, device='cuda'),
        'a_real': torch.full(
            (batch, network.num_poles), 0.9, device='cuda'),
        'a_imag': torch.zeros(batch, network.num_poles, device='cuda'),
        'phi_real': torch.ones(batch, network.num_poles, device='cuda'),
        'phi_imag': torch.zeros(batch, network.num_poles, device='cuda'),
    }
    mixer_output = []
    handle = cell.mixer.register_forward_hook(
        lambda _module, _inputs, output: mixer_output.append(output))
    _, state, alignment_gate = cell(
        current, aligned, spatial, None, None, None, poles, token)
    geometry_support, _ = cell._reliability(
        current, current, None, None)
    read_gate = (
        geometry_support
        * torch.sigmoid(mixer_output[-1]['read_logits']))
    assert torch.count_nonzero(geometry_support).item() == 0
    assert torch.count_nonzero(read_gate).item() == 0
    assert torch.count_nonzero(alignment_gate).item() == 0

    flow = torch.zeros(batch, 2, height, width, device='cuda')
    mixer_output.clear()
    _, _, alignment_gate = cell(
        current, aligned, spatial, state, flow, flow, poles, token)
    geometry_support, _ = cell._reliability(
        current, spatial, flow, flow)
    read_gate = (
        geometry_support
        * torch.sigmoid(mixer_output[-1]['read_logits']))
    assert torch.all(read_gate <= geometry_support + 1e-7)
    assert torch.all(alignment_gate <= geometry_support + 1e-7)
    handle.remove()
    print('GEOMETRY_SUPPORT_BOUNDS_OK', flush=True)


def dependency_hooks(network, frames):
    """Check actual concatenated tensors, not only module dimensions."""
    channels = network.mid_channels
    saved = {branch: {} for branch in network._BRANCHES}
    calls = {branch: 0 for branch in network._BRANCHES}
    handles = []

    def branch_hook(branch, branch_index):
        def inspect(module, args, output):
            del module
            index = calls[branch]
            frame = frames - 1 - index if branch.startswith('backward') else index
            chunks = args[0].detach().split(channels, dim=1)
            assert len(chunks) == branch_index + 2
            for previous_index, previous in enumerate(
                    network._BRANCHES[:branch_index]):
                assert len(saved[previous]) == frames, (branch, previous)
                assert torch.equal(
                    chunks[previous_index + 1],
                    saved[previous][frame]), (branch, previous, frame)
            saved[branch][frame] = chunks[-1] + output.detach()
            calls[branch] += 1
        return inspect

    for index, branch in enumerate(network._BRANCHES):
        assert network.backbone[branch].main[0].in_channels == (
            index + 2) * channels
        handles.append(network.backbone[branch].register_forward_hook(
            branch_hook(branch, index)))
    assert network.reconstruction.main[0].in_channels == 5 * channels
    reconstructed = [0]

    def reconstruction_hook(module, args):
        del module
        chunks = args[0].detach().split(channels, dim=1)
        assert len(chunks) == 5
        for index, branch in enumerate(network._BRANCHES):
            assert torch.equal(
                chunks[index + 1],
                saved[branch][reconstructed[0]])
        reconstructed[0] += 1

    handles.append(network.reconstruction.register_forward_pre_hook(
        reconstruction_hook))

    def finish():
        assert all(count == frames for count in calls.values())
        assert reconstructed[0] == frames
        for handle in handles:
            handle.remove()
        saved.clear()
        print('OFFICIAL_DENSE_INPUT_VALUES_AND_WIDTHS_OK', flush=True)
    return finish


def check_finite_gradients(network):
    missing = [name for name, parameter in network.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in network.parameters()
               if parameter.grad is not None)


def check_wavelet_boundary(network):
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(WAVELET_PREFIXES):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run this production-shape check on a reserved CUDA node.')
    torch.manual_seed(23)
    config = os.environ['CONFIG']
    option = yaml_load(config)
    network = build_network(deepcopy(option['network_g'])).cuda().train()
    assert network.__class__.__name__ == (
        'AverNetModalContentPoleWaveletDWT3OfficialDense')
    assert set(network.direction_attention) == {'stage_1'}
    assert network.wavelet_experts['w3'].spatial_film is None
    assert network.wavelet_experts['w2'].spatial_film is None
    assert network.wavelet_experts['w1'].spatial_film is not None
    forbidden = (
        'pole_wavelet_router',
        'direction_attention.stage_2',
        'modal_refiner',
        'inter_stage_reanchor',
        'keep_gate',
        'history_strength',
        'propagation_logit',
        'parent',
        'gain',
        'rho_delta',
        'omega_delta',
        'complex_w2',
    )
    assert not any(token in key.lower()
                   for key in network.state_dict()
                   for token in forbidden)
    check_geometry_support(network)
    with torch.no_grad():
        odd_ref = torch.rand(1, 3, 120, 213, device='cuda')
        odd_flow = network.spynet(odd_ref, torch.rand_like(odd_ref))
    assert odd_flow.shape == (1, 2, 120, 213)
    print('SPYNET_ODD_PYRAMID_OK', flush=True)

    frames = int(os.environ.get('CHECK_FRAMES', '12'))
    batch = int(os.environ.get('CHECK_BATCH', '4'))
    lq = torch.rand(batch, frames, 3, 256, 256, device='cuda')
    gt = torch.rand_like(lq)
    optimizer = torch.optim.Adam(
        [parameter for parameter in network.parameters()
         if parameter.requires_grad],
        lr=1e-4)

    finish = dependency_hooks(network, frames)
    outputs = network(lq, gt=gt)
    finish()
    assert torch.allclose(
        outputs['restored'],
        outputs['base_restored'],
        atol=3e-5,
        rtol=3e-5)
    wavelet_loss(outputs).backward()
    check_wavelet_boundary(network)
    for expert in network.wavelet_experts.values():
        assert all(norm(head.parameters()) > 0
                   for head in expert.residual_heads)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del outputs

    outputs = network(lq, gt=gt)
    wavelet_loss(outputs).backward()
    check_wavelet_boundary(network)
    assert norm(network.post_degradation_context.parameters()) > 0
    assert norm(network.w1_local_guide.parameters()) > 0
    assert norm(network.wavelet_experts.parameters()) > 0
    print('WAVELET_ONLY_GRADIENT_BOUNDARY_OK', flush=True)
    network.zero_grad(set_to_none=True)
    del outputs

    outputs = network(lq, gt=gt)
    (outputs['base_restored'] - gt).square().mean().backward()
    assert norm(network.backbone.parameters()) > 0
    assert all(parameter.grad is None
               for name, parameter in network.named_parameters()
               if name.startswith(WAVELET_PREFIXES))
    print('BASE_ONLY_GRADIENT_BOUNDARY_OK', flush=True)
    network.zero_grad(set_to_none=True)
    del outputs

    outputs = network(lq, gt=gt)
    (outputs['restored'] - gt).square().mean().backward()
    assert norm(network.backbone.parameters()) > 0
    assert norm(network.post_degradation_context.parameters()) > 0
    assert norm(network.w1_local_guide.parameters()) > 0
    assert norm(network.wavelet_experts.parameters()) > 0
    print('FINAL_JOINT_GRADIENT_BOUNDARY_OK', flush=True)
    network.zero_grad(set_to_none=True)
    del outputs

    outputs = network(lq, gt=gt)
    total_loss(outputs, gt).backward()
    check_finite_gradients(network)
    network.wavelet_residual_energy_ema.copy_(
        torch.arange(9, device='cuda').reshape(3, 3))
    network.wavelet_residual_energy_initialized.fill_(True)
    checkpoint = io.BytesIO()
    torch.save(network.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = build_network(deepcopy(option['network_g'])).eval()
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location='cpu', weights_only=True),
        strict=True)
    assert torch.equal(
        reloaded.wavelet_residual_energy_ema,
        torch.arange(9).reshape(3, 3))
    print(
        'NOROUTER_SINGLE_GPU_PEAK_GIB',
        f'allocated={torch.cuda.max_memory_allocated()/2**30:.3f}',
        f'reserved={torch.cuda.max_memory_reserved()/2**30:.3f}',
        flush=True)
    print('NOROUTER_PRODUCTION_NETWORK_OK', flush=True)


if __name__ == '__main__':
    main()
