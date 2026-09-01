"""CUDA checks for dense dependencies, direction-only gradients and reload."""

import io
import os
from copy import deepcopy

import torch

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


def norm(parameters):
    return sum(float(p.grad.detach().abs().sum())
               for p in parameters if p.grad is not None)


def check_direction_boundary(network):
    """Router gradients reach the scorer, not its evidence or pole states."""
    channels = network.mid_channels
    features = [torch.randn(1, channels, 8, 8, device='cuda', requires_grad=True)
                for _ in range(3)]
    trusts = [torch.rand(1, 1, 8, 8, device='cuda', requires_grad=True)
              for _ in range(2)]
    network._fuse_bidirectional_stage(
        'stage_2', *([value] for value in features + trusts))
    raw_states = []
    network._stage2_router_branches = {}
    for branch in ('backward_2', 'forward_2'):
        state = {
            'real': torch.randn(1, 16, 2, 8, 8, device='cuda', requires_grad=True),
            'imag': torch.randn(1, 16, 2, 8, 8, device='cuda', requires_grad=True),
            'confidence': torch.rand(1, 1, 8, 8, device='cuda', requires_grad=True),
            'history_strength': torch.rand(1, 1, 8, 8, device='cuda', requires_grad=True),
            'read_gate': torch.rand(1, 16, 8, 8, device='cuda', requires_grad=True),
        }
        raw_states.extend(state.values())
        network._stage2_router_branches[branch] = [
            network._detach_router_state(state)]
    network._combine_stage2_router_states()
    context, _, _ = network.pole_wavelet_router(
        'w2',
        torch.randn(1, 1, 9, 8, 8, device='cuda'),
        torch.randn(1, 1, 9, 8, 8, device='cuda'),
        torch.randn(1, 1, 16, 8, 8, device='cuda'),
        network._last_pole_router_frames)
    context.square().mean().backward()
    assert norm(network.direction_attention['stage_2'].parameters()) > 0
    assert all(value.grad is None for value in features + trusts + raw_states)
    network.zero_grad(set_to_none=True)
    network._last_pole_router_frames = None
    network._last_direction_weights = {}
    print('DENSE_DIRECTION_GRADIENT_BOUNDARY_OK', flush=True)


def dependency_hooks(network, frames):
    """Check the actual concatenated tensors, not just module dimensions."""
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
            for previous_index, previous in enumerate(network._BRANCHES[:branch_index]):
                assert len(saved[previous]) == frames, (branch, previous)
                assert torch.equal(chunks[previous_index + 1], saved[previous][frame]), (branch, previous, frame)
            saved[branch][frame] = chunks[-1] + output.detach()
            calls[branch] += 1
        return inspect

    for index, branch in enumerate(network._BRANCHES):
        assert network.backbone[branch].main[0].in_channels == (index + 2) * channels
        handles.append(network.backbone[branch].register_forward_hook(branch_hook(branch, index)))
    assert network.reconstruction.main[0].in_channels == 5 * channels
    reconstructed = [0]

    def reconstruction_hook(module, args):
        del module
        chunks = args[0].detach().split(channels, dim=1)
        assert len(chunks) == 5
        for index, branch in enumerate(network._BRANCHES):
            assert torch.equal(chunks[index + 1], saved[branch][reconstructed[0]])
        reconstructed[0] += 1

    handles.append(network.reconstruction.register_forward_pre_hook(reconstruction_hook))

    def finish():
        assert all(count == frames for count in calls.values())
        assert reconstructed[0] == frames
        for handle in handles:
            handle.remove()
        saved.clear()
        print('OFFICIAL_DENSE_INPUT_VALUES_AND_WIDTHS_OK', flush=True)
    return finish


def loss(outputs, gt):
    result = (outputs['restored'] - gt).square().mean()
    result = result + 0.05 * (outputs['stage1_preview'] - gt).square().mean()
    for level in ('w3', 'w2', 'w1'):
        target = outputs[f'{level}_target'] - outputs[f'{level}_base'].detach()
        result = result + 0.1 * (outputs[f'{level}_residual'] - target).square().mean()
    return result + sum(outputs['aux_losses'].values())


def check_gradients(network):
    missing = [name for name, p in network.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all()
               for p in network.parameters() if p.grad is not None)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run this production-shape check on a reserved CUDA node.')
    torch.manual_seed(23)
    config = os.environ['CONFIG']
    option = yaml_load(config)
    network = build_network(deepcopy(option['network_g'])).cuda().train()
    assert network.__class__.__name__ == 'AverNetModalContentPoleWaveletDWT3OfficialDense'
    check_direction_boundary(network)
    frames, batch = int(os.environ.get('CHECK_FRAMES', '12')), int(os.environ.get('CHECK_BATCH', '4'))
    lq = torch.rand(batch, frames, 3, 256, 256, device='cuda')
    gt = torch.rand_like(lq)
    optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=1e-4)
    finish = dependency_hooks(network, frames)
    output = network(lq, gt=gt)
    finish()
    assert torch.allclose(output['restored'], output['base_restored'], atol=3e-5, rtol=3e-5)
    loss(output, gt).backward()
    check_gradients(network)
    assert norm(network.pole_wavelet_router.parameters()) == 0
    assert norm(network.direction_attention['stage_2'].parameters()) == 0
    for expert in network.wavelet_experts.values():
        assert all(norm(head.parameters()) > 0 for head in expert.residual_heads)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del output
    output = network(lq, gt=gt)
    loss(output, gt).backward()
    check_gradients(network)
    router = norm(network.pole_wavelet_router.parameters())
    direction = norm(network.direction_attention['stage_2'].parameters())
    assert router > 0 and direction > 0, (router, direction)
    print(f'DENSE_SECOND_STEP_GRADIENTS router={router} direction={direction}', flush=True)
    forbidden = ('parent', 'gain', 'rho_delta', 'omega_delta', 'complex_w2')
    assert not any(token in key.lower() for key in network.state_dict() for token in forbidden)
    network.wavelet_residual_energy_ema.copy_(torch.arange(9, device='cuda').reshape(3, 3))
    network.wavelet_residual_energy_initialized.fill_(True)
    checkpoint = io.BytesIO()
    torch.save(network.state_dict(), checkpoint)
    checkpoint.seek(0)
    network.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
    assert torch.equal(network.wavelet_residual_energy_ema, torch.arange(9, device='cuda').reshape(3, 3))
    print(f'DENSE_SINGLE_GPU_PEAK_GIB allocated={torch.cuda.max_memory_allocated()/2**30:.3f} reserved={torch.cuda.max_memory_reserved()/2**30:.3f}', flush=True)
    print('OFFICIAL_DENSE_PRODUCTION_NETWORK_OK', flush=True)


if __name__ == '__main__':
    main()
