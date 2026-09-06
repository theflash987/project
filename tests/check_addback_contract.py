"""Verify the 6308179 boundary, matched initialization and live fusion paths."""

import importlib.util
import os
from copy import deepcopy

import torch


VARIANT = os.environ['ABLATION_VARIANT']
CONTEXT = 'time_pole_context' if VARIANT == 'time_pole_fusion' else 'uniform_pole_context'


def check_structure(network):
    assert set(network.direction_attention) == {'stage_1', 'stage_2'}
    assert not hasattr(network, 'inter_stage_reanchor')
    assert not hasattr(network, 'modal_refiner')
    assert set(network.local_guides) == {'w3', 'w2'}
    assert set(network.pole_fusions) == {'w3', 'w2', 'w1'}
    assert all(expert.spatial_film is not None for expert in network.wavelet_experts.values())
    assert network.num_poles == 16 and network.pole_state_channels == 2
    context = getattr(network, CONTEXT)
    assert hasattr(context, 'time_encoding') == (VARIANT == 'time_pole_fusion')
    print('FUSION_STRUCTURE_OK', VARIANT, flush=True)


def check_initialization(network, option):
    source = os.path.join(
        os.environ['BASELINE_CODE'], 'basicsr/archs/modal_content_pole_wavelet_dwt3_arch.py')
    spec = importlib.util.spec_from_file_location('baseline_initialization', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual_rng = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        baseline = module.AverNetModalContentPoleWaveletDWT3OfficialDense(
            **deepcopy(option['network_g']))
        assert torch.equal(actual_rng, torch.get_rng_state())
        restored = network.state_dict()
        for name, value in baseline.state_dict().items():
            mapped = name.replace('uniform_pole_context.', CONTEXT + '.')
            assert torch.equal(value.cpu(), restored[mapped].cpu()), name
    with torch.random.fork_rng(devices=[]):
        peer_source = (
            '/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_LocalPoleFusion/code/'
            'basicsr/archs/modal_content_pole_wavelet_dwt3_arch.py')
        peer_spec = importlib.util.spec_from_file_location('fusion_a_initialization', peer_source)
        peer_module = importlib.util.module_from_spec(peer_spec)
        peer_spec.loader.exec_module(peer_module)
        torch.manual_seed(23)
        peer = peer_module.AverNetModalContentPoleWaveletDWT3OfficialDense(
            **deepcopy(option['network_g']))
        for name, value in peer.state_dict().items():
            mapped = name.replace('uniform_pole_context.', CONTEXT + '.')
            assert torch.equal(value.cpu(), restored[mapped].cpu()), name
    print('A_B_SHARED_INITIALIZATION_IDENTICAL', flush=True)
    print('PARENT_6308179_INITIALIZATION_AND_RNG_IDENTICAL', flush=True)


def check_trained_path(network):
    network.eval()
    lq = torch.rand(1, 6, 3, 256, 256, device='cuda')
    with torch.no_grad():
        intact = network(lq)
    deltas = {}
    for level in ('w3', 'w2', 'w1'):
        for source_index, source_name in ((0, 'guide'), (1, 'pole')):
            def disable_source(module, args):
                inputs = list(args)
                inputs[source_index] = torch.zeros_like(inputs[source_index])
                return tuple(inputs)
            handle = network.pole_fusions[level].register_forward_pre_hook(disable_source)
            with torch.no_grad():
                disabled = network(lq)
            handle.remove()
            assert torch.equal(intact['base_restored'], disabled['base_restored'])
            delta = float((intact['restored'] - disabled['restored']).abs().max())
            assert delta > 1e-8, (level, source_name, delta)
            deltas[level + '_' + source_name] = delta
            del disabled
    print('ALL_SCALES_GUIDE_AND_POLE_PATHS_ACTIVE', deltas, flush=True)
    return deltas
