"""Verify the completed DDP smoke, strict weights/state reload, and live path."""

import json
import math
import re
import sys
from pathlib import Path

import torch

from basicsr.archs import build_network
from basicsr.models.lr_scheduler import CosineAnnealingRestartLR
from basicsr.utils.options import yaml_load
from check_addback_contract import VARIANT, check_trained_path


config_path, experiment_name, output_path = sys.argv[1:]
config = yaml_load(config_path)
assert config['train']['lambda_base_rec'] == 0.0
assert config['train']['lambda_base_ssim'] == 0.0
experiment = Path(config['path']['experiments_root']) / experiment_name
logs = sorted(experiment.glob('train_*.log'))
assert len(logs) == 1, logs
log = logs[0].read_text()


def values(name):
    return [float(value) for value in re.findall(
        rf'{re.escape(name)}:\s*([+\-0-9.eE]+)', log)]


for name in (
        'l_total', 'l_final_rec', 'l_final_ssim', 'l_base_rec', 'l_base_ssim',
        'l_wave_res', 'l_preview', 'l_flow_teacher', 'grad_norm',
        'base_to_final_residual_abs_mean', 'w1_local_guide_abs_mean'):
    data = values(name)
    assert len(data) >= 30 and all(math.isfinite(value) for value in data), name
for name in ('wavelet_branch_grad_norm', 'stage1_direction_grad_norm', 'addback_grad_norm'):
    data = values(name)
    assert len(data) == 30 and all(math.isfinite(value) for value in data)
    assert all(value > 0 for value in data[1:]), (name, data)
for name in ["local_guides_grad_norm","pole_fusions_grad_norm","time_pole_context_time_encoding_grad_norm","time_pole_context_query_grad_norm","time_pole_context_key_grad_norm"]:
    data = values(name)
    assert len(data) == 30 and all(math.isfinite(value) for value in data), name
    assert all(value > 0 for value in data[1:]), (name, data)
flow = values('spynet_grad_norm')
assert len(flow) == 30
assert all(value == 0 for value in flow[:10])
assert all(value > 0 and math.isfinite(value) for value in flow[10:])
for total, rec, ssim, wave, preview, flow_loss, flow_weight, pole in zip(
        values('l_total'), values('l_final_rec'), values('l_final_ssim'),
        values('l_wave_res'), values('l_preview'), values('l_flow_teacher'),
        values('lambda_flow_current'), values('l_aux_pole_separation'), strict=True):
    expected = rec + 0.05 * ssim + 0.1 * wave + 0.05 * preview + flow_weight * flow_loss + 0.001 * pole
    assert math.isclose(total, expected, rel_tol=1e-4, abs_tol=1e-6), (total, expected)
print('TOTAL_LOSS_EXCLUDES_BASE_OK', flush=True)
assert 'Validation DAVIS_T6_Test_As_Val @ 30' in log
for name in ('final_psnr', 'final_ssim', 'base_psnr', 'base_ssim'):
    assert values(name) and all(math.isfinite(value) for value in values(name)), name

checkpoint = torch.load(
    experiment / 'models/net_g_30.pth', map_location='cpu', weights_only=True)
network = build_network(config['network_g']).cuda().eval()
network.load_state_dict(checkpoint['params'], strict=True)
assert network.wavelet_residual_energy_initialized.item()
for key in ('wavelet_residual_energy_ema', 'wavelet_residual_energy_initialized'):
    assert torch.equal(checkpoint['params'][key], checkpoint['params_ema'][key])
assert all(torch.isfinite(value).all() for value in checkpoint['params'].values())
path_delta = check_trained_path(network)
network.load_state_dict(checkpoint['params_ema'], strict=True)
with torch.no_grad():
    sample = torch.rand(1, 20, 3, 256, 256, device='cuda')
    outputs = network(sample)
assert outputs['restored'].shape == sample.shape
assert torch.isfinite(outputs['restored']).all()
assert torch.isfinite(outputs['base_restored']).all()
del outputs, sample
network.zero_grad(set_to_none=True)

state = torch.load(
    experiment / 'training_states/30.state', map_location='cpu', weights_only=False)
assert state['iter'] == 30
normal, flow_parameters = [], []
for name, parameter in network.named_parameters():
    if not parameter.requires_grad:
        continue
    target = flow_parameters if any(
        f'spynet.basic_module.{level}.' in name for level in (4, 5)) else normal
    target.append(parameter)
lr = config['train']['optim_g']['lr']
optimizer = torch.optim.Adam(
    [{'params': normal, 'lr': lr},
     {'params': flow_parameters, 'lr': lr * config['train']['flow_lr_mul']}],
    **config['train']['optim_g'])
scheduler = CosineAnnealingRestartLR(optimizer, **config['train']['scheduler'])
optimizer.load_state_dict(state['optimizers'][0])
scheduler.load_state_dict(state['schedulers'][0])
assert len(optimizer.state) == len(normal) + len(flow_parameters)
assert scheduler.last_epoch == 29
for group, saved in zip(optimizer.param_groups, state['optimizers'][0]['param_groups']):
    assert group['lr'] == saved['lr']
for parameter in normal + flow_parameters:
    parameter.grad = torch.zeros_like(parameter)
optimizer.step()
scheduler.step()
assert scheduler.last_epoch == 30
assert all(torch.isfinite(parameter).all() for parameter in network.parameters())

summary = {
    'parent_job': '6308179',
    'base_loss_weights': [0.0, 0.0],
    'detach_unchanged': True,
    'variant': VARIANT, 'iterations': 30,
    'ddp': '4 ranks; batch 4/rank; 12 frames; 256 crop',
    'native_davis_validation': True,
    'strict_params_and_ema_reload': True,
    'optimizer_scheduler_reload_and_step': True,
    'chunked_20_frame_inference': True,
    'restored_path_output_max_delta': path_delta,
    'addback_grad_norm_last': values('addback_grad_norm')[-1],
    'spynet_grad_norm_last': flow[-1],
    'loss_last': values('l_total')[-1],
    'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
    'experiment_path': str(experiment),
}
Path(output_path, 'smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('ADDBACK_CHECKPOINT_AND_PATH_OK', json.dumps(summary), flush=True)
