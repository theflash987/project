"""Verify the on-disk state after the 20k-to-20.1k resume smoke."""

import math
import sys
from pathlib import Path

import torch


def check_checkpoint(source_state_path, output_root):
    output_root = Path(output_root)
    state = torch.load(output_root / 'training_states/20100.state',
                       map_location='cpu', weights_only=False)
    weights = torch.load(output_root / 'models/net_g_20100.pth',
                         map_location='cpu', weights_only=True)
    assert state['iter'] == 20100
    assert state['schedulers'][0]['last_epoch'] == 20099
    source_state = torch.load(source_state_path, map_location='cpu', weights_only=False)
    old_optimizer = source_state['optimizers'][0]
    optimizer = state['optimizers'][0]
    for index, old in old_optimizer['state'].items():
        current = optimizer['state'][index]
        assert current['step'].item() == old['step'].item() + 100, (
            index, old['step'].item(), current['step'].item())
        assert torch.isfinite(current['exp_avg']).all()
        assert torch.isfinite(current['exp_avg_sq']).all()
    for group, base_lr in zip(optimizer['param_groups'], (1e-4, 1.5e-5)):
        expected = 1e-7 + 0.5 * (base_lr - 1e-7) * (
            1 + math.cos(math.pi * 20099 / 150000))
        assert math.isclose(group['lr'], expected, rel_tol=1e-12)
    for key in ('params', 'params_ema'):
        assert all(torch.isfinite(value).all() for value in weights[key].values())
    logs = list(output_root.glob('train_*.log'))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert 'WALLTIME_SAVE_STOP iter=20100' in text
    assert 'TIME_POLE_RESUME_COMPLETE iter=20100' in text
    assert 'Validation DAVIS_T6_Test_As_Val @ 20100' in text
    print('TRUE_RESUME_100_STEPS_ADAM_COSINE_TIMED_SAVE_DAVIS_OK', flush=True)


if __name__ == '__main__':
    check_checkpoint(sys.argv[1], sys.argv[2])
