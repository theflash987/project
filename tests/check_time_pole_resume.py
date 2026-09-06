"""Exercise real 20k restoration, optimizer continuation and timed checkpoint save."""

import os
from pathlib import Path

import torch

from basicsr import train_resume as training
from check_time_pole_resume_checkpoint import check_checkpoint

restore = training.restore_training
output_root = None
source_state_path = None


def checked_restore(model, resume):
    global output_root, source_state_path
    state = restore(model, resume)
    weights = torch.load(resume['model'], map_location='cpu', weights_only=True)
    for network, key in ((model.get_bare_model(model.net_g), 'params'),
                         (model.net_g_ema, 'params_ema')):
        actual = network.state_dict()
        assert actual.keys() == weights[key].keys()
        assert all(torch.equal(actual[name].cpu(), value)
                   for name, value in weights[key].items())
    for optimizer, saved in zip(model.optimizers, state['optimizers']):
        actual = optimizer.state_dict()
        assert actual['param_groups'] == saved['param_groups']
        assert actual['state'].keys() == saved['state'].keys()
        for index, values in saved['state'].items():
            for name, value in values.items():
                assert torch.equal(actual['state'][index][name].cpu(), value.cpu())
    assert model.schedulers[0].state_dict() == state['schedulers'][0]
    assert state['iter'] == 20000
    assert model.schedulers[0].last_epoch == 19999
    output_root = Path(model.opt['path']['experiments_root'])
    source_state_path = resume['state']
    print('EXACT_20K_MODEL_EMA_BUFFERS_ADAM_COSINE_RESTORED', flush=True)
    return state


training.restore_training = checked_restore
training.train_pipeline(os.path.abspath(os.path.join(__file__, '..', '..')))

if int(os.environ['RANK']) == 0:
    check_checkpoint(source_state_path, output_root)
