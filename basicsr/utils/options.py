"""YAML and CLI parsing for the single 6220553 training configuration."""

import argparse
import os
import shutil
import sys
import time

import yaml

from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from basicsr.utils.misc import set_random_seed


def yaml_load(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def dict2str(value, indent=0):
    lines = []
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append('  ' * indent + f'{key}:')
            lines.append(dict2str(item, indent + 1))
        else:
            lines.append('  ' * indent + f'{key}: {item}')
    return '\n'.join(lines)


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True)
    parser.add_argument('--force_yml', nargs='+', required=True)
    args = parser.parse_args()
    opt = yaml_load(args.opt)
    init_dist(opt['dist_params']['backend'])
    opt['rank'], opt['world_size'] = get_dist_info()
    set_random_seed(opt['manual_seed'] + opt['rank'])

    for entry in args.force_yml:
        path, raw_value = entry.split('=', 1)
        target = opt
        keys = path.split(':')
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = yaml.safe_load(raw_value)

    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
    experiment_root = os.path.join(
        opt['path']['experiments_root'], opt['name'])
    opt['path'].update({
        'experiments_root': experiment_root,
        'models': os.path.join(experiment_root, 'models'),
        'training_states': os.path.join(experiment_root, 'training_states'),
        'log': experiment_root,
    })
    return opt, args


@master_only
def copy_opt_file(source, experiment_root):
    destination = os.path.join(experiment_root, os.path.basename(source))
    shutil.copyfile(source, destination)
    with open(destination, 'r+', encoding='utf-8') as stream:
        content = stream.read()
        stream.seek(0)
        stream.write(
            f'# GENERATED: {time.asctime()}\n# COMMAND: {" ".join(sys.argv)}\n\n'
            + content)
