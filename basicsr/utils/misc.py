"""Filesystem, time and seed helpers used by experiment 6220553."""

import os
import random
import time

import numpy as np
import torch

from .dist_util import master_only


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


@master_only
def make_exp_dirs(opt):
    os.makedirs(opt['path']['experiments_root'])
    os.makedirs(opt['path']['models'])
    os.makedirs(opt['path']['training_states'])
