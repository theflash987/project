"""PyTorch distributed helpers used by NoRouter-K16."""

import functools
import os

import torch
import torch.distributed as dist


def init_dist(backend='nccl'):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend)


def get_dist_info():
    return dist.get_rank(), dist.get_world_size()


def master_only(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return function(*args, **kwargs)
        return None
    return wrapper
