"""Optional torchrun support and synchronized numerical checks."""

import os

import torch
from torch import distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def rank_and_size():
    return (dist.get_rank(), dist.get_world_size()) if is_distributed() else (0, 1)


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("The model's MMCV deformable convolution requires a CUDA device.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank)


def require_finite(values, *, iteration, stage):
    flags = torch.stack([torch.isfinite(value.detach()).all() for value in values.values()])
    failed = (~flags.all()).to(dtype=torch.int32)
    if is_distributed():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        bad = [name for name, ok in zip(values, flags.cpu().tolist()) if not ok]
        rank, _ = rank_and_size()
        raise FloatingPointError(
            f"Nonfinite training values at iteration {iteration}, {stage}, rank {rank}: {bad}"
        )
