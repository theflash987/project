"""Strict generator loading and portable RNG records."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from basicsr.utils.dist_util import is_distributed, rank_and_size


def capture_rng():
    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, torch.from_numpy(keys.astype(np.int64)), position, gaussian, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def gather_rng(epoch_rng):
    record = {"current": capture_rng(), "epoch_start": epoch_rng}
    _, world_size = rank_and_size()
    records = [record]
    if is_distributed():
        records = [None] * world_size
        torch.distributed.all_gather_object(records, record)
    return records


def load_generator(model, checkpoint, key="params_ema"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state[key], strict=True)


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
