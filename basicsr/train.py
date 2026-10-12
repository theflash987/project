"""Single-GPU or torchrun training with explicit complete-state resume."""

import argparse
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.paired_video_dataset import PairedSequenceDataset, PairedVideoDataset
from basicsr.utils.checkpoint import capture_rng, restore_rng
from basicsr.utils.dist_util import is_distributed, rank_and_size, setup
from basicsr.utils.evaluation import evaluate, write_report


def seed_worker(worker_id, workers, rank, seed):
    worker_seed = workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, resume=None):
    from basicsr.models.modal_content_pole_wavelet_dwt3_video_model import Trainer

    device = setup()
    rank, world_size = rank_and_size()
    seed_all(config["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    output = Path(config["output"])
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output}; choose a new directory or --resume."
        )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    if is_distributed():
        torch.distributed.barrier()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / f"train_rank{rank}.log")],
    )
    options = config["datasets"]["train"]
    train_set = PairedVideoDataset(options)
    sampler = EnlargedSampler(train_set, world_size, rank, options["enlarge_ratio"])
    workers = int(options["workers"])
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=options["batch_size"],
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        worker_init_fn=partial(seed_worker, workers=workers, rank=rank, seed=config["seed"]),
    )
    if not len(loader):
        raise ValueError("The training dataset is smaller than one complete batch per rank.")
    validation_set = PairedSequenceDataset(config["datasets"]["val"])
    model = Trainer(config, device)
    step, start_epoch, skip_batches, resume_rng = 0, 0, 0, None
    if resume is not None:
        step, start_epoch, skip_batches, resume_rng = model.resume(resume)
        if skip_batches > len(loader):
            raise ValueError("Checkpoint cursor exceeds the current loader length.")
    total = int(config["train"]["total_iter"])
    if step >= total:
        raise ValueError("The checkpoint has already reached total_iter.")
    epoch = start_epoch
    while step < total:
        sampler.set_epoch(epoch)
        if resume_rng is not None:
            epoch_rng = resume_rng["epoch_start"]
            restore_rng(epoch_rng)
        else:
            epoch_rng = capture_rng()
        iterator = iter(loader)
        for _ in range(skip_batches):
            next(iterator)
        if resume_rng is not None:
            restore_rng(resume_rng["current"])
            resume_rng = None
        for batch_in_epoch, data in enumerate(iterator, start=skip_batches + 1):
            step += 1
            model.feed_data(data)
            losses = model.step(step)
            if rank == 0 and (step == 1 or step % config["logging"]["interval"] == 0):
                logging.info(
                    "iter=%d lr=%s losses=%s",
                    step,
                    [group["lr"] for group in model.optimizer_g.param_groups],
                    losses,
                )
            if step % config["logging"]["save_interval"] == 0 or step == total:
                model.save(
                    output / f"iter_{step}.pth",
                    step=step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    epoch_rng=epoch_rng,
                )
            if step % config["validation"]["interval"] == 0 or step == total:
                report = evaluate(
                    model.net_g_ema, validation_set, device, config["validation"]["cpu_offload"]
                )
                if rank == 0:
                    write_report(
                        {"iteration": step, **report}, output / "validation" / f"{step}.json"
                    )
                    logging.info("Validation @ %d: %s", step, report["final"]["sequence_macro"])
            if step == total:
                break
        epoch += 1
        skip_batches = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="A complete checkpoint created by this trainer")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for section, key in (
        ("logging", "interval"),
        ("logging", "save_interval"),
        ("validation", "interval"),
    ):
        if config[section][key] < 1:
            raise ValueError(f"{section}.{key} must be positive.")
    if not config["model"].get("spynet_pretrained"):
        raise ValueError("Training requires a pretrained SpyNet checkpoint.")
    try:
        train(config, args.resume)
    finally:
        if is_distributed():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
