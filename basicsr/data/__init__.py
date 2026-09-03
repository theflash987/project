"""The two datasets and loaders used by the NoRouter-K16 experiment."""

import random
from functools import partial

import numpy as np
import torch

from basicsr.utils.dist_util import get_dist_info

from .tud_video_train_dataset import TUDVideoDataset
from .video_test_dataset import VideoRecurrentTestDataset


def build_dataset(dataset_opt):
    options = dict(dataset_opt)
    dataset_class = (
        TUDVideoDataset
        if options['phase'] == 'train'
        else VideoRecurrentTestDataset)
    return dataset_class(options)


def build_dataloader(dataset, dataset_opt, sampler=None, seed=0):
    if dataset_opt['phase'] == 'train':
        rank, _ = get_dist_info()
        num_workers = dataset_opt['num_worker_per_gpu']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=dataset_opt['batch_size_per_gpu'],
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            worker_init_fn=partial(
                _worker_init,
                num_workers=num_workers,
                rank=rank,
                seed=seed))
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=0)


def _worker_init(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
