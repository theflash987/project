"""Deterministic distributed frame-index sampling."""

import math

import torch
from torch.utils.data import Sampler


class EnlargedSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, ratio=1):
        if not len(dataset) or ratio < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid dataset size, enlargement ratio, or distributed rank.")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) * ratio / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=generator).tolist()
        indices = [index % len(self.dataset) for index in indices]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
