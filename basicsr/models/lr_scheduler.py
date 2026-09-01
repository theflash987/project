"""Cosine schedule used by experiment 6220553."""

import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingRestartLR(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights, eta_min):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        self.cumulative_periods = [sum(periods[:index + 1]) for index in range(len(periods))]
        super().__init__(optimizer)

    def get_lr(self):
        cycle = next(
            index for index, end in enumerate(self.cumulative_periods)
            if self.last_epoch <= end)
        start = 0 if cycle == 0 else self.cumulative_periods[cycle - 1]
        phase = (self.last_epoch - start) / self.periods[cycle]
        weight = self.restart_weights[cycle]
        return [
            self.eta_min
            + weight * 0.5 * (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * phase))
            for base_lr in self.base_lrs
        ]
