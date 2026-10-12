"""The original cosine schedule, including its first-iteration convention."""

import math

from torch.optim.lr_scheduler import LRScheduler


class CosineSchedule(LRScheduler):
    def __init__(self, optimizer, total_iters, eta_min):
        if total_iters < 1 or eta_min < 0:
            raise ValueError("Invalid cosine schedule.")
        self.total_iters = int(total_iters)
        self.eta_min = float(eta_min)
        super().__init__(optimizer)

    def get_lr(self):
        phase = self.last_epoch / self.total_iters
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * phase))
            for base_lr in self.base_lrs
        ]
