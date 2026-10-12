"""Original-dtype host storage for frame-local inference tensors."""

import torch


def move_tensors(value, device):
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    return value.detach().to(device=device)


class CpuFrameCache:
    """Indexing restores only the requested frame; slices remain lazy."""

    def __init__(self, device, values=()):
        if torch.is_grad_enabled():
            raise RuntimeError("CPU frame storage is inference-only")
        self.device = device
        self.frames = []
        for value in values:
            self.append(value)

    def append(self, value):
        self.frames.append(move_tensors(value, "cpu"))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = CpuFrameCache(self.device)
            result.frames = self.frames[index]
            return result
        return move_tensors(self.frames[index], self.device)

    def release(self, index):
        self.frames[index] = None
