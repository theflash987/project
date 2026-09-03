"""Cached whole-sequence DAVIS validation dataset for NoRouter-K16."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



def _read_sequence(paths):
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(image.transpose(2, 0, 1)).float())
    return torch.stack(frames)


class VideoRecurrentTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        lq_root = Path(opt['dataroot_lq'])
        gt_root = Path(opt['dataroot_gt'])
        self.folders = sorted(path.name for path in lq_root.iterdir() if path.is_dir())
        self.imgs_lq = {}
        self.imgs_gt = {}
        self.data_info = {'folder': []}
        for folder in self.folders:
            lq_paths = sorted((lq_root / folder).iterdir())
            gt_paths = sorted((gt_root / folder).iterdir())
            self.imgs_lq[folder] = _read_sequence(lq_paths)
            self.imgs_gt[folder] = _read_sequence(gt_paths)
            self.data_info['folder'].extend([folder] * len(lq_paths))

    def __getitem__(self, index):
        folder = self.folders[index]
        return {
            'lq': self.imgs_lq[folder],
            'gt': self.imgs_gt[folder],
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
