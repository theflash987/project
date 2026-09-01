"""DAVIS 12-frame training dataset used by experiment 6220553."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.data.transforms import augment, paired_random_crop


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return image[..., :3]


def _to_rgb_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


class TUDVideoDataset(Dataset):
    def __init__(self, opt):
        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq'])
        self.num_frame = int(opt['num_frame'])
        self.patch_size = int(opt['gt_size'])
        self.filename_format = opt['filename_tmpl']
        self.filename_ext = opt['filename_ext']
        self.samples = []
        with open(opt['meta_info_file'], encoding='utf-8') as stream:
            for line in stream:
                folder, frame_count, _, first_frame = line.split()
                frame_count = int(frame_count)
                first_frame = int(first_frame)
                for frame in range(first_frame, first_frame + frame_count):
                    self.samples.append((folder, frame, first_frame, frame_count))

    def __getitem__(self, index):
        folder, start, first, frame_count = self.samples[index]
        last_start = first + frame_count - self.num_frame
        if start > last_start:
            start = random.randint(first, last_start)
        indices = range(start, start + self.num_frame)
        lq_frames = [
            _read_image(
                self.lq_root / folder /
                f'{frame:{self.filename_format}}.{self.filename_ext}')
            for frame in indices
        ]
        gt_frames = [
            _read_image(
                self.gt_root / folder /
                f'{frame:{self.filename_format}}.{self.filename_ext}')
            for frame in range(start, start + self.num_frame)
        ]
        gt_frames, lq_frames = paired_random_crop(
            gt_frames, lq_frames, self.patch_size)
        frames = augment(lq_frames + gt_frames)
        lq = torch.stack([_to_rgb_tensor(frame) for frame in frames[:self.num_frame]])
        gt = torch.stack([_to_rgb_tensor(frame) for frame in frames[self.num_frame:]])
        key = f'{folder}/{self.samples[index][1]:{self.filename_format}}'
        return {'lq': lq, 'gt': gt, 'key': key}

    def __len__(self):
        return len(self.samples)
