"""Spatial crop and augmentation used by DAVIS training."""

import random

import cv2


def paired_random_crop(gt_frames, lq_frames, patch_size):
    height, width = lq_frames[0].shape[:2]
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    region = (slice(top, top + patch_size), slice(left, left + patch_size))
    return (
        [frame[region[0], region[1], ...] for frame in gt_frames],
        [frame[region[0], region[1], ...] for frame in lq_frames],
    )


def augment(frames):
    horizontal = random.random() < 0.5
    vertical = random.random() < 0.5
    transpose = random.random() < 0.5
    output = []
    for frame in frames:
        if horizontal:
            cv2.flip(frame, 1, frame)
        if vertical:
            cv2.flip(frame, 0, frame)
        if transpose:
            frame = frame.transpose(1, 0, 2)
        output.append(frame)
    return output
