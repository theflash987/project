"""Paired frame-index training and complete-sequence evaluation datasets."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(path):
    """Read 8-bit RGB/BGR frames without rescaling their geometry."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an 8-bit, three-channel image: {path}")
    return image.astype(np.float32) / 255.0


def to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def read_sequence(paths):
    if not paths:
        raise ValueError("A sequence must contain frames.")
    images = [read_image(path) for path in paths]
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("All frames in a sequence must have the same dimensions.")
    return torch.stack([to_tensor(image) for image in images])


def frame_paths(folder):
    paths = sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(f"No image frames found in {folder}")
    return paths


def read_metadata(path):
    """Each row is: folder frame_count shape first_frame."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{number}: expected four metadata fields")
        folder, count, _, first = fields
        if Path(folder).name != folder or folder in {".", ".."}:
            raise ValueError(f"{path}:{number}: expected a sequence folder name")
        count, first = int(count), int(first)
        if count < 2 or first < 0:
            raise ValueError(f"{path}:{number}: invalid frame count or first index")
        rows.append((folder, count, first))
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError("Metadata must contain unique, nonempty sequences.")
    return rows


class PairedVideoDataset(Dataset):
    """One sample per source frame, preserving the original tail-resampling policy."""

    def __init__(self, options):
        self.lq_root = Path(options["lq"])
        self.gt_root = Path(options["gt"])
        self.frames = int(options["frames"])
        self.crop_size = int(options["crop_size"])
        self.filename_format = options.get("filename_format", "05d")
        self.extension = options.get("extension", "jpg")
        if self.frames < 2 or self.crop_size <= 128 or self.crop_size % 8:
            raise ValueError(
                "Training needs at least two frames and a crop above 128 divisible by 8."
            )
        self.samples = []
        for folder, count, first in read_metadata(options["metadata"]):
            if count < self.frames:
                raise ValueError(
                    f"{folder} has {count} frames, fewer than the requested {self.frames}."
                )
            self.samples.extend(
                (folder, frame, first, count) for frame in range(first, first + count)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        folder, start, first, count = self.samples[index]
        last_start = first + count - self.frames
        if start > last_start:
            start = random.randint(first, last_start)
        names = [
            f"{frame:{self.filename_format}}.{self.extension}"
            for frame in range(start, start + self.frames)
        ]
        lq = [read_image(self.lq_root / folder / name) for name in names]
        gt = [read_image(self.gt_root / folder / name) for name in names]
        height, width = lq[0].shape[:2]
        if any(image.shape != lq[0].shape for image in lq + gt):
            raise ValueError(f"LQ/GT frame shapes differ in {folder}.")
        if min(height, width) < self.crop_size:
            raise ValueError(f"{folder} is smaller than crop_size={self.crop_size}.")
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        region = (slice(top, top + self.crop_size), slice(left, left + self.crop_size))
        images = [image[region[0], region[1], ...] for image in lq + gt]
        horizontal = random.random() < 0.5
        vertical = random.random() < 0.5
        transpose = random.random() < 0.5
        tensors = []
        for image in images:
            if horizontal:
                cv2.flip(image, 1, image)
            if vertical:
                cv2.flip(image, 0, image)
            if transpose:
                image = image.transpose(1, 0, 2)
            tensors.append(to_tensor(image))
        return {
            "lq": torch.stack(tensors[: self.frames]),
            "gt": torch.stack(tensors[self.frames :]),
            "key": f"{folder}/{self.samples[index][1]:{self.filename_format}}",
        }


class PairedSequenceDataset(Dataset):
    """Load each complete video at its original resolution."""

    def __init__(self, options):
        self.lq_root = Path(options["lq"])
        self.gt_root = Path(options["gt"])
        self.folders = sorted(path.name for path in self.lq_root.iterdir() if path.is_dir())
        gt_folders = {path.name for path in self.gt_root.iterdir() if path.is_dir()}
        if not self.folders or set(self.folders) != gt_folders:
            raise ValueError("LQ and GT must contain the same nonempty sequence folders.")
        expected = None
        if options.get("metadata"):
            expected = {
                folder: (count, first)
                for folder, count, first in read_metadata(options["metadata"])
            }
            if set(expected) != set(self.folders):
                raise ValueError("Sequence folders differ from metadata.")
        self.paths = {}
        for folder in self.folders:
            lq = frame_paths(self.lq_root / folder)
            gt = frame_paths(self.gt_root / folder)
            if len(lq) < 2 or [path.name for path in lq] != [path.name for path in gt]:
                raise ValueError(f"Unpaired or too-short sequence: {folder}")
            if expected:
                count, first = expected[folder]
                indices = [int(path.stem) for path in lq]
                if indices != list(range(first, first + count)):
                    raise ValueError(f"Frame indices differ from metadata in {folder}.")
            self.paths[folder] = (lq, gt)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        lq, gt = self.paths[folder]
        lq, gt = read_sequence(lq), read_sequence(gt)
        if lq.shape != gt.shape:
            raise ValueError(f"LQ/GT frame shapes differ in {folder}.")
        return {"lq": lq, "gt": gt, "folder": folder}
