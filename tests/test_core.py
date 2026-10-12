"""CPU checks for numerical primitives and public data/checkpoint contracts."""

import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs.pole_conditioning import AdditivePoleConditioning
from basicsr.archs.wavelet_components import dwt3_bior44, idwt_level_bior44
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.paired_video_dataset import PairedSequenceDataset, PairedVideoDataset, read_image
from basicsr.models.lr_scheduler import CosineSchedule
from basicsr.utils.checkpoint import atomic_save, capture_rng, restore_rng
from basicsr.utils.evaluation import quantize_prediction


class NumericalTests(unittest.TestCase):
    def test_wavelet_roundtrip_and_gradient(self):
        video = torch.rand(1, 2, 3, 24, 32, requires_grad=True)
        bands = dwt3_bior44(video)
        restored = idwt_level_bior44(bands["a3"], bands["w3"])
        restored = idwt_level_bior44(restored, bands["w2"])
        restored = idwt_level_bior44(restored, bands["w1"])
        torch.testing.assert_close(restored, video, atol=2e-6, rtol=0)
        restored.square().sum().backward()
        torch.testing.assert_close(video.grad, 2 * video.detach(), atol=5e-6, rtol=0)

    def test_additive_mixing_is_conditioned_and_differentiable(self):
        torch.manual_seed(10)
        mixing = AdditivePoleConditioning(state_channels=2, context_channels=4, num_poles=3)
        frames = [
            {
                "real": torch.randn(1, 3, 2, 4, 4),
                "imag": torch.randn(1, 3, 2, 4, 4),
                "reliability": torch.ones(1, 3, 4, 4),
            }
            for _ in range(2)
        ]
        guide = torch.randn(1, 2, 3, 4, 4, 4, requires_grad=True)
        poles = {"lambdas": torch.full((1, 2, 3), 0.8), "omegas": torch.rand(1, 2, 3)}
        output = mixing("w2", frames, guide, poles)
        changed = mixing("w2", frames, guide + 1, poles)
        self.assertEqual(tuple(output.shape), (1, 2, 3, 4, 4, 4))
        self.assertGreater((changed - output).abs().max().item(), 0)
        output.square().mean().backward()
        for parameter in mixing.mixing_mlp.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(guide.grad.abs().sum().item(), 0)

    def test_cosine_initial_iteration_and_resume(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = CosineSchedule(optimizer, 300000, 1e-7)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        saved_optimizer = optimizer.state_dict()
        saved_scheduler = scheduler.state_dict()
        optimizer2 = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-4)
        scheduler2 = CosineSchedule(optimizer2, 300000, 1e-7)
        optimizer2.load_state_dict(saved_optimizer)
        scheduler2.load_state_dict(saved_scheduler)
        optimizer.step()
        scheduler.step()
        optimizer2.step()
        scheduler2.step()
        self.assertEqual(scheduler.last_epoch, scheduler2.last_epoch)
        self.assertEqual(scheduler.get_last_lr(), scheduler2.get_last_lr())

    def test_rng_checkpoint_can_load_with_weights_only(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pth"
            atomic_save(state, path)
            restored = torch.load(path, weights_only=True)
        restore_rng(restored)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(3), atol=0, rtol=0)

    def test_quantization_preserves_input_and_rgb_rounding(self):
        image = torch.tensor([-1, 0, 0.5, 1, 2], dtype=torch.float32).reshape(1, 1, 1, 5)
        original = image.clone()
        prediction = quantize_prediction(image)
        torch.testing.assert_close(image, original, atol=0, rtol=0)
        expected = torch.tensor([0, 0, 128 / 255, 1, 1]).reshape(1, 1, 1, 5)
        torch.testing.assert_close(prediction, expected, atol=0, rtol=0)

    def test_quantization_rejects_nonfinite_predictions(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                quantize_prediction(torch.tensor([value]).reshape(1, 1, 1, 1))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for domain in ("LQ", "GT"):
            folder = self.root / domain / "sequence"
            folder.mkdir(parents=True)
            for index in range(4):
                image = np.full((136, 144, 3), index * 40, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / f"{index:05d}.png"), image))
        self.metadata = self.root / "meta.txt"
        self.metadata.write_text("sequence 4 (136,144,3) 0\n")
        self.options = {
            "lq": str(self.root / "LQ"),
            "gt": str(self.root / "GT"),
            "metadata": str(self.metadata),
            "frames": 2,
            "crop_size": 136,
            "extension": "png",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_frame_index_grain_and_tail_sampling(self):
        dataset = PairedVideoDataset(self.options)
        self.assertEqual(len(dataset), 4)
        random.seed(8)
        sample = dataset[3]
        self.assertEqual(sample["lq"].shape, (2, 3, 136, 136))
        torch.testing.assert_close(sample["lq"], sample["gt"], atol=0, rtol=0)
        self.assertEqual(sample["key"], "sequence/00003")
        first = round(sample["lq"][0].mean().item() * 255 / 40)
        self.assertLessEqual(first, 2)
        self.assertAlmostEqual(
            sample["lq"][1].mean().item() - sample["lq"][0].mean().item(), 40 / 255, places=6
        )

    def test_whole_sequence_resolution_and_missing_pair(self):
        dataset = PairedSequenceDataset(self.options)
        sample = dataset[0]
        self.assertEqual(tuple(sample["lq"].shape), (4, 3, 136, 144))
        (self.root / "GT/sequence/00003.png").unlink()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            PairedSequenceDataset(self.options)

    def test_short_training_video_and_unreadable_image(self):
        with self.assertRaisesRegex(ValueError, "fewer"):
            PairedVideoDataset({**self.options, "frames": 5})
        path = self.root / "bad.png"
        path.write_text("invalid")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            read_image(path)

    def test_sampler_epoch_and_distributed_padding(self):
        dataset = PairedVideoDataset(self.options)
        first = EnlargedSampler(dataset, num_replicas=3, rank=0)
        second = EnlargedSampler(dataset, num_replicas=3, rank=1)
        self.assertEqual(len(first), 2)
        self.assertEqual(list(first), list(first))
        expected = torch.randperm(6, generator=torch.Generator().manual_seed(0)).tolist()
        self.assertEqual(list(first), [index % 4 for index in expected[0::3]])
        self.assertEqual(list(second), [index % 4 for index in expected[1::3]])
        first.set_epoch(1)
        expected = torch.randperm(6, generator=torch.Generator().manual_seed(1)).tolist()
        self.assertEqual(list(first), [index % 4 for index in expected[0::3]])


if __name__ == "__main__":
    unittest.main()
