"""Complete native model checks; CPU-only environments explicitly skip these."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/MMCV integration check")
class ModelTests(unittest.TestCase):
    def test_full_model_reload_and_cpu_cache(self):
        from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import ModalContentPoleWaveletDWT3

        torch.manual_seed(4)
        model = ModalContentPoleWaveletDWT3().cuda().eval()
        # Activate residual heads so this covers conditioning, not only zero-output initialization.
        with torch.no_grad():
            for expert in model.wavelet_experts.values():
                for head in expert.residual_heads:
                    head.weight.normal_(std=1e-4)
        video = torch.rand(1, 3, 3, 257, 259, device="cuda")
        with torch.no_grad():
            ordinary = model(video)["restored"]
            model.inference_cpu_offload = True
            offloaded = model(video)["restored"]
        torch.testing.assert_close(ordinary.cpu(), offloaded.cpu(), atol=0, rtol=0)
        self.assertEqual(tuple(offloaded.shape), tuple(video.shape))
        self.assertIsNone(model._last_frame_tokens)
        state = model.state_dict()
        clone = ModalContentPoleWaveletDWT3().cuda().eval()
        clone.load_state_dict(state, strict=True)
        with torch.no_grad():
            reloaded = clone(video)["restored"]
        torch.testing.assert_close(ordinary, reloaded, atol=0, rtol=0)

    def test_all_conditioning_paths_receive_gradients(self):
        from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import ModalContentPoleWaveletDWT3

        torch.manual_seed(5)
        model = ModalContentPoleWaveletDWT3().cuda().train()
        with torch.no_grad():
            for expert in model.wavelet_experts.values():
                for head in expert.residual_heads:
                    head.weight.normal_(std=1e-4)
        video = torch.rand(1, 2, 3, 256, 256, device="cuda")
        outputs = model(video, gt=video)
        loss = (outputs["restored"] - video).square().mean()
        loss = loss + 0.05 * (outputs["stage1_preview"] - video).square().mean()
        loss = loss + 1e-3 * outputs["aux_losses"]["pole_separation"]
        loss.backward()
        groups = (
            "time_pole_context.mixing_mlp",
            "preview_condition",
            "stage1_preview",
            "direction_attention.stage_2",
            "history_cells.backward_2",
        )
        for group in groups:
            gradients = [
                p.grad
                for name, p in model.named_parameters()
                if name.startswith(group) and p.grad is not None
            ]
            self.assertTrue(gradients, group)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), group)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, group)


if __name__ == "__main__":
    unittest.main()
