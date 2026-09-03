# ModalContent Pole-Wavelet NoRouter-K16

This directory contains the minimal source closure for the NoRouter-K16
experiment: one generator, one training wrapper, the DAVIS train/validation
datasets, the exact 20k configuration, and its train/validation scripts.

## Model

`AverNetModalContentPoleWaveletDWT3OfficialDense` uses:

- 12-frame ModalContent preprocessing;
- official four-pass dense propagation with DCN alignment;
- a global learned `K=16`, `C=2` fixed complex-pole bank;
- Stage-1 bidirectional attention for preview supervision;
- geometry-bounded Pole memory in all four propagation branches;
- independent W3/W2/W1 direct-residual experts over critically sampled
  three-level bior4.4 wavelets;
- a frame-local orientation-specific guide only for W1;
- detached DWT anchors with a live base reconstruction path;
- deterministic W3 -> W2 -> W1 inverse reconstruction.

The loss implemented by `ModalContentPoleWaveletDWT3VideoModel` is exactly:

```text
L_final_rec + 0.05 L_final_ssim
+ L_base_rec + 0.05 L_base_ssim
+ 0.1 L_wave_res + 0.05 L_preview
+ lambda_flow(t) L_flow + 1e-3 L_pole_separation
```

SpyNet levels 0-3 are permanently frozen. Levels 4-5 receive `0.15x` the
backbone learning rate after iteration 1250. The clean-GT SpyNet teacher is
training-only.

## Isambard execution

The existing project environment must provide PyTorch 2.3.1/CUDA 12.6 and a
compatible MMCV build containing `mmcv.ops.modulated_deform_conv`. Remaining
Python dependencies are listed in `requirements-isambard.txt`.

Formal training:

```bash
sbatch scripts/isambard/train_norouter_k16_4gpu_20k.sh
```

Reserved-node validation must be launched inside a four-GPU interactive
reservation:

```bash
bash scripts/isambard/validate_norouter_k16_reserved.sh
```

`KEEP_MANIFEST.txt` is the authoritative file whitelist for this directory.
