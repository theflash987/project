# 6240289 + UniformPoleW32

Independent uniform_pole_w32 add-back. See ../../README.md for the study protocol.

- Config: options/isambard/train_addback_20k.yml
- Reserved smoke: bash scripts/isambard/validate_addback_reserved.sh
- Fresh 20k: sbatch scripts/isambard/train_addback_4gpu_20k.sh
- Submit only after this exact copy passes its smoke.
- Baseline losses, seed, 12-frame/256-crop/batch-4-per-GPU protocol are unchanged.
