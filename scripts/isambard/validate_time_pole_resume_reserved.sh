#!/bin/bash
set -euo pipefail
module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME
REPO_DIR=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/code
PYTHON=/projects/b6bd/AverNet/conda/envs/avernet/bin/python
export PYTHONPATH="${REPO_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${REPO_DIR}"
"${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  tests/check_time_pole_resume.py -opt options/isambard/resume_time_pole_24h.yml \
  --force_yml "name=smoke_TimePoleFusion_resume20k_${SLURM_JOB_ID}" \
  "train:total_iter=20110" "resume:max_seconds=0" "logger:print_freq=10"
echo "TIME_POLE_TRUE_RESUME_SMOKE_OK job=${SLURM_JOB_ID}"
