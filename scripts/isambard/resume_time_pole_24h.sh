#!/bin/bash
#SBATCH --job-name=time-pole-resume20k
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=72
#SBATCH --mem=460000M
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/results/resume.%j.out

set -euo pipefail
module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME
REPO_DIR=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/code
PYTHON=/projects/b6bd/AverNet/conda/envs/avernet/bin/python
CONFIG=options/isambard/resume_time_pole_24h.yml
export ABLATION_VARIANT=time_pole_fusion
export BASELINE_CODE=/scratch/b6bd/zhuodong26.b6bd/AverNet_6286184_NoBaseLoss/code
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/tests"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$(mktemp -d /tmp/addback-${SLURM_JOB_ID}.XXXXXX)"
cd "${REPO_DIR}"

[[ "${SLURM_GPUS_ON_NODE}" == 4 ]]
EXP_NAME=AverNet_TimePoleFusion_resume20000_24h_cos150k
EXP_DIR=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/experiments/${EXP_NAME}
if [[ -e "${EXP_DIR}" ]]; then
  echo "Resume output refuses an existing experiment: ${EXP_DIR}" >&2
  exit 1
fi
"${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train_resume.py -opt "${CONFIG}"
echo "TIME_POLE_RESUME_JOB_COMPLETE source=6336419 job=${SLURM_JOB_ID}"
