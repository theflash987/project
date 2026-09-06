#!/bin/bash
#SBATCH --job-name=time-pole-resume-next
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=72
#SBATCH --mem=460000M
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:6360830
#SBATCH --no-requeue
#SBATCH --output=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/results/resume_next.%j.out

set -euo pipefail
module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME
ROOT=/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion
REPO_DIR="${ROOT}/code"
PYTHON=/projects/b6bd/AverNet/conda/envs/avernet/bin/python
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/tests"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$(mktemp -d /tmp/timepole-next-${SLURM_JOB_ID}.XXXXXX)"
cd "${REPO_DIR}"

[[ "${SLURM_GPUS_ON_NODE}" == 4 ]]
EXP_NAME=AverNet_TimePoleFusion_after6360830_24h_cos150k
EXP_DIR="${ROOT}/experiments/${EXP_NAME}"
if [[ -e "${EXP_DIR}" ]]; then
  echo "Resume output refuses an existing experiment: ${EXP_DIR}" >&2
  exit 1
fi
CONFIG="${ROOT}/results/dependency_config.${SLURM_JOB_ID}.yml"
"${PYTHON}" scripts/prepare_time_pole_dependency.py \
  options/isambard/resume_time_pole_24h.yml \
  "${ROOT}/experiments/AverNet_TimePoleFusion_resume20000_24h_cos150k" \
  "${EXP_NAME}" "${CONFIG}"
"${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train_resume.py -opt "${CONFIG}"
echo "TIME_POLE_DEPENDENCY_RESUME_COMPLETE predecessor=6360830 job=${SLURM_JOB_ID}"
