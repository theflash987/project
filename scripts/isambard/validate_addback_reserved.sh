#!/bin/bash
set -euo pipefail
[[ "${SLURM_JOB_RESERVATION}" == interactive ]]
[[ "${SLURM_GPUS_ON_NODE}" == 4 ]]
module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/projects/b6bd/AverNet/conda/envs/avernet/bin/python
CONFIG=options/isambard/train_addback_20k.yml
export ABLATION_VARIANT=time_pole_fusion
export BASELINE_CODE=/scratch/b6bd/zhuodong26.b6bd/AverNet_6286184_NoBaseLoss/code
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/tests"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$(mktemp -d /tmp/addback-${SLURM_JOB_ID}.XXXXXX)"
cd "${REPO_DIR}"

OUTPUT_DIR="/scratch/b6bd/zhuodong26.b6bd/AverNet_6308179_TimePoleFusion/results/smoke_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1
echo "ADDBACK_SMOKE_START variant=time_pole_fusion job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
"${PYTHON}" tests/check_addback_config.py "${CONFIG}"
"${PYTHON}" tests/check_addback_cli.py "${CONFIG}"
"${PYTHON}" tests/check_official_dense_components.py
"${PYTHON}" tests/check_fusion_components.py
CONFIG="${CONFIG}" CHECK_BATCH=4 CHECK_FRAMES=12 \
  "${PYTHON}" tests/check_official_dense_propagation.py
EXP_NAME="smoke_6308179_TimePoleFusion_${SLURM_JOB_ID}"
SMOKE_CONFIG="${OUTPUT_DIR}/train_smoke.yml"
"${PYTHON}" - "${CONFIG}" "${SMOKE_CONFIG}" "${EXP_NAME}" <<'PY'
import sys
import yaml

source, destination, name = sys.argv[1:]
with open(source, encoding='utf-8') as stream:
    config = yaml.safe_load(stream)
config['name'] = name
config['train'].update(total_iter=30, fix_flow=10, flow_teacher_schedule=[11, 20])
config['logger'].update(print_freq=1, save_checkpoint_freq=30)
config['val']['val_freq'] = 1000000
with open(destination, 'w', encoding='utf-8') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
"${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train.py -opt "${SMOKE_CONFIG}"
"${PYTHON}" tests/check_addback_checkpoint.py "${CONFIG}" "${EXP_NAME}" "${OUTPUT_DIR}"
echo "ADDBACK_SMOKE_OK variant=time_pole_fusion job=${SLURM_JOB_ID}"
