#!/bin/bash

set -euo pipefail

if [[ "${SLURM_JOB_RESERVATION:-}" != "interactive" ]]; then
  echo "Run this script inside an Isambard interactive reservation." >&2
  exit 1
fi
if [[ "${SLURM_GPUS_ON_NODE:-0}" != "4" ]]; then
  echo "The exact DDP smoke requires one reserved 4-GPU node." >&2
  exit 1
fi

module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME

REPO_DIR="${SCRATCHDIR}/AverNet_BasicSR_ModalContentPoleWaveletDWT3NoRouterK16/code"
BASE_DIR="${PROJECTDIR}/AverNet"
SCRATCH_BASE="${SCRATCHDIR}/AverNet"
CONFIG="options/isambard/train_AverNet_ModalContentPoleWaveletDWT3NoRouterK16_numframe12_20k_cos150k.yml"
EXP_NAME="smoke_ModalContentPoleWaveletNoRouterK16_${SLURM_JOB_ID}"
PYTHON="${BASE_DIR}/conda/envs/avernet/bin/python"

test -x "${PYTHON}"
test -f "${REPO_DIR}/${CONFIG}"
EXPERIMENT_ROOT="$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["path"]["experiments_root"])' "${REPO_DIR}/${CONFIG}")"
EXP_DIR="${EXPERIMENT_ROOT}/${EXP_NAME}"
export PYTHONPATH="${REPO_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="/tmp/nr-${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${SCRATCH_BASE}/cache/pycache_modal_norouter_k16"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${SCRATCH_BASE}/logs"
cd "${REPO_DIR}"

"${PYTHON}" -m py_compile \
  basicsr/archs/modal_content_dwt3_components.py \
  basicsr/archs/official_dense_components.py \
  basicsr/archs/modal_content_pole_wavelet_dwt3_arch.py \
  basicsr/models/modal_content_pole_wavelet_dwt3_video_model.py \
  tests/check_official_dense_components.py \
  tests/check_official_dense_propagation.py

"${PYTHON}" tests/check_official_dense_components.py

# One-GPU production shape: the same B=4, T=12 and 256 crop used per DDP rank.
CONFIG="${CONFIG}" CHECK_BATCH=4 CHECK_FRAMES=12 \
  "${PYTHON}" tests/check_official_dense_propagation.py

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  basicsr/train.py \
  -opt "${CONFIG}" \
  --force_yml \
  "name=${EXP_NAME}" \
  "datasets:train:num_frame=12" \
  "datasets:train:gt_size=256" \
  "datasets:train:batch_size_per_gpu=4" \
  "datasets:train:num_worker_per_gpu=4" \
  "train:total_iter=30" \
  "train:fix_flow=10" \
  "train:flow_teacher_schedule=[11,20]" \
  "logger:print_freq=1" \
  "logger:save_checkpoint_freq=30" \
  "val:val_freq=1000000"

EXP_DIR="${EXP_DIR}" CONFIG="${CONFIG}" "${PYTHON}" - <<'PY'
import glob
import math
import os
import re
import torch
from basicsr.archs import build_network
from basicsr.utils.options import yaml_load

exp_dir = os.environ['EXP_DIR']
logs = sorted(glob.glob(os.path.join(exp_dir, 'train_*.log')))
assert logs, exp_dir
text = open(logs[-1], encoding='utf-8').read()

def values(name):
    pattern = rf'{re.escape(name)}:\s*([+\-0-9.eE]+)'
    return [float(value) for value in re.findall(pattern, text)]

wavelet = values('wavelet_branch_grad_norm')
spynet = values('spynet_grad_norm')
direction = values('stage1_direction_grad_norm')
totals = values('l_total')
assert len(wavelet) >= 20 and all(value > 0 for value in wavelet)
assert len(spynet) >= 20 and any(value > 0 for value in spynet[10:])
assert len(direction) >= 20 and all(value > 0 for value in direction[1:])
assert totals and all(math.isfinite(value) for value in totals)
for name in (
        'l_final_rec', 'l_final_ssim', 'l_base_rec', 'l_base_ssim',
        'l_wave_res', 'l_preview', 'l_flow_teacher',
        'base_to_final_residual_abs_mean', 'w1_local_guide_abs_mean',
        'grad_norm'):
    assert values(name) and all(math.isfinite(value) for value in values(name)), name
assert all(value == 0 for value in spynet[:10])
assert all(value > 0 for value in spynet[10:])

checkpoint_path = os.path.join(exp_dir, 'models', 'net_g_30.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
assert 'wavelet_residual_energy_ema' in checkpoint['params']
assert 'wavelet_residual_energy_initialized' in checkpoint['params']
option = yaml_load(os.environ['CONFIG'])['network_g']
network = build_network(option).cuda().eval()
network.load_state_dict(checkpoint['params'], strict=True)
ema_network = build_network(option).cuda().eval()
ema_network.load_state_dict(checkpoint['params_ema'], strict=True)
assert network.wavelet_residual_energy_initialized.item()
assert torch.equal(
    network.wavelet_residual_energy_ema.cpu(),
    ema_network.wavelet_residual_energy_ema.cpu())

# Exercise the inference-only chunk path on a sequence longer than one chunk.
with torch.no_grad():
    long_input = torch.rand(1, 20, 3, 256, 256, device='cuda')
    result = ema_network(long_input)
assert result['restored'].shape == long_input.shape
assert all(torch.isfinite(value).all() for value in (
    result['restored'], result['base_restored']))
print('NOROUTER_DDP_METRICS', {'wavelet_last': wavelet[-1], 'direction_last': direction[-1], 'spynet_last': spynet[-1], 'loss_last': totals[-1]})
print('NOROUTER_LONG_SEQUENCE_PEAK_GIB', torch.cuda.max_memory_reserved() / 2**30)
print('MODAL_CONTENT_NOROUTER_K16_RESERVED_VALIDATION_OK')
PY

echo "MODAL_CONTENT_NOROUTER_K16_SMOKE_OK ${EXP_DIR}"
