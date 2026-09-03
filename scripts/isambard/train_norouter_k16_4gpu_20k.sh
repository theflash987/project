#!/bin/bash
#SBATCH --job-name=modal-norouter-k16-20k
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=72
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/b6bd/zhuodong26.b6bd/AverNet/logs/%x.%j.out

set -euo pipefail

module load cray-python/3.11.7
module load cuda/12.6
unset PYTHONHOME

REPO_DIR="${SCRATCHDIR}/AverNet_BasicSR_ModalContentPoleWaveletDWT3NoRouterK16/code"
BASE_DIR="${PROJECTDIR}/AverNet"
SCRATCH_BASE="${SCRATCHDIR}/AverNet"
CONFIG="options/isambard/train_AverNet_ModalContentPoleWaveletDWT3NoRouterK16_numframe12_20k_cos150k.yml"
EXP_NAME="AverNet_ModalContentPoleWaveletDWT3NoRouterK16_num_frame12_bs4_4gpu_20k_cos150k"
PYTHON="${BASE_DIR}/conda/envs/avernet/bin/python"

[[ "${SLURM_GPUS_ON_NODE:-4}" == "4" ]]
test -x "${PYTHON}"
test -f "${REPO_DIR}/${CONFIG}"

export PYTHONPATH="${REPO_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="/tmp/nr-${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${SCRATCH_BASE}/cache/pycache_modal_norouter_k16"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${SCRATCH_BASE}/logs"

cd "${REPO_DIR}"
EXP_DIR="${SCRATCH_BASE}/experiments/${EXP_NAME}"
if [[ -e "${EXP_DIR}" ]]; then
  echo "Fresh run refuses existing experiment: ${EXP_DIR}" >&2
  exit 1
fi

"${PYTHON}" -m py_compile \
  basicsr/archs/official_dense_components.py \
  basicsr/archs/modal_content_dwt3_components.py \
  basicsr/archs/modal_content_pole_wavelet_dwt3_arch.py \
  basicsr/models/modal_content_pole_wavelet_dwt3_video_model.py

CONFIG="${CONFIG}" "${PYTHON}" - <<'PY'
import os
from basicsr.archs import build_network
from basicsr.utils.options import yaml_load

opt = yaml_load(os.environ['CONFIG'])
net_opt = opt['network_g']
train = opt['train']
assert opt['datasets']['train']['num_frame'] == 12
assert opt['datasets']['train']['gt_size'] == 256
assert opt['datasets']['train']['batch_size_per_gpu'] == 4
assert opt['datasets']['train']['num_worker_per_gpu'] == 4
assert train['total_iter'] == 20000
assert train['scheduler']['periods'] == [150000]
assert train['fix_flow'] == 1250
assert train['trainable_spynet_levels'] == [4, 5]
assert train['flow_lr_mul'] == 0.15
assert train['flow_teacher_schedule'] == [1251, 2250]
assert train['lambda_rec'] == 1.0
assert train['lambda_ssim'] == 0.05
assert train['lambda_base_rec'] == 1.0
assert train['lambda_base_ssim'] == 0.05
assert train['lambda_wave_res'] == 0.1
assert train['lambda_preview'] == 0.05
assert train['lambda_flow'] == 0.02
assert train['energy_ema_decay'] == 0.99
assert train['energy_floor'] == 1e-4
assert train['energy_weight_bounds'] == [0.25, 4.0]
assert set(train['aux_loss_weights']) == {'pole_separation'}
assert net_opt['w1_guide_channels'] == 32
for forbidden in (
        'router_channels', 'modal_model_channels', 'modal_num_heads',
        'modal_num_blocks', 'pole_scene_reset_threshold',
        'pole_scene_reset_temperature', 'pole_dead_occupancy_floor',
        'inter_stage_demand_scale'):
    assert forbidden not in net_opt
for forbidden in (
        'lambda_band', 'lambda_pyramid', 'lambda_pc', 'lambda_gate',
        'gate_teacher_window', 'gain_teacher_eps'):
    assert forbidden not in train
network = build_network(net_opt)
assert network.__class__.__name__ == 'AverNetModalContentPoleWaveletDWT3OfficialDense'
assert not hasattr(network.shared_poles, 'adapter')
assert network.num_poles == 16 and network.pole_state_channels == 2
assert set(network.direction_attention) == {'stage_1'}
assert set(network.wavelet_experts) == {'w3', 'w2', 'w1'}
assert network.wavelet_experts['w3'].spatial_film is None
assert network.wavelet_experts['w2'].spatial_film is None
assert network.wavelet_experts['w1'].spatial_film is not None
assert sum(len(expert.residual_heads)
           for expert in network.wavelet_experts.values()) == 9
for level in range(4):
    assert not any(parameter.requires_grad
                   for parameter in network.spynet.basic_module[level].parameters())
for level in (4, 5):
    assert all(parameter.requires_grad
               for parameter in network.spynet.basic_module[level].parameters())
for key in network.state_dict():
    assert not any(token in key.lower() for token in (
        'pole_wavelet_router', 'direction_attention.stage_2',
        'modal_refiner', 'inter_stage_reanchor', 'keep_gate',
        'history_strength', 'propagation_logit', 'parent', 'gain',
        'rho_delta', 'omega_delta', 'complex_w2'))
print('MODAL_CONTENT_NOROUTER_K16_FULL_CONFIG_OK')
PY

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  basicsr/train.py \
  -opt "${CONFIG}" \
  --force_yml \
  "name=${EXP_NAME}" \
  "datasets:train:num_frame=12" \
  "datasets:train:batch_size_per_gpu=4"

echo "MODAL_CONTENT_NOROUTER_K16_20K_OK ${EXP_DIR}"
