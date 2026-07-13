#!/usr/bin/env bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"


# =========================
# First-round research-one run:
# keep it clean, compare only latent prior against the verified baseline.
# =========================
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="python"
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
VAL_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/val.txt"
SAVE_PATH="runs/research1_latent_prior_frozen_head_fp32"

ENCODER="vits"
IMG_SIZE=518
EPOCHS=10
BS=4
LR=1e-4
PRIOR_LR=1e-4
PRIOR_HEAD_LR=5e-5
HEAD_LR=1e-5
BACKBONE_LR=0
MIN_DEPTH=0.1
MAX_DEPTH=40.0
NUM_WORKERS=4
PRIOR_BASE_CH=32
PRIOR_CHANNELS="32,64,128,256"
LATENT_DIM=128
PRIOR_FFT_SIZE=64
PRIOR_STAT_HIDDEN=64
DEG_MAP_SCALE=0.2
LOSS_MODE="depthdive_relative"
CONSISTENCY_HARDNESS_WEIGHT=0.0
CONSISTENCY_AUG_PROB=0.0
AMP="false"
GRAD_CLIP=1.0

# First-step strategy:
# 1) freeze DINOv2 backbone
# 2) train latent prior encoder
# 3) train new degradation-map / global-modulation path
# 4) keep the verified standard depth head unchanged for the first ablation
FREEZE_BACKBONE="true"
FREEZE_BASE_HEAD="true"
FREEZE_LATENT_PRIOR="false"

EXTRA_ARGS=("$@")

if [[ "${CKPT}" == "/path/to/depth_anything_v2_vits.pth" ]]; then
  echo "Please edit CKPT in scripts/train_flsea_latent_prior.sh"
  exit 1
fi
if [[ "${TRAIN_LIST}" == "/path/to/flsea_train.txt" ]]; then
  echo "Please edit TRAIN_LIST in scripts/train_flsea_latent_prior.sh"
  exit 1
fi
if [[ "${VAL_LIST}" == "/path/to/flsea_val.txt" ]]; then
  echo "Please edit VAL_LIST in scripts/train_flsea_latent_prior.sh"
  exit 1
fi

mkdir -p "${SAVE_PATH}"

set -x
CMD=(
  "${PYTHON_BIN}" train_latent_prior.py
  --encoder "${ENCODER}"
  --img-size "${IMG_SIZE}"
  --epochs "${EPOCHS}"
  --bs "${BS}"
  --lr "${LR}"
  --prior-lr "${PRIOR_LR}"
  --prior-head-lr "${PRIOR_HEAD_LR}"
  --head-lr "${HEAD_LR}"
  --backbone-lr "${BACKBONE_LR}"
  --pretrained-from "${CKPT}"
  --save-path "${SAVE_PATH}"
  --train-list "${TRAIN_LIST}"
  --val-list "${VAL_LIST}"
  --min-depth "${MIN_DEPTH}"
  --max-depth "${MAX_DEPTH}"
  --prior-base-ch "${PRIOR_BASE_CH}"
  --prior-channels "${PRIOR_CHANNELS}"
  --latent-dim "${LATENT_DIM}"
  --prior-fft-size "${PRIOR_FFT_SIZE}"
  --prior-stat-hidden "${PRIOR_STAT_HIDDEN}"
  --deg-map-scale "${DEG_MAP_SCALE}"
  --loss-mode "${LOSS_MODE}"
  --consistency-hardness-weight "${CONSISTENCY_HARDNESS_WEIGHT}"
  --consistency-aug-prob "${CONSISTENCY_AUG_PROB}"
  --grad-clip "${GRAD_CLIP}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${AMP}" == "true" ]]; then
  CMD+=(--amp)
fi

if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
  CMD+=(--freeze-backbone)
fi
if [[ "${FREEZE_BASE_HEAD}" == "true" ]]; then
  CMD+=(--freeze-base-head)
fi
if [[ "${FREEZE_LATENT_PRIOR}" == "true" ]]; then
  CMD+=(--freeze-latent-prior)
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
