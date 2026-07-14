#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Usage:
#   bash scripts/train_flsea_latent_prior_ablation.sh full
#   bash scripts/train_flsea_latent_prior_ablation.sh global_only
#   bash scripts/train_flsea_latent_prior_ablation.sh local_only
#   bash scripts/train_flsea_latent_prior_ablation.sh no_fft
#   bash scripts/train_flsea_latent_prior_ablation.sh no_deg_map
VARIANT="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="python"
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt"

ENCODER="vits"
IMG_SIZE=518
EPOCHS=5
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
GRAD_CLIP=1.0

STRUCTURE_ARGS=()
case "${VARIANT}" in
  full)
    ;;
  global_only)
    STRUCTURE_ARGS+=(--disable-local-prior)
    ;;
  local_only)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    ;;
  no_fft)
    STRUCTURE_ARGS+=(--disable-fft-prior)
    ;;
  no_deg_map)
    STRUCTURE_ARGS+=(--disable-deg-map)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose one of: full, global_only, local_only, no_fft, no_deg_map"
    exit 2
    ;;
esac

SAVE_PATH="runs/ablation_${VARIANT}"
mkdir -p "${SAVE_PATH}"

set -x
"${PYTHON_BIN}" train_latent_prior.py \
  --encoder "${ENCODER}" \
  --img-size "${IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --lr "${LR}" \
  --prior-lr "${PRIOR_LR}" \
  --prior-head-lr "${PRIOR_HEAD_LR}" \
  --head-lr "${HEAD_LR}" \
  --backbone-lr "${BACKBONE_LR}" \
  --pretrained-from "${CKPT}" \
  --save-path "${SAVE_PATH}" \
  --train-list "${TRAIN_LIST}" \
  --val-list "${VAL_LIST}" \
  --min-depth "${MIN_DEPTH}" \
  --max-depth "${MAX_DEPTH}" \
  --prior-base-ch "${PRIOR_BASE_CH}" \
  --prior-channels "${PRIOR_CHANNELS}" \
  --latent-dim "${LATENT_DIM}" \
  --prior-fft-size "${PRIOR_FFT_SIZE}" \
  --prior-stat-hidden "${PRIOR_STAT_HIDDEN}" \
  --deg-map-scale "${DEG_MAP_SCALE}" \
  --loss-mode "${LOSS_MODE}" \
  --consistency-hardness-weight "${CONSISTENCY_HARDNESS_WEIGHT}" \
  --consistency-aug-prob "${CONSISTENCY_AUG_PROB}" \
  --grad-clip "${GRAD_CLIP}" \
  --num-workers "${NUM_WORKERS}" \
  --freeze-backbone \
  --freeze-base-head \
  --eval-before-train \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
