#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Stable, reproducible research-one ablations.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/train_flsea_latent_prior_clean_ablation.sh local_only
VARIANT="${1:-local_only}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"

CKPT="${CKPT:-/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt}"
VAL_LIST="${VAL_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt}"

STRUCTURE_ARGS=()
GAUGE_ANCHOR_WEIGHT=0.02
case "${VARIANT}" in
  local_only)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    ;;
  local_spectral)
    STRUCTURE_ARGS+=(--disable-global-prior)
    ;;
  no_fft)
    STRUCTURE_ARGS+=(--disable-fft-prior)
    ;;
  full)
    ;;
  global_only)
    STRUCTURE_ARGS+=(--disable-local-prior)
    ;;
  local_only_no_anchor)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    GAUGE_ANCHOR_WEIGHT=0.0
    ;;
  full_no_anchor)
    GAUGE_ANCHOR_WEIGHT=0.0
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose: local_only, local_spectral, no_fft, full, global_only, local_only_no_anchor, full_no_anchor"
    exit 2
    ;;
esac

SAVE_PATH="runs/clean_${VARIANT}_${RUN_TAG}"
if [[ -e "${SAVE_PATH}" ]]; then
  echo "Refusing to overwrite existing run: ${SAVE_PATH}"
  exit 3
fi
mkdir -p "${SAVE_PATH}"

set -x
python train_latent_prior.py \
  --encoder vits --img-size 518 --epochs "${EPOCHS}" --bs 4 --seed "${SEED}" \
  --lr 2e-5 --prior-lr 2e-5 --prior-head-lr 2e-5 \
  --head-lr 0 --backbone-lr 0 \
  --pretrained-from "${CKPT}" --save-path "${SAVE_PATH}" \
  --train-list "${TRAIN_LIST}" --val-list "${VAL_LIST}" \
  --min-depth 0.1 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.1 \
  --loss-mode depthdive_relative \
  --l1-weight 0.5 --silog-weight 0.5 --metric-weight 1.0 --grad-weight 0.05 \
  --gauge-anchor-weight "${GAUGE_ANCHOR_WEIGHT}" \
  --consistency-hardness-weight 0 --consistency-aug-prob 0 \
  --warmup-steps 200 --min-lr-ratio 0.2 --weight-decay 0 \
  --grad-clip 1.0 --num-workers 4 \
  --freeze-backbone --freeze-base-head --eval-before-train \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
