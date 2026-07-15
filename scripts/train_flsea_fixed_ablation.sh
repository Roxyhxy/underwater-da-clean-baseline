#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Fixed research-one ablations. Existing runs are never overwritten.
# Usage: SEED=42 bash scripts/train_flsea_fixed_ablation.sh <variant>
VARIANT="${1:-local_only}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
EPOCHS="${EPOCHS:-5}"

CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt"

STRUCTURE_ARGS=()
CONSISTENCY_WEIGHT=0.0
CONSISTENCY_PROB=0.0
case "${VARIANT}" in
  global_only)
    STRUCTURE_ARGS+=(--disable-local-prior)
    ;;
  full)
    ;;
  no_fft)
    STRUCTURE_ARGS+=(--disable-fft-prior)
    ;;
  local_only)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    ;;
  local_scalar)
    # Dataset-level learned scalar per scale; no spatial or sample-specific map.
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior --disable-deg-map)
    ;;
  local_spatial_mean)
    # Sample-specific mean gate with exactly the same average map magnitude.
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior --deg-map-spatial-mean)
    ;;
  local_consistency)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    CONSISTENCY_WEIGHT=0.08
    CONSISTENCY_PROB=1.0
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose: global_only, full, no_fft, local_only, local_scalar, local_spatial_mean, local_consistency"
    exit 2
    ;;
esac

SAVE_PATH="runs/fixed_${VARIANT}_${RUN_TAG}"
mkdir -p "${SAVE_PATH}"

set -x
python train_latent_prior.py \
  --encoder vits --img-size 518 --epochs "${EPOCHS}" --bs 4 --seed "${SEED}" \
  --lr 1e-4 --prior-lr 1e-4 --prior-head-lr 5e-5 --head-lr 1e-5 --backbone-lr 0 \
  --pretrained-from "${CKPT}" --save-path "${SAVE_PATH}" \
  --train-list "${TRAIN_LIST}" --val-list "${VAL_LIST}" \
  --min-depth 0.1 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.2 \
  --loss-mode depthdive_relative \
  --consistency-hardness-weight "${CONSISTENCY_WEIGHT}" \
  --consistency-aug-prob "${CONSISTENCY_PROB}" \
  --consistency-hardness-clamp-min 0.90 --consistency-hardness-clamp-max 1.10 \
  --consistency-blur-prob 0.30 --consistency-noise-prob 0.20 --consistency-noise-std 0.01 \
  --grad-clip 1.0 --num-workers 4 \
  --freeze-backbone --freeze-base-head --eval-before-train \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
