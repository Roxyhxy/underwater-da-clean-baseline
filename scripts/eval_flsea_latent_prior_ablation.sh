#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VARIANT="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
TEST_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt"
LOAD_FROM="runs/ablation_${VARIANT}/best_abs_rel.pth"
SAVE_DIR="eval/ablation_${VARIANT}_test"

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
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose one of: full, global_only, local_only, no_fft"
    exit 2
    ;;
esac

set -x
python eval_latent_prior.py \
  --encoder vits \
  --load-from "${LOAD_FROM}" \
  --pretrained-from "${CKPT}" \
  --val-list "${TEST_LIST}" \
  --img-size 518 \
  --max-depth 40.0 \
  --prior-base-ch 32 \
  --prior-channels 32,64,128,256 \
  --latent-dim 128 \
  --prior-fft-size 64 \
  --prior-stat-hidden 64 \
  --deg-map-scale 0.2 \
  --save-dir "${SAVE_DIR}" \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
