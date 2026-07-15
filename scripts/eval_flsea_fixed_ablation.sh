#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VARIANT="${1:-local_only}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SPLIT_LIST="${SPLIT_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt}"
LOAD_FROM="${LOAD_FROM:-runs/fixed_${VARIANT}_${RUN_TAG}/best_abs_rel.pth}"
SAVE_DIR="${SAVE_DIR:-eval/fixed_${VARIANT}_${RUN_TAG}_${SPLIT_NAME}}"

STRUCTURE_ARGS=()
case "${VARIANT}" in
  global_only)
    STRUCTURE_ARGS+=(--disable-local-prior)
    ;;
  full)
    ;;
  no_fft)
    STRUCTURE_ARGS+=(--disable-fft-prior)
    ;;
  local_only|local_consistency)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    ;;
  local_scalar)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior --disable-deg-map)
    ;;
  local_spatial_mean)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior --deg-map-spatial-mean)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    exit 2
    ;;
esac

mkdir -p "${SAVE_DIR}"
set -x
python eval_latent_prior.py \
  --encoder vits --load-from "${LOAD_FROM}" \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --val-list "${SPLIT_LIST}" --img-size 518 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.2 \
  --save-dir "${SAVE_DIR}" \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
