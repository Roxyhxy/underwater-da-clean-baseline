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
TEST_LIST="${TEST_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt}"
LOAD_FROM="${LOAD_FROM:-runs/ablation_${VARIANT}/best_abs_rel.pth}"
SAVE_DIR="${SAVE_DIR:-eval/ablation_${VARIANT}_test}"
SAVE_DEPTH="${SAVE_DEPTH:-true}"
DEPTH_OUTPUT_DIR="${DEPTH_OUTPUT_DIR:-${SAVE_DIR}/depth}"
DEPTH_COLORMAP="${DEPTH_COLORMAP:-Spectral_r}"
DEPTH_VIS_SPACE="${DEPTH_VIS_SPACE:-disparity}"

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

OUTPUT_ARGS=()
if [[ "${SAVE_DEPTH}" == "true" ]]; then
  mkdir -p "${DEPTH_OUTPUT_DIR}"
  OUTPUT_ARGS+=(
    --save-depth
    --depth-output-dir "${DEPTH_OUTPUT_DIR}"
    --depth-colormap "${DEPTH_COLORMAP}"
    --depth-vis-space "${DEPTH_VIS_SPACE}"
  )
fi

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
  "${OUTPUT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
