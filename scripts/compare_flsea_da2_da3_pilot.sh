#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_DIR="${IMAGE_DIR:-/data1/hxy/flsea_colmap_pilot/dense/images}"
DA2_CHECKPOINT="${DA2_CHECKPOINT:-/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth}"
DA3_DEPTH_DIR="${DA3_DEPTH_DIR:-/data1/hxy/flsea_colmap_pilot/da3_dense_756/depth_tiff}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/hxy/flsea_colmap_pilot/da2_da3_comparison}"
GPU_ID="${GPU_ID:-1}"

# DA2 uses short-side lower-bound resize. For FLSea, 462 produces about
# 462x756 input, matching DA3 process-res=756 (long-side upper bound).
DA2_INPUT_SIZE="${DA2_INPUT_SIZE:-462}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u tools/compare_da2_da3_pilot.py \
  --image-dir "${IMAGE_DIR}" \
  --da2-checkpoint "${DA2_CHECKPOINT}" \
  --da3-depth-dir "${DA3_DEPTH_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --encoder vits \
  --da2-input-size "${DA2_INPUT_SIZE}" \
  --device cuda \
  "$@"
