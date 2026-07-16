#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Edit these paths on the server.
DA3_ROOT="/data1/hxy/Depth-Anything-3"
COLMAP_DIR="/data1/hxy/flsea_colmap_pilot/dense"
MODEL="depth-anything/DA3-LARGE-1.1"
OUTPUT_DIR="/data1/hxy/flsea_colmap_pilot/da3_dense"

GPU_ID=1
PROCESS_RES=504
WINDOW_SIZE=24
OVERLAP=12

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python tools/generate_da3_colmap_depth.py \
  --da3-root "${DA3_ROOT}" \
  --colmap-dir "${COLMAP_DIR}" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --process-res "${PROCESS_RES}" \
  --window-size "${WINDOW_SIZE}" \
  --overlap "${OVERLAP}" \
  --ref-view-strategy middle \
  --save-npy \
  --save-visualization \
  "$@"
