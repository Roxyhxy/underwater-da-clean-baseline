#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Run once per FLSea scene. Example:
# SCENE=flatiron IMAGE_DIR=/data1/hxy/DATASET/FLSeaVI/canyons/flatiron/imgs \
#   bash scripts/generate_flsea_wat3r_teacher.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

WAT3R_ROOT="${WAT3R_ROOT:-/data1/hxy/Wat3R}"
WAT3R_CKPT="${WAT3R_CKPT:-/data1/hxy/Wat3R/checkpoints/wat3r.pth}"
SCENE="${SCENE:-flatiron}"
IMAGE_DIR="${IMAGE_DIR:-/data1/hxy/DATASET/FLSeaVI/canyons/flatiron/imgs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/hxy/flsea_wat3r_teacher}"

TARGET_SIZE="${TARGET_SIZE:-518}"
WINDOW_SIZE="${WINDOW_SIZE:-24}"
OVERLAP="${OVERLAP:-12}"
FRAMES_CHUNK_SIZE="${FRAMES_CHUNK_SIZE:-4}"

EXTRA_ARGS=("$@")

set -x
python tools/generate_wat3r_teacher_windows.py \
  --wat3r-root "${WAT3R_ROOT}" \
  --checkpoint "${WAT3R_CKPT}" \
  --image-dir "${IMAGE_DIR}" \
  --image-glob '*.tiff' \
  --output-dir "${OUTPUT_ROOT}/${SCENE}" \
  --device cuda \
  --target-size "${TARGET_SIZE}" \
  --window-size "${WINDOW_SIZE}" \
  --overlap "${OVERLAP}" \
  --frames-chunk-size "${FRAMES_CHUNK_SIZE}" \
  --min-visible-views 3 \
  --relative-depth-threshold 0.05 \
  --boundary 4 \
  --save-visualization \
  "${EXTRA_ARGS[@]}"
