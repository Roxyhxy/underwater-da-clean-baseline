#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt}"
DATASET_ROOT="${DATASET_ROOT:-/data1/hxy/DATASET/FLSeaVI}"
WAT3R_ROOT="${WAT3R_ROOT:-/data1/hxy/Wat3R}"
WAT3R_CKPT="${WAT3R_CKPT:-/data1/hxy/Wat3R/checkpoints/wat3r.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/hxy/flsea_wat3r_train_teacher}"

set -x
python tools/generate_wat3r_teacher_dataset.py \
  --train-list "${TRAIN_LIST}" \
  --dataset-root "${DATASET_ROOT}" \
  --wat3r-root "${WAT3R_ROOT}" \
  --checkpoint "${WAT3R_CKPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --target-size 518 \
  --window-size 24 \
  --overlap 12 \
  --frames-chunk-size 4 \
  --min-visible-views 3 \
  "$@"
