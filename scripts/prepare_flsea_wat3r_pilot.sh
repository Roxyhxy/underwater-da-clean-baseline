#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SCENE_ROOT="${SCENE_ROOT:-/data1/hxy/flsea_wat3r_train_teacher/canyons/flatiron}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${SCENE_ROOT}/manifest.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${SCENE_ROOT}/manifest_overlap.csv}"
MASK_ROOT="${MASK_ROOT:-${SCENE_ROOT}/overlap_masks}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt}"
RELATIVE_THRESHOLD="${RELATIVE_THRESHOLD:-0.05}"

set -x
python tools/build_wat3r_overlap_masks.py \
  --manifest "${INPUT_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --mask-root "${MASK_ROOT}" \
  --relative-threshold "${RELATIVE_THRESHOLD}" \
  --min-align-pixels 1000 \
  --min-copies 2

python tools/audit_wat3r_teacher.py \
  --manifest "${OUTPUT_MANIFEST}" \
  --train-list "${TRAIN_LIST}" \
  --max-frames 300 \
  --require-overlap
