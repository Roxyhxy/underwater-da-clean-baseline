#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TEACHER_ROOT="${TEACHER_ROOT:-/data1/hxy/flsea_wat3r_train_teacher}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${TEACHER_ROOT}/manifest_all.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${TEACHER_ROOT}/manifest_all_overlap.csv}"
MASK_ROOT="${MASK_ROOT:-${TEACHER_ROOT}/overlap_masks}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt}"
RELATIVE_THRESHOLD="${RELATIVE_THRESHOLD:-0.05}"

if [[ ! -s "${INPUT_MANIFEST}" ]]; then
  echo "Missing complete all-scene manifest: ${INPUT_MANIFEST}"
  echo "Finish scripts/generate_flsea_wat3r_all_train.sh first."
  exit 2
fi

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
  --max-frames 600 \
  --require-overlap
