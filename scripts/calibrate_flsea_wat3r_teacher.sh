#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

WAT3R_MANIFEST="${WAT3R_MANIFEST:-/data1/hxy/flsea_wat3r_train_teacher/manifest_all_overlap.csv}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt}"
OUTPUT_CSV="${OUTPUT_CSV:-analysis/wat3r_reliability_calibration.csv}"
MAX_ROWS="${MAX_ROWS:-2400}"
CONFIDENCE_QUANTILE="${CONFIDENCE_QUANTILE:-0.6}"
MIN_HOLE_PIXELS="${MIN_HOLE_PIXELS:-256}"

if [[ ! -f "${WAT3R_MANIFEST}" ]]; then
  echo "Missing Wat3R manifest: ${WAT3R_MANIFEST}"
  exit 2
fi

set -x
python tools/calibrate_wat3r_teacher.py \
  --manifest "${WAT3R_MANIFEST}" \
  --train-list "${TRAIN_LIST}" \
  --output-csv "${OUTPUT_CSV}" \
  --quantiles 0,0.2,0.4,0.6,0.8,0.9 \
  --max-rows "${MAX_ROWS}" \
  --min-align-pixels 1000 \
  --max-depth 40

python tools/audit_wat3r_teacher.py \
  --manifest "${WAT3R_MANIFEST}" \
  --train-list "${TRAIN_LIST}" \
  --max-frames 600 \
  --confidence-quantile "${CONFIDENCE_QUANTILE}" \
  --min-hole-pixels "${MIN_HOLE_PIXELS}" \
  --require-overlap
