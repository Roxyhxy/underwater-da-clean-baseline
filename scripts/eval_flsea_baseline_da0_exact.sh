#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="python"
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt"
SAVE_DIR="eval/flsea_baseline_da0_exact"

ENCODER="vits"
INPUT_SIZE=518
MAX_DEPTH=40.0

EXTRA_ARGS=("$@")

mkdir -p "${SAVE_DIR}"

set -x
CMD=(
  "${PYTHON_BIN}" eval_baseline_da0_exact.py
  --img-path "${VAL_LIST}"
  --encoder "${ENCODER}"
  --load-from "${CKPT}"
  --input-size "${INPUT_SIZE}"
  --max-depth "${MAX_DEPTH}"
  --outdir "${SAVE_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
