#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# =========================
# Recommended final baseline reproduction:
# DA_0-compatible FLSea evaluation at original image resolution.
# =========================
PYTHON_BIN="python"
# Match DA_0/run_FLSea.sh by default.
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt"
SAVE_DIR="eval/flsea_baseline_legacy"

ENCODER="vits"
INPUT_SIZE=518
MAX_DEPTH=40.0

EXTRA_ARGS=("$@")

if [[ "${CKPT}" == "/path/to/depth_anything_v2_vits.pth" ]]; then
  echo "Please edit CKPT in scripts/eval_flsea_baseline_legacy.sh"
  exit 1
fi
if [[ "${VAL_LIST}" == "/path/to/flsea_val.txt" ]]; then
  echo "Please edit VAL_LIST in scripts/eval_flsea_baseline_legacy.sh"
  exit 1
fi

mkdir -p "${SAVE_DIR}"

set -x
CMD=(
  "${PYTHON_BIN}" eval_baseline_legacy.py
  --img-path "${VAL_LIST}"
  --encoder "${ENCODER}"
  --load-from "${CKPT}"
  --input-size "${INPUT_SIZE}"
  --max-depth "${MAX_DEPTH}"
  --save-dir "${SAVE_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
