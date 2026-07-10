#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# =========================
# Edit this block directly
# =========================
PYTHON_BIN="python"
CKPT="/data1/hxy/DPV2/checkpoints/depth_anything_v2_vits.pth"
VAL_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/val.txt"
SAVE_DIR="eval/flsea_baseline"

ENCODER="vits"
IMG_SIZE=518
MIN_DEPTH=0.1
MAX_DEPTH=40.0
NUM_WORKERS=2

EXTRA_ARGS=("$@")

if [[ "${CKPT}" == "/path/to/depth_anything_v2_vits.pth" ]]; then
  echo "Please edit CKPT in scripts/eval_flsea_baseline.sh"
  exit 1
fi
if [[ "${VAL_LIST}" == "/path/to/flsea_val.txt" ]]; then
  echo "Please edit VAL_LIST in scripts/eval_flsea_baseline.sh"
  exit 1
fi

mkdir -p "${SAVE_DIR}"

set -x
CMD=(
  "${PYTHON_BIN}" eval_baseline.py
  --encoder "${ENCODER}"
  --pretrained-from "${CKPT}"
  --val-list "${VAL_LIST}"
  --img-size "${IMG_SIZE}"
  --min-depth "${MIN_DEPTH}"
  --max-depth "${MAX_DEPTH}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
