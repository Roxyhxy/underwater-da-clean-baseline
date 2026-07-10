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
LOAD_FROM="/path/to/best_abs_rel.pth"
VAL_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/val.txt"
SAVE_DIR="eval/flsea_latent_prior_eval"

ENCODER="vits"
IMG_SIZE=518
MIN_DEPTH=0.1
MAX_DEPTH=40.0
NUM_WORKERS=2
PRIOR_BASE_CH=32
PRIOR_CHANNELS="32,64,128,256"
LATENT_DIM=128
PRIOR_FFT_SIZE=64
PRIOR_STAT_HIDDEN=64
DEG_MAP_SCALE=0.2

SAVE_DEPTH="true"
DEPTH_OUTPUT_DIR="eval/flsea_latent_prior_eval/depth"

EXTRA_ARGS=("$@")

if [[ "${CKPT}" == "/path/to/depth_anything_v2_vits.pth" ]]; then
  echo "Please edit CKPT in scripts/eval_flsea_latent_prior.sh"
  exit 1
fi
if [[ "${LOAD_FROM}" == "/path/to/best_abs_rel.pth" ]]; then
  echo "Please edit LOAD_FROM in scripts/eval_flsea_latent_prior.sh"
  exit 1
fi
if [[ "${VAL_LIST}" == "/path/to/flsea_val.txt" ]]; then
  echo "Please edit VAL_LIST in scripts/eval_flsea_latent_prior.sh"
  exit 1
fi

mkdir -p "${SAVE_DIR}"

set -x
CMD=(
  "${PYTHON_BIN}" eval_latent_prior.py
  --encoder "${ENCODER}"
  --load-from "${LOAD_FROM}"
  --pretrained-from "${CKPT}"
  --val-list "${VAL_LIST}"
  --img-size "${IMG_SIZE}"
  --min-depth "${MIN_DEPTH}"
  --max-depth "${MAX_DEPTH}"
  --prior-base-ch "${PRIOR_BASE_CH}"
  --prior-channels "${PRIOR_CHANNELS}"
  --latent-dim "${LATENT_DIM}"
  --prior-fft-size "${PRIOR_FFT_SIZE}"
  --prior-stat-hidden "${PRIOR_STAT_HIDDEN}"
  --deg-map-scale "${DEG_MAP_SCALE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
)

if [[ "${SAVE_DEPTH}" == "true" ]]; then
  CMD+=(--save-depth)
fi
if [[ -n "${DEPTH_OUTPUT_DIR}" ]]; then
  CMD+=(--depth-output-dir "${DEPTH_OUTPUT_DIR}")
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
