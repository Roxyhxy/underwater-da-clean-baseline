#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# =========================
# Recommended latent-prior evaluation:
# use the same FLSea legacy protocol as the verified baseline.
# =========================
PYTHON_BIN="python"
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
LOAD_FROM="/path/to/best_abs_rel.pth"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt"
SAVE_DIR="eval/flsea_latent_prior_eval"

ENCODER="vits"
IMG_SIZE=518
MAX_DEPTH=40.0
PRIOR_BASE_CH=32
PRIOR_CHANNELS="32,64,128,256"
LATENT_DIM=128
PRIOR_FFT_SIZE=64
PRIOR_STAT_HIDDEN=64
DEG_MAP_SCALE=0.2

SAVE_DEPTH="false"
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
  --max-depth "${MAX_DEPTH}"
  --prior-base-ch "${PRIOR_BASE_CH}"
  --prior-channels "${PRIOR_CHANNELS}"
  --latent-dim "${LATENT_DIM}"
  --prior-fft-size "${PRIOR_FFT_SIZE}"
  --prior-stat-hidden "${PRIOR_STAT_HIDDEN}"
  --deg-map-scale "${DEG_MAP_SCALE}"
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
