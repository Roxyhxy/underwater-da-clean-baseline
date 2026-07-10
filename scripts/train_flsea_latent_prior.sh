#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
CKPT=${CKPT:-}
TRAIN_LIST=${TRAIN_LIST:-}
VAL_LIST=${VAL_LIST:-}
SAVE_PATH=${SAVE_PATH:-runs/flsea_latent_prior}

if [[ -z "${CKPT}" ]]; then
  echo "CKPT is empty. Example: export CKPT=/path/to/depth_anything_v2_vits.pth"
  exit 1
fi
if [[ -z "${TRAIN_LIST}" ]]; then
  echo "TRAIN_LIST is empty. Example: export TRAIN_LIST=/path/to/train.txt"
  exit 1
fi
if [[ -z "${VAL_LIST}" ]]; then
  echo "VAL_LIST is empty. Example: export VAL_LIST=/path/to/val.txt"
  exit 1
fi

ENCODER=${ENCODER:-vits}
IMG_SIZE=${IMG_SIZE:-518}
EPOCHS=${EPOCHS:-10}
BS=${BS:-4}
LR=${LR:-1e-4}
PRIOR_LR=${PRIOR_LR:-1e-4}
PRIOR_HEAD_LR=${PRIOR_HEAD_LR:-5e-5}
HEAD_LR=${HEAD_LR:-1e-5}
BACKBONE_LR=${BACKBONE_LR:-0}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-40.0}
NUM_WORKERS=${NUM_WORKERS:-4}
PRIOR_BASE_CH=${PRIOR_BASE_CH:-32}
PRIOR_CHANNELS=${PRIOR_CHANNELS:-32,64,128,256}
LATENT_DIM=${LATENT_DIM:-128}
PRIOR_FFT_SIZE=${PRIOR_FFT_SIZE:-64}
PRIOR_STAT_HIDDEN=${PRIOR_STAT_HIDDEN:-64}
DEG_MAP_SCALE=${DEG_MAP_SCALE:-0.2}
LOSS_MODE=${LOSS_MODE:-depthdive_relative}

EXTRA_ARGS=("$@")

mkdir -p "${SAVE_PATH}"

set -x
"${PYTHON_BIN}" train_latent_prior.py \
  --encoder "${ENCODER}" \
  --img-size "${IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --lr "${LR}" \
  --prior-lr "${PRIOR_LR}" \
  --prior-head-lr "${PRIOR_HEAD_LR}" \
  --head-lr "${HEAD_LR}" \
  --backbone-lr "${BACKBONE_LR}" \
  --pretrained-from "${CKPT}" \
  --save-path "${SAVE_PATH}" \
  --train-list "${TRAIN_LIST}" \
  --val-list "${VAL_LIST}" \
  --min-depth "${MIN_DEPTH}" \
  --max-depth "${MAX_DEPTH}" \
  --prior-base-ch "${PRIOR_BASE_CH}" \
  --prior-channels "${PRIOR_CHANNELS}" \
  --latent-dim "${LATENT_DIM}" \
  --prior-fft-size "${PRIOR_FFT_SIZE}" \
  --prior-stat-hidden "${PRIOR_STAT_HIDDEN}" \
  --deg-map-scale "${DEG_MAP_SCALE}" \
  --loss-mode "${LOSS_MODE}" \
  --freeze-backbone \
  "${EXTRA_ARGS[@]}"
