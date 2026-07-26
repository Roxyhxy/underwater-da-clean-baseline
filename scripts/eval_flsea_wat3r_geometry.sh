#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VARIANT="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
CKPT="${CKPT:-/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth}"
LOAD_FROM="${LOAD_FROM:-runs/wat3r_geometry_${VARIANT}_seed${SEED}/best_abs_rel.pth}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt}"
WAT3R_MANIFEST="${WAT3R_MANIFEST:-/data1/hxy/flsea_wat3r_train_teacher/manifest_all_overlap.csv}"
SAVE_DIR="${SAVE_DIR:-eval/wat3r_geometry_${VARIANT}_seed${SEED}}"
MAX_SAMPLES="${MAX_SAMPLES:-1200}"
CONFIDENCE_QUANTILE="${CONFIDENCE_QUANTILE:-0.6}"

if [[ ! -f "${LOAD_FROM}" ]]; then
  echo "Missing geometry checkpoint: ${LOAD_FROM}"
  exit 2
fi

set -x
python tools/eval_wat3r_geometry.py \
  --encoder vits \
  --load-from "${LOAD_FROM}" \
  --pretrained-from "${CKPT}" \
  --train-list "${TRAIN_LIST}" \
  --wat3r-manifest "${WAT3R_MANIFEST}" \
  --save-dir "${SAVE_DIR}" \
  --img-size 518 --max-depth 40 --max-samples "${MAX_SAMPLES}" --num-workers 2 \
  --confidence-quantile "${CONFIDENCE_QUANTILE}" \
  --relative-depth-threshold 0.05 --min-align-pixels 100 --require-overlap \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.1 \
  --disable-global-prior \
  --encoder-lora --encoder-lora-mode gated --encoder-lora-condition-source fft \
  --encoder-lora-rank 8 --encoder-lora-alpha 16 --encoder-lora-dropout 0 \
  --encoder-lora-last-n-blocks 12 \
  --geometry-adapter --geometry-adapter-hidden 32 --geometry-adapter-scale 0.1 \
  "${EXTRA_ARGS[@]}"
