#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CKPT="${CKPT:-/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth}"
LOAD_FROM="${LOAD_FROM:-runs/wat3r_distill_seed42/best_abs_rel.pth}"
VAL_LIST="${VAL_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt}"
SAVE_DIR="${SAVE_DIR:-eval/wat3r_distill_seed42}"

mkdir -p "${SAVE_DIR}"
set -x
python eval_latent_prior.py \
  --encoder vits --load-from "${LOAD_FROM}" --pretrained-from "${CKPT}" \
  --val-list "${VAL_LIST}" --img-size 518 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.1 \
  --disable-global-prior \
  --encoder-lora --encoder-lora-mode gated --encoder-lora-condition-source fft \
  --encoder-lora-rank 8 --encoder-lora-alpha 16 --encoder-lora-dropout 0 \
  --encoder-lora-last-n-blocks 12 \
  --save-dir "${SAVE_DIR}" \
  --save-raw-disparity --raw-output-dir "${SAVE_DIR}/raw_disparity" \
  --raw-colormap Spectral_r \
  "$@"
