#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VARIANT="${1:-local_only}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
SPLIT_NAME="${SPLIT_NAME:-val}"
SPLIT_LIST="${SPLIT_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt}"
LOAD_FROM="${LOAD_FROM:-runs/clean_${VARIANT}_${RUN_TAG}/best_abs_rel.pth}"
SAVE_DIR="${SAVE_DIR:-eval/clean_${VARIANT}_${RUN_TAG}_${SPLIT_NAME}}"

STRUCTURE_ARGS=()
case "${VARIANT}" in
  encoder_lora_plain)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-local-prior --disable-fft-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode plain)
    ;;
  encoder_lora_gated)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-local-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode gated)
    ;;
  encoder_lora_aqua)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-local-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode gated --encoder-lora-condition-source fft)
    ;;
  hybrid_lora_plain)
    STRUCTURE_ARGS+=(--disable-global-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode plain)
    ;;
  hybrid_lora_gated|hybrid_lora_gated_hole)
    STRUCTURE_ARGS+=(--disable-global-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode gated)
    ;;
  hybrid_lora_aqua|hybrid_lora_aqua_hole)
    STRUCTURE_ARGS+=(--disable-global-prior)
    STRUCTURE_ARGS+=(--encoder-lora --encoder-lora-mode gated --encoder-lora-condition-source fft)
    ;;
  local_only|local_only_no_anchor|local_only_hole)
    STRUCTURE_ARGS+=(--disable-global-prior --disable-fft-prior)
    ;;
  local_spectral|local_spectral_hole)
    STRUCTURE_ARGS+=(--disable-global-prior)
    ;;
  no_fft)
    STRUCTURE_ARGS+=(--disable-fft-prior)
    ;;
  full|full_no_anchor)
    ;;
  global_only)
    STRUCTURE_ARGS+=(--disable-local-prior)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    exit 2
    ;;
esac

mkdir -p "${SAVE_DIR}"
set -x
python eval_latent_prior.py \
  --encoder vits --load-from "${LOAD_FROM}" \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --val-list "${SPLIT_LIST}" --img-size 518 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.1 \
  --encoder-lora-rank 8 --encoder-lora-alpha 16 --encoder-lora-dropout 0 \
  --encoder-lora-last-n-blocks 12 \
  --save-dir "${SAVE_DIR}" \
  --save-raw-disparity --raw-output-dir "${SAVE_DIR}/raw_disparity" \
  --raw-colormap Spectral_r \
  "${STRUCTURE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
