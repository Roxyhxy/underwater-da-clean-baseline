#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENCODER="${ENCODER:-vits}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
UWD_ROOT="${UWD_ROOT:-/data1/hxy/DATASET/uw-d}"
UWD_SPLIT_DIR="${UWD_SPLIT_DIR:-dataset/splits/uwd}"
CKPT="${CKPT:-../Depth-Anything-V2/checkpoints/depth_anything_v2_${ENCODER}.pth}"
SAVE="${SAVE:-exp_clean/uwd_dense_${ENCODER}_seed${SEED}}"

if [[ ! -f "${UWD_SPLIT_DIR}/train.jsonl" || ! -f "${UWD_SPLIT_DIR}/val.jsonl" ]]; then
  echo "UW-D manifests are missing; generating deterministic scene splits first."
  UWD_ROOT="${UWD_ROOT}" UWD_SPLIT_DIR="${UWD_SPLIT_DIR}" SEED="${SEED}" \
    bash scripts/build_uwd_splits.sh
fi

mkdir -p "${SAVE}"

"${PYTHON}" train_uwd_dense.py \
  --encoder "${ENCODER}" \
  --pretrained-from "${CKPT}" \
  --dataset-root "${UWD_ROOT}" \
  --train-list "${UWD_SPLIT_DIR}/train.jsonl" \
  --val-list "${UWD_SPLIT_DIR}/val.jsonl" \
  --save-path "${SAVE}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS:-20}" \
  --bs "${BS:-4}" \
  --lr "${LR:-5e-6}" \
  --img-size "${IMG_SIZE:-518}" \
  --min-depth "${MIN_DEPTH:-0.1}" \
  --max-depth "${MAX_DEPTH:-20.0}" \
  --gradient-weight "${GRADIENT_WEIGHT:-0.5}" \
  --warmup-steps "${WARMUP_STEPS:-100}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.2}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --num-workers "${NUM_WORKERS:-4}"
