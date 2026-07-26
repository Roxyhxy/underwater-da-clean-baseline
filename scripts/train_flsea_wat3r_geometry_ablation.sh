#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Four matched experiments:
# control: adapter + valid GT only
# hole:    control + reliable hole teacher
# mv:      control + temporal reprojection
# full:    control + both privileged geometry losses
VARIANT="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
BS="${BS:-2}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"

CKPT="${CKPT:-/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth}"
INIT_FROM="${INIT_FROM:-runs/clean_hybrid_lora_aqua_seed42/best_abs_rel.pth}"
TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt}"
VAL_LIST="${VAL_LIST:-/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt}"
WAT3R_MANIFEST="${WAT3R_MANIFEST:-/data1/hxy/flsea_wat3r_train_teacher/manifest_all_overlap.csv}"
CONFIDENCE_QUANTILE="${CONFIDENCE_QUANTILE:-0.6}"
MIN_HOLE_PIXELS="${MIN_HOLE_PIXELS:-256}"

WAT3R_HOLE_WEIGHT=0
WAT3R_MV_WEIGHT=0
case "${VARIANT}" in
  control)
    ;;
  hole)
    WAT3R_HOLE_WEIGHT=0.1
    ;;
  mv)
    WAT3R_MV_WEIGHT=0.05
    ;;
  full)
    WAT3R_HOLE_WEIGHT=0.1
    WAT3R_MV_WEIGHT=0.05
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose one of: control, hole, mv, full"
    exit 2
    ;;
esac

SAVE_PATH="${SAVE_PATH:-runs/wat3r_geometry_${VARIANT}_${RUN_TAG}}"
if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing Stage-1 checkpoint: ${INIT_FROM}"
  exit 2
fi
if [[ ! -f "${WAT3R_MANIFEST}" ]]; then
  echo "Missing Wat3R manifest: ${WAT3R_MANIFEST}"
  exit 2
fi
if [[ -e "${SAVE_PATH}" ]]; then
  echo "Refusing to overwrite existing run: ${SAVE_PATH}"
  exit 3
fi
mkdir -p "${SAVE_PATH}"

set -x
python train_latent_prior.py \
  --encoder vits --img-size 518 --epochs "${EPOCHS}" --bs "${BS}" --seed "${SEED}" \
  --lr 0 --prior-lr 0 --prior-head-lr 0 --encoder-lora-lr 0 \
  --geometry-adapter-lr 1e-4 --head-lr 0 --backbone-lr 0 \
  --pretrained-from "${CKPT}" --init-from "${INIT_FROM}" --save-path "${SAVE_PATH}" \
  --train-list "${TRAIN_LIST}" --val-list "${VAL_LIST}" \
  --wat3r-manifest "${WAT3R_MANIFEST}" --wat3r-frame-stride 1 \
  --wat3r-hole-weight "${WAT3R_HOLE_WEIGHT}" --wat3r-hole-grad-weight 0.25 \
  --wat3r-mv-weight "${WAT3R_MV_WEIGHT}" \
  --wat3r-confidence-quantile "${CONFIDENCE_QUANTILE}" \
  --wat3r-relative-depth-threshold 0.05 --wat3r-min-align-pixels 100 \
  --wat3r-min-hole-pixels "${MIN_HOLE_PIXELS}" \
  --wat3r-hole-boundary-radius 5 --wat3r-hole-boundary-weight 2 \
  --wat3r-far-quantile 0.7 --wat3r-far-weight 2 \
  --wat3r-require-overlap \
  --min-depth 0.1 --max-depth 40 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.1 \
  --disable-global-prior \
  --encoder-lora --encoder-lora-mode gated --encoder-lora-condition-source fft \
  --encoder-lora-rank 8 --encoder-lora-alpha 16 --encoder-lora-dropout 0 \
  --encoder-lora-last-n-blocks 12 \
  --geometry-adapter --geometry-adapter-only \
  --geometry-adapter-hidden 32 --geometry-adapter-scale 0.1 \
  --loss-mode depthdive_relative \
  --l1-weight 0.5 --silog-weight 0.5 --metric-weight 1 --grad-weight 0.05 \
  --gauge-anchor-weight 0 --hole-geometry-weight 0 \
  --consistency-hardness-weight 0 --consistency-aug-prob 0 \
  --warmup-steps 100 --min-lr-ratio 0.2 --weight-decay 0 \
  --grad-clip 1 --num-workers 4 --amp \
  --freeze-backbone --freeze-base-head --freeze-latent-prior --eval-before-train \
  "${EXTRA_ARGS[@]}"
