#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# L1: full latent prior plus the previously verified consistency-hardness weighting.
set -x
python train_latent_prior.py \
  --encoder vits --img-size 518 --epochs 5 --bs 4 --lr 1e-4 \
  --prior-lr 1e-4 --prior-head-lr 5e-5 --head-lr 1e-5 --backbone-lr 0 \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --save-path runs/loss_full_consistency \
  --train-list /data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt \
  --val-list /data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt \
  --min-depth 0.1 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.2 \
  --loss-mode depthdive_relative \
  --consistency-hardness-weight 0.08 --consistency-aug-prob 1.0 \
  --consistency-hardness-clamp-min 0.90 --consistency-hardness-clamp-max 1.10 \
  --consistency-blur-prob 0.30 --consistency-noise-prob 0.20 --consistency-noise-std 0.01 \
  --grad-clip 1.0 --num-workers 4 --freeze-backbone --freeze-base-head --eval-before-train \
  "$@"
