#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

set -x
python eval_latent_prior.py \
  --encoder vits \
  --load-from runs/loss_full_consistency/best_abs_rel.pth \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --val-list /data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt \
  --img-size 518 --max-depth 40.0 \
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128 \
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.2 \
  --save-dir eval/loss_full_consistency_test \
  "$@"
