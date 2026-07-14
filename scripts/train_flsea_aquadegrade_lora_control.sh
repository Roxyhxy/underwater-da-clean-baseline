#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# B3: previous degradation-gated LoRA method, without consistency loss.
set -x
python train.py \
  --encoder vits --img-size 518 --epochs 5 --bs 4 --lr 5e-6 \
  --encoder-lr 5e-6 --decoder-lr 1e-5 --style-lr 1e-4 \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --save-path runs/control_aquadegrade_lora \
  --train-list /data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt \
  --val-list /data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt \
  --min-depth 0.1 --max-depth 40.0 \
  --lora-rank 8 --lora-alpha 16 --lora-target qkv --lora-last-n-blocks 12 \
  --lora-mode aquadegrade --style-dim 128 --style-hidden 64 --style-fft-size 64 \
  --use-decoder-lora --decoder-lora-rank 2 --decoder-lora-alpha 4 --decoder-lora-target tail \
  --loss-mode depthdive_relative --consistency-hardness-weight 0 --consistency-aug-prob 0 \
  --num-workers 4 \
  "$@"
