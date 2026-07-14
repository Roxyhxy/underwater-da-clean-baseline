#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

set -x
python eval_lora_control.py \
  --encoder vits --input-size 518 --max-depth 40.0 \
  --pretrained-from /data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --load-from runs/control_aquadegrade_lora/best_abs_rel.pth \
  --val-list /data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/test.txt \
  --save-dir eval/control_aquadegrade_lora_test \
  --lora-rank 8 --lora-alpha 16 --lora-target qkv --lora-last-n-blocks 12 \
  --style-dim 128 --style-hidden 64 --style-fft-size 64 \
  --decoder-lora-rank 2 --decoder-lora-alpha 4 --decoder-lora-target tail \
  "$@"
