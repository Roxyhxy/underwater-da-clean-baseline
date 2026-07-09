#!/usr/bin/env bash
set -euo pipefail

ENCODER=${ENCODER:-vits}
SEED=${SEED:-42}
PYTHON=${PYTHON:-python}
CKPT=${CKPT:-../Depth-Anything-V2/checkpoints/depth_anything_v2_${ENCODER}.pth}
TRAIN_LIST=${TRAIN_LIST:-dataset/splits/flsea/train_sub2.txt}
VAL_LIST=${VAL_LIST:-dataset/splits/flsea/test.txt}
SAVE=${SAVE:-exp_clean/flsea_aquadegrade_consistency_seed${SEED}}

mkdir -p "$SAVE"

$PYTHON train.py \
  --encoder "$ENCODER" \
  --pretrained-from "$CKPT" \
  --save-path "$SAVE" \
  --train-list "$TRAIN_LIST" \
  --val-list "$VAL_LIST" \
  --seed "$SEED" \
  --epochs 5 \
  --bs 4 \
  --lr 5e-6 \
  --encoder-lr 2e-6 \
  --decoder-lr 5e-6 \
  --style-lr 5e-6 \
  --img-size 518 \
  --min-depth 0.1 \
  --max-depth 40 \
  --lora-mode aquadegrade \
  --style-dim 128 \
  --style-hidden 64 \
  --style-fft-size 64 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --lora-target qkv \
  --lora-last-n-blocks 12 \
  --use-decoder-lora \
  --decoder-lora-rank 2 \
  --decoder-lora-alpha 4 \
  --decoder-lora-dropout 0.0 \
  --decoder-lora-target tail \
  --loss-mode depthdive_relative \
  --l1-weight 0.5 \
  --silog-weight 0.5 \
  --metric-weight 1.0 \
  --grad-weight 0.05 \
  --consistency-hardness-weight 0.08 \
  --consistency-hardness-clamp-min 0.90 \
  --consistency-hardness-clamp-max 1.10 \
  --warmup-steps 100 \
  --min-lr-ratio 0.2 \
  --weight-decay 0.0 \
  --num-workers 4

