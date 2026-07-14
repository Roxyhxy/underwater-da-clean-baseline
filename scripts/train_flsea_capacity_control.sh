#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Controls whether gains come from the proposed prior or merely extra capacity.
# Usage: bash scripts/train_flsea_capacity_control.sh conv_adapter|decoder_ft|full_decoder
VARIANT="${1:-conv_adapter}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="python"
CKPT="/data1/hxy/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
VAL_LIST="/data1/hxy/Depth-Anything-V2/DA_0/dataset/splits/flsea/val.txt"

COMMON_ARGS=(
  --encoder vits --img-size 518 --epochs 5 --bs 4 --lr 1e-4
  --prior-lr 1e-4 --prior-head-lr 5e-5 --adapter-lr 1e-4
  --head-lr 1e-5 --backbone-lr 0
  --pretrained-from "${CKPT}" --train-list "${TRAIN_LIST}" --val-list "${VAL_LIST}"
  --min-depth 0.1 --max-depth 40.0
  --prior-base-ch 32 --prior-channels 32,64,128,256 --latent-dim 128
  --prior-fft-size 64 --prior-stat-hidden 64 --deg-map-scale 0.2
  --loss-mode depthdive_relative --consistency-hardness-weight 0 --consistency-aug-prob 0
  --grad-clip 1.0 --num-workers 4 --freeze-backbone --eval-before-train
)

VARIANT_ARGS=()
case "${VARIANT}" in
  conv_adapter)
    # About 2.49M trainable parameters; backbone and original decoder stay frozen.
    VARIANT_ARGS+=(--freeze-base-head --freeze-latent-prior --disable-global-prior --disable-local-prior --disable-fft-prior --plain-adapter --adapter-hidden 256)
    ;;
  decoder_ft)
    # Only the original DPT decoder is trainable; no proposed prior branch is active.
    VARIANT_ARGS+=(--freeze-latent-prior --disable-global-prior --disable-local-prior --disable-fft-prior)
    ;;
  full_decoder)
    # Capacity ceiling: full latent prior plus the original DPT decoder.
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    echo "Choose one of: conv_adapter, decoder_ft, full_decoder"
    exit 2
    ;;
esac

SAVE_PATH="runs/control_${VARIANT}"
mkdir -p "${SAVE_PATH}"

set -x
"${PYTHON_BIN}" train_latent_prior.py \
  --save-path "${SAVE_PATH}" \
  "${COMMON_ARGS[@]}" \
  "${VARIANT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
