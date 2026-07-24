#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-42}"

export TRAIN_LIST="${TRAIN_LIST:-/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt}"
export WAT3R_MANIFEST="${WAT3R_MANIFEST:-/data1/hxy/flsea_wat3r_train_teacher/canyons/flatiron/manifest_overlap.csv}"
export SAVE_PATH="${SAVE_PATH:-runs/wat3r_flatiron_pilot_seed${SEED}}"
export EPOCHS="${EPOCHS:-1}"
export BS="${BS:-2}"
export WAT3R_HOLE_WEIGHT="${WAT3R_HOLE_WEIGHT:-0.01}"
export WAT3R_MV_WEIGHT="${WAT3R_MV_WEIGHT:-0.02}"
export REQUIRE_OVERLAP="true"

if [[ ! -s "${WAT3R_MANIFEST}" ]]; then
  echo "Missing pilot overlap manifest: ${WAT3R_MANIFEST}"
  echo "Run scripts/prepare_flsea_wat3r_pilot.sh first."
  exit 2
fi

SEED="${SEED}" bash scripts/train_flsea_wat3r_distill.sh "$@"
