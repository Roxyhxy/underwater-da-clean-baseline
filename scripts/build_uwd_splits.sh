#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

UWD_ROOT="${UWD_ROOT:-/data1/hxy/DATASET/uw-d}"
UWD_SPLIT_DIR="${UWD_SPLIT_DIR:-dataset/splits/uwd}"

python tools/build_uwd_manifests.py \
  --root "${UWD_ROOT}" \
  --output-dir "${UWD_SPLIT_DIR}" \
  --seed "${SEED:-42}" \
  --train-fraction 0.8 \
  --val-fraction 0.1
