#!/usr/bin/env bash

set -euo pipefail

# Continue from prepare_flsea_colmap_pilot.sh.
WORK_DIR="/data1/hxy/flsea_colmap_pilot"
PHYSICAL_GPU="${CUDA_VISIBLE_DEVICES:-1}"
COLMAP_GPU_INDEX=0

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/colmap" ]]; then
  COLMAP_BIN="${CONDA_PREFIX}/bin/colmap"
elif command -v colmap >/dev/null 2>&1; then
  COLMAP_BIN="$(command -v colmap)"
else
  echo "COLMAP was not found. Activate the environment containing COLMAP first." >&2
  exit 1
fi

IMAGES_DIR="${WORK_DIR}/images"
SPARSE_DIR="${WORK_DIR}/sparse_opencv"
DATABASE_PATH="${SPARSE_DIR}/database.db"
MODEL_DIR="${SPARSE_DIR}/model"
TEXT_DIR="${SPARSE_DIR}/model_txt"
MATCH_LOG="${SPARSE_DIR}/exhaustive_matching.log"
MAPPER_LOG="${SPARSE_DIR}/mapper.log"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "Prepared image directory not found: ${IMAGES_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATABASE_PATH}" ]]; then
  echo "Feature database not found: ${DATABASE_PATH}" >&2
  echo "Run scripts/prepare_flsea_colmap_pilot.sh first." >&2
  exit 1
fi

image_count="$(find "${IMAGES_DIR}" -maxdepth 1 -type l | wc -l)"
if [[ "${image_count}" -ne 100 ]]; then
  echo "Expected exactly 100 linked images, found ${image_count}." >&2
  exit 1
fi

if [[ -d "${MODEL_DIR}" ]] && find "${MODEL_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Sparse model output is not empty: ${MODEL_DIR}" >&2
  echo "Refusing to mix a new reconstruction with an existing result." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

# With only 100 frames, all-pairs matching is affordable and finds wider-baseline pairs.
echo "Matching all image pairs on physical GPU ${PHYSICAL_GPU}."
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${COLMAP_BIN}" exhaustive_matcher \
  --database_path "${DATABASE_PATH}" \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index "${COLMAP_GPU_INDEX}" \
  --FeatureMatching.guided_matching 1 \
  2>&1 | tee "${MATCH_LOG}"

echo "Running incremental mapping with calibrated intrinsics fixed."
"${COLMAP_BIN}" mapper \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --output_path "${MODEL_DIR}" \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  2>&1 | tee "${MAPPER_LOG}"

mapfile -t models < <(find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#models[@]}" -eq 0 ]]; then
  echo "COLMAP did not produce a sparse model. Inspect: ${MAPPER_LOG}" >&2
  exit 1
fi

echo "Produced ${#models[@]} sparse model(s)."
for model in "${models[@]}"; do
  echo "===== Model: ${model} ====="
  "${COLMAP_BIN}" model_analyzer --path "${model}"
done | tee "${SPARSE_DIR}/model_analysis.log"

for model in "${models[@]}"; do
  model_name="$(basename "${model}")"
  model_text_dir="${TEXT_DIR}/${model_name}"
  mkdir -p "${model_text_dir}"
  "${COLMAP_BIN}" model_converter \
    --input_path "${model}" \
    --output_path "${model_text_dir}" \
    --output_type TXT
done

echo "Sparse reconstruction completed."
echo "Binary model root: ${MODEL_DIR}"
echo "Text models: ${TEXT_DIR}"
echo "Analysis: ${SPARSE_DIR}/model_analysis.log"
