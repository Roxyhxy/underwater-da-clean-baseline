#!/usr/bin/env bash

set -euo pipefail

# =========================
# Edit this block on server
# =========================
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
WORK_DIR="/data1/hxy/flsea_colmap_pilot"
NUM_FRAMES=100

CALIB_FILE="/data1/hxy/DATASET/FLSeaVI/canyons/calibration/calibration_seaErra_imu_interp-kalibr-results-imucam.txt"
CAMERA_MODEL="OPENCV"
CAMERA_PARAMS="1175.3913431656817,1174.2805075232263,466.2595428966926,271.2116633091501,-0.13280386913948822,0.09799479194607102,-0.004731205238184176,0.0007132375646502103"

# Physical GPU 1 becomes logical GPU 0 inside this process.
PHYSICAL_GPU="${CUDA_VISIBLE_DEVICES:-1}"
COLMAP_GPU_INDEX=0
RUN_FEATURE_EXTRACTION="true"

if [[ ! -f "${TRAIN_LIST}" ]]; then
  echo "Training list not found: ${TRAIN_LIST}" >&2
  exit 1
fi

if [[ ! -f "${CALIB_FILE}" ]]; then
  echo "Calibration file not found: ${CALIB_FILE}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/colmap" ]]; then
  COLMAP_BIN="${CONDA_PREFIX}/bin/colmap"
elif command -v colmap >/dev/null 2>&1; then
  COLMAP_BIN="$(command -v colmap)"
else
  echo "COLMAP was not found. Activate the environment containing COLMAP first." >&2
  exit 1
fi

IMAGES_DIR="${WORK_DIR}/images"
PAIRS_FILE="${WORK_DIR}/pairs_${NUM_FRAMES}.txt"
RGB_FILE="${WORK_DIR}/rgb_${NUM_FRAMES}.txt"
SPARSE_DIR="${WORK_DIR}/sparse_opencv"
DATABASE_PATH="${SPARSE_DIR}/database.db"
FEATURE_LOG="${SPARSE_DIR}/feature_extraction.log"

mkdir -p "${IMAGES_DIR}" "${SPARSE_DIR}"

head -n "${NUM_FRAMES}" "${TRAIN_LIST}" > "${PAIRS_FILE}"
pair_count="$(wc -l < "${PAIRS_FILE}")"
if [[ "${pair_count}" -ne "${NUM_FRAMES}" ]]; then
  echo "Expected ${NUM_FRAMES} pairs, but the list only provided ${pair_count}." >&2
  exit 1
fi

: > "${RGB_FILE}"
declare -A seen_names=()

while read -r rgb depth extra; do
  if [[ -z "${rgb:-}" || -z "${depth:-}" || -n "${extra:-}" ]]; then
    echo "Each list line must contain exactly: RGB_PATH DEPTH_PATH" >&2
    exit 1
  fi
  if [[ ! -f "${rgb}" ]]; then
    echo "Missing RGB image: ${rgb}" >&2
    exit 1
  fi

  name="$(basename "${rgb}")"
  if [[ -n "${seen_names[${name}]:-}" ]]; then
    echo "Duplicate RGB basename: ${name}" >&2
    exit 1
  fi
  seen_names["${name}"]=1

  printf '%s\n' "${rgb}" >> "${RGB_FILE}"
  link_path="${IMAGES_DIR}/${name}"
  if [[ -L "${link_path}" ]]; then
    if [[ "$(readlink -f "${link_path}")" != "$(readlink -f "${rgb}")" ]]; then
      echo "Existing link points to another file: ${link_path}" >&2
      exit 1
    fi
  elif [[ -e "${link_path}" ]]; then
    echo "Expected a symbolic link but found another file: ${link_path}" >&2
    exit 1
  else
    ln -s "${rgb}" "${link_path}"
  fi
done < "${PAIRS_FILE}"

linked_count="$(find "${IMAGES_DIR}" -maxdepth 1 -type l | wc -l)"
if [[ "${linked_count}" -ne "${NUM_FRAMES}" ]]; then
  echo "Expected ${NUM_FRAMES} image links, but created ${linked_count}." >&2
  exit 1
fi

echo "Prepared ${linked_count} RGB frames."
echo "COLMAP binary: ${COLMAP_BIN}"
echo "Images: ${IMAGES_DIR}"
echo "Pairs with GT paths: ${PAIRS_FILE}"
echo "Calibration source: ${CALIB_FILE}"
echo "Camera model: ${CAMERA_MODEL}"
echo "Camera parameters: ${CAMERA_PARAMS}"

if [[ "${RUN_FEATURE_EXTRACTION}" != "true" ]]; then
  exit 0
fi

if [[ -e "${DATABASE_PATH}" ]]; then
  echo "Formal COLMAP database already exists: ${DATABASE_PATH}" >&2
  echo "Refusing to mix new settings into an existing database." >&2
  echo "Inspect it first, or choose a new SPARSE_DIR in this script." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${COLMAP_BIN}" feature_extractor \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --ImageReader.camera_model "${CAMERA_MODEL}" \
  --ImageReader.camera_params "${CAMERA_PARAMS}" \
  --ImageReader.single_camera 1 \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index "${COLMAP_GPU_INDEX}" \
  2>&1 | tee "${FEATURE_LOG}"

echo "Formal GPU feature extraction completed successfully."
echo "Database: ${DATABASE_PATH}"
echo "Log: ${FEATURE_LOG}"
