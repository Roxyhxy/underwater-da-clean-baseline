#!/usr/bin/env bash

set -euo pipefail

# =========================
# Edit this block on server
# =========================
TRAIN_LIST="/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt"
WORK_DIR="/data1/hxy/flsea_colmap_pilot"
NUM_FRAMES=100

# Set to false after the TIFF readability test succeeds once.
RUN_READ_TEST="true"
CAMERA_MODEL="SIMPLE_RADIAL"

if [[ ! -f "${TRAIN_LIST}" ]]; then
  echo "Training list not found: ${TRAIN_LIST}" >&2
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
READ_TEST_DB="${WORK_DIR}/read_test.db"

mkdir -p "${IMAGES_DIR}"

# Do not silently mix an earlier pilot set with this one.
existing_count="$(find "${IMAGES_DIR}" -mindepth 1 -maxdepth 1 | wc -l)"
if [[ "${existing_count}" -ne 0 ]]; then
  echo "Image directory is not empty: ${IMAGES_DIR}" >&2
  echo "Use a new WORK_DIR, or inspect and clear this pilot directory manually." >&2
  exit 1
fi

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
  ln -s "${rgb}" "${IMAGES_DIR}/${name}"
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

if [[ "${RUN_READ_TEST}" != "true" ]]; then
  exit 0
fi

if [[ -e "${READ_TEST_DB}" ]]; then
  echo "Read-test database already exists: ${READ_TEST_DB}" >&2
  echo "Set a new WORK_DIR or remove only this test database after inspecting it." >&2
  exit 1
fi

"${COLMAP_BIN}" feature_extractor \
  --database_path "${READ_TEST_DB}" \
  --image_path "${IMAGES_DIR}" \
  --ImageReader.camera_model "${CAMERA_MODEL}" \
  --ImageReader.single_camera 1 \
  --FeatureExtraction.use_gpu 0

echo "COLMAP read test completed successfully."
