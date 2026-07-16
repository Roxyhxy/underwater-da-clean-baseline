#!/usr/bin/env bash

set -euo pipefail

# Continue from run_flsea_colmap_sparse.sh after model/0 passes analysis.
WORK_DIR="/data1/hxy/flsea_colmap_pilot"
MAX_IMAGE_SIZE=1600
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/colmap" ]]; then
  COLMAP_BIN="${CONDA_PREFIX}/bin/colmap"
elif command -v colmap >/dev/null 2>&1; then
  COLMAP_BIN="$(command -v colmap)"
else
  echo "COLMAP was not found. Activate the environment containing COLMAP first." >&2
  exit 1
fi

IMAGES_DIR="${WORK_DIR}/images"
MODEL_DIR="${WORK_DIR}/sparse_opencv/model/0"
NORMALIZED_TEXT_DIR="${WORK_DIR}/sparse_opencv/model_normalized_txt"
NORMALIZED_MODEL_DIR="${WORK_DIR}/sparse_opencv/model_normalized"
DENSE_DIR="${WORK_DIR}/dense"
UNDISTORT_LOG="${WORK_DIR}/image_undistorter.log"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "Prepared image directory not found: ${IMAGES_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/cameras.bin" || ! -f "${MODEL_DIR}/images.bin" ]]; then
  echo "Valid COLMAP sparse model not found: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ -d "${DENSE_DIR}" ]] && find "${DENSE_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Dense workspace is not empty: ${DENSE_DIR}" >&2
  echo "Refusing to overwrite an existing undistortion or MVS result." >&2
  exit 1
fi
if [[ -d "${NORMALIZED_TEXT_DIR}" ]] && find "${NORMALIZED_TEXT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Normalized text model output is not empty: ${NORMALIZED_TEXT_DIR}" >&2
  exit 1
fi
if [[ -d "${NORMALIZED_MODEL_DIR}" ]] && find "${NORMALIZED_MODEL_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Normalized binary model output is not empty: ${NORMALIZED_MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${NORMALIZED_TEXT_DIR}" "${NORMALIZED_MODEL_DIR}" "${DENSE_DIR}"

# Symlink targets were stored as ../../DATASET/... names. MVS tools need stable basenames.
"${COLMAP_BIN}" model_converter \
  --input_path "${MODEL_DIR}" \
  --output_path "${NORMALIZED_TEXT_DIR}" \
  --output_type TXT

"${PYTHON_BIN}" - "${NORMALIZED_TEXT_DIR}/images.txt" <<'PY'
from pathlib import Path, PurePosixPath
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
output = []
expect_image = True
seen = set()

for line in lines:
    if line.startswith("#"):
        output.append(line)
        continue

    if expect_image:
        parts = line.strip().split()
        if len(parts) < 10:
            raise RuntimeError(f"Malformed COLMAP image row: {line!r}")
        name = PurePosixPath(parts[9].replace("\\", "/")).name
        if name in seen:
            raise RuntimeError(f"Duplicate normalized image name: {name}")
        seen.add(name)
        parts[9] = name
        output.append(" ".join(parts) + "\n")
    else:
        output.append(line)
    expect_image = not expect_image

if not expect_image:
    raise RuntimeError("images.txt ended before the final POINTS2D row")
if len(seen) != 100:
    raise RuntimeError(f"Expected 100 registered images, found {len(seen)}")

path.write_text("".join(output), encoding="utf-8")
print(f"Normalized {len(seen)} COLMAP image names to basenames.")
PY

"${COLMAP_BIN}" model_converter \
  --input_path "${NORMALIZED_TEXT_DIR}" \
  --output_path "${NORMALIZED_MODEL_DIR}" \
  --output_type BIN

"${COLMAP_BIN}" image_undistorter \
  --image_path "${IMAGES_DIR}" \
  --input_path "${NORMALIZED_MODEL_DIR}" \
  --output_path "${DENSE_DIR}" \
  --output_type COLMAP \
  --max_image_size "${MAX_IMAGE_SIZE}" \
  2>&1 | tee "${UNDISTORT_LOG}"

undistorted_count="$(find "${DENSE_DIR}/images" -maxdepth 1 -type f | wc -l)"
if [[ "${undistorted_count}" -ne 100 ]]; then
  echo "Expected 100 undistorted images, found ${undistorted_count}." >&2
  exit 1
fi

if [[ ! -f "${DENSE_DIR}/sparse/cameras.bin" || ! -f "${DENSE_DIR}/sparse/images.bin" ]]; then
  echo "Undistorted sparse camera model is incomplete: ${DENSE_DIR}/sparse" >&2
  exit 1
fi

echo "COLMAP dense workspace preparation completed."
echo "Undistorted images: ${undistorted_count}"
echo "Dense workspace: ${DENSE_DIR}"
echo "Log: ${UNDISTORT_LOG}"
