#!/usr/bin/env python3

import argparse
import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PATH_COLUMNS = (
    "image",
    "teacher_depth",
    "teacher_confidence",
    "static_mask",
    "camera",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build cross-window agreement masks for repeated Wat3R frame predictions."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--relative-threshold", default=0.05, type=float)
    parser.add_argument("--min-align-pixels", default=1000, type=int)
    parser.add_argument("--min-copies", default=2, type=int)
    parser.add_argument("--max-align-pixels", default=100000, type=int)
    return parser.parse_args()


def normalized(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def resolve(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_single_channel(path, name):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(f"Failed to read {name}: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    return value


def fit_affine(source, target, mask, min_pixels, max_pixels):
    indices = np.flatnonzero(mask)
    if indices.size < min_pixels:
        return None
    if max_pixels > 0 and indices.size > max_pixels:
        step = max(1, indices.size // max_pixels)
        indices = indices[::step][:max_pixels]
    x = source.reshape(-1)[indices].astype(np.float64)
    y = target.reshape(-1)[indices].astype(np.float64)
    a00 = np.dot(x, x)
    a01 = x.sum()
    a11 = float(x.size)
    b0 = np.dot(x, y)
    b1 = y.sum()
    determinant = a00 * a11 - a01 * a01
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        return None
    scale = (a11 * b0 - a01 * b1) / determinant
    shift = (-a01 * b0 + a00 * b1) / determinant
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        return None
    return float(scale), float(shift)


def safe_mask_path(image_path, mask_root):
    digest = hashlib.sha1(normalized(image_path).encode("utf-8")).hexdigest()[:12]
    return mask_root / f"{Path(image_path).stem}_{digest}_overlap.png"


def build_consensus(records, manifest_root, args):
    depths = []
    valid_masks = []
    shape = None
    for row in records:
        depth = read_single_channel(
            resolve(row["teacher_depth"], manifest_root), "teacher depth"
        ).astype(np.float32)
        static = read_single_channel(
            resolve(row["static_mask"], manifest_root), "static mask"
        )
        if shape is None:
            shape = depth.shape
        elif depth.shape != shape or static.shape != shape:
            raise ValueError(f"Shape mismatch for repeated frame: {row['image']}")
        valid = np.isfinite(depth) & (depth > 0) & (static > 0)
        disparity = np.zeros_like(depth, dtype=np.float32)
        disparity[valid] = 1.0 / depth[valid]
        depths.append(disparity)
        valid_masks.append(valid)

    if len(depths) < args.min_copies:
        return np.zeros(shape, dtype=np.uint8), 0

    reference_index = int(np.argmax([mask.sum() for mask in valid_masks]))
    reference = depths[reference_index]
    reference_valid = valid_masks[reference_index]
    aligned = []
    aligned_valid = []
    for disparity, valid in zip(depths, valid_masks):
        common = reference_valid & valid
        solution = fit_affine(
            disparity,
            reference,
            common,
            args.min_align_pixels,
            args.max_align_pixels,
        )
        if solution is None:
            continue
        scale, shift = solution
        value = scale * disparity + shift
        aligned.append(value)
        aligned_valid.append(valid & np.isfinite(value) & (value > 0))

    if len(aligned) < args.min_copies:
        return np.zeros(shape, dtype=np.uint8), len(aligned)

    aligned = np.stack(aligned)
    aligned_valid = np.stack(aligned_valid)
    masked = np.where(aligned_valid, aligned, np.nan)
    consensus = np.nanmedian(masked, axis=0)
    denominator = np.maximum(np.abs(consensus), 1e-6)
    agreement = (
        aligned_valid
        & np.isfinite(consensus)[None]
        & (np.abs(aligned - consensus[None]) / denominator[None] <= args.relative_threshold)
    )
    reliable = agreement.sum(axis=0) >= args.min_copies
    return reliable.astype(np.uint8) * 255, len(aligned)


def main():
    args = parse_args()
    if args.relative_threshold <= 0:
        raise ValueError("--relative-threshold must be positive")
    if args.min_copies < 2:
        raise ValueError("--min-copies must be at least 2")
    if args.min_align_pixels <= 0:
        raise ValueError("--min-align-pixels must be positive")

    manifest = args.manifest.expanduser().resolve()
    manifest_root = manifest.parent
    output_manifest = args.output_manifest.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"image", "teacher_depth", "static_mask"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError("Manifest contains no teacher rows")

    groups = defaultdict(list)
    for row in rows:
        groups[normalized(row["image"])].append(row)

    mask_by_image = {}
    coverages = []
    usable_groups = 0
    for index, records in enumerate(groups.values(), start=1):
        mask, aligned_copies = build_consensus(records, manifest_root, args)
        mask_path = safe_mask_path(records[0]["image"], mask_root)
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write overlap mask: {mask_path}")
        mask_by_image[normalized(records[0]["image"])] = mask_path
        coverage = float((mask > 0).mean())
        coverages.append(coverage)
        if aligned_copies >= args.min_copies and coverage > 0:
            usable_groups += 1
        if index % 100 == 0 or index == len(groups):
            print(
                f"Processed {index}/{len(groups)} unique frames; "
                f"latest copies={len(records)} aligned={aligned_copies} coverage={coverage:.4f}"
            )

    output_fields = fieldnames[:]
    if "overlap_mask" not in output_fields:
        output_fields.append("overlap_mask")
    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            row["overlap_mask"] = str(mask_by_image[normalized(row["image"])])
            writer.writerow(row)
    temporary.replace(output_manifest)

    coverage_array = np.asarray(coverages, dtype=np.float64)
    duplicate_counts = np.asarray([len(records) for records in groups.values()])
    print(f"Input rows: {len(rows)}")
    print(f"Unique frames: {len(groups)}")
    print(f"Frames with >=2 predictions: {(duplicate_counts >= 2).sum()}")
    print(f"Usable overlap masks: {usable_groups}")
    print(
        "Overlap coverage p10/p50/p90: "
        + "/".join(f"{value:.4f}" for value in np.percentile(coverage_array, [10, 50, 90]))
    )
    print(f"Output manifest: {output_manifest}")


if __name__ == "__main__":
    main()
