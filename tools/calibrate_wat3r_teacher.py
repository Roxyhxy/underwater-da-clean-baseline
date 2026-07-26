#!/usr/bin/env python3

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def normalized(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def resolve(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_map(path, name):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(f"Unreadable {name}: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    return value.astype(np.float32)


def fit_affine(prediction, target, mask, min_pixels):
    x = prediction[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    if x.size < min_pixels or np.std(x) < 1e-8:
        return None
    design = np.column_stack((x, np.ones_like(x)))
    solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(solution[0]), float(solution[1])


def parse_quantiles(value):
    quantiles = [float(item) for item in value.split(",") if item.strip()]
    if not quantiles or any(item < 0 or item > 1 for item in quantiles):
        raise ValueError("--quantiles must contain values in [0, 1]")
    return quantiles


def metric_accumulator():
    return {
        "pixels": 0,
        "possible": 0,
        "abs_rel_sum": 0.0,
        "sq_error_sum": 0.0,
        "delta1_sum": 0.0,
        "frames": 0,
    }


def update_metrics(accumulator, prediction, target, mask, possible):
    accumulator["possible"] += int(possible)
    pred = prediction[mask]
    gt = target[mask]
    if pred.size == 0:
        return
    ratio = np.maximum(pred / np.maximum(gt, 1e-6), gt / np.maximum(pred, 1e-6))
    accumulator["pixels"] += int(pred.size)
    accumulator["abs_rel_sum"] += float(np.sum(np.abs(pred - gt) / np.maximum(gt, 1e-6)))
    accumulator["sq_error_sum"] += float(np.sum((pred - gt) ** 2))
    accumulator["delta1_sum"] += float(np.sum(ratio < 1.25))
    accumulator["frames"] += 1


def finalize(name, quantile, values):
    pixels = max(values["pixels"], 1)
    return {
        "mask": name,
        "confidence_quantile": quantile,
        "frames": values["frames"],
        "pixels": values["pixels"],
        "coverage": values["pixels"] / max(values["possible"], 1),
        "abs_rel": values["abs_rel_sum"] / pixels,
        "rmse": np.sqrt(values["sq_error_sum"] / pixels),
        "d1": values["delta1_sum"] / pixels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Wat3R confidence and overlap masks against valid FLSea GT."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--quantiles", default="0,0.2,0.4,0.6,0.8")
    parser.add_argument("--max-rows", default=1200, type=int)
    parser.add_argument("--min-align-pixels", default=1000, type=int)
    parser.add_argument("--max-depth", default=40.0, type=float)
    args = parser.parse_args()

    gt_by_image = {}
    gt_by_stem = defaultdict(list)
    with args.train_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            gt_by_image[normalized(parts[0])] = parts[1]
            gt_by_stem[Path(parts[0]).stem].append(parts[1])

    manifest = args.manifest.expanduser().resolve()
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if args.max_rows > 0 and len(rows) > args.max_rows:
        indices = np.linspace(0, len(rows) - 1, args.max_rows).round().astype(int)
        rows = [rows[index] for index in indices]

    quantiles = parse_quantiles(args.quantiles)
    mask_names = ("all", "static", "static_overlap")
    accumulators = {
        (name, quantile): metric_accumulator()
        for name in mask_names
        for quantile in quantiles
    }
    aligned_frames = 0
    skipped_frames = 0

    for row in rows:
        image_path = normalized(row["image"])
        gt_path = gt_by_image.get(image_path)
        if gt_path is None:
            candidates = gt_by_stem.get(Path(row["image"]).stem, [])
            gt_path = candidates[0] if len(candidates) == 1 else None
        if gt_path is None:
            skipped_frames += 1
            continue

        teacher_depth = read_map(
            resolve(row["teacher_depth"], manifest.parent), "teacher depth"
        )
        confidence = read_map(
            resolve(row["teacher_confidence"], manifest.parent), "teacher confidence"
        )
        static = read_map(resolve(row["static_mask"], manifest.parent), "static mask") > 0
        overlap = (
            read_map(resolve(row["overlap_mask"], manifest.parent), "overlap mask") > 0
            if row.get("overlap_mask")
            else np.ones_like(static)
        )
        gt_depth = read_map(Path(gt_path), "GT depth")
        if not (
            teacher_depth.shape
            == confidence.shape
            == static.shape
            == overlap.shape
            == gt_depth.shape
        ):
            skipped_frames += 1
            continue

        teacher_valid = np.isfinite(teacher_depth) & (teacher_depth > 0)
        gt_valid = (
            np.isfinite(gt_depth)
            & (gt_depth > 0)
            & (gt_depth <= float(args.max_depth))
        )
        common = teacher_valid & gt_valid
        teacher_disparity = np.zeros_like(teacher_depth, dtype=np.float32)
        gt_disparity = np.zeros_like(gt_depth, dtype=np.float32)
        teacher_disparity[teacher_valid] = 1.0 / teacher_depth[teacher_valid]
        gt_disparity[gt_valid] = 1.0 / gt_depth[gt_valid]
        solution = fit_affine(
            teacher_disparity,
            gt_disparity,
            common,
            args.min_align_pixels,
        )
        if solution is None:
            skipped_frames += 1
            continue
        scale, shift = solution
        aligned_disparity = scale * teacher_disparity + shift
        aligned_valid = common & np.isfinite(aligned_disparity) & (aligned_disparity > 1e-6)
        aligned_depth = np.zeros_like(gt_depth, dtype=np.float32)
        aligned_depth[aligned_valid] = 1.0 / aligned_disparity[aligned_valid]
        aligned_frames += 1

        base_masks = {
            "all": aligned_valid,
            "static": aligned_valid & static,
            "static_overlap": aligned_valid & static & overlap,
        }
        confidence_reference = teacher_valid & static & np.isfinite(confidence)
        for quantile in quantiles:
            cutoff = (
                np.quantile(confidence[confidence_reference], quantile)
                if confidence_reference.any()
                else np.inf
            )
            confidence_mask = np.isfinite(confidence) & (confidence >= cutoff)
            for name, base_mask in base_masks.items():
                selected = base_mask & confidence_mask
                update_metrics(
                    accumulators[(name, quantile)],
                    aligned_depth,
                    gt_depth,
                    selected,
                    possible=int(gt_valid.sum()),
                )

    results = [
        finalize(name, quantile, accumulators[(name, quantile)])
        for name in mask_names
        for quantile in quantiles
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"aligned frames: {aligned_frames}")
    print(f"skipped frames: {skipped_frames}")
    print("mask              q      coverage    abs_rel     rmse       d1")
    for result in results:
        print(
            f"{result['mask']:<17} {result['confidence_quantile']:<6.2f} "
            f"{result['coverage']:<11.4f} {result['abs_rel']:<11.4f} "
            f"{result['rmse']:<10.4f} {result['d1']:.4f}"
        )
    print(f"Saved calibration table: {args.output_csv}")


if __name__ == "__main__":
    main()
