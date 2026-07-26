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


def main():
    parser = argparse.ArgumentParser(description="Audit Wat3R teacher coverage before training.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--max-frames", default=300, type=int)
    parser.add_argument("--frame-stride", default=1, type=int)
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--confidence-quantile", default=0.6, type=float)
    parser.add_argument("--min-hole-pixels", default=256, type=int)
    args = parser.parse_args()
    if not 0.0 <= args.confidence_quantile <= 1.0:
        raise ValueError("--confidence-quantile must be in [0, 1]")
    if args.min_hole_pixels < 0:
        raise ValueError("--min-hole-pixels must be non-negative")

    train_images = set()
    train_stems = set()
    gt_by_image = {}
    gt_by_stem = defaultdict(list)
    with args.train_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                image = line.split()[0]
                image_key = normalized(image)
                train_images.add(image_key)
                train_stems.add(Path(image).stem)
                if len(line.split()) >= 2:
                    depth_path = line.split()[1]
                    gt_by_image[image_key] = depth_path
                    gt_by_stem[Path(image).stem].append(depth_path)

    manifest = args.manifest.expanduser().resolve()
    groups = defaultdict(list)
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["local_index"] = int(row["local_index"])
            groups[row["window"]].append(row)

    triplets = 0
    matched_centers = set()
    for records in groups.values():
        indices = {record["local_index"]: record for record in records}
        for index, record in indices.items():
            if index - args.frame_stride not in indices or index + args.frame_stride not in indices:
                continue
            image = record["image"]
            if normalized(image) in train_images or Path(image).stem in train_stems:
                triplets += 1
                matched_centers.add(normalized(image))

    all_rows = [record for records in groups.values() for record in records]
    if args.max_frames > 0 and len(all_rows) > args.max_frames:
        positions = np.linspace(0, len(all_rows) - 1, args.max_frames).round().astype(int)
        audit_rows = [all_rows[position] for position in positions]
    else:
        audit_rows = all_rows

    depth_coverages = []
    static_coverages = []
    overlap_coverages = []
    reliable_coverages = []
    confidence_values = []
    hole_coverages = []
    reliable_hole_ratios = []
    eligible_hole_frames = 0
    shape_errors = 0
    for row in audit_rows:
        depth_path = resolve(row["teacher_depth"], manifest.parent)
        confidence_path = resolve(row["teacher_confidence"], manifest.parent)
        static_path = resolve(row["static_mask"], manifest.parent)
        overlap_path = (
            resolve(row["overlap_mask"], manifest.parent)
            if row.get("overlap_mask")
            else None
        )
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        static = cv2.imread(str(static_path), cv2.IMREAD_UNCHANGED)
        overlap = (
            cv2.imread(str(overlap_path), cv2.IMREAD_UNCHANGED)
            if overlap_path is not None
            else None
        )
        if depth is None or confidence is None or static is None:
            raise FileNotFoundError(f"Unreadable teacher row: {row}")
        if args.require_overlap and overlap is None:
            raise FileNotFoundError(f"Unreadable overlap mask: {overlap_path}")
        if depth.shape[:2] != confidence.shape[:2] or depth.shape[:2] != static.shape[:2]:
            shape_errors += 1
            continue
        if overlap is not None and overlap.shape[:2] != depth.shape[:2]:
            shape_errors += 1
            continue
        valid = np.isfinite(depth) & (depth > 0)
        static_valid = valid & (static > 0)
        overlap_valid = (
            static_valid & (overlap > 0) if overlap is not None else static_valid
        )
        depth_coverages.append(valid.mean())
        static_coverages.append(static_valid.mean())
        overlap_coverages.append(overlap_valid.mean())
        finite_static_confidence = static_valid & np.isfinite(confidence)
        finite_confidence = overlap_valid & np.isfinite(confidence)
        if finite_static_confidence.any():
            cutoff = np.quantile(
                confidence[finite_static_confidence], args.confidence_quantile
            )
            reliable = finite_confidence & (confidence >= cutoff)
            reliable_coverages.append(reliable.mean())
            image_key = normalized(row["image"])
            gt_path = gt_by_image.get(image_key)
            if gt_path is None:
                candidates = gt_by_stem.get(Path(row["image"]).stem, [])
                gt_path = candidates[0] if len(candidates) == 1 else None
            if gt_path is not None:
                gt_depth = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                if gt_depth is not None and gt_depth.shape[:2] == depth.shape[:2]:
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth[..., 0]
                    hole = ~np.isfinite(gt_depth) | (gt_depth <= 0)
                    reliable_hole = reliable & hole
                    hole_coverages.append(reliable_hole.mean())
                    if hole.any():
                        reliable_hole_ratios.append(reliable_hole.sum() / hole.sum())
                        eligible_hole_frames += int(
                            reliable_hole.sum() >= args.min_hole_pixels
                        )
        confidence_values.append(confidence[finite_confidence])

    confidence_values = [values for values in confidence_values if values.size]
    confidence_values = np.concatenate(confidence_values) if confidence_values else np.array([])
    print(f"windows: {len(groups)}")
    print(f"manifest rows: {len(all_rows)}")
    print(f"audited rows: {len(audit_rows)}")
    print(f"matched unique train centers: {len(matched_centers)}")
    print(f"available training triplets (duplicates across windows included): {triplets}")
    print(f"teacher valid coverage mean: {np.mean(depth_coverages):.4f}")
    print(f"static coverage mean: {np.mean(static_coverages):.4f}")
    print(f"static + overlap coverage mean: {np.mean(overlap_coverages):.4f}")
    reliable_mean = np.mean(reliable_coverages) if reliable_coverages else 0.0
    print(f"estimated final reliable coverage mean: {reliable_mean:.4f}")
    if reliable_hole_ratios:
        print(f"reliable Wat3R pixels in holes / full image: {np.mean(hole_coverages):.4f}")
        print(f"reliable Wat3R pixels / all hole pixels: {np.mean(reliable_hole_ratios):.4f}")
        print(
            f"frames with >={args.min_hole_pixels} reliable hole pixels: "
            f"{eligible_hole_frames}/{len(reliable_hole_ratios)}"
        )
    print(f"shape errors: {shape_errors}")
    if confidence_values.size:
        values = np.percentile(confidence_values, [10, 50, 90])
        print(f"static confidence p10/p50/p90: {values[0]:.4f}/{values[1]:.4f}/{values[2]:.4f}")
    if not matched_centers or not triplets:
        raise RuntimeError("Teacher manifest does not provide usable triplets for this train split")


if __name__ == "__main__":
    main()
