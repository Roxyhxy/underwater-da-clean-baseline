#!/usr/bin/env python3

import argparse
import csv
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate frame-wise dense DA3 depth using COLMAP camera poses."
    )
    parser.add_argument("--da3-root", required=True, type=Path)
    parser.add_argument("--colmap-dir", required=True, type=Path)
    parser.add_argument("--sparse-subdir", default="")
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", default=504, type=int)
    parser.add_argument("--window-size", default=24, type=int)
    parser.add_argument("--overlap", default=12, type=int)
    parser.add_argument(
        "--ref-view-strategy",
        default="middle",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
    )
    parser.add_argument(
        "--no-overlap-scale-correction",
        action="store_true",
        help="Disable robust scale correction between adjacent windows.",
    )
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    return parser.parse_args()


def window_starts(num_frames, window_size, overlap):
    if num_frames <= window_size:
        return [0]
    stride = window_size - overlap
    starts = list(range(0, num_frames - window_size + 1, stride))
    final_start = num_frames - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def robust_overlap_scale(depth, indices, depth_sum, weight_sum, conf):
    ratios = []
    for local_idx, global_idx in enumerate(indices):
        if not np.any(weight_sum[global_idx] > 0):
            continue
        previous = depth_sum[global_idx] / np.maximum(weight_sum[global_idx], 1e-6)
        current = depth[local_idx]
        confidence = conf[local_idx]
        finite_confidence = confidence[np.isfinite(confidence)]
        if finite_confidence.size == 0:
            continue
        conf_cutoff = np.percentile(finite_confidence, 60)
        valid = (
            (weight_sum[global_idx] > 0)
            & np.isfinite(previous)
            & np.isfinite(current)
            & (previous > 0)
            & (current > 0)
            & (confidence >= conf_cutoff)
        )
        if np.any(valid):
            ratios.append((previous[valid] / current[valid]).reshape(-1))
    if not ratios:
        return 1.0
    ratio = np.concatenate(ratios)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return float(np.median(ratio)) if ratio.size else 1.0


def colorize_depth(depth):
    import matplotlib

    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.sum() < 8:
        return result
    low, high = np.percentile(depth[valid], [1, 99])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    rgb = matplotlib.colormaps["Spectral_r"](normalized)[..., :3]
    result = (rgb[..., ::-1] * 255).round().astype(np.uint8)
    result[~valid] = 0
    return result


def main():
    args = parse_args()
    if args.window_size < 3:
        raise ValueError("--window-size must be at least 3")
    if args.overlap < 0 or args.overlap >= args.window_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < window-size")

    sys.path.insert(0, str(args.da3_root / "src"))
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.services.input_handlers import ColmapHandler

    image_paths, extrinsics, intrinsics = ColmapHandler.process(
        str(args.colmap_dir), args.sparse_subdir
    )
    records = sorted(
        zip(image_paths, extrinsics, intrinsics), key=lambda item: Path(item[0]).name
    )
    image_paths = [item[0] for item in records]
    extrinsics = np.stack([item[1] for item in records])
    intrinsics = np.stack([item[2] for item in records])

    print(f"Loading {args.model} on {args.device}")
    model = DepthAnything3.from_pretrained(args.model).to(args.device).eval()
    starts = window_starts(len(image_paths), args.window_size, args.overlap)

    depth_sum = None
    depth_square_sum = None
    confidence_sum = None
    weight_sum = None
    support_count = None
    scale_log = []

    for window_idx, start in enumerate(starts):
        end = min(start + args.window_size, len(image_paths))
        indices = list(range(start, end))
        print(f"Window {window_idx + 1}/{len(starts)}: frames {start}:{end}")
        prediction = model.inference(
            [image_paths[idx] for idx in indices],
            extrinsics=extrinsics[indices],
            intrinsics=intrinsics[indices],
            align_to_input_ext_scale=True,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy=args.ref_view_strategy,
        )
        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = (
            np.asarray(prediction.conf, dtype=np.float32)
            if prediction.conf is not None
            else np.ones_like(depth, dtype=np.float32)
        )
        if depth_sum is None:
            shape = (len(image_paths), *depth.shape[-2:])
            depth_sum = np.zeros(shape, dtype=np.float32)
            depth_square_sum = np.zeros(shape, dtype=np.float32)
            confidence_sum = np.zeros(shape, dtype=np.float32)
            weight_sum = np.zeros(shape, dtype=np.float32)
            support_count = np.zeros(shape, dtype=np.uint8)
        elif depth.shape[-2:] != depth_sum.shape[-2:]:
            raise RuntimeError("DA3 returned inconsistent spatial shapes across windows")

        scale = 1.0
        if window_idx and not args.no_overlap_scale_correction:
            scale = robust_overlap_scale(depth, indices, depth_sum, weight_sum, conf)
            depth *= scale
        scale_log.append((window_idx, start, end, scale))
        print(f"  overlap depth scale: {scale:.6f}")

        weights = np.maximum(conf, 1e-6)
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(weights)
        for local_idx, global_idx in enumerate(indices):
            local_weight = np.where(valid[local_idx], weights[local_idx], 0.0)
            depth_sum[global_idx] += np.where(
                valid[local_idx], depth[local_idx] * local_weight, 0.0
            )
            depth_square_sum[global_idx] += np.where(
                valid[local_idx], depth[local_idx] ** 2 * local_weight, 0.0
            )
            confidence_sum[global_idx] += np.where(
                valid[local_idx], conf[local_idx] * local_weight, 0.0
            )
            weight_sum[global_idx] += local_weight
            support_count[global_idx] += valid[local_idx].astype(np.uint8)

        del prediction, depth, conf, weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fused_depth = depth_sum / np.maximum(weight_sum, 1e-6)
    fused_conf = confidence_sum / np.maximum(weight_sum, 1e-6)
    depth_variance = np.maximum(
        depth_square_sum / np.maximum(weight_sum, 1e-6) - fused_depth**2, 0.0
    )
    relative_disagreement = np.sqrt(depth_variance) / np.maximum(fused_depth, 1e-6)
    fused_depth[weight_sum == 0] = 0
    fused_conf[weight_sum == 0] = 0
    relative_disagreement[weight_sum == 0] = 0

    depth_dir = args.output_dir / "depth_tiff"
    conf_dir = args.output_dir / "confidence_tiff"
    disagreement_dir = args.output_dir / "relative_disagreement_tiff"
    support_dir = args.output_dir / "support_count"
    npy_dir = args.output_dir / "npy"
    vis_dir = args.output_dir / "visualization"
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    disagreement_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualization:
        vis_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        depth = cv2.resize(fused_depth[idx], (width, height), interpolation=cv2.INTER_LINEAR)
        conf = cv2.resize(fused_conf[idx], (width, height), interpolation=cv2.INTER_LINEAR)
        disagreement = cv2.resize(
            relative_disagreement[idx], (width, height), interpolation=cv2.INTER_LINEAR
        )
        support = cv2.resize(
            support_count[idx], (width, height), interpolation=cv2.INTER_NEAREST
        )
        stem = Path(image_path).stem
        depth_path = depth_dir / f"{stem}_da3_depth.tif"
        conf_path = conf_dir / f"{stem}_da3_conf.tif"
        disagreement_path = disagreement_dir / f"{stem}_da3_rel_disagreement.tif"
        support_path = support_dir / f"{stem}_da3_support.png"
        cv2.imwrite(str(depth_path), depth.astype(np.float32))
        cv2.imwrite(str(conf_path), conf.astype(np.float32))
        cv2.imwrite(str(disagreement_path), disagreement.astype(np.float32))
        cv2.imwrite(str(support_path), support.astype(np.uint8))
        if args.save_npy:
            np.save(npy_dir / f"{stem}_da3_depth.npy", depth.astype(np.float32))
            np.save(npy_dir / f"{stem}_da3_conf.npy", conf.astype(np.float32))
        if args.save_visualization:
            cv2.imwrite(str(vis_dir / f"{stem}_da3_depth.png"), colorize_depth(depth))
        manifest_rows.append(
            (
                idx,
                image_path,
                depth_path,
                conf_path,
                disagreement_path,
                support_path,
                height,
                width,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "image",
                "depth",
                "confidence",
                "relative_disagreement",
                "support_count",
                "height",
                "width",
            ]
        )
        writer.writerows(manifest_rows)
    with (args.output_dir / "window_scales.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["window", "start", "end", "depth_scale"])
        writer.writerows(scale_log)
    print(f"Saved {len(image_paths)} dense depth maps to {args.output_dir}")


if __name__ == "__main__":
    main()
