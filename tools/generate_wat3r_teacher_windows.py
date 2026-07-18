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
        description="Generate internally consistent Wat3R teacher windows for FLSea distillation."
    )
    parser.add_argument("--wat3r-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=Path)
    source.add_argument("--image-list", type=Path)
    parser.add_argument("--image-glob", default="*.tiff")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-size", default=518, type=int)
    parser.add_argument("--window-size", default=24, type=int)
    parser.add_argument("--overlap", default=12, type=int)
    parser.add_argument("--frames-chunk-size", default=4, type=int)
    parser.add_argument("--teacher-weights", action="store_true", help="Load ema_models from a training checkpoint")
    parser.add_argument("--skip-static-mask", action="store_true")
    parser.add_argument("--static-foreground", action="store_true")
    parser.add_argument(
        "--min-visible-views",
        default=3,
        type=int,
        help="Minimum geometrically consistent views used by the static mask",
    )
    parser.add_argument("--relative-depth-threshold", default=0.05, type=float)
    parser.add_argument("--boundary", default=4, type=int)
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


def load_image_paths(args):
    if args.image_dir is not None:
        paths = sorted(args.image_dir.glob(args.image_glob), key=lambda path: path.name)
    else:
        with args.image_list.open("r", encoding="utf-8") as handle:
            paths = [Path(line.split()[0]) for line in handle if line.strip()]
    paths = [path.expanduser().resolve() for path in paths]
    if not paths:
        raise RuntimeError("No input images found")
    return paths


def colorize_inverse_depth(depth):
    import matplotlib

    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.sum() < 8:
        return output
    disparity = np.zeros_like(depth)
    disparity[valid] = 1.0 / depth[valid]
    low, high = np.percentile(disparity[valid], [1, 99])
    normalized = np.clip((disparity - low) / max(high - low, 1e-6), 0, 1)
    rgb = matplotlib.colormaps["Spectral_r"](normalized)[..., :3]
    output = (rgb[..., ::-1] * 255).round().astype(np.uint8)
    output[~valid] = 0
    return output


def main():
    args = parse_args()
    if args.window_size < 3:
        raise ValueError("--window-size must be at least 3")
    if args.overlap < 0 or args.overlap >= args.window_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < window-size")
    if args.min_visible_views < 2:
        raise ValueError("--min-visible-views must be at least 2")

    sys.path.insert(0, str(args.wat3r_root))
    from wat3r.models.wat3r import Wat3R
    from wat3r.utils.load_fn import load_and_preprocess_images
    from wat3r.utils.pose_enc import pose_encoding_to_extri_intri
    from wat3r.utils.static_mask import build_static_masks, depth_foreground_mask

    image_paths = load_image_paths(args)
    device = torch.device(args.device)
    model = Wat3R(enable_track=False, enable_camera=True, enable_point=False, enable_depth=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.teacher_weights:
        if not isinstance(checkpoint, dict) or "ema_models" not in checkpoint:
            raise KeyError("--teacher-weights requires a checkpoint containing ema_models")
        state = checkpoint["ema_models"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    important_missing = [
        key for key in missing if key.startswith(("aggregator.", "camera_head.", "depth_head."))
    ]
    if important_missing:
        raise RuntimeError(f"Checkpoint is missing Wat3R inference keys: {important_missing[:10]}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected checkpoint keys")
    model = model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    starts = window_starts(len(image_paths), args.window_size, args.overlap)
    manifest_rows = []
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    for window_index, start in enumerate(starts):
        end = min(start + args.window_size, len(image_paths))
        paths = image_paths[start:end]
        print(f"Window {window_index + 1}/{len(starts)}: frames {start}:{end}")
        images = load_and_preprocess_images(
            [str(path) for path in paths], mode="max", target_size=args.target_size
        ).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            prediction = model(
                images,
                frames_chunk_size=args.frames_chunk_size,
                need_point=False,
                need_depth=True,
                need_camera=True,
            )
        depth = prediction["depth"].float()
        confidence = prediction["depth_conf"].float()
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            prediction["pose_enc"].float(), images.shape[-2:]
        )
        if args.skip_static_mask:
            static = torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0)
        else:
            candidates = depth_foreground_mask(depth[..., 0]) if args.static_foreground else None
            static = build_static_masks(
                depth=depth[..., 0],
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                candidate_masks=candidates,
                visibility_tolerance=max(0, len(paths) - args.min_visible_views),
                relative_depth_threshold=args.relative_depth_threshold,
                boundary=args.boundary,
            )
        print(
            "  static coverage: "
            f"{static.float().mean().item():.4f}, "
            f"depth confidence mean: {confidence.float().mean().item():.4f}"
        )

        depth = depth[0, ..., 0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()
        static = static[0].cpu().numpy()
        extrinsics = extrinsics[0].cpu().numpy()
        intrinsics = intrinsics[0].cpu().numpy()
        processed_height, processed_width = images.shape[-2:]

        window_name = f"window_{window_index:04d}_{start:06d}_{end:06d}"
        window_dir = args.output_dir / window_name
        depth_dir = window_dir / "depth"
        confidence_dir = window_dir / "confidence"
        static_dir = window_dir / "static"
        camera_dir = window_dir / "camera"
        visualization_dir = window_dir / "visualization"
        for directory in (depth_dir, confidence_dir, static_dir, camera_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if args.save_visualization:
            visualization_dir.mkdir(parents=True, exist_ok=True)

        for local_index, image_path in enumerate(paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            height, width = image.shape[:2]
            stem = image_path.stem
            frame_tag = f"{local_index:03d}_{stem}"
            frame_depth = cv2.resize(
                depth[local_index], (width, height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            frame_confidence = cv2.resize(
                confidence[local_index], (width, height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            frame_static = cv2.resize(
                static[local_index].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            frame_intrinsics = intrinsics[local_index].copy()
            frame_intrinsics[0] *= width / float(processed_width)
            frame_intrinsics[1] *= height / float(processed_height)

            depth_path = depth_dir / f"{frame_tag}_depth.tif"
            confidence_path = confidence_dir / f"{frame_tag}_confidence.tif"
            static_path = static_dir / f"{frame_tag}_static.png"
            camera_path = camera_dir / f"{frame_tag}_camera.npz"
            cv2.imwrite(str(depth_path), frame_depth)
            cv2.imwrite(str(confidence_path), frame_confidence)
            cv2.imwrite(str(static_path), frame_static * 255)
            np.savez(
                camera_path,
                intrinsics=frame_intrinsics.astype(np.float32),
                extrinsics=extrinsics[local_index].astype(np.float32),
            )
            if args.save_visualization:
                cv2.imwrite(
                    str(visualization_dir / f"{frame_tag}_inverse_depth.png"),
                    colorize_inverse_depth(frame_depth),
                )
            manifest_rows.append(
                {
                    "window": window_name,
                    "local_index": local_index,
                    "global_index": start + local_index,
                    "image": str(image_path),
                    "teacher_depth": str(depth_path.relative_to(args.output_dir)),
                    "teacher_confidence": str(confidence_path.relative_to(args.output_dir)),
                    "static_mask": str(static_path.relative_to(args.output_dir)),
                    "camera": str(camera_path.relative_to(args.output_dir)),
                    "height": height,
                    "width": width,
                }
            )

        del prediction, images, depth, confidence, static, extrinsics, intrinsics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "window",
            "local_index",
            "global_index",
            "image",
            "teacher_depth",
            "teacher_confidence",
            "static_mask",
            "camera",
            "height",
            "width",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved {len(manifest_rows)} window-frame labels to {manifest_path}")


if __name__ == "__main__":
    main()
