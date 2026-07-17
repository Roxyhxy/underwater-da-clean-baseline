#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": (48, 96, 192, 384)},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": (96, 192, 384, 768)},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": (256, 512, 1024, 1024)},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": (1536, 1536, 1536, 1536)},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate matched DA2/DA3 geometry visualizations for a COLMAP pilot."
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--da2-checkpoint", required=True, type=Path)
    parser.add_argument("--da3-depth-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--encoder", default="vits", choices=MODEL_CONFIGS)
    parser.add_argument(
        "--da2-input-size",
        default=462,
        type=int,
        help="DA2 short-side resolution. 462 approximately matches DA3 long-side 756 on FLSea.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--colormap", default="Spectral_r")
    return parser.parse_args()


def colorize_geometry(disparity, colormap):
    import matplotlib

    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    colored = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    if valid.sum() < 8:
        return colored
    low, high = np.percentile(disparity[valid], [1, 99])
    normalized = np.clip((disparity - low) / max(high - low, 1e-6), 0, 1)
    rgb = matplotlib.colormaps.get_cmap(colormap)(normalized)[..., :3]
    colored = (rgb[..., ::-1] * 255).round().astype(np.uint8)
    colored[~valid] = 0
    return colored


def add_label(image, label):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (260, 42), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def load_da2(args):
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(args.da2_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(args.device).eval()


def main():
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    da2_dir = args.output_dir / "da2_raw_disparity"
    da2_vis_dir = args.output_dir / "da2_visualization"
    da3_vis_dir = args.output_dir / "da3_inverse_depth_visualization"
    comparison_dir = args.output_dir / "comparison"
    for directory in (da2_dir, da2_vis_dir, da3_vis_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = load_da2(args)
    manifest = []
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        da3_path = args.da3_depth_dir / f"{image_path.stem}_da3_depth.tif"
        da3_depth = cv2.imread(str(da3_path), cv2.IMREAD_UNCHANGED)
        if da3_depth is None:
            raise FileNotFoundError(da3_path)

        da2_disparity = model.infer_image(image, args.da2_input_size).astype(np.float32)
        da3_depth = da3_depth.astype(np.float32)
        da3_disparity = np.zeros_like(da3_depth)
        valid_da3 = np.isfinite(da3_depth) & (da3_depth > 0)
        da3_disparity[valid_da3] = 1.0 / da3_depth[valid_da3]

        da2_path = da2_dir / f"{image_path.stem}_da2_disparity.tif"
        cv2.imwrite(str(da2_path), da2_disparity)
        np.save(da2_dir / f"{image_path.stem}_da2_disparity.npy", da2_disparity)

        da2_vis = colorize_geometry(da2_disparity, args.colormap)
        da3_vis = colorize_geometry(da3_disparity, args.colormap)
        da2_vis_path = da2_vis_dir / f"{image_path.stem}_da2_disparity.png"
        da3_vis_path = da3_vis_dir / f"{image_path.stem}_da3_inverse_depth.png"
        cv2.imwrite(str(da2_vis_path), da2_vis)
        cv2.imwrite(str(da3_vis_path), da3_vis)

        rgb = cv2.resize(image, (da2_vis.shape[1], da2_vis.shape[0]))
        montage = np.concatenate(
            [
                add_label(rgb, "RGB"),
                add_label(da2_vis, f"DA2 disparity ({args.da2_input_size})"),
                add_label(da3_vis, "DA3 inverse depth"),
            ],
            axis=1,
        )
        comparison_path = comparison_dir / f"{index:04d}_{image_path.stem}.jpg"
        cv2.imwrite(str(comparison_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append((index, image_path, da2_path, da3_path, comparison_path))
        if (index + 1) % 10 == 0 or index + 1 == len(image_paths):
            print(f"Processed {index + 1}/{len(image_paths)}")

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "image", "da2_disparity", "da3_depth", "comparison"])
        writer.writerows(manifest)
    print(f"Saved comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
