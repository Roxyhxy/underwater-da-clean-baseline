#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.flsea_wat3r import FLSeaWat3R
from eval_latent_prior import load_model
from util.utils import init_log
from util.wat3r_distillation import (
    build_teacher_reliability_mask,
    wat3r_hole_distillation_loss,
    wat3r_hole_edge_metrics,
    wat3r_multiview_geometry_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Wat3R hole, temporal, and edge geometry proxies."
    )
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--wat3r-manifest", required=True)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--max-samples", default=1200, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--confidence-quantile", default=0.6, type=float)
    parser.add_argument("--relative-depth-threshold", default=0.05, type=float)
    parser.add_argument("--min-align-pixels", default=100, type=int)
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--prior-base-ch", default=32, type=int)
    parser.add_argument("--prior-channels", default="32,64,128,256")
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--prior-fft-size", default=64, type=int)
    parser.add_argument("--prior-stat-hidden", default=64, type=int)
    parser.add_argument("--deg-map-scale", default=0.1, type=float)
    parser.add_argument("--disable-global-prior", action="store_true")
    parser.add_argument("--disable-local-prior", action="store_true")
    parser.add_argument("--disable-fft-prior", action="store_true")
    parser.add_argument("--disable-deg-map", action="store_true")
    parser.add_argument("--deg-map-spatial-mean", action="store_true")
    parser.add_argument("--plain-adapter", action="store_true")
    parser.add_argument("--adapter-hidden", default=256, type=int)
    parser.add_argument("--encoder-lora", action="store_true")
    parser.add_argument("--encoder-lora-mode", default="gated", choices=["plain", "gated"])
    parser.add_argument(
        "--encoder-lora-condition-source", default="fft", choices=["fused", "fft"]
    )
    parser.add_argument("--encoder-lora-rank", default=8, type=int)
    parser.add_argument("--encoder-lora-alpha", default=16.0, type=float)
    parser.add_argument("--encoder-lora-dropout", default=0.0, type=float)
    parser.add_argument("--encoder-lora-last-n-blocks", default=12, type=int)
    parser.add_argument("--geometry-adapter", action="store_true")
    parser.add_argument("--geometry-adapter-hidden", default=32, type=int)
    parser.add_argument("--geometry-adapter-scale", default=0.1, type=float)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logger = init_log(
        "eval_wat3r_geometry",
        logging.INFO,
        str(args.save_dir / "geometry_eval.log"),
    )
    logger.propagate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, logger)
    dataset = FLSeaWat3R(
        args.train_list,
        args.wat3r_manifest,
        size=(args.img_size, args.img_size),
        require_overlap=args.require_overlap,
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    totals = {
        "hole_teacher_value": 0.0,
        "hole_teacher_gradient": 0.0,
        "reliable_hole_ratio": 0.0,
        "hole_coverage": 0.0,
        "temporal_reprojection": 0.0,
        "temporal_coverage": 0.0,
        "edge_mae": 0.0,
        "edge_f1": 0.0,
    }
    counts = {
        "hole": 0,
        "temporal": 0,
        "edge": 0,
    }
    processed = 0
    for sample in loader:
        if args.max_samples > 0 and processed >= args.max_samples:
            break
        images = sample["images"].to(device)
        batch, views = images.shape[:2]
        prediction = model(images.flatten(0, 1)).reshape(
            batch, views, *images.shape[-2:]
        )
        teacher_depth = sample["teacher_depth"].to(device)
        confidence = sample["teacher_confidence"].to(device)
        static = sample["teacher_static_mask"].to(device).bool()
        overlap = sample["teacher_overlap_mask"].to(device).bool()
        reliable = build_teacher_reliability_mask(
            teacher_depth.flatten(0, 1),
            confidence.flatten(0, 1),
            static.flatten(0, 1),
            args.confidence_quantile,
        ).reshape_as(static)
        if args.require_overlap:
            reliable = reliable & overlap

        gt_depth = sample["depth"].to(device)
        observed = sample["valid_mask"].to(device).bool()
        hole = wat3r_hole_distillation_loss(
            prediction[:, 1],
            teacher_depth[:, 1],
            gt_depth,
            observed,
            reliable[:, 1],
            min_align_pixels=args.min_align_pixels,
            min_hole_pixels=1,
            boundary_weight=1.0,
            far_weight=1.0,
        )
        temporal = wat3r_multiview_geometry_loss(
            prediction,
            teacher_depth,
            reliable,
            sample["intrinsics"].to(device),
            sample["extrinsics"].to(device),
            source_focus_mask=torch.stack(
                (
                    torch.zeros_like(observed),
                    ~observed,
                    torch.zeros_like(observed),
                ),
                dim=1,
            ),
            relative_depth_threshold=args.relative_depth_threshold,
            min_align_pixels=args.min_align_pixels,
        )
        edge = wat3r_hole_edge_metrics(
            prediction[:, 1],
            teacher_depth[:, 1],
            gt_depth,
            observed,
            reliable[:, 1],
            min_align_pixels=args.min_align_pixels,
        )
        totals["reliable_hole_ratio"] += float(hole["hole_ratio"].item())
        totals["hole_coverage"] += float(hole["coverage"].item())
        totals["temporal_coverage"] += float(temporal["coverage"].item())
        if int(hole["pixels"].item()) > 0:
            totals["hole_teacher_value"] += float(hole["value"].item())
            totals["hole_teacher_gradient"] += float(hole["gradient"].item())
            counts["hole"] += 1
        if int(temporal["pixels"].item()) > 0:
            totals["temporal_reprojection"] += float(temporal["loss"].item())
            counts["temporal"] += 1
        if int(edge["edge_pixels"].item()) > 0:
            totals["edge_mae"] += float(edge["edge_mae"].item())
            totals["edge_f1"] += float(edge["edge_f1"].item())
            counts["edge"] += 1
        processed += 1
        if processed % 100 == 0:
            logger.info("Processed %d geometry triplets" % processed)

    if processed == 0:
        raise RuntimeError("No Wat3R geometry samples were evaluated")
    metrics = {
        "hole_teacher_value": totals["hole_teacher_value"] / max(counts["hole"], 1),
        "hole_teacher_gradient": totals["hole_teacher_gradient"] / max(counts["hole"], 1),
        "reliable_hole_ratio": totals["reliable_hole_ratio"] / processed,
        "hole_coverage": totals["hole_coverage"] / processed,
        "temporal_reprojection": totals["temporal_reprojection"]
        / max(counts["temporal"], 1),
        "temporal_coverage": totals["temporal_coverage"] / processed,
        "edge_mae": totals["edge_mae"] / max(counts["edge"], 1),
        "edge_f1": totals["edge_f1"] / max(counts["edge"], 1),
        "hole_valid_samples": counts["hole"],
        "temporal_valid_samples": counts["temporal"],
        "edge_valid_samples": counts["edge"],
    }
    metrics["samples"] = processed
    with (args.save_dir / "geometry_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Final Wat3R geometry metrics:")
    for key, value in metrics.items():
        logger.info("%28s: %s" % (key, f"{value:.6f}" if isinstance(value, float) else value))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
