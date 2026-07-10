import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset.flsea import FLSea
from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import MODEL_CONFIGS, evaluate
from util.utils import init_log


def _colorize_depth(depth, valid_mask=None):
    depth = depth.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 0)
    else:
        valid_mask = valid_mask.astype(bool)

    colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid_mask.sum() < 8:
        return colored

    values = depth[valid_mask]
    near = np.percentile(values, 5)
    far = np.percentile(values, 95)
    if far <= near:
        far = near + 1e-6

    norm = np.clip((depth - near) / (far - near), 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored[~valid_mask] = 0
    return colored


@torch.no_grad()
def export_depth_predictions(model, loader, device, output_dir, min_depth, max_depth, logger):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for idx, sample in enumerate(loader):
        image = sample["image"].to(device, non_blocking=True)
        depth = sample["depth"].to(device, non_blocking=True)
        valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()
        image_path = sample["image_path"][0]

        pred = model(image)
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
        pred = pred[0].detach().cpu().numpy().astype(np.float32)

        gt = depth[0].detach().cpu().numpy().astype(np.float32)
        valid = (valid_mask[0].detach().cpu().numpy().astype(bool)) & np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth)

        stem = Path(image_path).stem
        npy_path = os.path.join(output_dir, f"{idx:04d}_{stem}_pred.npy")
        png_path = os.path.join(output_dir, f"{idx:04d}_{stem}_pred.png")

        np.save(npy_path, pred)
        color = _colorize_depth(pred, valid_mask=valid if valid.any() else None)
        cv2.imwrite(png_path, color)

    logger.info("Saved depth predictions to %s" % output_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent-prior Depth Anything on FLSea")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--min-depth", default=0.1, type=float)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--prior-base-ch", default=32, type=int)
    parser.add_argument("--prior-channels", default="32,64,128,256")
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--prior-fft-size", default=64, type=int)
    parser.add_argument("--prior-stat-hidden", default=64, type=int)
    parser.add_argument("--deg-map-scale", default=0.2, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--depth-output-dir", default="")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_log("eval_latent_prior", logging.INFO, os.path.join(args.save_dir, "eval.log"))
    logger.propagate = 0

    prior_channels = tuple(int(x) for x in args.prior_channels.split(",") if x.strip())
    if len(prior_channels) != 4:
        raise ValueError("--prior-channels must contain exactly 4 comma-separated integers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(args.pretrained_from, map_location="cpu")
    if isinstance(base_ckpt, dict) and "model" in base_ckpt:
        base_ckpt = base_ckpt["model"]

    model = DepthAnythingLatentPrior(
        **MODEL_CONFIGS[args.encoder],
        max_depth=args.max_depth,
        prior_base_ch=args.prior_base_ch,
        prior_channels=prior_channels,
        latent_dim=args.latent_dim,
        prior_fft_size=args.prior_fft_size,
        prior_stat_hidden=args.prior_stat_hidden,
        deg_map_scale=args.deg_map_scale,
    ).to(device)
    model.load_base_weights(base_ckpt, strict=False)

    state = torch.load(args.load_from, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=False)
    logger.info("Loaded checkpoint: %s" % args.load_from)

    dataset = FLSea(args.val_list, "val", size=(args.img_size, args.img_size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Validation dataset loaded: %d samples" % len(dataset))

    metrics = evaluate(model, loader, device, args.min_depth, args.max_depth)
    if metrics is None:
        logger.info("No valid samples for evaluation.")
        return

    logger.info("Final Evaluation Results:")
    for key in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]:
        logger.info("%8s: %.4f" % (key, metrics[key]))

    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("Final Evaluation Results:\n")
        for key in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]:
            handle.write(f"{key:>8}: {metrics[key]:.4f}\n")
    logger.info("Saved metrics to %s" % metrics_path)

    if args.save_depth:
        output_dir = args.depth_output_dir or os.path.join(args.save_dir, "depth_predictions")
        export_depth_predictions(model, loader, device, output_dir, args.min_depth, args.max_depth, logger)


if __name__ == "__main__":
    main()
