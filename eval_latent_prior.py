import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import MODEL_CONFIGS
from util.metric import eval_depth
from util.utils import init_log


METRIC_KEYS = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]


class Disparity2Depth:
    def __init__(self, depth_cap=40.0):
        self.depth_cap = float(depth_cap)

    def compute_scale_and_shift(self, prediction, target, mask):
        prediction_valid = prediction[mask == 1]
        target_valid = target[mask == 1]

        if prediction_valid.numel() == 0:
            return torch.tensor(1.0, device=prediction.device), torch.tensor(0.0, device=prediction.device)

        a_00 = torch.sum(prediction_valid * prediction_valid)
        a_01 = torch.sum(prediction_valid)
        a_11 = target_valid.numel()

        b_0 = torch.sum(prediction_valid * target_valid)
        b_1 = torch.sum(target_valid)

        det = a_00 * a_11 - a_01 * a_01
        if det > 0:
            scale = (a_11 * b_0 - a_01 * b_1) / det
            shift = (-a_01 * b_0 + a_00 * b_1) / det
        else:
            scale = torch.tensor(1.0, device=prediction.device)
            shift = torch.tensor(0.0, device=prediction.device)
        return scale, shift

    def __call__(self, prediction, target, mask):
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale * prediction + shift

        disparity_cap = 1.0 / self.depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap
        prediction_depth = 1.0 / prediction_aligned
        return prediction_depth


def load_file_list(file_list_path):
    pairs = []
    with open(file_list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("Each line must contain image_path and depth_path")
            pairs.append((parts[0], parts[1]))
    return pairs


def colorize_depth(depth):
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.sum() < 8:
        return colored

    values = depth[valid]
    near = np.percentile(values, 5)
    far = np.percentile(values, 95)
    if far <= near:
        far = near + 1e-6

    norm = np.clip((depth - near) / (far - near), 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def load_model(args, device, logger):
    prior_channels = tuple(int(x) for x in args.prior_channels.split(",") if x.strip())
    if len(prior_channels) != 4:
        raise ValueError("--prior-channels must contain exactly 4 comma-separated integers")

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
        use_global_prior=not args.disable_global_prior,
        use_local_prior=not args.disable_local_prior,
        use_fft_prior=not args.disable_fft_prior,
        use_deg_map=not args.disable_deg_map,
        deg_map_spatial_mean=args.deg_map_spatial_mean,
        use_plain_adapter=args.plain_adapter,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    model.load_base_weights(base_ckpt, strict=False)
    base_load_stats = model.base_load_stats
    logger.info(
        "Base checkpoint coverage: loaded=%d expected=%d missing=%d"
        % (
            base_load_stats["loaded"],
            base_load_stats["expected"],
            len(base_load_stats["missing"]),
        )
    )
    if base_load_stats["missing"]:
        preview = ", ".join(base_load_stats["missing"][:10])
        raise RuntimeError(
            "Base checkpoint is not architecture-compatible; missing %d pretrained keys. First keys: %s"
            % (len(base_load_stats["missing"]), preview)
        )

    state = torch.load(args.load_from, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    load_msg = model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s" % args.load_from)
    logger.info("Checkpoint load message: %s" % str(load_msg))
    return model


@torch.no_grad()
def evaluate_latent_prior(model, file_pairs, input_size, device, max_depth, logger, save_depth=False, depth_output_dir=""):
    results = {key: 0.0 for key in METRIC_KEYS}
    nsamples = 0
    metric = Disparity2Depth(depth_cap=max_depth)

    if save_depth:
        os.makedirs(depth_output_dir, exist_ok=True)

    for idx, (image_path, depth_path) in enumerate(file_pairs):
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            logger.info("Skip unreadable image: %s" % image_path)
            continue

        pred = model.infer_image(raw_image, input_size)
        pred_t = torch.tensor(pred, device=device, dtype=torch.float32)

        gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if gt_depth is None:
            logger.info("Skip unreadable depth: %s" % depth_path)
            continue
        if gt_depth.dtype == np.uint16:
            gt_depth = gt_depth.astype(np.float32) / 1000.0
        gt_depth_t = torch.tensor(gt_depth, device=device, dtype=torch.float32)

        valid_mask = (gt_depth_t > 0) & (gt_depth_t < max_depth)
        if valid_mask.sum() < 10:
            logger.info("Skip sample with too few valid pixels: %s" % image_path)
            continue

        if nsamples == 0:
            target_disp = torch.zeros_like(gt_depth_t)
            target_disp[valid_mask] = 1.0 / gt_depth_t[valid_mask]
            scale, shift = metric.compute_scale_and_shift(pred_t, target_disp, valid_mask)
            logger.info(
                "Validation output diagnostic: pred_min=%.6f pred_max=%.6f pred_mean=%.6f "
                "pred_std=%.6f align_scale=%.6f align_shift=%.6f"
                % (
                    pred_t.min().item(),
                    pred_t.max().item(),
                    pred_t.mean().item(),
                    pred_t.std().item(),
                    scale.item(),
                    shift.item(),
                )
            )

        prediction = metric(pred_t, gt_depth_t, valid_mask)
        cur_results = eval_depth(prediction[valid_mask], gt_depth_t[valid_mask])

        for key in METRIC_KEYS:
            results[key] += cur_results[key]
        nsamples += 1

        if save_depth:
            stem = Path(image_path).stem
            pred_np = prediction.detach().cpu().numpy().astype(np.float32)
            np.save(os.path.join(depth_output_dir, f"{idx:04d}_{stem}_pred.npy"), pred_np)
            cv2.imwrite(os.path.join(depth_output_dir, f"{idx:04d}_{stem}_pred.png"), colorize_depth(pred_np))

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d samples" % (idx + 1, len(file_pairs)))

    if nsamples == 0:
        return None
    return {key: value / nsamples for key, value in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent-prior model on FLSea with legacy baseline protocol")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--prior-base-ch", default=32, type=int)
    parser.add_argument("--prior-channels", default="32,64,128,256")
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--prior-fft-size", default=64, type=int)
    parser.add_argument("--prior-stat-hidden", default=64, type=int)
    parser.add_argument("--deg-map-scale", default=0.2, type=float)
    parser.add_argument("--disable-global-prior", action="store_true")
    parser.add_argument("--disable-local-prior", action="store_true")
    parser.add_argument("--disable-fft-prior", action="store_true")
    parser.add_argument(
        "--disable-deg-map",
        action="store_true",
        help="Use learned scalar gates instead of spatial degradation maps",
    )
    parser.add_argument(
        "--deg-map-spatial-mean",
        action="store_true",
        help="Use each generated map's per-image spatial mean",
    )
    parser.add_argument("--plain-adapter", action="store_true")
    parser.add_argument("--adapter-hidden", default=256, type=int)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--depth-output-dir", default="")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_log("eval_latent_prior", logging.INFO, os.path.join(args.save_dir, "eval.log"))
    logger.propagate = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, logger)
    file_pairs = load_file_list(args.val_list)
    logger.info("Validation file list loaded: %d samples" % len(file_pairs))

    depth_output_dir = args.depth_output_dir or os.path.join(args.save_dir, "depth_predictions")
    metrics = evaluate_latent_prior(
        model,
        file_pairs,
        args.img_size,
        device,
        args.max_depth,
        logger,
        save_depth=args.save_depth,
        depth_output_dir=depth_output_dir,
    )
    if metrics is None:
        logger.info("No valid samples for evaluation.")
        return

    print("Evaluation Results:")
    logger.info("Final Evaluation Results:")
    for key in METRIC_KEYS:
        print(f"{key}: {metrics[key]:.4f}")
        logger.info("%8s: %.4f" % (key, metrics[key]))

    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("Final Evaluation Results:\n")
        for key in METRIC_KEYS:
            handle.write(f"{key:>8}: {metrics[key]:.4f}\n")
    logger.info("Saved metrics to %s" % metrics_path)
    print(f"Log saved to: {os.path.join(args.save_dir, 'eval.log')}")


if __name__ == "__main__":
    main()
