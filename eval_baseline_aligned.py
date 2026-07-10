import argparse
import logging
import os

import torch
import torch.nn.functional as F

from dataset.flsea import FLSea
from depth_anything_v2.dpt import DepthAnythingV2
from train import MODEL_CONFIGS, _align_scale_shift
from util.metric import eval_depth
from util.utils import init_log


@torch.no_grad()
def evaluate_aligned(model, loader, device, min_depth, max_depth, eps=1e-6):
    model.eval()
    metrics = {key: 0.0 for key in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]}
    count = 0

    for sample in loader:
        image = sample["image"].to(device, non_blocking=True)
        depth = sample["depth"].to(device, non_blocking=True)
        valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()

        pred = model(image)
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[:, 0]

        valid = valid_mask & (depth >= min_depth) & (depth <= max_depth)
        target_disp = 1.0 / depth.clamp_min(eps)

        for batch_idx in range(pred.shape[0]):
            cur_valid = valid[batch_idx]
            if cur_valid.sum() < 32:
                continue

            scale, shift = _align_scale_shift(pred[batch_idx], target_disp[batch_idx], cur_valid, eps=eps)
            if scale is None:
                continue

            aligned_disp = (scale.detach() * pred[batch_idx] + shift.detach()).clamp_min(1.0 / max(max_depth, eps))
            pred_depth = (1.0 / aligned_disp).clamp(min=min_depth, max=max_depth)

            cur_metrics = eval_depth(
                pred_depth[cur_valid].clamp_min(min_depth),
                depth[batch_idx][cur_valid].clamp_min(min_depth),
            )
            for key, value in cur_metrics.items():
                metrics[key] += value
            count += 1

    if count == 0:
        return None
    return {key: value / count for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate original Depth Anything V2 with aligned metric on FLSea")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--min-depth", default=0.1, type=float)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_log("eval_baseline_aligned", logging.INFO, os.path.join(args.save_dir, "eval.log"))
    logger.propagate = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(args.pretrained_from, map_location="cpu")
    if isinstance(base_ckpt, dict) and "model" in base_ckpt:
        base_ckpt = base_ckpt["model"]

    model = DepthAnythingV2(
        **MODEL_CONFIGS[args.encoder],
        use_bn=False,
        use_clstoken=False,
        max_depth=args.max_depth,
    ).to(device)
    model.load_state_dict(base_ckpt, strict=False)
    logger.info("Loaded checkpoint: %s" % args.pretrained_from)

    dataset = FLSea(args.val_list, "val", size=(args.img_size, args.img_size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Validation dataset loaded: %d samples" % len(dataset))

    metrics = evaluate_aligned(model, loader, device, args.min_depth, args.max_depth)
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


if __name__ == "__main__":
    main()
