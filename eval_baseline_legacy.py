import argparse
import logging
import os

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth
from util.utils import init_log


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": (48, 96, 192, 384)},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": (96, 192, 384, 768)},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": (256, 512, 1024, 1024)},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": (1536, 1536, 1536, 1536)},
}


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


@torch.no_grad()
def evaluate_legacy(model, file_pairs, input_size, device, max_depth, logger):
    results = {key: 0.0 for key in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]}
    nsamples = 0
    metric = Disparity2Depth(depth_cap=max_depth)

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

        prediction = metric(pred_t, gt_depth_t, valid_mask)
        cur_results = eval_depth(prediction[valid_mask], gt_depth_t[valid_mask])

        for key in results.keys():
            results[key] += cur_results[key]
        nsamples += 1

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d samples" % (idx + 1, len(file_pairs)))

    if nsamples == 0:
        return None
    return {key: value / nsamples for key, value in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Legacy FLSea baseline evaluation at original resolution")
    parser.add_argument("--img-path", required=True, help="TXT file with image_path depth_path per line")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--input-size", default=518, type=int)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_log("eval_baseline_legacy", logging.INFO, os.path.join(args.save_dir, "eval.log"))
    logger.propagate = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(args.load_from, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        load_msg = model.load_state_dict(state["model"], strict=True)
    else:
        load_msg = model.load_state_dict(torch.load(args.load_from, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s" % args.load_from)
    logger.info("Checkpoint load message: %s" % str(load_msg))

    file_pairs = load_file_list(args.img_path)
    logger.info("Validation file list loaded: %d samples" % len(file_pairs))

    metrics = evaluate_legacy(model, file_pairs, args.input_size, device, args.max_depth, logger)
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
