import argparse
import logging
import os

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class Disparity2Depth:
    def __init__(self, threshold=1.25, depth_cap=40):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

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

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap
        prediction_depth = 1.0 / prediction_aligned
        return prediction_depth


def load_pairs(file_list_path):
    pairs = []
    with open(file_list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Exact DA_0-style FLSea baseline evaluation")
    parser.add_argument("--img-path", required=True, help="TXT file with image_path depth_path per line")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--outdir", type=str, default="./eval/flsea_baseline_da0_exact")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--load-from", type=str, required=True)
    parser.add_argument("--max-depth", type=float, default=40.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_file = os.path.join(args.outdir, "evaluation.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    checkpoint = torch.load(args.load_from, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(torch.load(args.load_from, map_location="cpu"))
    model = model.to(device).eval()

    pairs = load_pairs(args.img_path)
    results = {
        "d1": torch.tensor([0.0], device=device),
        "d2": torch.tensor([0.0], device=device),
        "d3": torch.tensor([0.0], device=device),
        "abs_rel": torch.tensor([0.0], device=device),
        "sq_rel": torch.tensor([0.0], device=device),
        "rmse": torch.tensor([0.0], device=device),
        "rmse_log": torch.tensor([0.0], device=device),
        "log10": torch.tensor([0.0], device=device),
        "silog": torch.tensor([0.0], device=device),
    }
    nsamples = torch.tensor([0.0], device=device)
    metric = Disparity2Depth(depth_cap=args.max_depth)

    for idx, (image_path, depth_path) in enumerate(pairs):
        print(f"Progress {idx + 1}/{len(pairs)}: {image_path}")
        raw_image = cv2.imread(image_path)
        depth = model.infer_image(raw_image, args.input_size)

        if depth_path.endswith(".npy"):
            gt_depth = np.load(depth_path)
        else:
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if gt_depth.dtype == np.uint16:
                gt_depth = gt_depth.astype(np.float32) / 1000.0

        depth_t = torch.tensor(depth, device=device, dtype=torch.float32)
        gt_depth_t = torch.tensor(gt_depth, device=device, dtype=torch.float32)
        valid_mask = (gt_depth_t > 0) & (gt_depth_t < args.max_depth)

        prediction = metric(depth_t, gt_depth_t, valid_mask)
        cur_results = eval_depth(prediction[valid_mask], gt_depth_t[valid_mask])
        for key in results.keys():
            results[key] += cur_results[key]
        nsamples += 1

    print("Evaluation Results:")
    logging.info("Evaluation Results:")
    for key in results.keys():
        avg_value = results[key].item() / nsamples.item()
        print(f"{key}: {avg_value:.4f}")
        logging.info(f"{key}: {avg_value:.4f}")

    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
