import argparse

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import MODEL_CONFIGS


def tensor_diff(name, lhs, rhs):
    lhs = lhs.float()
    rhs = rhs.float()
    diff = (lhs - rhs).abs()
    print(
        f"{name}: shape={tuple(lhs.shape)} max_abs={diff.max().item():.9f} "
        f"mean_abs={diff.mean().item():.9f} lhs_mean={lhs.mean().item():.9f} "
        f"rhs_mean={rhs.mean().item():.9f}"
    )


def first_image_path(file_list):
    with open(file_list, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                return parts[0]
    raise ValueError(f"No samples found in {file_list}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare the exact baseline and zero-prior forward paths")
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", default=518, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.pretrained_from, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state

    baseline = DepthAnythingV2(**MODEL_CONFIGS[args.encoder]).to(device).eval()
    baseline.load_state_dict(state, strict=True)

    latent = DepthAnythingLatentPrior(
        **MODEL_CONFIGS[args.encoder],
        max_depth=40.0,
    ).to(device).eval()
    latent.load_base_weights(state, strict=False)
    print(f"base checkpoint coverage: {latent.base_load_stats}")

    image_path = first_image_path(args.val_list)
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(image_path)
    print(f"image: {image_path}")

    baseline_image, _ = baseline.image2tensor(raw_image, args.input_size)
    latent_image, _ = latent.image2tensor(raw_image, args.input_size)
    tensor_diff("input", baseline_image, latent_image)

    patch_h, patch_w = baseline_image.shape[-2] // 14, baseline_image.shape[-1] // 14
    baseline_features = baseline.pretrained.get_intermediate_layers(
        baseline_image,
        baseline.intermediate_layer_idx[args.encoder],
        return_class_token=True,
    )
    latent_features = latent.pretrained.get_intermediate_layers(
        latent_image,
        latent.intermediate_layer_idx[args.encoder],
        return_class_token=True,
    )
    for index, (base_feature, latent_feature) in enumerate(zip(baseline_features, latent_features)):
        tensor_diff(f"feature_{index}_tokens", base_feature[0], latent_feature[0])
        tensor_diff(f"feature_{index}_cls", base_feature[1], latent_feature[1])

    base_head = baseline.depth_head(baseline_features, patch_h, patch_w)
    pure_base_from_latent_features = baseline.depth_head(latent_features, patch_h, patch_w)
    tensor_diff("base_head_feature_check", base_head, pure_base_from_latent_features)

    z_deg, prior_pyramid = latent.latent_prior_encoder(latent_image)
    latent_head = latent.depth_head(
        latent_features,
        patch_h,
        patch_w,
        z_deg,
        prior_pyramid,
    )
    tensor_diff("base_vs_zero_prior_head", base_head, latent_head)

    baseline_pred = baseline.infer_image(raw_image, args.input_size)
    latent_pred = latent.infer_image(raw_image, args.input_size)
    baseline_pred = torch.from_numpy(baseline_pred) / float(baseline.max_depth)
    latent_pred = torch.from_numpy(latent_pred) / float(latent.max_depth)
    tensor_diff("normalized_final_prediction", baseline_pred, latent_pred)


if __name__ == "__main__":
    main()
