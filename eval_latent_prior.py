import argparse

import torch

from dataset.flsea import FLSea
from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import MODEL_CONFIGS, evaluate


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
    args = parser.parse_args()

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

    loader = torch.utils.data.DataLoader(
        FLSea(args.val_list, "val", size=(args.img_size, args.img_size)),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    metrics = evaluate(model, loader, device, args.min_depth, args.max_depth)
    if metrics is None:
        print("No valid samples for evaluation.")
        return
    print("\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()]))


if __name__ == "__main__":
    main()
