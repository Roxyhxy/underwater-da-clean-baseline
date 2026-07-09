import argparse
import os

import torch

from dataset.flsea import FLSea
from models.depth_anything_lora import DepthAnythingLoRA
from train import MODEL_CONFIGS, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate the clean FLSea baseline")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--min-depth", default=0.1, type=float)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16.0, type=float)
    parser.add_argument("--lora-dropout", default=0.0, type=float)
    parser.add_argument("--lora-target", default="qkv", choices=["qkv", "qkv_proj", "qkv_mlp"])
    parser.add_argument("--lora-last-n-blocks", default=12, type=int)
    parser.add_argument("--lora-mode", default="aquadegrade", choices=["plain", "aquadegrade"])
    parser.add_argument("--style-dim", default=128, type=int)
    parser.add_argument("--style-hidden", default=64, type=int)
    parser.add_argument("--style-fft-size", default=64, type=int)
    parser.add_argument("--use-decoder-lora", action="store_true")
    parser.add_argument("--decoder-lora-rank", default=2, type=int)
    parser.add_argument("--decoder-lora-alpha", default=4.0, type=float)
    parser.add_argument("--decoder-lora-dropout", default=0.0, type=float)
    parser.add_argument("--decoder-lora-target", default="tail", choices=["post_vfe", "refinenet", "tail", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(args.pretrained_from, map_location="cpu")
    if isinstance(base_ckpt, dict) and "model" in base_ckpt:
        base_ckpt = base_ckpt["model"]

    model = DepthAnythingLoRA(
        **MODEL_CONFIGS[args.encoder],
        max_depth=args.max_depth,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        lora_last_n_blocks=args.lora_last_n_blocks,
        lora_mode=args.lora_mode,
        style_dim=args.style_dim,
        style_hidden=args.style_hidden,
        style_fft_size=args.style_fft_size,
        use_decoder_lora=args.use_decoder_lora,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
        decoder_lora_target=args.decoder_lora_target,
    ).to(device)
    model.load_base_weights(base_ckpt, strict=False)
    model.freeze_base_and_inject_lora()

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

