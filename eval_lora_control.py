import argparse
import logging
import os

import torch

from eval_latent_prior import METRIC_KEYS, evaluate_latent_prior, load_file_list
from models.depth_anything_lora import DepthAnythingLoRA
from train import MODEL_CONFIGS
from util.utils import init_log


def main():
    parser = argparse.ArgumentParser(description="Evaluate AquaDegrade-LoRA with the legacy FLSea protocol")
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", default=518, type=int)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16.0, type=float)
    parser.add_argument("--lora-target", default="qkv", choices=["qkv", "qkv_proj", "qkv_mlp"])
    parser.add_argument("--lora-last-n-blocks", default=12, type=int)
    parser.add_argument("--style-dim", default=128, type=int)
    parser.add_argument("--style-hidden", default=64, type=int)
    parser.add_argument("--style-fft-size", default=64, type=int)
    parser.add_argument("--decoder-lora-rank", default=2, type=int)
    parser.add_argument("--decoder-lora-alpha", default=4.0, type=float)
    parser.add_argument("--decoder-lora-target", default="tail", choices=["post_vfe", "refinenet", "tail", "all"])
    parser.add_argument("--save-raw-disparity", action="store_true")
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--raw-colormap", default="Spectral_r")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_log("eval_lora_control", logging.INFO, os.path.join(args.save_dir, "eval.log"))
    logger.propagate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthAnythingLoRA(
        **MODEL_CONFIGS[args.encoder],
        max_depth=args.max_depth,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
        lora_last_n_blocks=args.lora_last_n_blocks,
        lora_mode="aquadegrade",
        style_dim=args.style_dim,
        style_hidden=args.style_hidden,
        style_fft_size=args.style_fft_size,
        use_decoder_lora=True,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_target=args.decoder_lora_target,
    )
    base_state = torch.load(args.pretrained_from, map_location="cpu")
    base_state = base_state["model"] if isinstance(base_state, dict) and "model" in base_state else base_state
    model.load_base_weights(base_state, strict=True)
    model.freeze_base_and_inject_lora()

    checkpoint = torch.load(args.load_from, map_location="cpu")
    lora_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_msg = model.load_state_dict(lora_state, strict=False)
    if load_msg.unexpected_keys:
        raise RuntimeError("Unexpected LoRA checkpoint keys: %s" % load_msg.unexpected_keys[:10])
    missing_checkpoint_keys = sorted(set(lora_state) - set(model.state_dict()))
    if missing_checkpoint_keys:
        raise RuntimeError("LoRA checkpoint keys not loaded: %s" % missing_checkpoint_keys[:10])

    model = model.to(device).eval()
    pairs = load_file_list(args.val_list)
    logger.info("Loaded AquaDegrade-LoRA checkpoint: %s" % args.load_from)
    logger.info("Validation file list loaded: %d samples" % len(pairs))
    raw_output_dir = args.raw_output_dir or os.path.join(args.save_dir, "raw_disparity")
    metrics = evaluate_latent_prior(
        model,
        pairs,
        args.input_size,
        device,
        args.max_depth,
        logger,
        save_raw=args.save_raw_disparity,
        raw_output_dir=raw_output_dir,
        raw_colormap=args.raw_colormap,
    )
    if metrics is None:
        raise RuntimeError("No valid FLSea samples were evaluated")

    print("Evaluation Results:")
    logger.info("Final Evaluation Results:")
    with open(os.path.join(args.save_dir, "metrics.txt"), "w", encoding="utf-8") as handle:
        handle.write("Final Evaluation Results:\n")
        for key in METRIC_KEYS:
            print(f"{key}: {metrics[key]:.4f}")
            logger.info("%8s: %.4f" % (key, metrics[key]))
            handle.write(f"{key:>8}: {metrics[key]:.4f}\n")


if __name__ == "__main__":
    main()
