import argparse
import logging
import math
import os
import pprint
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.uwd import UWD
from depth_anything_v2.dpt import DepthAnythingV2
from util.depth_losses import dense_relative_depth_loss
from util.metric import eval_depth
from util.utils import init_log


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": (48, 96, 192, 384)},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": (96, 192, 384, 768)},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": (256, 512, 1024, 1024)},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": (1536, 1536, 1536, 1536)},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def freeze_encoder_train_decoder(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.depth_head.parameters():
        param.requires_grad = True


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def make_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def align_disparity(prediction, target, valid_mask, eps=1e-6):
    """Least-squares scale/shift alignment in inverse-depth space."""

    pred = prediction[valid_mask].float()
    target = target[valid_mask].float()
    if pred.numel() < 32:
        return None

    pred_mean = pred.mean()
    target_mean = target.mean()
    variance = ((pred - pred_mean) ** 2).mean()
    if variance < eps:
        return None
    scale = ((pred - pred_mean) * (target - target_mean)).mean() / variance
    shift = target_mean - scale * pred_mean
    return scale * prediction.float() + shift


@torch.no_grad()
def evaluate(model, loader, device, min_depth, max_depth):
    model.eval()
    totals = {
        key: 0.0
        for key in [
            "d1",
            "d2",
            "d3",
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "log10",
            "silog",
        ]
    }
    count = 0

    for sample in loader:
        image = sample["image"].to(device, non_blocking=True)
        depth = sample["depth"].to(device, non_blocking=True)
        valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()

        pred_disp = model(image)
        pred_disp = F.interpolate(
            pred_disp[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
        )[:, 0]

        valid_mask = (
            valid_mask
            & torch.isfinite(depth)
            & (depth >= min_depth)
            & (depth <= max_depth)
        )
        target_disp = depth.clamp_min(1e-6).reciprocal()

        for batch_index in range(pred_disp.shape[0]):
            valid = valid_mask[batch_index] & torch.isfinite(pred_disp[batch_index])
            aligned_disp = align_disparity(
                pred_disp[batch_index], target_disp[batch_index], valid
            )
            if aligned_disp is None:
                continue

            aligned_disp = aligned_disp.clamp_min(1.0 / max_depth)
            pred_depth = aligned_disp.reciprocal().clamp(min_depth, max_depth)
            valid = valid & torch.isfinite(pred_depth)
            if valid.sum() < 32:
                continue

            metrics = eval_depth(pred_depth[valid], depth[batch_index][valid])
            for key, value in metrics.items():
                totals[key] += value
            count += 1

    if count == 0:
        return None
    return {key: value / count for key, value in totals.items()}


def extract_decoder_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("depth_head.")
    }


def save_checkpoint(model, optimizer, scheduler, path, args, epoch, metrics):
    torch.save(
        {
            "model": extract_decoder_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="UW-D dense relative-depth baseline for Depth Anything V2"
    )
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--init-from", default="")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min-depth", default=0.1, type=float)
    parser.add_argument("--max-depth", default=20.0, type=float)
    parser.add_argument("--gradient-weight", default=0.5, type=float)
    parser.add_argument("--warmup-steps", default=100, type=int)
    parser.add_argument("--min-lr-ratio", default=0.2, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-interval", default=20, type=int)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log("uwd_dense", logging.INFO, os.path.join(args.save_path, "train.log"))
    logger.propagate = 0
    logger.info(pprint.pformat(vars(args)))
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = UWD(
        args.dataset_root,
        args.train_list,
        mode="train",
        size=(args.img_size, args.img_size),
    )
    val_set = UWD(
        args.dataset_root,
        args.val_list,
        mode="val",
        size=(args.img_size, args.img_size),
        random_flip=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = DepthAnythingV2(
        **MODEL_CONFIGS[args.encoder],
        use_bn=False,
        use_clstoken=False,
        max_depth=args.max_depth,
    ).to(device)
    load_result = model.load_state_dict(load_state_dict(args.pretrained_from), strict=False)
    logger.info(
        "Loaded DA2 checkpoint: missing=%d unexpected=%d",
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    freeze_encoder_train_decoder(model)

    if args.init_from:
        init_result = model.load_state_dict(load_state_dict(args.init_from), strict=False)
        logger.info(
            "Loaded decoder checkpoint: missing=%d unexpected=%d",
            len(init_result.missing_keys),
            len(init_result.unexpected_keys),
        )

    total_params, trainable_params = count_parameters(model)
    logger.info(
        "Parameters: total=%d trainable=%d (%.2f%%)",
        total_params,
        trainable_params,
        100.0 * trainable_params / max(total_params, 1),
    )

    optimizer = AdamW(
        model.depth_head.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(
        optimizer, total_steps, args.warmup_steps, args.min_lr_ratio
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_abs_rel = float("inf")
    best_d1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        model.pretrained.eval()
        running_loss = 0.0

        for step, sample in enumerate(train_loader):
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()
            valid_mask = (
                valid_mask
                & torch.isfinite(depth)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth)
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_disp = model(image)
                pred_disp = F.interpolate(
                    pred_disp[:, None],
                    depth.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
                losses = dense_relative_depth_loss(
                    pred_disp,
                    depth,
                    valid_mask,
                    gradient_weight=args.gradient_weight,
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

            if step % args.log_interval == 0:
                logger.info(
                    "Epoch %02d Iter %04d/%04d Loss=%.6f Data=%.6f Grad=%.6f Valid=%.4f LR=%.2e",
                    epoch + 1,
                    step,
                    len(train_loader),
                    loss.item(),
                    losses["data_loss"].item(),
                    losses["gradient_loss"].item(),
                    valid_mask.float().mean().item(),
                    optimizer.param_groups[0]["lr"],
                )

        metrics = evaluate(
            model, val_loader, device, args.min_depth, args.max_depth
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            os.path.join(args.save_path, "latest.pth"),
            args,
            epoch,
            metrics,
        )
        logger.info(
            "Epoch %02d mean_train_loss=%.6f",
            epoch + 1,
            running_loss / max(len(train_loader), 1),
        )
        if metrics is None:
            logger.info("Validation skipped: no valid samples")
            continue

        logger.info(
            "Validation: "
            + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        )
        if metrics["d1"] > best_d1:
            best_d1 = metrics["d1"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                os.path.join(args.save_path, "best_d1.pth"),
                args,
                epoch,
                metrics,
            )
        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                os.path.join(args.save_path, "best_abs_rel.pth"),
                args,
                epoch,
                metrics,
            )


if __name__ == "__main__":
    main()
