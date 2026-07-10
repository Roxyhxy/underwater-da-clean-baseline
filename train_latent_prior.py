import argparse
import logging
import os
import pprint

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.flsea import FLSea
from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import (
    MODEL_CONFIGS,
    affine_disparity_gradient_loss,
    aligned_depth_metric_loss,
    depth_anything_consistency_hardness_weight,
    depthdive_l1_silog_loss,
    evaluate,
    make_scheduler,
    normalized_disparity_shape_loss,
    random_underwater_consistency_augment,
    set_seed,
)
from util.utils import init_log


def save_checkpoint(model, path, args, epoch=None, metrics=None):
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def build_optimizer(model, args):
    groups = []
    latent_prior_params = []
    prior_head_params = []
    base_head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("latent_prior_encoder."):
            latent_prior_params.append(param)
        elif name.startswith("depth_head.global_mod.") or name.startswith("depth_head.deg_map_generator.") or name.startswith(
            "depth_head.prior_to_feat."
        ):
            prior_head_params.append(param)
        elif name.startswith("depth_head."):
            base_head_params.append(param)
        elif name.startswith("pretrained."):
            backbone_params.append(param)

    if latent_prior_params:
        groups.append({"params": latent_prior_params, "lr": args.prior_lr or args.lr})
    if prior_head_params:
        groups.append({"params": prior_head_params, "lr": args.prior_head_lr or args.lr})
    if base_head_params:
        groups.append({"params": base_head_params, "lr": args.head_lr or args.lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr or args.lr})
    return AdamW(groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)


def main():
    parser = argparse.ArgumentParser(description="Train FLSea with latent-prior Depth Anything V2")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--prior-lr", default=0.0, type=float)
    parser.add_argument("--prior-head-lr", default=0.0, type=float)
    parser.add_argument("--head-lr", default=0.0, type=float)
    parser.add_argument("--backbone-lr", default=0.0, type=float)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--init-from", default="")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min-depth", default=0.1, type=float)
    parser.add_argument("--max-depth", default=40.0, type=float)
    parser.add_argument("--prior-base-ch", default=32, type=int)
    parser.add_argument("--prior-channels", default="32,64,128,256")
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--prior-fft-size", default=64, type=int)
    parser.add_argument("--prior-stat-hidden", default=64, type=int)
    parser.add_argument("--deg-map-scale", default=0.2, type=float)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-base-head", action="store_true")
    parser.add_argument("--freeze-latent-prior", action="store_true")
    parser.add_argument("--loss-mode", default="depthdive_relative", choices=["relative_metric", "depthdive_relative"])
    parser.add_argument("--l1-weight", default=0.5, type=float)
    parser.add_argument("--silog-weight", default=0.5, type=float)
    parser.add_argument("--metric-weight", default=1.0, type=float)
    parser.add_argument("--grad-weight", default=0.05, type=float)
    parser.add_argument("--consistency-hardness-weight", default=0.08, type=float)
    parser.add_argument("--consistency-hardness-clamp-min", default=0.90, type=float)
    parser.add_argument("--consistency-hardness-clamp-max", default=1.10, type=float)
    parser.add_argument("--consistency-aug-prob", default=1.0, type=float)
    parser.add_argument("--consistency-blur-prob", default=0.30, type=float)
    parser.add_argument("--consistency-noise-prob", default=0.20, type=float)
    parser.add_argument("--consistency-noise-std", default=0.01, type=float)
    parser.add_argument("--warmup-steps", default=100, type=int)
    parser.add_argument("--min-lr-ratio", default=0.2, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log("latent_prior", logging.INFO)
    logger.propagate = 0
    logger.info(pprint.pformat(vars(args)))
    set_seed(args.seed)

    prior_channels = tuple(int(x) for x in args.prior_channels.split(",") if x.strip())
    if len(prior_channels) != 4:
        raise ValueError("--prior-channels must contain exactly 4 comma-separated integers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = FLSea(args.train_list, "train", size=(args.img_size, args.img_size))
    val_set = FLSea(args.val_list, "val", size=(args.img_size, args.img_size))
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
    model.configure_trainable(
        freeze_backbone=args.freeze_backbone,
        freeze_base_head=args.freeze_base_head,
        train_latent_prior=not args.freeze_latent_prior,
    )

    if args.init_from:
        init_state = torch.load(args.init_from, map_location="cpu")
        init_state = init_state["model"] if isinstance(init_state, dict) and "model" in init_state else init_state
        model.load_state_dict(init_state, strict=False)
        logger.info(f"Loaded latent-prior checkpoint from {args.init_from}")

    optimizer = build_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps, args.min_lr_ratio)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_abs_rel = float("inf")
    best_d1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        for step, sample in enumerate(train_loader):
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()
            valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_disp = model(image)
                sup_weight = valid_mask.float()

                if args.consistency_hardness_weight > 0:
                    aug_image = random_underwater_consistency_augment(
                        image,
                        blur_prob=args.consistency_blur_prob,
                        noise_prob=args.consistency_noise_prob,
                        noise_std=args.consistency_noise_std,
                    )
                    with torch.no_grad():
                        aug_disp = model(aug_image).detach()
                    cons_weight, cons_map = depth_anything_consistency_hardness_weight(
                        pred_disp,
                        aug_disp,
                        valid_mask,
                        alpha=args.consistency_hardness_weight,
                        clamp_min=args.consistency_hardness_clamp_min,
                        clamp_max=args.consistency_hardness_clamp_max,
                    )
                    sup_weight = sup_weight * cons_weight
                else:
                    cons_map = pred_disp.new_zeros(pred_disp.shape[0], pred_disp.shape[-2], pred_disp.shape[-1])

                if args.loss_mode == "depthdive_relative":
                    loss_l1, loss_silog = depthdive_l1_silog_loss(
                        pred_disp, depth, valid_mask, weight=sup_weight, max_depth=args.max_depth
                    )
                else:
                    loss_l1, loss_silog = normalized_disparity_shape_loss(
                        pred_disp, depth, valid_mask, weight=sup_weight
                    )

                loss_metric = aligned_depth_metric_loss(
                    pred_disp, depth, valid_mask, weight=sup_weight, max_depth=args.max_depth
                )
                loss_grad = affine_disparity_gradient_loss(pred_disp, depth, valid_mask, weight=sup_weight)
                loss = (
                    args.l1_weight * loss_l1
                    + args.silog_weight * loss_silog
                    + args.metric_weight * loss_metric
                    + args.grad_weight * loss_grad
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 20 == 0:
                logger.info(
                    "Epoch %02d Iter %04d/%04d Loss=%.6f L1=%.6f SiLog=%.6f Metric=%.6f Grad=%.6f HardMean=%.6f LR=%.2e"
                    % (
                        epoch + 1,
                        step,
                        len(train_loader),
                        loss.item(),
                        loss_l1.item(),
                        loss_silog.item(),
                        loss_metric.item(),
                        loss_grad.item(),
                        cons_map.mean().item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        metrics = evaluate(model, val_loader, device, args.min_depth, args.max_depth)
        save_checkpoint(model, os.path.join(args.save_path, "latest.pth"), args, epoch=epoch, metrics=metrics)
        if metrics is None:
            logger.info("Validation skipped: no valid samples")
            continue

        logger.info("Validation: " + ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()]))
        if metrics["d1"] > best_d1:
            best_d1 = metrics["d1"]
            save_checkpoint(model, os.path.join(args.save_path, "best_d1.pth"), args, epoch=epoch, metrics=metrics)
        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            save_checkpoint(model, os.path.join(args.save_path, "best_abs_rel.pth"), args, epoch=epoch, metrics=metrics)


if __name__ == "__main__":
    main()
