import argparse
import logging
import os
import pprint
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.flsea import FLSea
from dataset.flsea_wat3r import FLSeaWat3R
from eval_latent_prior import evaluate_latent_prior, load_file_list
from models.depth_anything_latent_prior import DepthAnythingLatentPrior
from train import (
    MODEL_CONFIGS,
    affine_disparity_gradient_loss,
    aligned_depth_metric_loss,
    depth_anything_consistency_hardness_weight,
    depthdive_l1_silog_loss,
    make_scheduler,
    normalized_disparity_shape_loss,
    random_underwater_consistency_augment,
    set_seed,
)
from util.hole_geometry import hole_geometry_preservation_loss
from util.utils import init_log
from util.wat3r_distillation import (
    build_teacher_reliability_mask,
    wat3r_hole_distillation_loss,
    wat3r_multiview_geometry_loss,
)


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


def _count_params(params):
    return sum(param.numel() for param in params)


def seed_data_worker(worker_id):
    """Keep NumPy/Python transforms reproducible inside each loader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def raw_disparity_gauge_loss(prediction, reference, eps=1e-6):
    """Keep the relative-disparity gauge near DA2 without copying its pixels."""
    reference = reference.detach()
    pred_mean = prediction.mean(dim=(-2, -1))
    ref_mean = reference.mean(dim=(-2, -1))
    pred_std = prediction.std(dim=(-2, -1), unbiased=False)
    ref_std = reference.std(dim=(-2, -1), unbiased=False)
    normalizer = ref_std.clamp_min(eps)
    mean_loss = (pred_mean - ref_mean).abs() / normalizer
    std_loss = (pred_std - ref_std).abs() / normalizer
    return (mean_loss + std_loss).mean()


def summarize_trainable_parameters(model):
    summary = {
        "latent_prior": [],
        "prior_head": [],
        "encoder_lora": [],
        "plain_adapter": [],
        "base_head": [],
        "backbone": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("pretrained.") and any(
            token in name for token in (".lora_a.", ".lora_b.", ".condition_gate.")
        ):
            summary["encoder_lora"].append((name, param))
        elif name.startswith("latent_prior_encoder."):
            summary["latent_prior"].append((name, param))
        elif name.startswith("depth_head.plain_adapters."):
            summary["plain_adapter"].append((name, param))
        elif name.startswith("depth_head.global_mod.") or name.startswith("depth_head.deg_map_generator.") or name.startswith(
            "depth_head.prior_to_feat."
        ) or name == "depth_head.scalar_gate_logits":
            summary["prior_head"].append((name, param))
        elif name.startswith("depth_head."):
            summary["base_head"].append((name, param))
        elif name.startswith("pretrained."):
            summary["backbone"].append((name, param))
    return summary


def build_optimizer(model, args):
    groups = []
    summary = summarize_trainable_parameters(model)

    latent_prior_params = [param for _, param in summary["latent_prior"]]
    prior_head_params = [param for _, param in summary["prior_head"]]
    encoder_lora_params = [param for _, param in summary["encoder_lora"]]
    plain_adapter_params = [param for _, param in summary["plain_adapter"]]
    base_head_params = [param for _, param in summary["base_head"]]
    backbone_params = [param for _, param in summary["backbone"]]

    if latent_prior_params:
        groups.append({"params": latent_prior_params, "lr": args.prior_lr or args.lr, "name": "latent_prior"})
    if prior_head_params:
        groups.append({"params": prior_head_params, "lr": args.prior_head_lr or args.lr, "name": "prior_head"})
    if encoder_lora_params:
        groups.append({"params": encoder_lora_params, "lr": args.encoder_lora_lr or args.lr, "name": "encoder_lora"})
    if plain_adapter_params:
        groups.append({"params": plain_adapter_params, "lr": args.adapter_lr or args.lr, "name": "plain_adapter"})
    if base_head_params:
        groups.append({"params": base_head_params, "lr": args.head_lr or args.lr, "name": "base_head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr or args.lr, "name": "backbone"})
    if not groups:
        raise ValueError("No trainable parameters. Check the freeze and structure flags.")
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
    parser.add_argument("--encoder-lora-lr", default=0.0, type=float)
    parser.add_argument("--adapter-lr", default=0.0, type=float)
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
    parser.add_argument("--disable-global-prior", action="store_true")
    parser.add_argument("--disable-local-prior", action="store_true")
    parser.add_argument("--disable-fft-prior", action="store_true")
    parser.add_argument(
        "--disable-deg-map",
        action="store_true",
        help="Replace spatial degradation maps with four learned scalar gates",
    )
    parser.add_argument(
        "--deg-map-spatial-mean",
        action="store_true",
        help="Replace each generated map by its per-image spatial mean",
    )
    parser.add_argument("--plain-adapter", action="store_true")
    parser.add_argument("--adapter-hidden", default=256, type=int)
    parser.add_argument("--encoder-lora", action="store_true")
    parser.add_argument("--encoder-lora-mode", default="gated", choices=["plain", "gated"])
    parser.add_argument(
        "--encoder-lora-condition-source", default="fused", choices=["fused", "fft"]
    )
    parser.add_argument("--encoder-lora-rank", default=8, type=int)
    parser.add_argument("--encoder-lora-alpha", default=16.0, type=float)
    parser.add_argument("--encoder-lora-dropout", default=0.0, type=float)
    parser.add_argument("--encoder-lora-last-n-blocks", default=12, type=int)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-base-head", action="store_true")
    parser.add_argument("--freeze-latent-prior", action="store_true")
    parser.add_argument("--loss-mode", default="depthdive_relative", choices=["relative_metric", "depthdive_relative"])
    parser.add_argument("--l1-weight", default=0.5, type=float)
    parser.add_argument("--silog-weight", default=0.5, type=float)
    parser.add_argument("--metric-weight", default=1.0, type=float)
    parser.add_argument("--grad-weight", default=0.05, type=float)
    parser.add_argument(
        "--gauge-anchor-weight",
        default=0.0,
        type=float,
        help="Anchor raw disparity mean/std to frozen DA2 output; 0 disables it",
    )
    parser.add_argument(
        "--hole-geometry-weight",
        default=0.0,
        type=float,
        help="Preserve frozen-DA2 multi-scale disparity gradients inside true depth holes",
    )
    parser.add_argument(
        "--hole-geometry-scales",
        default="1,2,4",
        help="Comma-separated pixel offsets used by the affine-invariant hole geometry loss",
    )
    parser.add_argument(
        "--wat3r-manifest",
        default="",
        help="Offline Wat3R window manifest; empty keeps the original single-frame loader",
    )
    parser.add_argument("--wat3r-frame-stride", default=1, type=int)
    parser.add_argument("--wat3r-hole-weight", default=0.0, type=float)
    parser.add_argument("--wat3r-hole-grad-weight", default=0.25, type=float)
    parser.add_argument("--wat3r-mv-weight", default=0.0, type=float)
    parser.add_argument("--wat3r-confidence-quantile", default=0.6, type=float)
    parser.add_argument("--wat3r-relative-depth-threshold", default=0.05, type=float)
    parser.add_argument("--wat3r-min-align-pixels", default=100, type=int)
    parser.add_argument("--consistency-hardness-weight", default=0.0, type=float)
    parser.add_argument("--consistency-hardness-clamp-min", default=0.90, type=float)
    parser.add_argument("--consistency-hardness-clamp-max", default=1.10, type=float)
    parser.add_argument("--consistency-aug-prob", default=0.0, type=float)
    parser.add_argument("--consistency-blur-prob", default=0.30, type=float)
    parser.add_argument("--consistency-noise-prob", default=0.20, type=float)
    parser.add_argument("--consistency-noise-std", default=0.01, type=float)
    parser.add_argument("--warmup-steps", default=100, type=int)
    parser.add_argument("--min-lr-ratio", default=0.2, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-interval", default=20, type=int)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    parser.add_argument("--grad-clip", default=1.0, type=float, help="Max gradient norm; <= 0 disables clipping")
    parser.add_argument(
        "--eval-before-train",
        action="store_true",
        help="Run the legacy aligned validation once before the first optimizer step",
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log("latent_prior", logging.INFO, os.path.join(args.save_path, "train.log"))
    logger.propagate = 0
    logger.info(pprint.pformat(vars(args)))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Deterministic training enabled with seed=%d" % args.seed)

    prior_channels = tuple(int(x) for x in args.prior_channels.split(",") if x.strip())
    if len(prior_channels) != 4:
        raise ValueError("--prior-channels must contain exactly 4 comma-separated integers")
    hole_geometry_scales = tuple(int(x) for x in args.hole_geometry_scales.split(",") if x.strip())
    if not hole_geometry_scales or any(scale <= 0 for scale in hole_geometry_scales):
        raise ValueError("--hole-geometry-scales must contain positive comma-separated integers")
    if args.hole_geometry_weight < 0:
        raise ValueError("--hole-geometry-weight must be non-negative")
    if args.hole_geometry_weight > 0 and (not args.freeze_backbone or not args.freeze_base_head):
        logger.warning(
            "Hole geometry reference is only immutable when both backbone and base head are frozen"
        )
    if args.wat3r_hole_weight < 0 or args.wat3r_mv_weight < 0:
        raise ValueError("Wat3R loss weights must be non-negative")
    if not 0.0 <= args.wat3r_confidence_quantile <= 1.0:
        raise ValueError("--wat3r-confidence-quantile must be in [0, 1]")
    if args.wat3r_frame_stride <= 0:
        raise ValueError("--wat3r-frame-stride must be positive")
    use_wat3r = bool(args.wat3r_manifest)
    if (args.wat3r_hole_weight > 0 or args.wat3r_mv_weight > 0) and not use_wat3r:
        raise ValueError("Wat3R losses require --wat3r-manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_wat3r:
        train_set = FLSeaWat3R(
            args.train_list,
            args.wat3r_manifest,
            size=(args.img_size, args.img_size),
            frame_stride=args.wat3r_frame_stride,
        )
    else:
        train_set = FLSea(args.train_list, "train", size=(args.img_size, args.img_size))
    val_pairs = load_file_list(args.val_list)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_data_worker,
        generator=loader_generator,
    )
    logger.info(
        "Dataset loaded: train=%d samples, val=%d samples, train_iters_per_epoch=%d"
        % (len(train_set), len(val_pairs), len(train_loader))
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
        use_global_prior=not args.disable_global_prior,
        use_local_prior=not args.disable_local_prior,
        use_fft_prior=not args.disable_fft_prior,
        use_deg_map=not args.disable_deg_map,
        deg_map_spatial_mean=args.deg_map_spatial_mean,
        use_plain_adapter=args.plain_adapter,
        adapter_hidden=args.adapter_hidden,
        use_encoder_lora=args.encoder_lora,
        encoder_lora_mode=args.encoder_lora_mode,
        encoder_lora_condition_source=args.encoder_lora_condition_source,
        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
        encoder_lora_dropout=args.encoder_lora_dropout,
        encoder_lora_last_n_blocks=args.encoder_lora_last_n_blocks,
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
    model.configure_trainable(
        freeze_backbone=args.freeze_backbone,
        freeze_base_head=args.freeze_base_head,
        train_latent_prior=not args.freeze_latent_prior,
    )
    logger.info(
        "Prior structure: global=%s local=%s fft=%s spectral_map=%s deg_map=%s spatial_mean=%s plain_adapter=%s encoder_lora=%s lora_mode=%s condition_source=%s lora_blocks=%d"
        % (
            str(not args.disable_global_prior),
            str(not args.disable_local_prior),
            str(not args.disable_fft_prior),
            str(not args.disable_fft_prior and not args.disable_local_prior),
            str(not args.disable_deg_map),
            str(args.deg_map_spatial_mean),
            str(args.plain_adapter),
            str(args.encoder_lora),
            args.encoder_lora_mode,
            args.encoder_lora_condition_source,
            args.encoder_lora_last_n_blocks if args.encoder_lora else 0,
        )
    )

    if args.init_from:
        init_state = torch.load(args.init_from, map_location="cpu")
        init_state = init_state["model"] if isinstance(init_state, dict) and "model" in init_state else init_state
        model.load_state_dict(init_state, strict=False)
        logger.info(f"Loaded latent-prior checkpoint from {args.init_from}")

    param_summary = summarize_trainable_parameters(model)
    logger.info(
        "Trainable params | latent_prior=%d | prior_head=%d | encoder_lora=%d | plain_adapter=%d | base_head=%d | backbone=%d"
        % (
            _count_params([param for _, param in param_summary["latent_prior"]]),
            _count_params([param for _, param in param_summary["prior_head"]]),
            _count_params([param for _, param in param_summary["encoder_lora"]]),
            _count_params([param for _, param in param_summary["plain_adapter"]]),
            _count_params([param for _, param in param_summary["base_head"]]),
            _count_params([param for _, param in param_summary["backbone"]]),
        )
    )

    optimizer = build_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps, args.min_lr_ratio)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for group in optimizer.param_groups:
        logger.info(
            "Optimizer group %-12s lr=%.2e params=%d"
            % (group.get("name", "default"), group["lr"], _count_params(group["params"]))
        )
    logger.info(
        "Consistency branch: enabled=%s weight=%.4f aug_prob=%.4f"
        % (
            str(args.consistency_hardness_weight > 0 and args.consistency_aug_prob > 0),
            args.consistency_hardness_weight,
            args.consistency_aug_prob,
        )
    )
    logger.info(
        "Hole geometry branch: enabled=%s weight=%.4f scales=%s"
        % (
            str(args.hole_geometry_weight > 0),
            args.hole_geometry_weight,
            ",".join(str(scale) for scale in hole_geometry_scales),
        )
    )
    logger.info(
        "Wat3R privileged geometry: enabled=%s hole_weight=%.4f mv_weight=%.4f "
        "conf_quantile=%.3f frame_stride=%d"
        % (
            str(use_wat3r),
            args.wat3r_hole_weight,
            args.wat3r_mv_weight,
            args.wat3r_confidence_quantile,
            args.wat3r_frame_stride,
        )
    )
    logger.info("Numerics: amp=%s grad_clip=%.4f" % (str(amp_enabled), args.grad_clip))

    best_abs_rel = float("inf")
    best_d1 = 0.0
    global_step = 0

    if args.eval_before_train:
        model.eval()
        initial_metrics = evaluate_latent_prior(
            model,
            val_pairs,
            args.img_size,
            device,
            args.max_depth,
            logger,
        )
        if initial_metrics is None:
            raise RuntimeError("Pre-training validation found no valid samples")
        logger.info(
            "Pre-training baseline: "
            + ", ".join([f"{key}={value:.4f}" for key, value in initial_metrics.items()])
        )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, sample in enumerate(train_loader):
            if epoch == 0 and step == 0:
                logger.info("First training batch: %s" % ", ".join(sample["image_path"]))
            if use_wat3r:
                images = sample["images"].to(device, non_blocking=True)
                batch_size, num_views = images.shape[:2]
                image = images[:, 1]
                teacher_depth_views = sample["teacher_depth"].to(device, non_blocking=True)
                teacher_confidence_views = sample["teacher_confidence"].to(device, non_blocking=True)
                teacher_static_views = sample["teacher_static_mask"].to(device, non_blocking=True).bool()
                teacher_intrinsics = sample["intrinsics"].to(device, non_blocking=True)
                teacher_extrinsics = sample["extrinsics"].to(device, non_blocking=True)
            else:
                image = sample["image"].to(device, non_blocking=True)
                images = None
            depth = sample["depth"].to(device, non_blocking=True)
            observed_mask = sample["valid_mask"].to(device, non_blocking=True).bool()
            valid_mask = observed_mask & (depth >= args.min_depth) & (depth <= args.max_depth)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                need_base_reference = args.gauge_anchor_weight > 0 or args.hole_geometry_weight > 0
                model_input = images.flatten(0, 1) if use_wat3r else image
                if need_base_reference:
                    prediction_output, base_output = model(model_input, return_base=True)
                    if use_wat3r:
                        pred_disp_views = prediction_output.reshape(
                            batch_size, num_views, *prediction_output.shape[-2:]
                        )
                        base_disp_views = base_output.reshape(
                            batch_size, num_views, *base_output.shape[-2:]
                        )
                        pred_disp = pred_disp_views[:, 1]
                        base_disp = base_disp_views[:, 1]
                    else:
                        pred_disp, base_disp = prediction_output, base_output
                    if args.gauge_anchor_weight > 0:
                        loss_gauge = raw_disparity_gauge_loss(pred_disp, base_disp)
                    else:
                        loss_gauge = pred_disp.new_zeros(())
                    if args.hole_geometry_weight > 0:
                        loss_hole_geometry = hole_geometry_preservation_loss(
                            pred_disp,
                            base_disp,
                            observed_mask,
                            scales=hole_geometry_scales,
                        )
                    else:
                        loss_hole_geometry = pred_disp.new_zeros(())
                else:
                    prediction_output = model(model_input)
                    if use_wat3r:
                        pred_disp_views = prediction_output.reshape(
                            batch_size, num_views, *prediction_output.shape[-2:]
                        )
                        pred_disp = pred_disp_views[:, 1]
                    else:
                        pred_disp = prediction_output
                    loss_gauge = pred_disp.new_zeros(())
                    loss_hole_geometry = pred_disp.new_zeros(())

                if use_wat3r:
                    teacher_reliable_views = build_teacher_reliability_mask(
                        teacher_depth_views.flatten(0, 1),
                        teacher_confidence_views.flatten(0, 1),
                        teacher_static_views.flatten(0, 1),
                        confidence_quantile=args.wat3r_confidence_quantile,
                    ).reshape(batch_size, num_views, *teacher_depth_views.shape[-2:])
                    if args.wat3r_hole_weight > 0:
                        wat3r_hole = wat3r_hole_distillation_loss(
                            pred_disp,
                            teacher_depth_views[:, 1],
                            depth,
                            observed_mask,
                            teacher_reliable_views[:, 1],
                            gradient_scales=hole_geometry_scales,
                            gradient_weight=args.wat3r_hole_grad_weight,
                            min_align_pixels=args.wat3r_min_align_pixels,
                        )
                    else:
                        wat3r_hole = {
                            "loss": pred_disp.new_zeros(()),
                            "value": pred_disp.new_zeros(()),
                            "gradient": pred_disp.new_zeros(()),
                            "coverage": pred_disp.new_zeros(()),
                        }
                    if args.wat3r_mv_weight > 0:
                        wat3r_mv = wat3r_multiview_geometry_loss(
                            pred_disp_views,
                            teacher_depth_views,
                            teacher_reliable_views,
                            teacher_intrinsics,
                            teacher_extrinsics,
                            relative_depth_threshold=args.wat3r_relative_depth_threshold,
                            min_align_pixels=args.wat3r_min_align_pixels,
                        )
                    else:
                        wat3r_mv = {
                            "loss": pred_disp.new_zeros(()),
                            "coverage": pred_disp.new_zeros(()),
                        }
                else:
                    wat3r_hole = {
                        "loss": pred_disp.new_zeros(()),
                        "value": pred_disp.new_zeros(()),
                        "gradient": pred_disp.new_zeros(()),
                        "coverage": pred_disp.new_zeros(()),
                    }
                    wat3r_mv = {
                        "loss": pred_disp.new_zeros(()),
                        "coverage": pred_disp.new_zeros(()),
                    }
                sup_weight = valid_mask.float()

                if args.consistency_hardness_weight > 0 and random.random() < args.consistency_aug_prob:
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
                    + args.gauge_anchor_weight * loss_gauge
                    + args.hole_geometry_weight * loss_hole_geometry
                    + args.wat3r_hole_weight * wat3r_hole["loss"]
                    + args.wat3r_mv_weight * wat3r_mv["loss"]
                )

            if not torch.isfinite(loss):
                logger.warning(
                    "Skip non-finite loss at epoch=%d iter=%d: %s" % (epoch + 1, step, str(loss.item()))
                )
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            global_step += 1

            if step % args.log_interval == 0:
                logger.info(
                    "Epoch %02d Iter %04d/%04d Step %06d Loss=%.6f L1=%.6f SiLog=%.6f Metric=%.6f Grad=%.6f Gauge=%.6f HoleGeom=%.6f WatHole=%.6f WatHoleCov=%.4f WatMV=%.6f WatMVCov=%.4f HoleRatio=%.4f HardMean=%.6f LR=%.2e"
                    % (
                        epoch + 1,
                        step,
                        len(train_loader),
                        global_step,
                        loss.item(),
                        loss_l1.item(),
                        loss_silog.item(),
                        loss_metric.item(),
                        loss_grad.item(),
                        loss_gauge.item(),
                        loss_hole_geometry.item(),
                        wat3r_hole["loss"].item(),
                        wat3r_hole["coverage"].item(),
                        wat3r_mv["loss"].item(),
                        wat3r_mv["coverage"].item(),
                        (~observed_mask).float().mean().item(),
                        cons_map.mean().item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        mean_train_loss = running_loss / max(len(train_loader), 1)
        logger.info("Epoch %02d finished | mean_train_loss=%.6f" % (epoch + 1, mean_train_loss))
        if model.use_global_prior:
            global_mod = model.depth_head.global_mod
            condition_norm = global_mod.to_gamma[2].weight.norm() + global_mod.to_beta[2].weight.norm()
            logger.info("Global conditional weight norm: %.6f" % condition_norm.item())
        if model.depth_head.scalar_gate_logits is not None:
            gates = torch.sigmoid(model.depth_head.scalar_gate_logits).detach().cpu().tolist()
            logger.info("Learned scalar gates: %s" % ", ".join(f"{gate:.4f}" for gate in gates))
        if model.use_encoder_lora:
            lora_b_norm = sum(
                module.lora_b.weight.detach().float().norm().item()
                for module in model.encoder_lora_modules
            )
            condition_norm = sum(
                module.condition_gate.weight.detach().float().norm().item()
                for module in model.encoder_lora_modules
                if module.condition_gate is not None
            )
            logger.info(
                "Encoder LoRA norms: lora_b_sum=%.6f condition_gate_sum=%.6f"
                % (lora_b_norm, condition_norm)
            )
        model.eval()
        metrics = evaluate_latent_prior(
            model,
            val_pairs,
            args.img_size,
            device,
            args.max_depth,
            logger,
        )
        save_checkpoint(model, os.path.join(args.save_path, "latest.pth"), args, epoch=epoch, metrics=metrics)
        if metrics is None:
            logger.info("Validation skipped: no valid samples")
            continue

        logger.info("Validation: " + ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()]))
        if metrics["d1"] > best_d1:
            best_d1 = metrics["d1"]
            save_checkpoint(model, os.path.join(args.save_path, "best_d1.pth"), args, epoch=epoch, metrics=metrics)
            logger.info("Saved new best_d1 checkpoint: best_d1=%.4f" % best_d1)
        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            save_checkpoint(model, os.path.join(args.save_path, "best_abs_rel.pth"), args, epoch=epoch, metrics=metrics)
            logger.info("Saved new best_abs_rel checkpoint: best_abs_rel=%.4f" % best_abs_rel)


if __name__ == "__main__":
    main()
