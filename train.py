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

from dataset.flsea import FLSea
from models.depth_anything_lora import DepthAnythingLoRA, extract_lora_state_dict
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


def weighted_mean(loss_map, weight, eps=1e-6):
    denom = weight.sum().clamp_min(eps)
    return (loss_map * weight).sum() / denom


def _as_bhw(x):
    return x.squeeze(1) if x.dim() == 4 else x


def _weighted_stats(values, weight=None, eps=1e-6):
    if weight is None:
        center = values.mean()
        scale = torch.mean(torch.abs(values - center)).clamp_min(eps)
        return center, scale
    w = weight.float().clamp_min(0.0)
    if w.sum() < eps:
        center = values.mean()
        scale = torch.mean(torch.abs(values - center)).clamp_min(eps)
        return center, scale
    w = w / (w.sum() + eps)
    center = (w * values).sum()
    scale = (w * torch.abs(values - center)).sum().clamp_min(eps)
    return center, scale


def _align_scale_shift(pred, target, mask, weight=None, eps=1e-6):
    p = pred[mask].float()
    t = target[mask].float()
    if p.numel() < 32:
        return None, None
    if weight is None:
        w = torch.ones_like(p) / p.numel()
    else:
        w = weight[mask].float().clamp_min(0.0)
        if w.sum() < eps:
            return None, None
        w = w / (w.sum() + eps)
    p_mean = (w * p).sum()
    t_mean = (w * t).sum()
    var = (w * (p - p_mean) ** 2).sum()
    if var < eps:
        return None, None
    scale = (w * (p - p_mean) * (t - t_mean)).sum() / (var + eps)
    shift = t_mean - scale * p_mean
    return scale, shift


def normalized_disparity_shape_loss(pred_disp, target_depth, mask, weight=None, eps=1e-6):
    pred = _as_bhw(pred_disp)
    gt = _as_bhw(target_depth)
    valid = _as_bhw(mask).bool()
    if weight is not None:
        weight = _as_bhw(weight)

    target_disp = 1.0 / gt.clamp_min(eps)
    l1_losses = []
    si_losses = []
    for batch_idx in range(pred.shape[0]):
        cur_mask = valid[batch_idx]
        if cur_mask.sum() < 32:
            continue
        cur_weight = weight[batch_idx][cur_mask] if weight is not None else None
        p = pred[batch_idx][cur_mask].float()
        t = target_disp[batch_idx][cur_mask].float()
        p_center, p_scale = _weighted_stats(p, cur_weight, eps=eps)
        t_center, t_scale = _weighted_stats(t, cur_weight, eps=eps)
        diff = (p - p_center) / p_scale - (t - t_center) / t_scale
        if cur_weight is not None:
            w = cur_weight.float().clamp_min(0.0)
            if w.sum() > eps:
                w = w / w.mean().clamp_min(eps)
                l1 = (torch.abs(diff) * w).mean()
                diff_for_si = diff * torch.sqrt(w.clamp_min(0.0))
            else:
                l1 = torch.abs(diff).mean()
                diff_for_si = diff
        else:
            l1 = torch.abs(diff).mean()
            diff_for_si = diff
        si = torch.sqrt((diff_for_si ** 2).mean() - 0.5 * (diff_for_si.mean() ** 2) + eps)
        l1_losses.append(l1)
        si_losses.append(si)
    if not l1_losses:
        zero = pred.sum() * 0.0
        return zero, zero
    return torch.stack(l1_losses).mean(), torch.stack(si_losses).mean()


def depthdive_l1_silog_loss(pred_disp, target_depth, mask, weight=None, max_depth=40.0, eps=1e-6):
    pred = _as_bhw(pred_disp)
    gt = _as_bhw(target_depth)
    valid = _as_bhw(mask).bool()
    if weight is not None:
        weight = _as_bhw(weight)

    target_disp = 1.0 / gt.clamp_min(eps)
    l1_losses = []
    silog_losses = []
    for batch_idx in range(pred.shape[0]):
        cur_mask = valid[batch_idx]
        cur_weight = weight[batch_idx] if weight is not None else None
        scale, shift = _align_scale_shift(pred[batch_idx], target_disp[batch_idx], cur_mask, weight=cur_weight, eps=eps)
        if scale is None:
            continue
        aligned_disp = (scale.detach() * pred[batch_idx] + shift.detach()).clamp_min(1.0 / max(max_depth, eps))
        pred_depth = (1.0 / aligned_disp).clamp(max=max_depth)
        gt_depth = gt[batch_idx].clamp_min(eps)
        valid_pred = cur_mask & torch.isfinite(pred_depth) & (pred_depth > 0)
        if valid_pred.sum() < 32:
            continue
        abs_depth = torch.abs(pred_depth - gt_depth)
        log_diff = torch.log(pred_depth.clamp_min(eps)) - torch.log(gt_depth)
        diff = log_diff[valid_pred]
        silog = torch.sqrt((diff ** 2).mean() - 0.5 * (diff.mean() ** 2) + eps)
        if cur_weight is not None:
            l1 = weighted_mean(abs_depth, cur_weight.float().clamp_min(0.0) * valid_pred.float())
        else:
            l1 = abs_depth[valid_pred].mean()
        l1_losses.append(l1)
        silog_losses.append(silog)
    if not l1_losses:
        return normalized_disparity_shape_loss(pred_disp, target_depth, mask, weight=weight, eps=eps)
    return torch.stack(l1_losses).mean(), torch.stack(silog_losses).mean()


def aligned_depth_metric_loss(pred_disp, target_depth, mask, weight=None, max_depth=40.0, eps=1e-6):
    pred = _as_bhw(pred_disp)
    gt = _as_bhw(target_depth)
    valid = _as_bhw(mask).bool()
    if weight is not None:
        weight = _as_bhw(weight)

    target_disp = 1.0 / gt.clamp_min(eps)
    losses = []
    for batch_idx in range(pred.shape[0]):
        cur_mask = valid[batch_idx]
        cur_weight = weight[batch_idx] if weight is not None else None
        scale, shift = _align_scale_shift(pred[batch_idx], target_disp[batch_idx], cur_mask, weight=cur_weight, eps=eps)
        if scale is None:
            continue
        aligned_disp = (scale.detach() * pred[batch_idx] + shift.detach()).clamp_min(1.0 / max(max_depth, eps))
        pred_depth = 1.0 / aligned_disp
        gt_depth = gt[batch_idx].clamp_min(eps)
        valid_pred = cur_mask & torch.isfinite(pred_depth) & (pred_depth > 0)
        if valid_pred.sum() < 32:
            continue
        abs_rel_map = torch.abs(pred_depth - gt_depth) / gt_depth
        log_diff = torch.log(pred_depth.clamp_min(eps)) - torch.log(gt_depth)
        log_l1_map = torch.abs(log_diff)
        diff = log_diff[valid_pred]
        silog = torch.sqrt((diff ** 2).mean() - 0.5 * (diff.mean() ** 2) + eps)
        pixel_loss = abs_rel_map + 0.5 * log_l1_map
        if cur_weight is not None:
            loss = weighted_mean(pixel_loss, cur_weight.float().clamp_min(0.0) * valid_pred.float()) + 0.5 * silog
        else:
            loss = pixel_loss[valid_pred].mean() + 0.5 * silog
        losses.append(loss)
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def affine_disparity_gradient_loss(pred_disp, target_depth, mask, weight=None, eps=1e-6):
    pred = _as_bhw(pred_disp)
    gt = _as_bhw(target_depth)
    valid = _as_bhw(mask).bool()
    if weight is not None:
        weight = _as_bhw(weight)
    target_disp = 1.0 / gt.clamp_min(eps)
    aligned = []
    for batch_idx in range(pred.shape[0]):
        cur_mask = valid[batch_idx]
        cur_weight = weight[batch_idx] if weight is not None else None
        scale, shift = _align_scale_shift(pred[batch_idx], target_disp[batch_idx], cur_mask, weight=cur_weight, eps=eps)
        if scale is None:
            aligned.append(pred[batch_idx : batch_idx + 1] * 0.0)
        else:
            aligned.append((scale.detach() * pred[batch_idx : batch_idx + 1] + shift.detach()))
    aligned = torch.cat(aligned, dim=0)
    diff = (aligned - target_disp) * valid.float()
    dx_mask = valid[:, :, 1:] & valid[:, :, :-1]
    dy_mask = valid[:, 1:, :] & valid[:, :-1, :]
    gx = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    gy = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    if weight is not None:
        wx = 0.5 * (weight[:, :, 1:] + weight[:, :, :-1])
        wy = 0.5 * (weight[:, 1:, :] + weight[:, :-1, :])
        gx = gx * wx / wx[dx_mask].mean().clamp_min(eps) if dx_mask.any() else gx
        gy = gy * wy / wy[dy_mask].mean().clamp_min(eps) if dy_mask.any() else gy
    loss_x = gx[dx_mask].mean() if dx_mask.any() else diff.sum() * 0.0
    loss_y = gy[dy_mask].mean() if dy_mask.any() else diff.sum() * 0.0
    return loss_x + loss_y


def denormalize_imagenet(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image.float() * std + mean).clamp(0.0, 1.0)


def normalize_imagenet(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image.float().clamp(0.0, 1.0) - mean) / std


def random_underwater_consistency_augment(
    image,
    gamma_min=0.80,
    gamma_max=1.25,
    brightness=0.10,
    contrast=0.15,
    saturation=0.15,
    channel_scale=0.10,
    blur_prob=0.30,
    noise_prob=0.20,
    noise_std=0.01,
):
    with torch.amp.autocast(device_type=image.device.type, enabled=False):
        x = denormalize_imagenet(image).float()
        batch = x.shape[0]
        gamma = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(float(gamma_min), float(gamma_max))
        x = x.clamp_min(1e-4).pow(gamma)

        gain = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(1.0 - float(brightness), 1.0 + float(brightness))
        x = x * gain

        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast_scale = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(1.0 - float(contrast), 1.0 + float(contrast))
        x = (x - mean) * contrast_scale + mean

        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
        sat = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(1.0 - float(saturation), 1.0 + float(saturation))
        x = gray + sat * (x - gray)

        channel = torch.empty(batch, 3, 1, 1, device=x.device).uniform_(1.0 - float(channel_scale), 1.0 + float(channel_scale))
        x = x * channel

        if blur_prob > 0:
            blur_mask = (torch.rand(batch, 1, 1, 1, device=x.device) < float(blur_prob)).float()
            blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            x = x * (1.0 - blur_mask) + blurred * blur_mask

        if noise_prob > 0 and noise_std > 0:
            noise_mask = (torch.rand(batch, 1, 1, 1, device=x.device) < float(noise_prob)).float()
            sigma = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(0.5 * float(noise_std), float(noise_std))
            x = x + torch.randn_like(x) * sigma * noise_mask

        return normalize_imagenet(x)


def _normalize_error_map(err, mask=None, eps=1e-6):
    err = _as_bhw(err)
    out = []
    for batch_idx in range(err.shape[0]):
        values = err[batch_idx]
        cur_mask = mask[batch_idx].bool() if mask is not None else torch.ones_like(values, dtype=torch.bool)
        valid_values = values[cur_mask]
        if valid_values.numel() < 16:
            valid_values = values.flatten()
        scale = torch.quantile(valid_values.float(), 0.90).clamp_min(eps)
        out.append((values / scale).clamp(0.0, 2.0) * 0.5)
    return torch.stack(out, dim=0)


def _softmax_residual_weight(err, mask=None, min_value=0.5, max_value=1.5, eps=1e-7):
    err = _as_bhw(err)
    batch, height, width = err.shape
    flat = err.flatten(1).float()
    if mask is not None:
        flat_mask = mask.flatten(1).bool()
        flat = torch.where(flat_mask, flat, flat.new_full(flat.shape, -1e4))
    weight = F.softmax(flat, dim=-1) * float(height * width) + float(eps)
    weight = weight.view(batch, height, width)
    if mask is not None:
        mask_f = mask.float()
        denom = (weight * mask_f).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        count = mask_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        weight = weight * count / denom
        weight = torch.where(mask, weight, torch.ones_like(weight))
    else:
        weight = weight / weight.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return weight.clamp(float(min_value), float(max_value))


def depth_anything_consistency_hardness_weight(
    pred_disp,
    aug_disp,
    mask,
    alpha=0.08,
    clamp_min=0.90,
    clamp_max=1.10,
    eps=1e-6,
):
    if alpha <= 0:
        zero = pred_disp.sum() * 0.0
        return mask.float(), zero

    pred = _as_bhw(pred_disp).float().detach()
    aug = _as_bhw(aug_disp).float().detach()
    valid = _as_bhw(mask).bool()
    errs = []
    for batch_idx in range(pred.shape[0]):
        cur_mask = valid[batch_idx]
        if cur_mask.sum() < 32:
            cur_mask = torch.ones_like(cur_mask, dtype=torch.bool)
        scale, shift = _align_scale_shift(aug[batch_idx], pred[batch_idx], cur_mask, eps=eps)
        aligned_aug = aug[batch_idx] if scale is None else scale.detach() * aug[batch_idx] + shift.detach()
        errs.append(torch.abs(aligned_aug - pred[batch_idx]))
    err = torch.stack(errs, dim=0)
    soft = _softmax_residual_weight(err.detach(), valid, min_value=0.5, max_value=1.5)
    weight = (1.0 + float(alpha) * (soft - 1.0)).clamp(float(clamp_min), float(clamp_max))
    weight = torch.where(valid, weight, torch.ones_like(weight))
    return weight.detach(), _normalize_error_map(err, valid).detach()


def evaluate(model, loader, device, min_depth, max_depth):
    model.eval()
    metrics = {key: 0.0 for key in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]}
    count = 0
    with torch.no_grad():
        for sample in loader:
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()

            pred = model(image)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[:, 0]

            valid = valid_mask & (depth >= min_depth) & (depth <= max_depth)
            for batch_idx in range(pred.shape[0]):
                cur_valid = valid[batch_idx]
                if cur_valid.sum() < 10:
                    continue
                cur_metrics = eval_depth(
                    pred[batch_idx][cur_valid].clamp_min(min_depth),
                    depth[batch_idx][cur_valid].clamp_min(min_depth),
                )
                for key, value in cur_metrics.items():
                    metrics[key] += value
                count += 1
    if count == 0:
        return None
    return {key: value / count for key, value in metrics.items()}


def save_checkpoint(model, path, args, epoch=None, metrics=None):
    torch.save(
        {
            "model": extract_lora_state_dict(model),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def build_optimizer(model, args):
    encoder_params = []
    decoder_params = []
    style_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("style_encoder."):
            style_params.append(param)
        elif "depth_head" in name and ".lora_" in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr or args.lr})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": args.decoder_lr or args.lr})
    if style_params:
        groups.append({"params": style_params, "lr": args.style_lr or args.lr})
    return AdamW(groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)


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


def main():
    parser = argparse.ArgumentParser(description="Clean FLSea baseline for Depth Anything V2")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--encoder-lr", default=0.0, type=float)
    parser.add_argument("--decoder-lr", default=0.0, type=float)
    parser.add_argument("--style-lr", default=0.0, type=float)
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--init-from", default="")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--seed", default=42, type=int)
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
    logger = init_log("clean_baseline", logging.INFO)
    logger.propagate = 0
    logger.info(pprint.pformat(vars(args)))
    set_seed(args.seed)

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

    if args.init_from:
        init_state = torch.load(args.init_from, map_location="cpu")
        init_state = init_state["model"] if isinstance(init_state, dict) and "model" in init_state else init_state
        model.load_state_dict(init_state, strict=False)
        logger.info(f"Loaded adaptation checkpoint from {args.init_from}")

    optimizer = build_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps, args.min_lr_ratio)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_abs_rel = float("inf")
    best_d1 = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, sample in enumerate(train_loader):
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True).bool()
            valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_disp = model(image)
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

            running_loss += loss.item()
            global_step += 1

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

