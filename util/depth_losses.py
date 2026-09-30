"""Masked relative-depth objectives for dense and sparse supervision."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def _as_bhw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"expected BxHxW or Bx1xHxW, got {tuple(tensor.shape)}")
    return tensor


def _masked_standardize(
    values: torch.Tensor, mask: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        valid = mask[batch_index]
        if valid.sum() < 2:
            continue
        selected = values[batch_index][valid]
        median = selected.median()
        scale = (selected - median).abs().mean().clamp_min(eps)
        normalized[batch_index] = (values[batch_index] - median) / scale
    return normalized


def scale_shift_invariant_inverse_depth_loss(
    prediction: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 loss after independently normalizing predicted and target disparity."""

    prediction = _as_bhw(prediction).float()
    target_depth = _as_bhw(target_depth).float()
    valid_mask = _as_bhw(valid_mask).bool()
    valid_mask = valid_mask & torch.isfinite(target_depth) & (target_depth > 0)
    valid_mask = valid_mask & torch.isfinite(prediction)

    target_inverse = torch.where(
        valid_mask, target_depth.clamp_min(eps).reciprocal(), torch.zeros_like(target_depth)
    )
    pred_norm = _masked_standardize(prediction, valid_mask, eps)
    target_norm = _masked_standardize(target_inverse, valid_mask, eps)

    if not valid_mask.any():
        return prediction.sum() * 0.0
    return (pred_norm - target_norm).abs()[valid_mask].mean()


def _gradient_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    losses = []
    horizontal_mask = mask[:, :, 1:] & mask[:, :, :-1]
    vertical_mask = mask[:, 1:, :] & mask[:, :-1, :]

    if horizontal_mask.any():
        pred_dx = prediction[:, :, 1:] - prediction[:, :, :-1]
        target_dx = target[:, :, 1:] - target[:, :, :-1]
        losses.append((pred_dx - target_dx).abs()[horizontal_mask].mean())
    if vertical_mask.any():
        pred_dy = prediction[:, 1:, :] - prediction[:, :-1, :]
        target_dy = target[:, 1:, :] - target[:, :-1, :]
        losses.append((pred_dy - target_dy).abs()[vertical_mask].mean())

    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def multiscale_gradient_loss(
    prediction: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    scales: Iterable[int] = (1, 2, 4, 8),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient matching on normalized inverse depth at multiple scales."""

    prediction = _as_bhw(prediction).float()
    target_depth = _as_bhw(target_depth).float()
    mask = _as_bhw(valid_mask).bool()
    mask = mask & torch.isfinite(target_depth) & (target_depth > 0)
    mask = mask & torch.isfinite(prediction)
    target_inverse = torch.where(
        mask, target_depth.clamp_min(eps).reciprocal(), torch.zeros_like(target_depth)
    )

    prediction = _masked_standardize(prediction, mask, eps)
    target_inverse = _masked_standardize(target_inverse, mask, eps)
    losses = []
    for scale in scales:
        if scale <= 0:
            raise ValueError("gradient scales must be positive")
        scaled_pred = prediction[:, ::scale, ::scale]
        scaled_target = target_inverse[:, ::scale, ::scale]
        scaled_mask = mask[:, ::scale, ::scale]
        losses.append(_gradient_l1(scaled_pred, scaled_target, scaled_mask))
    return torch.stack(losses).mean()


def dense_relative_depth_loss(
    prediction: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    gradient_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    data = scale_shift_invariant_inverse_depth_loss(
        prediction, target_depth, valid_mask
    )
    gradient = multiscale_gradient_loss(prediction, target_depth, valid_mask)
    return {
        "loss": data + gradient_weight * gradient,
        "data_loss": data,
        "gradient_loss": gradient,
    }
