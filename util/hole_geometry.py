import torch


def _masked_standardize(disparity, mask, eps=1e-6):
    """Remove per-image affine gauge using only observed FLSea pixels."""
    weight = mask.to(disparity.dtype)
    count = weight.sum(dim=(-2, -1), keepdim=True)
    fallback = count < 2

    safe_count = count.clamp_min(1.0)
    mean = (disparity * weight).sum(dim=(-2, -1), keepdim=True) / safe_count
    variance = ((disparity - mean).square() * weight).sum(dim=(-2, -1), keepdim=True) / safe_count

    full_mean = disparity.mean(dim=(-2, -1), keepdim=True)
    full_variance = disparity.var(dim=(-2, -1), keepdim=True, unbiased=False)
    mean = torch.where(fallback, full_mean, mean)
    variance = torch.where(fallback, full_variance, variance)
    return (disparity - mean) / variance.clamp_min(eps).sqrt()


def hole_geometry_preservation_loss(
    prediction,
    reference,
    observed_mask,
    scales=(1, 2, 4),
    charbonnier_eps=1e-3,
):
    """Preserve frozen-DA2 local geometry inside true FLSea depth holes.

    The loss compares multi-scale disparity differences after independently
    removing the affine gauge of prediction and reference. It therefore keeps
    local edges and ordinal structure without treating DA2 values as metric GT.
    Only pixel pairs whose two endpoints are both in a true depth hole are used.
    """
    if prediction.shape != reference.shape or prediction.shape != observed_mask.shape:
        raise ValueError("prediction, reference, and observed_mask must have identical shapes")

    reference = reference.detach()
    prediction_norm = _masked_standardize(prediction, observed_mask)
    reference_norm = _masked_standardize(reference, observed_mask)
    hole_mask = ~observed_mask.bool()

    total = prediction.new_zeros(())
    total_weight = prediction.new_zeros(())
    for scale in scales:
        if scale <= 0:
            raise ValueError("hole geometry scales must be positive integers")
        if scale >= prediction.shape[-1] or scale >= prediction.shape[-2]:
            continue

        pred_dx = prediction_norm[..., scale:] - prediction_norm[..., :-scale]
        ref_dx = reference_norm[..., scale:] - reference_norm[..., :-scale]
        mask_dx = hole_mask[..., scale:] & hole_mask[..., :-scale]

        pred_dy = prediction_norm[..., scale:, :] - prediction_norm[..., :-scale, :]
        ref_dy = reference_norm[..., scale:, :] - reference_norm[..., :-scale, :]
        mask_dy = hole_mask[..., scale:, :] & hole_mask[..., :-scale, :]

        scale_weight = 1.0 / float(scale)
        for residual, pair_mask in ((pred_dx - ref_dx, mask_dx), (pred_dy - ref_dy, mask_dy)):
            pair_weight = pair_mask.to(residual.dtype) * scale_weight
            robust_error = torch.sqrt(residual.square() + charbonnier_eps**2) - charbonnier_eps
            total = total + (robust_error * pair_weight).sum()
            total_weight = total_weight + pair_weight.sum()

    if total_weight.item() == 0:
        return prediction.sum() * 0.0
    return total / total_weight
