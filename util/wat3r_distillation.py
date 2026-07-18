import torch
import torch.nn.functional as F


def _as_bhw(tensor):
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(tensor.shape)}")
    return tensor


def align_disparity_scale_shift(
    prediction,
    target,
    mask,
    min_pixels=100,
    eps=1e-6,
    detach_solution=True,
):
    """Align a relative disparity to a target disparity with per-image least squares."""
    prediction = _as_bhw(prediction).float()
    target = _as_bhw(target).float()
    mask = _as_bhw(mask).bool()
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical spatial shapes")

    weight = mask.to(prediction.dtype)
    count = weight.sum(dim=(-2, -1))
    a00 = (weight * prediction.square()).sum(dim=(-2, -1))
    a01 = (weight * prediction).sum(dim=(-2, -1))
    a11 = count
    b0 = (weight * prediction * target).sum(dim=(-2, -1))
    b1 = (weight * target).sum(dim=(-2, -1))
    determinant = a00 * a11 - a01.square()
    valid = (count >= float(min_pixels)) & (determinant.abs() > eps)

    safe_det = torch.where(valid, determinant, torch.ones_like(determinant))
    scale = (a11 * b0 - a01 * b1) / safe_det
    shift = (-a01 * b0 + a00 * b1) / safe_det
    scale = torch.where(valid, scale, torch.ones_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))
    if detach_solution:
        scale = scale.detach()
        shift = shift.detach()
    aligned = scale[:, None, None] * prediction + shift[:, None, None]
    return aligned, valid, scale, shift


def build_teacher_reliability_mask(
    teacher_depth,
    teacher_confidence,
    static_mask,
    confidence_quantile=0.6,
):
    """Select finite, static teacher pixels above a per-frame confidence quantile."""
    if not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must be in [0, 1]")
    teacher_depth = _as_bhw(teacher_depth)
    teacher_confidence = _as_bhw(teacher_confidence)
    static_mask = _as_bhw(static_mask).bool()
    valid = (
        torch.isfinite(teacher_depth)
        & (teacher_depth > 0)
        & torch.isfinite(teacher_confidence)
        & static_mask
    )
    flattened_conf = teacher_confidence.masked_fill(~valid, float("nan")).flatten(1)
    cutoff = torch.nanquantile(flattened_conf.float(), confidence_quantile, dim=1)
    cutoff = torch.where(torch.isfinite(cutoff), cutoff, torch.full_like(cutoff, float("inf")))
    return valid & (teacher_confidence >= cutoff[:, None, None].to(teacher_confidence.dtype))


def wat3r_hole_distillation_loss(
    prediction_disparity,
    teacher_depth,
    gt_depth,
    observed_mask,
    teacher_reliable_mask,
    gradient_scales=(1, 2, 4),
    gradient_weight=0.25,
    min_align_pixels=100,
    charbonnier_eps=1e-3,
):
    """Distill Wat3R geometry only into reliable FLSea holes.

    Teacher and student disparities are independently aligned to sparse metric GT.
    This removes their unrelated affine gauges while keeping GT as the authority.
    """
    prediction_disparity = _as_bhw(prediction_disparity)
    teacher_depth = _as_bhw(teacher_depth)
    gt_depth = _as_bhw(gt_depth)
    observed_mask = _as_bhw(observed_mask).bool()
    teacher_reliable_mask = _as_bhw(teacher_reliable_mask).bool()

    gt_valid = observed_mask & torch.isfinite(gt_depth) & (gt_depth > 0)
    teacher_valid = teacher_reliable_mask & torch.isfinite(teacher_depth) & (teacher_depth > 0)
    teacher_disparity = torch.where(
        teacher_valid,
        teacher_depth.clamp_min(1e-6).reciprocal(),
        torch.zeros_like(teacher_depth),
    )
    gt_disparity = torch.where(
        gt_valid,
        gt_depth.clamp_min(1e-6).reciprocal(),
        torch.zeros_like(gt_depth),
    )

    aligned_prediction, valid_student, _, _ = align_disparity_scale_shift(
        prediction_disparity,
        gt_disparity,
        gt_valid,
        min_pixels=min_align_pixels,
    )
    aligned_teacher, valid_teacher, _, _ = align_disparity_scale_shift(
        teacher_disparity,
        gt_disparity,
        gt_valid & teacher_valid,
        min_pixels=min_align_pixels,
    )
    valid_image = valid_student & valid_teacher
    hole_mask = (~observed_mask) & teacher_valid & valid_image[:, None, None]

    residual = aligned_prediction - aligned_teacher.detach()
    robust = torch.sqrt(residual.square() + charbonnier_eps**2) - charbonnier_eps
    weight = hole_mask.to(robust.dtype)
    value_loss = (robust * weight).sum() / weight.sum().clamp_min(1.0)

    gradient_total = prediction_disparity.new_zeros(())
    gradient_count = prediction_disparity.new_zeros(())
    for scale in gradient_scales:
        if scale <= 0:
            raise ValueError("gradient scales must be positive")
        if scale >= residual.shape[-2] or scale >= residual.shape[-1]:
            continue
        dx = residual[..., scale:] - residual[..., :-scale]
        mask_x = hole_mask[..., scale:] & hole_mask[..., :-scale]
        dy = residual[..., scale:, :] - residual[..., :-scale, :]
        mask_y = hole_mask[..., scale:, :] & hole_mask[..., :-scale, :]
        scale_weight = 1.0 / float(scale)
        for difference, pair_mask in ((dx, mask_x), (dy, mask_y)):
            pair_weight = pair_mask.to(difference.dtype) * scale_weight
            pair_error = torch.sqrt(difference.square() + charbonnier_eps**2) - charbonnier_eps
            gradient_total = gradient_total + (pair_error * pair_weight).sum()
            gradient_count = gradient_count + pair_weight.sum()
    gradient_loss = gradient_total / gradient_count.clamp_min(1.0)
    loss = value_loss + float(gradient_weight) * gradient_loss
    return {
        "loss": loss,
        "value": value_loss,
        "gradient": gradient_loss,
        "coverage": hole_mask.float().mean(),
        "pixels": hole_mask.sum(),
    }


def _sample_map(values, grid, mode="bilinear"):
    batch, views, height, width = values.shape
    sampled = F.grid_sample(
        values.reshape(batch * views, 1, height, width).float(),
        grid.reshape(batch * views, height, width, 2).float(),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0].reshape(batch, views, height, width)


def wat3r_multiview_geometry_loss(
    prediction_disparity,
    teacher_depth,
    teacher_reliable_mask,
    intrinsics,
    extrinsics,
    relative_depth_threshold=0.05,
    min_align_pixels=100,
    charbonnier_eps=1e-3,
):
    """Transfer a Wat3R window's geometry to independent monocular predictions."""
    if prediction_disparity.ndim != 4:
        raise ValueError("prediction_disparity must be [B,V,H,W]")
    batch, views, height, width = prediction_disparity.shape
    if views < 2:
        return {
            "loss": prediction_disparity.sum() * 0.0,
            "coverage": prediction_disparity.new_zeros(()),
            "pixels": prediction_disparity.new_zeros((), dtype=torch.long),
        }
    if teacher_depth.shape != prediction_disparity.shape:
        raise ValueError("teacher_depth must match prediction_disparity")
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError("intrinsics must be [B,V,3,3]")
    if extrinsics.shape[-2:] == (4, 4):
        extrinsics = extrinsics[..., :3, :]
    if extrinsics.shape != (batch, views, 3, 4):
        raise ValueError("extrinsics must be [B,V,3,4] or [B,V,4,4]")

    reliable = teacher_reliable_mask.bool() & torch.isfinite(teacher_depth) & (teacher_depth > 0)
    teacher_disparity = torch.where(
        reliable,
        teacher_depth.clamp_min(1e-6).reciprocal(),
        torch.zeros_like(teacher_depth),
    )
    aligned_views = []
    valid_views = []
    for view_idx in range(views):
        aligned, valid, _, _ = align_disparity_scale_shift(
            prediction_disparity[:, view_idx],
            teacher_disparity[:, view_idx],
            reliable[:, view_idx],
            min_pixels=min_align_pixels,
        )
        aligned_views.append(aligned)
        valid_views.append(valid)
    aligned_disparity = torch.stack(aligned_views, dim=1)
    valid_views = torch.stack(valid_views, dim=1)
    aligned_depth = aligned_disparity.clamp_min(1e-4).reciprocal()

    ys, xs = torch.meshgrid(
        torch.arange(height, device=prediction_disparity.device),
        torch.arange(width, device=prediction_disparity.device),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=0).float()
    pixels = pixels.reshape(1, 3, -1).expand(batch, -1, -1)

    total = prediction_disparity.new_zeros(())
    total_weight = prediction_disparity.new_zeros(())
    for ref_idx in range(views):
        k_ref_inv = torch.linalg.inv(intrinsics[:, ref_idx].float())
        ref_depth = teacher_depth[:, ref_idx].float().reshape(batch, 1, -1)
        points_ref = torch.bmm(k_ref_inv, pixels) * ref_depth
        r_ref = extrinsics[:, ref_idx, :3, :3].float()
        t_ref = extrinsics[:, ref_idx, :3, 3:].float()
        points_world = torch.bmm(r_ref.transpose(1, 2), points_ref - t_ref)

        for target_idx in range(views):
            if target_idx == ref_idx:
                continue
            r_target = extrinsics[:, target_idx, :3, :3].float()
            t_target = extrinsics[:, target_idx, :3, 3:].float()
            points_target = torch.bmm(r_target, points_world) + t_target
            projected = torch.bmm(intrinsics[:, target_idx].float(), points_target)
            projected_depth = points_target[:, 2].reshape(batch, height, width)
            safe_z = projected[:, 2].clamp_min(1e-6)
            x = (projected[:, 0] / safe_z).reshape(batch, height, width)
            y = (projected[:, 1] / safe_z).reshape(batch, height, width)
            x_norm = 2.0 * x / max(width - 1, 1) - 1.0
            y_norm = 2.0 * y / max(height - 1, 1) - 1.0
            grid_single = torch.stack((x_norm, y_norm), dim=-1)
            grid = grid_single[:, None].expand(-1, views, -1, -1, -1).contiguous()

            sampled_student = _sample_map(aligned_depth, grid)[:, target_idx]
            sampled_teacher = _sample_map(teacher_depth, grid)[:, target_idx]
            sampled_reliable = _sample_map(reliable.float(), grid, mode="nearest")[:, target_idx] > 0.5
            inside = (x_norm.abs() <= 1.0) & (y_norm.abs() <= 1.0) & (projected_depth > 0)
            teacher_consistent = (
                (projected_depth - sampled_teacher).abs()
                / torch.maximum(projected_depth.abs(), sampled_teacher.abs()).clamp_min(1e-6)
                < float(relative_depth_threshold)
            )
            pair_mask = (
                reliable[:, ref_idx]
                & sampled_reliable
                & inside
                & teacher_consistent
                & valid_views[:, ref_idx, None, None]
                & valid_views[:, target_idx, None, None]
                & torch.isfinite(sampled_student)
                & (sampled_student > 0)
            )
            residual = torch.log(sampled_student.clamp_min(1e-6)) - torch.log(
                projected_depth.clamp_min(1e-6)
            )
            robust = torch.sqrt(residual.square() + charbonnier_eps**2) - charbonnier_eps
            pair_weight = pair_mask.to(robust.dtype)
            total = total + (robust * pair_weight).sum()
            total_weight = total_weight + pair_weight.sum()

    loss = total / total_weight.clamp_min(1.0)
    possible = float(batch * views * max(views - 1, 1) * height * width)
    return {
        "loss": loss,
        "coverage": total_weight / possible,
        "pixels": total_weight.detach(),
    }
