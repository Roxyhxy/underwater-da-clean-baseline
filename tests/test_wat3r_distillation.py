import torch

from util.wat3r_distillation import (
    build_teacher_reliability_mask,
    wat3r_hole_distillation_loss,
    wat3r_multiview_geometry_loss,
)


def _slanted_depth(height=16, width=16):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return (1.0 + 0.02 * x + 0.01 * y).float()


def test_reliability_uses_confidence_quantile():
    depth = torch.ones(2, 16, 16)
    confidence = torch.arange(256).reshape(1, 16, 16).float().repeat(2, 1, 1)
    static = torch.ones_like(depth, dtype=torch.bool)
    mask = build_teacher_reliability_mask(depth, confidence, static, 0.5)
    assert 0.45 < mask.float().mean().item() < 0.55


def test_hole_distillation_is_zero_for_affine_equivalent_disparity():
    teacher_depth = _slanted_depth()[None]
    prediction = (2.0 / teacher_depth + 0.3).requires_grad_()
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    observed = ((x + y) % 2 == 0)[None]
    reliable = torch.ones_like(observed)
    result = wat3r_hole_distillation_loss(
        prediction,
        teacher_depth,
        teacher_depth,
        observed,
        reliable,
        min_align_pixels=10,
    )
    assert result["loss"].item() < 1e-4
    assert result["coverage"].item() > 0.4
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_multiview_loss_is_zero_for_consistent_identity_cameras():
    batch, views, height, width = 1, 3, 16, 16
    teacher_depth = _slanted_depth(height, width)[None, None].repeat(batch, views, 1, 1)
    prediction = (2.0 / teacher_depth + 0.3).requires_grad_()
    reliable = torch.ones_like(teacher_depth, dtype=torch.bool)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, views, 1, 1)
    intrinsics[:, :, 0, 0] = 10
    intrinsics[:, :, 1, 1] = 10
    intrinsics[:, :, 0, 2] = 7.5
    intrinsics[:, :, 1, 2] = 7.5
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, views, 1, 1)
    result = wat3r_multiview_geometry_loss(
        prediction,
        teacher_depth,
        reliable,
        intrinsics,
        extrinsics,
        min_align_pixels=10,
    )
    assert result["loss"].item() < 1e-4
    assert result["coverage"].item() > 0.5
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()
