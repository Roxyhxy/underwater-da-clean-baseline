import torch

from util.wat3r_distillation import (
    build_hole_focus_weight,
    build_teacher_reliability_mask,
    wat3r_hole_distillation_loss,
    wat3r_hole_edge_metrics,
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
        min_hole_pixels=1,
    )
    assert result["loss"].item() < 1e-4
    assert result["coverage"].item() > 0.4
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_hole_distillation_skips_images_with_too_few_reliable_pixels():
    teacher_depth = _slanted_depth()[None]
    prediction = (2.0 / teacher_depth + 0.3).requires_grad_()
    observed = torch.ones_like(teacher_depth, dtype=torch.bool)
    observed[:, :2, :2] = False
    reliable = torch.ones_like(observed)
    result = wat3r_hole_distillation_loss(
        prediction,
        teacher_depth,
        teacher_depth,
        observed,
        reliable,
        min_align_pixels=10,
        min_hole_pixels=8,
    )
    assert result["loss"].item() == 0.0
    assert result["eligible_fraction"].item() == 0.0
    assert result["hole_ratio"].item() == 0.0


def test_hole_focus_increases_boundary_and_far_weights():
    depth = _slanted_depth()[None]
    observed = torch.zeros_like(depth, dtype=torch.bool)
    observed[:, 4:12, 4:12] = True
    hole = ~observed
    teacher_disparity = depth.reciprocal()
    weight, boundary, far = build_hole_focus_weight(
        observed,
        depth,
        teacher_disparity,
        hole,
        boundary_radius=2,
        boundary_weight=2.0,
        far_quantile=0.5,
        far_weight=3.0,
    )
    assert boundary.any()
    assert far.any()
    assert weight[boundary].min().item() >= 2.0
    assert weight[far].min().item() >= 3.0


def test_hole_edge_metrics_reward_affine_equivalent_geometry():
    teacher_depth = _slanted_depth()[None]
    prediction = 2.0 / teacher_depth + 0.3
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    observed = ((x + y) % 2 == 0)[None]
    result = wat3r_hole_edge_metrics(
        prediction,
        teacher_depth,
        teacher_depth,
        observed,
        torch.ones_like(observed),
        min_align_pixels=10,
    )
    assert result["edge_mae"].item() < 1e-5
    assert result["edge_f1"].item() > 0.99


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


def test_multiview_focus_mask_can_disable_all_source_pixels():
    batch, views, height, width = 1, 3, 16, 16
    teacher_depth = _slanted_depth(height, width)[None, None].repeat(batch, views, 1, 1)
    prediction = (2.0 / teacher_depth + 0.3).requires_grad_()
    reliable = torch.ones_like(teacher_depth, dtype=torch.bool)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, views, 1, 1)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, views, 1, 1)
    result = wat3r_multiview_geometry_loss(
        prediction,
        teacher_depth,
        reliable,
        intrinsics,
        extrinsics,
        source_focus_mask=torch.zeros_like(reliable),
        min_align_pixels=10,
    )
    assert result["loss"].item() == 0.0
    assert result["pixels"].item() == 0.0
