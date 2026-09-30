import torch

from util.depth_losses import dense_relative_depth_loss


def test_dense_relative_loss_is_affine_invariant_in_disparity():
    y, x = torch.meshgrid(torch.arange(16), torch.arange(20), indexing="ij")
    depth = (1.0 + 0.02 * x + 0.01 * y)[None]
    prediction = (2.5 / depth + 0.4).requires_grad_()
    valid = torch.ones_like(depth, dtype=torch.bool)

    result = dense_relative_depth_loss(prediction, depth, valid)
    assert result["loss"].item() < 1e-5
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_dense_relative_loss_ignores_invalid_pixels():
    depth = torch.ones(1, 8, 8)
    prediction = torch.ones(1, 8, 8, requires_grad=True)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:, :2, :2] = False
    depth[:, :2, :2] = float("nan")

    result = dense_relative_depth_loss(prediction, depth, valid)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()
