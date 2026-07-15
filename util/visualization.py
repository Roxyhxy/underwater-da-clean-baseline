import os
from pathlib import Path

import cv2
import numpy as np


def colorize_raw_disparity(disparity, colormap="Spectral_r"):
    """Colorize dense model output without GT alignment or depth conversion."""
    disparity = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(disparity)
    colored = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    if finite.sum() < 8:
        return colored

    values = disparity[finite]
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        high = low + 1e-6

    normalized = np.clip((disparity - low) / (high - low), 0.0, 1.0)
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("Saving Spectral disparity maps requires matplotlib") from exc

    cmap = matplotlib.colormaps.get_cmap(colormap)
    colored_rgb = (cmap(normalized)[..., :3] * 255.0).round().astype(np.uint8)
    colored = colored_rgb[..., ::-1].copy()
    colored[~finite] = 0
    return colored


def save_raw_disparity(disparity, output_dir, index, image_path, colormap="Spectral_r"):
    """Save raw relative disparity as float32 NPY and a shared color rendering."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    prefix = f"{index:04d}_{stem}_raw_disp"
    disparity = np.asarray(disparity, dtype=np.float32)
    np.save(os.path.join(output_dir, f"{prefix}.npy"), disparity)
    colored = colorize_raw_disparity(disparity, colormap=colormap)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}.png"), colored)
