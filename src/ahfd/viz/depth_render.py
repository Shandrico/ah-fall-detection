"""Fixed-scale depth preview; never used as measurement or saved to disk."""

import math

import cv2
import numpy as np


def render_depth(
    depth, depth_scale: float, max_width: int = 640,
    *, min_depth_m: float = 0.3, max_depth_m: float = 6.0,
):
    """Colour aligned, non-hole-filled depth. Invalid/out-of-range stays black.

    A fixed scale matching the source avoids making a stationary scene change
    colour when someone enters view. Downsampling uses nearest neighbours, not invented
    depths across object boundaries. Returns None when there is no depth plane.
    """
    if (
        not math.isfinite(min_depth_m) or not math.isfinite(max_depth_m)
        or min_depth_m < 0 or max_depth_m <= min_depth_m
    ):
        raise ValueError("depth preview range must be finite and increasing")
    if (
        depth is None or depth.size == 0 or depth.ndim != 2
        or not math.isfinite(depth_scale) or depth_scale <= 0
    ):
        return None
    height, width = depth.shape
    if width > max_width:
        depth = cv2.resize(
            depth, (max_width, max(1, round(height * max_width / width))),
            interpolation=cv2.INTER_NEAREST,
        )
    metres = depth.astype(np.float64) * depth_scale
    valid = (
        np.isfinite(metres) & (metres > 0)
        & (metres >= min_depth_m) & (metres <= max_depth_m)
    )
    values = np.zeros(metres.shape, dtype=np.uint8)
    values[valid] = (
        (metres[valid] - min_depth_m) / (max_depth_m - min_depth_m) * 255
    ).astype(np.uint8)
    canvas = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    canvas[~valid] = 0
    return canvas
