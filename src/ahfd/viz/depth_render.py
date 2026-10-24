"""Fixed-scale depth preview; never used as measurement or saved to disk."""

import math

import cv2
import numpy as np

# Light direction for the relief shading: from the upper left, towards the scene.
_LIGHT = np.array([-0.45, -0.55, 0.70])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def render_depth(
    depth, depth_scale: float, max_width: int = 640,
    *, min_depth_m: float = 0.3, max_depth_m: float = 6.0,
    shade: bool = False, legend: bool = False, clamp: bool = False,
    colormap: int = cv2.COLORMAP_TURBO,
):
    """Colour aligned, non-hole-filled depth. Invalid/out-of-range stays black.

    A fixed scale matching the source avoids making a stationary scene change
    colour when someone enters view. Downsampling uses nearest neighbours, not invented
    depths across object boundaries. Returns None when there is no depth plane.

    ``shade`` adds relief lighting from the depth gradient, so surfaces and body
    outlines read as 3D shapes rather than flat bands of colour. ``legend`` draws
    a colour bar with metre ticks on the right edge. Both are display only.
    ``clamp`` paints real depth outside the range in the end colours instead of
    black -- for a scene-fitted range, where black should still mean "no depth".
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
    valid = np.isfinite(metres) & (metres > 0)
    if not clamp:
        valid &= (metres >= min_depth_m) & (metres <= max_depth_m)
    values = np.zeros(metres.shape, dtype=np.uint8)
    values[valid] = (
        np.clip((metres[valid] - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0) * 255
    ).astype(np.uint8)
    canvas = cv2.applyColorMap(values, colormap)
    if shade:
        canvas = shade_depth(canvas, metres, valid)
    canvas[~valid] = 0
    if legend:
        draw_depth_legend(canvas, min_depth_m, max_depth_m, colormap=colormap)
    return canvas


def shade_depth(canvas, metres, valid, strength: float = 0.55):
    """Multiply ``canvas`` by Lambertian lighting from the depth surface normals.

    The normal comes from the gradient of log-depth, which is scale-free: a
    slope looks the same near or far. The image width stands in for the focal
    length (a ~70 deg lens has fx ~ 0.72 * width). Holes are excluded from the
    gradient so they do not paint dark halos onto valid neighbours.
    """
    if not valid.any():
        return canvas
    logd = np.zeros(metres.shape, dtype=np.float32)
    logd[valid] = np.log(metres[valid])
    mask = valid.astype(np.float32)

    def masked_blur(sigma):
        # Average of valid neighbours only, so holes never pull the surface to 0.
        return cv2.GaussianBlur(logd, (0, 0), sigma) / np.maximum(
            cv2.GaussianBlur(mask, (0, 0), sigma), 1e-3
        )

    # Lightly smooth valid depth (sensor speckle) and fill holes from a wider
    # neighbourhood so the gradient across them stays small.
    logd = np.where(valid, masked_blur(1.0), masked_blur(3.0))
    focal = 0.72 * metres.shape[1]
    gy, gx = np.gradient(logd)
    nx, ny = -gx * focal, -gy * focal
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    lit = (nx * _LIGHT[0] + ny * _LIGHT[1] + _LIGHT[2]) / norm
    light = (1.0 - strength) + strength * np.clip(lit, 0.0, 1.0)
    return np.clip(canvas.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)


def _nice_step(span: float) -> float:
    """A 1/2/5 x 10^n tick step giving roughly four to eight ticks over ``span``."""
    raw = span / 5.0
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        if raw <= mult * mag:
            return mult * mag
    return 10 * mag


def draw_depth_legend(
    img, lo: float, hi: float, *, colormap: int = cv2.COLORMAP_TURBO,
    unit: str = "m", invert: bool = False,
):
    """Draw a vertical colour bar with value ticks on the right edge of ``img``."""
    h, w = img.shape[:2]
    bar_w = max(8, w // 60)
    bar_h = int(h * 0.6)
    if bar_h < 40 or hi <= lo:
        return img
    x1 = w - 12 - 46
    x0 = x1 - bar_w
    y0 = (h - bar_h) // 2
    # Top of the bar is the far (high) end, like a map legend.
    ramp = np.linspace(255, 0, bar_h).astype(np.uint8)
    if invert:
        ramp = 255 - ramp
    colours = cv2.applyColorMap(ramp[:, None], colormap)
    img[y0:y0 + bar_h, x0:x1] = colours.repeat(bar_w, axis=1)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x1, y0 + bar_h), (255, 255, 255), 1)
    step = _nice_step(hi - lo)
    tick = math.ceil(lo / step) * step
    while tick <= hi + 1e-9:
        y = int(round(y0 + (hi - tick) / (hi - lo) * (bar_h - 1)))
        cv2.line(img, (x1, y), (x1 + 4, y), (255, 255, 255), 1)
        label = ("%g" % round(tick, 2)) + unit
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, label, (x1 + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, colour, thick, cv2.LINE_AA)
        tick += step
    return img


def scene_depth_range(
    depth, depth_scale: float, *, floor_m: float = 0.3, ceiling_m: float = 6.0,
):
    """A (near, far) colour range fitted to the scene, rounded to 0.25 m.

    Uses the 2nd-98th percentile of valid depth so a few stray pixels do not
    stretch the ramp. Returns None if there is too little depth to judge.
    """
    if depth is None or depth.size == 0 or not math.isfinite(depth_scale) or depth_scale <= 0:
        return None
    metres = depth[::4, ::4].astype(np.float64) * depth_scale
    vals = metres[np.isfinite(metres) & (metres >= floor_m) & (metres <= ceiling_m)]
    if vals.size < 500:
        return None
    lo, hi = np.percentile(vals, [2, 98])
    lo = max(floor_m, math.floor(lo * 4) / 4)
    hi = min(ceiling_m, math.ceil(hi * 4) / 4)
    if hi - lo < 0.5:
        hi = min(ceiling_m, lo + 0.5)
    return (lo, hi) if hi > lo else None
