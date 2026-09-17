"""Height-above-floor of body joints, measured from depth.

This is the piece that fixes the nadir failure of the monocular geometry. A
single image ray cannot recover a joint's height directly beneath the camera --
the triangulation in ``GroundPlane.joint_height`` goes degenerate and a standing
person collapses toward the floor (reads as on-ground). Depth does not: it gives
the joint's 3-D position outright, so its height is a direct measurement, best
exactly where the image geometry is worst.

The maths is the same convention as ``GroundPlane``: the camera sits at
``height_m`` above a floor at world Z = 0, and ``rotation`` maps camera axes to
world axes. For a pixel (u, v) at metric depth d, the camera-frame point is
``P = d * ((u-cx)/fx, (v-cy)/fy, 1)`` and its height above the floor is
``height_m + (rotation @ P).z``.

Kept dependency-light and free of any capture/pose import so it can be unit
tested with synthetic depth, reused by the live probe, and later dropped into
the feature extractor unchanged.
"""

from __future__ import annotations

import numpy as np

from ahfd.pose.skeleton import ANKLES, HEAD, HIPS, KNEES, SHOULDERS


def sample_depth_m(depth_m: np.ndarray, u: float, v: float, patch: int = 2) -> float:
    """Median valid depth (metres) in a small window around (u, v).

    A single depth pixel at a joint is often a hole (0) or an edge outlier; the
    median over a (2*patch+1) square rejects both. Returns NaN when the point is
    off-image or the whole window is invalid.
    """
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return float("nan")
    x0, x1 = max(0, ui - patch), min(w, ui + patch + 1)
    y0, y1 = max(0, vi - patch), min(h, vi + patch + 1)
    window = depth_m[y0:y1, x0:x1]
    valid = window[window > 0.0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def keypoint_heights_from_depth(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depth_m: np.ndarray,
    intrinsics,
    ground,
    *,
    min_score: float = 0.4,
    patch: int = 2,
) -> np.ndarray:
    """Height above the floor (metres) for each of the 17 joints, from depth.

    ``depth_m`` must be aligned to the same image as ``keypoints`` (pixel u, v)
    and already in metres. Entries are NaN for joints below ``min_score`` or
    with no valid depth nearby.
    """
    R = ground.rotation
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    h0 = float(ground.height_m)

    heights = np.full(len(keypoints), np.nan, dtype=float)
    for i in range(len(keypoints)):
        if scores[i] < min_score:
            continue
        u, v = float(keypoints[i][0]), float(keypoints[i][1])
        d = sample_depth_m(depth_m, u, v, patch)
        if not np.isfinite(d) or d <= 0.0:
            continue
        px = (u - cx) / fx * d
        py = (v - cy) / fy * d
        pz = d
        heights[i] = h0 + (R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz)
    return heights


def region_height(heights: np.ndarray, region) -> float:
    """Mean height over a joint group (head / hips / ...), NaN if none valid."""
    vals = [heights[i] for i in region if i < len(heights) and np.isfinite(heights[i])]
    return float(np.mean(vals)) if vals else float("nan")


def coarse_posture(heights: np.ndarray) -> str:
    """A rough posture label from depth heights alone -- a sanity read, not the
    classifier. The learned model will set real thresholds; this just shows
    whether depth heights separate the postures at all.
    """
    top = region_height(heights, HEAD)
    if not np.isfinite(top):
        top = region_height(heights, SHOULDERS)
    hip = region_height(heights, HIPS)

    if not np.isfinite(top) and not np.isfinite(hip):
        return "?"
    # Standing: torso well off the floor.
    if np.isfinite(hip) and hip >= 0.6 and (not np.isfinite(top) or top >= 1.2):
        return "standing"
    # On the ground: even the highest confident joint is low.
    if np.isfinite(top) and top < 0.6:
        return "on-ground"
    if np.isfinite(hip) and hip < 0.35 and (not np.isfinite(top) or top < 0.9):
        return "on-ground"
    return "sitting / low"
