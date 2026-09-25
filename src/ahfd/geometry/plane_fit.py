"""Detect flat surfaces (floor, bed) directly in depth -- IMU-independent.

The IMU gives the camera tilt so a joint's height is measured against a floor at
z = 0. The D435f has no IMU. But the floor and the bed are both flat surfaces the
depth sensor can SEE, so we can recover them from the depth itself:

* ``ground_from_floor`` fits the floor plane and returns the same (pitch, roll,
  height) the IMU would -- a self-calibration from a downward-looking view, so
  the D435f needs neither an IMU nor a hand-entered ``--pitch``; and
* ``surface_height`` measures the CURRENT height of a surface (the mattress)
  inside a region, so an adjustable bed that shifts up/down can be tracked live
  and the patient measured relative to the bed, not a stale fixed value.

STANDALONE and not wired into detection yet -- it exists to be tested on its own
against real depth first. numpy only; no capture/pose imports, so it unit tests
with synthetic point clouds.
"""

from __future__ import annotations

import math

import numpy as np


def deproject(depth_m: np.ndarray, intrinsics, *, stride: int = 4,
              zmin: float = 0.3, zmax: float = 8.0) -> np.ndarray:
    """Valid depth pixels -> (N, 3) camera-frame points (X right, Y down, Z fwd)."""
    h, w = depth_m.shape[:2]
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride].astype(np.float64)
    m = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    z = z[m]
    u = us[m].astype(np.float64)
    v = vs[m].astype(np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x, y, z], axis=1)


def fit_plane_ransac(points: np.ndarray, *, iters: int = 300, thresh: float = 0.03,
                     seed: int = 0):
    """RANSAC + SVD plane fit. Returns (normal(3,), d, inliers) for n.p + d = 0.

    ``normal`` is unit length; a point p lies on the plane when |n.p + d| < thresh.
    Returns (None, None, all-False) if there are too few points.
    """
    n = len(points)
    if n < 3:
        return None, None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inliers = None
    best = (None, None)
    for _ in range(iters):
        a, b, c = points[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = -float(nrm @ a)
        inliers = np.abs(points @ nrm + d) < thresh
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers, best = inliers, (nrm, d)
    nrm, d = best
    if best_inliers is not None and int(best_inliers.sum()) >= 3:  # refine on inliers
        pts = points[best_inliers]
        centre = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - centre)
        nrm = vt[2] / np.linalg.norm(vt[2])
        d = -float(nrm @ centre)
        best_inliers = np.abs(points @ nrm + d) < thresh
    return nrm, d, (best_inliers if best_inliers is not None else np.zeros(n, bool))


def ground_from_floor(depth_m: np.ndarray, intrinsics, *, thresh: float = 0.03):
    """Recover (pitch_deg, roll_deg, height_m) by fitting the floor plane.

    The floor is taken as the dominant plane; its normal is oriented to point up
    toward the camera. The tilt reuses the IMU convention (gravity = -normal),
    and the height is the camera's distance to the plane. Returns None if no
    plane is found. NOTE: in a cluttered room the dominant plane may be a wall or
    the bed -- this is why it ships standalone, to be checked on real depth.
    """
    from ahfd.geometry.ground import GroundPlane

    pts = deproject(depth_m, intrinsics)
    nrm, d, inliers = fit_plane_ransac(pts, thresh=thresh)
    if nrm is None or int(inliers.sum()) < 50:
        return None
    # Orient the normal so the camera origin sits on its positive side (d > 0),
    # i.e. the normal points up toward the camera; then gravity is -normal.
    if d < 0:
        nrm, d = -nrm, -d
    pitch, roll = GroundPlane.pitch_roll_from_gravity(-nrm)
    return abs(pitch), roll, float(d)


def surface_height(depth_m: np.ndarray, intrinsics, ground, mask: np.ndarray) -> float:
    """Median height above the floor (m) of the depth inside ``mask``.

    Point ``mask`` at the bed region to read the CURRENT mattress height -- the
    value an adjustable bed changes -- so the patient can be measured relative to
    it. NaN if nothing valid is in the mask.
    """
    h, w = depth_m.shape[:2]
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    R = ground.rotation
    ys, xs = np.nonzero(mask & np.isfinite(depth_m) & (depth_m > 0))
    if ys.size == 0:
        return float("nan")
    z = depth_m[ys, xs].astype(np.float64)
    px = (xs - cx) / fx * z
    py = (ys - cy) / fy * z
    world_z = R[2, 0] * px + R[2, 1] * py + R[2, 2] * z
    return float(np.median(ground.height_m + world_z))
