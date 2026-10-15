"""Verify a fixed D435f mount against a deliberately selected floor patch.

Never pick the dominant plane of the entire room: a mattress or wall can win.
Only scalar evidence leaves this module; depth stays in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ahfd.geometry.plane_fit import fit_plane_ransac


def validate_floor_roi(value) -> tuple[float, float, float, float]:
    try:
        roi = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("floor_roi must be four normalised xyxy numbers") from exc
    if len(roi) != 4 or not all(np.isfinite(item) for item in roi):
        raise ValueError("floor_roi must be four finite normalised xyxy numbers")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(
            "floor_roi must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
        )
    if (x1 - x0) * (y1 - y0) < 0.005:
        raise ValueError("floor_roi must cover at least 0.5 percent of the image")
    return roi


@dataclass(frozen=True)
class FloorEvidence:
    pitch_delta_deg: float
    roll_delta_deg: float
    height_delta_m: float
    inlier_fraction: float

    @property
    def valid(self) -> bool:
        return (
            abs(self.pitch_delta_deg) <= 2.0
            and abs(self.roll_delta_deg) <= 2.0
            and abs(self.height_delta_m) <= 0.10
        )


def floor_orientation(calib, frame) -> FloorEvidence | None:
    """Compare raw aligned depth with the measured ground plane, or abstain."""
    roi = getattr(calib, "floor_roi", None)
    depth = frame.depth_raw
    intr = frame.intrinsics
    if roi is None or depth is None or intr is None:
        return None
    x0, y0, x1, y1 = validate_floor_roi(roi)
    if depth.ndim != 2 or depth.shape != (intr.height, intr.width):
        return None
    h, w = depth.shape
    step = max(1, int(np.ceil(np.sqrt((x1 - x0) * (y1 - y0) * h * w / 1600))))
    vs, us = np.mgrid[
        int(y0 * h) : int(y1 * h) : step, int(x0 * w) : int(x1 * w) : step
    ]
    z = depth[vs, us].astype(float) * float(frame.depth_scale)
    good = np.isfinite(z) & (z >= 0.3) & (z <= 6.0)
    if np.count_nonzero(good) < 100 or np.mean(good) < 0.6:
        return None
    z, u, v = z[good], us[good], vs[good]
    points = np.column_stack(
        ((u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z)
    )
    normal, distance, inliers = fit_plane_ransac(points, iters=80, thresh=0.025)
    if normal is None or np.mean(inliers) < 0.8:
        return None
    # Reject line-like patches: a floor plane needs two independent dimensions.
    singular = np.linalg.svd(
        points[inliers] - points[inliers].mean(0), compute_uv=False
    )
    if singular[1] / np.sqrt(np.count_nonzero(inliers)) < 0.04:
        return None
    if distance < 0:
        normal, distance = -normal, -distance
    # The ground rotation's third row is the upward normal. Invert that
    # convention directly; do not fold wall/upward fits into valid downtilts.
    pitch = np.rad2deg(np.arcsin(np.clip(-normal[2], -1.0, 1.0)))
    roll = np.rad2deg(np.arctan2(-normal[0], -normal[1]))
    return FloorEvidence(
        float(pitch - calib.ground.pitch_deg),
        float((roll - calib.ground.roll_deg + 180) % 360 - 180),
        float(distance - calib.height_m),
        float(np.mean(inliers)),
    )


class FloorMonitor:
    """Sample at 5 Hz; require 2 s of valid floor, fail closed on loss/drift."""

    def __init__(self, calib):
        self.calib = calib
        self.next_check = 0.0
        self.stable_since: float | None = None
        self.last_sample: float | None = None
        self.stable_samples = 0
        self.evidence: FloorEvidence | None = None
        self.status = "WARMING"

    def update(self, frame, now: float) -> str:
        if now < self.next_check:
            return self.status
        self.next_check = now + 0.2
        if self.last_sample is not None and now - self.last_sample > 0.8:
            self.stable_since = None
            self.stable_samples = 0
        self.last_sample = now
        self.evidence = floor_orientation(self.calib, frame)
        if self.evidence is None or not self.evidence.valid:
            self.stable_since = None
            self.stable_samples = 0
            self.status = "MISSING" if self.evidence is None else "DRIFT"
        else:
            if self.stable_since is None:
                self.stable_since = now
            self.stable_samples += 1
            self.status = (
                "OK"
                if now - self.stable_since >= 2.0 and self.stable_samples >= 10
                else "WARMING"
            )
        return self.status
