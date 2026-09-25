"""Detect the floor and a raised bed surface straight from depth.

These prove the IMU-free path: fit the floor plane -> recover the same (pitch,
roll, height) an IMU would give (so the D435f needs no IMU and no --pitch), and
read the current height of a raised surface (an adjustable bed) from a masked
region.
"""

from __future__ import annotations

import numpy as np
import pytest

from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.plane_fit import fit_plane_ransac, ground_from_floor, surface_height
from ahfd.types import Intrinsics


def _scene(pitch_deg=25.0, mount_m=2.4):
    intr = Intrinsics(width=320, height=240, fx=230.0, fy=230.0, cx=160.0, cy=120.0)
    ground = GroundPlane(intrinsics=intr, height_m=mount_m, pitch_deg=pitch_deg, roll_deg=0.0)
    return intr, ground


def _plane_depth(intr, ground, world_z):
    """Depth image where every ray hits the horizontal plane at world height z."""
    R = ground.rotation
    cam_h = ground.height_m
    us, vs = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    dx = (us - intr.cx) / intr.fx
    dy = (vs - intr.cy) / intr.fy
    rz = R[2, 0] * dx + R[2, 1] * dy + R[2, 2]  # (R @ (dx,dy,1))[2]
    t = (world_z - cam_h) / np.where(np.abs(rz) < 1e-6, np.nan, rz)
    depth = np.where(np.isfinite(t) & (t > 0.2), t, 0.0)
    return depth.astype(np.float32)


def test_fit_plane_recovers_normal():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (500, 2))
    z = 0.1 * xy[:, 0] + 0.2 * xy[:, 1] + 1.5
    pts = np.column_stack([xy, z]) + rng.normal(0, 0.005, (500, 3))
    n, d, inliers = fit_plane_ransac(pts, thresh=0.03)
    truth = np.array([-0.1, -0.2, 1.0])
    truth /= np.linalg.norm(truth)
    assert int(inliers.sum()) > 400
    assert abs(abs(float(n @ truth)) - 1.0) < 1e-2  # normals parallel


def test_ground_from_floor_recovers_tilt_and_height():
    intr, ground = _scene(pitch_deg=25.0, mount_m=2.4)
    depth = _plane_depth(intr, ground, world_z=0.0)  # the whole view is floor
    got = ground_from_floor(depth, intr)
    assert got is not None
    pitch, roll, height = got
    assert pitch == pytest.approx(25.0, abs=1.5)
    assert abs(roll) < 2.0
    assert height == pytest.approx(2.4, abs=0.1)


def test_surface_height_reads_a_raised_bed():
    intr, ground = _scene(pitch_deg=25.0, mount_m=2.4)
    depth = _plane_depth(intr, ground, world_z=0.0)
    bed = _plane_depth(intr, ground, world_z=0.6)  # a 0.6 m raised mattress
    mask = np.zeros(depth.shape, bool)
    mask[110:170, 120:220] = True
    depth[mask] = bed[mask]
    assert surface_height(depth, intr, ground, mask) == pytest.approx(0.6, abs=0.08)
    floor = np.zeros(depth.shape, bool)
    floor[20:70, 20:100] = True
    assert surface_height(depth, intr, ground, floor) == pytest.approx(0.0, abs=0.08)
