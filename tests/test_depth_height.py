"""Depth-derived joint heights recover the truth, including at the nadir.

The whole point of measuring height from depth is that it works where the
monocular geometry fails -- directly beneath the camera. These tests build a
synthetic depth image by projecting joints at known heights, then check the
recovered heights and the coarse posture read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ahfd.geometry.depth_height import (
    DEPTH_FEATURES,
    coarse_posture,
    depth_features,
    keypoint_heights_from_depth,
    region_height,
)
from ahfd.geometry.ground import GroundPlane
from ahfd.pose.skeleton import ANKLES, HIPS, SHOULDERS
from ahfd.types import Intrinsics


def _scene(pitch_deg=30.0, mount_m=2.5):
    intr = Intrinsics(width=1280, height=720, fx=640.0, fy=640.0, cx=640.0, cy=360.0)
    th = math.radians(pitch_deg)
    ground = GroundPlane.from_gravity(
        intr, mount_m, np.array([0.0, math.cos(th), math.sin(th)])
    )
    return intr, ground


def _project(intr, ground, world_xyz):
    """Pixel (u, v) and depth for a world point, using the ground convention."""
    R = ground.rotation
    cam_origin = np.array([0.0, 0.0, ground.height_m])
    p_cam = R.T @ (np.asarray(world_xyz, dtype=float) - cam_origin)
    u = intr.cx + intr.fx * p_cam[0] / p_cam[2]
    v = intr.cy + intr.fy * p_cam[1] / p_cam[2]
    return u, v, p_cam[2]


def _synthetic_person(intr, ground, joint_heights):
    """A depth image + keypoints/scores for joints at given world heights.

    ``joint_heights`` maps a keypoint index to its height above the floor; the
    body is placed on the camera's optical axis so it lands in frame.
    """
    axis = ground.rotation @ np.array([0.0, 0.0, 1.0])
    t = -ground.height_m / axis[2]
    foot = np.array([0.0, 0.0, ground.height_m]) + t * axis  # floor point, in view

    depth = np.zeros((intr.height, intr.width), np.float32)
    kp = np.zeros((17, 2), np.float32)
    sc = np.zeros(17, np.float32)
    for idx, (dx, z) in joint_heights.items():
        u, v, d = _project(intr, ground, (foot[0] + dx, foot[1], z))
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < intr.width and 0 <= vi < intr.height:
            depth[vi, ui] = d
            kp[idx] = (u, v)
            sc[idx] = 1.0
    return depth, kp, sc


def test_recovers_joint_heights():
    intr, ground = _scene()
    joints = {
        5: (-0.10, 1.40), 6: (0.10, 1.40),   # shoulders
        11: (-0.10, 0.90), 12: (0.10, 0.90),  # hips
        15: (-0.10, 0.06), 16: (0.10, 0.06),  # ankles
    }
    depth, kp, sc = _synthetic_person(intr, ground, joints)
    heights = keypoint_heights_from_depth(kp, sc, depth, intr, ground, patch=0)

    assert region_height(heights, SHOULDERS) == pytest.approx(1.40, abs=0.05)
    assert region_height(heights, HIPS) == pytest.approx(0.90, abs=0.05)
    assert region_height(heights, ANKLES) == pytest.approx(0.06, abs=0.05)


def test_posture_standing_vs_on_ground():
    intr, ground = _scene()

    standing = {5: (-0.1, 1.40), 6: (0.1, 1.40), 11: (-0.1, 0.90), 12: (0.1, 0.90)}
    depth, kp, sc = _synthetic_person(intr, ground, standing)
    heights = keypoint_heights_from_depth(kp, sc, depth, intr, ground, patch=0)
    assert coarse_posture(heights) == "standing"

    # A body flat on the floor: every joint near zero height.
    fallen = {5: (-0.1, 0.12), 6: (0.1, 0.12), 11: (-0.1, 0.10), 12: (0.1, 0.10)}
    depth, kp, sc = _synthetic_person(intr, ground, fallen)
    heights = keypoint_heights_from_depth(kp, sc, depth, intr, ground, patch=0)
    assert coarse_posture(heights) == "on-ground"


def test_low_score_joints_ignored():
    intr, ground = _scene()
    depth, kp, sc = _synthetic_person(intr, ground, {11: (0.0, 0.9), 12: (0.1, 0.9)})
    sc[:] = 0.1  # everything below threshold
    heights = keypoint_heights_from_depth(kp, sc, depth, intr, ground, min_score=0.4)
    assert np.all(np.isnan(heights))


def test_depth_features_none_when_no_depth():
    feats = depth_features(None)
    assert set(feats) == set(DEPTH_FEATURES)
    assert all(v is None for v in feats.values())


def test_depth_features_from_heights():
    h = np.full(17, np.nan, np.float32)
    h[list((5, 6))] = 1.40   # shoulders
    h[list((11, 12))] = 0.90  # hips
    h[list((13, 14))] = 0.50  # knees
    h[list((15, 16))] = 0.05  # ankles
    feats = depth_features(h)
    assert feats["dh_shoulder"] == pytest.approx(1.40, abs=1e-3)
    assert feats["dh_hip"] == pytest.approx(0.90, abs=1e-3)
    assert feats["dh_torso"] == pytest.approx((1.40 + 0.90) / 2, abs=1e-3)
    assert feats["dh_vextent"] == pytest.approx(1.40 - 0.05, abs=1e-3)  # tallest - lowest
    assert feats["dh_hip_above_knee"] == pytest.approx(0.40, abs=1e-3)
    assert feats["dh_ankle"] == pytest.approx(0.05, abs=1e-3)


def test_heights_survive_track_serialisation_roundtrip():
    from ahfd.io.tracks_io import dict_to_pose, pose_to_dict
    from ahfd.types import PersonPose, PoseFrame

    heights = np.array([np.nan, 1.6] + [0.9] * 15, np.float32)
    person = PersonPose(
        keypoints=np.zeros((17, 2), np.float32),
        scores=np.ones(17, np.float32),
        score=0.9,
        track_id=1,
        heights=heights,
    )
    frame = PoseFrame(t=0.0, index=0, width=1920, height=1080, people=(person,))
    back = dict_to_pose(pose_to_dict(frame))
    got = back.people[0].heights
    assert got is not None
    assert np.isnan(got[0])                       # NaN -> null -> NaN preserved
    assert got[1] == pytest.approx(1.6, abs=1e-3)
    # a plain RGB pose (no heights) round-trips to None, not an array
    rgb = PersonPose(keypoints=np.zeros((17, 2), np.float32), scores=np.ones(17, np.float32), score=0.9)
    assert dict_to_pose(pose_to_dict(PoseFrame(0.0, 0, 1920, 1080, (rgb,)))).people[0].heights is None
