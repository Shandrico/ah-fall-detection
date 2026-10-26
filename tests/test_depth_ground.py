"""Depth heights use the depth stream's own floor-fitted pose (calibration depth_ground)."""

import numpy as np
import pytest
import yaml

from ahfd.geometry.calibration import load_calibration
from ahfd.geometry.depth_height import keypoint_heights_from_depth, reground_heights
from ahfd.geometry.ground import GroundPlane

CAMERA = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
          "height_m": 2.6, "pitch_deg": 30.0}


def _write(tmp_path, **extra):
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump({"camera_id": "cam", "camera": CAMERA, **extra}))
    return path


def test_without_depth_ground_depth_uses_the_colour_geometry(tmp_path):
    calib = load_calibration(_write(tmp_path))
    assert calib.depth_ground is None and calib.ground_for_depth is calib.ground


def test_depth_ground_is_loaded_and_validated(tmp_path):
    calib = load_calibration(_write(tmp_path, depth_ground={"height_m": 2.58, "pitch_deg": 24.1, "roll_deg": 1.2}))
    g = calib.ground_for_depth
    assert (g.height_m, g.pitch_deg, g.roll_deg) == (2.58, 24.1, 1.2)
    assert g.intrinsics == calib.ground.intrinsics
    with pytest.raises(ValueError):
        load_calibration(_write(tmp_path, depth_ground={"height_m": 2.58, "pitch_deg": 24.1}))


def test_reground_matches_recomputing_from_depth():
    k = load_calibration.__globals__["_intrinsics_from"](CAMERA)
    colour = GroundPlane(intrinsics=k, height_m=2.6, pitch_deg=30.0, roll_deg=0.0)
    depth_pose = GroundPlane(intrinsics=k, height_m=2.58, pitch_deg=24.1, roll_deg=1.2)
    rng = np.random.default_rng(0)
    depth = rng.uniform(2.0, 5.0, (480, 640)).astype(np.float32)
    kp = np.column_stack([rng.uniform(50, 590, 17), rng.uniform(50, 430, 17)])
    scores = np.ones(17)
    via_colour = keypoint_heights_from_depth(kp, scores, depth, k, colour, patch=0)
    direct = keypoint_heights_from_depth(kp, scores, depth, k, depth_pose, patch=0)
    assert np.allclose(reground_heights(kp, via_colour, colour, depth_pose), direct, atol=1e-6)


def test_extractor_defaults_depth_ground_to_ground():
    from ahfd.features import FeatureExtractor

    k = load_calibration.__globals__["_intrinsics_from"](CAMERA)
    g = GroundPlane(intrinsics=k, height_m=2.6, pitch_deg=30.0, roll_deg=0.0)
    assert FeatureExtractor(g).depth_ground is g
