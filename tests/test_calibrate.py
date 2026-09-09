"""`ahfd calibrate` writes a calibration that loads back correctly.

No camera: the command's YAML writer is exercised directly with synthetic
intrinsics and either a gravity vector (the D435i path) or an explicit pitch
(the webcam path), then round-tripped through load_calibration. Getting the
round-trip right is the whole point -- a calibration that writes but does not
load, or loads to a different tilt, would silently corrupt every metric.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ahfd.cli import _write_calibration_yaml
from ahfd.geometry.calibration import load_calibration
from ahfd.types import Intrinsics

INTR = Intrinsics.from_hfov(1920, 1080, hfov_deg=69.4, vfov_deg=42.5)


class TestPitchPath:
    def test_round_trips_pitch_and_height(self, tmp_path):
        out = tmp_path / "cam.yaml"
        _write_calibration_yaml(out, "cam_a", INTR, 2.6, pitch_deg=20.0, roll_deg=0.0)
        c = load_calibration(out)
        assert c.camera_id == "cam_a"
        assert c.height_m == pytest.approx(2.6)
        assert c.ground.pitch_deg == pytest.approx(20.0)
        assert (c.ground.intrinsics.width, c.ground.intrinsics.height) == (1920, 1080)

    def test_intrinsics_survive(self, tmp_path):
        out = tmp_path / "cam.yaml"
        _write_calibration_yaml(out, "cam_a", INTR, 2.6, pitch_deg=20.0)
        c = load_calibration(out)
        assert c.ground.intrinsics.fx == pytest.approx(INTR.fx, abs=0.01)
        assert c.ground.intrinsics.fy == pytest.approx(INTR.fy, abs=0.01)

    def test_starts_with_no_zones(self, tmp_path):
        out = tmp_path / "cam.yaml"
        _write_calibration_yaml(out, "cam_a", INTR, 2.6, pitch_deg=20.0)
        assert load_calibration(out).zones.zones == []


class TestMissingFile:
    def test_missing_calibration_gives_actionable_error(self, tmp_path):
        """A missing calib file must say how to make it, not a raw errno."""
        missing = tmp_path / "d435i.yaml"
        with pytest.raises(FileNotFoundError) as excinfo:
            load_calibration(missing)
        msg = str(excinfo.value)
        assert "ahfd calibrate" in msg
        assert "d435i.yaml" in msg


class TestGravityPath:
    @pytest.mark.parametrize("pitch", [10.0, 20.0, 35.0])
    def test_imu_gravity_round_trips_to_pitch(self, tmp_path, pitch):
        """The D435i path: a gravity vector must load back to the right tilt."""
        g = np.array(
            [0.0, math.cos(math.radians(pitch)), math.sin(math.radians(pitch))]
        )
        out = tmp_path / "cam.yaml"
        _write_calibration_yaml(out, "cam_imu", INTR, 2.6, gravity=g)
        c = load_calibration(out)
        assert c.ground.pitch_deg == pytest.approx(pitch, abs=0.05)

    def test_gravity_preferred_over_pitch_when_both_given(self, tmp_path):
        """If a gravity vector is supplied it wins -- the writer emits gravity,
        not pitch, so the IMU value is what loads back."""
        g = np.array([0.0, math.cos(math.radians(25.0)), math.sin(math.radians(25.0))])
        out = tmp_path / "cam.yaml"
        _write_calibration_yaml(out, "c", INTR, 2.6, gravity=g, pitch_deg=99.0)
        assert load_calibration(out).ground.pitch_deg == pytest.approx(25.0, abs=0.05)
