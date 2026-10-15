"""D435f support must supply real floor evidence, never pretend to have an IMU."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from ahfd.capture.devices import RealSenseDevice, RealSenseProbe
from ahfd.cli import (
    _canonical_onsite_source,
    _collection_completion_issues,
    _floor_health_details,
    _probe_onsite_d435i,
    _probe_onsite_realsense,
    _write_calibration_yaml,
)
from ahfd.config import Config
from ahfd.geometry.calibration import load_calibration
from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.orientation import (
    FloorMonitor,
    floor_orientation,
    validate_floor_roi,
)
from ahfd.types import Intrinsics


def scene(pitch=25.0, roll=0.0, height=2.4, surface=0.0):
    intr = Intrinsics(320, 240, 230.0, 230.0, 160.0, 120.0)
    ground = GroundPlane(intr, height, pitch, roll)
    us, vs = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    rays = np.stack(
        ((us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones(us.shape)), axis=-1
    )
    denom = rays @ ground.rotation[2]
    z = (surface - height) / np.where(np.abs(denom) < 1e-6, np.nan, denom)
    z = np.where(np.isfinite(z) & (z >= 0.3) & (z <= 6.0), z, 0.0)
    frame = SimpleNamespace(depth_raw=z, depth_scale=1.0, intrinsics=intr, gravity=None)
    calib = SimpleNamespace(
        ground=ground, height_m=height, floor_roi=(0.1, 0.7, 0.9, 0.98)
    )
    return calib, frame


@pytest.mark.parametrize("pitch,roll", [(15, 0), (30, 7), (40, -12)])
def test_floor_recovers_measured_mount_including_roll(pitch, roll):
    calib, frame = scene(pitch, roll)
    evidence = floor_orientation(calib, frame)
    assert evidence is not None and evidence.valid
    assert evidence.pitch_delta_deg == pytest.approx(0, abs=0.01)
    assert evidence.roll_delta_deg == pytest.approx(0, abs=0.01)
    assert evidence.height_delta_m == pytest.approx(0, abs=0.001)


@pytest.mark.parametrize("change", ["tilt", "height", "roll", "mattress"])
def test_floor_rejects_moved_mount_and_raised_surface(change):
    calib, _ = scene()
    params = (
        {"pitch": 29}
        if change == "tilt"
        else {"height": 2.65}
        if change == "height"
        else {"roll": 5}
        if change == "roll"
        else {"surface": 0.6}
    )
    _, frame = scene(**params)
    evidence = floor_orientation(calib, frame)
    assert evidence is not None and not evidence.valid


def test_patch_cannot_silently_use_the_rest_of_the_room():
    calib, frame = scene()
    frame.depth_raw[168:236, 32:289] = 0
    assert floor_orientation(calib, frame) is None


def test_wall_does_not_pass_as_floor():
    calib, frame = scene()
    frame.depth_raw[:] = 3.0
    evidence = floor_orientation(calib, frame)
    assert evidence is not None and not evidence.valid


def test_floor_monitor_requires_warmup_and_restarts_after_loss():
    calib, frame = scene()
    monitor = FloorMonitor(calib)
    assert monitor.update(frame, 0) == "WARMING"
    for now in np.arange(0.25, 2.1, 0.25):
        assert monitor.update(frame, now) == "WARMING"
    assert monitor.update(frame, 2.25) == "OK"
    good = frame.depth_raw.copy()
    frame.depth_raw[:] = 0
    assert monitor.update(frame, 2.5) == "MISSING"
    frame.depth_raw = good
    for now in np.arange(2.75, 5.0, 0.25):
        assert monitor.update(frame, now) == "WARMING"
    assert monitor.update(frame, 5.0) == "OK"
    details = _floor_health_details(monitor)
    assert details["floor_orientation_check"] == "OK"
    assert not any("imu" in key for key in details)


@pytest.mark.parametrize(
    "roi",
    [
        None,
        [0, 0, 1],
        [0, 0, 2, 1],
        [0.8, 0, 0.2, 1],
        [0, 0, 0.01, 0.01],
        [0, 0, float("nan"), 1],
    ],
)
def test_bad_floor_rois_rejected(roi):
    with pytest.raises(ValueError, match="floor_roi"):
        validate_floor_roi(roi)


def test_floor_roi_calibration_round_trip(tmp_path):
    calib, _ = scene()
    path = tmp_path / "camera.yaml"
    _write_calibration_yaml(
        path,
        "cam_0123abcd",
        calib.ground.intrinsics,
        calib.height_m,
        pitch_deg=25.0,
        floor_roi=calib.floor_roi,
        device_serial_sha256="a" * 64,
    )
    saved = load_calibration(path)
    assert saved.floor_roi == calib.floor_roi
    assert saved.verified_for_onsite is False
    assert "gravity" not in path.read_text()


def test_d435f_device_is_identified_and_old_imu_probe_stays_strict(monkeypatch):
    import ahfd.capture

    monkeypatch.setattr(
        ahfd.capture,
        "probe_realsense",
        lambda: RealSenseProbe(
            installed=True, devices=(RealSenseDevice("RealSense D435F", "test", "3.2"),)
        ),
    )
    assert _probe_onsite_realsense()[2] == "d435f"
    with pytest.raises(Exception, match="D435i"):
        _probe_onsite_d435i()


def test_d435f_record_provenance_is_truthful():
    cfg = Config(source="rs://?depth=1&emitter=1&max_laser=1")
    source = _canonical_onsite_source(
        cfg.source,
        SimpleNamespace(width=1920, height=1080, fps=30.0, has_depth=True),
        cfg,
        "a" * 64,
        {
            "emitter_enabled": True,
            "laser_at_max": True,
            "laser_power": 360.0,
            "laser_power_max": 360.0,
        },
        model="d435f",
    )
    assert source["kind"] == "intel_realsense_d435f"
    assert source["orientation_method"] == "depth_floor_roi"


def test_d435f_completion_requires_floor_even_in_empty_room():
    for subject in (None, "sub_0123456789abcdef"):
        assert _collection_completion_issues(
            subject,
            labeled_available_rows=1,
            imu_orientation_ready=False,
            floor_orientation_ready=False,
        ) == ("FLOOR_PREFLIGHT_NOT_READY",)
        assert (
            _collection_completion_issues(
                subject,
                labeled_available_rows=1,
                imu_orientation_ready=False,
                floor_orientation_ready=True,
            )
            == ()
        )
    assert _collection_completion_issues(
        "sub_0123456789abcdef",
        labeled_available_rows=0,
        imu_orientation_ready=False,
        floor_orientation_ready=True,
    ) == ("NO_LABELED_AVAILABLE_TARGET_ROWS",)


def test_d435f_dataset_requires_floor_provenance(tmp_path):
    from test_bed_dataset import make_session
    from ahfd.ml.bed_dataset import load_bed_session

    directory = make_session(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["kind"] = "intel_realsense_d435f"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="depth_floor_roi"):
        load_bed_session(directory)
    manifest["source"]["orientation_method"] = "depth_floor_roi"
    path.write_text(json.dumps(manifest))
    assert len(load_bed_session(directory).phase.metadata) > 0
