"""Onsite collection refuses unsafe modes before touching the camera."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from ahfd.cli import (
    _bed_observation_valid,
    _canonical_onsite_source,
    _collection_completion_issues,
    _collection_disk_free_bytes,
    _collection_exception_abort_code,
    _collection_marker,
    _collection_person_record,
    _collection_scope_violation,
    _credible_pose_count,
    _depth_provenance,
    _finalize_collection_recorder,
    _imu_orientation_delta,
    _intrinsics_match,
    _is_labeled_available_target_row,
    _next_collection_target,
    _probe_onsite_d435i,
    _robust_ankle_baseline,
    _returned_associations,
    _standing_preflight_eligible,
    _standing_depth_ankle_height,
    _target_depth_healthy,
    _validate_approved_output,
    _validate_calibration_approval,
    _validate_onsite_calibration,
    app,
)
from ahfd.config import Config
from ahfd.detect import BedExitSnapshot
from ahfd.features import Features
from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import Zone, ZoneMap
from ahfd.privacy import ENV_VAR
from ahfd.types import Intrinsics


runner = CliRunner()


def test_collect_refuses_every_source_override_before_opening_camera(tmp_path):
    result = runner.invoke(
        app,
        [
            "collect",
            "--out-root",
            str(tmp_path),
            "--site-id",
            "site_0123abcd",
            "--source",
            "file://not-a-camera.mp4",
        ],
    )
    assert result.exit_code != 0
    assert "--source overrides are disabled onsite" in result.output


def test_collect_refuses_when_raw_capture_environment_is_armed(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    result = runner.invoke(
        app,
        [
            "collect",
            "--out-root",
            str(tmp_path),
            "--site-id",
            "site_0123abcd",
        ],
    )
    assert result.exit_code != 0
    assert "raw capture is armed" in result.output


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("enabled", "bed_activity.enabled"),
        ("emit_exit_event", "bed_activity.emit_exit_event"),
    ],
)
def test_collect_requires_complete_bed_shadow_pipeline(
    tmp_path, monkeypatch, field, message
):
    config_path = tmp_path / "onsite.yaml"
    config_path.write_text("test: config\n", encoding="utf-8")
    cfg = Config(source="rs://?depth=1&emitter=1&max_laser=1")
    cfg.detect.enabled = True
    setattr(cfg.bed_activity, field, False)

    import ahfd.cli

    monkeypatch.setattr(ahfd.cli, "load_config", lambda _path: cfg)
    result = runner.invoke(
        app,
        [
            "collect",
            "--out-root",
            str(tmp_path / "output"),
            "--site-id",
            "site_0123abcd",
            "--config",
            str(config_path),
            "--calibration",
            str(tmp_path / "calibration.yaml"),
            "--approval",
            str(tmp_path / "approval.json"),
        ],
    )

    assert result.exit_code != 0
    assert message in result.output


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("rs://?depth=1&emitter=1", "max_laser=1"),
        ("rs://?depth=1&max_laser=1", "emitter=1"),
    ],
)
def test_collect_requires_approved_depth_emitter_policy(
    tmp_path, monkeypatch, source, message
):
    config_path = tmp_path / "onsite.yaml"
    config_path.write_text("test: config\n", encoding="utf-8")
    cfg = Config(source=source)

    import ahfd.cli

    monkeypatch.setattr(ahfd.cli, "load_config", lambda _path: cfg)
    result = runner.invoke(
        app,
        [
            "collect",
            "--out-root",
            str(tmp_path / "output"),
            "--site-id",
            "site_0123abcd",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert message in result.output


def test_collection_record_is_json_only_and_redacts_care_risk():
    person = SimpleNamespace(
        track_id=4,
        keypoints=np.zeros((17, 2), dtype=float),
        scores=np.ones(17, dtype=float),
        heights=np.full(17, np.nan),
    )
    features = Features(
        track_id=4,
        t=1.2,
        contact_xy=(0.0, 4.0),
        range_m=4.5,
        h_torso=0.6,
        h_head=0.8,
        h_max=0.9,
        h_min=0.3,
        h_ankle_min=0.4,
        floor_spread=1.5,
        v_z=0.1,
        motion=0.05,
        n_valid_kp=17,
        mean_conf=0.9,
        zones=("bed_a",),
        supported_by_bed="bed_a",
        bed_risk="high",
        bed_overlap=0.8,
        torso_tilt=65.0,
        h_shoulder=0.7,
        associated_bed="bed_a",
        bed_edge_distance_m=0.3,
    )
    snapshot = BedExitSnapshot(
        track_id=4,
        t=1.2,
        phase="RECLINED",
        phase_since=0.0,
        support="SUPPORTED",
        observation="VALID",
        observation_since=0.0,
        bed_id="bed_a",
        bed_risk="high",
    )
    machine = SimpleNamespace(
        state_of=lambda _track_id: "IN_BED",
        bed_snapshot_of=lambda _track_id, _t: snapshot,
    )

    row = _collection_person_record(person, features, machine, 2)
    assert row["joint_heights_m"] == [None] * 17
    assert "bed_risk" not in row["features"]
    assert "bed_risk" not in row["bed_activity"]
    assert isinstance(row["keypoints_xy"], list)


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        ("VALID", True),
        ("LOW_CONFIDENCE", False),
        ("MONITORING_UNAVAILABLE", False),
    ],
)
def test_collection_temporal_mask_follows_bed_observation(observation, expected):
    snapshot = SimpleNamespace(observation=observation)
    assert _bed_observation_valid(snapshot) is expected


def test_drift_uses_two_real_depth_ankles_not_monocular_fallback():
    features = SimpleNamespace(supported_by_bed=None, h_ankle_min=0.7)
    person = SimpleNamespace(
        heights=None,
        scores=np.ones(17, dtype=float),
    )
    assert _standing_depth_ankle_height(
        person, features, "UPRIGHT", min_score=0.5, feature_valid=True
    ) is None

    heights = np.full(17, np.nan)
    heights[15], heights[16] = 0.07, 0.09
    person.heights = heights
    assert _standing_depth_ankle_height(
        person, features, "UPRIGHT", min_score=0.5, feature_valid=True
    ) == pytest.approx(0.08)

    heights[16] = 0.30
    assert _standing_depth_ankle_height(
        person, features, "UPRIGHT", min_score=0.5, feature_valid=True
    ) is None


def test_markers_require_one_explicit_selected_association():
    assert _collection_marker(
        "phase_marker", "RECLINED", None, {1, 2}, {1: 10, 2: 11}
    ) is None
    marker = _collection_marker(
        "phase_marker", "RECLINED", 2, {1, 2}, {1: 10, 2: 11}
    )
    assert marker == {
        "kind": "phase_marker",
        "phase": "RECLINED",
        "track_ids": [2],
        "associations": [{"track_id": 2, "association_epoch": 11}],
    }


def test_collection_scope_is_one_explicit_participant_and_returns_reassociate():
    assert _collection_scope_violation(None, set()) is None
    assert (
        _collection_scope_violation(None, {4})
        == "UNEXPECTED_PERSON_IN_EMPTY_ROOM"
    )
    assert _collection_scope_violation("sub_0123456789abcdef", {4}) is None
    assert (
        _collection_scope_violation("sub_0123456789abcdef", {4, 5})
        == "UNAPPROVED_PERSON_PRESENT"
    )
    assert _returned_associations({4}, {4: 1.0}) == {4}
    assert _returned_associations({5}, {4: 1.0}) == set()


def test_scope_counts_credible_untracked_people_before_persistence():
    tracked = SimpleNamespace(score=0.95, track_id=4)
    untracked = SimpleNamespace(score=0.75, track_id=None)
    low_confidence_noise = SimpleNamespace(score=0.2, track_id=None)

    assert _credible_pose_count(
        [tracked, untracked, low_confidence_noise], min_person_score=0.5
    ) == 2
    assert (
        _collection_scope_violation(
            "sub_0123456789abcdef", {4}, credible_person_count=2
        )
        == "UNAPPROVED_PERSON_PRESENT"
    )
    assert (
        _collection_scope_violation(None, set(), credible_person_count=1)
        == "UNEXPECTED_PERSON_IN_EMPTY_ROOM"
    )


def test_planned_finish_completes_but_requested_stop_aborts():
    planned = SimpleNamespace(
        complete_calls=0,
        abort_reasons=[],
        complete=lambda: setattr(planned, "complete_calls", planned.complete_calls + 1),
        abort=lambda reason: planned.abort_reasons.append(reason),
    )
    requested = SimpleNamespace(
        complete_calls=0,
        abort_reasons=[],
        complete=lambda: setattr(
            requested, "complete_calls", requested.complete_calls + 1
        ),
        abort=lambda reason: requested.abort_reasons.append(reason),
    )

    assert _finalize_collection_recorder(planned, requested_abort=False) == "complete"
    assert planned.complete_calls == 1
    assert planned.abort_reasons == []
    assert _finalize_collection_recorder(requested, requested_abort=True) == "aborted"
    assert requested.complete_calls == 0
    assert requested.abort_reasons == ["REQUESTED_STOP"]

    incomplete = SimpleNamespace(
        complete_calls=0,
        abort_reasons=[],
        complete=lambda: setattr(
            incomplete, "complete_calls", incomplete.complete_calls + 1
        ),
        abort=lambda reason: incomplete.abort_reasons.append(reason),
    )
    assert (
        _finalize_collection_recorder(
            incomplete,
            requested_abort=False,
            abort_reason="PROTOCOL_INCOMPLETE",
        )
        == "aborted"
    )
    assert incomplete.complete_calls == 0
    assert incomplete.abort_reasons == ["PROTOCOL_INCOMPLETE"]


def test_participant_completion_requires_bound_labeled_available_imu_data():
    participant_id = "sub_0123456789abcdef"
    assert _collection_completion_issues(
        participant_id,
        labeled_available_rows=0,
        imu_orientation_ready=False,
    ) == (
        "NO_LABELED_AVAILABLE_TARGET_ROWS",
        "IMU_PREFLIGHT_NOT_READY",
    )
    assert _collection_completion_issues(
        participant_id,
        labeled_available_rows=1,
        imu_orientation_ready=True,
    ) == ()


def test_completion_row_requires_same_epoch_nonunknown_label_and_availability():
    participant = "sub_0123456789abcdef"
    row = {"track_id": 7, "association_epoch": 3}
    available = {"status": "AVAILABLE"}
    phases = {(7, 3): "EDGE_SITTING"}
    assert _is_labeled_available_target_row(
        participant, 7, 3, [row], available, phases
    )
    assert not _is_labeled_available_target_row(
        participant, 7, 4, [row], available, phases
    )
    assert not _is_labeled_available_target_row(
        participant, 7, 3, [row], available, {(7, 3): "UNKNOWN"}
    )
    assert not _is_labeled_available_target_row(
        participant, 7, 3, [row], {"status": "DEGRADED"}, phases
    )

    # The empty-room protocol intentionally has no target or observer labels.
    assert _collection_completion_issues(
        None,
        labeled_available_rows=0,
        imu_orientation_ready=False,
    ) == ()


def test_reselecting_only_visible_target_is_a_noop():
    assert _next_collection_target([7], 7, step=1) == (7, False)
    assert _next_collection_target([7], 7, step=-1) == (7, False)
    assert _next_collection_target([7], None, step=1) == (7, True)
    assert _next_collection_target([4, 7], 4, step=1) == (7, True)


def test_only_unclassified_capture_boundary_failure_is_camera_disconnect():
    assert (
        _collection_exception_abort_code(
            "PIPELINE_ERROR", processing_frame=False
        )
        == "CAMERA_DISCONNECTED"
    )
    assert (
        _collection_exception_abort_code(
            "FINALIZATION_ERROR", processing_frame=False
        )
        == "FINALIZATION_ERROR"
    )
    assert (
        _collection_exception_abort_code("DISK_ERROR", processing_frame=False)
        == "DISK_ERROR"
    )


def test_disk_usage_oserror_is_a_disk_failure_not_camera_disconnect(
    tmp_path, monkeypatch
):
    import shutil

    def fail(_path):
        raise OSError("volume unavailable")

    monkeypatch.setattr(shutil, "disk_usage", fail)
    with pytest.raises(OSError, match="volume unavailable"):
        _collection_disk_free_bytes(tmp_path)
    assert (
        _collection_exception_abort_code("DISK_ERROR", processing_frame=True)
        == "DISK_ERROR"
    )


def test_depth_provenance_distinguishes_sparse_depth_from_fallback():
    person = SimpleNamespace(heights=None, scores=np.ones(17))
    assert _depth_provenance(person, min_score=0.5) == {
        "joint_depth_valid_fraction": 0.0,
        "shoulder_depth_available": 0.0,
        "torso_depth_available": 0.0,
    }
    heights = np.full(17, np.nan)
    heights[5] = 0.7
    person.heights = heights
    context = _depth_provenance(person, min_score=0.5)
    assert context["joint_depth_valid_fraction"] == pytest.approx(1 / 17)
    assert context["shoulder_depth_available"] == 1.0
    assert context["torso_depth_available"] == 0.0
    assert not _target_depth_healthy(context)

    heights[6] = 0.71
    heights[11] = 0.65
    heights[12] = 0.66
    context = _depth_provenance(person, min_score=0.5)
    assert context["joint_depth_valid_fraction"] == pytest.approx(4 / 17)
    assert context["torso_depth_available"] == 1.0
    assert _target_depth_healthy(context)


def test_live_intrinsics_and_imu_are_compared_with_calibration():
    intrinsics = Intrinsics(1920, 1080, 1000.0, 1001.0, 960.0, 540.0)
    assert _intrinsics_match(intrinsics, intrinsics)
    shifted = Intrinsics(1920, 1080, 1003.0, 1001.0, 960.0, 540.0)
    assert not _intrinsics_match(intrinsics, shifted)

    gravity = np.array([0.0, np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0))])
    calibration = SimpleNamespace(
        height_m=2.5,
        ground=GroundPlane.from_gravity(intrinsics, 2.5, gravity),
    )
    pitch_delta, roll_delta = _imu_orientation_delta(calibration, gravity)
    assert pitch_delta == pytest.approx(0.0)
    assert roll_delta == pytest.approx(0.0)


def test_onsite_probe_requires_one_usb3_d435i(monkeypatch):
    import hashlib

    import ahfd.capture
    from ahfd.capture.devices import RealSenseDevice, RealSenseProbe

    monkeypatch.setattr(
        ahfd.capture,
        "probe_realsense",
        lambda: RealSenseProbe(
            installed=True,
            devices=(RealSenseDevice("Intel RealSense D435I", "abc123", "3.2"),),
        ),
    )
    assert _probe_onsite_d435i() == (
        "abc123",
        hashlib.sha256(b"abc123").hexdigest(),
    )

    monkeypatch.setattr(
        ahfd.capture,
        "probe_realsense",
        lambda: RealSenseProbe(
            installed=True,
            devices=(RealSenseDevice("Intel RealSense D435I", "abc123", "2.1"),),
        ),
    )
    with pytest.raises(Exception, match="USB 2.1"):
        _probe_onsite_d435i()

    for descriptor in ("", "unknown"):
        monkeypatch.setattr(
            ahfd.capture,
            "probe_realsense",
            lambda descriptor=descriptor: RealSenseProbe(
                installed=True,
                devices=(
                    RealSenseDevice(
                        "Intel RealSense D435I", "abc123", descriptor
                    ),
                ),
            ),
        )
        with pytest.raises(Exception, match="requires a verified USB 3"):
            _probe_onsite_d435i()


def test_canonical_onsite_provenance_includes_emitter_not_raw_serial():
    cfg = Config(source="rs://?depth=1&emitter=1&max_laser=1")
    meta = SimpleNamespace(width=1920, height=1080, fps=30.0, has_depth=True)
    provenance = _canonical_onsite_source(
        cfg.source,
        meta,
        cfg,
        approval_sha256="a" * 64,
        verified_depth_controls={
            "emitter_enabled": True,
            "laser_at_max": True,
            "laser_power": 360.0,
            "laser_power_max": 360.0,
        },
    )
    assert provenance["emitter"] is True
    assert provenance["max_laser"] is True
    assert provenance["laser_power"] == 360.0
    assert provenance["laser_power_max"] == 360.0
    assert "serial" not in json.dumps(provenance).lower()

    with pytest.raises(Exception, match="requires verified emitter"):
        _canonical_onsite_source(
            cfg.source,
            meta,
            cfg,
            approval_sha256="a" * 64,
            verified_depth_controls={
                "emitter_enabled": False,
                "laser_at_max": True,
            },
        )

    with pytest.raises(Exception, match="maximum laser power"):
        _canonical_onsite_source(
            cfg.source,
            meta,
            cfg,
            approval_sha256="a" * 64,
            verified_depth_controls={
                "emitter_enabled": True,
                "laser_at_max": True,
                "laser_power": 300.0,
                "laser_power_max": 360.0,
            },
        )


def test_onsite_calibration_binds_pseudonymous_camera_and_device():
    serial_hash = "a" * 64
    calibration = SimpleNamespace(
        camera_id="cam_01234567",
        verified_for_onsite=True,
        device_serial_sha256=serial_hash,
        ankle_height_baseline_m=0.08,
        height_m=2.5,
        zones=ZoneMap(
            [
                Zone(
                    name="bed_a",
                    kind="bed",
                    polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)],
                    top_m=0.5,
                    risk_level="unknown",
                )
            ]
        ),
    )
    _validate_onsite_calibration(calibration, serial_hash)
    calibration.camera_id = "ward_6_bed_2"
    with pytest.raises(Exception, match="pseudonymous"):
        _validate_onsite_calibration(calibration, serial_hash)


def test_robust_ankle_baseline_requires_stable_plausible_sample():
    stable = [0.08 + (index % 3 - 1) * 0.002 for index in range(45)]
    assert _robust_ankle_baseline(stable) == pytest.approx(0.08)
    with pytest.raises(ValueError, match="at least 30"):
        _robust_ankle_baseline(stable[:10])
    with pytest.raises(ValueError, match="plausible"):
        _robust_ankle_baseline([0.5] * 40)
    with pytest.raises(ValueError, match="unstable"):
        _robust_ankle_baseline([0.0, 0.2] * 20)


def test_output_and_calibration_need_external_custodian_records(tmp_path):
    site_id = "site_0123abcd"
    output = tmp_path / "approved"
    output.mkdir()
    marker = {
        "schema": "ahfd.approved-output",
        "schema_version": 1,
        "site_id": site_id,
        "encrypted_storage_attested": True,
        "purpose": "research_shadow_collection",
    }
    (output / ".ahfd-approved-output.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    _validate_approved_output(output, site_id)

    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("camera: approved\n", encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval_record = {
        "schema": "ahfd.calibration.approval",
        "schema_version": 2,
        "site_id": site_id,
        "purpose": "research_shadow_collection",
        "config_sha256": "b" * 64,
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "pose_model_sha256": "c" * 64,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
    }
    approval.write_text(json.dumps(approval_record), encoding="utf-8")
    approval_hash, calibration_hash = _validate_calibration_approval(
        calibration, approval, site_id, "b" * 64, "c" * 64
    )
    assert len(approval_hash) == 64
    assert calibration_hash == hashlib.sha256(calibration.read_bytes()).hexdigest()

    with pytest.raises(Exception, match="does not exactly match"):
        _validate_calibration_approval(
            calibration, approval, site_id, "b" * 64, "d" * 64
        )

    approval_record["calibration_sha256"] = "0" * 64
    approval.write_text(json.dumps(approval_record), encoding="utf-8")
    with pytest.raises(Exception, match="does not exactly match"):
        _validate_calibration_approval(
            calibration, approval, site_id, "b" * 64, "c" * 64
        )


def test_standing_preflight_rejects_nonupright_or_excluded_pose():
    cfg = SimpleNamespace(
        detect=SimpleNamespace(
            min_valid_kp=8,
            min_mean_conf=0.4,
            upright_h=0.7,
            seated_tilt_max=35.0,
        )
    )
    features = SimpleNamespace(
        n_valid_kp=15,
        mean_conf=0.9,
        has_geometry=lambda: True,
        h_torso=1.0,
        torso_tilt=10.0,
        supported_by_bed=None,
        in_excluded_zone=False,
    )
    assert _standing_preflight_eligible(features, cfg)
    features.torso_tilt = 65.0
    assert not _standing_preflight_eligible(features, cfg)
    features.torso_tilt = 10.0
    features.in_excluded_zone = True
    assert not _standing_preflight_eligible(features, cfg)
