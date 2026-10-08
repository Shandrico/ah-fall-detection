"""Monitoring gaps and weak observations remain explicit in collected data."""

from __future__ import annotations

import pytest

from ahfd.collect import HealthMonitor, HealthReason, HealthStatus


def test_starts_unavailable_until_a_real_frame_arrives():
    health = HealthMonitor()
    assert health.status == HealthStatus.UNAVAILABLE
    assert health.reasons == (HealthReason.STARTING,)
    transition = health.observe_frame(0.1)
    assert transition.previous == HealthStatus.UNAVAILABLE
    assert transition.current == HealthStatus.AVAILABLE
    assert health.snapshot()["reasons"] == []


def test_low_confidence_is_degraded_not_a_normal_negative():
    health = HealthMonitor()
    health.observe_frame(0.0)
    transition = health.observe_frame(
        0.1, pose_confident=False, details={"mean_conf": 0.22}
    )
    assert transition.current == HealthStatus.DEGRADED
    assert transition.reasons == (HealthReason.LOW_CONFIDENCE,)
    assert transition.details["mean_conf"] == 0.22


def test_missing_depth_and_reassociation_reasons_are_retained_together():
    health = HealthMonitor()
    transition = health.observe_frame(
        0.0, depth_valid=False, reassociated=True
    )
    assert transition.current == HealthStatus.DEGRADED
    assert set(transition.reasons) == {
        HealthReason.DEPTH_MISSING,
        HealthReason.TRACK_REASSOCIATED,
    }
    assert set(transition.as_record()["reasons"]) == {
        "DEPTH_MISSING",
        "TRACK_REASSOCIATED",
    }


def test_watchdog_opens_unavailable_interval_after_frame_gap():
    health = HealthMonitor(stale_after_s=2.0)
    health.observe_frame(1.0)
    assert health.tick(2.9) is None
    transition = health.tick(3.01)
    assert transition.current == HealthStatus.UNAVAILABLE
    assert transition.reasons == (HealthReason.NO_FRAMES,)
    assert transition.details["frame_age_s"] == pytest.approx(2.01)


def test_startup_gets_the_same_stale_grace_period():
    health = HealthMonitor(stale_after_s=2.0)
    assert health.tick(1.9) is None
    transition = health.tick(2.01)
    assert transition.current == HealthStatus.UNAVAILABLE
    assert transition.reasons == (HealthReason.NO_FRAMES,)


def test_unbound_target_does_not_hide_real_frame_freshness():
    health = HealthMonitor(stale_after_s=2.0)
    health.note_frame(1.0)
    health.mark_unavailable(1.01, HealthReason.TARGET_NOT_BOUND)

    assert health.tick(2.9) is None
    transition = health.tick(3.01)
    assert transition is not None
    assert transition.reasons == (HealthReason.NO_FRAMES,)


def test_healthy_frame_closes_degraded_or_unavailable_state():
    health = HealthMonitor(stale_after_s=1.0)
    health.observe_frame(0.0, pose_confident=False)
    health.tick(1.1)
    recovered = health.observe_frame(1.2)
    assert recovered.previous == HealthStatus.UNAVAILABLE
    assert recovered.current == HealthStatus.AVAILABLE
    assert recovered.reasons == ()


def test_calibration_drift_makes_monitoring_unavailable():
    health = HealthMonitor()
    transition = health.observe_frame(
        0.0, calibration_valid=False, details={"ankle_error_m": 0.18}
    )
    assert transition.current == HealthStatus.UNAVAILABLE
    assert transition.reasons == (HealthReason.CALIBRATION_DRIFT,)


def test_disconnect_and_disk_failure_can_be_marked_explicitly():
    health = HealthMonitor()
    health.observe_frame(0.0)
    disconnected = health.mark_unavailable(
        1.0, HealthReason.CAMERA_DISCONNECTED, details={"device_code": "cam_01"}
    )
    assert disconnected.current == HealthStatus.UNAVAILABLE
    assert disconnected.reasons == (HealthReason.CAMERA_DISCONNECTED,)
    disk = health.mark_unavailable(2.0, HealthReason.DISK_ERROR)
    assert disk.reasons == (HealthReason.DISK_ERROR,)


@pytest.mark.parametrize(
    "reason",
    [
        HealthReason.TARGET_NOT_BOUND,
        HealthReason.PRIVACY_SCOPE_VIOLATION,
        HealthReason.PROTOCOL_COMPLETE,
        HealthReason.PROTOCOL_INCOMPLETE,
        HealthReason.REQUESTED_STOP,
    ],
)
def test_fail_closed_collection_reasons_can_be_marked_unavailable(reason):
    health = HealthMonitor()

    transition = health.mark_unavailable(0.5, reason)

    assert transition is not None
    assert transition.current == HealthStatus.UNAVAILABLE
    assert transition.reasons == (reason,)


def test_same_state_and_evidence_does_not_spam_transitions_when_details_change():
    seen = []
    health = HealthMonitor(on_transition=seen.append)
    first = health.observe_frame(
        0.0, pose_confident=False, details={"fps": 12.0}
    )
    duplicate = health.observe_frame(
        0.1, pose_confident=False, details={"fps": 11.5}
    )
    assert first is not None
    assert duplicate is None
    assert len(seen) == 1
    assert health.snapshot()["details"] == {"fps": 11.5}


def test_reason_change_is_logged_even_when_status_stays_degraded():
    health = HealthMonitor()
    health.observe_frame(0.0, pose_confident=False)
    changed = health.observe_frame(0.1, depth_valid=False)
    assert changed.previous == HealthStatus.DEGRADED
    assert changed.current == HealthStatus.DEGRADED
    assert changed.reasons == (HealthReason.DEPTH_MISSING,)


def test_health_time_is_monotonic():
    health = HealthMonitor()
    health.observe_frame(2.0)
    with pytest.raises(ValueError, match="backwards"):
        health.tick(1.0)


def test_health_details_must_be_scalar_json():
    health = HealthMonitor()
    with pytest.raises(TypeError, match="scalar"):
        health.observe_frame(0.0, pose_confident=False, details={"values": [0.1]})
