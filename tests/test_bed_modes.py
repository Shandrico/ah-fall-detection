"""Per-bed dashboard policy is independent from activity recognition."""

from __future__ import annotations

from ahfd.dashboard.bed_modes import (
    apply_bed_mode,
    is_dashboard_alert,
    mode_from_legacy_risk,
)
from ahfd.detect.events import Event


def event(kind: str) -> Event:
    return Event(
        type=kind,
        track_id=7,
        t_trigger=1.0,
        t_alert=2.0,
        zone="bed_a",
        evidence={"trigger": "early_warning" if kind.endswith("WARNING") else "out_of_bed"},
    )


def test_low_admits_falls_only():
    assert apply_bed_mode(event("BED_EXIT_WARNING"), "low") is None
    assert apply_bed_mode(event("BED_EXIT"), "low") is None
    fall = event("FALL_CONFIRMED")
    assert apply_bed_mode(fall, "low") is fall


def test_medium_admits_completed_exit_not_early_warning():
    assert apply_bed_mode(event("BED_EXIT_WARNING"), "medium") is None
    admitted = apply_bed_mode(event("BED_EXIT"), "medium")
    assert admitted is not None
    assert admitted.severity == 2
    assert admitted.evidence["monitoring_mode"] == "medium"
    assert is_dashboard_alert(admitted)


def test_high_admits_corroborated_warning_and_exit():
    for kind in ("BED_EXIT_WARNING", "BED_EXIT"):
        admitted = apply_bed_mode(event(kind), "high")
        assert admitted is not None
        assert admitted.severity == 3
        assert admitted.evidence["monitoring_mode"] == "high"
        assert is_dashboard_alert(admitted)


def test_legacy_sit_up_event_is_never_relabelled_as_temporal_exit():
    legacy = Event(
        type="BED_EXIT",
        track_id=7,
        t_trigger=1.0,
        t_alert=2.0,
        zone="bed_a",
        evidence={"trigger": "sit_up"},
    )
    assert apply_bed_mode(legacy, "medium") is None
    assert apply_bed_mode(legacy, "high") is None


def test_legacy_risk_only_supplies_initial_mode():
    assert mode_from_legacy_risk("low") == "low"
    assert mode_from_legacy_risk("high") == "high"
    assert mode_from_legacy_risk("unknown") == "medium"
    assert mode_from_legacy_risk(None) == "medium"
