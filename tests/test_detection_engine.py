"""Composition keeps fall posture, bed phase, support and visibility separate."""

from types import SimpleNamespace

from ahfd.detect import DetectionEngine, Event, FallStateMachine
from ahfd.features import Features


def frame(t, *, motion=0.0, confidence=0.9):
    return Features(
        track_id=1,
        t=t,
        contact_xy=(0.0, 5.0),
        range_m=5.5,
        h_torso=0.55,
        h_head=0.75,
        h_max=0.85,
        h_min=0.4,
        h_ankle_min=0.5,
        floor_spread=1.6,
        v_z=0.0,
        motion=motion,
        n_valid_kp=15,
        mean_conf=confidence,
        zones=("bed_a",),
        supported_by_bed="bed_a",
        bed_top_m=0.55,
        bed_risk="high",
        bed_overlap=0.9,
        torso_tilt=70.0,
        h_shoulder=0.65,
        associated_bed="bed_a",
        bed_edge_distance_m=0.4,
    )


def test_engine_uses_independent_shadow_bed_machine_not_legacy_movement_alarm():
    legacy = FallStateMachine()
    assert legacy.update(frame(0.0, motion=0.2)) is not None

    engine = DetectionEngine()
    assert engine.update(frame(0.0, motion=0.2)) is None
    assert engine.state_of(1) == "IN_BED"
    snapshot = engine.bed_snapshot_of(1, 0.0)
    assert snapshot.support == "SUPPORTED"
    assert snapshot.observation == "VALID"
    assert snapshot.phase != engine.state_of(1)


def test_low_confidence_changes_observation_not_last_activity_phase():
    engine = DetectionEngine()
    for index in range(20):
        engine.update(frame(index * 0.1))
    before = engine.bed_phase_of(1)
    engine.update(frame(2.1, confidence=0.1))
    after = engine.bed_snapshot_of(1, 2.1)
    assert after.phase == before
    assert after.observation == "LOW_CONFIDENCE"


def test_retain_only_drops_both_submachines():
    engine = DetectionEngine()
    engine.update(frame(0.0))
    engine.retain_only(set())
    assert engine.state_of(1) == "UNKNOWN"
    assert engine.bed_phase_of(1) == "UNKNOWN"


def test_update_all_preserves_same_frame_fall_and_bed_events():
    fall_event = Event(
        type="FALL_CONFIRMED",
        track_id=1,
        t_trigger=1.0,
        t_alert=1.1,
        zone="bed_a",
        evidence={},
    )
    bed_event = Event(
        type="BED_EXIT",
        track_id=1,
        t_trigger=1.0,
        t_alert=1.1,
        zone="bed_a",
        evidence={},
    )
    engine = DetectionEngine()
    engine.fall = SimpleNamespace(update=lambda _features: fall_event)
    engine.bed = SimpleNamespace(update=lambda _features: bed_event)

    assert engine.update_all(frame(1.1)) == (fall_event, bed_event)


def test_legacy_update_keeps_urgent_event_when_two_occur_together():
    fall_event = Event(
        type="FALL_CONFIRMED",
        track_id=1,
        t_trigger=1.0,
        t_alert=1.1,
        zone="bed_a",
        evidence={},
    )
    bed_event = Event(
        type="BED_EXIT",
        track_id=1,
        t_trigger=1.0,
        t_alert=1.1,
        zone="bed_a",
        evidence={},
    )
    engine = DetectionEngine()
    engine.fall = SimpleNamespace(update=lambda _features: fall_event)
    engine.bed = SimpleNamespace(update=lambda _features: bed_event)

    assert engine.update(frame(1.1)) is fall_event
