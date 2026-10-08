"""Synthetic tests for the standalone hierarchical bed-activity machine."""

from dataclasses import dataclass

import pytest

from ahfd.detect.bed_exit import BedExitStateMachine, BedExitThresholds
from ahfd.detect.cusum import CusumConfig


DT = 0.1


@dataclass(frozen=True)
class Sample:
    track_id: int
    t: float
    contact_xy: tuple[float, float] | None = (0.0, 4.0)
    h_torso: float | None = 0.60
    h_shoulder: float | None = 0.60
    h_torso_source: str | None = "depth"
    h_shoulder_source: str | None = "depth"
    n_valid_kp: int = 15
    mean_conf: float = 0.9
    in_excluded_zone: bool = False
    supported_by_bed: str | None = "ward-A-2026-10-08"
    associated_bed: str | None = "ward-A-2026-10-08"
    bed_risk: str | None = "high"
    bed_overlap: float | None = 0.85
    bed_support_fraction: float | None = 0.85
    bed_edge_distance_m: float | None = 0.70
    torso_tilt: float | None = 70.0
    motion: float = 0.01

    def has_geometry(self):
        return self.contact_xy is not None and self.h_torso is not None


def feed(machine, start, seconds, *, track_id=1, values=None, **changes):
    count = max(1, int(round(seconds / DT)))
    events = []
    for i in range(count):
        current = dict(changes)
        if values is not None:
            current.update(values(i, count))
        event = machine.update(Sample(track_id=track_id, t=start + i * DT, **current))
        if event is not None:
            events.append(event)
    return start + count * DT, events


def establish_recline(machine, start=0.0, *, track_id=1):
    t, events = feed(machine, start, 2.0, track_id=track_id)
    assert events == []
    assert machine.phase_of(track_id) == "RECLINED"
    assert machine.snapshot_of(track_id).cusum_armed
    return t


def test_common_path_retains_support_independently_and_emits_only_on_exit():
    machine = BedExitStateMachine()
    t = establish_recline(machine)

    t, events = feed(
        machine, t, 0.6, h_shoulder=0.90, torso_tilt=50.0, motion=0.10
    )
    assert events == []
    assert machine.phase_of(1) == "TORSO_RISING"
    assert machine.snapshot_of(1).support == "SUPPORTED"

    t, _ = feed(machine, t, 0.6, h_shoulder=0.92, torso_tilt=30.0)
    assert machine.phase_of(1) == "UPRIGHT_IN_BED"

    t, _ = feed(
        machine,
        t,
        0.6,
        h_shoulder=0.92,
        torso_tilt=30.0,
        values=lambda i, _n: {"bed_edge_distance_m": 0.65 - i * 0.08},
    )
    assert machine.phase_of(1) == "SHIFTING_TO_EDGE"
    assert machine.snapshot_of(1).support == "SUPPORTED"

    t, _ = feed(
        machine, t, 0.6, h_shoulder=0.92, torso_tilt=30.0,
        bed_edge_distance_m=0.15,
    )
    assert machine.phase_of(1) == "EDGE_SITTING"

    t, _ = feed(
        machine, t, 0.6, h_shoulder=1.05, torso_tilt=20.0,
        supported_by_bed=None, bed_overlap=0.42, bed_support_fraction=0.42,
        bed_edge_distance_m=0.10,
    )
    assert machine.phase_of(1) == "ATTEMPTING_STAND"
    assert machine.snapshot_of(1).support == "PARTIAL"

    _t, events = feed(
        machine, t, 0.5, h_shoulder=1.10, torso_tilt=15.0,
        supported_by_bed=None, bed_overlap=0.0, bed_support_fraction=0.0,
        bed_edge_distance_m=-0.10,
    )
    assert machine.phase_of(1) == "OUT_OF_BED"
    assert len(events) == 1
    assert events[0].evidence["trigger"] == "out_of_bed"
    assert events[0].evidence["support"] == "UNSUPPORTED"
    assert events[0].evidence["onset_to_alert_s"] > 0


def test_fast_exit_can_skip_intermediate_phases():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    _t, events = feed(
        machine, t, 0.4, h_shoulder=1.1, torso_tilt=15.0,
        supported_by_bed=None, bed_overlap=0.0, bed_support_fraction=0.0,
        bed_edge_distance_m=-0.12, motion=0.4,
    )
    assert machine.phase_of(1) == "OUT_OF_BED"
    assert [event.evidence["trigger"] for event in events] == ["out_of_bed"]


def test_fast_support_loss_works_when_edge_distance_disappears_outside_zone():
    """The extractor cannot measure a bed-relative edge once association is
    gone, so recent support loss itself must remain evidence through the dwell.
    """
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    _t, events = feed(
        machine, t, 0.5, h_shoulder=1.1, torso_tilt=15.0,
        associated_bed=None, supported_by_bed=None,
        bed_overlap=0.0, bed_support_fraction=0.0,
        bed_edge_distance_m=None, motion=0.4,
    )
    assert machine.phase_of(1) == "OUT_OF_BED"
    assert len(events) == 1


def test_walking_through_bed_footprint_is_not_a_bed_exit():
    """Association alone is not proof that the mattress supported the track."""

    machine = BedExitStateMachine()
    t, events = feed(
        machine,
        0.0,
        0.8,
        h_shoulder=1.10,
        torso_tilt=15.0,
        supported_by_bed=None,
        bed_overlap=0.0,
        bed_support_fraction=0.0,
        bed_edge_distance_m=0.60,
        motion=0.1,
    )
    _t, later = feed(
        machine,
        t,
        0.8,
        h_shoulder=1.10,
        torso_tilt=15.0,
        supported_by_bed=None,
        bed_overlap=0.0,
        bed_support_fraction=0.0,
        values=lambda i, _n: {"bed_edge_distance_m": 0.50 - i * 0.12},
        motion=0.2,
    )
    assert events + later == []
    assert machine.phase_of(1) != "OUT_OF_BED"


def test_policy_evidence_reset_preserves_support_for_an_exit_in_progress():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, events = feed(
        machine,
        t,
        0.7,
        h_shoulder=0.95,
        torso_tilt=30.0,
        supported_by_bed=None,
        bed_overlap=0.40,
        bed_support_fraction=0.40,
        bed_edge_distance_m=0.12,
        motion=0.1,
    )
    assert events == []
    assert machine.snapshot_of(1).support == "PARTIAL"

    machine.reset_alert_evidence(1)
    snap = machine.snapshot_of(1)
    assert snap.bed_id == "ward-A-2026-10-08"
    assert snap.support == "PARTIAL"
    assert not snap.cusum_armed

    _t, events = feed(
        machine,
        t,
        0.8,
        h_shoulder=1.10,
        torso_tilt=15.0,
        supported_by_bed=None,
        bed_overlap=0.0,
        bed_support_fraction=0.0,
        bed_edge_distance_m=-0.12,
        motion=0.3,
    )
    assert [event.type for event in events] == ["BED_EXIT"]


def test_reclined_slide_reaches_shifting_without_a_rise():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, events = feed(
        machine,
        t,
        0.6,
        torso_tilt=70.0,
        h_shoulder=0.60,
        motion=0.15,
        values=lambda i, _n: {"bed_edge_distance_m": 0.65 - i * 0.08},
    )
    assert events == []
    assert machine.phase_of(1) == "SHIFTING_TO_EDGE"
    assert machine.snapshot_of(1).onset_t is None

    _t, events = feed(
        machine, t, 0.5, torso_tilt=75.0, h_shoulder=0.45,
        supported_by_bed=None, bed_overlap=0.0, bed_support_fraction=0.0,
        bed_edge_distance_m=-0.08, motion=0.2,
    )
    assert machine.phase_of(1) == "OUT_OF_BED"
    assert len(events) == 1
    assert events[0].t_trigger < events[0].t_alert
    assert events[0].latency_s == pytest.approx(machine.th.out_of_bed_dwell_s)


def test_reclined_tilt_hysteresis_keeps_rest_baseline_eligible():
    thresholds = BedExitThresholds(
        phase_dwell_s=0.0,
        cusum=CusumConfig(warmup_samples=5),
    )
    machine = BedExitStateMachine(thresholds)

    # Entering recline requires the higher threshold. The transition resets
    # the CUSUM, so the following samples must warm a fresh baseline.
    machine.update(Sample(track_id=1, t=0.0, torso_tilt=70.0))
    machine.update(Sample(track_id=1, t=0.1, torso_tilt=70.0))
    assert machine.phase_of(1) == "RECLINED"
    assert not machine.snapshot_of(1).cusum_armed

    # Once reclined, angles in the enter/exit deadband remain eligible stable
    # rest instead of silently ignoring the configured exit threshold.
    for index in range(5):
        machine.update(
            Sample(track_id=1, t=0.2 + index * DT, torso_tilt=52.0)
        )
    assert machine.phase_of(1) == "RECLINED"
    assert machine.snapshot_of(1).cusum_armed


def test_rail_climb_path_can_shift_while_still_supported():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    _t, events = feed(
        machine,
        t,
        0.7,
        torso_tilt=65.0,
        motion=0.2,
        values=lambda i, _n: {"bed_edge_distance_m": 0.70 - i * 0.07},
    )
    snap = machine.snapshot_of(1)
    assert events == []
    assert snap.phase == "SHIFTING_TO_EDGE"
    assert snap.support == "SUPPORTED"
    assert "edge_directed_motion" in snap.reasons


def test_brief_rise_can_return_to_recline_and_rearm():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, _ = feed(
        machine, t, 0.5, h_shoulder=0.92, torso_tilt=50.0, motion=0.1
    )
    assert machine.phase_of(1) == "TORSO_RISING"
    assert machine.snapshot_of(1).onset_t is not None

    t, events = feed(machine, t, 1.2, h_shoulder=0.60, torso_tilt=70.0)
    assert events == []
    assert machine.phase_of(1) == "RECLINED"
    # Explicit return resets the old elevated episode; it must warm a new rest
    # baseline instead of carrying a stale accumulator through the pause.
    assert machine.snapshot_of(1).onset_t is None
    assert not machine.snapshot_of(1).cusum_armed
    _t, _ = feed(machine, t, 1.5, h_shoulder=0.60, torso_tilt=70.0)
    assert machine.snapshot_of(1).cusum_armed


def test_edge_hysteresis_prevents_boundary_flicker():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, _ = feed(
        machine, t, 0.7, h_shoulder=0.9, torso_tilt=30.0,
        bed_edge_distance_m=0.15,
    )
    assert machine.phase_of(1) == "EDGE_SITTING"
    t, _ = feed(
        machine, t, 0.5, h_shoulder=0.9, torso_tilt=30.0,
        bed_edge_distance_m=0.28,
    )
    assert machine.phase_of(1) == "EDGE_SITTING"
    _t, _ = feed(
        machine, t, 0.7, h_shoulder=0.9, torso_tilt=30.0,
        bed_edge_distance_m=0.40,
    )
    assert machine.phase_of(1) == "UPRIGHT_IN_BED"


def test_low_confidence_and_unavailable_are_separate_from_retained_phase():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, _ = feed(
        machine,
        t,
        0.6,
        motion=0.2,
        values=lambda i, _n: {"bed_edge_distance_m": 0.65 - i * 0.08},
    )
    assert machine.phase_of(1) == "SHIFTING_TO_EDGE"

    machine.update(Sample(track_id=1, t=t, mean_conf=0.1))
    snap = machine.snapshot_of(1, t)
    assert snap.phase == "SHIFTING_TO_EDGE"
    assert snap.support == "SUPPORTED"
    assert snap.observation == "LOW_CONFIDENCE"

    machine.mark_unobserved(1, t + 0.2)
    snap = machine.snapshot_of(1, t + 0.2)
    assert snap.phase == "SHIFTING_TO_EDGE"
    assert snap.observation == "MONITORING_UNAVAILABLE"
    assert snap.reasons == ("track_not_observed",)


def test_long_unavailable_gap_keeps_stale_phase_but_drops_temporal_evidence():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, _ = feed(
        machine, t, 0.5, h_shoulder=0.92, torso_tilt=50.0, motion=0.1
    )
    assert machine.snapshot_of(1).onset_t is not None
    retained_phase = machine.phase_of(1)

    missing_t = t + machine.th.cusum.max_gap_s + 0.1
    machine.mark_unobserved(1, missing_t)
    snap = machine.snapshot_of(1, missing_t)
    assert snap.phase == retained_phase
    assert snap.observation == "MONITORING_UNAVAILABLE"
    assert snap.onset_t is None
    assert not snap.cusum_armed
    assert snap.edge_velocity_mps is None


def test_early_warning_is_explainable_but_shadow_only_by_default():
    machine = BedExitStateMachine()
    t = establish_recline(machine)
    t, events = feed(
        machine,
        t,
        1.2,
        h_shoulder=0.92,
        torso_tilt=45.0,
        motion=0.15,
        values=lambda i, _n: {"bed_edge_distance_m": 0.65 - i * 0.035},
    )
    snap = machine.snapshot_of(1, t - DT)
    assert events == []
    assert snap.early_warning_candidate
    assert snap.cusum_g > 0
    assert snap.onset_t is not None
    assert "edge_directed_motion" in snap.reasons


def test_early_warning_requires_explicit_opt_in_and_emits_once():
    machine = BedExitStateMachine(
        BedExitThresholds(emit_early_warning=True, emit_exit_event=False)
    )
    t = establish_recline(machine)
    _t, events = feed(
        machine,
        t,
        2.0,
        h_shoulder=0.92,
        torso_tilt=45.0,
        motion=0.15,
        values=lambda i, _n: {"bed_edge_distance_m": 0.65 - i * 0.02},
    )
    assert len(events) == 1
    assert events[0].type == "BED_EXIT_WARNING"
    assert events[0].evidence["trigger"] == "early_warning"
    assert "baseline_mean" in events[0].evidence


def test_static_near_edge_rise_does_not_become_operational_warning():
    machine = BedExitStateMachine(
        BedExitThresholds(emit_early_warning=True, emit_exit_event=False)
    )
    t = establish_recline(machine)
    _t, events = feed(
        machine,
        t,
        2.0,
        h_shoulder=0.92,
        torso_tilt=45.0,
        motion=0.02,
        bed_edge_distance_m=0.15,
    )
    assert events == []
    assert not machine.snapshot_of(1).early_warning_candidate


def test_depth_dropout_fallback_cannot_accumulate_as_cusum_movement():
    machine = BedExitStateMachine(
        BedExitThresholds(emit_early_warning=True, emit_exit_event=False)
    )
    t = establish_recline(machine)
    _t, events = feed(
        machine,
        t,
        2.0,
        torso_tilt=50.0,
        motion=0.1,
        values=lambda i, _n: {
            "h_shoulder": 1.40 if i % 2 == 0 else 0.60,
            "h_shoulder_source": "monocular" if i % 2 == 0 else "depth",
            "bed_edge_distance_m": 0.65 - i * 0.01,
        },
    )
    assert events == []
    snap = machine.snapshot_of(1)
    assert not snap.cusum_armed
    assert not snap.early_warning_candidate


def test_tracks_are_independent_and_lifecycle_is_explicit():
    machine = BedExitStateMachine()
    t1 = establish_recline(machine, track_id=1)
    _t2 = establish_recline(machine, track_id=2)
    _t, _ = feed(
        machine, t1, 0.6, track_id=1, h_shoulder=0.9, torso_tilt=50.0
    )
    assert machine.phase_of(1) == "TORSO_RISING"
    assert machine.phase_of(2) == "RECLINED"

    machine.retain_only({2})
    assert machine.phase_of(1) == "UNKNOWN"
    machine.reset(2)
    assert machine.phase_of(2) == "UNKNOWN"


def test_dated_zone_name_does_not_need_the_word_bed():
    machine = BedExitStateMachine()
    establish_recline(machine)
    snap = machine.snapshot_of(1)
    assert snap.bed_id == "ward-A-2026-10-08"


def test_threshold_validation():
    with pytest.raises(ValueError):
        BedExitThresholds(support_exit=0.8, support_enter=0.5)
