"""Fall decision tests.

Every test here is a synthetic sequence of metric features fed through the
state machine at 15 fps. No camera, no video, no pose model, no fixtures on
disk. That is the property that makes the decision layer developable at all
while the hardware is on somebody else's desk, and it means these run in
milliseconds and never flake.

The scenarios are chosen to mirror what actually happens in a ward cubicle.
The negatives matter more than the positives: a detector that catches every
fall and also cries wolf twice a shift will be switched off, so most of these
tests assert that **nothing** fires.
"""

from __future__ import annotations

import math

import pytest

from ahfd.detect import Event, FallStateMachine, FallThresholds
from ahfd.features import Features

FPS = 15.0
DT = 1.0 / FPS

# Representative floor_spread values, from the geometry: an upright body's
# head ray hits the floor many metres past its feet, a fallen body's joints
# genuinely lie within about a body length.
SPREAD_UPRIGHT = 11.0
SPREAD_DOWN = 1.8

BED_TOP = 0.60


def feats(
    t: float,
    *,
    track_id: int = 1,
    h_torso: float | None = 1.15,
    floor_spread: float = SPREAD_UPRIGHT,
    v_z: float = 0.0,
    motion: float = 0.0,
    n_valid_kp: int = 15,
    mean_conf: float = 0.8,
    zones: tuple[str, ...] = (),
    supported_by_bed: str | None = None,
    bed_top_m: float | None = None,
    bed_risk: str | None = None,
    contact: tuple[float, float] | None = (0.0, 5.0),
    excluded: bool = False,
) -> Features:
    return Features(
        track_id=track_id,
        t=t,
        contact_xy=contact,
        range_m=5.6,
        h_torso=h_torso,
        h_head=None if h_torso is None else h_torso + 0.45,
        h_max=None if h_torso is None else h_torso + 0.5,
        h_min=0.05,
        h_ankle_min=0.05,
        floor_spread=floor_spread,
        v_z=v_z,
        motion=motion,
        n_valid_kp=n_valid_kp,
        mean_conf=mean_conf,
        zones=zones,
        supported_by_bed=supported_by_bed,
        bed_top_m=bed_top_m,
        bed_risk=bed_risk,
        in_excluded_zone=excluded,
    )


def run(
    machine: FallStateMachine, frames: list[Features]
) -> list[Event]:
    events = []
    for f in frames:
        event = machine.update(f)
        if event is not None:
            events.append(event)
    return events


def hold(start: float, seconds: float, **kwargs) -> list[Features]:
    """A steady stretch of identical frames."""
    n = int(seconds * FPS)
    return [feats(start + i * DT, **kwargs) for i in range(n)]


def types_of(events: list[Event]) -> list[str]:
    return [e.type for e in events]


def fall_sequence(
    *,
    settle_s: float = 2.0,
    down_s: float = 10.0,
    down_motion: float = 0.02,
    zones: tuple[str, ...] = (),
) -> list[Features]:
    """Standing, then a fast drop, then down on the floor.

    The drop supplies both a sustained negative v_z and a real loss of height,
    which is what the trigger looks for.
    """
    frames = hold(0.0, settle_s, zones=zones)
    t = settle_s

    # 0.4 s of falling: torso from 1.15 m to about 0.2 m.
    for i in range(int(0.4 * FPS)):
        frac = (i + 1) / (0.4 * FPS)
        frames.append(
            feats(
                t + i * DT,
                h_torso=1.15 - 0.95 * frac,
                v_z=-2.2,
                floor_spread=SPREAD_UPRIGHT if frac < 0.6 else SPREAD_DOWN,
                motion=0.9,
                zones=zones,
            )
        )
    t += 0.4

    frames += hold(
        t,
        down_s,
        h_torso=0.20,
        floor_spread=SPREAD_DOWN,
        motion=down_motion,
        zones=zones,
    )
    return frames


# ---------------------------------------------------------------- negatives


class TestNothingFiresWhenItShouldNot:
    def test_standing_still_is_silent(self):
        m = FallStateMachine()
        assert run(m, hold(0.0, 30.0)) == []
        assert m.state_of(1) == "UPRIGHT"

    def test_walking_about_is_silent(self):
        """Moving around the cubicle, with mild vertical noise."""
        m = FallStateMachine()
        frames = [
            feats(i * DT, h_torso=1.15 + 0.03 * math.sin(i * 0.7), v_z=0.1 * math.cos(i * 0.7))
            for i in range(int(60 * FPS))
        ]
        assert run(m, frames) == []

    def test_lying_in_bed_is_silent(self):
        """The dominant false positive. A patient in bed is horizontal and low,
        so only the bed zone distinguishes it from a fall.
        """
        m = FallStateMachine()
        frames = hold(
            0.0,
            60.0,
            h_torso=BED_TOP + 0.05,
            floor_spread=SPREAD_DOWN,
            motion=0.01,
            zones=("bed_2",),
            supported_by_bed="bed_2",
            bed_top_m=BED_TOP,
        )
        assert run(m, frames) == []
        assert m.state_of(1) == "IN_BED"

    def test_getting_into_bed_is_not_a_fall(self):
        """A real drop in height that ends supported by a bed.

        Getting into bed produces the same velocity signature as falling out
        of one. Only the bed geometry tells them apart, which is why the
        supporting-bed test is geometric rather than a height comparison.
        """
        m = FallStateMachine()
        frames = fall_sequence(down_s=20.0, zones=("bed_2",))
        frames = [
            Features(
                **{
                    **f.__dict__,
                    "bed_top_m": BED_TOP,
                    # Supported only once they have come to rest on it.
                    "supported_by_bed": (
                        "bed_2"
                        if f.h_torso is not None and f.h_torso < 0.5
                        else None
                    ),
                    "h_torso": (
                        BED_TOP + 0.05
                        if f.h_torso is not None and f.h_torso < 0.5
                        else f.h_torso
                    ),
                }
            )
            for f in frames
        ]
        assert types_of(run(m, frames)) == []

    def test_low_confidence_track_is_excluded(self):
        m = FallStateMachine()
        frames = fall_sequence()
        frames = [Features(**{**f.__dict__, "n_valid_kp": 4}) for f in frames]
        assert run(m, frames) == []
        assert m.state_of(1) == "LOW_CONFIDENCE"

    def test_missing_geometry_is_excluded(self):
        """No floor contact means no metric height, so no decision."""
        m = FallStateMachine()
        frames = hold(0.0, 20.0, h_torso=None, contact=None)
        assert run(m, frames) == []

    def test_excluded_zone_is_ignored(self):
        """A doorway or a mirror is not our patient."""
        m = FallStateMachine()
        frames = [Features(**{**f.__dict__, "in_excluded_zone": True}) for f in fall_sequence()]
        assert run(m, frames) == []

    def test_trigger_never_predates_the_track_age_guard(self):
        """The id-switch guard, stated precisely.

        A genuine fall in the first second is still caught -- and should be.
        What the guard rules out is a *decision* taken while the track is too
        young to be trusted, so no trigger timestamp may predate
        min_track_age_s. Asserting "nothing fires at all" would be wrong: it
        would mean discarding real falls that happen shortly after a patient
        walks into view.
        """
        m = FallStateMachine()
        th = FallThresholds()
        events = run(m, fall_sequence(settle_s=0.3, down_s=12.0))

        assert events, "a real fall should still be reported"
        for event in events:
            assert event.t_trigger >= th.min_track_age_s

    def test_a_track_appearing_already_down_claims_no_impact(self):
        """The real id-switch shape: a track materialises mid-scene.

        With no height history there is no observed drop, so the machine must
        not manufacture one. It may eventually report PERSON_DOWN -- which is
        honest, and says 'somebody is on the floor' rather than 'I saw them
        fall'.
        """
        m = FallStateMachine()
        frames = hold(0.0, 12.0, h_torso=0.20, floor_spread=SPREAD_DOWN, motion=0.02)
        events = types_of(run(m, frames))
        assert "FALL_CONFIRMED" not in events
        assert "FALL_SUSPECTED" not in events

    def test_person_down_but_moving_is_not_confirmed(self):
        """Someone doing floor exercises, or actively getting up."""
        m = FallStateMachine()
        frames = fall_sequence(down_s=20.0, down_motion=0.6)
        assert "FALL_CONFIRMED" not in types_of(run(m, frames))


# ---------------------------------------------------------------- positives


class TestFallDetection:
    def test_confirms_a_fall_after_the_person_stays_down(self):
        m = FallStateMachine()
        events = run(m, fall_sequence(down_s=12.0))
        assert "FALL_CONFIRMED" in types_of(events)

    def test_suspects_early_then_confirms_late(self):
        """Two-tier alerting: the screen reacts fast, the pager waits."""
        m = FallStateMachine()
        events = run(m, fall_sequence(down_s=12.0))

        suspected = next(e for e in events if e.type == "FALL_SUSPECTED")
        confirmed = next(e for e in events if e.type == "FALL_CONFIRMED")

        th = FallThresholds()
        assert suspected.t_alert < confirmed.t_alert
        assert confirmed.t_alert - suspected.t_alert > 5.0
        # Confirmation must not arrive before the person has stayed down.
        assert confirmed.evidence["down_s"] >= th.confirm_s - 0.1

    def test_evidence_explains_the_alert(self):
        """The first question after any alert is 'why'. It must be answerable."""
        m = FallStateMachine()
        confirmed = next(
            e for e in run(m, fall_sequence(down_s=12.0)) if e.type == "FALL_CONFIRMED"
        )
        for key in ("peak_vz", "h_before", "h_after", "down_s", "floor_spread"):
            assert key in confirmed.evidence

        assert confirmed.evidence["peak_vz"] < -0.9
        assert confirmed.evidence["h_before"] > confirmed.evidence["h_after"]

    def test_only_one_alert_per_fall(self):
        """A cooldown stops one incident becoming a stream of pages."""
        m = FallStateMachine()
        events = run(m, fall_sequence(down_s=40.0))
        assert types_of(events).count("FALL_CONFIRMED") == 1

    def test_recovery_cancels_the_alarm(self):
        """Went down, got straight back up. Logged, but nobody is paged."""
        m = FallStateMachine()
        frames = fall_sequence(down_s=3.0)  # shorter than confirm_s
        frames += hold(frames[-1].t + DT, 5.0, h_torso=1.15)

        events = run(m, frames)
        assert "NEAR_MISS" in types_of(events)
        assert "FALL_CONFIRMED" not in types_of(events)

    def test_latency_is_reported(self):
        m = FallStateMachine()
        confirmed = next(
            e for e in run(m, fall_sequence(down_s=12.0)) if e.type == "FALL_CONFIRMED"
        )
        # Trigger to page: the confirmation window plus the fall itself.
        assert 7.0 < confirmed.latency_s < 12.0


class TestSlowSlump:
    def test_slow_descent_with_no_impact_is_caught(self):
        """A frail patient sliding down produces no velocity spike at all.

        A pure impact detector misses this entirely, and clinicians ask about
        it immediately, so it has its own path.
        """
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        # Ease to the floor over 12 s -- far too slow to trigger on velocity.
        n = int(12.0 * FPS)
        for i in range(n):
            frac = (i + 1) / n
            frames.append(
                feats(
                    2.0 + i * DT,
                    h_torso=1.15 - 0.95 * frac,
                    v_z=-0.08,
                    floor_spread=SPREAD_UPRIGHT if frac < 0.8 else SPREAD_DOWN,
                    motion=0.05,
                )
            )
        frames += hold(14.0, 30.0, h_torso=0.20, floor_spread=SPREAD_DOWN, motion=0.02)

        events = run(m, frames)
        assert "PERSON_DOWN" in types_of(events)
        down = next(e for e in events if e.type == "PERSON_DOWN")
        assert down.evidence["no_impact_detected"] is True

    def test_person_down_needs_the_full_wait(self):
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        frames += hold(2.0, 10.0, h_torso=0.2, floor_spread=SPREAD_DOWN, motion=0.02)
        assert "PERSON_DOWN" not in types_of(run(m, frames))

    def test_sitting_on_the_floor_deliberately_still_reports(self):
        """Accepted false positive, and an honest one.

        Someone who sits on the floor on purpose and stays there is
        indistinguishable from someone who has been down a long time.
        Reporting it is the safe direction, and the evidence lets a nurse
        dismiss it in a second.

        Note the descent is deliberately gradual. Lowering yourself takes a
        couple of seconds, which is far too slow to trip the impact trigger --
        so this arrives as PERSON_DOWN rather than FALL_CONFIRMED, and the
        distinction is visible to whoever reads the alert.
        """
        m = FallStateMachine()
        frames = hold(0.0, 2.0)

        n = int(2.5 * FPS)
        for i in range(n):
            frac = (i + 1) / n
            frames.append(
                feats(
                    2.0 + i * DT,
                    h_torso=1.15 - 0.70 * frac,
                    v_z=-0.28,
                    floor_spread=SPREAD_UPRIGHT if frac < 0.7 else SPREAD_DOWN,
                    motion=0.1,
                )
            )
        frames += hold(4.5, 30.0, h_torso=0.45, floor_spread=SPREAD_DOWN, motion=0.02)

        events = types_of(run(m, frames))
        assert "PERSON_DOWN" in events
        assert "FALL_CONFIRMED" not in events


class TestBedExit:
    def test_sitting_on_the_bed_edge_raises_a_precursor(self):
        """Preventing a fall beats detecting one, and this is far easier."""
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        frames += hold(
            2.0, 8.0, h_torso=0.55, zones=("bed_2",), bed_top_m=None, motion=0.05
        )
        events = run(m, frames)
        assert "BED_EXIT" in types_of(events)

    def test_bed_exit_fires_once_not_continuously(self):
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        frames += hold(2.0, 30.0, h_torso=0.55, zones=("bed_2",), motion=0.05)
        assert types_of(run(m, frames)).count("BED_EXIT") == 1

    def test_sitting_away_from_a_bed_is_not_a_bed_exit(self):
        """A visitor in a chair is not a patient about to get up."""
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        frames += hold(2.0, 20.0, h_torso=0.55, zones=("floor",), motion=0.05)
        assert "BED_EXIT" not in types_of(run(m, frames))

    def _bed_exit_event(self, bed_risk):
        m = FallStateMachine()
        frames = hold(0.0, 2.0)
        frames += hold(
            2.0, 8.0, h_torso=0.55, zones=("bed_2",), bed_risk=bed_risk, motion=0.05
        )
        events = [e for e in run(m, frames) if e.type == "BED_EXIT"]
        assert len(events) == 1
        return events[0]

    def test_high_risk_bed_exit_is_an_alert(self):
        """The graded response: a patient who should not self-exit -> alert."""
        e = self._bed_exit_event("high")
        assert e.severity == 3
        assert e.evidence["bed_risk"] == "high"

    def test_low_risk_bed_exit_is_awareness_not_alarm(self):
        """A patient cleared to mobilise -> low priority, no alarm. This is the
        alarm-fatigue fix: same action, gentler response."""
        assert self._bed_exit_event("low").severity == 1

    def test_none_risk_bed_exit_is_informational(self):
        assert self._bed_exit_event("none").severity == 0

    def test_unknown_risk_defaults_to_a_cautious_warning(self):
        """Not-yet-assessed is not the same as safe."""
        assert self._bed_exit_event("unknown").severity == 2
        # A bed with no risk set at all lands on the same cautious default.
        assert self._bed_exit_event(None).severity == 2

    def test_same_action_different_urgency(self):
        """The whole point, stated as one assertion: identical bed exit,
        severity driven entirely by the bed's risk level."""
        sev = {r: self._bed_exit_event(r).severity for r in ("none", "low", "high")}
        assert sev["none"] < sev["low"] < sev["high"]


class TestMultipleTracks:
    def test_tracks_are_independent(self):
        """One patient falls while a nurse stands beside the bed."""
        m = FallStateMachine()
        falling = fall_sequence(down_s=12.0)
        events = []
        for f in falling:
            nurse = Features(**{**f.__dict__, "track_id": 2, "h_torso": 1.15,
                                "floor_spread": SPREAD_UPRIGHT, "v_z": 0.0,
                                "motion": 0.1})
            for frame in (f, nurse):
                event = m.update(frame)
                if event is not None:
                    events.append(event)

        confirmed = [e for e in events if e.type == "FALL_CONFIRMED"]
        assert len(confirmed) == 1
        assert confirmed[0].track_id == 1
        assert m.state_of(2) == "UPRIGHT"

    def test_retain_only_drops_dead_tracks(self):
        m = FallStateMachine()
        run(m, hold(0.0, 2.0))
        assert m.state_of(1) == "UPRIGHT"
        m.retain_only(set())
        assert m.state_of(1) == "UNKNOWN"


class TestThresholdsAreConfigurable:
    def test_confirm_window_is_respected(self):
        """Tuning must actually change behaviour, or config is decoration."""
        quick = FallStateMachine(FallThresholds(confirm_s=2.0))
        slow = FallStateMachine(FallThresholds(confirm_s=30.0))
        frames = fall_sequence(down_s=6.0)

        assert "FALL_CONFIRMED" in types_of(run(quick, frames))
        assert "FALL_CONFIRMED" not in types_of(run(slow, frames))

    def test_no_learned_parameters(self):
        """The machine is pure configuration.

        Documents an intentional design property: there is nothing to train,
        so it works before any fall data exists and every alert is explainable.
        """
        th = FallThresholds()
        for name, value in vars(th).items():
            assert isinstance(value, (int, float, tuple)), name
