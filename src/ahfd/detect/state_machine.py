"""The fall decision.

A fall is a sequence, not a frame. Treating it as a frame -- "vertical speed
exceeded a threshold, therefore CRITICAL" -- is the single most common way
these systems become unusable, because a nurse sitting down quickly, a
keypoint flicker, or a tracker id swap all produce one bad frame. The nurses
then stop trusting the alerts, and the project has failed even though the
detector "works".

So this is a state machine over three phases:

    trigger   the torso drops fast, or drops a long way quickly
    rest      the body is genuinely down on the floor, not on a bed
    confirm   it stays down and still for several seconds

Only the third emits the alert that pages anybody. Something appears on screen
at 1.5 s so the system looks responsive, but the pager waits ~8 s, by which
point the person has demonstrably stayed down. That buys a large reduction in
false alarms for a latency cost that does not matter clinically -- nobody is
helped meaningfully faster by 8 seconds, and everybody is harmed by an alarm
that cried wolf.

There is a second, independent path. A frail patient sliding slowly to the
floor never produces a velocity spike at all, and neither does a fall the
system only saw the aftermath of, through a curtain or an occlusion. So a
track that simply *is* down, outside a bed, and still for long enough raises
`PERSON_DOWN` regardless of how it got there. Clinicians ask about this case
immediately, and a pure impact detector misses all of it.

Every threshold lives in `FallThresholds` and comes from config. None are
learned: the machine has no trained parameters, which is what lets it work
before any fall data exists and lets any alert be explained in plain language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ahfd.detect.events import Event
from ahfd.features.extractor import Features

# How urgent a bed exit is, by the bed's fall-risk level. This is the answer to
# alarm fatigue: the identical action produces a quiet dashboard status for a
# patient cleared to self-exit and a real alert for a high-risk one.
#   0 informational (dashboard status, no alarm)   1 low-priority notice
#   2 warning                                       3 alert (page a nurse)
# Patient severity tier, set per bed. It follows the three Morse Fall Scale
# risk bands (score 0-125 over 6 items): high >=45, moderate 25-44, low 0-24.
# The ward assigns it from the patient's Morse assessment; "unknown" stays
# cautious (not-yet-assessed is not safe). Our scope is bed-exit PREVENTION, so
# the tier drives how a sit-up is treated:
#   high     (Morse >=45)   must not exit unassisted -> any sit-up is an alert
#   moderate (Morse 25-44)  transfer risk            -> a sustained sit-up alerts
#   low      (Morse 0-24)   supervised mobilisation  -> awareness on bed-exit
#   none                    below scope / cleared    -> logged, no alarm
# "moderate" and "medium" are accepted as the same band (Morse uses "moderate").
BED_EXIT_SEVERITY_BY_RISK: dict[str, int] = {
    "none": 0,     # independent / cleared -> awareness only
    "low": 1,
    "medium": 2,
    "moderate": 2,  # Morse term for "medium"
    "high": 3,     # must not exit unassisted -> alert
    "confused": 3,  # mentally-ill / exit-seeking: allowed to move, but high exit risk
    "wander": 3,    # alias for "confused"
    "unknown": 2,
}

# How long a sit near the bed must persist before it alerts, BY tier. Only the
# "high" (immobile) tier alarms on the sit-up itself -- any movement is an
# alert. Every other tier is ALLOWED to move/sit in bed, so the sit-up never
# alarms (infinite dwell); their *exit* is caught by the stand-up trigger in
# _steady_state instead. "unknown" stays cautious and alerts on the sit-up.
BED_EXIT_DWELL_BY_RISK: dict[str, float] = {
    "high": 1.0,        # immobile -> alert on the first sustained sit-up
    "unknown": 2.0,     # not assessed -> cautious, alerts on the sit-up
    "medium": float("inf"),
    "moderate": float("inf"),
    "confused": float("inf"),
    "wander": float("inf"),
    "low": float("inf"),
    "none": float("inf"),
}

# Tiers cleared to leave the bed on their own (a fit patient, rare): NO bed-exit
# alert at all -- neither the sit-up nor the stand-up fires. Fall detection still
# applies to them, which is their safety net.
SELF_EXIT_ALLOWED: frozenset[str] = frozenset({"low", "none"})

State = Literal[
    "UNKNOWN",
    "UPRIGHT",
    "SITTING",
    "IN_BED",
    "FALLING",
    "ON_GROUND",
    "RECOVERED",
    "LOW_CONFIDENCE",
]


@dataclass(frozen=True)
class FallThresholds:
    """Every tunable, in metres, seconds and metres per second.

    The starting values assume an adult patient and a camera at 2.4-2.8 m
    looking down 18-25 degrees, which is the Class C ward geometry. They are
    a *starting point to be tuned against staged clips*, not settled numbers.
    """

    # --- posture bands -------------------------------------------------
    upright_h: float = 0.70  # torso above this, and not down -> UPRIGHT
    sitting_h: tuple[float, float] = (0.40, 0.70)
    bed_band: tuple[float, float] = (-0.25, 0.45)  # relative to bed top_m

    # --- what "down on the floor" means --------------------------------
    # floor_spread is the primary test, and it is the only one that is truly
    # range-independent. Measured on projected bodies, a prone person gives
    # floor_spread 1.64 m at both 4 m and 6 m -- identical -- while their
    # h_torso reads 0.62 m at 4 m but 0.47 m at 6 m, because the
    # vertical-line bias grows as people get closer.
    #
    # So h_torso is a loose sanity guard, not a discriminator. It sits well
    # above any prone reading (worst case ~0.7 m at close range) and well
    # below a standing one (~1.06 m at every range). Tightening it to
    # something that looks reasonable, like 0.60 m, silently loses falls at
    # the near bed -- which is exactly what happened before this comment
    # existed.
    down_spread: tuple[float, float] = (0.9, 3.0)  # metres, ~a body length
    down_h_torso: float = 0.90

    # Seated bodies keep a near-vertical torso even when floor_spread reads like
    # a fallen one's; torso_tilt (image angle, calibration-free) tells them
    # apart. A torso standing up in frame (tilt <= seated_tilt_max) within
    # seated_h is SITTING -- decided *before* the down test, so sitting stops
    # being swallowed as ON_GROUND without loosening the fall thresholds.
    seated_tilt_max: float = 35.0
    seated_h: tuple[float, float] = (0.70, 1.00)

    # --- trigger -------------------------------------------------------
    vz_trigger: float = -0.90  # m/s sustained
    vz_frames: int = 3
    drop_trigger: float = 0.45  # or this much height lost quickly
    drop_window_s: float = 0.8
    min_track_age_s: float = 1.0  # id-switch guard
    rest_deadline_s: float = 2.5  # trigger must be followed by rest this soon

    # --- confirmation --------------------------------------------------
    suspect_s: float = 1.5  # on-screen flag
    confirm_s: float = 8.0  # pages a nurse
    confirm_motion_max: float = 0.15  # m/s
    recover_h: float = 0.70  # back above this -> NEAR_MISS, no alarm

    # --- slow-slump path -----------------------------------------------
    slow_down_s: float = 20.0

    # --- bed exit ------------------------------------------------------
    bed_exit_s: float = 3.0
    # In-bed movement speed (m/s) that alerts for the immobile ("high") tier:
    # any sustained motion this fast -- turning, reaching, starting to rise --
    # is itself an alert, before it becomes a sit-up.
    in_bed_move: float = 0.10

    # --- quality gate --------------------------------------------------
    min_valid_kp: int = 8
    min_mean_conf: float = 0.40

    # --- hygiene -------------------------------------------------------
    cooldown_s: float = 60.0


@dataclass
class _TrackState:
    state: State = "UNKNOWN"
    first_seen: float | None = None
    state_since: float = 0.0

    vz_streak: int = 0
    trigger_t: float | None = None
    trigger_h: float | None = None
    peak_vz: float = 0.0

    down_since: float | None = None
    suspected: bool = False

    sitting_since: float | None = None
    bed_exit_emitted: bool = False
    # Cross-transition bed-exit state. The sit->stand reset clears sitting_since,
    # so remember separately that the person was on/at a bed, the bed's tier
    # (the contact leaves the bed zone as they stand), and whether this
    # departure has already alerted -- so an exit fires at most once.
    bed_assoc_recent: bool = False
    bed_risk_recent: str | None = None
    exit_alerted: bool = False
    bed_move_emitted: bool = False  # one in-bed-movement alert per movement burst

    last_alert_t: float | None = None
    height_log: list[tuple[float, float]] = field(default_factory=list)

    def age(self, now: float) -> float:
        return 0.0 if self.first_seen is None else now - self.first_seen


class FallStateMachine:
    """Per-track fall decision.

    `update` takes the timestamp as an argument and holds no wall clock, so the
    whole decision layer is testable from synthetic feature sequences -- no
    camera, no video, no pose model. That is deliberate, and it is what let
    this be written and verified while the D435i was unplugged.
    """

    def __init__(self, thresholds: FallThresholds | None = None):
        self.th = thresholds or FallThresholds()
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------ helpers

    def state_of(self, track_id: int) -> State:
        ts = self._tracks.get(track_id)
        return ts.state if ts else "UNKNOWN"

    def _set_state(self, ts: _TrackState, state: State, now: float) -> None:
        if ts.state != state:
            ts.state = state
            ts.state_since = now

    def _is_down(self, f: Features) -> bool:
        """Is this body horizontal and on the floor?

        Leans on `floor_spread` rather than joint heights. A fallen person
        really is on the floor, so projecting their joints onto it gives a
        consistent patch about a body long. An upright person's head ray hits
        the floor metres past their feet, or misses it, giving a spread far
        outside the band.
        """
        lo, hi = self.th.down_spread
        if not (lo <= f.floor_spread <= hi):
            return False
        if f.h_torso is not None and f.h_torso > self.th.down_h_torso:
            return False
        return True

    def _on_bed(self, f: Features) -> bool:
        """Is this body supported by a bed rather than the floor?

        Answered geometrically upstream, in FeatureExtractor._supporting_bed,
        by testing each bed at its own surface height. Doing it here from
        h_torso would not work: the vertical-line height of somebody lying in
        bed is computed from a floor contact point that is itself wrong for an
        elevated body, so the number it produces has no fixed relationship to
        the bed's height.
        """
        return f.supported_by_bed is not None

    def _height_lost(self, ts: _TrackState, now: float, current: float) -> float:
        """Height dropped within the recent window, metres."""
        window = [h for (t, h) in ts.height_log if now - t <= self.th.drop_window_s]
        return (max(window) - current) if window else 0.0

    def _cooling_down(self, ts: _TrackState, now: float) -> bool:
        return (
            ts.last_alert_t is not None
            and now - ts.last_alert_t < self.th.cooldown_s
        )

    # ------------------------------------------------------------- update

    def update(self, f: Features) -> Event | None:
        """Advance one track by one frame. Returns an event, or None."""
        ts = self._tracks.setdefault(f.track_id, _TrackState())
        now = f.t
        if ts.first_seen is None:
            ts.first_seen = now

        # --- quality gate ------------------------------------------------
        # Below this, exclude the track from decisions entirely rather than
        # guessing. Excluded time is reported by the evaluator so a good
        # false-alarm rate cannot hide behind poor coverage.
        if (
            f.n_valid_kp < self.th.min_valid_kp
            or f.mean_conf < self.th.min_mean_conf
            or not f.has_geometry()
        ):
            self._set_state(ts, "LOW_CONFIDENCE", now)
            ts.vz_streak = 0
            return None

        if f.in_excluded_zone:
            self._set_state(ts, "UNKNOWN", now)
            return None

        assert f.h_torso is not None  # has_geometry() guarantees this
        ts.height_log.append((now, f.h_torso))
        ts.height_log = [
            (t, h) for (t, h) in ts.height_log if now - t <= self.th.drop_window_s * 3
        ]

        zone = f.zones[0] if f.zones else None
        down = self._is_down(f)
        on_bed = self._on_bed(f)

        # --- a fall already in progress: confirm, or cancel --------------
        # Gated on an *active trigger*, not on the state alone. The slow-slump
        # path also parks a track in ON_GROUND, and without this check the
        # next frame would be handed to the impact machinery and a gradual
        # slide would be reported as a confirmed impact fall -- inventing a
        # peak velocity that never happened, in the evidence a nurse reads.
        if ts.trigger_t is not None and ts.state in ("FALLING", "ON_GROUND"):
            event = self._progress_fall(ts, f, now, down, on_bed, zone)
            if event is not None:
                return event
            if ts.state in ("FALLING", "ON_GROUND"):
                return None

        # --- trigger detection -------------------------------------------
        if f.v_z <= self.th.vz_trigger:
            ts.vz_streak += 1
            ts.peak_vz = min(ts.peak_vz, f.v_z)
        else:
            ts.vz_streak = 0

        fast_enough = ts.vz_streak >= self.th.vz_frames
        far_enough = self._height_lost(ts, now, f.h_torso) >= self.th.drop_trigger
        old_enough = ts.age(now) >= self.th.min_track_age_s

        if (fast_enough or far_enough) and old_enough and not on_bed:
            ts.trigger_t = now
            ts.trigger_h = max(
                (h for (_, h) in ts.height_log), default=f.h_torso
            )
            ts.peak_vz = min(ts.peak_vz, f.v_z)
            ts.suspected = False
            ts.down_since = now if down else None
            self._set_state(ts, "FALLING", now)
            return None

        # --- steady states, and the slow-slump path ----------------------
        return self._steady_state(ts, f, now, down, on_bed, zone)

    # ------------------------------------------------------- sub-machines

    def _progress_fall(
        self,
        ts: _TrackState,
        f: Features,
        now: float,
        down: bool,
        on_bed: bool,
        zone: str | None,
    ) -> Event | None:
        assert f.h_torso is not None

        # Got back up: a near miss, and free training data.
        if f.h_torso >= self.th.recover_h and not down:
            evidence = {
                "peak_vz": round(ts.peak_vz, 2),
                "h_before": round(ts.trigger_h or 0.0, 2),
                "recovered_after_s": round(
                    now - (ts.down_since or ts.trigger_t or now), 1
                ),
            }
            trigger_t = ts.trigger_t or now
            self._reset_fall(ts)
            self._set_state(ts, "RECOVERED", now)
            ts.last_alert_t = now
            return Event(
                type="NEAR_MISS",
                track_id=f.track_id,
                t_trigger=trigger_t,
                t_alert=now,
                zone=zone,
                evidence=evidence,
            )

        # Landed on a bed, not the floor: that is getting into bed.
        if on_bed:
            self._reset_fall(ts)
            self._set_state(ts, "IN_BED", now)
            return None

        if not down:
            # Trigger fired but the body never came to rest in time.
            if (
                ts.trigger_t is not None
                and now - ts.trigger_t > self.th.rest_deadline_s
            ):
                self._reset_fall(ts)
                self._set_state(ts, "UNKNOWN", now)
            return None

        if ts.down_since is None:
            ts.down_since = now
        self._set_state(ts, "ON_GROUND", now)

        elapsed = now - ts.down_since
        still = f.motion <= self.th.confirm_motion_max

        if not ts.suspected and elapsed >= self.th.suspect_s:
            ts.suspected = True
            return Event(
                type="FALL_SUSPECTED",
                track_id=f.track_id,
                t_trigger=ts.trigger_t or ts.down_since,
                t_alert=now,
                zone=zone,
                evidence={
                    "peak_vz": round(ts.peak_vz, 2),
                    "h_torso": round(f.h_torso, 2),
                    "floor_spread": round(f.floor_spread, 2),
                },
            )

        if elapsed >= self.th.confirm_s and still and not self._cooling_down(ts, now):
            evidence = {
                "peak_vz": round(ts.peak_vz, 2),
                "h_before": round(ts.trigger_h or 0.0, 2),
                "h_after": round(f.h_torso, 2),
                "floor_spread": round(f.floor_spread, 2),
                "down_s": round(elapsed, 1),
                "motion": round(f.motion, 3),
                "range_m": round(f.range_m or 0.0, 1),
                "n_valid_kp": f.n_valid_kp,
            }
            trigger_t = ts.trigger_t or ts.down_since
            ts.last_alert_t = now
            self._reset_fall(ts)
            self._set_state(ts, "ON_GROUND", now)
            return Event(
                type="FALL_CONFIRMED",
                track_id=f.track_id,
                t_trigger=trigger_t,
                t_alert=now,
                zone=zone,
                evidence=evidence,
            )

        return None

    def _steady_state(
        self,
        ts: _TrackState,
        f: Features,
        now: float,
        down: bool,
        on_bed: bool,
        zone: str | None,
    ) -> Event | None:
        assert f.h_torso is not None
        th = self.th

        if on_bed:
            self._set_state(ts, "IN_BED", now)
            ts.sitting_since = None
            ts.bed_exit_emitted = False
            ts.down_since = None
            ts.bed_assoc_recent = True
            ts.bed_risk_recent = f.bed_risk
            ts.exit_alerted = False  # settled back in bed -> ready for the next exit
            # The immobile ("high") tier tracks EVERYTHING: any sustained in-bed
            # movement -- turning, reaching, starting to rise -- alerts on its
            # own, before a sit-up. One alert per movement burst; resets when the
            # patient is still again.
            if (f.bed_risk or "unknown") == "high":
                if f.motion >= th.in_bed_move:
                    if not ts.bed_move_emitted and not self._cooling_down(ts, now):
                        ts.bed_move_emitted = True
                        ts.last_alert_t = now
                        return Event(
                            type="BED_EXIT",
                            track_id=f.track_id,
                            t_trigger=now,
                            t_alert=now,
                            zone=zone,
                            severity_override=3,
                            evidence={
                                "bed_risk": "high",
                                "trigger": "in_bed_movement",
                                "motion": round(f.motion, 3),
                            },
                        )
                else:
                    ts.bed_move_emitted = False  # still again -> ready for next move
            return None

        # A clearly seated body keeps a near-vertical torso even when its
        # floor_spread reads like a fallen one's -- so recognise it BEFORE the
        # down test, or sitting gets swallowed as ON_GROUND (the whole reason
        # sitting never showed). torso_tilt is an image angle, calibration-free,
        # so this holds even when the metric heights are off. The fall
        # thresholds are untouched.
        lo, hi = th.seated_h
        if (
            f.torso_tilt is not None
            and f.torso_tilt <= th.seated_tilt_max
            and lo <= f.h_torso <= hi
        ):
            ts.down_since = None
            return self._enter_sitting(ts, f, now, zone)

        # Down without ever triggering: slow slump, or joined late.
        if down:
            if ts.down_since is None:
                ts.down_since = now
            self._set_state(ts, "ON_GROUND", now)
            if (
                now - ts.down_since >= th.slow_down_s
                and f.motion <= th.confirm_motion_max
                and not self._cooling_down(ts, now)
            ):
                ts.last_alert_t = now
                return Event(
                    type="PERSON_DOWN",
                    track_id=f.track_id,
                    t_trigger=ts.down_since,
                    t_alert=now,
                    zone=zone,
                    evidence={
                        "down_s": round(now - ts.down_since, 1),
                        "h_torso": round(f.h_torso, 2),
                        "floor_spread": round(f.floor_spread, 2),
                        "motion": round(f.motion, 3),
                        "no_impact_detected": True,
                    },
                )
            return None

        ts.down_since = None

        # Sitting fallback: the height band alone, for frames where torso_tilt
        # is unavailable (e.g. shoulders or hips not confident).
        lo, hi = th.sitting_h
        if lo <= f.h_torso <= hi:
            return self._enter_sitting(ts, f, now, zone)

        ts.sitting_since = None
        ts.bed_exit_emitted = False

        if f.h_torso >= th.upright_h:
            self._set_state(ts, "UPRIGHT", now)
            # Standing up after being on/at a bed IS the exit. Catch it here when
            # the sit-up precursor never fired -- a fast riser, or an exit-seeking
            # ("confused") patient whose sit-up is allowed. Fires once per exit.
            risk = ts.bed_risk_recent or "unknown"
            if (
                ts.bed_assoc_recent
                and not ts.exit_alerted
                and risk not in SELF_EXIT_ALLOWED  # cleared to self-exit -> no alert
                and not self._cooling_down(ts, now)
            ):
                ts.exit_alerted = True
                ts.last_alert_t = now
                return Event(
                    type="BED_EXIT",
                    track_id=f.track_id,
                    t_trigger=now,
                    t_alert=now,
                    zone=zone,
                    severity_override=BED_EXIT_SEVERITY_BY_RISK.get(risk, 2),
                    evidence={
                        "bed_risk": risk,
                        "trigger": "stood_up",
                        "h_torso": round(f.h_torso, 2),
                    },
                )
        else:
            self._set_state(ts, "UNKNOWN", now)
        return None

    def _enter_sitting(
        self, ts: _TrackState, f: Features, now: float, zone: str | None
    ) -> Event | None:
        """Mark the track SITTING, emitting a graded BED_EXIT if it is a
        prolonged sit on a bed edge -- the highest-value precursor, since
        preventing a fall beats detecting one.
        """
        assert f.h_torso is not None
        self._set_state(ts, "SITTING", now)
        if ts.sitting_since is None:
            ts.sitting_since = now
        near_bed = any("bed" in z.lower() for z in f.zones)
        risk = f.bed_risk or "unknown"
        if near_bed:
            ts.bed_assoc_recent = True
            ts.bed_risk_recent = f.bed_risk
        # The dwell before alerting is set by the patient's severity tier: an
        # immobile patient alerts on the first sit-up, an independent one only
        # after a longer sit. "confused"/"wander" use an infinite dwell -- they
        # are allowed to sit up, so the sit-up never alarms (the stand-up does).
        # Falls back to the global threshold for any tier not in the map.
        dwell = BED_EXIT_DWELL_BY_RISK.get(risk, self.th.bed_exit_s)
        if (
            near_bed
            and not ts.bed_exit_emitted
            and not ts.exit_alerted
            and now - ts.sitting_since >= dwell
        ):
            ts.bed_exit_emitted = True
            ts.exit_alerted = True
            return Event(
                type="BED_EXIT",
                track_id=f.track_id,
                t_trigger=ts.sitting_since,
                t_alert=now,
                zone=zone,
                severity_override=BED_EXIT_SEVERITY_BY_RISK.get(risk, 2),
                evidence={
                    "bed_risk": risk,
                    "dwell_s": dwell,
                    "trigger": "sit_up",
                    "h_torso": round(f.h_torso, 2),
                    "seated_s": round(now - ts.sitting_since, 1),
                },
            )
        return None

    @staticmethod
    def _reset_fall(ts: _TrackState) -> None:
        ts.trigger_t = None
        ts.trigger_h = None
        ts.peak_vz = 0.0
        ts.down_since = None
        ts.suspected = False
        ts.vz_streak = 0

    def retain_only(self, live_ids: set[int]) -> None:
        for tid in list(self._tracks):
            if tid not in live_ids:
                del self._tracks[tid]
