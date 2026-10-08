"""The bed-exit decision.

A bed exit, like a fall, is a sequence rather than a frame. The difference is
what the sequence is *about*: not a body dropping to the floor, but a body core
leaving the footprint of the bed it is bound to. This machine runs in parallel
with `FallStateMachine` and never touches it -- a quiet bed-exit notice must
never get in the way of an urgent fall alert, and the two share no state.

The guarantee that makes this worth deploying
----------------------------------------------
**Limbs never trigger an exit.** The decision is made on the body core -- the
mid-hip and mid-shoulder -- so an arm over the rail, or a leg dangling off the
side, cannot raise a bed-exit alert. A leg crossing is reported as a separate,
low-priority `BED_EXIT_LIMB` signal (an early heads-up), and only the *core*
crossing and staying out raises `BED_EXIT_RISK` / `BED_EXIT_CONFIRMED`. This is
the whole point: the thing that makes these systems unusable is alarming on a
hand reaching for a water cup, and the core/limb split is how that is avoided.

Grading, not a binary alarm
---------------------------
The same physical exit is a silent dashboard status for a patient cleared to
mobilise and a nurse page for a high-risk one. Urgency comes from the bed's
`risk_level` via `BED_EXIT_SEVERITY_BY_RISK`, and a confirmed exit sits one notch
above the earlier risk warning for the same bed.

Unknown is never treated as safe
---------------------------------
If the joints needed to judge the bed are not observable -- blankets, curtains,
night lighting, the pose model losing the body behind a rail -- the machine goes
`DEGRADED` and says so, rather than silently reporting "still in bed".

Like the fall machine, `update` takes its time from the pose timestamp and holds
no wall clock, so the whole thing is testable from synthetic sequences with no
camera or pose model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from ahfd.detect.events import Event
from ahfd.detect.state_machine import BED_EXIT_SEVERITY_BY_RISK
from ahfd.features.bed_frame import BedFrameExtractor, BedObservation
from ahfd.types import PersonPose

BedExitState = Literal[
    "NOT_IN_BED",     # no bed episode bound to this track
    "IN_BED_STABLE",  # bound to a bed, no approach evidence
    "EDGE_APPROACH",  # core moving toward the locked edge
    "LEGS_OVER",      # lower limbs beyond the rail line (core still inside)
    "CORE_CROSSING",  # core at or past the rail line
    "EXITED",         # core beyond the line, sustained -- the real exit
    "DEGRADED",       # not enough observable evidence to judge
]

_SIDES = ("left", "right", "head", "foot")


@dataclass(frozen=True)
class BedExitThresholds:
    """Every tunable for the bed-exit branch, in metres, seconds, m/s.

    Untuned starting points, in the same spirit as `FallThresholds`: a baseline
    for `ahfd sweep` to tune against staged clips, not settled clinical numbers.
    """

    enabled: bool = True

    # evidence windows -- "sustained" means this fraction of observable frames
    # in the window, not N in a row, so one bad frame does not reset progress.
    evidence_window_s: float = 1.0
    evidence_fraction: float = 0.60
    baseline_window_s: float = 30.0

    # approach
    edge_near_m: float = 0.20
    approach_delta_m: float = 0.12
    approach_s: float = 1.5

    # legs (the low-priority limb signal)
    legs_quorum: int = 2
    legs_over_s: float = 0.8

    # crossing and exit
    crossing_margin_m: float = 0.0
    exit_margin_m: float = 0.10
    exit_confirm_s: float = 1.0

    # abort (hysteresis: return margin is looser than the exit margin it cancels)
    return_margin_m: float = 0.08
    abort_s: float = 3.0

    # profile
    rapid_core_speed: float = 0.50  # m/s of d_core decrease toward the edge

    # observation quality (also enforced in BedFrameExtractor)
    min_joint_conf: float = 0.30
    min_core_valid: int = 1
    min_total_valid: int = 4
    degraded_after_s: float = 2.0
    recover_s: float = 1.0

    # binding
    bind_s: float = 5.0
    unbind_s: float = 5.0

    # hygiene
    risk_cooldown_s: float = 30.0


def _escalate(sev: int) -> int:
    """Confirmed exit is one notch above the risk warning, but a bed cleared to
    self-mobilise (severity 0) still never alarms."""
    return min(3, sev + 1) if sev >= 1 else 0


@dataclass
class _TrackState:
    state: BedExitState = "NOT_IN_BED"
    state_since: float = 0.0
    first_seen: float | None = None

    bound_bed: str | None = None
    on_bed_since: float | None = None   # for binding
    off_bed_since: float | None = None  # for unbinding after exit

    locked_edge: str | None = None
    baseline_d: float | None = None     # rolling-median d to nearest edge in bed

    # rolling evidence: (t, observable, d_locked, legs_over_locked, d_nearest)
    window: deque = field(default_factory=lambda: deque())
    baseline_samples: deque = field(default_factory=lambda: deque())

    degraded_since: float | None = None
    observable_since: float | None = None
    state_before_degraded: BedExitState | None = None

    # per-episode emission flags and per-event cooldown
    risk_emitted: bool = False
    limb_emitted: bool = False
    confirmed_emitted: bool = False
    degraded_emitted: bool = False
    passed_approach: bool = False
    passed_legs: bool = False
    last_emit_t: dict[str, float] = field(default_factory=dict)

    def age(self, now: float) -> float:
        return 0.0 if self.first_seen is None else now - self.first_seen


class BedExitStateMachine:
    """Per-track bed-exit decision, parallel to the fall machine."""

    def __init__(
        self,
        thresholds: BedExitThresholds | None = None,
        extractor: BedFrameExtractor | None = None,
    ):
        self.th = thresholds or BedExitThresholds()
        self.extractor = extractor
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------ helpers

    def state_of(self, track_id: int) -> BedExitState:
        ts = self._tracks.get(track_id)
        return ts.state if ts else "NOT_IN_BED"

    def bound_bed_of(self, track_id: int) -> str | None:
        ts = self._tracks.get(track_id)
        return ts.bound_bed if ts else None

    def _set_state(self, ts: _TrackState, state: BedExitState, now: float) -> None:
        if ts.state != state:
            ts.state = state
            ts.state_since = now

    def _trim(self, dq: deque, now: float, window_s: float) -> None:
        while dq and now - dq[0][0] > window_s:
            dq.popleft()

    def _sustained(
        self, ts: _TrackState, now: float, predicate, *, window_s=None, fraction=None
    ) -> bool:
        """Did `predicate(sample)` hold over enough of the recent window?

        Samples are (t, observable, d_locked, legs_over, d_nearest). Only
        observable frames count toward both numerator and denominator, so a
        stretch of bad observation neither advances nor fakes progress.
        """
        window_s = self.th.evidence_window_s if window_s is None else window_s
        fraction = self.th.evidence_fraction if fraction is None else fraction
        obs = [s for s in ts.window if now - s[0] <= window_s and s[1]]
        if len(obs) < 2:
            return False
        hits = sum(1 for s in obs if predicate(s))
        return hits / len(obs) >= fraction

    def _cooling(self, ts: _TrackState, key: str, now: float) -> bool:
        last = ts.last_emit_t.get(key)
        return last is not None and now - last < self.th.risk_cooldown_s

    # ------------------------------------------------------------- update

    def update(self, person: PersonPose, t: float) -> Event | None:
        """Advance one track by one frame from its raw pose. Returns an event."""
        if self.extractor is None or not self.extractor.has_beds():
            return None
        if person.track_id is None:
            return None

        ts = self._tracks.setdefault(person.track_id, _TrackState())
        if ts.first_seen is None:
            ts.first_seen = t

        # Which bed to measure against: the bound one if any, else the bed the
        # hips are currently in (for binding). Resolving the bound bed by name
        # is what lets the machine keep watching after support is lost -- which
        # is the moment the exit actually happens.
        candidate = self.extractor.candidate_bed(person)
        bed_name = ts.bound_bed or (candidate.name if candidate else None)
        zone = self.extractor.zones.by_name(bed_name) if bed_name else None
        if zone is None:
            # Not near any bed and not bound -- nothing to decide.
            self._set_state(ts, "NOT_IN_BED", t)
            return None

        obs = self.extractor.observe(person, t, zone)
        return self._advance(ts, obs, t)

    def update_from_observation(
        self, track_id: int, obs: BedObservation, t: float
    ) -> Event | None:
        """Test seam: drive the machine directly from a `BedObservation`."""
        ts = self._tracks.setdefault(track_id, _TrackState())
        if ts.first_seen is None:
            ts.first_seen = t
        return self._advance(ts, obs, t)

    # ------------------------------------------------------- the decision

    def _advance(self, ts: _TrackState, obs: BedObservation, now: float) -> Event | None:
        th = self.th

        # --- observation quality / DEGRADED --------------------------------
        if not obs.observable:
            if ts.observable_since is not None:
                ts.observable_since = None
            if ts.degraded_since is None:
                ts.degraded_since = now
            ts.window.append((now, False, None, None, None))
            self._trim(ts.window, now, th.baseline_window_s)
            if (
                now - ts.degraded_since >= th.degraded_after_s
                and ts.state not in ("NOT_IN_BED", "DEGRADED")
            ):
                ts.state_before_degraded = ts.state
                self._set_state(ts, "DEGRADED", now)
                if not ts.degraded_emitted:
                    ts.degraded_emitted = True
                    return self._emit(ts, obs, now, "BED_MONITORING_DEGRADED", 1)
            return None

        # observable this frame
        ts.degraded_since = None
        if ts.observable_since is None:
            ts.observable_since = now

        # Distances: nearest edge overall, and the locked edge if we have one.
        d_by_edge = {s: obs.edges[s].d_core for s in _SIDES if obs.edges[s].d_core is not None}
        d_nearest = min(d_by_edge.values()) if d_by_edge else None
        nearest_edge = min(d_by_edge, key=d_by_edge.get) if d_by_edge else None
        if ts.locked_edge and ts.locked_edge in d_by_edge:
            d_locked = d_by_edge[ts.locked_edge]
            legs_over = obs.edges[ts.locked_edge].legs_over
        elif nearest_edge is not None:
            d_locked = d_by_edge[nearest_edge]
            legs_over = obs.edges[nearest_edge].legs_over
        else:
            d_locked = None
            legs_over = 0

        ts.window.append((now, True, d_locked, legs_over, d_nearest))
        self._trim(ts.window, now, th.baseline_window_s)

        # --- recover from DEGRADED -----------------------------------------
        if ts.state == "DEGRADED":
            if now - ts.observable_since >= th.recover_s:
                # Do not inherit counters across the gap: re-evaluate fresh.
                prev = ts.state_before_degraded or "IN_BED_STABLE"
                self._set_state(ts, prev if prev != "DEGRADED" else "IN_BED_STABLE", now)
                ts.degraded_emitted = False
            else:
                return None

        # --- binding -------------------------------------------------------
        if obs.on_bed:
            ts.off_bed_since = None
            if ts.on_bed_since is None:
                ts.on_bed_since = now
            if ts.bound_bed is None and now - ts.on_bed_since >= th.bind_s:
                ts.bound_bed = obs.bed_name
                self._set_state(ts, "IN_BED_STABLE", now)
        else:
            ts.on_bed_since = None
            if ts.off_bed_since is None:
                ts.off_bed_since = now

        if ts.bound_bed is None:
            self._set_state(ts, "NOT_IN_BED", now)
            return None

        # --- baseline (only while genuinely stable) ------------------------
        if ts.state == "IN_BED_STABLE" and d_nearest is not None:
            ts.baseline_samples.append((now, d_nearest))
            self._trim(ts.baseline_samples, now, th.baseline_window_s)
            vals = sorted(s[1] for s in ts.baseline_samples)
            ts.baseline_d = vals[len(vals) // 2]  # median

        # --- state transitions ---------------------------------------------
        return self._transitions(ts, obs, now, d_locked, legs_over, d_nearest, nearest_edge)

    def _transitions(
        self, ts, obs, now, d_locked, legs_over, d_nearest, nearest_edge
    ) -> Event | None:
        th = self.th

        # EXITED: core beyond the line, sustained -> the confirmed exit.
        if ts.state in ("CORE_CROSSING", "LEGS_OVER", "EDGE_APPROACH", "IN_BED_STABLE"):
            if self._sustained(
                ts, now,
                lambda s: s[2] is not None and s[2] <= -th.exit_margin_m,
                window_s=th.exit_confirm_s,
            ):
                self._lock_if_needed(ts, nearest_edge)
                self._set_state(ts, "EXITED", now)
                if not ts.confirmed_emitted:
                    ts.confirmed_emitted = True
                    base = BED_EXIT_SEVERITY_BY_RISK.get(obs.bed_risk or "unknown", 2)
                    return self._emit(
                        ts, obs, now, "BED_EXIT_CONFIRMED", _escalate(base),
                        profile=self._profile(ts, now),
                    )

        # CORE_CROSSING: core at or past the rail line.
        if ts.state in ("IN_BED_STABLE", "EDGE_APPROACH", "LEGS_OVER"):
            if self._sustained(
                ts, now, lambda s: s[2] is not None and s[2] <= th.crossing_margin_m
            ):
                self._lock_if_needed(ts, nearest_edge)
                self._set_state(ts, "CORE_CROSSING", now)
                if not ts.risk_emitted and not self._cooling(ts, "BED_EXIT_RISK", now):
                    ts.risk_emitted = True
                    base = BED_EXIT_SEVERITY_BY_RISK.get(obs.bed_risk or "unknown", 2)
                    return self._emit(
                        ts, obs, now, "BED_EXIT_RISK", base,
                        profile=self._profile(ts, now),
                    )

        # LEGS_OVER: legs beyond the line -- the low-priority limb signal. The
        # core is still inside, so this is explicitly NOT a bed exit.
        if ts.state in ("IN_BED_STABLE", "EDGE_APPROACH"):
            if self._sustained(
                ts, now, lambda s: s[3] >= th.legs_quorum, window_s=th.legs_over_s
            ):
                ts.passed_legs = True
                self._set_state(ts, "LEGS_OVER", now)
                if not ts.limb_emitted and not self._cooling(ts, "BED_EXIT_LIMB", now):
                    ts.limb_emitted = True
                    base = BED_EXIT_SEVERITY_BY_RISK.get(obs.bed_risk or "unknown", 2)
                    return self._emit(
                        ts, obs, now, "BED_EXIT_LIMB", min(1, base)
                    )

        # EDGE_APPROACH: core nearing the edge, measured against the baseline so
        # a patient who simply sleeps near a rail does not count as approaching.
        if ts.state == "IN_BED_STABLE":
            base_d = ts.baseline_d
            if base_d is not None and self._sustained(
                ts, now,
                lambda s: (
                    s[4] is not None
                    and s[4] <= th.edge_near_m
                    and (base_d - s[4]) >= th.approach_delta_m
                ),
                window_s=th.approach_s,
            ):
                ts.passed_approach = True
                self._lock_if_needed(ts, nearest_edge)
                self._set_state(ts, "EDGE_APPROACH", now)
                if not ts.risk_emitted and not self._cooling(ts, "BED_EXIT_RISK", now):
                    ts.risk_emitted = True
                    base = BED_EXIT_SEVERITY_BY_RISK.get(obs.bed_risk or "unknown", 2)
                    return self._emit(
                        ts, obs, now, "BED_EXIT_RISK", base,
                        profile=self._profile(ts, now),
                    )

        # ABORT: came back inside and legs back in, sustained -> reset.
        if ts.state in ("EDGE_APPROACH", "LEGS_OVER", "CORE_CROSSING"):
            if self._sustained(
                ts, now,
                lambda s: s[2] is not None and s[2] >= th.return_margin_m and s[3] == 0,
                window_s=th.abort_s,
            ):
                event = self._emit(ts, obs, now, "BED_EXIT_ABORTED", 0)
                self._reset_episode(ts, now)
                self._set_state(ts, "IN_BED_STABLE", now)
                return event

        # Unbind only well after a confirmed exit, so we never unbind mid-exit.
        if (
            ts.state == "EXITED"
            and ts.off_bed_since is not None
            and now - ts.off_bed_since >= th.unbind_s
        ):
            self._reset_episode(ts, now)
            ts.bound_bed = None
            self._set_state(ts, "NOT_IN_BED", now)

        return None

    # ------------------------------------------------------------ plumbing

    def _lock_if_needed(self, ts: _TrackState, nearest_edge: str | None) -> None:
        if ts.locked_edge is None and nearest_edge is not None:
            ts.locked_edge = nearest_edge

    def _profile(self, ts: _TrackState, now: float) -> str:
        """Descriptive label for nurse review -- NOT a claim about intent."""
        # Core speed toward the edge over the window.
        obs = [s for s in ts.window if s[1] and s[2] is not None]
        if len(obs) >= 2 and (obs[-1][0] - obs[0][0]) > 1e-6:
            speed = (obs[0][2] - obs[-1][2]) / (obs[-1][0] - obs[0][0])
        else:
            speed = 0.0
        if (ts.passed_approach and ts.passed_legs) and speed < self.th.rapid_core_speed:
            return "progressive"
        if speed >= self.th.rapid_core_speed or not (ts.passed_approach or ts.passed_legs):
            return "rapid"
        return "progressive"

    def _reset_episode(self, ts: _TrackState, now: float) -> None:
        ts.locked_edge = None
        ts.risk_emitted = False
        ts.limb_emitted = False
        ts.confirmed_emitted = False
        ts.passed_approach = False
        ts.passed_legs = False

    def _emit(
        self,
        ts: _TrackState,
        obs: BedObservation,
        now: float,
        etype: str,
        severity: int,
        profile: str | None = None,
    ) -> Event:
        ts.last_emit_t[etype] = now
        edge_ev = obs.edges.get(ts.locked_edge) if ts.locked_edge else None
        observable_frac = self._observable_fraction(ts, now)
        evidence = {
            "bed": obs.bed_name,
            "bed_risk": obs.bed_risk,
            "edge": ts.locked_edge,
            "edge_has_rail": edge_ev.has_rail if edge_ev else None,
            "d_core_m": round(edge_ev.d_core, 3) if edge_ev and edge_ev.d_core is not None else None,
            "legs_over": edge_ev.legs_over if edge_ev else None,
            "n_core_valid": obs.n_core_valid,
            "n_leg_valid": obs.n_leg_valid,
            "baseline_d_core_m": round(ts.baseline_d, 3) if ts.baseline_d is not None else None,
            "observable_fraction": round(observable_frac, 2),
        }
        if profile is not None:
            evidence["profile"] = profile
        return Event(
            type=etype,  # type: ignore[arg-type]
            track_id=obs.track_id,
            t_trigger=ts.state_since,
            t_alert=now,
            zone=obs.bed_name,
            severity_override=severity,
            evidence=evidence,
        )

    def _observable_fraction(self, ts: _TrackState, now: float) -> float:
        recent = [s for s in ts.window if now - s[0] <= self.th.evidence_window_s]
        if not recent:
            return 0.0
        return sum(1 for s in recent if s[1]) / len(recent)

    def retain_only(self, live_ids: set[int]) -> None:
        for tid in list(self._tracks):
            if tid not in live_ids:
                del self._tracks[tid]
