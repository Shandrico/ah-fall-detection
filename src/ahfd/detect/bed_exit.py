"""Hierarchical, explainable bed-activity state machine.

Activity phase, physical bed support and observation quality are deliberately
separate. Losing pose confidence therefore does not rewrite a patient's last
known activity as "normal", and a patient may progress toward an exit while
still physically supported by the mattress or rail.

The defaults run early warning in shadow mode: the candidate and its evidence
are visible in :class:`BedExitSnapshot`, but no early-warning event is emitted
until a deployment explicitly opts in after nurse review. A completed
``OUT_OF_BED`` transition may still emit the existing ``BED_EXIT`` event.

The older :class:`BedExitMonitor` precursor is retained below for compatibility
with Shandrico's original detector and offline comparisons. New runtime paths
should use :class:`BedExitStateMachine`, whose warning output requires temporal
and bed-edge corroboration instead of emitting on raw CUSUM onset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from ahfd.detect.cusum import CusumConfig, CusumOnset, CusumSample
from ahfd.detect.events import Event
from ahfd.detect.state_machine import BED_EXIT_SEVERITY_BY_RISK, SELF_EXIT_ALLOWED
from ahfd.features.extractor import Features

BedActivityPhase = Literal[
    "UNKNOWN",
    "RECLINED",
    "TORSO_RISING",
    "UPRIGHT_IN_BED",
    "SHIFTING_TO_EDGE",
    "EDGE_SITTING",
    "ATTEMPTING_STAND",
    "OUT_OF_BED",
]
BedSupport = Literal["UNKNOWN", "SUPPORTED", "PARTIAL", "UNSUPPORTED"]
ObservationStatus = Literal["VALID", "LOW_CONFIDENCE", "MONITORING_UNAVAILABLE"]


_RISK_SEVERITY = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "unknown": 2,
}


@dataclass(frozen=True)
class BedExitThresholds:
    """Engineering starting points, not validated clinical thresholds."""

    min_valid_kp: int = 8
    min_mean_conf: float = 0.40

    reclined_tilt_enter_deg: float = 58.0
    reclined_tilt_exit_deg: float = 48.0
    upright_tilt_enter_deg: float = 38.0
    upright_tilt_exit_deg: float = 48.0
    stable_rest_motion_max: float = 0.08

    support_enter: float = 0.55
    support_exit: float = 0.30
    support_loss_rate: float = 0.20
    near_edge_enter_m: float = 0.22
    near_edge_exit_m: float = 0.34
    outside_edge_m: float = 0.05
    edge_velocity_enter_mps: float = -0.05
    edge_velocity_exit_mps: float = -0.015

    rise_delta_m: float = 0.12
    phase_dwell_s: float = 0.30
    fast_transition_dwell_s: float = 0.10
    return_recline_dwell_s: float = 0.75
    out_of_bed_dwell_s: float = 0.20
    early_warning_dwell_s: float = 0.60

    emit_early_warning: bool = False
    emit_exit_event: bool = True
    cusum: CusumConfig = field(default_factory=CusumConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.support_exit < self.support_enter <= 1.0:
            raise ValueError("support thresholds need 0 <= exit < enter <= 1")
        if self.near_edge_exit_m <= self.near_edge_enter_m:
            raise ValueError("near_edge_exit_m must exceed near_edge_enter_m")
        for name in (
            "phase_dwell_s",
            "fast_transition_dwell_s",
            "return_recline_dwell_s",
            "out_of_bed_dwell_s",
            "early_warning_dwell_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(name + " cannot be negative")


@dataclass(frozen=True)
class BedExitSnapshot:
    """Current JSON-friendly state and the evidence behind it."""

    track_id: int
    t: float
    phase: BedActivityPhase
    phase_since: float
    support: BedSupport
    observation: ObservationStatus
    observation_since: float
    bed_id: str | None = None
    bed_risk: str | None = None
    shoulder_elevation_m: float | None = None
    support_fraction: float | None = None
    edge_distance_m: float | None = None
    edge_velocity_mps: float | None = None
    torso_tilt_deg: float | None = None
    cusum_armed: bool = False
    cusum_z: float = 0.0
    cusum_g: float = 0.0
    baseline_mean: float | None = None
    baseline_std: float | None = None
    onset_t: float | None = None
    early_warning_candidate: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def phase_elapsed_s(self) -> float:
        return max(0.0, self.t - self.phase_since)

    @property
    def observation_elapsed_s(self) -> float:
        return max(0.0, self.t - self.observation_since)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["phase_elapsed_s"] = round(self.phase_elapsed_s, 3)
        result["observation_elapsed_s"] = round(self.observation_elapsed_s, 3)
        result["reasons"] = list(self.reasons)
        return result


@dataclass
class _BedTrack:
    track_id: int
    phase: BedActivityPhase = "UNKNOWN"
    phase_since: float = 0.0
    support: BedSupport = "UNKNOWN"
    observation: ObservationStatus = "MONITORING_UNAVAILABLE"
    observation_since: float = 0.0
    last_update_t: float = 0.0
    bed_id: str | None = None
    bed_risk: str | None = None
    candidate_phase: BedActivityPhase | None = None
    candidate_since: float | None = None
    last_valid_t: float | None = None
    last_edge_distance: float | None = None
    last_edge_t: float | None = None
    edge_velocity: float | None = None
    last_support_fraction: float | None = None
    last_support_t: float | None = None
    support_velocity: float | None = None
    onset_t: float | None = None
    warning_candidate_since: float | None = None
    warning_emitted: bool = False
    exit_emitted: bool = False
    support_seen: bool = False
    support_lost_since: float | None = None
    reasons: tuple[str, ...] = ()
    last_signal: float | None = None
    signal_source: str | None = None
    last_tilt: float | None = None
    last_support_fraction_seen: float | None = None
    last_cusum: CusumSample | None = None
    cusum: CusumOnset | None = None


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class BedExitStateMachine:
    """Per-track bed activity, independent from the fall state machine."""

    def __init__(self, thresholds: BedExitThresholds | None = None) -> None:
        self.th = thresholds or BedExitThresholds()
        self._tracks: dict[int, _BedTrack] = {}

    def _track(self, track_id: int, now: float) -> _BedTrack:
        ts = self._tracks.get(track_id)
        if ts is None:
            ts = _BedTrack(
                track_id=track_id,
                phase_since=now,
                observation_since=now,
                last_update_t=now,
                cusum=CusumOnset(self.th.cusum),
            )
            self._tracks[track_id] = ts
        return ts

    @staticmethod
    def _set_observation(
        ts: _BedTrack, status: ObservationStatus, now: float
    ) -> None:
        if ts.observation != status:
            ts.observation = status
            ts.observation_since = now

    def phase_of(self, track_id: int) -> BedActivityPhase:
        ts = self._tracks.get(track_id)
        return ts.phase if ts is not None else "UNKNOWN"

    def snapshot_of(self, track_id: int, now: float | None = None) -> BedExitSnapshot:
        ts = self._tracks.get(track_id)
        if ts is None:
            t = 0.0 if now is None else now
            return BedExitSnapshot(
                track_id=track_id,
                t=t,
                phase="UNKNOWN",
                phase_since=t,
                support="UNKNOWN",
                observation="MONITORING_UNAVAILABLE",
                observation_since=t,
            )
        t = ts.last_update_t if now is None else now
        c = ts.last_cusum
        return BedExitSnapshot(
            track_id=track_id,
            t=t,
            phase=ts.phase,
            phase_since=ts.phase_since,
            support=ts.support,
            observation=ts.observation,
            observation_since=ts.observation_since,
            bed_id=ts.bed_id,
            bed_risk=ts.bed_risk,
            shoulder_elevation_m=ts.last_signal,
            support_fraction=ts.last_support_fraction_seen,
            edge_distance_m=ts.last_edge_distance,
            edge_velocity_mps=ts.edge_velocity,
            torso_tilt_deg=ts.last_tilt,
            cusum_armed=bool(c and c.armed),
            cusum_z=(c.z if c else 0.0),
            cusum_g=(c.g if c else 0.0),
            baseline_mean=(c.baseline_mean if c and math.isfinite(c.baseline_mean) else None),
            baseline_std=(c.baseline_std if c and math.isfinite(c.baseline_std) else None),
            onset_t=ts.onset_t,
            early_warning_candidate=(
                ts.warning_candidate_since is not None
                and t - ts.warning_candidate_since >= self.th.early_warning_dwell_s
            ),
            reasons=ts.reasons,
        )

    def reset_alert_evidence(self, track_id: int) -> None:
        """Forget warning/dwell evidence while preserving physical bed state.

        A runtime policy toggle must not reuse CUSUM evidence accumulated under
        another mode, but it also must not erase the fact that this track was
        supported by the bed.  The latter is required to recognise a completed
        exit if staff change the mode while the patient is already at the edge.
        """

        ts = self._tracks.get(track_id)
        if ts is None:
            return
        assert ts.cusum is not None
        ts.cusum.reset()
        ts.last_cusum = None
        ts.onset_t = None
        ts.warning_candidate_since = None
        ts.warning_emitted = False
        ts.candidate_phase = None
        ts.candidate_since = None
        ts.edge_velocity = None
        ts.support_velocity = None
        ts.last_edge_t = None
        ts.last_support_t = None

    def _quality_ok(self, f: Features) -> bool:
        try:
            geometry_ok = f.has_geometry()
        except AttributeError:
            geometry_ok = getattr(f, "contact_xy", None) is not None and getattr(
                f, "h_torso", None
            ) is not None
        return (
            int(getattr(f, "n_valid_kp", 0)) >= self.th.min_valid_kp
            and float(getattr(f, "mean_conf", 0.0)) >= self.th.min_mean_conf
            and geometry_ok
        )

    def _support(self, ts: _BedTrack, f: Features) -> BedSupport:
        if getattr(f, "supported_by_bed", None) is not None:
            return "SUPPORTED"
        raw_fraction = getattr(f, "bed_support_fraction", None)
        if raw_fraction is None:
            raw_fraction = getattr(f, "bed_overlap", None)
        fraction = _finite(raw_fraction)
        if fraction is None:
            return "UNKNOWN"
        if fraction >= self.th.support_enter:
            return "SUPPORTED"
        if fraction <= self.th.support_exit:
            return "UNSUPPORTED"
        return "PARTIAL"

    def _update_kinematics(
        self, ts: _BedTrack, now: float, edge: float | None, support: float | None
    ) -> None:
        if edge is not None:
            if ts.last_edge_distance is not None and ts.last_edge_t is not None:
                dt = now - ts.last_edge_t
                if 1e-6 < dt <= self.th.cusum.max_gap_s:
                    ts.edge_velocity = (edge - ts.last_edge_distance) / dt
            ts.last_edge_distance = edge
            ts.last_edge_t = now
        if support is not None:
            if ts.last_support_fraction is not None and ts.last_support_t is not None:
                dt = now - ts.last_support_t
                if 1e-6 < dt <= self.th.cusum.max_gap_s:
                    ts.support_velocity = (support - ts.last_support_fraction) / dt
            ts.last_support_fraction = support
            ts.last_support_t = now

    def _advance(
        self,
        ts: _BedTrack,
        desired: BedActivityPhase,
        now: float,
        *,
        fast: bool = False,
    ) -> tuple[bool, float | None]:
        """Apply phase dwell and return ``(changed, first_evidence_t)``.

        ``phase_since`` remains the time at which the phase was accepted.  The
        separate evidence timestamp preserves the beginning of the dwell so
        event latency includes hysteresis instead of incorrectly reading zero.
        """
        if desired == ts.phase:
            ts.candidate_phase = None
            ts.candidate_since = None
            return False, None
        if ts.candidate_phase != desired:
            ts.candidate_phase = desired
            ts.candidate_since = now
            return False, None

        assert ts.candidate_since is not None
        dwell = self.th.fast_transition_dwell_s if fast else self.th.phase_dwell_s
        if desired == "RECLINED" and ts.phase not in ("UNKNOWN", "RECLINED"):
            dwell = self.th.return_recline_dwell_s
        elif desired == "OUT_OF_BED":
            dwell = self.th.out_of_bed_dwell_s
        if now - ts.candidate_since < dwell:
            return False, None

        first_evidence_t = ts.candidate_since
        previous = ts.phase
        ts.phase = desired
        ts.phase_since = now
        ts.candidate_phase = None
        ts.candidate_since = None
        if desired == "RECLINED" and previous != "RECLINED":
            assert ts.cusum is not None
            # Re-arm only after the activity owner has confirmed a sustained
            # return to recline. CUSUM never decides this from a flat elevated
            # signal on its own.
            ts.cusum.reset()
            ts.last_cusum = None
            ts.onset_t = None
            ts.warning_candidate_since = None
            ts.warning_emitted = False
            ts.exit_emitted = False
        return True, first_evidence_t

    @staticmethod
    def _break_continuity(ts: _BedTrack) -> None:
        """Discard temporal evidence that cannot cross an observation gap.

        The last phase/support remain as an explicitly stale observation for the
        dashboard; onset, velocity and candidate dwell do not survive.
        """
        ts.candidate_phase = None
        ts.candidate_since = None
        ts.onset_t = None
        ts.warning_candidate_since = None
        ts.edge_velocity = None
        ts.support_velocity = None
        ts.last_edge_t = None
        ts.last_support_t = None

    def _evidence(self, ts: _BedTrack, now: float, trigger: str) -> dict:
        snap = self.snapshot_of(ts.track_id, now)
        evidence = {
            "trigger": trigger,
            "phase": snap.phase,
            "bed_id": snap.bed_id,
            "bed_risk": snap.bed_risk or "unknown",
            "support": snap.support,
            "observation": snap.observation,
            "reasons": list(snap.reasons),
            "cusum_g": round(snap.cusum_g, 3),
            "cusum_z": round(snap.cusum_z, 3),
            "height_source": ts.signal_source or "unknown",
        }
        for key, value in (
            ("shoulder_elevation_m", snap.shoulder_elevation_m),
            ("support_fraction", snap.support_fraction),
            ("edge_distance_m", snap.edge_distance_m),
            ("edge_velocity_mps", snap.edge_velocity_mps),
            ("baseline_mean", snap.baseline_mean),
            ("baseline_std", snap.baseline_std),
        ):
            if value is not None:
                evidence[key] = round(value, 3)
        if ts.onset_t is not None:
            evidence["onset_to_alert_s"] = round(now - ts.onset_t, 3)
        return evidence

    def update(self, f: Features) -> Event | None:
        """Advance one track and optionally emit a warning or completed exit."""
        now = float(f.t)
        ts = self._track(int(f.track_id), now)
        ts.last_update_t = now
        assert ts.cusum is not None

        unavailable = bool(getattr(f, "in_excluded_zone", False))
        if unavailable or not self._quality_ok(f):
            status: ObservationStatus = (
                "MONITORING_UNAVAILABLE" if unavailable else "LOW_CONFIDENCE"
            )
            self._set_observation(ts, status, now)
            if (
                ts.last_valid_t is not None
                and now - ts.last_valid_t > self.th.cusum.max_gap_s
            ):
                self._break_continuity(ts)
            else:
                ts.candidate_phase = None
                ts.candidate_since = None
                ts.warning_candidate_since = None
            ts.last_cusum = ts.cusum.update(
                None, now, baseline_eligible=False, valid=False
            )
            ts.reasons = ("excluded_zone" if unavailable else "low_confidence",)
            return None

        self._set_observation(ts, "VALID", now)
        if (
            ts.last_valid_t is not None
            and now - ts.last_valid_t > self.th.cusum.max_gap_s
        ):
            self._break_continuity(ts)
        ts.last_valid_t = now

        associated = getattr(f, "associated_bed", None) or getattr(
            f, "supported_by_bed", None
        )
        if associated is not None:
            ts.bed_id = str(associated)
        risk = getattr(f, "bed_risk", None)
        if risk is not None:
            ts.bed_risk = str(risk)

        raw_support = getattr(f, "bed_support_fraction", None)
        if raw_support is None:
            raw_support = getattr(f, "bed_overlap", None)
        support_fraction = _finite(raw_support)
        edge = _finite(getattr(f, "bed_edge_distance_m", None))
        signal = _finite(getattr(f, "h_shoulder", None))
        signal_source = getattr(f, "h_shoulder_source", None)
        if signal is None:
            signal = _finite(getattr(f, "h_torso", None))
            signal_source = getattr(f, "h_torso_source", None)
        if signal is not None and signal_source is None:
            # Backward-compatible synthetic/replay inputs have one stable,
            # unspecified estimator rather than alternating depth/monocular.
            signal_source = "unspecified"
        if ts.signal_source is not None and signal_source != ts.signal_source:
            # A depth hole may make the extractor fall back to monocular
            # geometry. Those estimators have different offsets, so their
            # discontinuity is not patient movement and must not enter CUSUM.
            self.reset_alert_evidence(ts.track_id)
        ts.signal_source = signal_source
        tilt = _finite(getattr(f, "torso_tilt", None))
        motion = _finite(getattr(f, "motion", None)) or 0.0

        previous_support = ts.support
        ts.support = self._support(ts, f)
        if ts.support in ("SUPPORTED", "PARTIAL"):
            ts.support_seen = True
        if ts.support == "UNSUPPORTED" and previous_support != "UNSUPPORTED":
            ts.support_lost_since = now
        elif ts.support == "SUPPORTED":
            ts.support_lost_since = None
        ts.last_signal = signal
        ts.last_tilt = tilt
        ts.last_support_fraction_seen = support_fraction
        self._update_kinematics(ts, now, edge, support_fraction)

        edge_velocity_limit = (
            self.th.edge_velocity_exit_mps
            if ts.phase == "SHIFTING_TO_EDGE"
            else self.th.edge_velocity_enter_mps
        )
        edge_directed = bool(
            ts.edge_velocity is not None
            and ts.edge_velocity <= edge_velocity_limit
        ) or bool(
            ts.support_velocity is not None
            and ts.support_velocity <= -self.th.support_loss_rate
        )
        reclined_tilt_limit = (
            self.th.reclined_tilt_exit_deg
            if ts.phase == "RECLINED"
            else self.th.reclined_tilt_enter_deg
        )
        reclined = bool(
            tilt is not None
            and tilt >= reclined_tilt_limit
            and ts.support == "SUPPORTED"
        )
        baseline_eligible = (
            reclined and motion <= self.th.stable_rest_motion_max and not edge_directed
        )
        ts.last_cusum = ts.cusum.update(
            signal,
            now,
            baseline_eligible=baseline_eligible,
            valid=signal is not None,
        )
        if ts.last_cusum.onset:
            ts.onset_t = now

        baseline_rise = bool(
            ts.last_cusum.armed
            and signal is not None
            and math.isfinite(ts.last_cusum.baseline_mean)
            and signal - ts.last_cusum.baseline_mean >= self.th.rise_delta_m
        )
        rise_active = ts.onset_t is not None or baseline_rise
        # Reclined-angle hysteresis must not hide a measured upper-body rise.
        # It only keeps a quiet dead-band posture eligible as rest; once rise
        # evidence is currently active, the phase machine is allowed to
        # advance.  Use current evidence here rather than the episode's latched
        # onset, otherwise that latch would make return-to-recline impossible.
        current_rise_evidence = baseline_rise or bool(
            ts.cusum.fired and ts.last_cusum.z > self.th.cusum.k
        )
        stable_reclined = baseline_eligible and not current_rise_evidence
        upright = bool(tilt is not None and tilt <= self.th.upright_tilt_enter_deg)
        if ts.phase in ("UPRIGHT_IN_BED", "EDGE_SITTING", "ATTEMPTING_STAND"):
            upright = bool(tilt is not None and tilt <= self.th.upright_tilt_exit_deg)

        near_edge = ts.support == "PARTIAL"
        if edge is not None:
            limit = (
                self.th.near_edge_exit_m
                if ts.phase in ("SHIFTING_TO_EDGE", "EDGE_SITTING")
                else self.th.near_edge_enter_m
            )
            near_edge = edge <= limit
        outside = edge is not None and edge < -self.th.outside_edge_m
        bed_known = ts.bed_id is not None

        reasons: list[str] = []
        if ts.last_cusum.onset:
            reasons.append("cusum_onset")
        elif baseline_rise:
            reasons.append("sustained_rise")
        if edge_directed:
            reasons.append("edge_directed_motion")
        if near_edge:
            reasons.append("near_edge")
        if ts.support in ("PARTIAL", "UNSUPPORTED"):
            reasons.append("support_" + ts.support.lower())

        fast = False
        if stable_reclined:
            desired: BedActivityPhase = "RECLINED"
        # A completed *bed exit* requires evidence that this track was actually
        # supported by the bed earlier in the episode.  Merely walking through
        # a calibrated bed footprint must not create a medium/high alert.
        elif bed_known and ts.support_seen and ts.support == "UNSUPPORTED" and (
            outside
            or edge_directed
            or ts.support_lost_since is not None
            or ts.phase in ("SHIFTING_TO_EDGE", "EDGE_SITTING", "ATTEMPTING_STAND")
        ):
            desired = "OUT_OF_BED"
            fast = outside or ts.phase == "RECLINED"
            reasons.append("bed_support_lost")
        elif bed_known and upright and ts.support in ("PARTIAL", "UNSUPPORTED") and (
            rise_active or near_edge or edge_directed
        ):
            desired = "ATTEMPTING_STAND"
            fast = ts.phase == "RECLINED"
        elif bed_known and upright and near_edge:
            desired = "EDGE_SITTING"
        elif bed_known and edge_directed:
            desired = "SHIFTING_TO_EDGE"
        elif bed_known and upright and ts.support in ("SUPPORTED", "PARTIAL"):
            desired = "UPRIGHT_IN_BED"
        elif bed_known and rise_active and ts.support in ("SUPPORTED", "PARTIAL"):
            desired = "TORSO_RISING"
        elif reclined:
            desired = "RECLINED"
        else:
            desired = "UNKNOWN" if ts.phase == "UNKNOWN" else ts.phase

        ts.reasons = tuple(reasons)
        changed, transition_evidence_t = self._advance(ts, desired, now, fast=fast)

        # An operational high-mode warning must never be raw CUSUM, and a
        # static person who merely happens to be near an edge is not enough.
        # Require the rest-trained CUSUM to have fired *and* current progression
        # evidence: edge/support movement or the committed stand-attempt phase.
        # This deliberately trades a little lead time for fewer nuisance pages.
        corroborated = bool(ts.cusum.fired) and rise_active and (
            edge_directed or ts.phase == "ATTEMPTING_STAND"
        )
        if corroborated and ts.phase != "OUT_OF_BED":
            if ts.warning_candidate_since is None:
                ts.warning_candidate_since = now
        else:
            ts.warning_candidate_since = None

        warning_ready = bool(
            ts.warning_candidate_since is not None
            and now - ts.warning_candidate_since >= self.th.early_warning_dwell_s
        )
        risk_name = ts.bed_risk or "unknown"
        severity = _RISK_SEVERITY.get(risk_name, 2)

        if (
            warning_ready
            and self.th.emit_early_warning
            and not ts.warning_emitted
        ):
            ts.warning_emitted = True
            return Event(
                type="BED_EXIT_WARNING",
                track_id=ts.track_id,
                t_trigger=(
                    ts.onset_t
                    if ts.onset_t is not None
                    else (
                        ts.warning_candidate_since
                        if ts.warning_candidate_since is not None
                        else now
                    )
                ),
                t_alert=now,
                zone=ts.bed_id,
                severity_override=severity,
                evidence=self._evidence(ts, now, "early_warning"),
            )

        if (
            changed
            and ts.phase == "OUT_OF_BED"
            and self.th.emit_exit_event
            and not ts.exit_emitted
        ):
            ts.exit_emitted = True
            return Event(
                type="BED_EXIT",
                track_id=ts.track_id,
                t_trigger=(
                    ts.onset_t
                    if ts.onset_t is not None
                    else (
                        transition_evidence_t
                        if transition_evidence_t is not None
                        else ts.phase_since
                    )
                ),
                t_alert=now,
                zone=ts.bed_id,
                severity_override=severity,
                evidence=self._evidence(ts, now, "out_of_bed"),
            )

        return None

    def mark_unobserved(self, track_id: int, t: float) -> None:
        """Mark a live-but-missed track without erasing its last activity phase."""
        ts = self._tracks.get(track_id)
        if ts is None:
            return
        ts.last_update_t = t
        assert ts.cusum is not None
        self._set_observation(ts, "MONITORING_UNAVAILABLE", t)
        if (
            ts.last_valid_t is not None
            and t - ts.last_valid_t > self.th.cusum.max_gap_s
        ):
            self._break_continuity(ts)
        else:
            ts.candidate_phase = None
            ts.candidate_since = None
            ts.warning_candidate_since = None
        ts.last_cusum = ts.cusum.update(
            None, t, baseline_eligible=False, valid=False
        )
        ts.reasons = ("track_not_observed",)

    def mark_frame_unobserved(
        self, t: float, live_ids: set[int], observed_ids: set[int]
    ) -> None:
        """Batch helper for a tracker-driven frame loop."""
        for track_id in live_ids - observed_ids:
            self.mark_unobserved(track_id, t)

    def reset(self, track_id: int | None = None) -> None:
        """Reset one reassociated track, or every track after a source seek."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def retain_only(self, live_ids: set[int]) -> None:
        for track_id in list(self._tracks):
            if track_id not in live_ids:
                del self._tracks[track_id]


@dataclass
class BedExitConfig:
    cusum: CusumConfig = field(default_factory=CusumConfig)
    recline_tilt_deg: float = 45.0
    """torso_tilt at or above this (image-space, calibration-free) counts as
    reclined/lying -- the baseline state the sit-up rises from."""
    cooldown_s: float = 5.0
    """Minimum gap between precursor alerts on one track, so a single drawn-out
    rise (sit-up then a stand attempt) does not page repeatedly."""
    overlap_context: float = 0.3
    """bed_overlap at or above this also counts as bed-context, not just a
    detected contact point. Bed-support (a contact point inside the footprint) is
    flickery at range -- on real clips it is present in only ~a third of frames --
    so a second, independent cue keeps the context from dropping out mid-sit-up."""
    support_grace_s: float = 3.0
    """Keep treating the patient as in-bed for this long after the last bed-context
    frame. Support/overlap flicker for a frame or two must not reset the detector
    (the earlier reset-on-unsupported wiped the accumulating rise mid-sit-up)."""


class BedExitMonitor:
    """Per-track bed-exit precursor detector. Feed it one ``Features`` per frame.

    Returns a ``BED_EXIT`` :class:`~ahfd.detect.events.Event` on the frame a
    sit-up onset is confirmed for a patient whose care plan does *not* clear them
    to self-exit; otherwise ``None``. Severity is graded by the bed's risk level,
    consistent with the fall machine's bed-exit tiers.
    """

    def __init__(self, config: BedExitConfig | None = None) -> None:
        self.cfg = config or BedExitConfig()
        self._cusum: dict[int, CusumOnset] = {}
        self._phase: dict[int, str] = {}
        self._last_alert: dict[int, float] = {}
        self._last_ctx_t: dict[int, float] = {}
        self._last_bed: dict[int, str | None] = {}

    def phase(self, track_id: int) -> str:
        """Current activity phase for a track: RECLINED / RISING / OUT_OF_BED."""
        return self._phase.get(track_id, "UNKNOWN")

    def _bed_context(self, f) -> bool:
        """Is the patient in/on a bed *right now*? Two independent cues, because
        the contact-point support is flickery at range."""
        if f.supported_by_bed is not None:
            return True
        return f.bed_overlap is not None and f.bed_overlap >= self.cfg.overlap_context

    def update(self, f) -> Event | None:
        tid = f.track_id
        cu = self._cusum.setdefault(tid, CusumOnset(self.cfg.cusum))

        if self._bed_context(f):
            self._last_ctx_t[tid] = f.t
            if f.supported_by_bed is not None:
                self._last_bed[tid] = f.supported_by_bed
        last_ctx = self._last_ctx_t.get(tid)
        in_bed_recent = last_ctx is not None and (f.t - last_ctx) <= self.cfg.support_grace_s

        # The shared CUSUM now requires callers to identify baseline-eligible
        # samples explicitly.  Teach this legacy comparator only from a
        # confidently reclined, recently bed-supported posture; an elevated
        # pause must not be absorbed as the new normal.  Bed context remains a
        # gate rather than a reset so brief support flicker cannot erase a rise.
        reclined = bool(
            in_bed_recent
            and f.torso_tilt is not None
            and f.torso_tilt >= self.cfg.recline_tilt_deg
        )
        s = cu.update(f.h_torso, f.t, baseline_eligible=reclined)

        if not in_bed_recent:
            # Well away from the bed: a confirmed stand/exit is the fall machine's job.
            self._phase[tid] = "OUT_OF_BED"
            return None

        if s.onset:
            self._phase[tid] = "RISING"
            risk = f.bed_risk or "unknown"
            # A patient cleared to mobilise (low/none) may sit up freely -- no alert.
            if risk in SELF_EXIT_ALLOWED:
                return None
            last = self._last_alert.get(tid)
            if last is not None and f.t - last < self.cfg.cooldown_s:
                return None
            self._last_alert[tid] = f.t
            return Event(
                type="BED_EXIT",
                track_id=tid,
                t_trigger=f.t,
                t_alert=f.t,
                zone=f.supported_by_bed or self._last_bed.get(tid),
                severity_override=BED_EXIT_SEVERITY_BY_RISK.get(risk, 2),
                evidence={
                    "trigger": "sit_up_onset",
                    "phase": "RISING",
                    "bed_risk": risk,
                    "cusum_g": round(s.g, 1),
                    "h_torso": round(f.h_torso, 2) if f.h_torso is not None else None,
                },
            )

        # Not rising: mark the reclined baseline state when the torso is flat.
        if reclined:
            self._phase[tid] = "RECLINED"
        return None
