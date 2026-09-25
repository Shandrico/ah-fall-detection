"""Bed-exit *precursor* monitor -- catches the sit-up while still bed-supported.

This runs in PARALLEL with ``FallStateMachine`` (the design review's
recommendation) and deliberately does not touch it. It exists to close a
verified gap: ``FallStateMachine._steady_state`` sets ``IN_BED`` and returns
*before* any sitting/rising check, so for every tier except the immobile one a
patient who sits up while still on the bed produces no early signal -- the exit
is only caught once they have already stood or left the footprint. That is too
late for an *early* warning.

The monitor watches the torso-elevation signal with a causal CUSUM
(:class:`ahfd.detect.cusum.CusumOnset`) while the patient is bed-supported, and
emits a ``BED_EXIT`` precursor the moment a sustained rise begins. CUSUM is used
rather than a height threshold because the elevation signal is biased at the
ward mount (see the depth error analysis); the *change* from the patient's own
reclined baseline survives that bias.

Scope, on purpose: this is the RISK precursor (RECLINED -> RISING). The confirmed
departure (standing / leaving the footprint) stays with the fall machine's exit
trigger. It emits the onset only; corroborating it with edge-directed motion --
once a bed-edge-distance feature exists -- is the natural next step and belongs
with the in-bed-precursor owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ahfd.detect.cusum import CusumConfig, CusumOnset
from ahfd.detect.events import Event
from ahfd.detect.state_machine import BED_EXIT_SEVERITY_BY_RISK, SELF_EXIT_ALLOWED


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

        # Run the change detector continuously (support is a *gate*, not a reset --
        # it flickers, and resetting on it wipes the accumulating rise). CUSUM
        # self-heals its baseline on stable rest, so a stint off the bed is fine.
        s = cu.update(f.h_torso, f.t)

        if self._bed_context(f):
            self._last_ctx_t[tid] = f.t
            if f.supported_by_bed is not None:
                self._last_bed[tid] = f.supported_by_bed
        last_ctx = self._last_ctx_t.get(tid)
        in_bed_recent = last_ctx is not None and (f.t - last_ctx) <= self.cfg.support_grace_s

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
        if f.torso_tilt is not None and f.torso_tilt >= self.cfg.recline_tilt_deg:
            self._phase[tid] = "RECLINED"
        return None
