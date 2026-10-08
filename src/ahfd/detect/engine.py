"""Composition of independent fall and bed-activity decisions."""

from __future__ import annotations

from ahfd.detect.bed_exit import BedExitStateMachine, BedExitThresholds
from ahfd.detect.state_machine import FallStateMachine, FallThresholds
from ahfd.features.extractor import Features


class DetectionEngine:
    """One public detector API backed by two independent state machines.

    The fall posture remains suitable for the existing overlay while the bed
    activity phase, physical support and observation availability stay
    separately inspectable.  Bed logic is not folded back into fall state.
    """

    def __init__(
        self,
        fall_thresholds: FallThresholds | None = None,
        bed_thresholds: BedExitThresholds | None = None,
        *,
        bed_activity_enabled: bool = True,
    ) -> None:
        self.fall = FallStateMachine(
            fall_thresholds, enable_legacy_bed_alerts=not bed_activity_enabled
        )
        self.bed = BedExitStateMachine(bed_thresholds) if bed_activity_enabled else None

    def update_all(self, features: Features) -> tuple:
        """Return every same-frame event, ordered from urgent fall to bed activity."""
        bed_event = self.bed.update(features) if self.bed is not None else None
        fall_event = self.fall.update(features)
        return tuple(event for event in (fall_event, bed_event) if event is not None)

    def update(self, features: Features):
        """Legacy one-event API; prefer :meth:`update_all` for persistence."""
        events = self.update_all(features)
        return events[0] if events else None

    def state_of(self, track_id: int):
        return self.fall.state_of(track_id)

    def bed_phase_of(self, track_id: int):
        return self.bed.phase_of(track_id) if self.bed is not None else "UNKNOWN"

    def bed_snapshot_of(self, track_id: int, now: float | None = None):
        return self.bed.snapshot_of(track_id, now) if self.bed is not None else None

    def mark_frame_unobserved(
        self, t: float, live_ids: set[int], observed_ids: set[int]
    ) -> None:
        if self.bed is not None:
            self.bed.mark_frame_unobserved(t, live_ids, observed_ids)

    def retain_only(self, live_ids: set[int]) -> None:
        self.fall.retain_only(live_ids)
        if self.bed is not None:
            self.bed.retain_only(live_ids)

    def reset(self) -> None:
        self.fall.retain_only(set())
        if self.bed is not None:
            self.bed.reset()
