"""Explicit monitoring availability for onsite collection.

Unobservable time is neither a negative example nor evidence that a patient is
safe.  This monitor turns frame gaps, weak pose, absent depth and identity
reassociation into timestamped state transitions suitable for the collection
telemetry stream and later coverage-adjusted evaluation.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping


class HealthStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class HealthReason(str, Enum):
    STARTING = "STARTING"
    NO_FRAMES = "NO_FRAMES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DEPTH_MISSING = "DEPTH_MISSING"
    TRACK_REASSOCIATED = "TRACK_REASSOCIATED"
    TARGET_NOT_BOUND = "TARGET_NOT_BOUND"
    PRIVACY_SCOPE_VIOLATION = "PRIVACY_SCOPE_VIOLATION"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    DISK_ERROR = "DISK_ERROR"
    SOURCE_ENDED = "SOURCE_ENDED"
    PROTOCOL_COMPLETE = "PROTOCOL_COMPLETE"
    PROTOCOL_INCOMPLETE = "PROTOCOL_INCOMPLETE"
    REQUESTED_STOP = "REQUESTED_STOP"
    OPERATOR_STOP = "OPERATOR_STOP"
    SESSION_LIMIT = "SESSION_LIMIT"


@dataclass(frozen=True)
class HealthTransition:
    """One change in status or its explanatory reason set."""

    t_rel_s: float
    previous: HealthStatus
    current: HealthStatus
    reasons: tuple[HealthReason, ...]
    details: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        """JSON-ready record body; pass ``t_rel_s`` to SessionRecorder separately."""
        return {
            "kind": "monitoring_health_transition",
            "previous": self.previous.value,
            "current": self.current.value,
            "reasons": [reason.value for reason in self.reasons],
            "details": dict(self.details),
        }


class HealthMonitor:
    """Track whether current observations support a trustworthy decision.

    ``observe_frame`` covers frames that reached the pipeline. ``tick`` must be
    called by a lightweight watchdog even when capture is blocked; after
    ``stale_after_s`` it opens an explicit ``NO_FRAMES`` unavailable interval.
    A transition callback can write each record directly to session telemetry.
    """

    def __init__(
        self,
        *,
        stale_after_s: float = 2.0,
        on_transition: Callable[[HealthTransition], None] | None = None,
    ) -> None:
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("stale_after_s must be finite and > 0")
        self.stale_after_s = float(stale_after_s)
        self._on_transition = on_transition
        self._lock = threading.RLock()
        self._status = HealthStatus.UNAVAILABLE
        self._reasons = (HealthReason.STARTING,)
        self._details: dict[str, Any] = {}
        self._last_frame_t: float | None = None
        self._last_update_t = 0.0
        self._transitions: list[HealthTransition] = []

    @property
    def status(self) -> HealthStatus:
        with self._lock:
            return self._status

    @property
    def reasons(self) -> tuple[HealthReason, ...]:
        with self._lock:
            return self._reasons

    @property
    def last_frame_t(self) -> float | None:
        with self._lock:
            return self._last_frame_t

    @property
    def transitions(self) -> tuple[HealthTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status.value,
                "reasons": [reason.value for reason in self._reasons],
                "details": dict(self._details),
                "last_frame_t_rel_s": self._last_frame_t,
            }

    def _time(self, t_rel_s: float) -> float:
        t = float(t_rel_s)
        if not math.isfinite(t) or t < 0:
            raise ValueError("t_rel_s must be finite and non-negative")
        if t + 1e-9 < self._last_update_t:
            raise ValueError("health time moved backwards")
        self._last_update_t = max(self._last_update_t, t)
        return round(t, 6)

    @staticmethod
    def _clean_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
        if details is None:
            return {}
        clean = dict(details)
        # Health details are intentionally scalar: identifiers and arrays belong
        # in derived records, not operational status messages.
        for key, value in clean.items():
            if not isinstance(key, str):
                raise TypeError("health detail keys must be strings")
            if not (
                value is None
                or isinstance(value, (str, bool, int))
                or (isinstance(value, float) and math.isfinite(value))
            ):
                raise TypeError("health details must contain JSON scalar values")
        return clean

    def _set(
        self,
        status: HealthStatus,
        reasons: tuple[HealthReason, ...],
        t_rel_s: float,
        details: Mapping[str, Any] | None = None,
    ) -> HealthTransition | None:
        t = self._time(t_rel_s)
        ordered = tuple(sorted(set(reasons), key=lambda reason: reason.value))
        clean = self._clean_details(details)
        if status == HealthStatus.AVAILABLE and ordered:
            raise ValueError("AVAILABLE cannot carry degradation reasons")
        if status != HealthStatus.AVAILABLE and not ordered:
            raise ValueError(status.value + " requires at least one reason")
        if status == self._status and ordered == self._reasons:
            # Details are live diagnostic context (for example FPS and free
            # disk space), not part of the health state identity.  Keep the
            # latest values for snapshots without emitting a transition on
            # every frame.
            self._details = clean
            return None

        transition = HealthTransition(
            t_rel_s=t,
            previous=self._status,
            current=status,
            reasons=ordered,
            details=clean,
        )
        self._status = status
        self._reasons = ordered
        self._details = clean
        self._transitions.append(transition)
        if self._on_transition is not None:
            self._on_transition(transition)
        return transition

    def observe_frame(
        self,
        t_rel_s: float,
        *,
        pose_confident: bool = True,
        depth_valid: bool = True,
        reassociated: bool = False,
        calibration_valid: bool = True,
        details: Mapping[str, Any] | None = None,
    ) -> HealthTransition | None:
        """Report one received frame and the quality of its derived observation."""
        with self._lock:
            return self._observe_frame_unlocked(
                t_rel_s,
                pose_confident=pose_confident,
                depth_valid=depth_valid,
                reassociated=reassociated,
                calibration_valid=calibration_valid,
                details=details,
            )

    def note_frame(self, t_rel_s: float) -> None:
        """Refresh capture freshness without claiming a usable observation.

        Target binding and calibration are separate availability gates. A real
        frame received while one of those gates is unavailable must still keep
        the watchdog from misreporting ``NO_FRAMES``.
        """
        with self._lock:
            self._last_frame_t = self._time(t_rel_s)

    def _observe_frame_unlocked(
        self,
        t_rel_s: float,
        *,
        pose_confident: bool,
        depth_valid: bool,
        reassociated: bool,
        calibration_valid: bool,
        details: Mapping[str, Any] | None,
    ) -> HealthTransition | None:
        t = self._time(t_rel_s)
        self._last_frame_t = t
        if not calibration_valid:
            return self._set(
                HealthStatus.UNAVAILABLE,
                (HealthReason.CALIBRATION_DRIFT,),
                t,
                details,
            )

        reasons: list[HealthReason] = []
        if not pose_confident:
            reasons.append(HealthReason.LOW_CONFIDENCE)
        if not depth_valid:
            reasons.append(HealthReason.DEPTH_MISSING)
        if reassociated:
            reasons.append(HealthReason.TRACK_REASSOCIATED)
        if reasons:
            return self._set(HealthStatus.DEGRADED, tuple(reasons), t, details)
        return self._set(HealthStatus.AVAILABLE, (), t, details)

    def tick(self, t_rel_s: float) -> HealthTransition | None:
        """Watchdog tick; marks a frame-starved pipeline unavailable."""
        with self._lock:
            return self._tick_unlocked(t_rel_s)

    def _tick_unlocked(self, t_rel_s: float) -> HealthTransition | None:
        t = self._time(t_rel_s)
        # Session time is relative and starts at zero. Give source/model startup
        # the same grace period as a mid-run frame before changing STARTING to
        # NO_FRAMES; otherwise the first watchdog tick would alarm immediately.
        age = t if self._last_frame_t is None else t - self._last_frame_t
        if age > self.stale_after_s:
            details = {} if not math.isfinite(age) else {"frame_age_s": round(age, 3)}
            return self._set(
                HealthStatus.UNAVAILABLE,
                (HealthReason.NO_FRAMES,),
                t,
                details,
            )
        return None

    def mark_unavailable(
        self,
        t_rel_s: float,
        reason: HealthReason,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> HealthTransition | None:
        """Open an explicit unavailable interval for a terminal/operational fault."""
        with self._lock:
            return self._mark_unavailable_unlocked(
                t_rel_s, reason, details=details
            )

    def _mark_unavailable_unlocked(
        self,
        t_rel_s: float,
        reason: HealthReason,
        *,
        details: Mapping[str, Any] | None,
    ) -> HealthTransition | None:
        allowed = {
            HealthReason.NO_FRAMES,
            HealthReason.TARGET_NOT_BOUND,
            HealthReason.PRIVACY_SCOPE_VIOLATION,
            HealthReason.CALIBRATION_DRIFT,
            HealthReason.CAMERA_DISCONNECTED,
            HealthReason.DISK_ERROR,
            HealthReason.SOURCE_ENDED,
            HealthReason.PROTOCOL_COMPLETE,
            HealthReason.PROTOCOL_INCOMPLETE,
            HealthReason.REQUESTED_STOP,
            HealthReason.OPERATOR_STOP,
            HealthReason.SESSION_LIMIT,
        }
        if reason not in allowed:
            raise ValueError(reason.value + " is degraded evidence, not an unavailable reason")
        return self._set(HealthStatus.UNAVAILABLE, (reason,), t_rel_s, details)
