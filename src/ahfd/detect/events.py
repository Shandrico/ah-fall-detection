"""Events the detector emits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "FALL_SUSPECTED",  # on screen only, fast
    "FALL_CONFIRMED",  # pages a nurse, after the person stays down
    "PERSON_DOWN",  # slow slump, or a fall we joined late
    "BED_EXIT",  # sitting on the bed edge -- a precursor, not an incident
    "NEAR_MISS",  # went down and got straight back up
]

# What each event is worth waking someone for.
SEVERITY: dict[str, int] = {
    "BED_EXIT": 1,
    "NEAR_MISS": 1,
    "FALL_SUSPECTED": 2,
    "PERSON_DOWN": 3,
    "FALL_CONFIRMED": 4,
}


@dataclass(frozen=True)
class Event:
    """A single detection.

    `evidence` carries the numbers that caused it. That is not decoration: the
    first question after any alert -- especially a false one -- is "why did it
    fire", and an answer like "torso dropped 0.95 m to 0.18 m in 0.6 s, peak
    -1.4 m/s, still for 8.0 s, on the floor beside Bed 2" can be discussed
    with a nurse. A confidence score cannot.
    """

    type: EventType
    track_id: int
    t_trigger: float  # when the causing motion happened
    t_alert: float  # when this event was emitted
    zone: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    # Some events grade their own urgency rather than taking it from the type.
    # A bed exit is the case that matters: the same action is a quiet status for
    # a low-risk patient and a real alert for a high-risk one, so its severity
    # is set per event from the bed's risk level, not fixed by "BED_EXIT".
    severity_override: int | None = None

    @property
    def severity(self) -> int:
        if self.severity_override is not None:
            return self.severity_override
        return SEVERITY.get(self.type, 0)

    @property
    def latency_s(self) -> float:
        return self.t_alert - self.t_trigger

    def describe(self) -> str:
        where = (" in " + self.zone) if self.zone else ""
        return (
            self.type
            + " track "
            + str(self.track_id)
            + where
            + " at t="
            + format(self.t_alert, ".1f")
            + "s"
        )
