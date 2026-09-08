"""Ground-truth annotations and the system's own event log, for evaluation.

Two record types, kept deliberately separate:

* `GroundTruth` -- what actually happened in a clip, labelled by a human. A
  fall has a start and an impact time; a clip with no fall carries an empty
  event list *and its duration*, because that duration is the denominator for
  the false-alarm rate. A boring 30-minute clip of nobody falling is the most
  valuable negative there is, and it is worthless to the metric without its
  length recorded.

* The system's emitted events, read back from the JSONL the alert sink wrote.

The matcher in `metrics.py` compares the two. Keeping them apart matters: it
must be impossible to accidentally score the system against its own output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrueFall:
    """One real fall in a clip, as labelled by a person."""

    t_impact: float  # when the body hit the floor -- the reference instant
    t_start: float | None = None  # when the fall began, if known
    subject: str = ""
    notes: str = ""


@dataclass(frozen=True)
class GroundTruth:
    """The truth for one clip."""

    clip_id: str
    duration_s: float
    falls: tuple[TrueFall, ...] = ()

    @property
    def is_negative(self) -> bool:
        """A clip where nothing should fire. The false-alarm denominator."""
        return len(self.falls) == 0

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        falls = tuple(
            TrueFall(
                t_impact=float(f["t_impact"]),
                t_start=(
                    float(f["t_start"]) if f.get("t_start") is not None else None
                ),
                subject=str(f.get("subject", "")),
                notes=str(f.get("notes", "")),
            )
            for f in data.get("falls", [])
        )
        if "duration_s" not in data:
            raise ValueError(
                "clip "
                + repr(data.get("clip_id", "?"))
                + " has no duration_s; it is the false-alarm denominator and "
                "must be recorded even for clips where nothing happens"
            )
        return cls(
            clip_id=str(data["clip_id"]),
            duration_s=float(data["duration_s"]),
            falls=falls,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruth":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass(frozen=True)
class PredictedEvent:
    """One event the system emitted, read back from the JSONL log."""

    type: str
    t_alert: float
    t_trigger: float
    track_id: int
    severity: int = 0
    zone: str | None = None
    evidence: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PredictedEvent":
        return cls(
            type=str(data["type"]),
            t_alert=float(data["t_alert"]),
            t_trigger=float(data.get("t_trigger", data["t_alert"])),
            track_id=int(data.get("track_id", -1)),
            severity=int(data.get("severity", 0)),
            zone=data.get("zone"),
            evidence=data.get("evidence", {}),
        )


def load_events(path: str | Path) -> list[PredictedEvent]:
    """Read an events JSONL log back into records."""
    path = Path(path)
    events: list[PredictedEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(PredictedEvent.from_dict(json.loads(line)))
    return events
