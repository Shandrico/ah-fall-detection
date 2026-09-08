"""Alert sinks: console and JSON Lines.

Every sink takes an `Event` and nothing else. `Event` holds no imagery, so no
alert path can leak a frame -- the privacy guarantee survives however this is
extended.

The JSONL writer is the event log the evaluation harness reads back, so the
format is append-only, one self-contained JSON object per line, and stable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ahfd.detect.events import Event


class AlertSink(Protocol):
    def emit(self, event: Event) -> None: ...

    def close(self) -> None: ...


class ConsoleSink:
    """Human-readable one-liners.

    Leads with the event type and the evidence, because the first question
    about any alert -- particularly a false one -- is why it fired.
    """

    _MARKS = {
        "FALL_CONFIRMED": "!!",
        "PERSON_DOWN": "!!",
        "FALL_SUSPECTED": " !",
        "BED_EXIT": " ~",
        "NEAR_MISS": " .",
    }

    def __init__(self, min_severity: int = 0):
        self.min_severity = min_severity

    def emit(self, event: Event) -> None:
        if event.severity < self.min_severity:
            return
        mark = self._MARKS.get(event.type, "  ")
        evidence = " ".join(
            key + "=" + str(value) for key, value in sorted(event.evidence.items())
        )
        print(
            mark
            + " "
            + format(event.t_alert, "7.1f")
            + "s  "
            + event.type.ljust(15)
            + " track "
            + str(event.track_id).ljust(3)
            + (event.zone or "-").ljust(12)
            + evidence
        )

    def close(self) -> None:
        pass


class JsonlSink:
    """Append-only event log, one JSON object per line."""

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._file = self.path.open("a", encoding="utf-8")

    def emit(self, event: Event) -> None:
        record = {
            "run_id": self.run_id,
            "type": event.type,
            "severity": event.severity,
            "track_id": event.track_id,
            "t_trigger": round(event.t_trigger, 3),
            "t_alert": round(event.t_alert, 3),
            "latency_s": round(event.latency_s, 3),
            "zone": event.zone,
            "evidence": event.evidence,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Flushed per event: an alert that exists only in a buffer when the
        # process is killed is an alert that never happened.
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class MultiSink:
    """Fan out to several sinks.

    One failing sink must not take down the others, or a broken log file would
    also silence the console -- so failures are reported and swallowed.
    """

    def __init__(self, *sinks: AlertSink):
        self.sinks = list(sinks)

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001 - see docstring
                print("alert sink " + type(sink).__name__ + " failed: " + str(exc))

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                pass
