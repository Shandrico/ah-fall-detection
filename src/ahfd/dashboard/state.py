"""Shared, thread-safe state between the pipeline and the web server.

The design here is a direct response to what went wrong in the reference
implementation, where the dashboard drove a Jetson to 99 C. Two rules:

1. **One producer.** The pipeline thread encodes the annotated frame to JPEG
   exactly once per frame and stores the bytes here. Viewers read those bytes;
   they never trigger encoding. Ten browsers cost the same as one.
2. **No per-viewer work in the store.** Reads copy a small snapshot under a
   short-held lock and return. Nothing here loops, sleeps, or blocks.

Session totals are kept as running counters that only increment, separate from
the bounded recent-events log. The log is capped so memory is bounded over a
long shift; the counters are not, so "falls today" stays correct even after the
log has scrolled past them.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# Which event types are a standing alert a nurse must clear, versus
# informational. Drives the triage queue and the "open alerts" count.
ALERTING_TYPES = frozenset({"FALL_CONFIRMED", "PERSON_DOWN"})


class DashboardState:
    """The latest annotated frame plus recent and cumulative detection state."""

    def __init__(self, max_events: int = 200):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._tracks: list[dict[str, Any]] = []
        self._people = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._acked: set[str] = set()
        self._counts: dict[str, int] = {}  # cumulative per event type
        self._fps = 0.0
        self._started = time.time()
        self._last_alert_ts = 0.0  # wall-clock of the most recent alerting event

    # ---- producer side (pipeline thread) -------------------------------

    def publish_frame(
        self, jpeg: bytes, tracks: list[dict[str, Any]], fps: float
    ) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self._tracks = list(tracks)
            self._people = len(tracks)
            self._fps = fps

    def publish_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            self._counts[event["type"]] = self._counts.get(event["type"], 0) + 1
            if event.get("type") in ALERTING_TYPES:
                self._last_alert_ts = time.time()

    # ---- consumer side (web server threads) ----------------------------

    def latest_frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._seq

    def snapshot(self) -> dict[str, Any]:
        """A small JSON-ready view of current state."""
        with self._lock:
            events = list(self._events)
            acked = set(self._acked)
            counts = dict(self._counts)
            open_alerts = [
                {**e, "acknowledged": e.get("event_id") in acked}
                for e in reversed(events)
                if e.get("type") in ALERTING_TYPES and e.get("event_id") not in acked
            ]
            return {
                "fps": round(self._fps, 1),
                "uptime_s": round(time.time() - self._started, 1),
                "people": self._people,
                "tracks": self._tracks,
                "counts": {
                    "fall_confirmed": counts.get("FALL_CONFIRMED", 0),
                    "person_down": counts.get("PERSON_DOWN", 0),
                    "fall_suspected": counts.get("FALL_SUSPECTED", 0),
                    "bed_exit": counts.get("BED_EXIT", 0),
                    "near_miss": counts.get("NEAR_MISS", 0),
                },
                "open_alerts": open_alerts,
                "open_count": len(open_alerts),
                "seconds_since_alert": (
                    round(time.time() - self._last_alert_ts, 1)
                    if self._last_alert_ts
                    else None
                ),
                "events": [
                    {**e, "acknowledged": e.get("event_id") in acked}
                    for e in reversed(events)
                ],
            }

    def acknowledge(self, event_id: str) -> None:
        with self._lock:
            self._acked.add(event_id)

    def unacknowledge(self, event_id: str) -> None:
        with self._lock:
            self._acked.discard(event_id)
