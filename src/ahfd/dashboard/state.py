"""Shared, thread-safe state between the pipeline and the web server.

The design here is a direct response to what went wrong in the reference
implementation, where the dashboard drove a Jetson to 99 C. Two rules:

1. **One producer.** The pipeline thread encodes the annotated frame to JPEG
   exactly once per frame and stores the bytes here. Viewers read those bytes;
   they never trigger encoding. Ten browsers cost the same as one.
2. **No per-viewer work in the store.** Reads copy a small snapshot under a
   short-held lock and return. Nothing here loops, sleeps, or blocks.

The failure that cooked the reference build was per-request encoding plus a
stream generator that never noticed the browser had gone, so reconnects piled
up threads. Keeping all the real work in the single producer thread removes
both.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class DashboardState:
    """The latest annotated frame plus recent detection state."""

    def __init__(self, max_events: int = 100):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._tracks: dict[int, str] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._acked: set[str] = set()
        self._fps = 0.0
        self._started = time.time()

    # ---- producer side (pipeline thread) -------------------------------

    def publish_frame(self, jpeg: bytes, tracks: dict[int, str], fps: float) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self._tracks = dict(tracks)
            self._fps = fps

    def publish_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    # ---- consumer side (web server threads) ----------------------------

    def latest_frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._seq

    def snapshot(self) -> dict[str, Any]:
        """A small JSON-ready view of current state."""
        with self._lock:
            events = list(self._events)
            acked = set(self._acked)
            return {
                "fps": round(self._fps, 1),
                "uptime_s": round(time.time() - self._started, 1),
                "tracks": [
                    {"track_id": tid, "state": state}
                    for tid, state in sorted(self._tracks.items())
                ],
                "events": [
                    {**e, "acknowledged": e.get("event_id") in acked}
                    for e in reversed(events)
                ],
                "open_alerts": sum(
                    1
                    for e in events
                    if e.get("severity", 0) >= 3 and e.get("event_id") not in acked
                ),
            }

    def acknowledge(self, event_id: str) -> None:
        with self._lock:
            self._acked.add(event_id)
