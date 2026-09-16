"""Shared, thread-safe state between the pipeline and the web server.

The design here is a direct response to what went wrong in the reference
implementation, where the dashboard drove a Jetson to 99 C. Two rules:

1. **One producer of frames.** The pipeline thread encodes the annotated
   frame to JPEG exactly once per frame and stores the bytes here. Viewers read
   those bytes; they never trigger encoding. Ten browsers cost the same as one.
   The control plane (a switch request, an acknowledgement) also writes here,
   but only small scalars under the same short-held lock -- the rule is about
   cost, not exclusivity.
2. **No per-viewer work in the store.** Reads copy a small snapshot under a
   short-held lock and return. Nothing here loops, sleeps, or blocks.

Publishing is fenced by a **generation** number. A switch bumps it before the
outgoing pipeline is even asked to stop, so everything that pipeline publishes
afterwards is dropped. That matters because `stop()` can time out: a thread
wedged in a blocking read on an unplugged camera stays alive, and without the
fence it would keep painting frames -- and raising alerts -- from a camera
nobody is looking at any more. Losing one frame period of real events is the
better half of that trade.

Session totals are kept as running counters that only increment, separate from
the bounded recent-events log. The history is capped over a long shift; the
counters survive rollover, and outstanding alerts remain until acknowledged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from functools import cache
from typing import Any

# Which event types are a standing alert a nurse must clear, versus
# informational. Drives the triage queue and the "open alerts" count.
ALERTING_TYPES = frozenset({"FALL_CONFIRMED", "PERSON_DOWN"})


@cache
def _privacy_frame() -> bytes:
    """One safe replacement, encoded once rather than once per viewer."""
    import cv2
    import numpy as np

    ok, jpeg = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))
    if not ok:
        raise RuntimeError("could not encode privacy frame")
    return jpeg.tobytes()


class DashboardState:
    """The latest annotated frame plus recent and cumulative detection state."""

    def __init__(self, max_events: int = 200):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._tracks: list[dict[str, Any]] = []
        self._people = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        # Outstanding alerts are not history: scrolling the log must never
        # silently clear an alert that still needs a human acknowledgement.
        self._outstanding_alerts: dict[str, dict[str, Any]] = {}
        self._acked: set[str] = set()
        # None preserves direct producer use; the controller explicitly sets
        # the session's view policy before starting any pipeline.
        self._rgb_enabled: bool | None = None
        self._counts: dict[str, int] = {}  # cumulative per event type
        self._fps = 0.0
        self._started = time.time()
        self._last_alert_ts = 0.0  # wall-clock of the most recent alerting event
        # Control plane: which pipeline may publish, and what it is doing.
        self._gen = 0
        self._switch_seq = 0
        self._status = "idle"
        self._status_since = time.time()
        self._error: str | None = None
        self._runtime: dict[str, Any] = {}
        # Replay control plane: on only for a seekable file source, so the page
        # can show a scrub bar and play/pause/step. A live camera leaves this
        # off (there is nothing to seek). Small scalars, same short-held lock.
        self._replay_seekable = False
        self._replay_total = 0
        self._replay_cur = 0
        self._replay_paused = False
        self._replay_seek: int | None = None
        self._replay_speed = 1.0

    # ---- producer side (pipeline thread) -------------------------------

    def publish_frame(
        self, jpeg: bytes, tracks: list[dict[str, Any]], fps: float, gen: int = 0,
        *, show_rgb: bool = False,
    ) -> None:
        with self._lock:
            if gen != self._gen:
                return  # a retired pipeline; see the module docstring
            if show_rgb and self._rgb_enabled is False:
                return  # RGB rendered before an off request must not reappear
            self._jpeg = jpeg
            self._seq += 1
            self._tracks = list(tracks)
            self._people = len(tracks)
            self._fps = fps

    def publish_event(self, event: dict[str, Any], gen: int = 0) -> None:
        with self._lock:
            if gen != self._gen:
                return
            self._events.append(event)
            self._counts[event["type"]] = self._counts.get(event["type"], 0) + 1
            if event.get("type") in ALERTING_TYPES:
                self._last_alert_ts = time.time()
                if event["event_id"] not in self._acked:
                    self._outstanding_alerts[event["event_id"]] = event

    # ---- control plane (controller thread) ------------------------------

    def set_rgb(self, on: bool) -> None:
        """Change the publication policy and replace cached RGB immediately.

        A paused recording or disconnected camera may never publish another
        frame. Sending a new safe JPEG (not just None) also replaces the last
        RGB part already displayed by existing MJPEG viewers.
        """
        replacement = _privacy_frame() if not on else None
        with self._lock:
            self._rgb_enabled = on
            self._runtime["show_rgb"] = on
            if not on and self._jpeg is not None:
                self._jpeg = replacement
                self._seq += 1

    def begin_generation(self, **switching_to: Any) -> int:
        """Claim the publishing slot for a new pipeline, fencing out the old.

        The live metrics are cleared with it: "three people in view" left over
        from a camera that is no longer running is worse than a blank. The
        last JPEG is deliberately kept -- MJPEG holds the last part on screen
        regardless, and clearing it would blank the feed for anyone who
        connects mid-switch.
        """
        with self._lock:
            self._gen += 1
            self._switch_seq += 1
            self._fps = 0.0
            self._tracks = []
            self._people = 0
            self._status = "switching"
            self._status_since = time.time()
            self._error = None
            self._runtime = dict(switching_to)
            # The new source might not be seekable; clear the player until it
            # announces itself. Otherwise stale controls would drive a camera.
            self._replay_seekable = False
            self._replay_total = 0
            self._replay_cur = 0
            self._replay_paused = False
            self._replay_seek = None
            self._replay_speed = 1.0
            return self._gen

    @property
    def generation(self) -> int:
        with self._lock:
            return self._gen

    def publish_status(
        self, gen: int, status: str, *, error: str | None = None, **info: Any
    ) -> None:
        """Report where a pipeline is: starting / running / ended / error."""
        with self._lock:
            if gen != self._gen:
                return  # a retired pipeline's last words
            self._status = status
            self._status_since = time.time()
            self._error = error
            self._runtime.update({k: v for k, v in info.items() if v is not None})

    def update_runtime(self, gen: int, **info: Any) -> None:
        """Amend the runtime detail without touching the status or its clock."""
        with self._lock:
            if gen != self._gen:
                return
            self._runtime.update(info)

    def announce_replay(self, total: int, gen: int) -> None:
        """A seekable-file pipeline turns the player on and reports its length."""
        with self._lock:
            if gen != self._gen:
                return
            self._replay_seekable = True
            self._replay_total = int(total)
            self._replay_cur = 0
            self._replay_paused = False
            self._replay_seek = None
            self._replay_speed = 1.0

    def publish_replay_pos(self, cur: int, gen: int) -> None:
        """The pipeline reports which frame it is now showing (for the slider)."""
        with self._lock:
            if gen != self._gen:
                return
            self._replay_cur = int(cur)

    def replay_control(self, action: str | None, value: Any = None) -> bool:
        """A player command from the web server. Returns whether it was applied.

        `step` and `seek` set a pending target frame the pipeline picks up on
        its next tick; `step` also pauses, so the frame it lands on stays put.
        """
        with self._lock:
            if not self._replay_seekable:
                return False
            if action == "pause":
                self._replay_paused = True
            elif action == "play":
                self._replay_paused = False
            elif action == "seek" and value is not None:
                self._replay_seek = int(value)
            elif action == "step" and value is not None:
                self._replay_seek = self._replay_cur + int(value)
                self._replay_paused = True
            elif action == "speed" and value is not None:
                self._replay_speed = max(0.1, min(4.0, float(value)))
            else:
                return False
            return True

    def take_replay_command(self) -> tuple[bool, int | None, float]:
        """The pipeline reads (paused, pending seek target, speed); clears the seek."""
        with self._lock:
            seek = self._replay_seek
            self._replay_seek = None
            return self._replay_paused, seek, self._replay_speed

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
                {**e, "acknowledged": False}
                for e in reversed(list(self._outstanding_alerts.values()))
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
                # Which camera and model are running, and whether a switch is
                # in flight. The authoritative keys go last so a stale entry in
                # _runtime can never shadow them.
                "runtime": {
                    **self._runtime,
                    # A pipeline may have read its flag before an off request;
                    # the control-plane policy is authoritative, not that read.
                    **(
                        {"show_rgb": self._rgb_enabled}
                        if self._rgb_enabled is not None else {}
                    ),
                    "status": self._status,
                    "since_s": round(time.time() - self._status_since, 1),
                    "error": self._error,
                    "switch_seq": self._switch_seq,
                },
                # Only present for a seekable file, so the page shows the player
                # for recordings and nothing for a live camera.
                "replay": (
                    {
                        "total": self._replay_total,
                        "cur": self._replay_cur,
                        "paused": self._replay_paused,
                        "speed": self._replay_speed,
                    }
                    if self._replay_seekable
                    else None
                ),
            }

    def acknowledge(self, event_id: str) -> None:
        with self._lock:
            self._acked.add(event_id)
            self._outstanding_alerts.pop(event_id, None)

    def unacknowledge(self, event_id: str) -> None:
        with self._lock:
            self._acked.discard(event_id)
            for event in reversed(self._events):
                if event.get("event_id") == event_id:
                    if event.get("type") in ALERTING_TYPES:
                        self._outstanding_alerts[event_id] = event
                    break
