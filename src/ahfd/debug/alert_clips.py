"""Save a short video around each alert, for evaluation and report evidence.

Frames wait in a few seconds of memory and are thrown away unless an alert happens,
so nothing reaches the disk during normal running. An alert keeps the seconds before
it, carries on for the seconds after, and writes one .mp4 with a .json of the alerts.
Alerts close together share a clip.

Two modes:

  skeleton  stick figures on a blank background: no camera pixels exist in the buffer
            or the file. The default.
  rgb       the camera image. Needs all three raw-capture switches, may not be
            written inside the repository, and only into a folder whose parent
            already exists, so an unmounted T9 cannot redirect it to the internal disk.

Lives in ``ahfd.debug`` because it can write imagery.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ahfd.privacy import PrivacyViolation, raw_capture_allowed

_REPO = Path(__file__).resolve().parents[3]


def check_clip_settings(mode: str, out_dir: str | Path, *, config_flag: bool, cli_flag: bool) -> None:
    """Raise if these settings may not save clips. Skeleton mode is always allowed."""
    if mode == "skeleton":
        return
    if mode != "rgb":
        raise ValueError("alert clip mode must be 'skeleton' or 'rgb', not " + repr(mode))
    if not raw_capture_allowed(config_flag=config_flag, cli_flag=cli_flag):
        raise PrivacyViolation(
            "RGB alert clips need all three raw-capture switches: config "
            "privacy.allow_raw_capture, AHFD_ALLOW_RAW=1 and --i-understand-raw-capture"
        )
    path = Path(out_dir).resolve()
    if path == _REPO or _REPO in path.parents:
        raise PrivacyViolation("RGB alert clips may not be written inside the repository: " + str(path))
    if not path.parent.is_dir():
        raise PrivacyViolation("RGB alert clip folder is not available (is the T9 mounted?): " + str(path))


@dataclass
class _Clip:
    frames: list = field(default_factory=list)  # (monotonic time, jpeg, is_rgb)
    events: list = field(default_factory=list)
    started_wall: float = 0.0
    first_alert_t: float = 0.0
    end_t: float = 0.0


class AlertClipRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        *,
        mode: str = "skeleton",
        pre_s: float = 10.0,
        post_s: float = 15.0,
        max_s: float = 60.0,
        min_free_gb: float = 2.0,
        source: str = "",
        log=print,
        clock=time.monotonic,
    ):
        self.out_dir = Path(out_dir)
        self.mode = mode
        self.pre_s, self.post_s, self.max_s = pre_s, post_s, max_s
        self.min_free_gb = min_free_gb
        self.source = source
        self._log = log
        self._clock = clock
        self._ring: deque = deque()
        self._clip: _Clip | None = None
        self._writers: list[threading.Thread] = []
        self._lock = threading.Lock()

    def add_frame(self, jpeg: bytes, *, is_rgb: bool = False) -> None:
        """Offer one rendered frame, on the pipeline thread."""
        if is_rgb and self.mode != "rgb":
            raise PrivacyViolation("an RGB frame reached a skeleton-mode alert clip recorder")
        now = self._clock()
        with self._lock:
            if not is_rgb:
                # RGB was switched off: what was on screen must not reach a file later.
                self._forget_rgb()
            self._ring.append((now, jpeg, is_rgb))
            while self._ring and now - self._ring[0][0] > self.pre_s:
                self._ring.popleft()
            clip = self._clip
            if clip is None:
                return
            clip.frames.append((now, jpeg, is_rgb))
            if now >= clip.end_t:
                self._clip = None
                self._write_later(clip)

    def _forget_rgb(self) -> None:
        if self._ring and self._ring[-1][2]:
            self._ring = deque(f for f in self._ring if not f[2])
            if self._clip is not None:
                self._clip.frames = [f for f in self._clip.frames if not f[2]]

    def on_alert(self, event: dict) -> None:
        """An alert fired: start a clip, or keep the open one going."""
        now = self._clock()
        with self._lock:
            clip = self._clip
            if clip is None:
                clip = self._clip = _Clip(
                    frames=list(self._ring),
                    started_wall=time.time() - (now - self._ring[0][0] if self._ring else 0.0),
                    first_alert_t=now,
                )
                # A camera that dies mid-incident sends no more frames, and that clip matters most.
                threading.Thread(target=self._finish_when_due, args=(clip,), daemon=True).start()
            clip.events.append(event)
            clip.end_t = min(now + self.post_s, clip.first_alert_t + self.max_s)

    def _finish_when_due(self, clip: _Clip) -> None:
        while True:
            time.sleep(0.25)
            with self._lock:
                if self._clip is not clip:
                    return
                if self._clock() >= clip.end_t + 1.0:
                    self._clip = None
                    self._write_later(clip)
                    return

    def close(self, timeout: float = 20.0) -> None:
        """Finish what is open: a clip must not be lost because the camera was switched."""
        with self._lock:
            clip, self._clip = self._clip, None
            if clip is not None:
                self._write_later(clip)
            writers = list(self._writers)
        for writer in writers:
            writer.join(timeout)

    def _write_later(self, clip: _Clip) -> None:
        # Encoding is slow: it must not hold up the detector.
        writer = threading.Thread(target=self._write, args=(clip,), daemon=True, name="alert-clip")
        self._writers = [w for w in self._writers if w.is_alive()] + [writer]
        writer.start()

    def _usable_dir(self) -> bool:
        try:
            if self.mode == "rgb":
                if not self.out_dir.parent.is_dir():
                    self._log("alert clip dropped: folder is gone (is the T9 mounted?): " + str(self.out_dir))
                    return False
                self.out_dir.mkdir(exist_ok=True)
            else:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(self.out_dir).free / 1e9
        except OSError as exc:
            self._log("alert clip dropped: " + str(exc))
            return False
        if free_gb < self.min_free_gb:
            self._log("alert clip dropped: only %.1f GB free in %s" % (free_gb, self.out_dir))
            return False
        return True

    def _write(self, clip: _Clip) -> None:
        if not clip.frames or not self._usable_dir():
            return
        first = clip.events[0]
        stamp = datetime.fromtimestamp(clip.started_wall).strftime("%Y%m%d_%H%M%S")
        where = re.sub(r"[^A-Za-z0-9_-]", "", str(first.get("zone") or first.get("near_bed") or "room"))
        stem = "_".join((stamp, re.sub(r"[^A-Z_]", "", str(first.get("type"))), where))
        duration = max(clip.frames[-1][0] - clip.frames[0][0], 0.1)
        fps = max(len(clip.frames) / duration, 1.0)
        video = self.out_dir / (stem + ".mp4")
        partial = self.out_dir / (stem + ".partial.mp4")
        try:
            encode = _encode_ffmpeg if shutil.which("ffmpeg") else _encode_cv2
            encode(clip.frames, fps, partial)
            partial.replace(video)
        except Exception as exc:  # a failed clip must never take the detector down
            self._log("alert clip failed: " + str(exc))
            partial.unlink(missing_ok=True)
            return
        contains_rgb = any(f[2] for f in clip.frames)
        (self.out_dir / (stem + ".json")).write_text(
            json.dumps(
                {
                    "video": video.name,
                    "source": self.source,
                    "picture": "camera image" if contains_rgb else "stick figures only",
                    "started": datetime.fromtimestamp(clip.started_wall).isoformat(timespec="seconds"),
                    "seconds_before_first_alert": round(clip.first_alert_t - clip.frames[0][0], 1),
                    "duration_s": round(duration, 1),
                    "frames": len(clip.frames),
                    "fps": round(fps, 1),
                    "alerts": [
                        {k: e.get(k) for k in ("type", "clock", "zone", "near_bed", "severity", "track_id", "monitoring_mode", "evidence")}
                        for e in clip.events
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._log("saved alert clip: " + str(video))


def _encode_ffmpeg(frames: list, fps: float, out: Path) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", "%.3f" % fps, "-i", "-",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(out),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, err = proc.communicate(b"".join(f[1] for f in frames))
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + err.decode(errors="replace")[-200:])


def _encode_cv2(frames: list, fps: float, out: Path) -> None:
    """No ffmpeg (a Windows laptop, say): plays in VLC, though not in every browser."""
    import cv2
    import numpy as np

    writer = None
    try:
        for _, jpeg, _ in frames:
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
