"""Raw video recorder for consented staged-fall sessions.

This is the one place raw RGB is legitimately written to disk, and it exists
for exactly one purpose: capturing staged falls by consenting volunteers (never
patients) so the team has footage to extract keypoints from and tune against.

It is fenced three ways:

* **Runtime gate.** Nothing records unless `ahfd.privacy.require_raw_capture`
  passes -- config flag AND env var AND CLI acknowledgement, all three. The
  default is off in every one of them.
* **Loud when on.** A banner is burned into every recorded frame and a warning
  is logged, so a session cannot be recording without everyone in the room
  being able to see that it is.
* **Structural.** This is the only module the privacy test permits to call an
  image writer, so a stray `VideoWriter` anywhere else fails the build.

The intended workflow is: record a session here, run `ahfd extract` on the
file to get keypoints, then **delete the raw video** and keep only the
tracks.jsonl. The footage is scaffolding for tuning, not something to retain.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ahfd.privacy import BANNER, require_raw_capture


class RawRecorder:
    """Writes frames to a video file, but only with full consent.

    Construction raises `PrivacyViolation` unless all three switches agree, so
    it is impossible to hold an armed recorder by accident.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        *,
        config_flag: bool,
        cli_flag: bool,
    ):
        # Refuse before opening any file if consent is incomplete.
        require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, fps if fps > 0 else 30.0, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError("could not open video writer for " + str(self.path))
        self._frames = 0

    def write(self, bgr: np.ndarray) -> None:
        """Record one frame, with the recording banner burned in."""
        marked = bgr.copy()
        cv2.rectangle(marked, (0, 0), (marked.shape[1], 34), (0, 0, 160), -1)
        cv2.putText(
            marked,
            "REC  " + BANNER,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._writer.write(marked)
        self._frames += 1

    @property
    def count(self) -> int:
        return self._frames

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "RawRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
