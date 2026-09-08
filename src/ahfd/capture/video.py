"""Webcam and video-file source, via OpenCV.

This is the development path that needs no RealSense: it works on any built-in
laptop camera and on any recorded clip, including public fall datasets that
ship as video.
"""

from __future__ import annotations

import time
from typing import Iterator

import cv2

from ahfd.capture.base import Frame, SourceMeta


class VideoSource:
    """Frames from a webcam index or a video file path.

    The timebase differs by kind, deliberately. A file uses frame_index / fps,
    so replay is deterministic and independent of how fast the machine
    decodes. A live camera uses elapsed monotonic time, because dropped frames
    are real and pretending otherwise would understate fall velocities.
    """

    def __init__(
        self,
        target: int | str,
        uri: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        self._live = isinstance(target, int)
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError("could not open video source: " + repr(target))

        # Request a resolution if asked. The camera may not honour it exactly,
        # so the reported size below is read back rather than assumed.
        if self._live and width and height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._meta = SourceMeta(
            uri=uri or str(target),
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,
            has_depth=False,
            frame_count=None if self._live or count <= 0 else count,
        )

    @property
    def meta(self) -> SourceMeta:
        return self._meta

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        start = time.monotonic()
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                break
            t = (time.monotonic() - start) if self._live else index / self._meta.fps
            yield Frame(index=index, t=t, bgr=bgr)
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
