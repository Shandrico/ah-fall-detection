"""Image-sequence source: a directory of frames.

Two uses, one code path:

* Public fall datasets. UR Fall Detection and Le2i ship as numbered PNG/JPG
  sequences rather than video, and this reads them directly so the same
  extract -> detect -> eval pipeline runs over them.
* Recorded frame dumps from a staged session.

Frames are ordered by filename, which is why datasets pad their numbers
(frame_0001.png). A natural sort is used so that an unpadded `frame_2` still
sorts before `frame_10` -- padded or not, the order is what a human expects.

Timestamps are synthesised from a supplied fps, because a directory of images
carries no timing of its own. That makes velocity meaningful and playback
deterministic -- the same directory yields the same timeline every run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import cv2

from ahfd.capture.base import Frame, SourceMeta

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _natural_key(path: Path):
    """Sort key that orders frame_2 before frame_10."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.name)
    ]


class ImageSequenceSource:
    """Frames from a directory of images, ordered by filename."""

    def __init__(self, directory: str | Path, fps: float = 30.0, uri: str | None = None):
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise RuntimeError("not a directory: " + str(self._dir))

        self._paths = sorted(
            (p for p in self._dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES),
            key=_natural_key,
        )
        if not self._paths:
            raise RuntimeError("no images found in " + str(self._dir))

        self._fps = fps if fps > 0 else 30.0

        # Read the first image for the frame size, rather than trusting a guess.
        first = cv2.imread(str(self._paths[0]))
        if first is None:
            raise RuntimeError("could not read " + str(self._paths[0]))
        h, w = first.shape[:2]

        self._meta = SourceMeta(
            uri=uri or ("seq://" + str(self._dir)),
            width=w,
            height=h,
            fps=self._fps,
            has_depth=False,
            frame_count=len(self._paths),
        )

    @property
    def meta(self) -> SourceMeta:
        return self._meta

    def __iter__(self) -> Iterator[Frame]:
        for index, path in enumerate(self._paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                # A single unreadable frame should not abort the whole clip.
                continue
            yield Frame(index=index, t=index / self._fps, bgr=bgr)

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageSequenceSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
