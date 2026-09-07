"""The frame source contract.

One interface covers the live RealSense, a recorded .bag, a video file, a
webcam and a directory of dataset images. That matters for two reasons:

* Development and evaluation proceed with no camera attached -- which is the
  situation this project has been in, and will be again whenever the hardware
  is on somebody else's desk.
* Offline replay is frame-exact, so the same clip produces the same events on
  every run. Without that, threshold tuning is guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from ahfd.types import Intrinsics


@dataclass(frozen=True)
class SourceMeta:
    """What a source can report about itself before you read from it."""

    uri: str
    width: int
    height: int
    fps: float
    has_depth: bool = False
    frame_count: int | None = None  # None for live sources
    intrinsics: Intrinsics | None = None


@dataclass(frozen=True, eq=False)
class Frame:
    """A single moment from a source.

    This is the *only* type in the project that holds pixels. It is consumed by
    the pose stage and dropped; see `ahfd.privacy` for why that is enforced
    rather than merely intended.
    """

    index: int
    t: float  # seconds in the source timebase, not wall clock
    bgr: np.ndarray | None  # (H, W, 3) uint8, OpenCV channel order
    depth: np.ndarray | None = None  # (H, W) uint16, aligned to bgr
    depth_raw: np.ndarray | None = None  # not hole-filled -- measure from this
    depth_scale: float = 0.001
    intrinsics: Intrinsics | None = None
    gravity: np.ndarray | None = None  # unit vector in camera frame, from IMU

    @property
    def shape(self) -> tuple[int, int]:
        if self.bgr is not None:
            return self.bgr.shape[0], self.bgr.shape[1]
        if self.depth is not None:
            return self.depth.shape[0], self.depth.shape[1]
        raise ValueError("frame carries neither colour nor depth")


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields Frames in order."""

    @property
    def meta(self) -> SourceMeta: ...

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...
