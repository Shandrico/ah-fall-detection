"""Core data types.

The single most important property of this module: **nothing here holds an
image**. RGB exists only inside `ahfd.capture.base.Frame`, which is consumed by
the pose stage and dropped. Everything downstream -- features, fall decisions,
alerts, recordings -- is built from these coordinate-only types, so raw video
cannot leak past the pose stage even by accident. There is no field to put it
in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# RTMO and the other COCO-trained pose models all emit these 17 joints.
NUM_KEYPOINTS = 17


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics, in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(
        cls, width: int, height: int, hfov_deg: float, vfov_deg: float | None = None
    ) -> "Intrinsics":
        """Approximate intrinsics from a datasheet field of view.

        A stopgap for cameras we have not calibrated -- a laptop webcam, or a
        D435i before its factory intrinsics are read off the device. Good
        enough to develop the geometry against; not good enough for the metric
        numbers that go in a report, because lens distortion is ignored
        entirely and a datasheet FOV is a nominal figure.

        Prefer real intrinsics wherever they exist: `rs.video_stream_profile`
        supplies them for the RealSense.
        """
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        # Square pixels unless a vertical FOV says otherwise.
        fy = (
            fx
            if vfov_deg is None
            else (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        )
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
        )


@dataclass(frozen=True, eq=False)
class PersonPose:
    """One person detected in one frame.

    Coordinates are in pixels of the source frame. `track_id` is None until a
    tracker has assigned a stable identity.
    """

    keypoints: np.ndarray  # (17, 2) float32 -- (u, v) pixel coordinates
    scores: np.ndarray  # (17,) float32 -- per-keypoint confidence
    score: float  # overall person confidence
    track_id: int | None = None

    def valid_mask(self, min_score: float) -> np.ndarray:
        """Boolean mask of keypoints confident enough to use."""
        return self.scores >= min_score

    def n_valid(self, min_score: float) -> int:
        return int(np.count_nonzero(self.valid_mask(min_score)))

    def bbox(self, min_score: float = 0.3) -> tuple[float, float, float, float] | None:
        """Axis-aligned box around the confident keypoints, as (x1, y1, x2, y2).

        Derived rather than stored: RTMO is one-stage and reports keypoints, and
        a box inferred from confident joints is more useful downstream than the
        raw detection box anyway.
        """
        mask = self.valid_mask(min_score)
        if not mask.any():
            return None
        pts = self.keypoints[mask]
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    def with_track_id(self, track_id: int) -> "PersonPose":
        return PersonPose(
            keypoints=self.keypoints,
            scores=self.scores,
            score=self.score,
            track_id=track_id,
        )

    def with_keypoints(self, keypoints: np.ndarray) -> "PersonPose":
        """Same person, replaced coordinates -- used by the smoother."""
        return PersonPose(
            keypoints=keypoints,
            scores=self.scores,
            score=self.score,
            track_id=self.track_id,
        )


@dataclass(frozen=True, eq=False)
class PoseFrame:
    """All people detected in a single frame. Carries no imagery."""

    t: float  # seconds, source timebase -- not wall clock
    index: int  # frame number within the source
    width: int
    height: int
    people: tuple[PersonPose, ...]

    def __len__(self) -> int:
        return len(self.people)

    def with_people(self, people: tuple[PersonPose, ...]) -> "PoseFrame":
        """Same moment, different person list -- keeps the timebase intact."""
        return PoseFrame(
            t=self.t,
            index=self.index,
            width=self.width,
            height=self.height,
            people=people,
        )
