"""Skeleton drawn on top of the RGB frame.

Two uses:

* A development view (`ahfd run --view overlay`) for judging whether the pose
  is actually tracking the body -- on a black background a bad skeleton and a
  good one look similar, but over the video the error is obvious.
* The nurse dashboard's RGB mode.

Both **display** RGB; neither writes it to disk. That distinction is the whole
privacy question here. The system's stored data is still keypoints only (see
`ahfd.types`), and nothing in this module persists a frame. But putting RGB on
a screen is a different decision from keeping it off disk, and turning it on in
the dashboard reverses the "skeleton only" stance chosen for the ward -- so it
lives behind an explicit toggle, off by default, documented as needing sign-off.

This function takes a `Frame` (which holds pixels) rather than a `PoseFrame`,
so it is the one render path that sees imagery -- deliberately, and only for
live display.
"""

from __future__ import annotations

import cv2
import numpy as np

from ahfd.capture.base import Frame
from ahfd.types import PoseFrame

# Loudest possible red for a confirmed fall, in BGR.
_ALERT_BGR = (0, 0, 255)


def render_overlay(
    frame: Frame,
    pose: PoseFrame,
    min_keypoint_score: float = 0.3,
    states: dict[int, str] | None = None,
    alert: str | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Return a copy of the frame's BGR with skeletons and status drawn on it."""
    from ahfd.viz.skeleton_render import draw_people

    if frame.bgr is None:
        # No colour (e.g. a depth-only source): fall back to the black canvas so
        # the caller still gets something to show rather than a crash.
        from ahfd.viz.skeleton_render import render_skeleton

        return render_skeleton(pose, min_keypoint_score, states=states, fps=fps)

    canvas = frame.bgr.copy()
    draw_people(
        canvas,
        pose,
        min_keypoint_score=min_keypoint_score,
        show_ids=True,
        show_bbox=False,
        states=states,
    )

    if fps is not None:
        cv2.putText(
            canvas,
            format(fps, ".1f") + " fps",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if alert:
        # A banner across the top on a confirmed fall -- the thing a nurse
        # glancing at the screen must not miss.
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), _ALERT_BGR, -1)
        cv2.putText(
            canvas,
            alert,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas
