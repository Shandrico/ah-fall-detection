"""Skeleton rendering on a black background.

This is the demo view, and the choice of a black background is an argument
rather than an aesthetic. What is drawn here is exactly what the system
retains -- joint coordinates and nothing else. Showing a stakeholder the
skeleton next to an empty background makes the privacy property visible in a
way no amount of documentation does: "this is everything we keep."

It also means the demo can be screen-shared or photographed without exposing
anybody, which matters in a real ward.
"""

from __future__ import annotations

import cv2
import numpy as np

from ahfd.pose.skeleton import EDGES, track_color
from ahfd.types import PoseFrame


def render_skeleton(
    pose: PoseFrame,
    min_keypoint_score: float = 0.3,
    show_ids: bool = True,
    show_bbox: bool = False,
    fps: float | None = None,
    joint_radius: int = 3,
    bone_thickness: int = 2,
) -> np.ndarray:
    """Draw a PoseFrame onto a fresh black canvas.

    Takes a PoseFrame, not a Frame: this function cannot draw over video even
    if someone later wants it to, because it never receives any.
    """
    canvas = np.zeros((pose.height, pose.width, 3), dtype=np.uint8)

    for person in pose.people:
        color = track_color(person.track_id)
        valid = person.valid_mask(min_keypoint_score)
        pts = person.keypoints

        for a, b in EDGES:
            if valid[a] and valid[b]:
                cv2.line(
                    canvas,
                    (int(pts[a, 0]), int(pts[a, 1])),
                    (int(pts[b, 0]), int(pts[b, 1])),
                    color,
                    bone_thickness,
                    lineType=cv2.LINE_AA,
                )

        for i in range(pts.shape[0]):
            if valid[i]:
                cv2.circle(
                    canvas,
                    (int(pts[i, 0]), int(pts[i, 1])),
                    joint_radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        box = person.bbox(min_keypoint_score)
        if box is None:
            continue

        if show_bbox:
            cv2.rectangle(
                canvas,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                1,
            )

        if show_ids and person.track_id is not None:
            cv2.putText(
                canvas,
                "id " + str(person.track_id),
                (int(box[0]), max(12, int(box[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    _draw_hud(canvas, pose, fps)
    return canvas


def _draw_hud(canvas: np.ndarray, pose: PoseFrame, fps: float | None) -> None:
    """Corner readout: enough to tell at a glance that the pipeline is alive."""
    lines = ["t " + format(pose.t, ".1f") + "s", "people " + str(len(pose.people))]
    if fps is not None:
        lines.insert(0, format(fps, ".1f") + " fps")

    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 18 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # A visible reminder of what this window is, for anyone who walks past it.
    label = "SKELETON ONLY -- NO VIDEO RETAINED"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(
        canvas,
        label,
        (max(8, canvas.shape[1] - tw - 8), canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
