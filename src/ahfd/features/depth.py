"""Attach sparse, per-joint depth heights to a pose frame.

The returned object still contains only coordinates, confidences and seventeen
height scalars per person.  Dense depth and RGB stay on the capture ``Frame``
and are never retained by this helper.
"""

from __future__ import annotations

import numpy as np

from ahfd.geometry.depth_height import keypoint_heights_from_depth


def attach_depth_heights(pose, frame, ground, *, min_score: float = 0.4):
    """Return ``pose`` with sparse joint heights when measurement depth exists."""
    if frame.depth_raw is None or frame.intrinsics is None or not pose.people:
        return pose
    depth_m = frame.depth_raw.astype(np.float32) * float(frame.depth_scale)
    return pose.with_people(
        tuple(
            person.with_heights(
                keypoint_heights_from_depth(
                    person.keypoints,
                    person.scores,
                    depth_m,
                    frame.intrinsics,
                    ground,
                    min_score=min_score,
                )
            )
            for person in pose.people
        )
    )
