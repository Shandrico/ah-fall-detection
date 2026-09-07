"""COCO-17 skeleton definition.

Every pose backend in the bake-off (RTMO, YOLO-pose, RTMPose) emits these same
17 joints in this same order, which is exactly why COCO-17 is the interface
currency: swapping the model does not change anything downstream.

Named indices matter for the fall features later on -- height thresholds are
expressed per joint group (head / trunk / ankles), not on an anonymous array.
"""

from __future__ import annotations

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NAME_TO_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Joint groups used by the fall features.
HEAD = (0, 1, 2, 3, 4)
SHOULDERS = (5, 6)
HIPS = (11, 12)
KNEES = (13, 14)
ANKLES = (15, 16)
TRUNK = SHOULDERS + HIPS

# Bones, as index pairs. Order is legs, pelvis, torso, arms, face.
EDGES: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

# Distinct, colour-blind-safe track colours in BGR (OpenCV order). Cycled by
# track id so a given person keeps one colour for the life of the track.
TRACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (180, 119, 31),
    (14, 127, 255),
    (44, 160, 44),
    (40, 39, 214),
    (189, 103, 148),
    (75, 86, 140),
    (194, 119, 227),
    (207, 190, 23),
)


def track_color(track_id: int | None) -> tuple[int, int, int]:
    """Stable colour for a track id; grey when the person is untracked."""
    if track_id is None:
        return (160, 160, 160)
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]
