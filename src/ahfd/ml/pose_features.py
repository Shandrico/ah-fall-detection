"""Rich per-frame features derived from the raw COCO-17 joints.

The rule-based classifier decides posture from a few aggregates -- torso
height and floor-spread. That is deliberate and interpretable, but it throws
away most of what the skeleton knows: a *sitting* person is not just "torso a
bit lower", they have **bent knees and a bent hip** while their torso stays
roughly vertical, which is exactly what separates them from someone lying on
the floor at the same torso height. This module extracts those joint-level
signals so the learned classifier can use the whole skeleton, and so we can
*measure* which joints actually distinguish the postures.

Everything here obeys the same rule the metric features do: **range- and
position-invariant**. Two kinds of quantity qualify, and nothing else is
allowed in:

* **Metric heights** (metres above the floor), recovered per joint through the
  calibrated ground plane. A metre is a metre at the near bed and the far one.
* **Angles and ratios**, which are dimensionless -- a bent knee is ~90 degrees
  whether the person is 2 m or 6 m away, and a tall-narrow bounding box stays
  tall-narrow at any scale.

Deliberately absent: any absolute pixel coordinate or absolute pixel size.
Those encode *where the person stood* and *how far away they were*, which a
model on single-session data will happily memorise -- the same location leak
that made `range_m` read as 65% important before it was removed. Angles and
ratios cannot leak it because they do not carry it.
"""

from __future__ import annotations

import math

import numpy as np

from ahfd.pose.skeleton import ANKLES, HEAD, HIPS, KNEES, SHOULDERS

# Groups the base skeleton module does not name, needed for the arm heights.
ELBOWS = (7, 8)
WRISTS = (9, 10)

# Left/right limb chains for the joint angles, as (proximal, joint, distal).
# The angle is measured *at* the middle joint.
_KNEE_CHAINS = ((11, 13, 15), (12, 14, 16))  # hip  -> knee -> ankle
_HIP_CHAINS = ((5, 11, 13), (6, 12, 14))  # shoulder -> hip -> knee

# The feature names this module produces, grouped by what they capture. The
# order is the column order everything downstream relies on.
JOINT_FEATURES: list[str] = [
    # per-joint metric heights (metres above floor)
    "h_head",
    "h_shoulder",
    "h_hip",
    "h_knee",
    "h_ankle",
    "h_elbow",
    "h_wrist",
    # vertical body structure -- metric gaps between joint groups. These shrink
    # or invert as a body folds up or lies down.
    "head_above_hip",
    "shoulder_above_hip",
    "hip_above_knee",
    "knee_above_ankle",
    # joint angles (degrees) -- the "use more than torso" signal
    "knee_angle",  # ~180 standing straight, ~90 sitting
    "hip_angle",  # ~180 standing, bent when seated
    "torso_tilt",  # 0 = torso vertical (up/sit), 90 = torso horizontal (lying)
    # body-shape ratio (dimensionless)
    "bbox_aspect",  # keypoint-box height/width: tall>1 upright, wide<1 lying
]

_EPS = 1e-6


def _group_height(person, ground, contact, valid, indices) -> float | None:
    """Mean floor-referenced height of a joint group, or None if none visible."""
    heights = []
    for i in indices:
        if not valid[i]:
            continue
        h = ground.joint_height(
            float(person.keypoints[i, 0]), float(person.keypoints[i, 1]), contact
        )
        if h is not None:
            heights.append(h)
    return float(np.mean(heights)) if heights else None


def _mid(person, valid, indices) -> np.ndarray | None:
    """Mean pixel position of the visible joints in a group, or None."""
    pts = [person.keypoints[i] for i in indices if valid[i]]
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=float), axis=0)


def _angle_at(person, valid, chain) -> float | None:
    """Interior angle in degrees at the middle joint of a (a, b, c) chain.

    Computed in pixel space, which is fine because an angle is scale-free: the
    triangle is similar at any range. None if any of the three joints is not
    confident or the limb is degenerately short.
    """
    a_i, b_i, c_i = chain
    if not (valid[a_i] and valid[b_i] and valid[c_i]):
        return None
    b = person.keypoints[b_i].astype(float)
    v1 = person.keypoints[a_i].astype(float) - b
    v2 = person.keypoints[c_i].astype(float) - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _paired_angle(person, valid, chains) -> float | None:
    """Average of the left/right versions of a joint angle that are available."""
    vals = [a for a in (_angle_at(person, valid, c) for c in chains) if a is not None]
    return float(np.mean(vals)) if vals else None


def _torso_tilt(shoulder_mid, hip_mid) -> float | None:
    """Angle of the hip->shoulder vector away from image-vertical, in degrees.

    0 means the torso points straight up the image (standing or sitting); 90
    means it lies flat (a fallen or in-bed body). Image-space is acceptable
    here for the same reason angles are: it is a direction, not a length. It
    does depend on which way the person faces relative to the camera, but that
    is a feature the model may legitimately weigh, not a location leak.
    """
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]  # image y grows downward
    n = math.hypot(dx, dy)
    if n < _EPS:
        return None
    # Image "up" is (0, -1); the cosine to it is (-dy)/n.
    return math.degrees(math.acos(float(np.clip(-dy / n, -1.0, 1.0))))


def _bbox_aspect(person, valid) -> float | None:
    """Height/width of the confident-keypoint box. Tall (>1) upright, wide (<1) lying."""
    pts = person.keypoints[valid]
    if len(pts) < 2:
        return None
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    if w < _EPS:
        return None
    return h / w


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def joint_features(person, ground, contact, min_score: float) -> dict[str, float | None]:
    """Extended range-invariant features for one person, keyed by ``JOINT_FEATURES``.

    ``contact`` is the person's floor position (``Features.contact_xy``), reused
    so per-joint heights match the aggregate metric features exactly. Any
    feature whose joints are not confident enough comes back as None, to be
    imputed downstream rather than faked.
    """
    valid = person.valid_mask(min_score)

    h_head = _group_height(person, ground, contact, valid, HEAD)
    h_shoulder = _group_height(person, ground, contact, valid, SHOULDERS)
    h_hip = _group_height(person, ground, contact, valid, HIPS)
    h_knee = _group_height(person, ground, contact, valid, KNEES)
    h_ankle = _group_height(person, ground, contact, valid, ANKLES)
    h_elbow = _group_height(person, ground, contact, valid, ELBOWS)
    h_wrist = _group_height(person, ground, contact, valid, WRISTS)

    return {
        "h_head": h_head,
        "h_shoulder": h_shoulder,
        "h_hip": h_hip,
        "h_knee": h_knee,
        "h_ankle": h_ankle,
        "h_elbow": h_elbow,
        "h_wrist": h_wrist,
        "head_above_hip": _diff(h_head, h_hip),
        "shoulder_above_hip": _diff(h_shoulder, h_hip),
        "hip_above_knee": _diff(h_hip, h_knee),
        "knee_above_ankle": _diff(h_knee, h_ankle),
        "knee_angle": _paired_angle(person, valid, _KNEE_CHAINS),
        "hip_angle": _paired_angle(person, valid, _HIP_CHAINS),
        "torso_tilt": _torso_tilt(
            _mid(person, valid, SHOULDERS), _mid(person, valid, HIPS)
        ),
        "bbox_aspect": _bbox_aspect(person, valid),
    }
