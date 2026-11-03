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


def draw_people(
    canvas: np.ndarray,
    pose: PoseFrame,
    min_keypoint_score: float = 0.3,
    show_ids: bool = True,
    show_bbox: bool = False,
    states: dict[int, str] | None = None,
    joint_radius: int = 3,
    bone_thickness: int = 2,
) -> None:
    """Draw skeletons onto an existing canvas, in place.

    Shared by the black-background view and the on-video overlay, so both draw
    identically. `states`, if given, labels each track with its fall-machine
    state (UPRIGHT / IN_BED / ON_GROUND ...), which is what makes the overlay
    readable to a person watching.
    """
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
            _draw_label(canvas, person.track_id, states.get(person.track_id) if states else None,
                        (int(box[0]), int(box[1])), color)


# Flags that get their own coloured line on the label (BGR).
_FLAG_COLOURS = {"POSSIBLE FALL": (0, 140, 255), "GETTING UP": (0, 215, 255)}


def _draw_label(canvas, track_id, state, top_left, color) -> None:
    """Readable label above a person: rule state large, model labels and flags below.

    ``state`` is the runner's "RULE | model: a, b | FLAG" string. Text is sized to
    the frame so it stays legible after the browser stream is shrunk to 960 px.
    """
    parts = [p.strip() for p in (state or "").split("|") if p.strip()]
    rule = parts[0] if parts else ""
    model = next((p for p in parts[1:] if p.startswith("model:")), None)
    flags = [p for p in parts[1:] if p in _FLAG_COLOURS]
    s = canvas.shape[0] / 1080.0
    big, small = 1.05 * s, 0.8 * s
    thick_big, thick_small = max(1, round(2.4 * s)), max(1, round(1.8 * s))
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [(f"#{track_id}  {rule}".strip(), big, thick_big, (255, 255, 255))]
    if model:
        lines.append(("model: " + model[len("model:"):].strip().replace(", ", " / "), small, thick_small, (210, 210, 210)))
    lines += [(f, small, thick_small, _FLAG_COLOURS[f]) for f in flags]
    sizes = [cv2.getTextSize(t, font, sc, th)[0] for t, sc, th, _ in lines]
    pad, gap = round(8 * s), round(8 * s)
    w = max(tw for tw, _ in sizes) + 2 * pad
    h = sum(th for _, th in sizes) + gap * (len(lines) - 1) + 2 * pad
    x0 = max(0, min(top_left[0], canvas.shape[1] - w))
    y0 = max(0, top_left[1] - h - round(6 * s))
    roi = canvas[y0:y0 + h, x0:x0 + w]
    if roi.size:
        roi[:] = (roi * 0.35).astype(roi.dtype)  # dark see-through box
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), color, max(1, round(2 * s)))
    y = y0 + pad
    for (text, sc, th, col), (_tw, tht) in zip(lines, sizes):
        y += tht
        cv2.putText(canvas, text, (x0 + pad, y), font, sc, col, th, cv2.LINE_AA)
        y += gap


def draw_metrics(
    canvas: np.ndarray,
    pose: PoseFrame,
    metrics: dict[int, dict],
    min_keypoint_score: float = 0.3,
) -> None:
    """Draw the per-person metric readout, and a corner calibration check.

    This is the development/tuning view: it shows the actual numbers the
    decision runs on -- height in metres, floor-spread, vertical velocity --
    so a bad calibration or a mis-tracked pose is visible rather than hidden
    behind a plausible-looking skeleton.

    The corner line is the calibration sanity signal: a standing person's
    ankles should read ~0.05 m, so if that drifts the mount has moved and every
    metric is quietly wrong.
    """
    from ahfd.pose.skeleton import track_color

    for person in pose.people:
        m = metrics.get(person.track_id) if person.track_id is not None else None
        if m is None:
            continue
        box = person.bbox(min_keypoint_score)
        if box is None:
            continue

        def fmt(key, unit=""):
            v = m.get(key)
            return "-" if v is None else (format(v, ".2f") + unit)

        lines = [
            "h " + fmt("h_torso", "m") + "  spread " + fmt("floor_spread", "m"),
            "vz " + fmt("v_z") + "  ankle " + fmt("h_ankle_min", "m"),
        ]
        if m.get("bed_risk"):
            lines.append("bed risk " + str(m["bed_risk"]))

        color = track_color(person.track_id)
        x = int(box[2]) + 6  # to the right of the person
        y0 = max(24, int(box[1]))
        for i, line in enumerate(lines):
            cv2.putText(
                canvas, line, (x, y0 + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

    # Calibration sanity: min ankle height across tracked people.
    ankles = [
        m["h_ankle_min"]
        for m in metrics.values()
        if m.get("h_ankle_min") is not None
    ]
    if ankles:
        from ahfd.geometry.calibration import drift_check

        lowest = min(ankles)
        ok = drift_check([lowest])
        text = "ankle " + format(lowest, ".2f") + "m " + ("ok" if ok else "CALIB?")
        cv2.putText(
            canvas, text, (8, canvas.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (150, 220, 150) if ok else (60, 60, 230), 1, cv2.LINE_AA,
        )


def render_skeleton(
    pose: PoseFrame,
    min_keypoint_score: float = 0.3,
    show_ids: bool = True,
    show_bbox: bool = False,
    fps: float | None = None,
    states: dict[int, str] | None = None,
    metrics: dict[int, dict] | None = None,
    joint_radius: int = 3,
    bone_thickness: int = 2,
) -> np.ndarray:
    """Draw a PoseFrame onto a fresh black canvas.

    Takes a PoseFrame, not a Frame: this function cannot draw over video even
    if someone later wants it to, because it never receives any.
    """
    canvas = np.zeros((pose.height, pose.width, 3), dtype=np.uint8)
    draw_people(
        canvas,
        pose,
        min_keypoint_score=min_keypoint_score,
        show_ids=show_ids,
        show_bbox=show_bbox,
        states=states,
        joint_radius=joint_radius,
        bone_thickness=bone_thickness,
    )
    if metrics:
        draw_metrics(canvas, pose, metrics, min_keypoint_score)
    _draw_hud(canvas, pose, fps)
    return canvas


def _draw_hud(canvas: np.ndarray, pose: PoseFrame, fps: float | None) -> None:
    """Corner readout: enough to tell at a glance that the pipeline is alive."""
    lines = ["t " + format(pose.t, ".1f") + "s", "people " + str(len(pose.people))]
    if fps is not None:
        lines.insert(0, format(fps, ".1f") + " fps")

    sc = canvas.shape[0] / 1080.0
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (round(12 * sc), round(34 * sc) + i * round(32 * sc)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * sc,
            (220, 220, 220),
            max(1, round(2 * sc)),
            cv2.LINE_AA,
        )

    # A visible reminder of what this window is, for anyone who walks past it.
    label = "SKELETON ONLY -- NO VIDEO RETAINED"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75 * sc, max(1, round(2 * sc)))
    cv2.putText(
        canvas,
        label,
        (max(8, canvas.shape[1] - tw - round(12 * sc)), canvas.shape[0] - round(16 * sc)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * sc,
        (130, 130, 130),
        max(1, round(2 * sc)),
        cv2.LINE_AA,
    )
