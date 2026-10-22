"""Scrub-and-mark segment labeller (postures, or bed-exit phases).

A small video viewer (OpenCV window) for turning a staged clip into the time
segments a classifier trains on. You scrub the clip, mark a START, scrub to the
END, and pick a class -- one segment written per hold. Transitions are simply
the gaps you leave between segments, which is what keeps the labels clean (see
``ahfd.ml.posture``).

It is the twin of ``ahfd calibrate-zones``: read a frame, let a human point at
what matters, write coordinates -- here time-codes, there floor metres. Like
that tool it **never saves imagery**; only the label JSON is written, so the
privacy guarantee holds (tests/test_privacy.py).

The labeller is parameterised by a :class:`LabelScheme` -- the set of classes,
the JSON field they are written under, and their timeline colours. Two schemes
ship: the 4-class :data:`POSTURE_SCHEME` (upright/sitting/in_bed/on_ground,
field ``posture``) and the 7-phase :data:`PHASE_SCHEME` (the bed-exit activity
phases, field ``phase``). ``run_labeler`` defaults to the posture scheme so
every existing caller is unchanged.

The heavy interactive loop is ``run_labeler``; the pure pieces it relies on
(reading any existing labels, formatting the output) are split out so they can
be unit-tested without a display.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Posture classes, in the order their number keys are shown (1..4). These match
# the system's own posture states, which is what makes the labels trainable.
POSTURE_CLASSES: tuple[str, ...] = ("upright", "sitting", "in_bed", "on_ground")

# Display colour per posture, BGR (OpenCV), for the timeline ribbon.
_POSTURE_BGR: dict[str, tuple[int, int, int]] = {
    "upright": (60, 180, 60),  # green
    "sitting": (40, 200, 220),  # yellow
    "in_bed": (220, 140, 40),  # blue
    "on_ground": (40, 40, 220),  # red
}

# Bed-exit activity phases, keys 1..7 -- the same names and order the live
# `collect` observer uses (cli.py) and the bed-exit state machine emits
# (ahfd.detect.bed_exit.BedActivityPhase). Marked OFFLINE here against a
# recorded clip; UNKNOWN is simply a gap left unlabelled, as with postures.
PHASE_CLASSES: tuple[str, ...] = (
    "RECLINED",
    "TORSO_RISING",
    "UPRIGHT_IN_BED",
    "SHIFTING_TO_EDGE",
    "EDGE_SITTING",
    "ATTEMPTING_STAND",
    "OUT_OF_BED",
)

# Ribbon colours, reclined (blue) -> out of bed (red), so the timeline reads as
# a left-to-right progression out of the bed.
_PHASE_BGR: dict[str, tuple[int, int, int]] = {
    "RECLINED": (220, 140, 40),  # blue
    "TORSO_RISING": (210, 180, 60),  # teal
    "UPRIGHT_IN_BED": (120, 200, 60),  # green
    "SHIFTING_TO_EDGE": (40, 210, 200),  # yellow
    "EDGE_SITTING": (40, 170, 230),  # orange
    "ATTEMPTING_STAND": (40, 110, 240),  # deep orange
    "OUT_OF_BED": (40, 40, 220),  # red
}


@dataclass(frozen=True)
class LabelScheme:
    """What the labeller marks: the classes, their JSON field, their colours.

    ``name`` captions the window; ``classes`` are shown on number keys 1..N;
    ``key`` is the per-segment JSON field (``posture`` or ``phase``); ``colours``
    maps each class to a BGR timeline colour.
    """

    name: str
    classes: tuple[str, ...]
    key: str
    colours: dict[str, tuple[int, int, int]]


POSTURE_SCHEME = LabelScheme("postures", POSTURE_CLASSES, "posture", _POSTURE_BGR)
PHASE_SCHEME = LabelScheme("phases", PHASE_CLASSES, "phase", _PHASE_BGR)


def load_existing_segments(
    path: Path, *, scheme: LabelScheme = POSTURE_SCHEME
) -> tuple[str | None, list[dict]]:
    """Return (clip_id, real_segments) from an existing label file, if any.

    Placeholder rows (``end_s <= start_s``) are dropped, so re-opening a file
    that still holds the generated template starts you from a clean slate but
    keeps any real segments you had already marked. Segments whose class is not
    in ``scheme.classes`` are dropped too (a stale/foreign label). The segments
    are returned under ``scheme.key`` -- the same field the labeller writes.
    """
    if not path.exists():
        return None, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, []
    key = scheme.key
    segs = [
        {
            "start_s": round(float(s["start_s"]), 2),
            "end_s": round(float(s["end_s"]), 2),
            key: str(s[key]),
        }
        for s in data.get("segments", [])
        if key in s
        and float(s.get("end_s", 0)) > float(s.get("start_s", 0))
        and s.get(key) in scheme.classes
    ]
    return data.get("clip_id"), segs


def dump_posture_json(
    clip_id: str, duration_s: float, segments: list[dict], *, label_key: str = "posture"
) -> str:
    """Format labels as the project's compact one-segment-per-line JSON.

    Segments are sorted by start time so the file reads in clip order regardless
    of the sequence they were marked in. ``label_key`` is the per-segment field
    (``posture`` by default, ``phase`` for the bed-exit phase scheme).
    """
    ordered = sorted(segments, key=lambda s: s["start_s"])
    lines = [
        '    { "start_s": %s, "end_s": %s, "%s": "%s" }'
        % (_num(s["start_s"]), _num(s["end_s"]), label_key, s[label_key])
        for s in ordered
    ]
    body = ",\n".join(lines)
    return (
        "{\n"
        '  "clip_id": "' + clip_id + '",\n'
        '  "duration_s": ' + _num(duration_s) + ",\n"
        '  "segments": [\n'
        + (body + "\n" if body else "")
        + "  ]\n"
        "}\n"
    )


def _num(x: float) -> str:
    """Trim a float to a tidy string: 18.0 not 18.00000001, 32.5 kept."""
    return format(round(float(x), 2), "g")


# --------------------------------------------------------------------------
# Interactive loop. Imports OpenCV lazily so importing this module (and the
# pure helpers above) never requires a GUI build.
# --------------------------------------------------------------------------

# Windows waitKeyEx codes for the arrow keys, so they can drive scrubbing too.
_ARROWS = {2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down"}

# Timeline ribbon geometry, shared by the drawer and the click-to-seek handler
# so a click lands on exactly the bar that is drawn.
_BAR_MARGIN = 12
_BAR_H = 14
_BAR_BOTTOM = 28  # ribbon top sits at (display height - _BAR_BOTTOM)


def _help_lines(scheme: LabelScheme) -> tuple[str, ...]:
    return (
        "click / drag the bar below to seek     a <- -1s   d -> +1s   , . 1frame   [ ] 5s",
        "s / SPACE mark START     f mark END (then 1-%d)     c cancel mark     r remove seg here"
        % len(scheme.classes),
        "0 start   g end     u undo last     w save     q save+quit",
    )


def run_labeler(
    clip_path: Path,
    out_path: Path,
    *,
    scheme: LabelScheme = POSTURE_SCHEME,
    max_display_width: int = 1280,
) -> int:
    """Open ``clip_path`` in a scrubber and write ``scheme`` segments to ``out_path``.

    Returns the number of segments saved. Requires a display (run it on the
    laptop, not the headless Jetson). Any segments already in ``out_path`` (for
    this scheme) are loaded for editing -- which is how a script-seeded draft is
    corrected rather than marked from scratch.
    """
    import cv2

    key_field = scheme.key

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError("could not open video: " + str(clip_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames <= 0:
        cap.release()
        raise RuntimeError("video reports no frames: " + str(clip_path))
    duration_s = n_frames / fps

    clip_id, segments = load_existing_segments(out_path, scheme=scheme)
    clip_id = clip_id or clip_path.stem

    step = max(1, int(round(fps)))  # frames in one second
    cur = 0
    need_read = True
    frame = None
    scale = 1.0

    start_mark: float | None = None
    pending: tuple[float, float] | None = None  # (start, end) awaiting a class
    status = "loaded " + str(len(segments)) + " existing segment(s)"

    win = "ahfd label-" + scheme.name + "  [" + clip_id + "]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # Click or drag on the timeline ribbon to seek there, like a normal video
    # scrubber. The callback runs during waitKey; it only records the requested
    # frame, and the loop applies it -- no shared-state gymnastics with `cur`.
    ui: dict = {"geom": None, "seek_frame": None}

    def on_mouse(event, x, y, flags, param):
        geom = ui["geom"]
        if geom is None:
            return
        bar_top, bar_bot, left, right = geom
        pressed = event == cv2.EVENT_LBUTTONDOWN
        dragging = event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)
        if (pressed or dragging) and bar_top - 10 <= y <= bar_bot + 10:
            frac = min(1.0, max(0.0, (x - left) / float(max(1, right - left))))
            ui["seek_frame"] = int(round(frac * (n_frames - 1)))

    cv2.setMouseCallback(win, on_mouse)

    def t_now() -> float:
        return round(cur / fps, 2)

    def seek(delta_frames: int) -> None:
        nonlocal cur, need_read
        cur = min(max(0, cur + delta_frames), n_frames - 1)
        need_read = True

    help_lines = _help_lines(scheme)

    while True:
        if need_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, img = cap.read()
            if ok and img is not None:
                frame = img
                h, w = frame.shape[:2]
                scale = min(1.0, max_display_width / float(w))
            need_read = False
        if frame is None:
            break

        disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        dh, dw = disp.shape[:2]
        ui["geom"] = (dh - _BAR_BOTTOM, dh - _BAR_BOTTOM + _BAR_H, _BAR_MARGIN, dw - _BAR_MARGIN)
        _draw_overlay(
            cv2, disp, t_now(), duration_s, cur, n_frames, start_mark,
            pending, segments, status, scheme, help_lines,
        )
        cv2.imshow(win, disp)

        code = cv2.waitKeyEx(20)
        if ui["seek_frame"] is not None:  # a click/drag landed on the ribbon
            cur = min(max(0, ui["seek_frame"]), n_frames - 1)
            need_read = True
            ui["seek_frame"] = None
        if code == -1:
            continue
        arrow = _ARROWS.get(code)
        key = code & 0xFF

        # --- pick a class while an end is pending ---------------------------
        if pending is not None:
            if ord("1") <= key <= ord("0") + len(scheme.classes):
                cls = scheme.classes[key - ord("1")]
                segments.append(
                    {"start_s": pending[0], "end_s": pending[1], key_field: cls}
                )
                status = "added %s  %.2f-%.2fs" % (cls, pending[0], pending[1])
                pending = None
                start_mark = None
            elif key in (ord("c"), 27):  # cancel the pending end
                pending = None
                status = "cancelled"
            continue

        # --- navigation -----------------------------------------------------
        if arrow == "right" or key == ord("d"):
            seek(step)
        elif arrow == "left" or key == ord("a"):
            seek(-step)
        elif arrow == "up" or key == ord("]"):
            seek(5 * step)
        elif arrow == "down" or key == ord("["):
            seek(-5 * step)
        elif key == ord("."):
            seek(1)
        elif key == ord(","):
            seek(-1)
        elif key == ord("0"):
            cur, need_read = 0, True
        elif key == ord("g"):
            cur, need_read = n_frames - 1, True
        # --- marking --------------------------------------------------------
        elif key in (ord("s"), ord(" ")):
            start_mark = t_now()
            status = "START @ %.2fs -- scrub to the end, then press f" % start_mark
        elif key == ord("f"):
            if start_mark is None:
                status = "set a START first (s)"
            elif t_now() <= start_mark:
                status = "END must be after START (%.2fs)" % start_mark
            else:
                pending = (start_mark, t_now())
                status = "pick: " + _class_menu(scheme)
        elif key == ord("u"):
            if segments:
                gone = segments.pop()
                status = "undo %s %.2f-%.2fs" % (
                    gone[key_field], gone["start_s"], gone["end_s"]
                )
            else:
                status = "nothing to undo"
        elif key == ord("c"):
            if start_mark is not None:
                start_mark = None
                status = "cleared START -- mark a new one"
            else:
                status = "no START to cancel"
        elif key == ord("r"):
            t = t_now()
            hit = next(
                (i for i, s in enumerate(segments) if s["start_s"] <= t <= s["end_s"]),
                None,
            )
            if hit is not None:
                gone = segments.pop(hit)
                status = "removed %s %.2f-%.2fs" % (
                    gone[key_field], gone["start_s"], gone["end_s"]
                )
            else:
                status = "no segment at %.2fs (scrub onto one to remove it)" % t
        elif key == ord("w"):
            _write(out_path, clip_id, duration_s, segments, scheme)
            status = "saved %d segment(s) -> %s" % (len(segments), out_path.name)
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    _write(out_path, clip_id, duration_s, segments, scheme)
    return len(segments)


def _class_menu(scheme: LabelScheme) -> str:
    return "   ".join(
        str(i + 1) + "=" + name for i, name in enumerate(scheme.classes)
    ) + "   c=cancel"


def _write(
    out_path: Path, clip_id: str, duration_s: float, segments: list[dict],
    scheme: LabelScheme = POSTURE_SCHEME,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        dump_posture_json(clip_id, duration_s, segments, label_key=scheme.key),
        encoding="utf-8",
    )


def _draw_overlay(
    cv2, disp, t, duration, cur, n_frames, start_mark, pending, segments, status,
    scheme: LabelScheme = POSTURE_SCHEME, help_lines: tuple[str, ...] | None = None,
) -> None:
    """Draw the HUD and the timeline ribbon onto the display frame (in memory)."""
    h, w = disp.shape[:2]
    yellow, white, green = (0, 255, 255), (255, 255, 255), (80, 255, 80)
    key_field = scheme.key
    if help_lines is None:
        help_lines = _help_lines(scheme)

    def text(s, x, y, color=white, scale=0.6, thick=2):
        cv2.putText(disp, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(disp, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    text("t=%6.2fs / %.1fs   frame %d/%d   segs:%d" % (t, duration, cur, n_frames - 1, len(segments)), 12, 28, yellow, 0.7)
    for i, line in enumerate(help_lines):
        text(line, 12, 54 + i * 22, white, 0.5, 1)

    if start_mark is not None:
        text("START @ %.2fs" % start_mark, 12, 130, green, 0.65)
    if pending is not None:
        text("PICK  %.2f-%.2fs :  %s" % (pending[0], pending[1], _class_menu(scheme)), 12, 158, yellow, 0.55)
    if status:
        text(status, 12, h - 46, (180, 220, 255), 0.55, 1)

    # Timeline ribbon along the bottom: labelled spans in colour, a cursor line.
    margin, bar_y, bar_h = _BAR_MARGIN, h - _BAR_BOTTOM, _BAR_H

    def x_of(sec):
        return int(margin + (w - 2 * margin) * (sec / duration if duration else 0))

    cv2.rectangle(disp, (margin, bar_y), (w - margin, bar_y + bar_h), (60, 60, 60), -1)
    for s in segments:
        cv2.rectangle(
            disp, (x_of(s["start_s"]), bar_y), (x_of(s["end_s"]), bar_y + bar_h),
            scheme.colours.get(s[key_field], (200, 200, 200)), -1,
        )
    if start_mark is not None:
        cv2.line(disp, (x_of(start_mark), bar_y - 4), (x_of(start_mark), bar_y + bar_h + 4), green, 2)
    xc = x_of(t)
    cv2.line(disp, (xc, bar_y - 6), (xc, bar_y + bar_h + 6), white, 2)
