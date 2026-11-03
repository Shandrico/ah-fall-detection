"""Annotated review video: a recording replayed through the live pipeline, saved.

The dashboard computes pose and posture as the clip plays, so on a busy or
remote link it stutters. This renders the same overlay -- skeleton, track id,
posture state, bed phase -- for EVERY frame of a consented staged recording and
encodes it once as 1080p H.264, which any browser then plays smoothly.

It writes identifiable RGB, so it lives in ``ahfd.debug`` and refuses unless all
three raw-capture switches are on (see ``ahfd.privacy``). The video is written
next to the recordings (approved storage), never into the repository.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ahfd.privacy import require_raw_capture


def export_review(
    recording: Path,
    out: Path,
    *,
    cfg,
    calibration: Path,
    config_flag: bool = False,
    cli_flag: bool = False,
    crf: int = 23,
    progress=print,
) -> int:
    """Render ``recording`` (.db3/.bag) with the dashboard overlay to ``out`` (.mp4).

    Returns the number of frames written. Every recorded colour frame is
    processed in order (no skipping), so the states match an offline replay of
    the live pipeline. Writes ``<out>.partial.mp4`` first and renames it only
    when complete.
    """
    require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)
    import time

    import cv2

    from ahfd.capture.realsense import BagSource
    from ahfd.cli import _build_detection
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.track import SimpleTracker
    from ahfd.viz import render_overlay

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to encode H.264 (browsers cannot play OpenCV's mp4v)")
    out = Path(out)
    if out.exists():
        raise FileExistsError("refusing to overwrite " + str(out))
    partial = out.with_name(out.stem + ".partial.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 2_000_000_000:
        raise RuntimeError("less than 2 GB free next to the output; nothing written")

    src = BagSource(str(recording))
    meta = src.meta
    extractor, machine, _sink, _calib = _build_detection(cfg, str(calibration), meta)
    estimator = build_estimator(cfg.pose)
    tracker = SimpleTracker(min_keypoint_score=cfg.pose.min_keypoint_score, max_miss_s=1.0)
    smoother = (
        KeypointSmoother(
            min_cutoff=cfg.smoothing.min_cutoff,
            beta=cfg.smoothing.beta,
            d_cutoff=cfg.smoothing.d_cutoff,
        )
        if cfg.smoothing.enabled
        else None
    )
    fps = meta.fps or 30.0
    total = meta.frame_count
    encoder = None
    n = 0
    t_first = None
    started = time.monotonic()
    try:
        for frame in src:
            pose = tracker.update(estimator.estimate(frame))
            if smoother is not None:
                smoother.retain_only(tracker.live_ids)
                pose = pose.with_people(tuple(
                    p.with_keypoints(smoother.smooth(p.track_id, pose.t, p.keypoints))
                    for p in pose.people
                ))
            extractor.retain_only(tracker.live_ids)
            machine.retain_only(tracker.live_ids)
            states, phases = {}, {}
            for person in pose.people:
                if person.track_id is None:
                    continue
                features = extractor.extract(person, pose.t)
                if features is not None:
                    machine.update_all(features)
                states[person.track_id] = machine.state_of(person.track_id)
                bed = machine.bed_snapshot_of(person.track_id, pose.t)
                if bed is not None and bed.phase not in (None, "UNKNOWN"):
                    phases[person.track_id] = bed.phase

            canvas = render_overlay(
                frame, pose, min_keypoint_score=cfg.pose.min_keypoint_score, states=states,
            )
            t_first = frame.t if t_first is None else t_first
            _draw_hud(cv2, canvas, frame.t - t_first, n, phases)

            if encoder is None:
                h, w = canvas.shape[:2]
                encoder = subprocess.Popen(
                    [
                        "ffmpeg", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
                        "-r", "%g" % fps, "-i", "-",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
                    ],
                    stdin=subprocess.PIPE,
                )
            encoder.stdin.write(canvas.tobytes())
            n += 1
            if n % 300 == 0:
                rate = n / max(time.monotonic() - started, 1e-6)
                left = (total - n) / rate if total else float("nan")
                progress("  %d/%s frames  (%.1f fps, ~%.0f s left)" % (n, total or "?", rate, left))
    finally:
        src.close()
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
    if encoder is None or encoder.returncode != 0 or n == 0:
        raise RuntimeError("encoding failed; partial file kept for inspection: " + str(partial))
    partial.rename(out)
    return n


def _draw_hud(cv2, canvas, t_s: float, index: int, phases: dict) -> None:
    """Recording time + frame number (top right) and each track's bed phase."""
    minutes = int(t_s // 60)
    lines = ["%d:%04.1f   frame %d" % (minutes, t_s - 60 * minutes, index)]
    lines += ["id %d  bed phase: %s" % (tid, phase) for tid, phase in sorted(phases.items())]
    w = canvas.shape[1]
    y = 40
    for text in lines:
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = w - tw - 24
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += 40


# Class colours (BGR) for the timeline strip, shared by labels, rules and models.
_PHASE_COLOURS = {
    "RECLINED": (180, 119, 31), "TORSO_RISING": (14, 127, 255), "UPRIGHT_IN_BED": (44, 160, 44),
    "SHIFTING_TO_EDGE": (189, 103, 148), "EDGE_SITTING": (75, 86, 140), "ATTEMPTING_STAND": (40, 39, 214),
    "OUT_OF_BED": (34, 189, 188),
}
_POSTURE_COLOURS = {
    "in_bed": (180, 119, 31), "sitting": (44, 160, 44), "upright": (34, 189, 188), "on_ground": (40, 39, 214),
}
_BANNER_COLOURS = {
    "FALL ALERT": (40, 39, 214), "BED EXIT": (180, 60, 200), "POSSIBLE FALL": (0, 165, 255), "GETTING UP": (0, 230, 230),
}
_ROWS = (  # (title, attribute, colours) top to bottom in the strip; "system" = the trained models
    ("7-phase label", "label_phase", _PHASE_COLOURS),
    ("7-phase system", "model_phase", _PHASE_COLOURS),
    ("4-state label", "label_posture", _POSTURE_COLOURS),
    ("4-state system", "model_posture", _POSTURE_COLOURS),
    ("alerts", "banner", _BANNER_COLOURS),
)


def export_labelled_review(
    recording: Path,
    tracks: Path,
    out: Path,
    *,
    cfg,
    calibration: Path,
    posture_model: Path | None = None,
    phase_model: Path | None = None,
    labels: Path | None = None,
    label_t0: float | None = None,
    posture_labels: Path | None = None,
    falls: Path | None = None,
    reviewed: dict[float, str] | None = None,
    preds_out: list | None = None,
    config_flag: bool = False,
    cli_flag: bool = False,
    crf: int = 23,
    out_fps: float = 15.0,
    progress=print,
) -> tuple[int, dict]:
    """Review video of a clip's stored keypoints with rules, models and labels.

    Pose comes from ``tracks`` (the extracted keypoints the models are scored
    on), not a fresh pose run, so what is drawn is exactly what is evaluated.
    Each frame shows the skeleton, the calibrated bed outlines, a table of the
    4-state and 7-phase answers from labels / rules / model (green = agrees
    with the label, red = not), and a timeline of the whole clip with a cursor.
    Frames are held to their timestamps at ``out_fps``, so a recording with
    dropped frames still plays at real speed. Returns (frames, agreement).
    """
    require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)
    import cv2
    import numpy as np

    from ahfd.capture.realsense import BagSource
    from ahfd.geometry.calibration import load_calibration
    from ahfd.io import read_tracks
    from ahfd.ml.clip_review import predict_clip
    from ahfd.viz import render_overlay

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to encode H.264 (browsers cannot play OpenCV's mp4v)")
    out = Path(out)
    if out.exists():
        raise FileExistsError("refusing to overwrite " + str(out))
    partial = out.with_name(out.stem + ".partial.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 2_000_000_000:
        raise RuntimeError("less than 2 GB free next to the output; nothing written")

    preds = predict_clip(
        tracks, calibration, cfg, posture_model=posture_model, phase_model=phase_model,
        labels=labels, label_t0=label_t0, posture_labels=posture_labels, falls=falls, reviewed=reviewed,
    )
    poses = list(read_tracks(tracks))
    times = np.array([p.t for p in poses])
    calib = load_calibration(calibration)
    duration = max((f.t for f in preds.frames.values()), default=1.0) or 1.0
    strip = _timeline_strip(cv2, np, preds, duration, width=1920)
    beds = _bed_outlines(np, calib)

    src = BagSource(str(recording))
    encoder = None
    n = written = 0
    previous = None
    held = None
    t_first = None
    try:
        for frame in src:
            # Nearest stored row; both carry the colour frame's own timestamp.
            i = int(np.searchsorted(times, frame.t))
            i = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - frame.t))
            if abs(times[i] - frame.t) > 0.005:
                continue
            pose = poses[i]
            t_first = frame.t if t_first is None else t_first
            pred = preds.at(pose.t)
            if pred is None and held is not None and pose.t - held[0] <= 1.0:
                pred = held[1]  # pose dropped out for a moment: keep showing the last answers
            elif pred is not None:
                held = (pose.t, pred)
            # The tag on the skeleton is the system's (trained model's) posture, as in the table.
            states = {pred.track_id: (pred.model_posture or "").upper()} if pred and pred.track_id is not None else {}
            canvas = render_overlay(frame, pose, min_keypoint_score=cfg.pose.min_keypoint_score, states=states)
            for poly in beds:
                cv2.polylines(canvas, [poly], True, (0, 255, 255), 3, cv2.LINE_AA)
            t_rel = pose.t - preds.t0
            _draw_table(cv2, canvas, pred, frame.t - t_first, n)
            if preds.marks or preds.impacts or preds.exits:
                _draw_alert_panel(cv2, canvas, preds, t_rel)
                _draw_truth(cv2, canvas, preds, t_rel)
                _draw_border(cv2, canvas, preds, t_rel)
            else:
                _draw_banner(cv2, canvas, pred)
            canvas = _with_strip(cv2, np, canvas, strip, (frame.t - t_first) / duration)

            if encoder is None:
                h, w = canvas.shape[:2]
                encoder = subprocess.Popen(
                    [
                        "ffmpeg", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
                        "-r", "%g" % out_fps, "-i", "-",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
                    ],
                    stdin=subprocess.PIPE,
                )
            # Hold the previous frame until this one's time: real-speed playback.
            if previous is not None:
                for _ in range(max(1, round((frame.t - t_first) * out_fps) - written)):
                    encoder.stdin.write(previous)
                    written += 1
            previous = canvas.tobytes()
            n += 1
            if n % 200 == 0:
                progress("  %d frames" % n)
        if previous is not None:
            encoder.stdin.write(previous)
    finally:
        src.close()
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
    if encoder is None or encoder.returncode != 0 or n == 0:
        raise RuntimeError("encoding failed; partial file kept for inspection: " + str(partial))
    partial.rename(out)
    if preds_out is not None:
        preds_out.append(preds)
    return n, preds.agreement()


def _bed_outlines(np, calib):
    """Each calibrated bed's outline at its surface height, in image pixels."""
    g = calib.ground
    k = g.intrinsics
    to_cam = g.rotation.T
    polys = []
    for zone in calib.zones.zones:
        if zone.kind != "bed":
            continue
        z = zone.top_m or 0.0
        pts = []
        for x, y in zone.polygon:
            c = to_cam @ np.array([x, y, z - g.height_m])
            if c[2] <= 1e-6:
                break
            pts.append((k.fx * c[0] / c[2] + k.cx, k.fy * c[1] / c[2] + k.cy))
        else:
            polys.append(np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2))
    return polys


# Alert colours (BGR) by event, and verdict colours. One look tells what fired and whether it was right.
_EVENT_STYLE = {
    "FALL_SUSPECTED": ("FALL ALERT", (40, 39, 214)), "FALL_CONFIRMED": ("FALL CONFIRMED", (40, 39, 214)),
    "PERSON_DOWN": ("PERSON DOWN", (40, 39, 214)), "NEAR_MISS": ("PERSON UP AGAIN", (60, 80, 200)),
    "POSSIBLE_FALL": ("POSSIBLE FALL (early)", (0, 140, 255)), "BED_EXIT": ("BED EXIT", (180, 60, 200)),
    "BED_EXIT_WARNING": ("BED-EXIT EARLY WARNING", (0, 215, 255)), "FALL": ("LABELLED FALL", (230, 230, 230)),
}
_VERDICT_COLOURS = {
    "TRUE": (60, 190, 60), "FALSE ALARM": (50, 50, 240), "RISKY": (0, 170, 255), "EXTRA": (0, 170, 255),
    "NO EXIT": (0, 170, 255), "MISSED": (50, 50, 240),
}
_VERDICT_TEXT = {"TRUE": "TRUE", "FALSE ALARM": "FALSE ALARM", "RISKY": "RISKY (reviewed)", "EXTRA": "EXTRA",
                 "NO EXIT": "NO EXIT FOLLOWED", "MISSED": "MISSED"}
# Most serious last: the panel names a group by its most serious alert.
_SEVERITY_ORDER = ("BED_EXIT_WARNING", "POSSIBLE_FALL", "NEAR_MISS", "BED_EXIT", "FALL_SUSPECTED", "PERSON_DOWN",
                   "FALL_CONFIRMED")
SHOW_ALERT_S = 6.0  # how long an alert stays in the panel
FLASH_S = 2.0  # how long the frame border flashes after an alert


def _pill(cv2, canvas, x2, y, text, colour, scale=0.9, text_colour=(0, 0, 0)):
    """Filled label ending at x2 (right-aligned); returns its left x."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x1 = x2 - w - 20
    cv2.rectangle(canvas, (x1, y), (x2, y + h + 18), colour, -1)
    cv2.putText(canvas, text, (x1 + 10, y + h + 9), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, 2, cv2.LINE_AA)
    return x1


def _draw_alert_panel(cv2, canvas, preds, t: float) -> None:
    """Top-right: every alert of the last SHOW_ALERT_S seconds, each with its verdict tag."""
    right = canvas.shape[1] - 16
    y = 16
    active = [m for m in preds.marks if m.kind != "missed" and 0.0 <= t - m.t <= SHOW_ALERT_S]
    if not active:
        _pill(cv2, canvas, right, y, "no alert", (90, 90, 90), 0.8, (235, 235, 235))
        return
    for m in active[-3:]:
        main = max(m.types, key=lambda k: _SEVERITY_ORDER.index(k) if k in _SEVERITY_ORDER else -1)
        name, colour = _EVENT_STYLE.get(main, (main, (200, 200, 200)))
        x = _pill(cv2, canvas, right, y, _VERDICT_TEXT[m.verdict], _VERDICT_COLOURS[m.verdict], 0.9,
                  (255, 255, 255) if m.verdict in ("FALSE ALARM", "MISSED") else (0, 0, 0))
        _pill(cv2, canvas, x - 6, y, "%s   %.1f s ago" % (name, t - m.t), colour, 0.9)
        chain = " > ".join(dict.fromkeys(_EVENT_STYLE.get(k, (k,))[0] for k in m.types))
        _pill(cv2, canvas, right, y + 50, chain + "   |   " + m.note, (30, 30, 30), 0.6, (235, 235, 235))
        y += 92


def _draw_truth(cv2, canvas, preds, t: float) -> None:
    """Bottom-left: the labelled fall / exit for 4 s from when it happens; MISSED if no alert caught it."""
    missed = {(round(m.t, 2), m.types[0]) for m in preds.marks if m.kind == "missed"}
    y = canvas.shape[0] - 70
    for kind, times in (("FALL", preds.impacts), ("BED_EXIT", preds.exits)):
        for ti in times:
            if 0.0 <= t - ti <= 4.0:
                text = "LABELLED FALL (impact)" if kind == "FALL" else "LABELLED BED EXIT"
                x = 16 + _pill_width(cv2, text)
                _pill(cv2, canvas, x, y, text, (240, 240, 240), 0.9)
                if (round(ti, 2), kind) in missed:
                    _pill(cv2, canvas, x + 12 + _pill_width(cv2, "MISSED: no alert"), y, "MISSED: no alert",
                          _VERDICT_COLOURS["MISSED"], 0.9, (255, 255, 255))
                y -= 56


def _pill_width(cv2, text, scale=0.9):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] + 20


def _draw_border(cv2, canvas, preds, t: float) -> None:
    """A thick frame border in the verdict colour for FLASH_S after an alert: false alarms stand out."""
    fresh = [m for m in preds.marks if m.kind != "missed" and 0.0 <= t - m.t <= FLASH_S]
    if not fresh:
        return
    worst = min(fresh, key=lambda m: ("FALSE ALARM", "RISKY", "NO EXIT", "EXTRA", "TRUE").index(m.verdict))
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), _VERDICT_COLOURS[worst.verdict], 14)


def _draw_banner(cv2, canvas, pred) -> None:
    """Top-right: what the system is showing staff at this moment."""
    if pred is None or pred.banner is None:
        return
    text = pred.banner + (" (%s)" % pred.alert_type if pred.alert_type else "")
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    x1 = canvas.shape[1] - 16
    cv2.rectangle(canvas, (x1 - w - 24, 16), (x1, 16 + h + 28), _BANNER_COLOURS[pred.banner], -1)
    cv2.putText(canvas, text, (x1 - w - 12, 16 + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)


def _draw_table(cv2, canvas, pred, t_s: float, index: int, show_rules: bool = False) -> None:
    """Top-left table: label / system for 4-state and 7-phase.

    "system" is what the dashboard shows: the trained models. Alerts come from the rule engine and
    are drawn separately; ``show_rules`` adds the rule engine's own states as an extra row.
    """
    rows = [("", "4-state", "7-phase")]
    keys = (("label", "label_posture", "label_phase"), ("system", "model_posture", "model_phase"))
    if show_rules:
        keys += (("rules", "rule_posture", "rule_phase"),)
    for title, posture_key, phase_key in keys:
        rows.append((title, getattr(pred, posture_key, None) if pred else None,
                     getattr(pred, phase_key, None) if pred else None))
    x0, y0, cols = 16, 16, (0, 110, 290)
    cv2.rectangle(canvas, (x0, y0), (x0 + 640, y0 + 60 + 34 * len(rows)), (0, 0, 0), -1)
    minutes = int(t_s // 60)
    cv2.putText(canvas, "%d:%04.1f  frame %d" % (minutes, t_s - 60 * minutes, index), (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    label_post = getattr(pred, "label_posture", None) if pred else None
    label_phase = getattr(pred, "label_phase", None) if pred else None
    for r, row in enumerate(rows):
        y = y0 + 60 + 34 * r
        for c, text in enumerate(row):
            colour = (255, 255, 255)
            if r >= 2 and c >= 1 and text is not None:
                truth = label_post if c == 1 else label_phase
                if truth is not None:
                    colour = (80, 220, 80) if text == truth else (80, 80, 255)
            cv2.putText(canvas, "-" if text is None else str(text), (x0 + 10 + cols[c], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, "system = trained models   |   alerts (right) = rule engine",
                (x0 + 10, y0 + 60 + 34 * (len(rows) - 1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)


def _timeline_strip(cv2, np, preds, duration: float, width: int):
    """Whole-clip coloured bars, one row per source, with a legend.

    With verdicts (``preds.marks``) the last two rows are the labelled events and every
    alert as a tick in its event colour, false alarms crossed out in red.
    """
    row_h, label_w, legend_h = 26, 190, 34
    judged = bool(getattr(preds, "marks", None) or getattr(preds, "impacts", None) or getattr(preds, "exits", None))
    rows = _ROWS[:-1] if judged else _ROWS
    n_rows = len(rows) + (2 if judged else 0)
    strip = np.full((row_h * n_rows + legend_h, width, 3), 24, np.uint8)
    frames = sorted(preds.frames.values(), key=lambda f: f.t)
    span = width - label_w - 10
    if judged:
        xt = lambda t: label_w + int(span * min(max(t / duration, 0.0), 1.0))  # noqa: E731
        y = len(rows) * row_h
        cv2.putText(strip, "labelled events", (8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        for ti in preds.impacts:
            cv2.rectangle(strip, (xt(ti) - 3, y + 3), (xt(ti) + 3, y + row_h - 3), (240, 240, 240), -1)
            cv2.putText(strip, "fall", (xt(ti) + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        for e in preds.exits:
            cv2.rectangle(strip, (xt(e) - 3, y + 3), (xt(e) + 3, y + row_h - 3), (200, 120, 230), -1)
            cv2.putText(strip, "exit", (xt(e) + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 120, 230), 1, cv2.LINE_AA)
        y += row_h
        cv2.putText(strip, "alerts", (8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        for m in preds.marks:
            x = xt(m.t)
            if m.kind == "missed":
                cv2.putText(strip, "MISSED", (x - 30, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2, cv2.LINE_AA)
                continue
            colour = _EVENT_STYLE.get(m.types[0], ("", (200, 200, 200)))[1]
            cv2.rectangle(strip, (x - 5, y + 3), (x + 5, y + row_h - 3), colour, -1)
            if m.verdict == "FALSE ALARM":
                cv2.line(strip, (x - 9, y + 2), (x + 9, y + row_h - 2), (60, 60, 255), 3)
                cv2.line(strip, (x - 9, y + row_h - 2), (x + 9, y + 2), (60, 60, 255), 3)
                cv2.putText(strip, "FALSE", (x + 12, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2, cv2.LINE_AA)
            elif m.verdict in ("RISKY", "NO EXIT", "EXTRA"):
                cv2.rectangle(strip, (x - 8, y + 1), (x + 8, y + row_h - 1), (0, 170, 255), 2)
    for r, (title, attr, colours) in enumerate(rows):
        y = r * row_h
        cv2.putText(strip, title, (8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        for a, b in zip(frames, frames[1:] + [None]):
            value = getattr(a, attr)
            if value is None:
                continue
            x1 = label_w + int(span * a.t / duration)
            x2 = label_w + int(span * ((b.t if b else duration) / duration))
            cv2.rectangle(strip, (x1, y + 3), (max(x1 + 1, x2), y + row_h - 3), colours.get(value, (128, 128, 128)), -1)
    x, y = 8, row_h * n_rows + 24
    for name, colour in list(_PHASE_COLOURS.items()) + [("|", (24, 24, 24))] + list(_POSTURE_COLOURS.items()):
        if name == "|":
            x += 20
            continue
        cv2.rectangle(strip, (x, y - 14), (x + 16, y + 2), colour, -1)
        cv2.putText(strip, name, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        x += 30 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    return strip


def _with_strip(cv2, np, canvas, strip, fraction: float):
    """Stack the timeline under the frame and draw the playback cursor."""
    if strip.shape[1] != canvas.shape[1]:
        strip = cv2.resize(strip, (canvas.shape[1], strip.shape[0]))
    strip = strip.copy()
    label_w, span = 190, strip.shape[1] - 190 - 10
    x = label_w + int(span * min(max(fraction, 0.0), 1.0))
    cv2.line(strip, (x, 0), (x, strip.shape[0] - 34), (255, 255, 255), 2)
    return np.vstack([canvas, strip])
