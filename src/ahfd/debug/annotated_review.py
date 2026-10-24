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
    tracker = SimpleTracker(min_keypoint_score=cfg.pose.min_keypoint_score)
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
