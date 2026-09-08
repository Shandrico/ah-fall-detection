"""The dashboard's pipeline thread.

Runs capture -> pose -> track -> smooth -> (detect) exactly like `ahfd run`,
but instead of showing a window it encodes each annotated frame to JPEG once
and publishes it to the shared `DashboardState`. Exactly one of these runs per
dashboard, regardless of how many browsers are watching -- which is the whole
point (see state.py).

`show_rgb` chooses the annotated frame: the RGB overlay (reverses the
skeleton-only stance; opt-in) or the skeleton on black (privacy-safe default).
"""

from __future__ import annotations

import threading
import time
import uuid

import cv2

from ahfd.dashboard.state import DashboardState


class PipelineRunner:
    def __init__(self, source_uri, cfg, calib_path, state: DashboardState, show_rgb: bool):
        self.source_uri = source_uri
        self.cfg = cfg
        self.calib_path = calib_path
        self.state = state
        self.show_rgb = show_rgb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg_quality = 80

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        from ahfd.capture import open_source
        from ahfd.pose import KeypointSmoother, build_estimator
        from ahfd.track import SimpleTracker
        from ahfd.viz import render_overlay, render_skeleton

        cfg = self.cfg
        src = open_source(self.source_uri)
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

        extractor = machine = None
        if cfg.detect.enabled:
            # Deferred import to avoid a hard dependency when detection is off.
            from ahfd.cli import _build_detection

            extractor, machine, _sink, _calib = _build_detection(
                cfg, self.calib_path, src.meta
            )

        fps_ema: float | None = None
        try:
            for frame in src:
                if self._stop.is_set():
                    break
                t0 = time.perf_counter()

                pose = estimator.estimate(frame)
                pose = tracker.update(pose)
                if smoother is not None:
                    smoother.retain_only(tracker.live_ids)
                    pose = pose.with_people(
                        tuple(
                            p.with_keypoints(
                                smoother.smooth(p.track_id, pose.t, p.keypoints)
                            )
                            for p in pose.people
                        )
                    )

                alert = None
                if machine is not None and extractor is not None:
                    extractor.retain_only(tracker.live_ids)
                    machine.retain_only(tracker.live_ids)
                    for person in pose.people:
                        features = extractor.extract(person, pose.t)
                        if features is None:
                            continue
                        event = machine.update(features)
                        if event is not None:
                            self.state.publish_event(
                                {
                                    "event_id": uuid.uuid4().hex[:12],
                                    "type": event.type,
                                    "severity": event.severity,
                                    "track_id": event.track_id,
                                    "t_alert": round(event.t_alert, 1),
                                    "zone": event.zone,
                                    "evidence": event.evidence,
                                }
                            )
                            if event.type in ("FALL_CONFIRMED", "PERSON_DOWN"):
                                alert = event.describe()

                states = {
                    p.track_id: machine.state_of(p.track_id)
                    for p in pose.people
                    if p.track_id is not None
                } if machine is not None else {
                    p.track_id: "TRACKED"
                    for p in pose.people
                    if p.track_id is not None
                }

                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

                if self.show_rgb:
                    canvas = render_overlay(
                        frame, pose,
                        min_keypoint_score=cfg.pose.min_keypoint_score,
                        states=states, alert=alert, fps=fps_ema,
                    )
                else:
                    canvas = render_skeleton(
                        pose, min_keypoint_score=cfg.pose.min_keypoint_score,
                        states=states, fps=fps_ema,
                    )

                ok, buf = cv2.imencode(
                    ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
                )
                if ok:
                    self.state.publish_frame(buf.tobytes(), states, fps_ema or 0.0)
        finally:
            src.close()
