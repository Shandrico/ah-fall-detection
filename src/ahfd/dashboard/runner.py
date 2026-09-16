"""The dashboard's pipeline thread.

Runs capture -> pose -> track -> smooth -> (detect) exactly like `ahfd run`,
but instead of showing a window it encodes each annotated frame to JPEG once
and publishes it to the shared `DashboardState`. Exactly one of these runs per
dashboard, regardless of how many browsers are watching -- which is the whole
point (see state.py).

`show_rgb` chooses the annotated frame: the RGB overlay (reverses the
skeleton-only stance; opt-in) or the skeleton on black (privacy-safe default).
It is read per frame, so it can be flipped live without restarting anything.
"""

from __future__ import annotations

import threading
import time
import uuid

import cv2

from ahfd.dashboard.state import DashboardState


class PipelineRunner:
    """One source, one model, one thread. Single-use by design.

    To change the source or the model, build a new one (see controller.py).
    Restarting in place would mean clearing the stop event and hoping the old
    thread had really let go of the camera -- exactly the situation where two
    threads end up fighting over one device.
    """

    def __init__(
        self,
        source_uri,
        cfg,
        calib_path,
        state: DashboardState,
        show_rgb: bool,
        *,
        gen: int = 0,
    ):
        self.source_uri = source_uri
        self.cfg = cfg
        self.calib_path = calib_path
        self.state = state
        self.show_rgb = show_rgb
        self.gen = gen
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._jpeg_quality = cfg.dashboard.jpeg_quality

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("PipelineRunner is single-use -- construct a new one")
        self._started = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ahfd-pipeline-" + str(self.gen)
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the loop to finish.

        May return with the thread still running: a blocking read on a camera
        that has been unplugged cannot be interrupted. Callers must check
        `alive` rather than assume the device has been released.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        from ahfd.capture import open_source
        from ahfd.pose import KeypointSmoother, build_estimator
        from ahfd.track import SimpleTracker

        state, gen, cfg = self.state, self.gen, self.cfg
        state.publish_status(
            gen,
            "starting",
            source=self.source_uri,
            backend=cfg.pose.backend,
            show_rgb=self.show_rgb,
        )

        src = None
        try:
            src = open_source(
                self.source_uri, width=cfg.capture.width, height=cfg.capture.height
            )
            state.publish_status(
                gen,
                "starting",
                resolution=str(src.meta.width) + "x" + str(src.meta.height),
                source_fps=round(src.meta.fps, 1),
            )

            extractor = machine = None
            if cfg.detect.enabled:
                # Before the model, not after: a missing or resolution-mismatched
                # calibration is a config error and should surface in a second,
                # not once a 35 MB download has finished. Deferred import to
                # avoid a hard dependency when detection is off.
                #
                # The dashboard controller disables detection for a camera it
                # deliberately listed without a calibration (see _spawn), so a
                # missing calibration reaching HERE is a genuine mistake worth an
                # error -- not the multi-camera "uncalibrated spare camera" case.
                from ahfd.cli import _build_detection

                extractor, machine, _sink, _calib = _build_detection(
                    cfg, self.calib_path, src.meta
                )

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

            # `detect` rides along so the page can explain why every chip reads
            # TRACKED: with detection off there is no posture to report, which
            # otherwise looks exactly like a broken state machine.
            state.publish_status(
                gen, "running", model=estimator.name, detect=cfg.detect.enabled
            )
            frames_seen = self._loop(
                src, estimator, tracker, smoother, extractor, machine
            )
            if self._stop.is_set():
                state.publish_status(gen, "stopped")
            elif frames_seen == 0:
                # The source opened but never produced a frame. On Windows the
                # webcam is exclusive, so the usual cause is another program (or
                # a leftover dashboard) still holding it -- which otherwise shows
                # as a silent dead feed with no error at all.
                state.publish_status(
                    gen,
                    "error",
                    error="no frames from "
                    + str(self.source_uri)
                    + " -- the camera may be in use by another program (only one "
                    "at a time on Windows) or disconnected.",
                )
            else:
                state.publish_status(gen, "ended")
        except Exception as exc:  # noqa: BLE001 -- a dead daemon thread tells nobody
            # typer.BadParameter is a click UsageError: the text is on .message,
            # and format_message() would prepend CLI framing that means nothing
            # in a browser.
            state.publish_status(
                gen, "error", error=getattr(exc, "message", None) or str(exc)
            )
        finally:
            if src is not None:
                src.close()

    def _loop(self, src, estimator, tracker, smoother, extractor, machine) -> int:
        """Run until the source ends or a stop is asked. Returns frames processed.

        A seekable file gets the player loop (scrub / play / pause / step); a
        live camera streams straight through.
        """
        if getattr(src, "seekable", False) and src.meta.frame_count:
            return self._loop_replay(src, estimator, tracker, smoother, extractor, machine)

        fps_ema: float | None = None
        frames_seen = 0
        for frame in src:
            if self._stop.is_set():
                break
            frames_seen += 1
            fps_ema = self._process_frame(
                frame, estimator, tracker, smoother, extractor, machine, fps_ema
            )
        return frames_seen

    def _loop_replay(self, src, estimator, tracker, smoother, extractor, machine) -> int:
        """Player loop for a recording: obey scrub / play / pause / step commands.

        Pose only runs when the shown frame changes, so pausing costs nothing.
        A seek is a jump in time, which would otherwise fabricate a huge
        velocity and a phantom fall, so the detector's per-track history is
        cleared on every seek -- fall EVENTS are therefore meaningful only while
        playing forward, which is the honest contract for a scrub tool.
        """
        total = src.meta.frame_count
        self.state.announce_replay(total, self.gen)
        fps = src.meta.fps or 30.0

        def reset_state():
            # A seek is a jump in time: motion continuity is gone, so clear
            # EVERYTHING that carries per-track history across frames -- the
            # tracker (else ids blend/confuse), the smoother (else the skeleton
            # sticks halfway between the old and new frame), and the detector
            # (else the time jump fabricates a huge velocity / phantom fall).
            tracker.reset()
            if smoother is not None:
                smoother.retain_only(set())
            if extractor is not None:
                extractor.retain_only(set())
            if machine is not None:
                machine.retain_only(set())

        fps_ema: float | None = None
        frames_seen = 0
        cur = 0
        last = -1
        anchor: tuple[float, int] | None = None  # (wall_time, frame) when playing
        while not self._stop.is_set():
            paused, seek, speed = self.state.take_replay_command()

            if seek is not None:
                cur = max(0, min(seek, total - 1))
                reset_state()
                anchor = None

            if not paused:
                # Play at real time (or speed x): the frame to show is derived
                # from wall-clock, so if pose can't keep up the loop SKIPS frames
                # to stay in sync -- exactly like a normal video player under
                # load, instead of the previous slow-motion.
                if anchor is None:
                    anchor = (time.monotonic(), cur)
                w0, f0 = anchor
                cur = min(total - 1, f0 + int((time.monotonic() - w0) * fps * speed))
            else:
                anchor = None

            if cur != last:
                frame = src.read_at(cur)
                if frame is not None:
                    fps_ema = self._process_frame(
                        frame, estimator, tracker, smoother, extractor, machine, fps_ema
                    )
                    frames_seen += 1
                    last = cur
                    self.state.publish_replay_pos(cur, self.gen)

            if not paused and cur >= total - 1:
                self.state.replay_control("pause")  # reached the end -> stop
            time.sleep(0.005)  # yield; the wall-clock target sets the real pace
        return frames_seen

    def _process_frame(
        self, frame, estimator, tracker, smoother, extractor, machine, fps_ema
    ) -> float:
        """Pose -> track -> detect -> encode -> publish for one frame.

        Returns the updated fps EMA. Shared by the live and replay loops so both
        annotate and publish identically.
        """
        cfg = self.cfg
        from ahfd.viz import render_overlay, render_skeleton

        if True:
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
            # Per-track feature snapshot for the dashboard (state + a metric
            # or two the nurse can read). Built even when detection is off,
            # so the "people in view" panel always populates.
            track_info: dict[int, dict] = {}
            if machine is not None and extractor is not None:
                extractor.retain_only(tracker.live_ids)
                machine.retain_only(tracker.live_ids)
                for person in pose.people:
                    if person.track_id is None:
                        continue
                    features = extractor.extract(person, pose.t)
                    info = {
                        "track_id": person.track_id,
                        "state": machine.state_of(person.track_id),
                    }
                    if features is not None:
                        if features.h_torso is not None:
                            info["height_m"] = round(features.h_torso, 2)
                        if features.zones:
                            info["zone"] = features.zones[0]
                    track_info[person.track_id] = info
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
                                "clock": time.strftime("%H:%M:%S"),
                                "zone": event.zone,
                                "evidence": event.evidence,
                            },
                            gen=self.gen,
                        )
                        if event.type in ("FALL_CONFIRMED", "PERSON_DOWN"):
                            alert = event.describe()
            else:
                for person in pose.people:
                    if person.track_id is not None:
                        track_info[person.track_id] = {
                            "track_id": person.track_id,
                            "state": "TRACKED",
                        }

            states = {tid: info["state"] for tid, info in track_info.items()}

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
                self.state.publish_frame(
                    buf.tobytes(), list(track_info.values()), fps_ema or 0.0,
                    gen=self.gen,
                )

        return fps_ema if fps_ema is not None else 0.0
