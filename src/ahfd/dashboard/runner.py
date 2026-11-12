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

import math
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from ahfd.dashboard.state import DashboardState
from ahfd.ml.clip_review import GETTING_UP_PHASES


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
        self._stream_max_width = cfg.dashboard.stream_max_width
        self._last_depth_preview = -math.inf
        self._depth_max_m = 6.0
        self._depth_spatial = 2
        self._sink = None
        self._track_bed_modes: dict[int, tuple[str, str, int]] = {}
        self._bed_incident_ids: dict[tuple[str, int], str] = {}
        self._posture_model = None
        self._phase_model = None
        self._last_frame_t0: float | None = None
        self._clips = None

    def _start_alert_clips(self) -> None:
        """Start saving clips around alerts, if configured. Fails closed: no clips."""
        clips = self.cfg.dashboard.alert_clips
        if not clips.enabled:
            return
        from ahfd.debug.alert_clips import AlertClipRecorder, check_clip_settings
        from ahfd.privacy import PrivacyViolation

        try:
            check_clip_settings(
                clips.mode, clips.dir,
                config_flag=self.cfg.privacy.allow_raw_capture, cli_flag=clips.cli_consent,
            )
        except (PrivacyViolation, ValueError) as exc:
            print("alert clips off: " + str(exc))
            return
        self._clips = AlertClipRecorder(
            clips.dir, mode=clips.mode, pre_s=clips.pre_s, post_s=clips.post_s,
            max_s=clips.max_s, min_free_gb=clips.min_free_gb, source=str(self.source_uri),
        )

    def _load_posture_model(self) -> None:
        """Load the display-only trained posture model, if configured.

        A missing or broken model only switches the comparison off (and says
        why on the page); it must never stop detection.
        """
        path = self.cfg.dashboard.posture_model
        if not path:
            return
        from ahfd.ml.live_posture import LivePostureModel

        try:
            self._posture_model = LivePostureModel(
                path, min_keypoint_score=self.cfg.pose.min_keypoint_score
            )
        except Exception as exc:  # noqa: BLE001 - shown on the page instead
            self._posture_model = None
            self.state.update_runtime(
                self.gen, posture_model_error=f"{path}: {exc}"[:200]
            )
            return
        self.state.update_runtime(
            self.gen, posture_model=Path(path).name, posture_model_error=None
        )

    def _load_phase_model(self) -> None:
        """Load the display-only 7-phase model for this camera's view, if configured."""
        dash = self.cfg.dashboard
        path = next((o.phase_model for o in dash.sources if o.uri == self.source_uri and o.phase_model), None)
        path = path or dash.phase_model
        if not path:
            return
        from ahfd.ml.live_phase import LivePhaseModel

        try:
            self._phase_model = LivePhaseModel(
                path, max_gap_s=self.cfg.bed_activity.cusum.max_gap_s,
                near_edge_m=self.cfg.bed_activity.near_edge_enter_m,
                min_keypoint_score=self.cfg.pose.min_keypoint_score,
            )
        except Exception as exc:  # noqa: BLE001 - shown on the page instead
            self._phase_model = None
            self.state.update_runtime(self.gen, phase_model_error=f"{path}: {exc}"[:200])
            return
        self.state.update_runtime(self.gen, phase_model=Path(path).name, phase_model_error=None)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _coalesce_bed_incident(self, event, fresh_id: str) -> tuple[str, bool]:
        """Keep warning/exit identity per patient track, even on one bed."""

        if not event.zone:
            return fresh_id, False
        key = (event.zone, event.track_id)
        if event.type == "BED_EXIT_WARNING":
            self._bed_incident_ids[key] = fresh_id
            return fresh_id, False
        if event.type == "BED_EXIT":
            existing = self._bed_incident_ids.get(key)
            if existing is not None:
                return existing, True
        return fresh_id, False

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
            self._start_alert_clips()
            src = open_source(
                self.source_uri, width=cfg.capture.width, height=cfg.capture.height
            )
            self._configure_depth_preview()
            state.publish_status(
                gen,
                "starting",
                resolution=str(src.meta.width) + "x" + str(src.meta.height),
                source_fps=round(src.meta.fps, 1),
                depth_stream_enabled=bool(src.meta.has_depth),
                depth_active=False,
                depth_valid_fraction=0.0,
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

                extractor, machine, self._sink, calibration = _build_detection(
                    cfg,
                    self.calib_path,
                    src.meta,
                    # Candidate generation stays inside the causal bed machine;
                    # per-bed dashboard policy decides whether it is admitted.
                    emit_bed_early_warning=True,
                )
                from ahfd.cli import _polygons_overlap
                from ahfd.dashboard.bed_modes import mode_from_legacy_risk

                beds = [
                    zone for zone in calibration.zones.zones if zone.kind == "bed"
                ]
                names = [bed.name for bed in beds]
                overlaps = [
                    (left.name, right.name)
                    for index, left in enumerate(beds)
                    for right in beds[index + 1 :]
                    if _polygons_overlap(left.polygon, right.polygon)
                ]
                if not cfg.bed_activity.enabled:
                    state.register_beds([], gen=gen)
                    state.update_runtime(
                        gen,
                        bed_policy_warning=(
                            "bed modes disabled: temporal bed activity is off; "
                            "fall detection remains active"
                        ),
                    )
                elif len(beds) != 2:
                    state.register_beds([], gen=gen)
                    state.update_runtime(
                        gen,
                        bed_policy_warning=(
                            "bed modes disabled: this deployment needs exactly "
                            "two calibrated bed zones; fall detection remains active"
                        ),
                    )
                elif len(names) != len(set(names)) or overlaps:
                    state.register_beds([], gen=gen)
                    state.update_runtime(
                        gen,
                        bed_policy_warning=(
                            "bed modes disabled: calibration has duplicate or "
                            "overlapping bed zones; fall detection remains active"
                        ),
                    )
                else:
                    state.register_beds(
                        [
                            (bed.name, mode_from_legacy_risk(bed.risk_level))
                            for bed in beds
                        ],
                        gen=gen,
                    )
                self._load_posture_model()
                self._load_phase_model()

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
            if self._sink is not None:
                self._sink.close()
                self._sink = None
            if self._clips is not None:
                self._clips.close()
                self._clips = None

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
        fps = getattr(src, "replay_fps", None) or src.meta.fps or 30.0

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
        anchor: tuple[float, int, float] | None = None  # (wall_time, frame, speed) when playing
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
                # A speed change re-anchors at the current frame; otherwise the new
                # speed would rescale all the time since play started and jump.
                if anchor is None or anchor[2] != speed:
                    anchor = (time.monotonic(), cur, speed)
                w0, f0, _ = anchor
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

    def _configure_depth_preview(self) -> None:
        """Use the same range/options as capture, not a second fixed 6 m cut."""
        from ahfd.capture.factory import parse_realsense_uri

        options = (
            parse_realsense_uri(self.source_uri)
            if self.source_uri.startswith("rs://") else {}
        )
        self._depth_max_m = float(options.get("max_range_m", 6.0))
        self._depth_spatial = max(1, min(5, int(options.get("spatial_magnitude", 2))))
        # Colour range fitted to this camera's scene on the first good frame,
        # then held fixed so colours stay stable when someone walks in.
        self._depth_range = None
        self.state.update_runtime(
            self.gen, depth_min_m=0.3, depth_max_m=self._depth_max_m,
            depth_spatial_passes=self._depth_spatial,
            depth_preview_hole_filled=False,
        )

    def _publish_depth_preview(self, frame, rendered_rgb: bool, pose=None, ground=None) -> None:
        """At most 10 Hz, one small encode for all viewers; no second camera.

        Skipped while nobody has the depth view open: it costs about 35 ms a
        frame (measured 2 Nov), a sixth of the whole loop at the default setup.
        In "height" mode (the default when the camera is calibrated) every
        pixel is coloured by its height above the floor, with the skeletons
        and their head/hip heights on top; "distance" colours by range.
        """
        if not rendered_rgb or not self.state.depth_wanted():
            return
        now = time.monotonic()
        height_mode = (
            self.state.depth_mode() == "height" and ground is not None
            and frame.intrinsics is not None and frame.depth_raw is not None
        )
        # The height map costs ~40 ms; 5 Hz is plenty for a picture of the room.
        if now - self._last_depth_preview < (0.2 if height_mode else 0.1):
            return
        self._last_depth_preview = now
        from ahfd.viz.depth_render import render_depth, render_height_map, scene_depth_range

        if height_mode:
            canvas = render_height_map(
                frame.depth_raw, frame.depth_scale, frame.intrinsics, ground,
                people=pose.people if pose is not None else None,
                min_keypoint_score=self.cfg.pose.min_keypoint_score,
            )
            if canvas is not None:
                ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
                self.state.publish_depth_frame(buf.tobytes() if ok else None, gen=self.gen)
                return

        if getattr(self, "_depth_range", None) is None:
            self._depth_range = scene_depth_range(
                frame.depth_raw, frame.depth_scale, ceiling_m=self._depth_max_m,
            )
            if self._depth_range is not None:
                self.state.update_runtime(
                    self.gen, depth_min_m=self._depth_range[0],
                    depth_max_m=self._depth_range[1],
                )
        lo, hi = self._depth_range or (0.3, self._depth_max_m)
        canvas = render_depth(
            frame.depth_raw, frame.depth_scale, min_depth_m=lo, max_depth_m=hi,
            shade=True, legend=True, clamp=self._depth_range is not None,
        )
        jpeg = None
        if canvas is not None:
            ok, buf = cv2.imencode(
                ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            )
            if ok:
                jpeg = buf.tobytes()
        self.state.publish_depth_frame(jpeg, gen=self.gen)

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

            raw_depth = frame.depth_raw
            depth_valid_fraction = 0.0
            if raw_depth is not None and raw_depth.size:
                depth_valid_fraction = float(np.count_nonzero(raw_depth)) / float(
                    raw_depth.size
                )
            self.state.update_runtime(
                self.gen,
                depth_active=bool(
                    frame.intrinsics is not None and depth_valid_fraction >= 0.01
                ),
                depth_valid_fraction=round(depth_valid_fraction, 3),
            )

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

            if extractor is not None and frame.depth_raw is not None:
                from ahfd.features import attach_depth_heights

                pose = attach_depth_heights(
                    pose,
                    frame,
                    extractor.depth_ground,
                    min_score=cfg.pose.min_keypoint_score,
                )

            alert = None
            # Per-track feature snapshot for the dashboard (state + a metric
            # or two the nurse can read). Built even when detection is off,
            # so the "people in view" panel always populates.
            track_info: dict[int, dict] = {}
            if machine is not None and extractor is not None:
                extractor.retain_only(tracker.live_ids)
                machine.retain_only(tracker.live_ids)
                if self._posture_model is not None:
                    self._posture_model.retain_only(tracker.live_ids)
                if self._phase_model is not None:
                    self._phase_model.retain_only(tracker.live_ids)
                observed_ids: set[int] = set()
                for person in pose.people:
                    if person.track_id is None:
                        continue
                    features = extractor.extract(person, pose.t)
                    info = {"track_id": person.track_id}
                    if features is not None:
                        if features.h_torso is not None:
                            info["height_m"] = round(features.h_torso, 2)
                        if features.zones:
                            info["zone"] = features.zones[0]
                        if self._posture_model is not None:
                            info["model_posture"] = self._posture_model.predict(
                                person.track_id, features, person, extractor.ground
                            )
                        associated_bed = (
                            features.associated_bed or features.supported_by_bed
                        )
                        previous_policy = self._track_bed_modes.get(person.track_id)
                        policy_bed = (
                            str(associated_bed)
                            if associated_bed
                            else (previous_policy[0] if previous_policy else None)
                        )
                        policy_snapshot = previous_policy
                        if policy_bed is not None:
                            selected_mode, policy_revision = self.state.bed_policy(
                                policy_bed
                            )
                            policy_snapshot = (
                                policy_bed,
                                selected_mode or "unassigned",
                                policy_revision,
                            )
                            if (
                                previous_policy is not None
                                and previous_policy[0] != policy_snapshot[0]
                            ):
                                # Physical support from Bed A must never make a
                                # later association with Bed B look like its exit.
                                if hasattr(machine, "reset_bed"):
                                    machine.reset_bed(person.track_id)
                                self._bed_incident_ids.pop(
                                    (previous_policy[0], person.track_id), None
                                )
                            elif (
                                previous_policy is not None
                                and previous_policy[2] != policy_snapshot[2]
                                and hasattr(machine, "reset_bed_alert_evidence")
                            ):
                                # A live mode change starts with fresh warning
                                # evidence but retains physical support so an
                                # exit already in progress can still complete.
                                machine.reset_bed_alert_evidence(person.track_id)
                            self._track_bed_modes[person.track_id] = policy_snapshot
                        observed_ids.add(person.track_id)
                        for raw_event in machine.update_all(features):
                            from ahfd.dashboard.bed_modes import (
                                apply_bed_mode,
                                is_dashboard_alert,
                            )

                            event = raw_event
                            mode = None
                            if event.type in {"BED_EXIT", "BED_EXIT_WARNING"}:
                                # One atomic policy snapshot governs reset and
                                # admission for this frame. A concurrent POST is
                                # observed on the next frame and cannot promote
                                # already-accumulated evidence into High mode.
                                if (
                                    policy_snapshot is None
                                    or event.zone != policy_snapshot[0]
                                    or policy_snapshot[1] == "unassigned"
                                ):
                                    continue
                                mode = policy_snapshot[1]
                                event = apply_bed_mode(event, mode)
                                if event is None:
                                    continue
                            event_id = uuid.uuid4().hex[:12]
                            event_id, escalated_incident = (
                                self._coalesce_bed_incident(event, event_id)
                            )
                            # The dashboard is a real detector entry point, so it
                            # must honour the same configured durable sinks as
                            # ``ahfd run``. Previously it only painted the browser.
                            if self._sink is not None:
                                self._sink.emit(event)
                            # A person getting up updates their open fall alert
                            # rather than raising a second one. With no fall
                            # alert yet it is the only alert for that fall.
                            already_alerted = (
                                event.type == "NEAR_MISS"
                                and self.state.mark_recovered(
                                    event.track_id, time.strftime("%H:%M:%S"), gen=self.gen
                                )
                            )
                            alert_record = {
                                    "event_id": event_id,
                                    "type": event.type,
                                    "severity": event.severity,
                                    "alerting": is_dashboard_alert(event) and not already_alerted,
                                    "reopen_on_escalation": escalated_incident,
                                    "monitoring_mode": mode,
                                    "track_id": event.track_id,
                                    "t_alert": round(event.t_alert, 1),
                                    "clock": time.strftime("%H:%M:%S"),
                                    "zone": event.zone,
                                    # Off a bed there is no zone: name the bed they were at.
                                    "near_bed": self._track_bed_modes.get(event.track_id, (None,))[0],
                                    "evidence": event.evidence,
                            }
                            self.state.publish_event(alert_record, gen=self.gen)
                            if self._clips is not None and is_dashboard_alert(event):
                                try:
                                    self._clips.on_alert(alert_record)
                                except Exception as exc:
                                    print("alert clips off: " + repr(exc))
                                    self._clips = None
                            if event.type in ("POSSIBLE_FALL", "FALL_SUSPECTED", "FALL_CONFIRMED", "PERSON_DOWN"):
                                alert = event.describe()

                    # Read states after update so the dashboard is not one frame
                    # behind. Bed phase/support/availability remain separate.
                    info["state"] = machine.state_of(person.track_id)
                    if machine.possible_fall_of(person.track_id, pose.t):
                        info["possible_fall"] = True
                    if hasattr(machine, "bed_snapshot_of"):
                        bed = machine.bed_snapshot_of(person.track_id, pose.t)
                        if bed is not None:
                            track_policy = self._track_bed_modes.get(person.track_id)
                            bed_mode = (
                                track_policy[1]
                                if track_policy is not None
                                and track_policy[0] == bed.bed_id
                                else "unassigned"
                            )
                            if bed.bed_id and bed.phase == "RECLINED":
                                self._bed_incident_ids.pop(
                                    (bed.bed_id, person.track_id), None
                                )
                            info.update(
                                {
                                    "bed_id": bed.bed_id,
                                    "bed_mode": bed_mode or "unassigned",
                                    "bed_phase": bed.phase,
                                    "bed_support": bed.support,
                                    "observation": bed.observation,
                                    "warning_candidate": (
                                        bed_mode == "high"
                                        and bed.early_warning_candidate
                                    ),
                                    "cusum_g": round(bed.cusum_g, 2),
                                }
                            )
                    if self._phase_model is not None and features is not None:
                        snap = (
                            machine.bed_snapshot_of(person.track_id, pose.t)
                            if hasattr(machine, "bed_snapshot_of") else None
                        )
                        try:
                            model_phase = self._phase_model.predict(
                                person.track_id, features, person, extractor.ground, snap
                            )
                        except Exception as exc:  # noqa: BLE001 - display only, never stops detection
                            self._phase_model = None
                            self.state.update_runtime(self.gen, phase_model_error=repr(exc)[:200])
                        else:
                            if model_phase:
                                info["model_phase"] = model_phase
                    if person.heights is not None:
                        values = [
                            float(value)
                            for value in person.heights
                            if math.isfinite(float(value))
                        ]
                        info["depth_valid_joints"] = len(values)
                        info["depth_total_joints"] = len(person.heights)
                        # Head and hip heights for the live height chart.
                        for key, idx in (("head_h", (0, 1, 2, 3, 4)), ("hip_h", (11, 12))):
                            hv = [
                                float(person.heights[i]) for i in idx
                                if i < len(person.heights)
                                and math.isfinite(float(person.heights[i]))
                                and person.scores[i] >= cfg.pose.min_keypoint_score
                            ]
                            if hv:
                                info[key] = round(sorted(hv)[len(hv) // 2], 2)
                    track_info[person.track_id] = info

                if hasattr(machine, "mark_frame_unobserved"):
                    machine.mark_frame_unobserved(
                        pose.t, set(tracker.live_ids), observed_ids
                    )
                for track_id in list(self._track_bed_modes):
                    if track_id not in tracker.live_ids:
                        bed_id, _mode, _revision = self._track_bed_modes.pop(track_id)
                        if bed_id:
                            self._bed_incident_ids.pop((bed_id, track_id), None)
            else:
                for person in pose.people:
                    if person.track_id is not None:
                        track_info[person.track_id] = {
                            "track_id": person.track_id,
                            "state": "TRACKED",
                        }

            states = {
                tid: info["state"]
                + (
                    " | model: " + ", ".join(
                        str(info[k]) for k in ("model_posture", "model_phase") if k in info
                    )
                    if "model_posture" in info or "model_phase" in info else ""
                )
                + (" | POSSIBLE FALL" if info.get("possible_fall") else "")
                + (
                    " | GETTING UP"
                    if info.get("bed_phase") in GETTING_UP_PHASES
                    and info["state"] not in ("ON_GROUND", "FALLING")
                    else ""
                )
                for tid, info in track_info.items()
            }

            # Frames handled per second, start to start, so the camera read, depth
            # alignment, drawing and encoding all count. The old figure timed only
            # this block and read 12.7 fps when the real rate was 4.9 (2 Nov).
            prev = self._last_frame_t0
            self._last_frame_t0 = t0
            if prev is not None and t0 > prev and t0 - prev < 5.0:
                inst = 1.0 / (t0 - prev)
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            # Tag the actual render, not a flag re-read after encoding. An off
            # request can arrive while rendering, and the store rejects that
            # in-flight RGB publication even if the runner flag has changed.
            rendered_rgb = self.show_rgb
            self._publish_depth_preview(
                frame, rendered_rgb, pose,
                getattr(extractor, "depth_ground", None) if extractor is not None else None,
            )
            if rendered_rgb:
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

            limit = self._stream_max_width
            if limit and canvas.shape[1] > limit:
                canvas = cv2.resize(
                    canvas, (limit, round(canvas.shape[0] * limit / canvas.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            )
            if ok:
                self.state.publish_frame(
                    buf.tobytes(), list(track_info.values()), fps_ema or 0.0,
                    gen=self.gen, show_rgb=rendered_rgb,
                )
                if self._clips is not None:
                    try:
                        self._offer_clip_frame(pose, states, fps_ema, buf, rendered_rgb)
                    except Exception as exc:
                        # Evidence clips must never stop the monitoring itself.
                        print("alert clips off: " + repr(exc))
                        self._clips = None

        return fps_ema if fps_ema is not None else 0.0

    def _offer_clip_frame(self, pose, states, fps, shown_jpeg, shown_rgb) -> None:
        """Hand the recorder this frame, with no camera pixels unless its mode allows them."""
        import cv2

        from ahfd.viz import render_skeleton

        if shown_rgb and self._clips.mode == "rgb":
            self._clips.add_frame(shown_jpeg.tobytes(), is_rgb=True)
            return
        if not shown_rgb:
            self._clips.add_frame(shown_jpeg.tobytes())
            return
        # The page shows the camera, but this recorder may only keep stick figures.
        canvas = render_skeleton(
            pose, min_keypoint_score=self.cfg.pose.min_keypoint_score, states=states, fps=fps,
        )
        width = self._stream_max_width or 960
        if canvas.shape[1] > width:
            canvas = cv2.resize(
                canvas, (width, round(canvas.shape[0] * width / canvas.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if ok:
            self._clips.add_frame(buf.tobytes())
