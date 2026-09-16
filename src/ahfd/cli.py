"""Command line entry points.

    ahfd run       webcam -> pose -> skeleton, and detection if calibrated
    ahfd extract   a clip -> tracks.jsonl (keypoints only). Run pose once.
    ahfd replay    tracks.jsonl -> detection, fast and deterministic
    ahfd sweep     tune one threshold against labelled tracks
    ahfd eval      score event logs against ground truth
    ahfd bench     pose backend bake-off
    ahfd info      environment and hardware report
"""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ahfd.config import load_config

app = typer.Typer(add_completion=False, help="Privacy-preserving fall detection.")


def _build_detection(cfg, calib_path, meta):
    """Wire up feature extractor + state machine + sinks from a calibration.

    Shared by `run` and `replay` so the detection path is defined once. Returns
    (extractor, machine, sink) or raises typer.BadParameter if the calibration
    is missing or its resolution does not match the stream. Kept out of the
    per-command bodies because getting the resolution guard wrong produces
    plausible-but-wrong metres, and it must be identical everywhere.
    """
    from ahfd.alert import ConsoleSink, JsonlSink, MultiSink
    from ahfd.detect import FallStateMachine
    from ahfd.features import FeatureExtractor
    from ahfd.geometry.calibration import load_calibration

    if not calib_path:
        raise typer.BadParameter(
            "fall detection needs a calibration: thresholds are metric, so the "
            "camera height and tilt are required. Pass --calibration or set "
            "'calibration:' in the config. See calib/example_ward6.yaml."
        )
    calib = load_calibration(calib_path)
    _check_calibration_resolution(calib, meta)

    extractor = FeatureExtractor(
        calib.ground, zones=calib.zones, min_keypoint_score=cfg.pose.min_keypoint_score
    )
    machine = FallStateMachine(cfg.detect.to_thresholds())

    sinks: list = []
    if cfg.alert.console:
        sinks.append(ConsoleSink(min_severity=cfg.alert.min_severity))
    if cfg.alert.jsonl_path:
        sinks.append(JsonlSink(cfg.alert.jsonl_path))
    return extractor, machine, MultiSink(*sinks), calib


@app.command()
def run(
    source: str = typer.Option(
        None, help="Source URI, e.g. webcam://0 or file://clip.mp4"
    ),
    config: Path = typer.Option(None, help="Path to a YAML config."),
    calibration: Path = typer.Option(
        None, help="Per-camera calibration YAML. Required for fall detection."
    ),
    backend: str = typer.Option(
        None, help="Override the pose backend: rtmo | rtmpose | yolo."
    ),
    detect: bool = typer.Option(
        None,
        "--detect/--no-detect",
        help="Turn fall detection on or off. On needs a calibration matching "
        "the capture resolution.",
    ),
    view: str = typer.Option(None, help="'skeleton' or 'none' for headless."),
    max_frames: int = typer.Option(
        0, help="Stop after N frames. 0 runs until you quit."
    ),
) -> None:
    """Run the pipeline: capture, pose, track, smooth, render."""
    import cv2

    from ahfd.capture import open_source
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.track import SimpleTracker
    from ahfd.viz import render_overlay, render_skeleton

    cfg = load_config(config)
    if backend:
        cfg.pose.backend = backend  # CLI override -- swap models without editing the config
    if detect is not None:
        cfg.detect.enabled = detect
    uri = source or cfg.source
    view_mode = view or cfg.view.mode

    # Open the source before loading the model: a busy webcam or a missing
    # file should fail immediately, not after a 35 MB download.
    src = open_source(uri)
    typer.echo(
        "source:  "
        + uri
        + "  "
        + str(src.meta.width)
        + "x"
        + str(src.meta.height)
        + " @ "
        + format(src.meta.fps, ".0f")
        + " fps"
    )
    typer.echo("loading pose model (the first run downloads weights)...")

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

    typer.echo("model:   " + estimator.name)

    # --- detection, only if a camera calibration exists ------------------
    # Fall detection is refused without calibration rather than guessed at.
    # Every threshold is a height in metres, and without the camera's height
    # and tilt there is no way to compute one -- a default would produce
    # confident, meaningless alerts, which is worse than none.
    extractor = None
    machine = None
    sink = None

    calib_path = calibration or cfg.calibration
    if cfg.detect.enabled:
        extractor, machine, sink, calib = _build_detection(cfg, calib_path, src.meta)
        typer.echo(
            "calib:   "
            + calib.camera_id
            + "  height "
            + format(calib.height_m, ".2f")
            + " m  pitch "
            + format(calib.ground.pitch_deg, ".1f")
            + " deg  zones "
            + str(len(calib.zones.zones))
        )
    else:
        typer.echo("detect:  off -- pose and tracking only")

    windowed = view_mode in ("skeleton", "overlay")
    if windowed:
        typer.echo("press q in the window to quit")
    if view_mode == "overlay":
        typer.echo(
            "NOTE: overlay shows live RGB. It is not persisted, but it is not "
            "the skeleton-only privacy view -- use it for debugging, not the ward."
        )

    window = "ahfd -- " + ("overlay (RGB)" if view_mode == "overlay" else "skeleton only")
    fps_ema: float | None = None
    n = 0
    events_seen = 0
    last_alert: str | None = None

    try:
        for frame in src:
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

            metrics: dict[int, dict] = {}
            if machine is not None and extractor is not None and sink is not None:
                extractor.retain_only(tracker.live_ids)
                machine.retain_only(tracker.live_ids)
                for person in pose.people:
                    features = extractor.extract(person, pose.t)
                    if features is None:
                        continue
                    if person.track_id is not None:
                        metrics[person.track_id] = {
                            "state": machine.state_of(person.track_id),
                            "h_torso": features.h_torso,
                            "floor_spread": features.floor_spread,
                            "v_z": features.v_z,
                            "h_ankle_min": features.h_ankle_min,
                            "bed_risk": features.bed_risk,
                            "range_m": features.range_m,
                        }
                    event = machine.update(features)
                    if event is not None:
                        sink.emit(event)
                        events_seen += 1
                        if event.type in ("FALL_CONFIRMED", "PERSON_DOWN"):
                            last_alert = event.describe()

            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            if windowed:
                states = (
                    {p.track_id: machine.state_of(p.track_id) for p in pose.people if p.track_id is not None}
                    if machine is not None
                    else None
                )
                fps_show = fps_ema if cfg.view.show_fps else None
                metrics_show = metrics if cfg.view.show_metrics else None
                if view_mode == "overlay":
                    canvas = render_overlay(
                        frame,
                        pose,
                        min_keypoint_score=cfg.pose.min_keypoint_score,
                        states=states,
                        metrics=metrics_show,
                        alert=last_alert,
                        fps=fps_show,
                    )
                else:
                    canvas = render_skeleton(
                        pose,
                        min_keypoint_score=cfg.pose.min_keypoint_score,
                        show_ids=cfg.view.show_ids,
                        show_bbox=cfg.view.show_bbox,
                        states=states,
                        metrics=metrics_show,
                        fps=fps_show,
                    )
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        src.close()
        if sink is not None:
            sink.close()
        if windowed:
            cv2.destroyAllWindows()

    typer.echo("processed " + str(n) + " frames")
    if fps_ema is not None:
        typer.echo("mean pipeline rate " + format(fps_ema, ".1f") + " fps")
    if machine is not None:
        typer.echo("events emitted " + str(events_seen))


@app.command()
def extract(
    source: str = typer.Argument(..., help="Source URI: file://, seq://, webcam://, bag://"),
    out: Path = typer.Argument(..., help="Output tracks.jsonl path."),
    config: Path = typer.Option(None, help="Path to a YAML config (for the pose backend)."),
    max_frames: int = typer.Option(0, help="Stop after N frames. 0 = whole clip."),
) -> None:
    """Run pose once over a clip and write keypoints to tracks.jsonl.

    This is the only command that touches imagery for a recorded clip: it reads
    the frames, extracts keypoints, and discards the pixels. Everything after
    this -- replay, sweep, eval -- works on the keypoints alone, so it is fast,
    deterministic, and privacy-safe. Run it once per clip; it is the slow step.
    """
    from ahfd.capture import open_source
    from ahfd.io import TracksWriter
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.track import SimpleTracker

    cfg = load_config(config)

    src = open_source(source)
    typer.echo(
        "source:  " + source + "  "
        + str(src.meta.width) + "x" + str(src.meta.height)
        + " @ " + format(src.meta.fps, ".0f") + " fps"
    )
    typer.echo("loading pose model (the first run downloads weights)...")
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
    typer.echo("model:   " + estimator.name)

    writer = TracksWriter(out)
    n = 0
    try:
        for frame in src:
            pose = estimator.estimate(frame)
            pose = tracker.update(pose)
            if smoother is not None:
                smoother.retain_only(tracker.live_ids)
                pose = pose.with_people(
                    tuple(
                        p.with_keypoints(smoother.smooth(p.track_id, pose.t, p.keypoints))
                        for p in pose.people
                    )
                )
            writer.write(pose)
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        src.close()
        writer.close()

    typer.echo("wrote " + str(writer.count) + " frames to " + str(out))


@app.command()
def replay(
    tracks: Path = typer.Argument(..., help="A tracks.jsonl from `ahfd extract`."),
    config: Path = typer.Option(None, help="Path to a YAML config (thresholds, calibration)."),
    calibration: Path = typer.Option(None, help="Per-camera calibration YAML."),
    view: str = typer.Option("none", help="'skeleton' to watch it, 'none' for headless."),
) -> None:
    """Replay tracks.jsonl through detection. No model, no camera.

    Reads the keypoints extracted earlier and runs features -> state machine ->
    alerts. Runs far faster than real time and gives byte-identical events every
    run, which is what makes threshold tuning and golden regression tests
    possible. `--view skeleton` renders the stick figures back for the demo, so
    a canned clip can stand in for the camera on demo day.
    """
    from ahfd.capture.base import SourceMeta
    from ahfd.io import read_tracks, tracks_meta

    cfg = load_config(config)
    calib_path = calibration or cfg.calibration

    size = tracks_meta(tracks)
    if size is None:
        typer.echo("empty tracks file: " + str(tracks))
        raise typer.Exit(code=1)

    meta = SourceMeta(uri="tracks://" + str(tracks), width=size[0], height=size[1], fps=0.0)
    extractor, machine, sink, calib = _build_detection(cfg, calib_path, meta)
    typer.echo(
        "calib:   " + calib.camera_id
        + "  " + str(size[0]) + "x" + str(size[1])
        + "  zones " + str(len(calib.zones.zones))
    )

    render = view == "skeleton"
    if render:
        import cv2

        from ahfd.viz import render_skeleton

    events_seen = 0
    n = 0
    live_ids: set[int] = set()
    try:
        for pose in read_tracks(tracks):
            live_ids = {p.track_id for p in pose.people if p.track_id is not None}
            extractor.retain_only(live_ids)
            machine.retain_only(live_ids)
            for person in pose.people:
                features = extractor.extract(person, pose.t)
                if features is None:
                    continue
                event = machine.update(features)
                if event is not None:
                    sink.emit(event)
                    events_seen += 1
            if render:
                canvas = render_skeleton(pose, min_keypoint_score=cfg.pose.min_keypoint_score)
                cv2.imshow("ahfd -- replay (skeleton only)", canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            n += 1
    finally:
        sink.close()
        if render:
            cv2.destroyAllWindows()

    typer.echo("replayed " + str(n) + " frames, emitted " + str(events_seen) + " events")


@app.command()
def sweep(
    param: str = typer.Argument(..., help="Threshold to vary, e.g. detect.vz_trigger."),
    range_: str = typer.Option(..., "--range", help="lo:hi:step, e.g. -1.5:-0.5:0.1"),
    tracks: Path = typer.Option(..., help="Directory of <clip>.jsonl track files."),
    annotations: Path = typer.Option(..., help="Directory of <clip>.json ground truth."),
    config: Path = typer.Option(None, help="Base config."),
    calibration: Path = typer.Option(None, help="Per-camera calibration YAML."),
) -> None:
    """Sweep one threshold and print the recall vs false-alarms/hour curve.

    For each value it replays every track file through detection and scores the
    result, so the whole curve comes from the fast keypoint replay rather than
    re-running pose. The crossover point on this curve is how an operating point
    gets chosen, and the curve itself is the strongest single figure for the
    report.
    """
    from ahfd.capture.base import SourceMeta
    from ahfd.eval import GroundTruth, evaluate
    from ahfd.io import read_tracks, tracks_meta

    lo, hi, step = (float(x) for x in range_.split(":"))
    section, _, field = param.partition(".")
    if section != "detect":
        raise typer.BadParameter("only detect.* thresholds can be swept, got " + repr(param))

    base = load_config(config)
    calib_path = calibration or base.calibration

    clips = sorted(tracks.glob("*.jsonl"))
    if not clips:
        raise typer.BadParameter("no *.jsonl track files in " + str(tracks))

    typer.echo(param + "  |  recall  |  FA/hour  |  latency(s)")
    typer.echo("-" * 48)

    value = lo
    while value <= hi + 1e-9:
        cfg = base.model_copy(deep=True)
        if not hasattr(cfg.detect, field):
            raise typer.BadParameter("unknown detect threshold: " + repr(field))
        setattr(cfg.detect, field, value)

        pairs = []
        for clip in clips:
            ann = annotations / (clip.stem + ".json")
            if not ann.exists():
                continue
            truth = GroundTruth.load(ann)
            size = tracks_meta(clip)
            if size is None:
                continue
            meta = SourceMeta(uri=str(clip), width=size[0], height=size[1], fps=0.0)
            extractor, machine, _sink, _calib = _build_detection(cfg, calib_path, meta)

            events = _replay_events(clip, extractor, machine, read_tracks)
            pairs.append((truth, events))

        report = evaluate(pairs)
        typer.echo(
            format(value, "8.3f")
            + "  |  " + format(report.recall, "6.3f")
            + "  |  " + format(report.false_alarms_per_hour, "7.2f")
            + "  |  " + format(report.latency_median(), "6.1f")
        )
        value += step


def _replay_events(clip, extractor, machine, read_tracks):
    """Replay one track file to an in-memory list of PredictedEvents."""
    from ahfd.eval import PredictedEvent

    events = []
    for pose in read_tracks(clip):
        live = {p.track_id for p in pose.people if p.track_id is not None}
        extractor.retain_only(live)
        machine.retain_only(live)
        for person in pose.people:
            features = extractor.extract(person, pose.t)
            if features is None:
                continue
            event = machine.update(features)
            if event is not None:
                events.append(
                    PredictedEvent(
                        type=event.type,
                        t_alert=event.t_alert,
                        t_trigger=event.t_trigger,
                        track_id=event.track_id,
                        severity=event.severity,
                        zone=event.zone,
                        evidence=event.evidence,
                    )
                )
    return events


@app.command()
def bench(
    source: str = typer.Option("webcam://0", help="Frames to benchmark on."),
    frames: int = typer.Option(40, help="How many frames to capture and time."),
    backends: str = typer.Option(
        "rtmo,rtmpose", help="Comma-separated backends to compare."
    ),
    device: str = typer.Option("cpu", help="cpu | gpu | cuda"),
    runtime: str = typer.Option("onnxruntime", help="onnxruntime | openvino"),
) -> None:
    """Pose backend bake-off: time each backend on the same captured frames.

    Every backend sees an identical set of frames, so the comparison is fair.
    This answers the report's 'which pose model' question with numbers from
    this machine rather than from a datasheet.
    """
    from ahfd.capture import open_source
    from ahfd.config import PoseConfig
    from ahfd.pose import benchmark, build_estimator

    # Capture once, reuse for every backend.
    src = open_source(source)
    captured = []
    for frame in src:
        captured.append(frame)
        if len(captured) >= frames:
            break
    src.close()
    typer.echo("captured " + str(len(captured)) + " frames from " + source)
    typer.echo("")

    for name in [b.strip() for b in backends.split(",") if b.strip()]:
        try:
            cfg = PoseConfig(backend=name, device=device, runtime=runtime)
            estimator = build_estimator(cfg)
            result = benchmark(estimator, captured)
            typer.echo(result.line())
        except Exception as exc:  # noqa: BLE001 - report and continue
            typer.echo(name.ljust(20) + "FAILED: " + str(exc)[:100])


@app.command()
def calibrate_zones(
    calibration: Path = typer.Argument(..., help="Calibration YAML to add the zone(s) to."),
    source: str = typer.Option("rs://", help="Camera to grab a frame from (ignored with --frame)."),
    frame: Path = typer.Option(None, help="Click on this saved image instead of a live frame."),
) -> None:
    """Draw bed/zone polygons by clicking on a frame; write them into the calibration.

    Click the corners of the bed SURFACE (the mattress edges). Each click is
    back-projected to floor metres at the bed height you enter, using the camera
    geometry already in the calibration (run `ahfd calibrate` first). Keys in the
    window: left-click adds a point, 'u' undo, 'f' finish the polygon, 'q'/Esc
    cancel. You can add several zones in one run. This is the only zone step that
    needs a frame -- zones are used purely from metres afterwards.
    """
    import cv2
    import yaml

    from ahfd.geometry.calibration import load_calibration
    from ahfd.geometry.zones import Zone, polygon_from_pixels

    calib = load_calibration(calibration)
    ground = calib.ground

    win = "ahfd calibrate-zones"
    cv2.namedWindow(win)

    if frame is not None:
        img = cv2.imread(str(frame))
        if img is None:
            cv2.destroyAllWindows()
            raise typer.BadParameter("could not read image: " + str(frame))
    else:
        # Live preview: watch the feed and press SPACE to freeze the moment you
        # want to draw on (so the subject can be positioned and the image is
        # steady). The frozen frame is held in memory only and never written to
        # disk -- the privacy guard forbids saving imagery here (see
        # tests/test_privacy.py), and none is needed: you click, it measures, done.
        from ahfd.capture import open_source

        src = open_source(source)
        img = None
        typer.echo("live view -- SPACE to freeze the frame, 'q' to quit")
        try:
            for f in src:
                if f.bgr is None:
                    continue
                view = f.bgr.copy()
                cv2.putText(
                    view, "SPACE = freeze   q = quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow(win, view)
                key = cv2.waitKey(20) & 0xFF
                if key == ord(" "):
                    img = f.bgr.copy()
                    break
                if key in (ord("q"), 27):
                    break
        finally:
            src.close()
        if img is None:
            cv2.destroyAllWindows()
            raise typer.BadParameter("no frame frozen (quit before pressing SPACE).")

    h, w = img.shape[:2]
    k = ground.intrinsics
    if (w, h) != (k.width, k.height):
        cv2.destroyAllWindows()
        raise typer.BadParameter(
            "frame is " + str(w) + "x" + str(h) + " but the calibration is for "
            + str(k.width) + "x" + str(k.height) + "; use a frame at the "
            "calibrated resolution or the metres will be wrong."
        )

    data = yaml.safe_load(calibration.read_text(encoding="utf-8")) or {}
    data.setdefault("zones", [])

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    added = 0
    while True:
        clicks.clear()
        typer.echo("\nclick the zone corners in the window; 'f' finish, 'u' undo, 'q' cancel")
        cancelled = False
        while True:
            disp = img.copy()
            cv2.putText(
                disp, "click corners   f = finish   u = undo   q = cancel", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )
            for i, (x, y) in enumerate(clicks):
                cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
                if i:
                    cv2.line(disp, clicks[i - 1], clicks[i], (0, 255, 0), 2)
            if len(clicks) >= 3:
                cv2.line(disp, clicks[-1], clicks[0], (0, 200, 0), 1)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cancelled = True
                break
            if key == ord("u") and clicks:
                clicks.pop()
            if key == ord("f") and len(clicks) >= 3:
                break

        if cancelled or len(clicks) < 3:
            typer.echo("  zone cancelled")
        else:
            pts = list(clicks)
            name = typer.prompt("  zone name", default="bed_" + str(added + 1))
            kind = typer.prompt("  kind (bed/chair/floor/exclude)", default="bed")
            if kind in ("bed", "chair"):
                top_m = float(
                    typer.prompt(
                        "  surface height top_m in metres (floor->mattress top)",
                        default="0.4",
                    )
                )
                risk = typer.prompt("  risk_level (none/low/medium/high)", default="high")
                plane_z = top_m
            else:
                top_m, risk, plane_z = None, "unknown", 0.0
            try:
                poly = polygon_from_pixels(ground, pts, plane_z)
                Zone(name=name, kind=kind, polygon=poly, top_m=top_m, risk_level=risk)
            except ValueError as exc:
                typer.echo("  ! invalid zone, not added: " + str(exc))
            else:
                entry: dict = {"name": name, "kind": kind}
                if top_m is not None:
                    entry["top_m"] = top_m
                    entry["risk_level"] = risk
                entry["polygon"] = [[p[0], p[1]] for p in poly]
                data["zones"].append(entry)
                added += 1
                typer.echo(
                    "  added " + name + " (" + kind + ", " + str(len(poly)) + " points"
                    + (", top_m=" + str(top_m) + ", risk=" + risk if top_m is not None else "")
                    + ")"
                )

        if not typer.confirm("add another zone?", default=False):
            break

    cv2.destroyAllWindows()
    if added:
        calibration.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        typer.echo("\nwrote " + str(added) + " zone(s) to " + str(calibration))
    else:
        typer.echo("\nno zones added; calibration unchanged")


@app.command()
def copy_zones(
    from_: Path = typer.Option(..., "--from", help="Calibration to copy zones FROM."),
    to: list[Path] = typer.Option(
        ..., "--to", help="Calibration(s) to copy zones INTO (repeat --to for several)."
    ),
    append: bool = typer.Option(
        False, help="Add to the target's existing zones instead of replacing them."
    ),
) -> None:
    """Copy bed/zone polygons from one calibration into others -- draw once, share.

    Zones are stored in floor METRES, not pixels, so they describe the same
    physical beds no matter which camera sees them. That means one set of zones
    can be shared across every calibration of the SAME camera mount -- e.g. the
    colour (calib/d435i.yaml) and infrared (calib/d435i_ir.yaml) views of one
    D435i -- without redrawing.

    It does NOT make sense to copy zones to a camera in a DIFFERENT position (a
    webcam across the room): its floor frame is unrelated, so it needs its own
    zones. A mismatched mount height usually means exactly that, and the copy to
    that target is skipped with a warning.
    """
    import yaml

    src = yaml.safe_load(from_.read_text(encoding="utf-8")) or {}
    zones = src.get("zones") or []
    if not zones:
        raise typer.BadParameter("no zones in " + str(from_) + " to copy.")
    src_h = (src.get("camera") or {}).get("height_m")

    copied_to = 0
    for target in to:
        if target.resolve() == from_.resolve():
            continue
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        tgt_h = (data.get("camera") or {}).get("height_m")
        if src_h is not None and tgt_h is not None and abs(float(src_h) - float(tgt_h)) > 0.1:
            typer.echo(
                "  ! SKIP " + target.name + ": mount height " + str(tgt_h) + " m != "
                + str(src_h) + " m (looks like a different camera position -- it "
                "needs its own zones). Draw them with `ahfd calibrate-zones`."
            )
            continue
        existing = data.get("zones") or []
        data["zones"] = (existing + list(zones)) if append else list(zones)
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        copied_to += 1
        typer.echo(
            "  " + str(len(zones)) + " zone(s) -> " + str(target)
            + (" (appended)" if append else (" (replaced " + str(len(existing)) + ")"))
        )
    typer.echo("\ncopied into " + str(copied_to) + " calibration(s)")


@app.command()
def train_posture(
    calibration: Path = typer.Option(..., help="Calibration YAML (for the metric features)."),
    labels: Path = typer.Option(Path("data/postures"), help="Dir of posture-label JSONs."),
    tracks: Path = typer.Option(Path("data/tracks"), help="Dir of extracted <clip>.jsonl."),
    out: Path = typer.Option(None, help="Save the trained model here (.joblib), optional."),
    max_depth: int = typer.Option(5, help="Tree depth (small = interpretable)."),
) -> None:
    """Train an interpretable posture classifier from labelled clips.

    Reads posture labels + extracted tracks + calibration, builds a table of
    metric features -> posture, and fits a small decision tree -- reporting
    accuracy, a confusion matrix, and which features matter most.

    Retraining is just re-running this on more labelled+extracted clips; the
    code never changes, only the data grows. On a single recording session the
    accuracy is NOT meaningful (it overfits) -- this is a plumbing/coverage
    check until you have several people and sessions.
    """
    from collections import Counter

    from ahfd.geometry.calibration import load_calibration
    from ahfd.ml.posture import build_dataset, train

    calib = load_calibration(calibration)
    rows, labels_list, _groups, used, skipped = build_dataset(labels, tracks, calib)

    if skipped:
        typer.echo("WARNING: skipped label files that failed to parse (fix the JSON):")
        for name, why in skipped:
            typer.echo("  ! " + name + ": " + why)
        typer.echo("")

    typer.echo("clips used:")
    for clip, n in used:
        typer.echo("  " + clip.ljust(24) + str(n) + " labelled frames")
    if not rows:
        raise typer.BadParameter(
            "no labelled frames found. Need posture files in "
            + str(labels)
            + " with real segments AND matching tracks in "
            + str(tracks)
            + " (run `ahfd extract` on the labelled clips first)."
        )

    dist = Counter(labels_list)
    typer.echo("")
    typer.echo(
        "samples per posture: "
        + ", ".join(k + "=" + str(v) for k, v in sorted(dist.items()))
    )
    typer.echo("total samples: " + str(len(rows)))

    result = train(rows, labels_list, max_depth=max_depth)

    typer.echo("")
    if result.split_done:
        typer.echo(
            "train/test split: " + str(result.n_train) + " train, "
            + str(result.n_test) + " test"
        )
        typer.echo("TEST accuracy: " + format(result.accuracy, ".3f"))
    else:
        typer.echo(
            "too few samples for a held-out test -- trained on all, reporting "
            "TRAINING accuracy (not a real score)"
        )
        typer.echo("TRAINING accuracy: " + format(result.accuracy, ".3f"))

    typer.echo("")
    typer.echo("confusion (rows=true, cols=pred): " + "  ".join(result.classes))
    for cls, row in zip(result.classes, result.confusion):
        typer.echo("  " + cls.ljust(12) + " ".join(str(x).rjust(4) for x in row))

    typer.echo("")
    typer.echo("tree feature importances (what THIS tree split on -- noisy on small data):")
    for feat, imp in result.importances:
        if imp <= 0:
            continue
        typer.echo("  " + feat.ljust(18) + format(imp, ".3f") + " " + "#" * int(round(imp * 40)))

    # The univariate ranking is the honest "which joint distinguishes them"
    # answer: it scores each feature alone, so it is stable where the tree's
    # importances are not. Show the top handful with their per-class means so
    # the separation is legible, not just a score.
    typer.echo("")
    typer.echo("most discriminative features (ANOVA F-score, higher = separates postures better):")
    header = "  " + "feature".ljust(18) + "F".rjust(8) + "  MI".ljust(8)
    header += "".join(c[:8].rjust(10) for c in result.classes)
    typer.echo(header)
    for feat, fscore, miscore in result.separability[:8]:
        line = "  " + feat.ljust(18) + format(fscore, ".1f").rjust(8) + ("  " + format(miscore, ".2f")).ljust(8)
        for c in result.classes:
            m = result.class_means[c].get(feat, float("nan"))
            line += (format(m, ".2f") if m == m else "  --").rjust(10)
        typer.echo(line)
    typer.echo("  (last columns = mean value of that feature per posture -- read across to see the gap)")

    typer.echo("")
    typer.echo(result.report)
    typer.echo(
        "NOTE: on one session this OVERFITS -- the number is a plumbing check, "
        "not a real accuracy. Retrain on more people/sessions by labelling + "
        "extracting more clips and re-running this exact command."
    )

    if out:
        import joblib

        joblib.dump(result.model, out)
        typer.echo("saved model -> " + str(out))


@app.command()
def export_features(
    calibration: Path = typer.Option(..., help="Calibration YAML (for the metric features)."),
    tracks: Path = typer.Option(Path("data/tracks"), help="Dir of extracted <clip>.jsonl."),
    postures: Path = typer.Option(Path("data/postures"), help="Dir of posture labels (fills the posture column)."),
    out: Path = typer.Option(Path("data/features"), help="Dir to write <clip>.csv into."),
    clip: str = typer.Option(None, help="Export only this clip."),
    labelled_only: bool = typer.Option(False, help="Keep only rows inside a labelled segment."),
) -> None:
    """Extract and STORE the per-frame key-joint features from the tracks.

    Writes one CSV per clip: t, frame, posture (your label, or blank in a gap),
    then every feature the classifier uses -- torso_tilt, knee heights,
    floor_spread, and the rest. The features come from the tracks, so NO labels
    are needed; labels only fill the `posture` column where they exist. All
    frames by default; --labelled-only keeps just the labelled ones. Only
    coordinates are written -- no imagery.
    """
    import csv

    from ahfd.annotate import load_existing_segments
    from ahfd.features import FeatureExtractor
    from ahfd.geometry.calibration import load_calibration
    from ahfd.io import read_tracks
    from ahfd.ml.posture import FEATURES, _main_person, _posture_at, features_row

    calib = load_calibration(calibration)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["t", "frame", "posture"] + list(FEATURES)

    written = []
    for track_path in sorted(Path(tracks).glob("*.jsonl")):
        stem = track_path.stem
        if clip and stem != clip:
            continue

        seg_path = Path(postures) / (stem + ".json")
        segments: list[dict] = []
        if seg_path.exists():
            _cid, segments = load_existing_segments(seg_path)

        ext = FeatureExtractor(calib.ground, zones=calib.zones)
        rows = []
        for pose in read_tracks(track_path):
            person = _main_person(pose)
            if person is None:
                continue
            feats = ext.extract(person, pose.t)  # keeps velocity history moving
            posture = _posture_at(segments, pose.t) if segments else None
            if labelled_only and posture is None:
                continue
            base = {"t": round(pose.t, 3), "frame": pose.index, "posture": posture or ""}
            if feats is not None and feats.has_geometry():
                fr = features_row(feats, person, ext.ground)
                for f in FEATURES:
                    v = fr.get(f)
                    base[f] = "" if v is None else round(v, 4)
            else:
                for f in FEATURES:
                    base[f] = ""
            rows.append(base)

        csv_path = out / (stem + ".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        n_lab = sum(1 for r in rows if r["posture"])
        written.append((stem, len(rows), n_lab))

    if not written:
        raise typer.BadParameter("no track files found in " + str(tracks))
    typer.echo("wrote per-frame features to " + str(out) + " (" + str(len(FEATURES)) + " features/row):")
    for stem, n, n_lab in written:
        typer.echo("  " + stem.ljust(24) + str(n).rjust(6) + " rows  (" + str(n_lab) + " labelled)")


@app.command()
def posture_check(
    calibration: Path = typer.Option(..., help="Calibration YAML (for the metric features)."),
    postures: Path = typer.Option(Path("data/postures"), help="Dir of posture-label JSONs."),
    tracks: Path = typer.Option(Path("data/tracks"), help="Dir of extracted <clip>.jsonl."),
    clip: str = typer.Option(None, help="Check only this clip."),
) -> None:
    """See what posture the LIVE detector assigns on your recordings vs labels.

    Replays each labelled clip's extracted tracks through the exact same
    features + state machine the dashboard uses, and compares the assigned
    posture to your labels -- so you can check, with no camera, whether
    sitting/upright/etc. classify correctly. It uses each recording's own
    geometry, so it sidesteps a wrong live-camera height entirely.
    """
    from collections import Counter

    from ahfd.annotate import POSTURE_CLASSES, load_existing_segments
    from ahfd.detect import FallStateMachine
    from ahfd.features import FeatureExtractor
    from ahfd.geometry.calibration import load_calibration
    from ahfd.io import read_tracks

    calib = load_calibration(calibration)
    state_to_posture = {
        "UPRIGHT": "upright", "SITTING": "sitting", "IN_BED": "in_bed",
        "ON_GROUND": "on_ground", "FALLING": "on_ground",
    }
    cols = list(POSTURE_CLASSES) + ["other"]

    def posture_at(segs, t):
        for a, b, p in segs:
            if a <= t <= b:
                return p
        return None

    overall = {r: Counter() for r in POSTURE_CLASSES}
    per_clip = []
    for path in sorted(Path(postures).glob("*.json")):
        if clip and path.stem != clip:
            continue
        _cid, raw = load_existing_segments(path)
        track_path = Path(tracks) / (path.stem + ".jsonl")
        if not raw or not track_path.exists():
            continue
        segs = [(float(s["start_s"]), float(s["end_s"]), s["posture"]) for s in raw]
        ext = FeatureExtractor(calib.ground, zones=calib.zones)
        fsm = FallStateMachine()
        conf = {r: Counter() for r in POSTURE_CLASSES}
        for pose in read_tracks(track_path):
            ids = {p.track_id for p in pose.people if p.track_id is not None}
            for p in pose.people:
                if p.track_id is None:
                    continue
                f = ext.extract(p, pose.t)
                if f is None:
                    continue
                fsm.update(f)
                truth = posture_at(segs, pose.t)
                if truth in POSTURE_CLASSES and f.has_geometry():
                    got = state_to_posture.get(fsm.state_of(p.track_id), "other")
                    conf[truth][got] += 1
                    overall[truth][got] += 1
            ext.retain_only(ids)
            fsm.retain_only(ids)
        per_clip.append((path.stem, conf))

    if not per_clip:
        raise typer.BadParameter(
            "no labelled clips with tracks found -- extract them first."
        )

    typer.echo("posture-check (system vs your labels) on " + str(len(per_clip)) + " clip(s):\n")
    for cid, conf in per_clip:
        parts = []
        for r in POSTURE_CLASSES:
            n = sum(conf[r].values())
            if n:
                parts.append(r + ":" + format(100 * conf[r][r] / n, ".0f") + "%")
        typer.echo("  " + cid.ljust(22) + "  ".join(parts))

    typer.echo("\noverall confusion (rows = your label, cols = system %):")
    typer.echo("  " + "".ljust(11) + "".join(c[:9].rjust(10) for c in cols))
    for r in POSTURE_CLASSES:
        tot = sum(overall[r].values()) or 1
        typer.echo(
            "  " + r.ljust(11)
            + "".join(format(100 * overall[r][c] / tot, ".0f").rjust(9) + "%" for c in cols)
        )
    typer.echo("  (each row sums to 100%; the diagonal is correct)")


@app.command()
def label_review(
    postures: Path = typer.Option(Path("data/postures"), help="Dir of posture-label JSONs."),
    posture: str = typer.Option(None, help="Show only segments of this posture (upright/sitting/in_bed/on_ground)."),
    clip: str = typer.Option(None, help="Show only this clip's segments."),
) -> None:
    """Review your posture labels: per-posture coverage, or filter to check one.

    With no options: a summary of how much of each posture you've labelled (so
    you can see, e.g., how many seconds of `sitting` exist and in which clips).
    `--posture sitting` lists every sitting segment across all clips; `--clip X`
    lists one clip's segments. Placeholder rows (start==end) are ignored.
    """
    from collections import defaultdict

    from ahfd.annotate import POSTURE_CLASSES, load_existing_segments

    if posture and posture not in POSTURE_CLASSES:
        raise typer.BadParameter(
            "posture must be one of: " + ", ".join(POSTURE_CLASSES)
        )

    # clip -> list of (start, end, posture)
    by_clip: dict[str, list[tuple[float, float, str]]] = {}
    for path in sorted(Path(postures).glob("*.json")):
        _cid, segs = load_existing_segments(path)
        if segs:
            by_clip[path.stem] = [
                (float(s["start_s"]), float(s["end_s"]), s["posture"]) for s in segs
            ]

    if not by_clip:
        typer.echo("no labelled segments found in " + str(postures))
        return

    def dur(a, b):
        return b - a

    # --- filter to one clip ------------------------------------------------
    if clip:
        segs = by_clip.get(clip)
        if not segs:
            raise typer.BadParameter("no real segments for clip " + repr(clip))
        typer.echo(clip + ":")
        for a, b, p in sorted(segs):
            typer.echo("  %6.1f - %6.1f s  (%5.1fs)  %s" % (a, b, dur(a, b), p))
        return

    # --- filter to one posture across all clips ----------------------------
    if posture:
        total = 0.0
        n = 0
        typer.echo(posture + " segments across all clips:")
        for cid in sorted(by_clip):
            for a, b, p in sorted(by_clip[cid]):
                if p == posture:
                    typer.echo("  %-24s %6.1f - %6.1f s  (%5.1fs)" % (cid, a, b, dur(a, b)))
                    total += dur(a, b)
                    n += 1
        typer.echo("\n%d segment(s), %.1f s total of %s" % (n, total, posture))
        return

    # --- default: coverage summary ----------------------------------------
    per_posture_secs: dict[str, float] = defaultdict(float)
    per_posture_segs: dict[str, int] = defaultdict(int)
    per_posture_clips: dict[str, set] = defaultdict(set)
    for cid, segs in by_clip.items():
        for a, b, p in segs:
            per_posture_secs[p] += dur(a, b)
            per_posture_segs[p] += 1
            per_posture_clips[p].add(cid)

    typer.echo("posture coverage (across " + str(len(by_clip)) + " labelled clips):")
    typer.echo("  " + "posture".ljust(12) + "segs".rjust(6) + "seconds".rjust(10) + "   clips")
    for p in POSTURE_CLASSES:
        typer.echo(
            "  " + p.ljust(12) + str(per_posture_segs[p]).rjust(6)
            + format(per_posture_secs[p], ".1f").rjust(10)
            + "   " + str(len(per_posture_clips[p]))
        )
    unknown = set(per_posture_secs) - set(POSTURE_CLASSES)
    for p in sorted(unknown):
        typer.echo("  ! " + p.ljust(10) + " (not a valid posture class) "
                   + str(per_posture_segs[p]) + " segs")

    typer.echo("\nper clip:")
    for cid in sorted(by_clip):
        counts: dict[str, float] = defaultdict(float)
        for a, b, p in by_clip[cid]:
            counts[p] += dur(a, b)
        summary = "  ".join(
            k + ":" + format(v, ".0f") + "s" for k, v in sorted(counts.items())
        )
        typer.echo("  " + cid.ljust(24) + summary)


@app.command()
def derive_falls(
    postures: Path = typer.Option(Path("data/postures"), help="Dir of posture-label JSONs."),
    out: Path = typer.Option(Path("data/annotations"), help="Dir to write fall ground-truth into."),
    prune_placeholders: bool = typer.Option(
        False,
        "--prune-placeholders",
        help="Delete stale t_impact:0.0 placeholder annotations for unlabelled fall clips.",
    ),
) -> None:
    """Derive fall ground-truth from posture labels (no separate impact labelling).

    A fall is an upright/sitting -> on_ground transition, so the posture
    segments already contain the falls: t_impact is the start of each on_ground
    hold, t_start the end of the posture before it. Negatives (neg_* / bedexit_*)
    are written as empty-falls from their duration even without posture labels.
    Unlabelled fall clips are left out of the ground truth until you label them.
    Re-run this after every labelling session -- it keeps the falls in sync.
    """
    from ahfd.annotate import derive_annotations

    written, unlabelled, pruned = derive_annotations(
        postures, out, prune_placeholders=prune_placeholders
    )

    if not written:
        raise typer.BadParameter(
            "no annotations written from " + str(postures)
            + " -- label some clips first with `ahfd label-postures`."
        )

    typer.echo("wrote fall ground-truth to " + str(out) + ":")
    n_falls = 0
    for clip, n in written:
        n_falls += n
        tag = (str(n) + " fall" + ("s" if n != 1 else "")) if n else "negative (0 falls)"
        typer.echo("  " + clip.ljust(26) + tag)
    typer.echo("\n" + str(len(written)) + " clip(s), " + str(n_falls) + " fall(s) total")

    if unlabelled:
        typer.echo(
            "\nUNLABELLED fall clips (no posture labels yet -- NOT in the ground "
            "truth, do not sweep them until labelled):\n  " + ", ".join(unlabelled)
        )
        if not prune_placeholders:
            typer.echo(
                "  (any leftover t_impact:0.0 placeholders remain; re-run with "
                "--prune-placeholders to delete them)"
            )
    if pruned:
        typer.echo("\npruned " + str(len(pruned)) + " stale placeholder annotation(s)")

    typer.echo(
        "\nnext: sweep a threshold against these, e.g.\n"
        "  ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 "
        "--tracks data/tracks --annotations " + str(out)
    )


@app.command()
def label_postures(
    clip: Path = typer.Argument(..., help="Video clip to label, e.g. data/clips/fall_slump_02.mp4."),
    out: Path = typer.Option(
        None, help="Posture JSON to write. Default: data/postures/<clip>.json."
    ),
) -> None:
    """Scrub a clip and mark posture segments -- a little video labeller.

    Opens the clip in a window with a timeline. Scrub with a/d (+/-1s), ,/.
    (+/-1 frame), [ / ] (+/-5s). Press 's' (or SPACE) to mark the START of a
    hold, scrub to its end, press 'f', then a number to pick the posture
    (1 upright, 2 sitting, 3 in_bed, 4 on_ground). 'u' undoes, 'w' saves, 'q'
    saves and quits. Leave gaps between segments for transitions -- they are
    excluded from training on purpose.

    Only the label JSON is written; no frame is ever saved. Run it on the
    laptop (it needs a display), not the headless Jetson.
    """
    from ahfd.annotate import run_labeler

    if not clip.exists():
        raise typer.BadParameter("clip not found: " + str(clip))
    out_path = out or (Path("data/postures") / (clip.stem + ".json"))

    typer.echo("labelling " + clip.stem + "  ->  " + str(out_path))
    typer.echo("  s/SPACE=start  f=end  1-4=posture  u=undo  w=save  q=save+quit")
    typer.echo("  click/drag the timeline to seek   c=cancel mark   r=remove segment here")
    n = run_labeler(clip, out_path)
    typer.echo("saved " + str(n) + " segment(s) to " + str(out_path))


@app.command()
def compare_posture(
    calibration: Path = typer.Option(..., help="Calibration YAML (for the metric features)."),
    labels: Path = typer.Option(Path("data/postures"), help="Dir of posture-label JSONs."),
    tracks: Path = typer.Option(Path("data/tracks"), help="Dir of extracted <clip>.jsonl."),
    by: str = typer.Option(
        "clip",
        help="Leave-one-out unit: 'clip' (cross-scenario, one person) or 'person' "
        "(train on N-1 people, test on the held-out one -- the real generalisation test).",
    ),
    show_rules: bool = typer.Option(
        True, "--show-rules/--no-show-rules", help="Print the learned tree thresholds."
    ),
) -> None:
    """Train the posture classifier several ways and rank them, honestly.

    Compares flat 4-class models (tree / forest / logistic / naive-Bayes), the
    coarse-to-fine cascade (upright -> sitting-vs-down -> ground-vs-bed) with a
    tree or logistic model at each node, and a learning-free threshold rule.

    Scoring is LEAVE-ONE-OUT by clip (default) or by person (`--by person`): each
    held-out unit is predicted by a model that never saw it. A random frame split
    would leak near-duplicate frames and inflate the score. `--by person` is the
    real generalisation test -- it needs at least two people labelled.
    """
    if by not in ("clip", "person"):
        raise typer.BadParameter("--by must be 'clip' or 'person'")

    from ahfd.geometry.calibration import load_calibration
    from ahfd.ml.compare import compare
    from ahfd.ml.posture import build_dataset

    calib = load_calibration(calibration)
    rows, labels_list, groups, used, skipped = build_dataset(labels, tracks, calib)

    if by == "person":
        def _person_of(clip_id):
            tail = clip_id.rsplit("_", 1)[-1]
            return "person_" + tail if tail.isdigit() else "person_?"

        groups = [_person_of(g) for g in groups]
    unit = by

    if skipped:
        typer.echo("WARNING: skipped label files that failed to parse (fix the JSON):")
        for name, why in skipped:
            typer.echo("  ! " + name + ": " + why)
        typer.echo("")

    typer.echo("clips used:")
    for clip, n in used:
        typer.echo("  " + clip.ljust(24) + str(n) + " labelled frames")
    n_groups = len(set(groups))
    typer.echo("\nleave-one-" + unit + "-out: " + str(n_groups) + " " + unit + "(s) = " + str(n_groups) + " fold(s)")
    if not rows or n_groups < 2:
        hint = (
            " -- only person 01 is labelled. Label + extract a few clips for "
            "persons 02-04, then re-run with `--by person`."
            if unit == "person"
            else ". Label + extract more clips first."
        )
        raise typer.BadParameter(
            "need at least two " + unit + "s to leave one out; found "
            + str(n_groups) + hint
        )

    from collections import Counter

    dist = Counter(labels_list)
    typer.echo(
        "\nsamples per posture: "
        + ", ".join(k + "=" + str(v) for k, v in sorted(dist.items()))
    )

    result = compare(rows, labels_list, groups)

    typer.echo("")
    typer.echo(
        "leave-one-clip-out ranking (macro-F1 = balanced across postures; "
        "bal-acc = mean recall):"
    )
    header = "  " + "model".ljust(18) + "macro-F1".rjust(9) + "bal-acc".rjust(9) + "  "
    header += "".join(("R:" + c[:6]).rjust(10) for c in result.classes)
    typer.echo(header)
    for s in result.scores:
        line = "  " + s.name.ljust(18)
        line += format(s.macro_f1, ".3f").rjust(9) + format(s.balanced_accuracy, ".3f").rjust(9) + "  "
        line += "".join(format(s.per_class_recall[c], ".2f").rjust(10) for c in result.classes)
        typer.echo(line)
    typer.echo("  (R:<posture> = recall = of all true frames of that posture, fraction caught)")

    best = result.scores[0]
    typer.echo("")
    typer.echo("BEST: " + best.name + "  -- confusion (rows=true, cols=pred): " + "  ".join(result.classes))
    for cls, row in zip(result.classes, best.confusion):
        typer.echo("  " + cls.ljust(12) + " ".join(str(x).rjust(5) for x in row))
    typer.echo("")
    typer.echo("per-clip accuracy for " + best.name + ":")
    for clip, acc in best.per_clip_acc:
        typer.echo("  " + clip.ljust(24) + format(acc, ".3f"))

    if show_rules:
        typer.echo("")
        typer.echo("how a tree decides -- learned thresholds (depth-3 flat tree on all data):")
        typer.echo(result.tree_rules)

    typer.echo(
        "NOTE: still ONE person/camera -- even leave-one-clip-out mostly tests "
        "cross-scenario, not cross-person. Label persons 02-04, extract, re-run: "
        "this becomes leave-one-person-out with no code change."
    )


@app.command()
def dashboard(
    source: str = typer.Option(None, help="Source URI. Defaults to the config's source."),
    config: Path = typer.Option(None, help="Path to a YAML config."),
    calibration: Path = typer.Option(None, help="Per-camera calibration (for detection)."),
    backend: str = typer.Option(None, help="Override the pose backend: rtmo | rtmpose | yolo."),
    detect: bool = typer.Option(
        None,
        "--detect/--no-detect",
        help="Turn fall detection on or off. On needs a calibration matching "
        "the capture resolution; without it there are no postures, only tracks.",
    ),
    host: str = typer.Option(None, help="Bind address. Default 127.0.0.1 (localhost)."),
    port: int = typer.Option(None, help="Port. Default 8000."),
    rgb: bool = typer.Option(
        False,
        "--rgb",
        help="Show live RGB video instead of skeleton-only. Reverses the ward "
        "privacy stance -- needs AH/DPO sign-off before real use.",
    ),
    allow_rgb: bool = typer.Option(
        False,
        "--allow-rgb",
        help="Start skeleton-only but let the page switch to RGB (virtual "
        "nursing). Same sign-off as --rgb; the difference is the default.",
    ),
) -> None:
    """Serve the nurse dashboard: live view, per-person state, alert log.

    One pipeline thread produces frames; the web server only forwards them, so
    extra viewers cost nothing and there is no per-request encoding. Skeleton-
    only by default; --rgb (or dashboard.show_rgb in config) shows video.

    --source and --backend set the *initial* camera and model; both can then be
    changed from the page (see dashboard.sources in the config). Each camera in
    dashboard.sources may carry its own `calibration:`, so switching cameras in
    the picker switches the calibration too.

    With no --config this loads configs/dashboard.yaml (detection on, a picker
    for the webcam and the RealSense), not the run default.
    """
    from ahfd.config import DASHBOARD_CONFIG_PATH
    from ahfd.dashboard import DashboardController, DashboardServer, DashboardState

    # Bare `ahfd dashboard` loads the dashboard default (detection on, several
    # cameras) rather than the run default -- see DASHBOARD_CONFIG_PATH.
    cfg = load_config(config or DASHBOARD_CONFIG_PATH)
    if backend:
        cfg.pose.backend = backend  # CLI override -- swap models without editing the config
    if detect is not None:
        cfg.detect.enabled = detect
    uri = source or cfg.source
    calib_path = calibration or cfg.calibration
    bind_host = host or cfg.dashboard.host
    bind_port = port or cfg.dashboard.port
    show_rgb = rgb or cfg.dashboard.show_rgb
    rgb_allowed = show_rgb or allow_rgb or cfg.dashboard.allow_rgb

    if show_rgb:
        typer.echo(
            "WARNING: RGB view is ON. Live video is shown (not stored). This "
            "reverses the skeleton-only privacy stance -- confirm AH/DPO approval."
        )

    state = DashboardState()
    controller = DashboardController(
        cfg,
        calib_path,
        state,
        source=uri,
        show_rgb=show_rgb,
        # Turning RGB off from the page is always allowed; turning it on takes
        # --rgb, --allow-rgb, or the matching config flag.
        rgb_authorised=rgb_allowed,
        log=typer.echo,
    )
    server = DashboardServer(
        state, host=bind_host, port=bind_port, controller=controller
    )

    typer.echo(
        "source:  "
        + uri
        + (
            "  [RGB]"
            if show_rgb
            else ("  [skeleton only, RGB allowed]" if rgb_allowed else "  [skeleton only]")
        )
    )
    typer.echo(
        "detect:  "
        + (
            "on -- calib " + str(calib_path)
            if cfg.detect.enabled
            else "off -- pose and tracking only, no posture states"
        )
    )
    typer.echo("serving: http://" + bind_host + ":" + str(bind_port) + "  (Ctrl+C to stop)")
    typer.echo("         camera and pose model can be changed from the page")

    controller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nstopping...")
    finally:
        server.shutdown()
        controller.stop()


@app.command()
def record(
    out: Path = typer.Argument(..., help="Output .mp4 path for the raw recording."),
    source: str = typer.Option("webcam://0", help="Source URI to record."),
    seconds: float = typer.Option(0.0, help="Auto-stop after N seconds. 0 = until you press q."),
    consent: bool = typer.Option(
        False,
        "--i-understand-raw-capture",
        help="Required. Confirms this session is consented raw capture.",
    ),
) -> None:
    """Record raw video for a consented staged-fall session.

    This is the ONE tool that writes video to disk, and it is only for staged
    sessions with volunteers who have consented -- never patients, never a live
    ward. It records on start and stops on 'q' or after --seconds; a red banner
    is burned into every frame so the recording is never invisible.

    The intended flow is: record here, run `ahfd extract` on the file to get
    keypoints, then DELETE the video and keep only the tracks.jsonl. The footage
    is scaffolding for tuning, not something to retain.

    Requires the explicit --i-understand-raw-capture flag: invoking this command
    with that flag is the deliberate, informed intent the privacy gate exists to
    check. (The stricter triple-switch gate stays on `run`/`dashboard`, where raw
    capture would be an accident rather than the whole point.)
    """
    import os

    import cv2

    from ahfd.capture import open_source
    from ahfd.debug import RawRecorder
    from ahfd.privacy import ENV_VAR

    if not consent:
        raise typer.BadParameter(
            "raw recording is off unless you pass --i-understand-raw-capture. "
            "This tool writes video to disk; use it only for consented staged "
            "sessions with volunteers, never patients or a live ward."
        )

    # The explicit flag IS the consent, so satisfy the gate here rather than
    # making the user also juggle an env var for a tool whose only job is to
    # record. The banner and the deliberate flag keep it non-accidental.
    os.environ[ENV_VAR] = "1"

    src = open_source(source)
    typer.echo(
        "RECORDING (raw video) from " + source + " -> " + str(out)
        + "  " + str(src.meta.width) + "x" + str(src.meta.height)
    )
    typer.echo("press q in the window to stop. Extract keypoints, then delete this file.")

    recorder = RawRecorder(
        out, src.meta.width, src.meta.height, src.meta.fps,
        config_flag=True, cli_flag=True,
    )
    window = "ahfd RECORDING -- raw video"
    try:
        for frame in src:
            if frame.bgr is None:
                continue
            recorder.write(frame.bgr)
            cv2.imshow(window, frame.bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            if seconds and frame.t >= seconds:
                break
    finally:
        src.close()
        recorder.close()
        cv2.destroyAllWindows()

    typer.echo("saved " + str(recorder.count) + " frames to " + str(out))
    typer.echo("next: ahfd extract file://" + str(out) + " data/tracks/<clip>.jsonl  (then delete the .mp4)")


@app.command(name="eval")
def eval_cmd(
    annotations: Path = typer.Argument(
        ..., help="Directory of <clip>.json ground-truth files."
    ),
    events: Path = typer.Argument(
        ..., help="Directory of <clip>.jsonl event logs."
    ),
    out: Path = typer.Option(None, help="Write the Markdown report here."),
    pre_s: float = typer.Option(2.0, help="Match window before impact."),
    post_s: float = typer.Option(30.0, help="Match window after impact."),
) -> None:
    """Score event logs against ground truth: recall, false alarms/hour, latency.

    Pairs each <clip>.json in the annotations directory with <clip>.jsonl in
    the events directory. A clip with annotations but no event log is scored as
    if the system emitted nothing -- a missed fall is not silently dropped.
    """
    from ahfd.eval import GroundTruth, evaluate, load_events, render_markdown

    pairs = []
    for ann_path in sorted(annotations.glob("*.json")):
        truth = GroundTruth.load(ann_path)
        event_path = events / (ann_path.stem + ".jsonl")
        clip_events = load_events(event_path) if event_path.exists() else []
        pairs.append((truth, clip_events))

    if not pairs:
        raise typer.BadParameter("no <clip>.json annotations found in " + str(annotations))

    report = evaluate(pairs, pre_s=pre_s, post_s=post_s)
    markdown = render_markdown(report)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        typer.echo("wrote " + str(out))
    else:
        typer.echo(markdown)


def _check_calibration_resolution(calib, meta) -> None:
    """Refuse a calibration whose resolution does not match the stream.

    Intrinsics are in pixels, so they only describe the resolution they were
    measured at. Feed 1080p intrinsics to a 640x480 stream and every ray is
    computed from the wrong focal length and principal point -- which yields
    metric heights that are confidently, invisibly wrong. The skeleton still
    looks perfect on screen, so nothing about the display hints at a problem;
    only the numbers are broken. Worth an error rather than a warning.
    """
    k = calib.ground.intrinsics
    if (k.width, k.height) != (meta.width, meta.height):
        raise typer.BadParameter(
            "calibration "
            + repr(calib.camera_id)
            + " is for "
            + str(k.width)
            + "x"
            + str(k.height)
            + " but the source is "
            + str(meta.width)
            + "x"
            + str(meta.height)
            + ". Intrinsics are resolution-specific, so this would give "
            "plausible-looking but wrong metric heights. Either run the "
            "source at the calibrated resolution, or write a calibration for "
            "this one."
        )


@app.command()
def level(
    source: str = typer.Option("rs://", help="Source with an IMU (the D435i)."),
    target_pitch: float = typer.Option(
        20.0, help="Desired downtilt, for the on-screen guidance."
    ),
) -> None:
    """Live camera pitch/roll from the IMU -- for aiming the mount.

    Watch the numbers update as you tilt the camera: set pitch to your target
    downtilt (~20 deg for the ward geometry) and roll near 0 (level). Needs the
    RealSense IMU; a plain webcam has no IMU and cannot report an angle.

    Ctrl+C to stop. Nothing is written -- this is a read-only aiming aid.
    """
    import numpy as np

    from ahfd.capture import open_source
    from ahfd.geometry.ground import GroundPlane

    src = open_source(source)
    typer.echo("aiming aid -- target pitch " + format(target_pitch, ".0f") + " deg, roll 0. Ctrl+C to stop.")

    seen_imu = False
    try:
        for frame in src:
            if frame.gravity is None:
                # First frame without gravity: this source has no IMU.
                if not seen_imu:
                    typer.echo(
                        "this source reports no IMU gravity -- angle readout "
                        "needs the RealSense (rs://). A webcam has no IMU."
                    )
                    break
                continue
            seen_imu = True
            pitch, roll = GroundPlane.pitch_roll_from_gravity(np.asarray(frame.gravity))
            level_hint = "LEVEL" if abs(roll) < 1.5 else ("tilt " + ("right" if roll > 0 else "left"))
            pitch_hint = (
                "on target"
                if abs(pitch - target_pitch) < 2.0
                else ("aim down" if pitch < target_pitch else "aim up")
            )
            # Overwrite one line in place.
            print(
                "\rpitch {:6.1f} deg ({:<8})  roll {:6.1f} deg ({:<10})".format(
                    pitch, pitch_hint, roll, level_hint
                ),
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        print()  # newline after the in-place line


@app.command()
def info() -> None:
    """Report versions and whether a RealSense is actually present.

    Worth having as a command rather than a note in the README: the camera not
    enumerating has already cost this project time, and the failure looks
    identical whether the cause is the cable, the port or the driver.
    """
    import sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    typer.echo("python      " + sys.version.split()[0])

    # (import name, distribution name). Some packages -- rtmlib is one -- do not
    # expose __version__ on the module, so the version is read from the
    # installed package metadata instead, which every package has.
    packages = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("onnxruntime", "onnxruntime"),
        ("rtmlib", "rtmlib"),
    ]
    for import_name, dist_name in packages:
        try:
            __import__(import_name)
        except ImportError:
            typer.echo(import_name.ljust(11) + " NOT INSTALLED")
            continue
        try:
            ver = pkg_version(dist_name)
        except PackageNotFoundError:
            ver = "(installed)"
        typer.echo(import_name.ljust(11) + " " + ver)

    # Same probe the dashboard's camera picker uses, so the two can never
    # disagree about what is plugged in.
    from ahfd.capture import probe_realsense

    probe = probe_realsense()
    if not probe.installed:
        typer.echo("pyrealsense2 NOT INSTALLED (install the 'realsense' extra)")
        return

    typer.echo("pyrealsense " + str(probe.version))
    if probe.error:
        typer.echo("  enumeration failed: " + probe.error)
        return

    typer.echo("realsense devices: " + str(len(probe.devices)))

    if not probe.devices:
        typer.echo("  none found -- check the cable is USB 3 and the port is host-mode")
        return

    for d in probe.devices:
        typer.echo("  " + d.name + "  usb " + d.usb)
        # A D435i on a USB 2 link silently loses stream profiles rather than
        # erroring, so say so plainly.
        if d.usb2:
            typer.echo(
                "  WARNING: negotiated USB " + d.usb + " -- depth+colour at 30 fps "
                "will not fit. Use a USB 3 cable, no passive extension."
            )


def _write_calibration_yaml(
    path, camera_id, intrinsics, height_m, gravity=None, pitch_deg=None, roll_deg=0.0
):
    """Write a calibration YAML that load_calibration reads back.

    Prefers an IMU gravity vector (D435i) so tilt cannot go stale on a sagging
    mount; falls back to explicit pitch/roll for a plain camera.
    """
    import yaml

    camera: dict = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": round(intrinsics.fx, 3),
        "fy": round(intrinsics.fy, 3),
        "cx": round(intrinsics.cx, 3),
        "cy": round(intrinsics.cy, 3),
        "height_m": round(float(height_m), 3),
    }
    if gravity is not None:
        camera["gravity"] = [round(float(v), 5) for v in gravity]
    else:
        camera["pitch_deg"] = round(float(pitch_deg), 2)
        camera["roll_deg"] = round(float(roll_deg), 2)

    doc = {
        "camera_id": camera_id,
        "camera": camera,
        "zones": [],  # add beds/floor next -- see calib/example_ward6.yaml
        "notes": "Auto-generated by `ahfd calibrate`. Add zones before detection.",
    }
    from pathlib import Path as _P

    _P(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)


@app.command()
def calibrate(
    out: Path = typer.Argument(..., help="Output calibration YAML path."),
    source: str = typer.Option("rs://", help="Source to calibrate (rs:// for the D435i)."),
    height: float = typer.Option(..., help="Lens height above the floor, in metres (measure it)."),
    camera_id: str = typer.Option(None, help="Name for this camera position."),
    pitch: float = typer.Option(None, help="Downtilt in degrees, if the source has no IMU."),
    roll: float = typer.Option(0.0, help="Roll in degrees (webcam only)."),
    hfov: float = typer.Option(69.4, help="Horizontal FOV, if the source has no intrinsics."),
    vfov: float = typer.Option(42.5, help="Vertical FOV, if the source has no intrinsics."),
) -> None:
    """Capture a camera's intrinsics + tilt + height into a calibration file.

    On the D435i this is nearly automatic: the IMU gives the tilt and the device
    gives the true intrinsics, so you only supply the measured height. On a plain
    webcam there is no IMU or real lens data, so you pass --pitch and the FOV.

    Height is the one thing no sensor provides -- measure the lens height above
    the floor with a tape. After this, add the bed zones (see
    calib/example_ward6.yaml) before turning detection on.
    """
    import numpy as np

    from ahfd.capture import open_source
    from ahfd.types import Intrinsics

    src = open_source(source)
    cam_id = camera_id or Path(out).stem

    # Grab the first frame that carries what we need.
    frame = None
    for i, f in enumerate(src):
        frame = f
        if i >= 5:  # let auto-exposure / the IMU settle a few frames
            break
    src.close()
    if frame is None:
        raise typer.BadParameter("no frames from " + source)

    intrinsics = frame.intrinsics or Intrinsics.from_hfov(
        frame.shape[1], frame.shape[0], hfov_deg=hfov, vfov_deg=vfov
    )

    # Report the IMU-derived tilt if a gravity vector is available, as a
    # cross-check regardless of which source we actually use.
    imu_pitch = imu_roll = None
    if frame.gravity is not None:
        from ahfd.geometry.ground import GroundPlane

        gp_imu = GroundPlane.from_gravity(intrinsics, height, np.asarray(frame.gravity))
        imu_pitch, imu_roll = gp_imu.pitch_deg, gp_imu.roll_deg
        typer.echo(
            "IMU reads: pitch " + format(imu_pitch, ".1f")
            + " deg, roll " + format(imu_roll, ".1f") + " deg"
        )

    if pitch is not None:
        # A supplied angle is a deliberate measurement and wins over the IMU.
        _write_calibration_yaml(
            out, cam_id, intrinsics, height, pitch_deg=pitch, roll_deg=roll
        )
        typer.echo("using your --pitch " + format(pitch, ".1f") + " deg (measurement overrides IMU)")
    elif frame.gravity is not None:
        _write_calibration_yaml(out, cam_id, intrinsics, height, gravity=frame.gravity)
        typer.echo("using the IMU tilt (pass --pitch to override with your own measurement)")
    else:
        raise typer.BadParameter(
            "no --pitch given and this source has no IMU. Provide --pitch "
            "(downtilt in degrees); a camera looking slightly down might be ~20."
        )

    typer.echo(
        "wrote " + str(out) + "  (" + str(intrinsics.width) + "x"
        + str(intrinsics.height) + ", height " + format(height, ".2f") + " m)"
    )
    typer.echo("next: add bed zones to it (see calib/example_ward6.yaml), then run with --calibration " + str(out))


if __name__ == "__main__":
    app()
