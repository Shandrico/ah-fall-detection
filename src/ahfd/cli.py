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

            if machine is not None and extractor is not None and sink is not None:
                extractor.retain_only(tracker.live_ids)
                machine.retain_only(tracker.live_ids)
                for person in pose.people:
                    features = extractor.extract(person, pose.t)
                    if features is None:
                        continue
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
                if view_mode == "overlay":
                    canvas = render_overlay(
                        frame,
                        pose,
                        min_keypoint_score=cfg.pose.min_keypoint_score,
                        states=states,
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
def dashboard(
    source: str = typer.Option(None, help="Source URI. Defaults to the config's source."),
    config: Path = typer.Option(None, help="Path to a YAML config."),
    calibration: Path = typer.Option(None, help="Per-camera calibration (for detection)."),
    host: str = typer.Option(None, help="Bind address. Default 127.0.0.1 (localhost)."),
    port: int = typer.Option(None, help="Port. Default 8000."),
    rgb: bool = typer.Option(
        False,
        "--rgb",
        help="Show live RGB video instead of skeleton-only. Reverses the ward "
        "privacy stance -- needs AH/DPO sign-off before real use.",
    ),
) -> None:
    """Serve the nurse dashboard: live view, per-person state, alert log.

    One pipeline thread produces frames; the web server only forwards them, so
    extra viewers cost nothing and there is no per-request encoding. Skeleton-
    only by default; --rgb (or dashboard.show_rgb in config) shows video.
    """
    from ahfd.dashboard import DashboardServer, DashboardState, PipelineRunner

    cfg = load_config(config)
    uri = source or cfg.source
    calib_path = calibration or cfg.calibration
    bind_host = host or cfg.dashboard.host
    bind_port = port or cfg.dashboard.port
    show_rgb = rgb or cfg.dashboard.show_rgb

    if show_rgb:
        typer.echo(
            "WARNING: RGB view is ON. Live video is shown (not stored). This "
            "reverses the skeleton-only privacy stance -- confirm AH/DPO approval."
        )

    state = DashboardState()
    runner = PipelineRunner(uri, cfg, calib_path, state, show_rgb=show_rgb)
    server = DashboardServer(state, host=bind_host, port=bind_port)

    typer.echo("source:  " + uri + ("  [RGB]" if show_rgb else "  [skeleton only]"))
    typer.echo("serving: http://" + bind_host + ":" + str(bind_port) + "  (Ctrl+C to stop)")

    runner.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nstopping...")
    finally:
        server.shutdown()
        runner.stop()


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

    try:
        import pyrealsense2 as rs
    except ImportError:
        typer.echo("pyrealsense2 NOT INSTALLED (install the 'realsense' extra)")
        return

    typer.echo("pyrealsense " + str(rs.__version__))
    devices = list(rs.context().query_devices())
    typer.echo("realsense devices: " + str(len(devices)))

    if not devices:
        typer.echo("  none found -- check the cable is USB 3 and the port is host-mode")
        return

    for d in devices:
        name = d.get_info(rs.camera_info.name)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        typer.echo("  " + name + "  usb " + usb)
        # A D435i on a USB 2 link silently loses stream profiles rather than
        # erroring, so say so plainly.
        if usb.startswith("2"):
            typer.echo(
                "  WARNING: negotiated USB " + usb + " -- depth+colour at 30 fps "
                "will not fit. Use a USB 3 cable, no passive extension."
            )


if __name__ == "__main__":
    app()
