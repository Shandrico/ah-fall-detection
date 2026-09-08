"""Command line entry points.

    ahfd run                            webcam -> pose -> skeleton on black
    ahfd run --source file://clip.mp4   the same pipeline over a recording
    ahfd info                           environment and hardware report
"""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ahfd.config import load_config

app = typer.Typer(add_completion=False, help="Privacy-preserving fall detection.")


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

    from ahfd.alert import ConsoleSink, JsonlSink, MultiSink
    from ahfd.capture import open_source
    from ahfd.detect import FallStateMachine
    from ahfd.features import FeatureExtractor
    from ahfd.geometry.calibration import load_calibration
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.track import SimpleTracker
    from ahfd.viz import render_skeleton

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
        if not calib_path:
            raise typer.BadParameter(
                "detect.enabled is set but no calibration was given. Fall "
                "thresholds are metric, so the camera height and tilt are "
                "required -- pass --calibration or set 'calibration:' in the "
                "config. See calib/example_ward6.yaml."
            )
        calib = load_calibration(calib_path)
        _check_calibration_resolution(calib, src.meta)
        extractor = FeatureExtractor(
            calib.ground,
            zones=calib.zones,
            min_keypoint_score=cfg.pose.min_keypoint_score,
        )
        machine = FallStateMachine(cfg.detect.to_thresholds())

        sinks: list = []
        if cfg.alert.console:
            sinks.append(ConsoleSink(min_severity=cfg.alert.min_severity))
        if cfg.alert.jsonl_path:
            sinks.append(JsonlSink(cfg.alert.jsonl_path))
        sink = MultiSink(*sinks)

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

    if view_mode == "skeleton":
        typer.echo("press q in the window to quit")

    window = "ahfd -- skeleton only"
    fps_ema: float | None = None
    n = 0
    events_seen = 0

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

            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            if view_mode == "skeleton":
                canvas = render_skeleton(
                    pose,
                    min_keypoint_score=cfg.pose.min_keypoint_score,
                    show_ids=cfg.view.show_ids,
                    show_bbox=cfg.view.show_bbox,
                    fps=fps_ema if cfg.view.show_fps else None,
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
        if view_mode == "skeleton":
            cv2.destroyAllWindows()

    typer.echo("processed " + str(n) + " frames")
    if fps_ema is not None:
        typer.echo("mean pipeline rate " + format(fps_ema, ".1f") + " fps")
    if machine is not None:
        typer.echo("events emitted " + str(events_seen))


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

    typer.echo("python      " + sys.version.split()[0])

    for module in ("numpy", "cv2", "onnxruntime", "rtmlib"):
        try:
            m = __import__(module)
            typer.echo(
                module.ljust(11) + " " + str(getattr(m, "__version__", "(no version)"))
            )
        except ImportError:
            typer.echo(module.ljust(11) + " NOT INSTALLED")

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
