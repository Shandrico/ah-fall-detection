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

# Fixed mount downtilt (degrees) for a camera with no IMU (D435f). This is the
# ONE place to change the angle: it is the default --pitch for `depth-view` and
# `extract`, so you never have to pass the flag. Change it if the mount changes.
MOUNT_PITCH_DEG = 15.0


def _build_detection(
    cfg, calib_path, meta, *, emit_bed_early_warning: bool | None = None
):
    """Wire up feature extractor + state machine + sinks from a calibration.

    Shared by `run` and `replay` so the detection path is defined once. Returns
    (extractor, machine, sink) or raises typer.BadParameter if the calibration
    is missing or its resolution does not match the stream. Kept out of the
    per-command bodies because getting the resolution guard wrong produces
    plausible-but-wrong metres, and it must be identical everywhere.
    """
    from ahfd.alert import ConsoleSink, JsonlSink, MultiSink
    from ahfd.detect import DetectionEngine
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
    bed_thresholds = cfg.bed_activity.to_thresholds()
    if emit_bed_early_warning is not None:
        from dataclasses import replace

        bed_thresholds = replace(
            bed_thresholds, emit_early_warning=emit_bed_early_warning
        )
    machine = DetectionEngine(
        cfg.detect.to_thresholds(),
        bed_thresholds,
        bed_activity_enabled=cfg.bed_activity.enabled,
    )

    sinks: list = []
    if cfg.alert.console:
        sinks.append(ConsoleSink(min_severity=cfg.alert.min_severity))
    if cfg.alert.jsonl_path:
        sinks.append(JsonlSink(cfg.alert.jsonl_path))
    return extractor, machine, MultiSink(*sinks), calib


def _validate_onsite_calibration(calib, device_serial_sha256: str) -> None:
    """Refuse geometry that has not passed the physical onsite preflight.

    This is intentionally scoped to ``ahfd collect``.  Development playback
    and the ordinary local runner may use an unverified calibration, but a
    hospital collection must make the human verification latch and the
    geometry/privacy invariants explicit before the camera loop starts.
    """
    import re

    if not re.fullmatch(r"cam_[0-9a-f]{8}", calib.camera_id):
        raise typer.BadParameter(
            "onsite pseudonymous camera_id must be cam_ plus exactly 8 random hex characters; "
            "do not encode a ward, room, bed, date, or serial"
        )
    if not calib.verified_for_onsite:
        raise typer.BadParameter(
            "calibration is not verified for onsite collection. Complete the "
            "mount/height/bed/depth preflight in docs/ONSITE_COLLECTION.md, "
            "then set verified_for_onsite: true in that calibration file."
        )
    if calib.device_serial_sha256 != device_serial_sha256:
        raise typer.BadParameter(
            "the connected D435i does not match this calibration's hashed device "
            "identity; recalibrate the exact camera or connect the correct unit"
        )
    if calib.ankle_height_baseline_m is None:
        raise typer.BadParameter(
            "onsite calibration needs ankle_height_baseline_m from the approved "
            "two-ankle standing preflight before verified_for_onsite is enabled"
        )
    if not 1.5 <= float(calib.height_m) <= 4.0:
        raise typer.BadParameter("onsite camera height must be within 1.5-4.0 m")
    if not -0.05 <= float(calib.ankle_height_baseline_m) <= 0.25:
        raise typer.BadParameter(
            "onsite ankle baseline must be within -0.05 to 0.25 m"
        )
    bed_zones = [zone for zone in calib.zones.zones if zone.kind == "bed"]
    if not bed_zones:
        raise typer.BadParameter("onsite calibration has no bed zones")
    names = [zone.name for zone in calib.zones.zones]
    if len(names) != len(set(names)):
        raise typer.BadParameter("onsite calibration has duplicate zone names")
    bad_names = [
        name
        for name in names
        if not re.fullmatch(r"(?:bed|chair|floor|exclude)_[a-z0-9]{1,20}", name)
    ]
    if bad_names:
        raise typer.BadParameter(
            "onsite zone names must be pseudonymous kind_codes "
            "(bed_a, floor_1); bad: " + ", ".join(bad_names)
        )
    if any(zone.risk_level != "unknown" for zone in bed_zones):
        raise typer.BadParameter(
            "onsite calibration must contain geometry only: set bed risk_level "
            "to unknown and keep care policy in the hospital-controlled system"
        )
    for zone in bed_zones:
        if zone.top_m is None or not 0.25 <= float(zone.top_m) <= 1.20:
            raise typer.BadParameter(
                "onsite bed top_m must be within the plausible 0.25-1.20 m range"
            )
        if _polygon_area(zone.polygon) < 0.20 or _polygon_self_intersects(
            zone.polygon
        ):
            raise typer.BadParameter(
                "onsite bed polygons must be non-self-intersecting with area >= 0.20 m^2"
            )
    for index, left in enumerate(bed_zones):
        for right in bed_zones[index + 1 :]:
            if _polygons_overlap(left.polygon, right.polygon):
                raise typer.BadParameter(
                    "onsite bed polygons overlap: " + left.name + " and " + right.name
                )


def _polygon_area(points) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def _segments_intersect(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    values = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    epsilon = 1e-9

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
            and min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon
        )

    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return (
        abs(values[0]) <= epsilon and on_segment(a, c, b)
        or abs(values[1]) <= epsilon and on_segment(a, d, b)
        or abs(values[2]) <= epsilon and on_segment(c, a, d)
        or abs(values[3]) <= epsilon and on_segment(c, b, d)
    )


def _polygon_self_intersects(points) -> bool:
    size = len(points)
    for first in range(size):
        a, b = points[first], points[(first + 1) % size]
        for second in range(first + 1, size):
            if second in (first, (first + 1) % size) or (second + 1) % size == first:
                continue
            c, d = points[second], points[(second + 1) % size]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _polygons_overlap(left, right) -> bool:
    from ahfd.geometry.zones import point_in_polygon

    if any(point_in_polygon(point, right) for point in left):
        return True
    if any(point_in_polygon(point, left) for point in right):
        return True
    return any(
        _segments_intersect(
            left[i],
            left[(i + 1) % len(left)],
            right[j],
            right[(j + 1) % len(right)],
        )
        for i in range(len(left))
        for j in range(len(right))
    )


def _validate_approved_output(root: Path, site_id: str) -> None:
    """Require a custodian-provisioned marker on the encrypted study volume."""
    import json

    root = Path(root).expanduser().resolve()
    marker = root / ".ahfd-approved-output.json"
    if not root.is_dir() or not marker.is_file():
        raise typer.BadParameter(
            "onsite output must already exist and contain the custodian-provisioned "
            ".ahfd-approved-output.json marker"
        )
    try:
        document = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter("approved-output marker is not valid JSON") from exc
    expected = {
        "schema": "ahfd.approved-output",
        "schema_version": 1,
        "site_id": site_id,
        "encrypted_storage_attested": True,
        "purpose": "research_shadow_collection",
    }
    if document != expected:
        raise typer.BadParameter(
            "approved-output marker must exactly match this site, purpose, and "
            "encrypted-storage attestation"
        )


def _validate_calibration_approval(
    calibration: Path,
    approval: Path,
    site_id: str,
    config_sha256: str,
    pose_model_sha256: str,
) -> tuple[str, str]:
    """Return approval and calibration hashes from one validated byte snapshot."""
    import hashlib
    import json
    from datetime import datetime, timezone

    calibration = Path(calibration).expanduser().resolve()
    approval = Path(approval).expanduser().resolve()
    if not approval.is_file():
        raise typer.BadParameter("external calibration approval record was not found")
    try:
        calibration_bytes = calibration.read_bytes()
        approval_bytes = approval.read_bytes()
        document = json.loads(approval_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter("calibration approval record is not valid JSON") from exc
    digest = hashlib.sha256(calibration_bytes).hexdigest()
    approved_utc = document.get("approved_utc") if isinstance(document, dict) else None
    try:
        approved_at = datetime.fromisoformat(str(approved_utc).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter("calibration approval needs an ISO-8601 approved_utc") from exc
    if approved_at.tzinfo is None:
        raise typer.BadParameter("calibration approval approved_utc must include a timezone")
    age_s = (datetime.now(timezone.utc) - approved_at.astimezone(timezone.utc)).total_seconds()
    if age_s < -300 or age_s > 12 * 60 * 60:
        raise typer.BadParameter(
            "calibration approval must be issued after the current pre-session "
            "mount/depth check and within the last 12 hours"
        )
    expected = {
        "schema": "ahfd.calibration.approval",
        "schema_version": 2,
        "site_id": site_id,
        "purpose": "research_shadow_collection",
        "config_sha256": config_sha256,
        "calibration_sha256": digest,
        "pose_model_sha256": pose_model_sha256,
        "approved_utc": approved_utc,
    }
    if document != expected:
        raise typer.BadParameter(
            "calibration approval does not exactly match this site, purpose, "
            "config SHA-256, calibration SHA-256, and pose-model SHA-256"
        )
    return hashlib.sha256(approval_bytes).hexdigest(), digest


def _canonical_onsite_source(
    uri: str,
    meta,
    cfg,
    approval_sha256: str,
    verified_depth_controls: dict[str, float | bool],
) -> dict:
    """Return allowlisted provenance based on applied sensor readback."""
    import math
    from importlib.metadata import PackageNotFoundError, version

    from ahfd.capture.factory import parse_realsense_uri

    options = parse_realsense_uri(uri)
    if options.get("infrared"):
        raise typer.BadParameter("onsite collection requires the calibrated colour stream")
    if (
        verified_depth_controls.get("emitter_enabled") is not True
        or verified_depth_controls.get("laser_at_max") is not True
    ):
        raise typer.BadParameter(
            "onsite source provenance requires verified emitter and laser readback"
        )
    try:
        laser_power = float(verified_depth_controls["laser_power"])
        laser_power_max = float(verified_depth_controls["laser_power_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "onsite source provenance requires numeric laser readback"
        ) from exc
    tolerance = max(0.01, abs(laser_power_max) * 1e-4)
    if (
        not math.isfinite(laser_power)
        or not math.isfinite(laser_power_max)
        or laser_power_max <= 0.0
        or abs(laser_power - laser_power_max) > tolerance
    ):
        raise typer.BadParameter(
            "onsite source provenance requires verified maximum laser power"
        )
    runtime_versions = {}
    for package in ("numpy", "opencv-python", "rtmlib", "openvino", "onnxruntime"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {
        "kind": "intel_realsense_d435i",
        "width": int(meta.width),
        "height": int(meta.height),
        "fps": float(meta.fps),
        "depth_enabled": bool(meta.has_depth),
        # These are applied/read-back states, not URI intent.
        "emitter": True,
        "max_laser": True,
        "laser_power": laser_power,
        "laser_power_max": laser_power_max,
        "max_range_m": float(options.get("max_range_m", 6.0)),
        "spatial_magnitude": int(options.get("spatial_magnitude", 2)),
        "pose_backend": str(cfg.pose.backend),
        "pose_model_size": str(cfg.pose.model_size),
        "pose_runtime": str(cfg.pose.runtime),
        "pose_device": str(cfg.pose.device),
        "runtime_versions": runtime_versions,
        "approval_sha256": approval_sha256,
    }


def _pose_artifact_sha256(estimator) -> str:
    """Hash the exact local ONNX artifact used by the approved RTMO runtime."""
    import hashlib

    model = getattr(estimator, "_model", None)
    path = getattr(model, "onnx_model", None)
    candidate = Path(path) if isinstance(path, str) else None
    if candidate is None or not candidate.is_file():
        raise typer.BadParameter(
            "cannot resolve the local pose model artifact for provenance; "
            "onsite collection requires the reviewed RTMO ONNX file"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _containing_git_worktree(path: Path) -> Path | None:
    """Return the Git worktree containing ``path`` (including future children)."""
    import subprocess

    target = path.expanduser().resolve()
    probe = target if target.is_dir() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(result.stdout.strip()).resolve()


def _validate_comparator_output(out_dir: Path, sessions: list[Path]) -> None:
    """Require a new analysis directory under the approved external site root."""
    import json

    worktree = _containing_git_worktree(out_dir)
    if worktree is not None:
        raise typer.BadParameter(
            "model output contains sensitive study-derived information and must "
            "stay outside Git; selected path is inside " + str(worktree)
        )
    site_ids: set[str] = set()
    try:
        for session in sessions:
            manifest = json.loads((Path(session) / "manifest.json").read_text("utf-8"))
            site_ids.add(str(manifest["site_id"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(
            "cannot bind comparator output to the input session site"
        ) from exc
    if len(site_ids) != 1:
        raise typer.BadParameter(
            "comparator output requires input sessions from exactly one approved site"
        )
    approved_root = Path(out_dir).expanduser().resolve().parent
    while not (approved_root / ".ahfd-approved-output.json").is_file():
        if approved_root == approved_root.parent:
            raise typer.BadParameter(
                "comparator output must be under a custodian-approved external root"
            )
        approved_root = approved_root.parent
    _validate_approved_output(approved_root, next(iter(site_ids)))


def _probe_onsite_d435i() -> tuple[str, str]:
    """Return the verified raw serial for binding plus its provenance digest."""
    import hashlib

    from ahfd.capture import probe_realsense

    probe = probe_realsense()
    if not probe.installed:
        raise typer.BadParameter("pyrealsense2 is not installed for the onsite D435i")
    if probe.error:
        raise typer.BadParameter("D435i enumeration failed: " + probe.error)
    if len(probe.devices) != 1:
        raise typer.BadParameter(
            "onsite collection requires exactly one connected RealSense; found "
            + str(len(probe.devices))
        )
    device = probe.devices[0]
    if "D435I" not in device.name.upper():
        raise typer.BadParameter(
            "onsite collection requires the approved D435i; found " + device.name
        )
    usb_descriptor = str(device.usb or "").strip()
    if not usb_descriptor.startswith("3"):
        raise typer.BadParameter(
            "D435i negotiated USB "
            + (usb_descriptor or "unknown")
            + "; onsite collection requires a verified USB 3 link without a hub"
        )
    if not device.serial:
        raise typer.BadParameter("D435i did not report a serial; device identity is unverifiable")
    return device.serial, hashlib.sha256(device.serial.encode("utf-8")).hexdigest()


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

            # A depth-enabled RealSense URI (``rs://?depth=1``) contributes
            # only seventeen per-joint height scalars.  Dense depth remains on
            # this local capture frame and is discarded after the iteration.
            if extractor is not None and frame.depth_raw is not None:
                from ahfd.features import attach_depth_heights

                pose = attach_depth_heights(
                    pose,
                    frame,
                    extractor.ground,
                    min_score=cfg.pose.min_keypoint_score,
                )

            metrics: dict[int, dict] = {}
            if machine is not None and extractor is not None and sink is not None:
                live_ids = set(tracker.live_ids)
                observed_ids: set[int] = set()
                extractor.retain_only(live_ids)
                machine.retain_only(live_ids)
                for person in pose.people:
                    features = extractor.extract(person, pose.t)
                    if features is None:
                        continue
                    if person.track_id is not None:
                        observed_ids.add(person.track_id)
                        metrics[person.track_id] = {
                            "state": machine.state_of(person.track_id),
                            "h_torso": features.h_torso,
                            "floor_spread": features.floor_spread,
                            "v_z": features.v_z,
                            "h_ankle_min": features.h_ankle_min,
                            "bed_risk": features.bed_risk,
                            "range_m": features.range_m,
                        }
                    for event in machine.update_all(features):
                        sink.emit(event)
                        events_seen += 1
                        if event.type in ("FALL_CONFIRMED", "PERSON_DOWN"):
                            last_alert = event.describe()
                # A tracker may keep an identity alive across a missed pose.
                # Preserve its last activity phase, but explicitly mark bed
                # monitoring unavailable for this frame.
                machine.mark_frame_unobserved(pose.t, live_ids, observed_ids)

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


def _finite_or_none(value):
    """JSON-safe finite float, or None for missing/invalid measurements."""
    import math

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 5) if math.isfinite(number) else None


def _collection_person_record(
    person, features, machine, association_epoch: int, temporal_summary=None
) -> dict:
    """Privacy-reduced, JSON-only row for one tracked person."""
    heights = None
    if person.heights is not None:
        heights = [_finite_or_none(value) for value in person.heights]
    record = {
        "track_id": int(person.track_id),
        "association_epoch": int(association_epoch),
        "keypoints_xy": [
            [_finite_or_none(point[0]), _finite_or_none(point[1])]
            for point in person.keypoints
        ],
        "keypoint_scores": [_finite_or_none(value) for value in person.scores],
        "joint_heights_m": heights,
        "features": {
            "contact_xy_m": (
                [_finite_or_none(features.contact_xy[0]), _finite_or_none(features.contact_xy[1])]
                if features.contact_xy is not None
                else None
            ),
            "range_m": _finite_or_none(features.range_m),
            "h_torso_m": _finite_or_none(features.h_torso),
            "h_shoulder_m": _finite_or_none(features.h_shoulder),
            "h_head_m": _finite_or_none(features.h_head),
            "floor_spread_m": _finite_or_none(features.floor_spread),
            "vertical_velocity_mps": _finite_or_none(features.v_z),
            "motion_mps": _finite_or_none(features.motion),
            "n_valid_keypoints": int(features.n_valid_kp),
            "mean_confidence": _finite_or_none(features.mean_conf),
            "zones": list(features.zones),
            "associated_bed": features.associated_bed,
            "supported_by_bed": features.supported_by_bed,
            "bed_support_fraction": _finite_or_none(features.bed_overlap),
            "bed_edge_distance_m": _finite_or_none(features.bed_edge_distance_m),
            "torso_tilt_deg": _finite_or_none(features.torso_tilt),
        },
        "fall_state": machine.state_of(person.track_id),
    }
    if hasattr(machine, "bed_snapshot_of"):
        snapshot = machine.bed_snapshot_of(person.track_id, features.t)
        if snapshot is not None:
            activity = snapshot.to_dict()
            # Care/risk policy is not a behavioural model input and must not be
            # copied out of the hospital-controlled system into research rows.
            activity.pop("bed_risk", None)
            record["bed_activity"] = activity
    if temporal_summary is not None:
        record["temporal_summary"] = {
            "schema_version": temporal_summary.schema_version,
            # The ordered feature names and their SHA-256 live once in the
            # manifest.  Repeating 929 JSON keys for every frame inflated an
            # hour-long session by roughly a gigabyte.
            "values": temporal_summary.as_vector(),
        }
    return record


def _standing_depth_ankle_height(
    person, features, state: str, *, min_score: float, feature_valid: bool
) -> float | None:
    """Return a real sparse-depth ankle check, never monocular fallback geometry."""
    import math

    from ahfd.pose.skeleton import ANKLES

    if (
        not feature_valid
        or state != "UPRIGHT"
        or features.supported_by_bed is not None
        or person.heights is None
    ):
        return None
    values = [
        float(person.heights[index])
        for index in ANKLES
        if (
            index < len(person.heights)
            and person.scores[index] >= min_score
            and math.isfinite(float(person.heights[index]))
        )
    ]
    # Both ankles are required.  A lone stereo sample at a limb boundary is too
    # easy to corrupt, and a lifted foot is not evidence that the mount moved.
    if len(values) != len(ANKLES) or max(values) - min(values) > 0.12:
        return None
    return sum(values) / len(values)


def _standing_preflight_eligible(features, cfg) -> bool:
    """Strict posture gate for the consenting-staff ankle preflight."""
    return bool(
        features is not None
        and features.n_valid_kp >= cfg.detect.min_valid_kp
        and features.mean_conf >= cfg.detect.min_mean_conf
        and features.has_geometry()
        and features.h_torso is not None
        and features.h_torso >= cfg.detect.upright_h
        and features.torso_tilt is not None
        and features.torso_tilt <= cfg.detect.seated_tilt_max
        and features.supported_by_bed is None
        and not features.in_excluded_zone
    )


def _bed_observation_valid(snapshot) -> bool:
    """Use the bed machine's full visibility gate for temporal training rows."""
    return snapshot is not None and snapshot.observation == "VALID"


def _collection_marker(
    kind: str,
    value: str,
    selected_track_id: int | None,
    observed_ids: set[int],
    epochs: dict[int, int],
) -> dict | None:
    """Build one observer marker only for an explicitly selected association."""
    if selected_track_id is None or selected_track_id not in observed_ids:
        return None
    record = {
        "kind": kind,
        "track_ids": [selected_track_id],
        "associations": [
            {
                "track_id": selected_track_id,
                "association_epoch": epochs[selected_track_id],
            }
        ],
    }
    record["phase" if kind == "phase_marker" else "value"] = value
    return record


def _collection_scope_violation(
    participant_id: str | None,
    observed_track_ids: set[int],
    *,
    credible_person_count: int | None = None,
) -> str | None:
    """Return the fail-closed privacy code for the current observed people."""
    count = max(
        len(observed_track_ids),
        len(observed_track_ids)
        if credible_person_count is None
        else int(credible_person_count),
    )
    if participant_id is None and count:
        return "UNEXPECTED_PERSON_IN_EMPTY_ROOM"
    if participant_id is not None and count > 1:
        return "UNAPPROVED_PERSON_PRESENT"
    return None


def _credible_pose_count(people, *, min_person_score: float) -> int:
    """Count privacy-relevant detections, including ones the tracker cannot box.

    RTMO is configured with the same person-score threshold.  Rechecking it
    here makes the collection boundary explicit and prevents a credible but
    keypoint-poor second pose (``track_id=None``) from bypassing scope checks.
    """
    import math

    return sum(
        1
        for person in people
        if math.isfinite(float(getattr(person, "score", float("nan"))))
        and float(person.score) >= float(min_person_score)
    )


def _returned_associations(
    observed_track_ids: set[int], recently_missing: dict[int, float]
) -> set[int]:
    """Identify reused tracker IDs before the missing map is pruned."""
    return set(observed_track_ids) & set(recently_missing)


def _next_collection_target(
    candidates: list[int], selected_track_id: int | None, *, step: int
) -> tuple[int | None, bool]:
    """Return the requested target and whether its association actually changed."""
    if not candidates:
        return None, selected_track_id is not None
    if selected_track_id not in candidates:
        return candidates[0], True
    index = (candidates.index(selected_track_id) + step) % len(candidates)
    next_track_id = candidates[index]
    return next_track_id, next_track_id != selected_track_id


def _collection_completion_issues(
    participant_id: str | None,
    *,
    labeled_available_rows: int,
    imu_orientation_ready: bool,
) -> tuple[str, ...]:
    """List missing evidence that makes a participant run incomplete."""
    if participant_id is None:
        return ()
    issues: list[str] = []
    if labeled_available_rows < 1:
        issues.append("NO_LABELED_AVAILABLE_TARGET_ROWS")
    if not imu_orientation_ready:
        issues.append("IMU_PREFLIGHT_NOT_READY")
    return tuple(issues)


def _is_labeled_available_target_row(
    participant_id: str | None,
    selected_track_id: int | None,
    association_epoch: int,
    people_rows: list[dict],
    monitoring: dict,
    current_phase_by_association: dict[tuple[int, int], str],
) -> bool:
    """True only for a persisted available row with a prior usable local label."""
    if participant_id is None or selected_track_id is None:
        return False
    association = (selected_track_id, association_epoch)
    phase = current_phase_by_association.get(association)
    if phase is None or phase == "UNKNOWN" or monitoring.get("status") != "AVAILABLE":
        return False
    return any(
        (row.get("track_id"), row.get("association_epoch")) == association
        for row in people_rows
    )


def _collection_exception_abort_code(
    abort_code: str, *, processing_frame: bool
) -> str:
    """Identify capture-boundary failures without relabeling later failures."""
    if abort_code == "PIPELINE_ERROR" and not processing_frame:
        return "CAMERA_DISCONNECTED"
    return abort_code


def _collection_disk_free_bytes(path: Path) -> int:
    """Return output-volume free bytes; callers classify OSError as disk failure."""
    import shutil

    return int(shutil.disk_usage(path).free)


def _finalize_collection_recorder(
    recorder,
    *,
    requested_abort: bool,
    abort_reason: str | None = None,
) -> str:
    """Complete an eligible run or persist a controlled abort reason."""
    if requested_abort:
        abort_reason = "REQUESTED_STOP"
    if abort_reason is not None:
        recorder.abort(abort_reason)
        return "aborted"
    recorder.complete()
    return "complete"


def _intrinsics_match(expected, actual, *, tolerance_px: float = 1.0) -> bool:
    """Check the live factory intrinsics against the file used for geometry."""
    if actual is None:
        return False
    if (expected.width, expected.height) != (actual.width, actual.height):
        return False
    return all(
        abs(float(left) - float(right)) <= tolerance_px
        for left, right in (
            (expected.fx, actual.fx),
            (expected.fy, actual.fy),
            (expected.cx, actual.cx),
            (expected.cy, actual.cy),
        )
    )


def _imu_orientation_delta(calib, gravity) -> tuple[float, float] | None:
    """Live D435i pitch/roll delta from the orientation stored in calibration."""
    if gravity is None:
        return None
    from ahfd.geometry.ground import GroundPlane

    live = GroundPlane.from_gravity(
        calib.ground.intrinsics,
        calib.height_m,
        gravity,
    )
    roll_delta = (live.roll_deg - calib.ground.roll_deg + 180.0) % 360.0 - 180.0
    return live.pitch_deg - calib.ground.pitch_deg, roll_delta


def _depth_provenance(person, *, min_score: float) -> dict[str, float | None]:
    """Causal masks that stop a model confusing depth loss with movement."""
    import math

    from ahfd.pose.skeleton import HIPS, SHOULDERS

    if person.heights is None:
        return {
            "joint_depth_valid_fraction": 0.0,
            "shoulder_depth_available": 0.0,
            "torso_depth_available": 0.0,
        }
    valid = [
        index
        for index in range(min(len(person.heights), len(person.scores)))
        if person.scores[index] >= min_score
        and math.isfinite(float(person.heights[index]))
    ]
    valid_set = set(valid)
    shoulder_available = any(index in valid_set for index in SHOULDERS)
    hip_available = any(index in valid_set for index in HIPS)
    return {
        "joint_depth_valid_fraction": len(valid) / 17.0,
        "shoulder_depth_available": float(shoulder_available),
        "torso_depth_available": float(shoulder_available and hip_available),
    }


def _target_depth_healthy(provenance: dict[str, float | None]) -> bool:
    """Require critical trunk depth plus four valid joints, not one limb hit."""
    return bool(
        float(provenance.get("joint_depth_valid_fraction") or 0.0) >= 4.0 / 17.0
        and provenance.get("shoulder_depth_available") == 1.0
        and provenance.get("torso_depth_available") == 1.0
    )


def _robust_ankle_baseline(values: list[float]) -> float:
    """Validate a standing preflight sample and return its robust median."""
    import numpy as np

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 30:
        raise ValueError("need at least 30 valid two-ankle samples")
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if not -0.05 <= median <= 0.25:
        raise ValueError("standing ankle baseline is outside the plausible floor band")
    if mad > 0.025:
        raise ValueError("standing ankle sample is too unstable; fix depth/mount/pose")
    return median


@app.command(name="collect")
def collect_onsite(
    out_root: Path = typer.Option(
        ..., "--out-root", help="Hospital-managed encrypted output directory."
    ),
    site_id: str = typer.Option(
        ...,
        help="Random pseudonymous site code, e.g. site_0123abcd (never a ward/room name).",
    ),
    participant_id: str = typer.Option(
        None,
        help=(
            "Optional random pseudonym, e.g. sub_0123456789abcdef; "
            "omit for empty-room tests."
        ),
    ),
    session_id: str = typer.Option(
        None,
        help="Optional ses_ plus 16 random hex characters; generated when omitted.",
    ),
    source: str = typer.Option(
        None,
        help="Disabled onsite: the reviewed config is the capture-source authority.",
    ),
    config: Path = typer.Option(
        Path("configs/onsite_collection.yaml"), help="Locked shadow-collection YAML."
    ),
    calibration: Path = typer.Option(
        None, help="Exact onsite mount calibration; defaults to config calibration."
    ),
    approval: Path = typer.Option(
        None,
        help=(
            "External governance approval JSON bound to config, calibration, "
            "and pose-model hashes."
        ),
    ),
    seconds: float = typer.Option(0.0, help="Stop after N wall-clock seconds; 0 = until q."),
    max_frames: int = typer.Option(0, help="Dry-run limit; 0 = no frame limit."),
    view: bool = typer.Option(
        True, "--view/--headless", help="Show only the derived skeleton and enable marker keys."
    ),
) -> None:
    """Collect derived D435i signals and observer markers in research shadow mode.

    No RGB frame or dense depth map is written. Output contains keypoints,
    seventeen optional joint-height scalars, metric/temporal features, explicit
    availability telemetry and controlled observer labels. It is sensitive
    research data and must not be committed to GitHub.
    """
    from collections import deque
    import hashlib
    import os
    import re
    import subprocess
    import threading
    import uuid

    import cv2
    import numpy as np

    from ahfd.capture import open_source
    from ahfd.capture.factory import parse_realsense_uri
    from ahfd.collect import HealthMonitor, HealthReason, SessionRecorder, sha256_file
    from ahfd.features import attach_depth_heights
    from ahfd.geometry.calibration import drift_check
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.privacy import ENV_VAR
    from ahfd.track import SimpleTracker
    from ahfd.viz import render_skeleton
    from ahfd.ml.temporal import (
        TEMPORAL_FEATURE_NAMES,
        TEMPORAL_SCHEMA_VERSION,
        CausalTemporalSummarizer,
    )

    # Hash around parsing so the runtime object and recorded provenance cannot
    # silently refer to different bytes if an operator/editor replaces a file
    # during preflight.
    config_sha256 = sha256_file(config)
    cfg = load_config(config)
    if sha256_file(config) != config_sha256:
        raise typer.BadParameter("onsite config changed while it was being loaded")
    if source is not None:
        raise typer.BadParameter(
            "--source overrides are disabled onsite; update and re-approve the config"
        )
    uri = cfg.source
    calib_path = calibration or cfg.calibration
    if not str(uri).startswith("rs://"):
        raise typer.BadParameter("onsite collection accepts only a live rs:// source")
    try:
        source_options = parse_realsense_uri(uri)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter("invalid onsite RealSense URI: " + str(exc)) from exc
    if source_options.get("infrared"):
        raise typer.BadParameter("onsite collection requires the calibrated colour stream")
    if not source_options.get("with_depth"):
        raise typer.BadParameter(
            "onsite collection requires depth=1 in the approved config"
        )
    if not source_options.get("max_laser"):
        raise typer.BadParameter(
            "onsite collection requires max_laser=1 in the approved config"
        )
    if source_options.get("emitter") is not True:
        raise typer.BadParameter(
            "onsite collection requires emitter=1 in the approved config"
        )
    # Refuse an armed raw path before calibration/repository preflight.  The
    # collection command must fail closed even when other required arguments
    # are missing and it must never reach a writer or camera in this state.
    if cfg.privacy.allow_raw_capture or os.environ.get(ENV_VAR) == "1":
        raise typer.BadParameter(
            "raw capture is armed; unset AHFD_ALLOW_RAW and keep privacy.allow_raw_capture false"
        )
    if calib_path is None:
        raise typer.BadParameter("onsite collection needs the exact mount calibration")
    if approval is None:
        raise typer.BadParameter("onsite collection needs an external --approval record")
    if participant_id is not None and not view:
        raise typer.BadParameter(
            "participant collection requires the skeleton view for explicit target binding"
        )
    if not cfg.detect.enabled:
        raise typer.BadParameter("onsite collection config must set detect.enabled: true")
    if not cfg.bed_activity.enabled:
        raise typer.BadParameter(
            "onsite collection config must set bed_activity.enabled: true"
        )
    if not cfg.bed_activity.emit_exit_event:
        raise typer.BadParameter(
            "onsite collection requires bed_activity.emit_exit_event: true "
            "for shadow outcome logging"
        )
    if cfg.pose.backend != "rtmo":
        raise typer.BadParameter(
            "first onsite protocol is pinned to the reviewed Apache-2.0 RTMO pose backend"
        )
    if cfg.bed_activity.emit_early_warning:
        raise typer.BadParameter(
            "onsite collection is shadow-only: bed_activity.emit_early_warning must be false"
        )
    if cfg.dashboard.show_rgb or cfg.dashboard.allow_rgb:
        raise typer.BadParameter(
            "onsite collection config must keep dashboard show_rgb/allow_rgb false"
        )
    if cfg.alert.console or cfg.alert.jsonl_path:
        raise typer.BadParameter(
            "onsite collection keeps shadow events only in its managed derived "
            "stream; disable alert.console and alert.jsonl_path"
        )
    if not re.fullmatch(r"site_[0-9a-f]{8}", site_id):
        raise typer.BadParameter("site_id must be site_ plus exactly 8 random hex characters")
    if participant_id is not None and not re.fullmatch(
        r"sub_[0-9a-f]{16}", participant_id
    ):
        raise typer.BadParameter(
            "participant_id must be sub_ plus exactly 16 random hex characters"
        )
    if session_id is not None and not re.fullmatch(r"ses_[0-9a-f]{16}", session_id):
        raise typer.BadParameter(
            "session_id must be ses_ plus exactly 16 random hex characters"
        )
    output_worktree = _containing_git_worktree(out_root)
    if output_worktree is not None:
        raise typer.BadParameter(
            "onsite output must be an approved encrypted directory outside every "
            "Git worktree; selected path is inside " + str(output_worktree)
        )
    _validate_approved_output(out_root, site_id)
    calibration_worktree = _containing_git_worktree(Path(calib_path))
    if calibration_worktree is not None:
        raise typer.BadParameter(
            "onsite calibration may reveal ward geometry and must live outside "
            "Git; selected file is inside " + str(calibration_worktree)
        )
    approval_worktree = _containing_git_worktree(Path(approval))
    if approval_worktree is not None:
        raise typer.BadParameter(
            "calibration approval must be a hospital-controlled external record, "
            "not a file inside " + str(approval_worktree)
        )
    repo_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        dirty = True
    if dirty:
        raise typer.BadParameter(
            "onsite collection requires a clean, reviewed Git revision; commit or "
            "discard local changes and resolve any merge before collection"
        )

    typer.echo("loading the reviewed pose model for provenance preflight...")
    review_estimator = build_estimator(cfg.pose)
    pose_model_sha256 = _pose_artifact_sha256(review_estimator)
    approval_sha256, calibration_sha256 = _validate_calibration_approval(
        Path(calib_path),
        Path(approval),
        site_id,
        config_sha256,
        pose_model_sha256,
    )
    if _pose_artifact_sha256(review_estimator) != pose_model_sha256:
        raise typer.BadParameter("reviewed pose model changed during approval preflight")
    del review_estimator
    estimator = build_estimator(cfg.pose)
    if _pose_artifact_sha256(estimator) != pose_model_sha256:
        raise typer.BadParameter("runtime pose model does not match the approved artifact")
    device_serial, device_serial_sha256 = _probe_onsite_d435i()
    src = open_source(
        uri,
        width=cfg.capture.width,
        height=cfg.capture.height,
        device_serial=device_serial,
        strict_depth_controls=True,
    )
    extractor = machine = sink = None
    recorder = None
    abort_code = "PIPELINE_ERROR"
    health = None
    watchdog_stop = None
    watchdog_thread = None
    watchdog_errors: list[Exception] = []
    window = "ahfd onsite collection -- DERIVED SKELETON ONLY"
    frames_seen = 0
    event_count = 0
    epoch_counter = 0
    epochs: dict[int, int] = {}
    previous_live: set[int] = set()
    previous_observed_tracks: set[int] = set()
    recently_missing_tracks: dict[int, float] = {}
    ankle_heights: deque[tuple[float, float]] = deque(maxlen=120)
    orientation_samples: deque[tuple[float, float, float]] = deque(maxlen=120)
    intrinsics_checked = False
    selected_track_id: int | None = None
    last_complete_wall: float | None = None
    fps_ema: float | None = None
    last_heartbeat_t = -10.0
    next_derived_write_t = 0.0
    pending_events: list[dict] = []
    stop_reason = None
    requested_abort = False
    completion_abort_reason: str | None = None
    processing_frame = False
    current_phase_by_association: dict[tuple[int, int], str] = {}
    labeled_available_rows = 0
    imu_orientation_ready = False

    phase_keys = {
        ord("1"): "RECLINED",
        ord("2"): "TORSO_RISING",
        ord("3"): "UPRIGHT_IN_BED",
        ord("4"): "SHIFTING_TO_EDGE",
        ord("5"): "EDGE_SITTING",
        ord("6"): "ATTEMPTING_STAND",
        ord("7"): "OUT_OF_BED",
        ord("0"): "UNKNOWN",
    }
    context_keys = {
        ord("x"): "RETURN_TO_RECLINE",
        ord("p"): "PAUSE",
        ord("f"): "FAST_TRANSITION",
        ord("z"): "SLIDE",
        ord("r"): "RAIL_CLIMB",
        ord("i"): "ASSISTED_TRANSFER",
        ord("c"): "STAFF_OCCLUSION",
        ord("v"): "BLANKET_OCCLUSION",
        ord("a"): "BED_ARTICULATION",
        ord("t"): "TRACK_ERROR",
        ord("u"): "OBSERVER_UNSURE",
    }

    try:
        extractor, machine, sink, calib = _build_detection(cfg, calib_path, src.meta)
        if sha256_file(calib_path) != calibration_sha256:
            raise typer.BadParameter(
                "onsite calibration changed while it was being loaded"
            )
        _validate_onsite_calibration(calib, device_serial_sha256)
        verified_depth_controls = src.preflight()
        tracker = SimpleTracker(min_keypoint_score=cfg.pose.min_keypoint_score)
        temporal = CausalTemporalSummarizer(
            max_gap_s=cfg.bed_activity.cusum.max_gap_s,
            near_edge_m=cfg.bed_activity.near_edge_enter_m,
        )
        smoother = (
            KeypointSmoother(
                min_cutoff=cfg.smoothing.min_cutoff,
                beta=cfg.smoothing.beta,
                d_cutoff=cfg.smoothing.d_cutoff,
            )
            if cfg.smoothing.enabled
            else None
        )

        source_provenance = _canonical_onsite_source(
            uri,
            src.meta,
            cfg,
            approval_sha256,
            verified_depth_controls,
        )
        source_provenance.update(
            {
                "camera_id": calib.camera_id,
                "pose_model": estimator.name,
                "pose_model_sha256": pose_model_sha256,
            }
        )
        if sha256_file(config) != config_sha256:
            raise typer.BadParameter("onsite config changed after preflight")
        if sha256_file(calib_path) != calibration_sha256:
            raise typer.BadParameter("onsite calibration changed after approval")
        if sha256_file(approval) != approval_sha256:
            raise typer.BadParameter("onsite approval record changed after preflight")
        if _pose_artifact_sha256(estimator) != pose_model_sha256:
            raise typer.BadParameter("reviewed pose model changed after approval")
        recorder = SessionRecorder(
            out_root,
            site_id=site_id,
            participant_id=participant_id,
            session_id=session_id,
            code_hash=revision,
            config_path=config,
            calibration_path=calib_path,
            source=source_provenance,
            expected_config_sha256=config_sha256,
            expected_calibration_sha256=calibration_sha256,
            temporal_schema_version=TEMPORAL_SCHEMA_VERSION,
            temporal_feature_names=TEMPORAL_FEATURE_NAMES,
        )
        # All JSONL records and the manifest must share the recorder's session
        # epoch.  Model/camera startup before this point is preflight, not
        # monitored session time.
        session_io_lock = threading.RLock()

        def _write_session(method, record, *, t_rel_s=None) -> None:
            nonlocal abort_code
            try:
                with session_io_lock:
                    timestamp = recorder.elapsed_s() if t_rel_s is None else t_rel_s
                    method(record, t_rel_s=timestamp)
            except OSError:
                abort_code = "DISK_ERROR"
                raise

        def _health_transition(transition) -> None:
            _write_session(
                recorder.write_telemetry,
                transition.as_record(),
                t_rel_s=transition.t_rel_s,
            )

        health = HealthMonitor(
            stale_after_s=2.0,
            on_transition=_health_transition,
        )

        def _health_call(method, *args, **kwargs):
            # Timestamp sampling, state transition and callback persistence are
            # serialized with every other session write.  A watchdog tick can
            # therefore never overtake an earlier main-loop observation.
            with session_io_lock:
                return method(recorder.elapsed_s(), *args, **kwargs)

        watchdog_stop = threading.Event()

        def _watch_capture() -> None:
            while not watchdog_stop.wait(0.5):
                try:
                    _health_call(health.tick)
                except Exception as exc:  # surfaced by the main loop or finalizer
                    watchdog_errors.append(exc)
                    return

        watchdog_thread = threading.Thread(
            target=_watch_capture,
            name="ahfd-collection-watchdog",
            daemon=True,
        )
        watchdog_thread.start()
        typer.echo("session: " + str(recorder.path))
        typer.echo(
            "markers: 1 reclined  2 rising  3 upright-in-bed  4 shifting  "
            "5 edge-sitting  6 stand  7 out  0 unknown"
        )
        typer.echo("target: [ / ] explicitly bind the one consented participant")
        typer.echo(
            "context: x return  p pause  f fast  z slide  r rail  i assisted  "
            "c staff-occlusion  v blanket  a bed-articulation  t track-error  "
            "u unsure  q planned-complete  Esc requested/safety-stop"
        )

        for frame in src:
            if watchdog_errors:
                if abort_code != "DISK_ERROR":
                    abort_code = "DISK_OR_WATCHDOG_ERROR"
                raise RuntimeError("collection watchdog failed") from watchdog_errors[0]
            processing_frame = True
            t_rel = recorder.elapsed_s()
            _health_call(health.note_frame)
            if not intrinsics_checked:
                if not _intrinsics_match(calib.ground.intrinsics, frame.intrinsics):
                    abort_code = "CALIBRATION_MISMATCH"
                    _health_call(
                        health.mark_unavailable,
                        HealthReason.CALIBRATION_DRIFT,
                        details={"check": "INTRINSICS_MISMATCH"},
                    )
                    raise RuntimeError(
                        "live D435i intrinsics do not match the approved calibration"
                    )
                intrinsics_checked = True
            orientation_delta = _imu_orientation_delta(calib, frame.gravity)
            if orientation_delta is not None:
                orientation_samples.append(
                    (t_rel, orientation_delta[0], orientation_delta[1])
                )
            while orientation_samples and t_rel - orientation_samples[0][0] > 3.0:
                orientation_samples.popleft()

            pose = tracker.update(estimator.estimate(frame))
            live = set(tracker.live_ids)
            new_ids = live - previous_live
            current_observed_tracks = {
                int(person.track_id)
                for person in pose.people
                if person.track_id is not None
            }
            credible_person_count = _credible_pose_count(
                pose.people, min_person_score=cfg.pose.min_score
            )
            previously_missing = set(recently_missing_tracks)
            returned_ids = _returned_associations(
                current_observed_tracks, recently_missing_tracks
            )
            new_observed_ids = current_observed_tracks - previous_observed_tracks
            replacement_seen = bool(
                new_observed_ids
                and (previously_missing - current_observed_tracks)
            )
            reassociated = bool(returned_ids or replacement_seen)

            # Collection consent is bound to one explicitly selected tracker
            # association.  Do not persist opportunistically detected staff,
            # visitors, or neighbouring patients under that participant ID.
            scope_violation = _collection_scope_violation(
                participant_id,
                current_observed_tracks,
                credible_person_count=credible_person_count,
            )
            if scope_violation == "UNEXPECTED_PERSON_IN_EMPTY_ROOM":
                abort_code = scope_violation
                _health_call(
                    health.mark_unavailable,
                    HealthReason.PRIVACY_SCOPE_VIOLATION,
                    details={"condition": "PERSON_IN_EMPTY_ROOM"},
                )
                raise RuntimeError(
                    "person detected during empty-room protocol; stopped before body data was written"
                )
            if scope_violation == "UNAPPROVED_PERSON_PRESENT":
                abort_code = scope_violation
                _health_call(
                    health.mark_unavailable,
                    HealthReason.PRIVACY_SCOPE_VIOLATION,
                    details={"condition": "MULTIPLE_PEOPLE"},
                )
                raise RuntimeError(
                    "more than one person detected; stopped before non-target body data was written"
                )

            for missing_id in previous_observed_tracks - current_observed_tracks:
                recently_missing_tracks[missing_id] = t_rel
            for track_id in returned_ids:
                recently_missing_tracks.pop(track_id, None)
            recently_missing_tracks = {
                track_id: missing_t
                for track_id, missing_t in recently_missing_tracks.items()
                if track_id in live and track_id not in current_observed_tracks
            }
            for track_id in sorted(new_ids | returned_ids):
                epoch_counter += 1
                epochs[track_id] = epoch_counter
            if reassociated:
                _write_session(
                    recorder.write_telemetry,
                    {
                        "kind": "association_reset",
                        "returned_track_ids": sorted(returned_ids),
                        "new_track_ids": sorted(new_observed_ids),
                        "association_epochs": [
                            {
                                "track_id": track_id,
                                "association_epoch": epochs[track_id],
                            }
                            for track_id in sorted(new_ids | returned_ids)
                            if track_id in epochs
                        ],
                    },
                )

            # Any missed observation breaks identity continuity, even if the
            # tracker later reuses the same numeric ID.  Clear every stateful
            # filter before processing the returned pose and require the
            # operator to bind the participant again.
            if returned_ids:
                if smoother is not None:
                    for track_id in returned_ids:
                        smoother.forget(track_id)
                extractor.retain_only(live - returned_ids)
                machine.retain_only(live - returned_ids)
                for track_id in returned_ids:
                    temporal.reset(track_id)
                if selected_track_id in returned_ids:
                    selected_track_id = None
                    typer.echo("target association was lost; select the participant again")
            if selected_track_id is not None and selected_track_id not in current_observed_tracks:
                temporal.mark_unobserved(selected_track_id, pose.t)
                selected_track_id = None
                typer.echo("target is not observable; select the participant again when visible")

            if smoother is not None:
                smoother.retain_only(tracker.live_ids)
                pose = pose.with_people(
                    tuple(
                        person.with_keypoints(
                            smoother.smooth(person.track_id, pose.t, person.keypoints)
                        )
                        for person in pose.people
                    )
                )
            pose = attach_depth_heights(
                pose,
                frame,
                extractor.ground,
                min_score=cfg.pose.min_keypoint_score,
            )
            previous_observed_tracks = current_observed_tracks
            previous_live = live

            extractor.retain_only(live)
            machine.retain_only(live)
            temporal.retain_only(
                {selected_track_id} if selected_track_id is not None else set()
            )
            observed_ids: set[int] = set()
            people_rows = []
            # A participant-free session is the explicit empty-room protocol;
            # no person there is a valid observation. In a participant session,
            # losing every usable pose is degraded coverage, never a negative.
            pose_confident = participant_id is None and not pose.people
            states = {}
            target_depth_fraction: float | None = None
            target_depth_provenance: dict[str, float | None] | None = None
            for person in pose.people:
                if person.track_id is None:
                    continue
                features = extractor.extract(person, pose.t)
                if features is None:
                    continue
                observed_ids.add(person.track_id)
                events = machine.update_all(features)
                bed_snapshot = machine.bed_snapshot_of(person.track_id, pose.t)
                states[person.track_id] = machine.state_of(person.track_id)
                is_target = (
                    participant_id is not None
                    and selected_track_id == person.track_id
                )
                if not is_target:
                    continue
                # The training validity mask must use the same full gate as
                # the bed activity machine.  In particular, an excluded-zone
                # observation is unavailable even if its keypoints/geometry
                # happen to look numerically good.
                feature_valid = _bed_observation_valid(bed_snapshot)
                pose_confident = pose_confident or feature_valid
                for event in events:
                    sink.emit(event)
                    event_count += 1
                    event_record = {
                        "event_id": "evt_" + uuid.uuid4().hex[:16],
                        "type": event.type,
                        "track_id": int(event.track_id),
                        "association_epoch": int(
                            epochs.get(int(event.track_id), 0)
                        ),
                        "frame_index": int(frame.index),
                        "source_t_s": _finite_or_none(frame.t),
                        "severity": int(event.severity),
                        "zone": event.zone,
                        "trigger_t_s": _finite_or_none(event.t_trigger),
                        "alert_t_s": _finite_or_none(event.t_alert),
                        "evidence": event.evidence,
                    }
                    pending_events.append(event_record)
                    # Feature rows are intentionally capped at 10 Hz, but an
                    # event must survive a crash or abort before the next row.
                    # Keep this append-only event copy in telemetry as the
                    # durable, immediate record; the frame copy retains local
                    # feature context for offline analysis.
                    _write_session(
                        recorder.write_telemetry,
                        {"kind": "shadow_event", **event_record},
                    )
                measured_ankle = _standing_depth_ankle_height(
                    person,
                    features,
                    states[person.track_id],
                    min_score=cfg.pose.min_keypoint_score,
                    feature_valid=feature_valid,
                )
                if measured_ankle is not None:
                    ankle_heights.append((t_rel, measured_ankle))
                provenance = _depth_provenance(
                    person, min_score=cfg.pose.min_keypoint_score
                )
                target_depth_fraction = provenance["joint_depth_valid_fraction"]
                target_depth_provenance = provenance
                temporal_summary = temporal.update(
                    features,
                    valid=feature_valid,
                    context={
                        "edge_velocity_mps": (
                            bed_snapshot.edge_velocity_mps if bed_snapshot else None
                        ),
                        "cusum_z": bed_snapshot.cusum_z if bed_snapshot else None,
                        "cusum_g": bed_snapshot.cusum_g if bed_snapshot else None,
                        "cusum_onset": (
                            1.0
                            if bed_snapshot and "cusum_onset" in bed_snapshot.reasons
                            else 0.0
                        ),
                        "cusum_armed": (
                            1.0 if bed_snapshot and bed_snapshot.cusum_armed else 0.0
                        ),
                        "joint_depth_valid_fraction": provenance[
                            "joint_depth_valid_fraction"
                        ],
                        "shoulder_depth_available": provenance[
                            "shoulder_depth_available"
                        ],
                        "torso_depth_available": provenance[
                            "torso_depth_available"
                        ],
                        "imu_pitch_delta_deg": (
                            orientation_delta[0] if orientation_delta else None
                        ),
                        "imu_roll_delta_deg": (
                            orientation_delta[1] if orientation_delta else None
                        ),
                        "episode_reclined": (
                            1.0
                            if bed_snapshot and bed_snapshot.phase == "RECLINED"
                            else 0.0
                        ),
                    },
                )
                people_rows.append(
                    _collection_person_record(
                        person,
                        features,
                        machine,
                        epochs.get(person.track_id, 0),
                        temporal_summary,
                    )
                )
            if (
                participant_id is not None
                and selected_track_id is not None
                and selected_track_id not in observed_ids
            ):
                temporal.mark_unobserved(selected_track_id, pose.t)
                selected_track_id = None
                typer.echo("target features unavailable; select the participant again")
            machine.mark_frame_unobserved(pose.t, live, observed_ids)

            depth_fraction = 0.0
            if frame.depth_raw is not None and frame.depth_raw.size:
                depth_fraction = float(np.count_nonzero(frame.depth_raw)) / float(
                    frame.depth_raw.size
                )
            while ankle_heights and t_rel - ankle_heights[0][0] > 3.0:
                ankle_heights.popleft()
            ankle_values = [value for _, value in ankle_heights]
            ankle_span = (
                ankle_heights[-1][0] - ankle_heights[0][0]
                if len(ankle_heights) >= 2
                else 0.0
            )
            # Compare with the baseline measured during this exact mount's
            # approved staff preflight; the ankle keypoint is not floor zero.
            # A two-second window avoids one-frame stereo-edge aborts.
            ankle_ready = len(ankle_values) >= 15 and ankle_span >= 2.0
            calibration_valid = (
                not ankle_ready
                or drift_check(
                    ankle_values,
                    tolerance_m=0.10,
                    baseline_m=calib.ankle_height_baseline_m,
                )
            )
            calibration_check = (
                "NOT_CHECKED"
                if not ankle_values
                else (
                    "WARMING"
                    if not ankle_ready
                    else ("OK" if calibration_valid else "DRIFT")
                )
            )
            ankle_mean = (
                round(float(np.mean(ankle_values)), 4) if ankle_values else None
            )
            orientation_span = (
                orientation_samples[-1][0] - orientation_samples[0][0]
                if len(orientation_samples) >= 2
                else 0.0
            )
            orientation_ready = (
                len(orientation_samples) >= 15 and orientation_span >= 2.0
            )
            pitch_delta = (
                float(np.median([value[1] for value in orientation_samples]))
                if orientation_samples
                else None
            )
            roll_delta = (
                float(np.median([value[2] for value in orientation_samples]))
                if orientation_samples
                else None
            )
            imu_missing = not orientation_samples and t_rel >= 2.0
            orientation_valid = (
                not imu_missing
                and (
                    not orientation_ready
                    or (
                        abs(float(pitch_delta)) <= 2.0
                        and abs(float(roll_delta)) <= 2.0
                    )
                )
            )
            if orientation_ready and orientation_valid:
                imu_orientation_ready = True
            orientation_check = (
                "MISSING"
                if imu_missing
                else (
                    "WARMING"
                    if not orientation_ready
                    else ("OK" if orientation_valid else "DRIFT")
                )
            )
            calibration_valid = calibration_valid and orientation_valid
            complete_wall = time.monotonic()
            if last_complete_wall is not None and complete_wall > last_complete_wall:
                instantaneous_fps = 1.0 / (complete_wall - last_complete_wall)
                fps_ema = (
                    instantaneous_fps
                    if fps_ema is None
                    else 0.9 * fps_ema + 0.1 * instantaneous_fps
                )
            last_complete_wall = complete_wall
            try:
                disk_free_bytes = _collection_disk_free_bytes(recorder.path)
            except OSError as exc:
                abort_code = "DISK_ERROR"
                _health_call(
                    health.mark_unavailable,
                    HealthReason.DISK_ERROR,
                )
                raise RuntimeError(
                    "cannot inspect free space in the approved output volume"
                ) from exc
            if t_rel - last_heartbeat_t >= 10.0:
                _write_session(
                    recorder.write_telemetry,
                    {
                        "kind": "heartbeat",
                        "frames_seen": frames_seen,
                        "effective_fps": round(fps_ema, 3) if fps_ema is not None else None,
                        "disk_free_bytes": int(disk_free_bytes),
                        "monitoring": health.snapshot(),
                    }
                )
                last_heartbeat_t = t_rel
            if disk_free_bytes < 512 * 1024 * 1024:
                abort_code = "DISK_SPACE_LOW"
                _health_call(
                    health.mark_unavailable,
                    HealthReason.DISK_ERROR,
                    details={"disk_free_bytes": int(disk_free_bytes)},
                )
                raise RuntimeError("less than 512 MiB free in the approved output volume")
            health_details = {
                    "people": len(people_rows),
                    "frame_depth_valid_fraction": round(depth_fraction, 4),
                    "target_joint_depth_fraction": (
                        round(target_depth_fraction, 4)
                        if target_depth_fraction is not None
                        else None
                    ),
                    "ankle_drift_samples": len(ankle_heights),
                    "ankle_drift_span_s": round(ankle_span, 3),
                    "ankle_height_mean_m": ankle_mean,
                    "calibration_check": calibration_check,
                    "imu_orientation_check": orientation_check,
                    "imu_pitch_delta_deg": (
                        round(pitch_delta, 3) if pitch_delta is not None else None
                    ),
                    "imu_roll_delta_deg": (
                        round(roll_delta, 3) if roll_delta is not None else None
                    ),
                    "effective_fps": round(fps_ema, 3) if fps_ema is not None else None,
                    "disk_free_bytes": int(disk_free_bytes),
                }
            if participant_id is not None and selected_track_id is None:
                _health_call(
                    health.mark_unavailable,
                    HealthReason.TARGET_NOT_BOUND,
                    details=health_details,
                )
            else:
                depth_valid = (
                    depth_fraction > 0.05
                    if participant_id is None
                    else _target_depth_healthy(target_depth_provenance or {})
                )
                _health_call(
                    health.observe_frame,
                    pose_confident=pose_confident,
                    depth_valid=depth_valid,
                    reassociated=reassociated,
                    calibration_valid=calibration_valid,
                    details=health_details,
                )
            # The planned model rate is 10 Hz.  Processing may run faster for
            # tracking/health, but writing keyed temporal rows more often adds
            # storage and frame correlation without useful signal.
            if t_rel + 1e-9 >= next_derived_write_t:
                monitoring_snapshot = health.snapshot()
                _write_session(
                    recorder.write_derived,
                    {
                        "kind": "frame_observation",
                        "frame_index": int(frame.index),
                        "source_t_s": _finite_or_none(frame.t),
                        "monitoring": monitoring_snapshot,
                        "people": people_rows,
                        "events": list(pending_events),
                    },
                )
                if _is_labeled_available_target_row(
                    participant_id,
                    selected_track_id,
                    epochs.get(selected_track_id, 0)
                    if selected_track_id is not None
                    else 0,
                    people_rows,
                    monitoring_snapshot,
                    current_phase_by_association,
                ):
                    labeled_available_rows += 1
                pending_events.clear()
                while next_derived_write_t <= t_rel + 1e-9:
                    next_derived_write_t += 0.1
            if not calibration_valid:
                abort_code = "CALIBRATION_DRIFT"
                raise RuntimeError(
                    "calibration drift or missing IMU detected; "
                    "session aborted -- secure the mount and recalibrate"
                )

            key = -1
            if view:
                canvas = render_skeleton(
                    pose,
                    min_keypoint_score=cfg.pose.min_keypoint_score,
                    states=states,
                    fps=None,
                )
                health_snapshot = health.snapshot()
                status = health_snapshot["status"]
                reasons = ",".join(health_snapshot["reasons"]) or "OK"
                status_color = (
                    (80, 220, 80)
                    if status == "AVAILABLE"
                    else ((0, 210, 255) if status == "DEGRADED" else (50, 50, 230))
                )
                cv2.putText(
                    canvas,
                    "RESEARCH SHADOW | "
                    + status
                    + " | "
                    + reasons
                    + " | %.1f fps" % (fps_ema or 0.0)
                    + " | depth %.0f%%" % (100.0 * depth_fraction)
                    + " | disk %.1f GB" % (disk_free_bytes / (1024**3)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                candidates = sorted(observed_ids)
                if selected_track_id not in observed_ids:
                    selected_track_id = None
                target_text = (
                    "LABEL TARGET: track " + str(selected_track_id)
                    if selected_track_id is not None
                    else "LABEL TARGET: NONE  ([ / ] to select)"
                )
                cv2.putText(
                    canvas,
                    target_text,
                    (12, max(24, canvas.shape[0] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("["), ord("]")):
                    if candidates:
                        next_track_id, target_changed = _next_collection_target(
                            candidates,
                            selected_track_id,
                            step=-1 if key == ord("[") else 1,
                        )
                        if target_changed:
                            selected_track_id = next_track_id
                            # Pre-binding observations must not seed CUSUM, dwell,
                            # velocity or smoothing state for the recorded target.
                            # Begin one clean causal association at the operator's
                            # explicit binding action.
                            if smoother is not None:
                                smoother.forget(selected_track_id)
                            extractor.retain_only(live - {selected_track_id})
                            machine.retain_only(live - {selected_track_id})
                            temporal.reset(selected_track_id)
                            _write_session(
                                recorder.write_telemetry,
                                {
                                    "kind": "target_binding",
                                    "track_id": selected_track_id,
                                    "association_epoch": epochs[selected_track_id],
                                },
                            )
                            typer.echo("label target: track " + str(selected_track_id))
                        else:
                            typer.echo(
                                "label target unchanged: track "
                                + str(selected_track_id)
                            )
                    else:
                        selected_track_id = None
                        typer.echo("label target: none visible")
                elif key in phase_keys:
                    marker = _collection_marker(
                        "phase_marker",
                        phase_keys[key],
                        selected_track_id,
                        observed_ids,
                        epochs,
                    )
                    if marker is None:
                        typer.echo("marker ignored: select one visible target with [ / ]")
                    else:
                        _write_session(recorder.write_label, marker)
                        association = marker["associations"][0]
                        current_phase_by_association[
                            (
                                association["track_id"],
                                association["association_epoch"],
                            )
                        ] = marker["phase"]
                elif key in context_keys:
                    marker = _collection_marker(
                        "context_marker",
                        context_keys[key],
                        selected_track_id,
                        observed_ids,
                        epochs,
                    )
                    if marker is None:
                        typer.echo("marker ignored: select one visible target with [ / ]")
                    else:
                        _write_session(recorder.write_label, marker)
                        if context_keys[key] == "BED_ARTICULATION":
                            abort_code = "BED_ARTICULATION_RECALIBRATE"
                            raise RuntimeError(
                                "bed articulation changes calibrated geometry; "
                                "session aborted -- recalibrate and start a new session"
                            )
                elif key == 27:
                    stop_reason = HealthReason.REQUESTED_STOP
                    requested_abort = True
                    processing_frame = False
                    break
                elif key == ord("q"):
                    stop_reason = HealthReason.PROTOCOL_COMPLETE
                    processing_frame = False
                    break

            frames_seen += 1
            processing_frame = False
            if max_frames and frames_seen >= max_frames:
                stop_reason = HealthReason.SESSION_LIMIT
                break
            if seconds and t_rel >= seconds:
                stop_reason = HealthReason.SESSION_LIMIT
                break

        if stop_reason is None:
            abort_code = "SOURCE_ENDED"
            if watchdog_stop is not None:
                watchdog_stop.set()
            if watchdog_thread is not None:
                watchdog_thread.join(timeout=2.0)
            _health_call(health.mark_unavailable, HealthReason.SOURCE_ENDED)
            raise RuntimeError("live D435i source ended unexpectedly")
        if watchdog_stop is not None:
            watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)
        completion_issues = _collection_completion_issues(
            participant_id,
            labeled_available_rows=labeled_available_rows,
            imu_orientation_ready=imu_orientation_ready,
        )
        if not requested_abort and completion_issues:
            completion_abort_reason = "PROTOCOL_INCOMPLETE"
            stop_reason = HealthReason.PROTOCOL_INCOMPLETE
            typer.echo(
                "protocol incomplete: " + ", ".join(completion_issues)
            )
        # Any failure from the terminal health write or recorder finalization is
        # not a camera disconnect.  A lower-level OSError still overrides this
        # with DISK_ERROR in _write_session.
        abort_code = "FINALIZATION_ERROR"
        _health_call(health.mark_unavailable, stop_reason)
        _finalize_collection_recorder(
            recorder,
            requested_abort=requested_abort,
            abort_reason=completion_abort_reason,
        )
    except KeyboardInterrupt:
        if watchdog_stop is not None:
            watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)
        if health is not None:
            try:
                _health_call(health.mark_unavailable, HealthReason.OPERATOR_STOP)
            except Exception:
                pass
        if recorder is not None and recorder.status == "incomplete":
            recorder.abort("OPERATOR_INTERRUPT")
        typer.echo("aborted by operator; the flushed derived prefix is retained")
        return
    except Exception:
        if watchdog_stop is not None:
            watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)
        classified_abort_code = _collection_exception_abort_code(
            abort_code, processing_frame=processing_frame
        )
        if classified_abort_code == "CAMERA_DISCONNECTED" and health is not None:
            abort_code = classified_abort_code
            try:
                _health_call(
                    health.mark_unavailable,
                    HealthReason.CAMERA_DISCONNECTED,
                )
            except Exception:
                pass
        else:
            abort_code = classified_abort_code
        if recorder is not None and recorder.status == "incomplete":
            recorder.abort(abort_code)
        raise
    finally:
        if watchdog_stop is not None:
            watchdog_stop.set()
        if watchdog_thread is not None and watchdog_thread.is_alive():
            watchdog_thread.join(timeout=2.0)
        src.close()
        if sink is not None:
            sink.close()
        if view:
            cv2.destroyAllWindows()

    if requested_abort:
        final_status = "aborted (requested/safety stop)"
    elif completion_abort_reason is not None:
        final_status = "aborted (protocol incomplete)"
    else:
        final_status = "complete"
    typer.echo(
        final_status + ": " + str(frames_seen) + " frames, " + str(event_count)
        + " shadow events -> " + str(recorder.path if recorder else out_root)
    )


@app.command(name="compare-bed-exit")
def compare_bed_exit(
    sessions: list[Path] = typer.Argument(
        ..., help="Two or more complete derived-only ses_* directories."
    ),
    group_by: str = typer.Option(
        "subject", help="Cross-validation group: subject (preferred) or session (development only)."
    ),
    out_dir: Path = typer.Option(
        None,
        help=(
            "Optional new directory for aggregate report + fitted research models; "
            "it must be beneath an external custodian-approved site root."
        ),
    ),
) -> None:
    """Compare logistic/tree temporal baselines on causal onsite summaries.

    This reports out-of-fold frame-level development metrics for activity phase
    and 5/10/20-second exit horizons. It does not coalesce alert events, measure
    clinical response, or enable either model in the live dashboard.
    """
    import hashlib
    import json
    import platform
    from collections import Counter
    from dataclasses import replace
    from datetime import datetime, timezone
    from importlib.metadata import PackageNotFoundError, version

    from ahfd.ml.bed_dataset import HORIZONS_S, load_bed_sessions
    from ahfd.ml.temporal import compare_grouped

    if group_by not in ("subject", "session"):
        raise typer.BadParameter("--group-by must be subject or session")
    dataset = load_bed_sessions(sessions)
    if group_by == "subject" and len(dataset.participant_ids) < 2:
        raise typer.BadParameter(
            "subject-held-out comparison needs at least two explicit participants"
        )
    if group_by == "session":
        typer.echo(
            "WARNING: session-held-out scores are development-only; repeated people "
            "can still leak person-specific movement."
        )
    if out_dir is not None:
        _validate_comparator_output(out_dir, sessions)
        out_dir.mkdir(parents=True, exist_ok=False)

    typer.echo(
        "FRAME-LEVEL RESEARCH BASELINE ONLY -- not an event-level or clinical validation"
    )
    typer.echo(
        "loaded "
        + str(len(dataset.session_ids))
        + " session(s), "
        + str(len(dataset.participant_ids))
        + " participant(s), sampled causally at "
        + format(dataset.sample_hz, ".1f")
        + " Hz"
    )

    tasks = [("phase", dataset.phase)] + [
        ("exit_" + str(horizon) + "s", dataset.for_horizon(horizon))
        for horizon in HORIZONS_S
    ]
    report = {
        "schema": "ahfd.bed_exit_comparator.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_shadow_only": True,
        "live_inference_enabled": False,
        "metric_scope": "out_of_fold_frame_level",
        "group_by": group_by,
        "session_ids": list(dataset.session_ids),
        "participant_ids": list(dataset.participant_ids),
        "inputs": [],
        "tasks": {},
    }
    dependency_versions = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "joblib"):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = "unavailable"
    report["dependency_versions"] = dict(sorted(dependency_versions.items()))
    for session_path in sessions:
        manifest_path = Path(session_path) / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        report["inputs"].append(
            {
                "session_id": manifest["session_id"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "code_hash": manifest["inputs"]["code_hash"],
                "config_sha256": manifest["inputs"]["config_sha256"],
                "calibration_sha256": manifest["inputs"]["calibration_sha256"],
            }
        )

    successful = 0
    for task_name, supervised in tasks:
        counts = Counter(supervised.labels)
        typer.echo(
            "\n"
            + task_name
            + ": "
            + str(len(supervised))
            + " eligible rows; "
            + ", ".join(label + "=" + str(count) for label, count in sorted(counts.items()))
        )
        try:
            result = compare_grouped(
                *supervised.compare_args(),
                group_by=group_by,
            )
        except ValueError as exc:
            typer.echo("  SKIPPED: " + str(exc))
            report["tasks"][task_name] = {
                "status": "skipped",
                "reason": str(exc),
                "n_rows": len(supervised),
                "class_counts": dict(sorted(counts.items())),
            }
            continue

        successful += 1
        score_rows = []
        for score in result.scores:
            per_group_accuracy = dict(score.per_group_accuracy)
            row = {
                "model": score.model_kind,
                "accuracy": score.accuracy,
                "balanced_accuracy": score.balanced_accuracy,
                "macro_f1": score.macro_f1,
                "macro_average_precision": score.macro_average_precision,
                "brier_score": score.brier_score,
                "per_class_average_precision": dict(score.per_class_average_precision),
                "per_group_accuracy": per_group_accuracy,
                "group_macro_accuracy": (
                    sum(per_group_accuracy.values()) / len(per_group_accuracy)
                    if per_group_accuracy
                    else None
                ),
                "group_macro_unit": group_by,
                "positive_class": score.positive_class,
                "positive_precision": score.positive_precision,
                "positive_recall": score.positive_recall,
                "false_positive_rate": score.false_positive_rate,
            }
            score_rows.append(row)
            line = (
                "  "
                + score.model_kind.ljust(9)
                + " macro-F1="
                + format(score.macro_f1, ".3f")
                + " bal-acc="
                + format(score.balanced_accuracy, ".3f")
                + " AP="
                + format(score.macro_average_precision, ".3f")
                + " Brier="
                + format(score.brier_score, ".3f")
            )
            if score.positive_class == "exit":
                line += (
                    " exit-precision="
                    + format(score.positive_precision or 0.0, ".3f")
                    + " exit-recall="
                    + format(score.positive_recall or 0.0, ".3f")
                    + " FPR="
                    + format(score.false_positive_rate or 0.0, ".3f")
                )
            typer.echo(line)

        report["tasks"][task_name] = {
            "status": "ok",
            "n_rows": len(supervised),
            "class_counts": dict(sorted(counts.items())),
            "classes": list(result.classes),
            "folds": [
                {
                    "train_groups": list(fold.train_groups),
                    "test_groups": list(fold.test_groups),
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                }
                for fold in result.folds
            ],
            "scores": score_rows,
        }
        if out_dir is not None:
            import joblib

            for model_kind, model in result.models.items():
                horizon_s = (
                    int(task_name.removeprefix("exit_").removesuffix("s"))
                    if task_name.startswith("exit_")
                    else None
                )
                model.metadata = replace(
                    model.metadata,
                    task=task_name,
                    horizon_s=horizon_s,
                    input_manifest_sha256=tuple(
                        item["manifest_sha256"] for item in report["inputs"]
                    ),
                    dependency_versions=tuple(sorted(dependency_versions.items())),
                )
                joblib.dump(model, out_dir / (task_name + "_" + model_kind + ".joblib"))

    if successful == 0:
        if out_dir is not None:
            (out_dir / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise typer.BadParameter(
            "no task had enough class-complete independent groups for comparison"
        )
    if out_dir is not None:
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        typer.echo("\nwrote research-only comparator artifacts to " + str(out_dir))
    typer.echo(
        "\nNext gate: coalesce predictions into events and measure false alerts per "
        "available hour plus lead time before considering any live shadow inference."
    )


@app.command()
def extract(
    source: str = typer.Argument(..., help="Source URI: file://, seq://, webcam://, bag://"),
    out: Path = typer.Argument(..., help="Output tracks.jsonl path."),
    config: Path = typer.Option(None, help="Path to a YAML config (for the pose backend)."),
    max_frames: int = typer.Option(0, help="Stop after N frames. 0 = whole clip."),
    depth: bool = typer.Option(
        False, "--depth",
        help="Also measure each joint's height above the floor from depth and "
        "store it in the tracks (adds the dh_* features to training). Needs a "
        "RealSense depth source: bag://<clip>.bag or rs://.",
    ),
    height: float = typer.Option(2.5, help="Camera mount height above the floor (m), for --depth."),
    pitch: float = typer.Option(
        MOUNT_PITCH_DEG,
        help="Mount downtilt in degrees, for --depth on a camera with no IMU (D435f). "
        "Defaults to MOUNT_PITCH_DEG (set once at the top of cli.py). Ignored when the "
        "clip carries IMU gravity (D435i).",
    ),
) -> None:
    """Run pose once over a clip and write keypoints to tracks.jsonl.

    This is the only command that touches imagery for a recorded clip: it reads
    the frames, extracts keypoints, and discards the pixels. Everything after
    this -- replay, sweep, eval -- works on the keypoints alone, so it is fast,
    deterministic, and privacy-safe. Run it once per clip; it is the slow step.

    With --depth (on a .bag recorded by `ahfd record-depth`), it also samples
    the aligned depth at each joint and stores the joint's height above the
    floor -- keypoint scalars, never a depth image. Those feed the dh_* depth
    features; clips extracted without --depth simply lack them and train as RGB.
    """
    from ahfd.capture import open_source
    from ahfd.io import TracksWriter
    from ahfd.pose import KeypointSmoother, build_estimator
    from ahfd.track import SimpleTracker

    cfg = load_config(config)

    if depth:
        # Depth needs a RealSense stream that carries the aligned depth + IMU.
        if source.startswith("bag://"):
            from ahfd.capture.realsense import BagSource

            src = BagSource(source[len("bag://") :], with_depth=True)
        elif source.startswith("rs://"):
            from ahfd.capture.realsense import RealSenseSource

            src = RealSenseSource(with_depth=True)
        else:
            raise typer.BadParameter(
                "--depth needs a RealSense source (bag://<clip>.bag or rs://); got " + source
            )
    else:
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
    if depth:
        import numpy as _np

        from ahfd.geometry.depth_height import keypoint_heights_from_depth
        from ahfd.geometry.ground import GroundPlane

        typer.echo("depth:   ON -- storing joint heights above floor (mount " + str(height) + " m)")

    writer = TracksWriter(out)
    n = 0
    depth_frames = 0
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
            if (
                depth
                and frame.depth_raw is not None
                and frame.intrinsics is not None
                and pose.people
            ):
                # Ground tilt from the live IMU (D435i), or the fixed --pitch when
                # the recording has no gravity (D435f). depth_raw = measurement
                # depth (no hole filling), the right one for a per-joint reading.
                if frame.gravity is not None:
                    gp = GroundPlane.from_gravity(frame.intrinsics, height, _np.asarray(frame.gravity))
                elif pitch is not None:
                    gp = GroundPlane(intrinsics=frame.intrinsics, height_m=height,
                                     pitch_deg=float(pitch), roll_deg=0.0)
                else:
                    gp = None
                if gp is not None:
                    depth_m = frame.depth_raw.astype(_np.float32) * float(frame.depth_scale)
                    pose = pose.with_people(
                        tuple(
                            p.with_heights(
                                keypoint_heights_from_depth(
                                    p.keypoints, p.scores, depth_m, frame.intrinsics, gp
                                )
                            )
                            for p in pose.people
                        )
                    )
                    depth_frames += 1
            writer.write(pose)
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        src.close()
        writer.close()

    typer.echo("wrote " + str(writer.count) + " frames to " + str(out))
    if depth:
        if depth_frames:
            typer.echo("depth:   heights attached on " + str(depth_frames) + " frames")
        else:
            typer.echo(
                "WARNING: --depth was set but no heights were stored. Needs depth + a "
                "tilt: an IMU (D435i) or --pitch <deg> (D435f). Is this a depth .bag "
                "from `ahfd record-depth`?"
            )


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
                for event in machine.update_all(features):
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
            for event in machine.update_all(features):
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
                risk = typer.prompt(
                    "  risk_level (legacy metadata; dashboard mode is selected at runtime)",
                    default="unknown",
                )
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

    Retraining is just re-running this on more labelled+extracted clips. Test
    scores hold out whole people (numeric clip suffixes) or recordings, never
    random frames from the same recording. Without an honest group hold-out,
    only training accuracy is reported, not a generalisation score.
    """
    from collections import Counter

    from ahfd.geometry.calibration import load_calibration
    from ahfd.ml.posture import build_dataset, train

    calib = load_calibration(calibration)
    rows, labels_list, groups, used, skipped = build_dataset(labels, tracks, calib)

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

    result = train(rows, labels_list, groups=groups, max_depth=max_depth)

    typer.echo("")
    if result.split_done:
        typer.echo(
            "group-held-out split (" + result.split_unit + "): "
            + str(result.n_train) + " train, "
            + str(result.n_test) + " test"
        )
        typer.echo("training groups: " + ", ".join(result.train_groups))
        typer.echo("held-out groups: " + ", ".join(result.test_groups))
        typer.echo("TEST accuracy: " + format(result.accuracy, ".3f"))
    else:
        typer.echo(
            "no honest group-held-out test: " + result.split_reason
            + " -- trained on all, reporting TRAINING accuracy (not a real score)"
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
        "NOTE: reliable evaluation needs varied people and sessions. The saved "
        "model is trained on all labelled rows; any TEST score above comes from "
        "a separate group-held-out fit."
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
    clip: Path = typer.Argument(..., help="Clip to label: an .mp4, or a depth .db3/.bag (auto colour-exported)."),
    out: Path = typer.Option(
        None, help="Posture JSON to write. Default: data/postures/<clip>.json."
    ),
    config: Path = typer.Option(
        None,
        help="For .bag/.db3 only: YAML whose privacy.allow_raw_capture is true.",
    ),
    consent: bool = typer.Option(
        False,
        "--i-understand-raw-capture",
        help="Required to label a .db3/.bag directly (it writes a temp colour mp4).",
    ),
) -> None:
    """Scrub a clip and mark posture segments -- a little video labeller.

    Opens the clip in a window with a timeline. Scrub with a/d (+/-1s), ,/.
    (+/-1 frame), [ / ] (+/-5s). Press 's' (or SPACE) to mark the START of a
    hold, scrub to its end, press 'f', then a number to pick the posture
    (1 upright, 2 sitting, 3 in_bed, 4 on_ground). 'u' undoes, 'w' saves, 'q'
    saves and quits. Leave gaps between segments for transitions -- they are
    excluded from training on purpose.

    A depth .db3/.bag can be labelled directly: the colour stream is exported to
    a temporary .mp4 (the labeller cannot scrub a .db3), labelled, then deleted.
    The label's <clip>.json stem matches the <clip>.jsonl tracks either way.

    Only the label JSON is kept; any temp mp4 is removed. Run it on the laptop
    (it needs a display), not the headless Jetson.
    """
    from ahfd.annotate import run_labeler

    if not clip.exists():
        raise typer.BadParameter("clip not found: " + str(clip))
    out_path = out or (Path("data/postures") / (clip.stem + ".json"))

    # A depth recording is not a video the labeller can open, so colour-export
    # it to a temp .mp4 first (transparently) and remove it afterwards.
    tmp_mp4 = None
    label_src = clip
    if clip.suffix.lower() in (".db3", ".bag"):
        if not consent:
            raise typer.BadParameter(
                "labelling a .db3/.bag writes a temporary colour mp4; pass "
                "--i-understand-raw-capture (consented staged data only)."
            )
        import tempfile

        from ahfd.debug.color_export import export_color

        cfg = load_config(config)
        tmp_mp4 = Path(tempfile.gettempdir()) / (clip.stem + "_label.mp4")
        typer.echo("colour-exporting " + clip.name + " -> temp mp4 for labelling ...")
        frames = export_color(
            clip,
            tmp_mp4,
            config_flag=cfg.privacy.allow_raw_capture,
            cli_flag=consent,
        )
        typer.echo("exported " + str(frames) + " frames; opening labeller ...")
        label_src = tmp_mp4

    typer.echo("labelling " + clip.stem + "  ->  " + str(out_path))
    typer.echo("  s/SPACE=start  f=end  1-4=posture  u=undo  w=save  q=save+quit")
    typer.echo("  click/drag the timeline to seek   c=cancel mark   r=remove segment here")
    try:
        n = run_labeler(label_src, out_path)
    finally:
        if tmp_mp4 is not None and tmp_mp4.exists():
            tmp_mp4.unlink()
            typer.echo("removed temp colour mp4")
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
    test_person: str = typer.Option(
        None,
        help="Fixed holdout instead of leave-one-out: train on every OTHER person, "
        "test on just this one (e.g. '--test-person 03' = train 01+02, test 03). "
        "Implies '--by person'. Use this for a clean train/validate/test split.",
    ),
    show_rules: bool = typer.Option(
        True, "--show-rules/--no-show-rules", help="Print the learned tree thresholds."
    ),
    rgb_only: bool = typer.Option(
        False, "--rgb-only",
        help="Ignore the depth (dh_*) features. Run with and without this on the "
        "same clips to isolate what depth adds (paired RGB vs RGB+depth).",
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

    Pass `--test-person NN` for a single fixed holdout instead: train on every
    other person and test on that one only (the plain train/validate/test split).
    """
    if by not in ("clip", "person"):
        raise typer.BadParameter("--by must be 'clip' or 'person'")

    from ahfd.geometry.calibration import load_calibration
    from ahfd.ml.compare import compare
    from ahfd.ml.posture import build_dataset

    calib = load_calibration(calibration)
    rows, labels_list, groups, used, skipped = build_dataset(labels, tracks, calib)

    # A fixed test person forces person-level grouping (you can't hold out a
    # single person while grouping by clip).
    holdout = None
    if test_person is not None:
        by = "person"
        digits = "".join(ch for ch in test_person if ch.isdigit())
        if not digits:
            raise typer.BadParameter("--test-person must contain a person number, e.g. '03'")
        holdout = "person_" + digits.zfill(2)

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
    group_set = set(groups)
    n_groups = len(group_set)
    if holdout is not None:
        if holdout not in group_set:
            raise typer.BadParameter(
                "--test-person resolves to " + repr(holdout) + " but that person "
                "has no labelled+extracted clips; found: "
                + ", ".join(sorted(group_set))
            )
        train_people = sorted(group_set - {holdout})
        typer.echo(
            "\nfixed holdout: train on " + ", ".join(train_people)
            + "  ->  test on " + holdout
        )
    else:
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

    result = compare(rows, labels_list, groups, holdout=holdout, rgb_only=rgb_only)
    if rgb_only:
        typer.echo("(RGB-only: depth dh_* features masked out)")

    typer.echo("")
    if holdout is not None:
        typer.echo(
            "train(" + ", ".join(train_people) + ") -> test(" + holdout + ") ranking "
            "(macro-F1 = balanced across postures; bal-acc = mean recall):"
        )
    else:
        typer.echo(
            "leave-one-" + unit + "-out ranking (macro-F1 = balanced across postures; "
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
    if holdout is None:
        typer.echo("")
        typer.echo("per-" + unit + " accuracy for " + best.name + ":")
        for clip, acc in best.per_clip_acc:
            typer.echo("  " + clip.ljust(24) + format(acc, ".3f"))

    if show_rules:
        typer.echo("")
        typer.echo("how a tree decides -- learned thresholds (depth-3 flat tree on all data):")
        typer.echo(result.tree_rules)

    if holdout is not None:
        typer.echo(
            "\nNOTE: fixed holdout -- trained on " + ", ".join(train_people)
            + " and scored ONLY on " + holdout + " (frames the model never saw). "
            "This is your train/validate split. When you add person 4, re-run with "
            "`--test-person 04` for the final held-out test; the deployed model is "
            "trained on ALL labelled people (`ahfd train-posture`)."
        )
    elif unit == "clip":
        typer.echo(
            "\nNOTE: leave-one-CLIP-out is cross-scenario, still one body/camera. "
            "Once >=2 people are labelled, `--by person` is the real cross-person test."
        )
    else:
        typer.echo(
            "\nNOTE: leave-one-PERSON-out -- each score is on a person the model "
            "never trained on. This is the honest generalisation number. More "
            "people tightens it further."
        )


@app.command()
def classify_view(
    source: str = typer.Argument(..., help="Depth recording to replay: bag://<clip>.db3 or the path."),
    calibration: Path = typer.Option(Path("calib/d435i.yaml"), help="Calibration YAML (ground + zones)."),
    height: float = typer.Option(2.5, help="Camera mount height above the floor (m)."),
    labels: Path = typer.Option(Path("data/postures_depth"), help="Dir of posture-label JSONs (for ground truth + training)."),
    tracks: Path = typer.Option(Path("data/tracks_depth"), help="Dir of extracted depth tracks (for training)."),
    backend: str = typer.Option("rtmo", help="Pose backend: rtmo | rtmpose | yolo."),
    runtime: str = typer.Option("openvino", help="Pose runtime: openvino (iGPU) | onnxruntime."),
    device: str = typer.Option("gpu", help="Pose device: gpu (iGPU) | cpu | cuda."),
    dmin: float = typer.Option(2.5, help="Near clip for the depth colour ramp (m)."),
    dmax: float = typer.Option(5.5, help="Far clip for the depth colour ramp (m)."),
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="Force re-processing instead of loading the saved cache (use after retraining).",
    ),
    config: Path = typer.Option(
        None,
        help="YAML whose privacy.allow_raw_capture is true for the RGB frame cache.",
    ),
    consent: bool = typer.Option(
        False,
        "--i-understand-raw-capture",
        help="Required because the scrubber cache persists JPEG RGB/depth panes.",
    ),
    pitch: float = typer.Option(
        MOUNT_PITCH_DEG,
        help="Mount downtilt in degrees for a D435f recording (no IMU). Defaults to "
        "MOUNT_PITCH_DEG. Ignored when the clip carries IMU gravity (D435i).",
    ),
) -> None:
    """Replay a clip with pose + posture on BOTH panes: RGB model vs RGB+depth.

    The colour pane shows the skeleton and the RGB-only model's posture call; the
    depth pane shows the skeleton and the RGB+depth model's call. The recording's
    own person is held out of training, so both calls are honest, and the
    ground-truth posture is shown when a label exists -- so you can watch where
    depth fixes a wrong RGB call (the sitting / nadir frames).
    """
    from ahfd.viz.classify_view import run_classify_viewer
    from ahfd.privacy import require_raw_capture

    cfg = load_config(config)
    require_raw_capture(
        config_flag=cfg.privacy.allow_raw_capture,
        cli_flag=consent,
    )

    run_classify_viewer(
        source,
        calibration=calibration,
        height_m=height,
        labels_dir=str(labels),
        tracks_dir=str(tracks),
        backend=backend,
        runtime=runtime,
        device=device,
        dmin=dmin,
        dmax=dmax,
        rebuild=rebuild,
        pitch=pitch,
        raw_config_flag=cfg.privacy.allow_raw_capture,
        raw_cli_flag=consent,
    )


@app.command()
def depth_view(
    source: str = typer.Option("rs://", help="rs:// for the live D435i, or a path to a .bag recording."),
    height: float = typer.Option(2.5, help="Camera mount height above the floor (m) -- used by the height-above-floor colour mode."),
    dmin: float = typer.Option(1.5, help="Near clip for the depth colour ramp (m)."),
    dmax: float = typer.Option(3.5, help="Far clip for the depth colour ramp (m)."),
    colormap: str = typer.Option("turbo", help="turbo | jet | viridis | inferno | magma."),
    raw: bool = typer.Option(False, "--raw", help="Show measurement depth (holes visible) instead of the hole-filled display depth."),
    color: bool = typer.Option(False, "--color", help="Show the RGB image beside the depth."),
    long_range: bool = typer.Option(False, "--long-range", help="Max the projector power for denser depth at 4-6 m, and point the colour ramp there. Live camera only."),
    max_range: float = typer.Option(6.0, help="Far depth cut in metres (the threshold filter). Raise it (e.g. 10) to see past 6 m; farther = noisier. Live camera only."),
    smooth: int = typer.Option(2, help="Spatial-filter strength 1-5 (smoothing passes). Higher = less noise but rounder edges. Live camera only."),
    pose: bool = typer.Option(False, "--pose", help="Overlay the skeleton + each joint's depth-measured height above the floor, and a coarse posture guess. Validates depth for posture before wiring it into detection."),
    backend: str = typer.Option("rtmo", help="Pose backend for --pose: rtmo | rtmpose | yolo."),
    runtime: str = typer.Option("openvino", help="Pose runtime for --pose: openvino (iGPU) | onnxruntime."),
    device: str = typer.Option("gpu", help="Pose device for --pose: gpu (iGPU) | cpu | cuda."),
    pitch: float = typer.Option(
        MOUNT_PITCH_DEG,
        help="Mount downtilt in degrees for the D435f (no IMU): supplies the tilt for "
        "height mode / --pose heights when there is no live gravity. Defaults to "
        "MOUNT_PITCH_DEG (set once at the top of cli.py). The D435i ignores it.",
    ),
) -> None:
    """Live depth viewer for tuning: denoised RealSense depth with a clamped colour ramp.

    Reuses the capture filter chain (disparity -> spatial -> temporal -> hole
    fill), so what you see is the same denoised depth the feature pipeline would
    consume. Clamping the colour ramp to the band the scene occupies (--dmin /
    --dmax) makes the head, torso and floor land on clearly different colours.

    Press 'f' in the window to switch to a height-above-floor colouring, built
    live from the IMU gravity vector and --height: it separates standing from
    on-ground even directly beneath the camera, where the image-only geometry
    breaks down. Keys: f mode, c colormap, i invert, a auto-range, [ ] far,
    , . near, h holes, v RGB, space pause, q quit.
    """
    from ahfd.viz.depth_view import run_depth_viewer

    run_depth_viewer(
        source,
        dmin=dmin,
        dmax=dmax,
        height_m=height,
        colormap=colormap,
        hole_filled=not raw,
        show_color=color,
        long_range=long_range,
        max_range_m=max_range,
        smooth=smooth,
        pose=pose,
        backend=backend,
        runtime=runtime,
        device=device,
        pitch=pitch,
    )


@app.command()
def estimate_ground(
    source: str = typer.Option("rs://", help="rs:// live, or bag://<clip>.db3 / a path."),
    frames: int = typer.Option(30, help="Frames to sample; the median is reported."),
) -> None:
    """Recover the mount tilt + height from the FLOOR in depth -- no IMU needed.

    Fits the floor plane in each depth frame and reports the median pitch / roll /
    height. Use it to self-calibrate the D435f (no IMU): compare the printed pitch
    to the D435i's IMU reading on the same mount, then use it as --pitch. Prints
    only -- it does not touch detection.
    """
    import numpy as np

    from ahfd.geometry.plane_fit import ground_from_floor

    path = source[len("bag://"):] if source.startswith("bag://") else source
    if source.startswith("bag://") or str(source).lower().endswith((".bag", ".db3")):
        from ahfd.capture.realsense import BagSource

        src = BagSource(path, with_depth=True)
    else:
        from ahfd.capture.realsense import RealSenseSource

        src = RealSenseSource(with_depth=True)

    typer.echo("sampling the floor plane from depth ...")
    got = []
    n = 0
    try:
        for frame in src:
            if frame.depth_raw is None or frame.intrinsics is None:
                continue
            depth_m = frame.depth_raw.astype(float) * float(frame.depth_scale)
            est = ground_from_floor(depth_m, frame.intrinsics)
            if est is not None:
                got.append(est)
            n += 1
            if n >= frames:
                break
    finally:
        src.close()

    if not got:
        typer.echo("no floor plane found -- aim the camera so the floor is visible.")
        return
    arr = np.array(got)
    typer.echo(
        "recovered from floor (median of %d):  pitch %.1f deg   roll %.1f deg   height %.2f m"
        % (len(got), float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))
    )
    typer.echo("compare the pitch to the D435i IMU on the same mount to validate, then pass it as --pitch.")


@app.command()
def compare_depth(
    dmin: float = typer.Option(1.0, help="Near clip for the depth colour ramp (m)."),
    dmax: float = typer.Option(6.0, help="Far clip for the depth colour ramp (m)."),
) -> None:
    """Live side-by-side depth from BOTH RealSense cameras, with quality gauges.

    Opens the two connected cameras (e.g. D435i + D435f) at once and shows each
    one's colourised depth with a centre-ROI readout -- fill %, mean distance,
    noise spread -- so you point both at the same target and see which gives
    denser, cleaner depth. Two projectors interfere (representative of a
    multi-camera ward); press 1/2 to toggle a camera's projector to isolate it.
    Keys: 1/2 projector on/off, q quit.
    """
    from ahfd.viz.compare_cameras import run_compare_cameras

    run_compare_cameras(dmin=dmin, dmax=dmax)


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
    config: Path = typer.Option(None, help="YAML whose privacy.allow_raw_capture is true."),
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

    Requires all three privacy switches: the YAML flag, ``AHFD_ALLOW_RAW=1``
    and the explicit command-line acknowledgement.  Staged volunteer capture
    must not weaken the same gate the rest of the application promises.
    """
    import cv2

    from ahfd.capture import open_source
    from ahfd.debug import RawRecorder
    from ahfd.privacy import require_raw_capture

    cfg = load_config(config)
    require_raw_capture(config_flag=cfg.privacy.allow_raw_capture, cli_flag=consent)

    src = open_source(source)
    typer.echo(
        "RECORDING (raw video) from " + source + " -> " + str(out)
        + "  " + str(src.meta.width) + "x" + str(src.meta.height)
    )
    typer.echo("press q in the window to stop. Extract keypoints, then delete this file.")

    recorder = RawRecorder(
        out, src.meta.width, src.meta.height, src.meta.fps,
        config_flag=cfg.privacy.allow_raw_capture, cli_flag=consent,
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


@app.command()
def record_depth(
    out: Path = typer.Argument(..., help="Output .bag path (stores colour + depth + IMU)."),
    config: Path = typer.Option(None, help="YAML whose privacy.allow_raw_capture is true."),
    seconds: float = typer.Option(0.0, help="Auto-stop after N seconds. 0 = until you press q."),
    rgb: str = typer.Option(
        "1080", help="RGB resolution: 1080 (1920x1080) or 720 (1280x720). 1080 keeps "
        "more detail on far/small subjects (multi-bed); use 720 for a close per-bed "
        "mount to halve the file size.",
    ),
    consent: bool = typer.Option(
        False,
        "--i-understand-raw-capture",
        help="Required. Confirms this session is consented raw capture.",
    ),
) -> None:
    """Record a depth `.bag` for a consented staged-fall session.

    Like `record`, but saves a RealSense `.bag` holding colour + DEPTH + IMU
    together -- the colour-only `record` (.mp4) path cannot carry depth. This is
    how you collect data for the depth-vs-RGB comparison: one `.bag` yields both
    the monocular and the depth features from the *same* frames, so the only
    difference between the two feature sets is depth itself.

    Raw capture: staged, consented volunteers only -- never patients, never a
    live ward. Delete the `.bag` once features are extracted.
    """
    from ahfd.debug.bag_writer import record_bag
    from ahfd.privacy import require_raw_capture

    cfg = load_config(config)
    require_raw_capture(config_flag=cfg.privacy.allow_raw_capture, cli_flag=consent)
    if out.suffix.lower() not in (".bag", ".db3"):
        raise typer.BadParameter(
            "output must be a .bag or .db3 file (it stores depth + IMU); got "
            + repr(out.suffix or out.name)
            + ". Newer librealsense builds require .db3 (rosbag2); older ones use .bag."
        )

    color_size = {"720": (1280, 720), "1080": (1920, 1080)}.get(str(rgb))
    if color_size is None:
        raise typer.BadParameter("--rgb must be 720 or 1080; got " + repr(rgb))

    typer.echo("RECORDING depth .bag -> " + str(out)
               + "  (colour %dx%d + depth + IMU-if-present)" % color_size)
    typer.echo("press q in the window to stop; the projector is on for depth.")
    n = record_bag(
        out,
        seconds=seconds,
        color_size=color_size,
        config_flag=cfg.privacy.allow_raw_capture,
        cli_flag=consent,
    )
    typer.echo("saved ~" + str(n) + " framesets to " + str(out))
    typer.echo("next: ahfd depth-view --pose --source " + str(out) + "  (verify), then extract features + DELETE the .bag")


@app.command()
def export_color(
    bag: Path = typer.Argument(..., help="Depth .bag/.db3 to pull the colour stream from."),
    out: Path = typer.Argument(..., help="Output .mp4 for the posture labeller."),
    config: Path = typer.Option(None, help="YAML whose privacy.allow_raw_capture is true."),
    consent: bool = typer.Option(
        False,
        "--i-understand-raw-capture",
        help="Required. Confirms this is consented staged data.",
    ),
) -> None:
    """Export a depth .bag/.db3's colour stream to an .mp4 so you can label it.

    The posture labeller (`ahfd label-postures`) reads an .mp4, not a .db3, so
    depth clips can't be labelled directly. This re-encodes just the colour into
    an .mp4 whose stem matches the clip, so the resulting <clip>.json label pairs
    with the <clip>.jsonl tracks. Label times line up because `compare-posture`
    normalises each clip's track timestamps to start at zero.

    Raw imagery: staged, consented volunteers only. Delete the .mp4 once the
    clip is labelled, the same as the .bag.
    """
    from ahfd.debug.color_export import export_color as _export
    from ahfd.privacy import require_raw_capture

    cfg = load_config(config)
    require_raw_capture(config_flag=cfg.privacy.allow_raw_capture, cli_flag=consent)
    if out.suffix.lower() != ".mp4":
        raise typer.BadParameter("output must be an .mp4 (the labeller reads video).")

    typer.echo("exporting colour  " + str(bag) + "  ->  " + str(out))
    n = _export(
        bag,
        out,
        config_flag=cfg.privacy.allow_raw_capture,
        cli_flag=consent,
    )
    typer.echo("wrote " + str(n) + " frames to " + str(out))
    typer.echo("next: ahfd label-postures " + str(out) + "   then DELETE the .mp4")


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


@app.command(name="measure-ankle-baseline")
def measure_ankle_baseline(
    calibration: Path = typer.Argument(
        ..., help="External onsite calibration YAML to update."
    ),
    config: Path = typer.Option(
        Path("configs/onsite_collection.yaml"), help="Pose/depth preflight config."
    ),
    source: str = typer.Option(
        None, help="Depth-enabled D435i URI; defaults to the config source."
    ),
    seconds: float = typer.Option(
        6.0, help="Seconds of one consenting staff member standing still."
    ),
    view: bool = typer.Option(
        True, "--view/--headless", help="Show a live skeleton; no imagery is saved."
    ),
) -> None:
    """Measure the sparse-depth ankle baseline used by the drift fail-safe.

    Run during physical preflight with exactly one consenting staff member
    standing naturally in the approved bed area. The command persists one
    scalar median, never RGB or dense depth, and resets the human verification
    latch to false so the rest of the checklist must still be completed.
    """
    import os
    import re
    import shutil
    import tempfile

    import cv2
    import numpy as np
    import yaml

    from ahfd.capture import open_source
    from ahfd.capture.factory import parse_realsense_uri
    from ahfd.features import FeatureExtractor, attach_depth_heights
    from ahfd.geometry.calibration import load_calibration
    from ahfd.pose import build_estimator
    from ahfd.viz import render_skeleton

    if seconds < 3.0:
        raise typer.BadParameter("--seconds must be at least 3 for a stable baseline")
    if _containing_git_worktree(calibration) is not None:
        raise typer.BadParameter(
            "onsite calibration may reveal ward geometry and must live outside Git"
        )
    cfg = load_config(config)
    uri = source or cfg.source
    options = parse_realsense_uri(uri) if str(uri).startswith("rs://") else {}
    if (
        not options.get("with_depth")
        or options.get("emitter") is not True
        or not options.get("max_laser")
    ):
        raise typer.BadParameter(
            "baseline measurement requires rs:// with depth=1, emitter=1 and max_laser=1"
        )
    device_serial, device_hash = _probe_onsite_d435i()
    calib = load_calibration(calibration)
    if not re.fullmatch(r"cam_[0-9a-f]{8}", calib.camera_id):
        raise typer.BadParameter("camera_id must be cam_ plus exactly 8 random hex characters")
    if calib.device_serial_sha256 != device_hash:
        raise typer.BadParameter("connected D435i does not match this calibration")

    src = open_source(
        uri,
        width=cfg.capture.width,
        height=cfg.capture.height,
        device_serial=device_serial,
        strict_depth_controls=True,
    )
    try:
        src.preflight()
        _check_calibration_resolution(calib, src.meta)
        typer.echo(
            "loading pose model; keep exactly one consenting staff member standing still..."
        )
        estimator = build_estimator(cfg.pose)
        extractor = FeatureExtractor(
            calib.ground,
            zones=calib.zones,
            min_keypoint_score=cfg.pose.min_keypoint_score,
        )
    except Exception:
        # preflight starts the pipeline; setup failures must not leave it owned
        # until interpreter exit.
        src.close()
        raise
    values: list[float] = []
    orientation: list[tuple[float, float]] = []
    started = None
    window = "ahfd ankle baseline -- skeleton only"
    try:
        for frame in src:
            if started is None:
                started = time.monotonic()
            if not _intrinsics_match(calib.ground.intrinsics, frame.intrinsics):
                raise typer.BadParameter("live intrinsics do not match this calibration")
            delta = _imu_orientation_delta(calib, frame.gravity)
            if delta is not None:
                orientation.append(delta)
            pose = estimator.estimate(frame)
            pose = attach_depth_heights(
                pose,
                frame,
                calib.ground,
                min_score=cfg.pose.min_keypoint_score,
            )
            if len(pose.people) > 1:
                raise typer.BadParameter(
                    "ankle preflight requires exactly one consenting staff member in view"
                )
            if len(pose.people) == 1:
                person = pose.people[0].with_track_id(0)
                features = extractor.extract(person, pose.t)
                eligible = _standing_preflight_eligible(features, cfg)
                value = _standing_depth_ankle_height(
                    person,
                    features,
                    "UPRIGHT",
                    min_score=cfg.pose.min_keypoint_score,
                    feature_valid=eligible,
                )
                if value is not None:
                    values.append(value)
            if view:
                canvas = render_skeleton(
                    pose,
                    min_keypoint_score=cfg.pose.min_keypoint_score,
                    fps=None,
                )
                cv2.putText(
                    canvas,
                    "STAND STILL | valid samples " + str(len(values)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if time.monotonic() - started >= seconds:
                break
    finally:
        src.close()
        if view:
            cv2.destroyAllWindows()

    if not orientation:
        raise typer.BadParameter("no D435i IMU orientation was observed")
    pitch_delta = float(np.median([item[0] for item in orientation]))
    roll_delta = float(np.median([item[1] for item in orientation]))
    if abs(pitch_delta) > 2.0 or abs(roll_delta) > 2.0:
        raise typer.BadParameter(
            "mount orientation differs from calibration by more than 2 degrees"
        )
    try:
        baseline = _robust_ankle_baseline(values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    document = yaml.safe_load(calibration.read_text(encoding="utf-8")) or {}
    document["ankle_height_baseline_m"] = round(baseline, 4)
    document["verified_for_onsite"] = False
    rendered = yaml.safe_dump(document, sort_keys=False)
    backup = calibration.with_name(calibration.name + ".pre-baseline.bak")
    if backup.exists():
        raise typer.BadParameter(
            "refusing to overwrite existing calibration backup " + str(backup)
        )
    shutil.copy2(calibration, backup)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=calibration.name + ".", suffix=".tmp", dir=str(calibration.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(calibration, temporary)
        os.replace(temporary, calibration)
    finally:
        if temporary.exists():
            temporary.unlink()
    typer.echo(
        "wrote ankle_height_baseline_m="
        + format(baseline, ".4f")
        + "; verified_for_onsite reset to false"
    )
    typer.echo("complete the remaining physical checklist, then set the latch true")
    typer.echo("previous calibration backup: " + str(backup))


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
    path,
    camera_id,
    intrinsics,
    height_m,
    gravity=None,
    pitch_deg=None,
    roll_deg=0.0,
    device_serial_sha256=None,
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
        "verified_for_onsite": False,
        "ankle_height_baseline_m": None,
        "camera": camera,
        "zones": [],  # add beds/floor next -- see calib/example_ward6.yaml
        "notes": "Auto-generated by `ahfd calibrate`. Add zones before detection.",
    }
    if device_serial_sha256 is not None:
        doc["device_serial_sha256"] = device_serial_sha256
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

    device_identity = _probe_onsite_d435i() if source.startswith("rs://") else None
    device_serial = device_identity[0] if device_identity is not None else None
    device_serial_sha256 = device_identity[1] if device_identity is not None else None
    src = open_source(source, device_serial=device_serial)
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
            out,
            cam_id,
            intrinsics,
            height,
            pitch_deg=pitch,
            roll_deg=roll,
            device_serial_sha256=device_serial_sha256,
        )
        typer.echo("using your --pitch " + format(pitch, ".1f") + " deg (measurement overrides IMU)")
    elif frame.gravity is not None:
        _write_calibration_yaml(
            out,
            cam_id,
            intrinsics,
            height,
            gravity=frame.gravity,
            device_serial_sha256=device_serial_sha256,
        )
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
