"""Record a RealSense .bag (colour + depth + IMU) for a consented session.

Lives in ``ahfd.debug`` for the same reason as the colour-only ``RawRecorder``
beside it: a .bag persists raw imagery, and ``ahfd.debug`` is the one package
the privacy guard permits to do that, gated by consent at the call site. The
colour-only recorder cannot capture depth (or the IMU gravity the height maths
needs), so this is the tool for collecting depth training data.

The flow mirrors ``record``: capture a staged, consented session; extract
features from the .bag; then delete the .bag and keep only the derived data.
"""

from __future__ import annotations

import time
from pathlib import Path

from ahfd.privacy import require_raw_capture


def _label(cv2, img, text):
    """Caption a preview pane at its bottom-left; returns the same image."""
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def record_bag(
    out_path,
    *,
    config_flag: bool = False,
    cli_flag: bool = False,
    seconds: float = 0.0,
    color_size=(1920, 1080),
    depth_size=(848, 480),
    fps: int = 30,
    preview: bool = True,
    emitter_on: bool = True,
) -> int:
    """Record colour + depth + IMU to a .bag. Returns the frameset count.

    ``config_flag`` and ``cli_flag`` are intentionally fail-closed defaults.
    The third switch is ``AHFD_ALLOW_RAW=1``; all three are checked here, at
    the writer boundary, even when a caller has already checked them.

    Stops on 'q'/ESC in the preview window, or after ``seconds`` (0 = manual).
    The depth projector is forced on (it is what makes depth work); that is the
    opposite of the clean-IR mode, and correct here.
    """
    # This must be the first executable operation.  In particular, refuse
    # before importing the optional camera dependency, creating a directory,
    # opening a pipeline or asking librealsense to create the output file.
    require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)

    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "pyrealsense2 is not installed. Install the realsense extra "
            "(uv pip install -e '.[realsense]') to record a .bag."
        ) from exc

    import cv2
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ahfd.capture.realsense import _device_has_imu

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
    # IMU only when present: the D435i has accel+gyro, the D435f does NOT, and
    # requesting absent streams makes pipeline.start fail. A D435f .bag then has
    # no gravity -- extraction uses --pitch for the tilt instead.
    if _device_has_imu(rs):
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
    config.enable_record_to_file(str(out_path))  # the whole point of this module

    profile = pipeline.start(config)
    try:
        sensor = profile.get_device().first_depth_sensor()
        if emitter_on and sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0)
        # Max the projector: denser depth / fewer holes at 4-6 m. This is baked
        # into the recording (a capture-time hardware setting), so it cannot be
        # recovered later -- unlike the colour ramp or the software filters.
        if sensor.supports(rs.option.laser_power):
            rng = sensor.get_option_range(rs.option.laser_power)
            sensor.set_option(rs.option.laser_power, rng.max)
    except Exception:  # pragma: no cover - best-effort hardware option
        pass

    win = "ahfd RECORDING -- depth .bag"
    start = time.monotonic()
    n_framesets = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as exc:  # USB drop mid-recording -- keep what we got
                print("\ncamera disconnected after %d framesets: %s" % (n_framesets, exc))
                print("the clip up to here is saved. This is a USB/bandwidth issue, not config:")
                print("use the shipped cable in a USB-3 port with no hub; 1080p is heavy -- try --rgb 720.")
                break
            n_framesets += 1
            elapsed = time.monotonic() - start
            if preview:
                # Show colour AND colourised depth side by side, so you can see
                # live that depth is actually landing (coverage, holes) while the
                # .bag records -- not just trust it after the fact.
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                panes = []
                if color:
                    cimg = np.asanyarray(color.get_data())
                    panes.append(_label(cv2, cv2.resize(cimg, (640, 360)), "RGB"))
                if depth:
                    dimg = np.asanyarray(depth.get_data())  # uint16, mm
                    norm = np.clip(dimg.astype(np.float32) / 6000.0, 0.0, 1.0)  # 0..6 m
                    dcol = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    dcol[dimg == 0] = 0  # holes -> black, so coverage is visible
                    panes.append(_label(cv2, cv2.resize(dcol, (640, 360)), "DEPTH (near=blue far=red)"))
                if panes:
                    disp = np.hstack(panes) if len(panes) > 1 else panes[0]
                    cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 160), -1)
                    cv2.putText(
                        disp, "REC  %.0fs   colour+depth+IMU   (q to stop)" % elapsed,
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.imshow(win, disp)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
            if seconds and elapsed >= seconds:
                break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
    return n_framesets


def _cam_panes(cv2, np, frames, label):
    """RGB + colourised-depth panes for one camera, captioned with its label."""
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    out = []
    if color:
        cimg = np.asanyarray(color.get_data())
        out.append(_label(cv2, cv2.resize(cimg, (560, 315)), label + "  RGB"))
    if depth:
        dimg = np.asanyarray(depth.get_data())
        norm = np.clip(dimg.astype(np.float32) / 6000.0, 0.0, 1.0)
        dcol = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dcol[dimg == 0] = 0
        out.append(_label(cv2, cv2.resize(dcol, (560, 315)), label + "  DEPTH"))
    return np.hstack(out) if out else None


def record_bag_dual(
    cams,
    *,
    config_flag: bool = False,
    cli_flag: bool = False,
    seconds: float = 0.0,
    color_size=(1920, 1080),
    depth_size=(848, 480),
    fps: int = 30,
    preview: bool = True,
    emitter_on: bool = True,
) -> dict:
    """Record TWO (or more) RealSense cameras at once, each to its own .db3.

    ``cams`` is a list of ``(out_path, serial, label)``. Each camera runs on its
    own pipeline, so the streams are *software*-synced only: both start as close
    together as the opens allow and every frame carries its own timestamp, which
    is enough to align the two recordings in post. It is NOT hardware frame-sync
    (that needs the inter-camera sync cable).

    Resilience is the point of recording them together here: each camera is
    polled independently, so one dropping on a USB/bandwidth stall does not stop
    the other -- its clip up to the drop is still saved. Returns {label: frames}.
    """
    require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)

    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "pyrealsense2 is not installed. Install the realsense extra "
            "(uv pip install -e '.[realsense]') to record."
        ) from exc

    import cv2
    import numpy as np

    from ahfd.capture.realsense import _device_has_imu

    pipes = []
    for out_path, serial, label in cams:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
        if _device_has_imu(rs, serial):
            cfg.enable_stream(rs.stream.accel)
            cfg.enable_stream(rs.stream.gyro)
        cfg.enable_record_to_file(str(out_path))
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        try:
            sensor = profile.get_device().first_depth_sensor()
            if emitter_on and sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1.0)
            if sensor.supports(rs.option.laser_power):
                rng = sensor.get_option_range(rs.option.laser_power)
                sensor.set_option(rs.option.laser_power, rng.max)
        except Exception:  # pragma: no cover - best-effort hardware option
            pass
        pipes.append({"path": out_path, "serial": serial, "label": label,
                      "pipe": pipe, "alive": True, "n": 0})
        print("  recording %s (%s) -> %s" % (label, serial[-4:], out_path))

    win = "ahfd RECORDING -- dual .db3  (q to stop)"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    start = time.monotonic()
    try:
        while any(p["alive"] for p in pipes):
            elapsed = time.monotonic() - start
            rows = []
            for p in pipes:
                if not p["alive"]:
                    continue
                try:
                    ok, frames = p["pipe"].try_wait_for_frames(200)
                except RuntimeError as exc:  # USB drop -- keep the OTHER camera going
                    print("\n%s (%s) disconnected after %d framesets: %s"
                          % (p["label"], p["serial"][-4:], p["n"], exc))
                    print("  that clip is saved; the other camera keeps recording.")
                    try:
                        p["pipe"].stop()
                    except Exception:
                        pass
                    p["alive"] = False
                    continue
                if not ok:
                    continue
                p["n"] += 1
                if preview:
                    pane = _cam_panes(cv2, np, frames, p["label"])
                    if pane is not None:
                        rows.append(pane)
            if preview:
                if rows:
                    w = max(r.shape[1] for r in rows)
                    rows = [r if r.shape[1] == w else cv2.copyMakeBorder(
                        r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
                    disp = np.vstack(rows)
                    cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 160), -1)
                    cv2.putText(disp, "REC  %.0fs   %d cameras   (q to stop)"
                                % (elapsed, sum(p["alive"] for p in pipes)),
                                (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win, disp)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if seconds and elapsed >= seconds:
                break
    finally:
        for p in pipes:
            try:
                if p["alive"]:
                    p["pipe"].stop()
            except Exception:
                pass
        if preview:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
    return {p["label"]: p["n"] for p in pipes}
