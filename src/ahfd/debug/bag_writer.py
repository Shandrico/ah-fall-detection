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


def _label(cv2, img, text):
    """Caption a preview pane at its bottom-left; returns the same image."""
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def record_bag(
    out_path,
    *,
    seconds: float = 0.0,
    color_size=(1920, 1080),
    depth_size=(848, 480),
    fps: int = 30,
    preview: bool = True,
    emitter_on: bool = True,
) -> int:
    """Record colour + depth + IMU to a .bag. Returns the frameset count.

    Stops on 'q'/ESC in the preview window, or after ``seconds`` (0 = manual).
    The depth projector is forced on (it is what makes depth work); that is the
    opposite of the clean-IR mode, and correct here.
    """
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

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
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
            frames = pipeline.wait_for_frames()
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
