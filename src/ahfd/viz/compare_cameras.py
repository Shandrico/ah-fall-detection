"""Side-by-side live depth comparison of two RealSense cameras (D435i vs D435f).

Opens both devices at once (by serial), shows each camera's colourised depth with
a centre-ROI quality gauge -- fill %, mean distance, and noise spread -- so you
can point both at the same target and read off which gives denser, cleaner depth.

Display only; nothing is written to disk.

Caveat: two RealSense projectors emit the same ~850 nm dots, so run together they
INTERFERE -- which is actually representative of a multi-camera ward. Press '1'
or '2' to toggle a camera's projector off and see the other one isolated.
"""

from __future__ import annotations

import numpy as np


def _gauge(depth_m: np.ndarray, roi_frac: float = 0.12):
    """Centre-ROI depth quality: (fill %, mean m, spread cm)."""
    h, w = depth_m.shape[:2]
    ry, rx = int(h * roi_frac), int(w * roi_frac)
    cy, cx = h // 2, w // 2
    roi = depth_m[cy - ry:cy + ry, cx - rx:cx + rx]
    valid = roi[(roi > 0.1) & (roi < 10.0)]
    fill = 100.0 * valid.size / max(roi.size, 1)
    if valid.size < 10:
        return fill, 0.0, 0.0, (cx - rx, cy - ry, cx + rx, cy + ry)
    mean = float(np.median(valid))
    spread = float(np.percentile(valid, 84) - np.percentile(valid, 16)) / 2.0 * 100.0
    return fill, mean, spread, (cx - rx, cy - ry, cx + rx, cy + ry)


def _draw_numbers(cv2, pane, depth_m, cols: int = 8, rows: int = 6):
    """Overlay a grid of median depth values (metres) across the pane."""
    ph, pw = pane.shape[:2]
    dh, dw = depth_m.shape[:2]
    for r in range(rows):
        for c in range(cols):
            cell = depth_m[int(dh * r / rows):int(dh * (r + 1) / rows),
                           int(dw * c / cols):int(dw * (c + 1) / cols)]
            v = cell[(cell > 0.1) & (cell < 10.0)]
            if v.size < 5:
                continue
            txt = "%.2f" % float(np.median(v))
            px = int(pw * (c + 0.5) / cols)
            py = int(ph * (r + 0.5) / rows)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(pane, (px - tw // 2 - 2, py - th - 2), (px + tw // 2 + 2, py + 3), (0, 0, 0), -1)
            cv2.putText(pane, txt, (px - tw // 2, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)


def _colorize(cv2, depth_m, dmin, dmax):
    norm = np.clip((depth_m - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[depth_m <= 0] = 0
    return col


def run_compare_cameras(*, dmin: float = 1.0, dmax: float = 6.0,
                        width: int = 848, height: int = 480, fps: int = 30,
                        pane_w: int = 640) -> None:
    import cv2
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if len(devices) < 2:
        print("need two RealSense cameras plugged in; found %d." % len(devices))
        return

    cams = []  # (pipeline, depth_scale, label, depth_sensor)
    for dev in devices[:2]:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name).replace("Intel RealSense ", "")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        sensor = profile.get_device().first_depth_sensor()
        scale = float(sensor.get_depth_scale())
        cams.append({"pipe": pipe, "scale": scale,
                     "label": "%s  %s" % (name, serial[-4:]), "sensor": sensor,
                     "emitter": True})
        print("opened", name, serial)

    win = "ahfd compare-depth   (1/2 projector, n numbers, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    F = cv2.FONT_HERSHEY_SIMPLEX
    numbers = False
    try:
        while True:
            panes = []
            for cam in cams:
                ok, frames = cam["pipe"].try_wait_for_frames(1000)
                if not ok:
                    continue
                d = frames.get_depth_frame()
                if not d:
                    continue
                depth_m = np.asanyarray(d.get_data()).astype(np.float32) * cam["scale"]
                fill, mean, spread, (x0, y0, x1, y1) = _gauge(depth_m)
                vis = _colorize(cv2, depth_m, dmin, dmax)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2)
                # header + gauge readout
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 78), (0, 0, 0), -1)
                emit = "proj ON" if cam["emitter"] else "proj OFF"
                cv2.putText(vis, cam["label"] + "   " + emit, (10, 26), F, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, "fill %.0f%%   mean %.2f m   noise +-%.1f cm" % (fill, mean, spread),
                            (10, 60), F, 0.7,
                            (60, 220, 60) if (fill > 85 and spread < 10) else (60, 200, 255),
                            2, cv2.LINE_AA)
                pane = cv2.resize(vis, (pane_w, int(vis.shape[0] * pane_w / vis.shape[1])))
                if numbers:
                    _draw_numbers(cv2, pane, depth_m)
                panes.append(pane)
            if panes:
                combo = np.hstack(panes) if len(panes) > 1 else panes[0]
                cv2.imshow(win, combo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("n"), ord("m")):
                numbers = not numbers
            elif key in (ord("1"), ord("2")):
                cam = cams[key - ord("1")]
                if cam["sensor"].supports(rs.option.emitter_enabled):
                    cam["emitter"] = not cam["emitter"]
                    cam["sensor"].set_option(rs.option.emitter_enabled, 1.0 if cam["emitter"] else 0.0)
    finally:
        for cam in cams:
            try:
                cam["pipe"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
