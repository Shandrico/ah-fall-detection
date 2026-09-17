"""Interactive depth viewer for tuning the RealSense depth stream.

A diagnostic window, not part of the detection path. It reuses the capture
layer's existing filter chain (disparity -> spatial -> temporal -> hole-fill,
see ``capture/realsense.py``) so what you see is the *same* denoised depth the
feature pipeline would consume -- and adds the two things a raw depth map lacks
for the eye:

* a **clamped colour ramp** -- spreading the whole colormap across only the
  depth band the scene occupies (e.g. 1.5-3.5 m) instead of 0-6 m, so a head,
  a torso and the floor fall on visibly different colours instead of three
  near-identical shades;
* a **height-above-floor** mode -- recolours every pixel by its metres above
  the floor, derived live from the IMU gravity vector and the mount height.
  This is the view that shows depth solving the "standing directly under the
  camera reads as on-ground" failure: at the nadir, image geometry collapses
  but the head/foot height difference is still plainly there in depth.

Display only -- it never writes a frame to disk (privacy).
"""

from __future__ import annotations

import time

import numpy as np


def _colormaps(cv2):
    return [
        ("turbo", cv2.COLORMAP_TURBO),
        ("jet", cv2.COLORMAP_JET),
        ("viridis", cv2.COLORMAP_VIRIDIS),
        ("inferno", cv2.COLORMAP_INFERNO),
        ("magma", cv2.COLORMAP_MAGMA),
    ]


def _open_source(source: str, max_laser: bool = False):
    from ahfd.capture.realsense import BagSource, RealSenseSource

    if str(source).lower().endswith(".bag"):
        return BagSource(str(source))  # a recording's laser power is fixed
    return RealSenseSource(with_depth=True, max_laser=max_laser)


def run_depth_viewer(
    source: str = "rs://",
    *,
    dmin: float = 1.5,
    dmax: float = 3.5,
    height_m: float = 2.5,
    colormap: str = "turbo",
    hole_filled: bool = True,
    show_color: bool = False,
    long_range: bool = False,
    max_width: int = 1280,
) -> None:
    """Open the live depth viewer. See the module docstring for the keys."""
    import cv2

    cmaps = _colormaps(cv2)
    ci = next((i for i, (n, _) in enumerate(cmaps) if n == colormap), 0)

    # --long-range maxes the projector for denser far depth, and (unless the
    # ramp was set explicitly) points the colour ramp at the 4-6 m band.
    if long_range and (dmin, dmax) == (1.5, 3.5):
        dmin, dmax = 4.0, 6.0

    st = {
        "dmin": float(dmin),
        "dmax": float(dmax),
        "hmin": 0.0,
        "hmax": 2.2,
        "mode": "depth",          # "depth" | "height"
        "hole": bool(hole_filled),
        "auto": False,
        "invert": False,
        "color": bool(show_color),
        "paused": False,
        "mouse": None,            # (x, y) in the displayed image
        "pane_x0": 0,             # x offset of the depth pane in the composite
        "scale": 1.0,             # displayed / full-res ratio
    }

    win = "ahfd depth viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _on_mouse(event, x, y, flags, _param):
        st["mouse"] = (x, y)

    cv2.setMouseCallback(win, _on_mouse)

    src = _open_source(source, max_laser=long_range)
    it = iter(src)

    # Cached per-resolution pixel-direction grids for the height projection.
    grid = {"shape": None, "xg": None, "yg": None}
    ground = {"R": None, "h": float(height_m)}

    last_frame = None
    t_prev = time.monotonic()
    fps = 0.0

    def _ensure_grid(intr, shape):
        if grid["shape"] == shape:
            return
        h, w = shape
        us = (np.arange(w, dtype=np.float32) - intr.cx) / intr.fx
        vs = (np.arange(h, dtype=np.float32) - intr.cy) / intr.fy
        grid["xg"] = np.tile(us[None, :], (h, 1))
        grid["yg"] = np.tile(vs[:, None], (1, w))
        grid["shape"] = shape

    def _height_map(depth_m, frame):
        """Metres above the floor for every pixel, or None if unavailable."""
        from ahfd.geometry.ground import GroundPlane

        if frame.intrinsics is None:
            return None
        if frame.gravity is not None:
            try:
                gp = GroundPlane.from_gravity(frame.intrinsics, ground["h"], np.asarray(frame.gravity))
                ground["R"] = gp.rotation
            except Exception:
                pass
        if ground["R"] is None:
            return None
        _ensure_grid(frame.intrinsics, depth_m.shape)
        R = ground["R"]
        px = depth_m * grid["xg"]
        py = depth_m * grid["yg"]
        pz = depth_m
        world_z = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
        return ground["h"] + world_z

    def _colorize(field, lo, hi, valid):
        rng = max(hi - lo, 1e-6)
        norm = np.clip((field - lo) / rng, 0.0, 1.0)
        if st["invert"]:
            norm = 1.0 - norm
        u8 = (norm * 255).astype(np.uint8)
        img = cv2.applyColorMap(u8, cmaps[ci][1])
        img[~valid] = (0, 0, 0)
        return img

    def _hud(img, frame, valid_depth_m):
        lines = []
        if st["mode"] == "height":
            lines.append("MODE height-above-floor  range %.2f-%.2f m  mount %.2f m" % (st["hmin"], st["hmax"], ground["h"]))
        else:
            lines.append("MODE depth  range %.2f-%.2f m%s" % (st["dmin"], st["dmax"], "  (AUTO)" if st["auto"] else ""))
        lines.append("colormap %s%s  |  %s  |  %.0f fps" % (
            cmaps[ci][0], " (inv)" if st["invert"] else "",
            "hole-filled" if st["hole"] else "raw (holes visible)", fps))
        v = valid_depth_m[valid_depth_m > 0]
        if v.size:
            lines.append("valid depth  min %.2f  median %.2f  max %.2f m" % (v.min(), np.median(v), v.max()))
        # cursor readout
        if st["mouse"] is not None:
            mx, my = st["mouse"]
            ox = int((mx / max(st["scale"], 1e-6)) - st["pane_x0"])
            oy = int(my / max(st["scale"], 1e-6))
            if 0 <= oy < valid_depth_m.shape[0] and 0 <= ox < valid_depth_m.shape[1]:
                d = float(valid_depth_m[oy, ox])
                txt = "cursor  %.2f m" % d if d > 0 else "cursor  (no depth)"
                lines.append(txt)
        y = 26
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30
        help_ln = "f mode  c colormap  i invert  a auto  [ ] max  , . min  h holes  v color  space pause  q quit"
        cv2.putText(img, help_ln, (14, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, help_ln, (14, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    print("depth viewer: press 'q' or ESC in the window to quit.")
    try:
        while True:
            if not st["paused"] or last_frame is None:
                try:
                    frame = next(it)
                except StopIteration:
                    print("source ended.")
                    break
                except Exception as exc:  # noqa: BLE001 - disconnect / USB stall
                    # A mid-stream unplug (or another program grabbing the camera)
                    # raises out of wait_for_frames. Exit cleanly instead of
                    # dumping a traceback and orphaning the window.
                    first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                    print("\ncamera stopped: " + first)
                    print("(device disconnected or held by another program) -- closing viewer.")
                    break
                last_frame = frame
                now = time.monotonic()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            else:
                frame = last_frame

            depth_u16 = frame.depth if st["hole"] else frame.depth_raw
            if depth_u16 is None:
                # IR/colour-only frame slipped through; skip.
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            depth_m = depth_u16.astype(np.float32) * float(frame.depth_scale)
            valid = depth_m > 0.0

            if st["mode"] == "height":
                hmap = _height_map(depth_m, frame)
                if hmap is None:
                    field, lo, hi = depth_m, st["dmin"], st["dmax"]
                    st["mode"] = "depth"  # fall back if no gravity/intrinsics
                else:
                    field, lo, hi = hmap, st["hmin"], st["hmax"]
            if st["mode"] == "depth":
                if st["auto"] and valid.any():
                    lo, hi = np.percentile(depth_m[valid], [2, 98])
                    st["dmin"], st["dmax"] = float(lo), float(hi)
                field, lo, hi = depth_m, st["dmin"], st["dmax"]

            vis = _colorize(field, lo, hi, valid)

            st["pane_x0"] = 0
            if st["color"] and frame.bgr is not None and frame.bgr.shape[:2] == vis.shape[:2]:
                vis = np.hstack([frame.bgr, vis])
                st["pane_x0"] = frame.bgr.shape[1]

            # fit to screen
            if vis.shape[1] > max_width:
                st["scale"] = max_width / vis.shape[1]
                vis = cv2.resize(vis, (max_width, int(vis.shape[0] * st["scale"])))
            else:
                st["scale"] = 1.0

            _hud(vis, frame, depth_m)
            cv2.imshow(win, vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("f"):
                st["mode"] = "height" if st["mode"] == "depth" else "depth"
            elif k == ord("c"):
                ci = (ci + 1) % len(cmaps)
            elif k == ord("i"):
                st["invert"] = not st["invert"]
            elif k == ord("a"):
                st["auto"] = not st["auto"]
            elif k == ord("h"):
                st["hole"] = not st["hole"]
            elif k == ord("v"):
                st["color"] = not st["color"]
            elif k == ord(" "):
                st["paused"] = not st["paused"]
            elif k == ord("r"):
                st["dmin"], st["dmax"], st["hmin"], st["hmax"] = 1.5, 3.5, 0.0, 2.2
            elif k in (ord("]"), ord("=")):
                key = "hmax" if st["mode"] == "height" else "dmax"
                st[key] += 0.1
            elif k in (ord("["), ord("-")):
                key = "hmax" if st["mode"] == "height" else "dmax"
                st[key] = max(st[key] - 0.1, (st["hmin"] if key == "hmax" else st["dmin"]) + 0.1)
            elif k == ord("."):
                key = "hmin" if st["mode"] == "height" else "dmin"
                st[key] = min(st[key] + 0.1, (st["hmax"] if key == "hmin" else st["dmax"]) - 0.1)
            elif k == ord(","):
                key = "hmin" if st["mode"] == "height" else "dmin"
                st[key] = max(st[key] - 0.1, -1.0 if key == "hmin" else 0.1)
    finally:
        try:
            src.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # On Windows a destroyed window only actually closes once the GUI event
        # loop is pumped -- without this the window can linger as "not responding".
        for _ in range(5):
            cv2.waitKey(1)
