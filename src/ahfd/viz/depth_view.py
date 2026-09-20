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


def _open_source(source: str, max_laser: bool = False, max_range_m: float = 6.0, smooth: int = 2):
    from ahfd.capture.realsense import BagSource, RealSenseSource

    if str(source).lower().endswith((".bag", ".db3")):
        # The filter chain re-runs on playback (the .bag holds RAW depth), so
        # --smooth / --max-range tune denoising on a recording too. with_depth
        # so playback carries depth.
        return BagSource(
            str(source), with_depth=True,
            max_range_m=max_range_m, spatial_magnitude=smooth,
        )
    return RealSenseSource(
        with_depth=True, max_laser=max_laser, max_range_m=max_range_m,
        spatial_magnitude=smooth,
    )


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
    max_range_m: float = 6.0,
    smooth: int = 2,
    pose: bool = False,
    backend: str = "rtmo",
    runtime: str = "openvino",
    device: str = "gpu",
    max_width: int = 1280,
) -> None:
    """Open the live depth viewer. See the module docstring for the keys."""
    import cv2

    cmaps = _colormaps(cv2)
    ci = next((i for i, (n, _) in enumerate(cmaps) if n == colormap), 0)

    # --long-range maxes the projector for denser far depth. Unless the ramp
    # was set explicitly, aim it at the far band and, crucially, end it at the
    # requested --max-range -- otherwise everything past 6 m clamps to one
    # colour (turbo's dark red) as the range is opened up.
    if (dmin, dmax) == (1.5, 3.5) and (long_range or max_range_m > 6.0):
        dmax = float(max_range_m)
        dmin = max(1.0, dmax - 4.0)  # a ~4 m window ending at the far cut

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
        "pose_txt": None,         # per-joint depth-height readout, when --pose
        "numbers": False,         # overlay a grid of depth values instead of colour
        "equalize": False,        # histogram-equalise the ramp (more contrast where the pixels are)
        "gauge": False,           # centre-ROI quality readout (fill %, mean, noise)
        "gauge_txt": None,
    }

    win = "ahfd depth viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _on_mouse(event, x, y, flags, _param):
        st["mouse"] = (x, y)

    cv2.setMouseCallback(win, _on_mouse)

    # Optional pose overlay: run the estimator on the colour frame, then read
    # each joint's height from the aligned depth. Loaded here so the plain
    # viewer keeps no pose dependency.
    estimator = None
    ph = None
    if pose:
        from ahfd.config import PoseConfig
        from ahfd.geometry.depth_height import (
            coarse_posture,
            keypoint_heights_from_depth,
            region_height,
        )
        from ahfd.pose import build_estimator
        from ahfd.pose.skeleton import ANKLES, EDGES, HEAD, HIPS, KNEES, SHOULDERS

        print("loading pose model (" + backend + " / " + runtime + ") ...")
        estimator = build_estimator(
            PoseConfig(backend=backend, runtime=runtime, device=device)
        )
        ph = {
            "EDGES": EDGES, "HEAD": HEAD, "SHOULDERS": SHOULDERS, "HIPS": HIPS,
            "KNEES": KNEES, "ANKLES": ANKLES, "kh": keypoint_heights_from_depth,
            "rh": region_height, "cp": coarse_posture,
        }

    src = _open_source(source, max_laser=long_range, max_range_m=max_range_m, smooth=smooth)
    it = iter(src)

    # Cached per-resolution pixel-direction grids for the height projection.
    grid = {"shape": None, "xg": None, "yg": None}
    ground = {"gp": None, "h": float(height_m)}

    def _get_ground(frame):
        """Latest GroundPlane from the IMU gravity + mount height, or None."""
        from ahfd.geometry.ground import GroundPlane

        if frame.gravity is not None and frame.intrinsics is not None:
            try:
                ground["gp"] = GroundPlane.from_gravity(
                    frame.intrinsics, ground["h"], np.asarray(frame.gravity)
                )
            except Exception:
                pass
        return ground["gp"]

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
        gp = _get_ground(frame)
        if gp is None or frame.intrinsics is None:
            return None
        _ensure_grid(frame.intrinsics, depth_m.shape)
        R = gp.rotation
        px = depth_m * grid["xg"]
        py = depth_m * grid["yg"]
        pz = depth_m
        world_z = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
        return ground["h"] + world_z

    def _pose_overlay(vis, depth_m, frame):
        """Draw the skeleton on the depth pane and read each joint's height."""
        st["pose_txt"] = None
        try:
            people = estimator.estimate(frame).people
        except Exception:
            return
        if not people:
            return
        # Largest confident bounding box = the person of interest.
        def _area(p):
            m = p.scores >= 0.3
            if not m.any():
                return 0.0
            pts = p.keypoints[m]
            return float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))

        person = max(people, key=_area)
        kp, sc = person.keypoints, person.scores
        for a, b in ph["EDGES"]:
            if sc[a] >= 0.3 and sc[b] >= 0.3:
                cv2.line(vis, (int(kp[a][0]), int(kp[a][1])),
                         (int(kp[b][0]), int(kp[b][1])), (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(len(kp)):
            if sc[i] >= 0.3:
                p = (int(kp[i][0]), int(kp[i][1]))
                cv2.circle(vis, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(vis, p, 3, (60, 220, 60), -1, cv2.LINE_AA)

        gp = _get_ground(frame)
        if gp is None or frame.intrinsics is None:
            return
        h = ph["kh"](kp, sc, depth_m, frame.intrinsics, gp, min_score=0.4, patch=2)
        rh = ph["rh"]

        def f(x):
            return ("%.2f" % x) if np.isfinite(x) else "--"

        st["pose_txt"] = [
            "POSE height (m):  head %s  shoulder %s  hip %s  knee %s  ankle %s" % (
                f(rh(h, ph["HEAD"])), f(rh(h, ph["SHOULDERS"])), f(rh(h, ph["HIPS"])),
                f(rh(h, ph["KNEES"])), f(rh(h, ph["ANKLES"]))),
            "depth posture guess:  " + ph["cp"](h),
        ]

    def _colorize(field, lo, hi, valid):
        rng = max(hi - lo, 1e-6)
        norm = np.clip((field - lo) / rng, 0.0, 1.0)
        if st["equalize"]:
            # Allocate colours by how many pixels sit at each depth, not by
            # linear distance: the person's body (a dense band) then spans a big
            # slice of the colormap and stands out, even inside a wide 3-6 m
            # window where a linear ramp would render it near-flat.
            vals = field[valid]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.size > 64:
                hist, _ = np.histogram(vals, bins=256, range=(lo, hi))
                cdf = np.cumsum(hist).astype(np.float32)
                if cdf[-1] > 0:
                    cdf /= cdf[-1]
                    norm = cdf[(norm * 255).astype(np.int32)]
        if st["invert"]:
            norm = 1.0 - norm
        u8 = (norm * 255).astype(np.uint8)
        img = cv2.applyColorMap(u8, cmaps[ci][1])
        img[~valid] = (0, 0, 0)
        return img

    def _numbers_overlay(img, depth_m):
        """Print a grid of raw depth values (metres) across the depth pane.

        The number under each grid point is the median depth in a small window
        (zeros ignored). '--' means no depth there (a hole). This is the depth
        matrix made legible -- the actual distances behind the colours.
        """
        h, w = depth_m.shape[:2]
        scale = st["scale"]
        x0 = st["pane_x0"]
        cols, rows = 12, 8

        def _val(oy, ox):
            y0, y1 = max(0, oy - 1), min(h, oy + 2)
            x1, x2 = max(0, ox - 1), min(w, ox + 2)
            patch = depth_m[y0:y1, x1:x2]
            good = patch[patch > 0]
            return float(np.median(good)) if good.size else 0.0

        for j in range(rows):
            oy = int((j + 0.5) / rows * h)
            dy = int(oy * scale)
            for i in range(cols):
                ox = int((i + 0.5) / cols * w)
                dx = int((x0 + ox) * scale)
                d = _val(oy, ox)
                txt = ("%.2f" % d) if d > 0 else "--"
                cv2.putText(img, txt, (dx - 15, dy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (dx - 15, dy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def _gauge(vis, depth_m):
        """Measure depth quality in a centre box: fill %, mean, noise spread.

        To find the max usable range, aim this box at a flat wall (or a standing
        person) and step back: watch fill % fall and the spread grow. The
        distance where fill drops below ~85% or the spread exceeds your
        tolerance (~5-10 cm for posture) is the practical limit at that mount.
        """
        h, w = depth_m.shape[:2]
        rw, rh = int(w * 0.06), int(h * 0.06)
        cx, cy = w // 2, h // 2
        x0, y0, x1, y1 = cx - rw, cy - rh, cx + rw, cy + rh
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2)
        roi = depth_m[y0:y1, x0:x1]
        good = roi[roi > 0]
        fill = 100.0 * good.size / max(roi.size, 1)
        if good.size:
            st["gauge_txt"] = (
                "GAUGE centre box: mean %.2f m   fill %.0f%%   spread +/-%.1f cm"
                "  (aim at a flat wall for true noise)"
                % (float(np.median(good)), fill, float(np.std(good)) * 100.0)
            )
        else:
            st["gauge_txt"] = "GAUGE centre box: no depth here (fill 0%)"

    def _hud(img, frame, valid_depth_m):
        lines = []
        if st["mode"] == "height":
            lines.append("MODE height-above-floor  range %.2f-%.2f m  mount %.2f m" % (st["hmin"], st["hmax"], ground["h"]))
        else:
            lines.append("MODE depth  range %.2f-%.2f m%s" % (st["dmin"], st["dmax"], "  (AUTO)" if st["auto"] else ""))
        lines.append("colormap %s%s%s  |  %s  |  %.0f fps" % (
            cmaps[ci][0], " (inv)" if st["invert"] else "",
            " (EQ)" if st["equalize"] else "",
            "hole-filled" if st["hole"] else "raw (holes visible)", fps))
        v = valid_depth_m[valid_depth_m > 0]
        if v.size:
            lines.append("valid depth  min %.2f  median %.2f  max %.2f m" % (v.min(), np.median(v), v.max()))
        if st.get("pose_txt"):
            lines.extend(st["pose_txt"])
        if st["gauge"] and st.get("gauge_txt"):
            lines.append(st["gauge_txt"])
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
        help_ln = "f mode  n numbers  e equalize  g gauge  c colormap  i invert  a auto  [ ] max  , . min  h holes  v color  space pause  q quit"
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

            if estimator is not None and frame.bgr is not None:
                _pose_overlay(vis, depth_m, frame)

            if st["gauge"]:
                _gauge(vis, depth_m)

            if st["numbers"]:
                # dim the colour so the overlaid numbers stay legible
                vis = (vis.astype(np.float32) * 0.4).astype(np.uint8)

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
            if st["numbers"]:
                _numbers_overlay(vis, depth_m)
            cv2.imshow(win, vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("f"):
                st["mode"] = "height" if st["mode"] == "depth" else "depth"
            elif k == ord("n"):
                st["numbers"] = not st["numbers"]
            elif k == ord("e"):
                st["equalize"] = not st["equalize"]
            elif k == ord("g"):
                st["gauge"] = not st["gauge"]
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
