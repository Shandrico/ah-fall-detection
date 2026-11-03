"""Quick re-calibration for a moved camera (demo day), then a live check.

    python scripts/quick_calibrate.py --camera d435f --view along --height 2.60
    python scripts/quick_calibrate.py --camera d435f --check          # live check only

What it does, in order:
 1. Opens the camera (RGB + depth) for about 2 s. Nothing is saved except numbers.
 2. Floor fit from depth -> the depth pose (``depth_ground``); D435i IMU -> colour tilt.
 3. You click the two bed outlines in a window (``ahfd calibrate-zones``, needs the Jetson
    desktop or remote desktop). Name them ``nearest`` and ``middle``.
 4. D435f only (no IMU): the colour tilt is chosen so the outlines you clicked come out at the
    real bed length (``--bed-length``, default 2.10 m: measure it). A measured ``--pitch`` wins.
 5. Writes ``calib/<camera>_<date>.yaml`` and points ``configs/dashboard_jetson.yaml`` at it,
    with the 7-phase model for the view you gave.
 6. ``--check``: 20 s of live pose. Stand beside a bed, then lie on it. It prints what the
    system sees so you can tell if the numbers are sane before the demo.

Measure the lens height with a tape (``--height``); the depth estimate is printed as a check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CONFIG = REPO / "configs/dashboard_jetson.yaml"
# Colour tilt minus depth floor-fit tilt seen on the pilot mount (the depth fit reads low).
DEPTH_PITCH_OFFSET_DEG = 4.0
# Image-tilt thresholds for a camera looking ALONG the beds (29 Oct D435f mount).
ALONG_BED_TILT = {"reclined_enter": 26, "reclined_exit": 22, "upright_enter": 18, "upright_exit": 22}


def uri(cam: str) -> str:
    return f"rs://?depth=1&emitter=1&max_laser=1&max_range=6&model={cam}"


def sample(cam: str, frames: int = 40):
    """Intrinsics, IMU gravity (if any) and floor fits from a short live capture."""
    import numpy as np

    from ahfd.capture import open_source
    from ahfd.geometry.plane_fit import ground_from_floor

    src = open_source(uri(cam))
    intr, grav, fits, valid, ranges = None, [], [], [], []
    try:
        for i, f in enumerate(src):
            if i < 5:
                continue  # let exposure settle
            intr = f.intrinsics or intr
            if f.gravity is not None:
                grav.append(np.asarray(f.gravity, float))
            if f.depth_raw is not None and f.intrinsics is not None:
                d = f.depth_raw
                valid.append(float((d > 0).mean()))
                if (d > 0).any():
                    ranges.append(float(np.median(d[d > 0])) * float(f.depth_scale))
                est = ground_from_floor(f.depth_raw.astype(float) * float(f.depth_scale), f.intrinsics)
                if est is not None:
                    fits.append(est)
            if i >= frames:
                break
    finally:
        src.close()
    if intr is None:
        raise SystemExit("no frames from the " + cam)
    g = np.median(np.array(grav), axis=0) if grav else None
    fit = tuple(float(np.median([x[k] for x in fits])) for k in range(3)) if fits else None
    v = float(np.median(valid)) if valid else 0.0
    r = float(np.median(ranges)) if ranges else 0.0
    print(f"    depth: {v:.0%} of pixels valid, median range {r:.2f} m")
    if v < 0.3 or r < 1.0:
        print("    ! the camera does not seem to see the room (covered, facing a wall, or too close).")
    if fit and not (1.5 <= fit[2] <= 3.5 and abs(fit[1]) <= 15 and 5 <= fit[0] <= 60):
        print(f"    ! floor fit rejected (height {fit[2]:.2f} m, tilt {fit[0]:.1f}, roll {fit[1]:.1f}): "
              "the biggest flat surface is not the floor. Clear the floor in view, or pass --height and --pitch.")
        fit = None
    return intr, g, fit


def project(gp, x: float, y: float, z: float) -> tuple[float, float]:
    """World point (metres, floor frame) to pixel: the inverse of GroundPlane.pixel_to_plane."""
    import numpy as np

    d_cam = gp.rotation.T @ np.array([x, y, z - gp.height_m])
    k = gp.intrinsics
    return (k.fx * d_cam[0] / d_cam[2] + k.cx, k.fy * d_cam[1] / d_cam[2] + k.cy)


def bed_lengths(gp, pixels, top_m):
    """Long side of each clicked outline (metres) under the ground plane ``gp``."""
    out = []
    for poly in pixels:
        pts = [gp.pixel_to_plane(u, v, top_m) for u, v in poly]
        if any(p is None for p in pts):
            return None
        sides = [math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        out.append(max(sides))
    return out


def fit_pitch(intr, height, roll, zones, start_pitch, target_len):
    """Colour tilt that makes the clicked beds ``target_len`` long (grid search, 0.1 deg)."""
    from ahfd.geometry.ground import GroundPlane

    gp0 = GroundPlane(intr, height, start_pitch, roll)
    pixels = [[project(gp0, x, y, z["top_m"]) for x, y in z["polygon"]] for z in zones]
    top = zones[0]["top_m"]
    best = None
    for tenth in range(100, 450):
        pitch = tenth / 10
        lens = bed_lengths(GroundPlane(intr, height, pitch, roll), pixels, top)
        if not lens:
            continue
        err = abs(st.mean(lens) - target_len)
        if best is None or err < best[0]:
            best = (err, pitch, lens)
    if best is None:
        raise SystemExit("could not fit a tilt from the bed outlines")
    gp = GroundPlane(intr, height, best[1], roll)
    new_polys = [[list(map(lambda v: round(v, 3), gp.pixel_to_plane(u, v, z["top_m"]))) for u, v in px]
                 for px, z in zip(pixels, zones)]
    return best[1], best[2], new_polys


def point_dashboard(cam: str, calib: Path, view: str) -> None:
    """Make this camera's picker entry (and the start-up camera) use the new calibration."""
    s = CONFIG.read_text()
    rel = calib.relative_to(REPO).as_posix()
    block = re.compile(r'(      uri: "' + re.escape(uri(cam)) + r'"\n      calibration: )"[^"]*"(\n(?:      #[^\n]*\n)?      phase_model: )"[^"]*"')
    s, n = block.subn(lambda m: m.group(1) + f'"{rel}"' + m.group(2) + f'"models/phase_v3_{view}_logistic.joblib"', s)
    if n != 1:
        raise SystemExit(f"could not find the {cam} entry in {CONFIG}; edit it by hand: calibration {rel}")
    s = re.sub(r'(?m)^source: "[^"]*"', f'source: "{uri(cam)}"', s, count=1)
    s = re.sub(r'(?m)^calibration: "[^"]*"', f'calibration: "{rel}"', s, count=1)
    CONFIG.write_text(s)


def calibrate(a) -> None:
    import yaml

    from ahfd.cli import _write_calibration_yaml
    from ahfd.geometry.ground import GroundPlane

    print(f"1/5 sampling the {a.camera} (about 2 s; keep the floor in view, nobody in front of it)")
    intr, grav, fit = sample(a.camera)
    if fit:
        print(f"    depth floor fit: height {fit[2]:.2f} m, tilt {fit[0]:.1f} deg, roll {fit[1]:.1f} deg")
    else:
        print("    ! no floor found in depth; point the camera so more floor is visible")
    height = a.height or (fit[2] if fit else None)
    if height is None:
        raise SystemExit("no --height and no floor fit: measure the lens height and pass --height")
    if not a.height:
        print(f"    ! using the depth height {height:.2f} m; a tape measurement is better (--height)")
    roll = fit[1] if fit else 0.0
    if grav is not None and a.pitch is None:
        pitch, roll = GroundPlane.from_gravity(intr, height, grav).pitch_deg, GroundPlane.from_gravity(intr, height, grav).roll_deg
        src_pitch = "IMU"
    elif a.pitch is not None:
        pitch, src_pitch = a.pitch, "your --pitch"
    elif fit:
        pitch, src_pitch = fit[0] + DEPTH_PITCH_OFFSET_DEG, "depth fit + 4 deg (refined from the beds below)"
    else:
        raise SystemExit("no tilt: pass --pitch (phone inclinometer on the camera)")
    print(f"2/5 colour tilt {pitch:.1f} deg from {src_pitch}, roll {roll:.1f} deg")

    out = REPO / "calib" / f"{a.camera}_{dt.date.today():%Y%m%d}.yaml"
    _write_calibration_yaml(out, out.stem, intr, height, pitch_deg=pitch, roll_deg=roll)
    doc = yaml.safe_load(out.read_text())
    if fit:
        doc["depth_ground"] = {"height_m": round(fit[2], 3), "pitch_deg": round(fit[0], 2), "roll_deg": round(fit[1], 2)}
    if a.view == "along":
        doc["bed_tilt_deg"] = dict(ALONG_BED_TILT)
    doc["notes"] = (f"scripts/quick_calibrate.py {dt.datetime.now():%Y-%m-%d %H:%M}; view {a.view}; "
                    f"height {'taped' if a.height else 'from depth'}; tilt from {src_pitch}.")
    out.write_text(yaml.safe_dump(doc, sort_keys=False))

    print("3/5 click the beds: a window opens with the live view.")
    print("    SPACE freezes the frame, click the 4 MATTRESS corners, 'f' to finish one bed.")
    print("    Names: 'nearest' then 'middle', kind 'bed', top_m = mattress height (about 0.60).")
    subprocess.run([sys.executable, "-m", "ahfd.cli", "calibrate-zones", str(out), "--source", uri(a.camera)], cwd=REPO, check=False)
    doc = yaml.safe_load(out.read_text())
    beds = [z for z in doc.get("zones") or [] if z.get("kind") == "bed"]
    names = sorted(z["name"] for z in beds)
    if len(beds) != 2:
        print(f"    ! {len(beds)} bed outline(s); bed modes need exactly two. Re-run to redraw.")

    if beds and grav is None and a.pitch is None:
        pitch, lens, polys = fit_pitch(intr, height, roll, beds, pitch, a.bed_length)
        for z, p in zip(beds, polys):
            z["polygon"] = p
        doc["camera"]["pitch_deg"] = round(pitch, 2)
        doc["notes"] += f" Tilt fitted to bed length {a.bed_length} m: {pitch:.1f} deg."
        out.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"4/5 tilt fitted from the bed outlines: {pitch:.1f} deg (beds {', '.join(f'{x:.2f}' for x in lens)} m long)")
    elif beds:
        gp = GroundPlane(intr, height, doc["camera"].get("pitch_deg", pitch), roll)
        print("4/5 tilt kept; bed long sides:", ", ".join(f"{max(math.dist(p[i], p[i - 1]) for i in range(len(p))):.2f} m" for p in (z["polygon"] for z in beds)),
              f"(expect about {a.bed_length} m)")

    point_dashboard(a.camera, out, a.view)
    print(f"5/5 wrote {out.relative_to(REPO)} (beds: {', '.join(names) or 'none'}); the dashboard now starts on it.")
    print("    next: python scripts/quick_calibrate.py --camera", a.camera, "--check")


def check(a) -> None:
    """20 s of live pose through the real feature extractor and rules."""
    import time

    import yaml

    from ahfd.capture import open_source
    from ahfd.cli import _build_detection
    from ahfd.config import load_config
    from ahfd.pose import build_estimator
    from ahfd.track import SimpleTracker

    s = CONFIG.read_text()
    m = re.search(r'uri: "' + re.escape(uri(a.camera)) + r'"\n      calibration: "([^"]+)"', s)
    calib = REPO / m.group(1)
    cfg = load_config(str(CONFIG)); cfg.alert.console = False; cfg.alert.jsonl_path = None
    src = open_source(uri(a.camera))
    ex, eng, _s, cal = _build_detection(cfg, str(calib), src.meta)
    est = build_estimator(cfg.pose); tr = SimpleTracker(min_keypoint_score=cfg.pose.min_keypoint_score, max_miss_s=1.0)
    print(f"checking {calib.relative_to(REPO)} for {a.seconds:.0f} s: stand beside a bed, then lie on it")
    t_end, last = time.monotonic() + a.seconds, 0.0
    try:
        for frame in src:
            pose = tr.update(est.estimate(frame))
            ex.retain_only(tr.live_ids); eng.retain_only(tr.live_ids)
            rows = []
            for p in pose.people:
                if p.track_id is None:
                    continue
                f = ex.extract(p, pose.t)
                if f is None:
                    continue
                eng.update_all(f)
                snap = eng.bed_snapshot_of(p.track_id, pose.t)
                rows.append(f"person {p.track_id}: torso {f.h_torso:.2f} m" if f.h_torso is not None else f"person {p.track_id}: torso ?")
                rows[-1] += f", {eng.state_of(p.track_id)}, bed {f.associated_bed or f.supported_by_bed or '-'}"
                rows[-1] += f", {snap.phase}" if snap else ""
            if time.monotonic() - last >= 1.0:
                print("  " + (" | ".join(rows) or "nobody detected")); last = time.monotonic()
            if time.monotonic() > t_end:
                break
    finally:
        src.close()
    print("Sane: standing torso about 0.9-1.3 m and UPRIGHT; lying on a bed shows that bed and IN_BED / RECLINED.")
    print("If standing reads well above 1.4 m or below 0.7 m, the height or tilt is off: re-run with --height / --pitch.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True, choices=("d435i", "d435f"))
    p.add_argument("--view", choices=("side", "along"), help="camera looks at the beds from the side, or along them")
    p.add_argument("--height", type=float, help="lens height above the floor, taped (m)")
    p.add_argument("--pitch", type=float, help="measured downtilt (deg); overrides the IMU and the bed fit")
    p.add_argument("--bed-length", type=float, default=2.10, help="real mattress length (m), for the D435f tilt fit")
    p.add_argument("--check", action="store_true", help="only run the live check")
    p.add_argument("--seconds", type=float, default=20.0)
    a = p.parse_args()
    if a.check:
        check(a)
    else:
        if not a.view:
            p.error("--view side|along is needed to calibrate")
        calibrate(a)


if __name__ == "__main__":
    main()
