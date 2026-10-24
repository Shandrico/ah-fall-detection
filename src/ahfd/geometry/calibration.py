"""Per-camera calibration: load a ground plane and its zones from YAML.

Calibration is separate from configuration, and the split is deliberate.

* **Calibration** describes one physical camera in one physical position --
  its height, its tilt, its lens, and the beds it can see. It is measured, it
  is different for every camera, and it is invalidated the moment anybody
  moves the mount.
* **Configuration** holds the thresholds. Because those are expressed in
  metres, they are properties of a *ward*, not of a camera, and one set
  covers every camera in the room.

Keeping them in different files is what stops the thresholds being quietly
re-tuned per camera, which is the failure mode that made the pixel-based
approach unmaintainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import yaml

from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import ZoneMap
from ahfd.types import Intrinsics


@dataclass(frozen=True)
class Calibration:
    """One camera position, fully described."""

    camera_id: str
    ground: GroundPlane
    zones: ZoneMap
    notes: str = ""
    # A human-controlled deployment latch. `ahfd calibrate` always writes
    # false; set true only after the onsite mount/height/zones/depth checks in
    # docs/ONSITE_COLLECTION.md have been completed for this exact position.
    verified_for_onsite: bool = False
    # SHA-256 of the factory serial. The raw serial is never written to a
    # research manifest, but this binds one calibration to one physical unit.
    device_serial_sha256: str | None = None
    # Median two-ankle sparse-depth height measured during the verified staff
    # preflight. Runtime drift is evaluated around this mount-specific value.
    ankle_height_baseline_m: float | None = None
    # D435f has no IMU. A human-selected, permanently visible floor rectangle
    # provides independent live mount verification. Normalised image xyxy.
    floor_roi: tuple[float, float, float, float] | None = None
    # Torso image-tilt thresholds for the bed-activity phases, when this camera's
    # viewpoint differs from the defaults (side-on to the beds). A camera looking
    # along a bed sees a lying torso foreshortened -- ~34 deg instead of ~73 --
    # so RECLINED never triggers at the default 58. Keys: reclined_enter,
    # reclined_exit, upright_enter, upright_exit (degrees). None = defaults.
    bed_tilt_deg: dict[str, float] | None = None

    @property
    def height_m(self) -> float:
        return self.ground.height_m


def _intrinsics_from(entry: dict) -> Intrinsics:
    width = int(entry["width"])
    height = int(entry["height"])
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")

    if "fx" in entry:
        intrinsics = Intrinsics(
            width=width,
            height=height,
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            cx=float(entry.get("cx", width / 2.0)),
            cy=float(entry.get("cy", height / 2.0)),
        )
        values = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if intrinsics.fx <= 0 or intrinsics.fy <= 0:
            raise ValueError("camera fx and fy must be positive")
        return intrinsics

    if "hfov_deg" in entry:
        hfov = float(entry["hfov_deg"])
        vfov = float(entry["vfov_deg"]) if entry.get("vfov_deg") is not None else None
        if not np.isfinite(hfov) or vfov is not None and not np.isfinite(vfov):
            raise ValueError("camera field of view must be finite")
        return Intrinsics.from_hfov(
            width,
            height,
            hfov_deg=hfov,
            vfov_deg=vfov,
        )

    raise ValueError(
        "camera intrinsics need either fx/fy or hfov_deg; got keys "
        + repr(sorted(entry))
    )


def load_calibration(path: str | Path) -> Calibration:
    """Read a calibration YAML."""
    path = Path(path)
    if not path.exists():
        # A raw "[Errno 2] No such file or directory" tells the user nothing.
        # This file is created by `ahfd calibrate`, so say that.
        raise FileNotFoundError(
            "calibration file not found: "
            + str(path)
            + "\nCreate it first with:  ahfd calibrate "
            + str(path)
            + " --source rs:// --height <metres>"
            + "\n(use --pitch <deg> instead of the IMU for a webcam). See docs/USAGE.md."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    camera = data.get("camera")
    if not camera:
        raise ValueError(str(path) + " has no 'camera' section")

    intrinsics = _intrinsics_from(camera)
    height_m = float(camera["height_m"])
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("camera height_m must be finite and positive")

    verified = data.get("verified_for_onsite", False)
    if not isinstance(verified, bool):
        raise ValueError("verified_for_onsite must be a YAML boolean true or false")
    serial_hash = data.get("device_serial_sha256")
    if serial_hash is not None and (
        not isinstance(serial_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", serial_hash)
    ):
        raise ValueError("device_serial_sha256 must be 64 lowercase hexadecimal characters")
    ankle_baseline = data.get("ankle_height_baseline_m")
    if ankle_baseline is not None:
        ankle_baseline = float(ankle_baseline)
        if not np.isfinite(ankle_baseline):
            raise ValueError("ankle_height_baseline_m must be finite")

    if "gravity" in camera:
        gravity = np.asarray(camera["gravity"], dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("camera gravity must contain exactly three finite values")
        if float(np.linalg.norm(gravity)) < 1e-6:
            raise ValueError("camera gravity vector is degenerate")
        # Preferred on a D435i: tilt comes from the IMU, so it cannot go stale
        # when the mount sags.
        ground = GroundPlane.from_gravity(
            intrinsics,
            height_m=height_m,
            gravity_cam=gravity,
        )
    else:
        pitch = float(camera["pitch_deg"])
        roll = float(camera.get("roll_deg", 0.0))
        if not np.isfinite(pitch) or not np.isfinite(roll):
            raise ValueError("camera pitch_deg and roll_deg must be finite")
        ground = GroundPlane(
            intrinsics=intrinsics,
            height_m=height_m,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    bed_tilt = data.get("bed_tilt_deg")
    if bed_tilt is not None:
        bed_tilt = _validate_bed_tilt(bed_tilt)

    floor_roi = data.get("floor_roi")
    if floor_roi is not None:
        from ahfd.geometry.orientation import validate_floor_roi

        floor_roi = validate_floor_roi(floor_roi)

    return Calibration(
        camera_id=str(data.get("camera_id", path.stem)),
        ground=ground,
        zones=ZoneMap.from_config(data.get("zones", [])),
        notes=str(data.get("notes", "")),
        verified_for_onsite=verified,
        device_serial_sha256=serial_hash,
        ankle_height_baseline_m=ankle_baseline,
        floor_roi=floor_roi,
        bed_tilt_deg=bed_tilt,
    )


_BED_TILT_KEYS = ("reclined_enter", "reclined_exit", "upright_enter", "upright_exit")


def _validate_bed_tilt(raw) -> dict[str, float]:
    """All four keys, finite degrees in [0, 90], ordered so the bands cannot cross."""
    if not isinstance(raw, dict) or set(raw) != set(_BED_TILT_KEYS):
        raise ValueError("bed_tilt_deg needs exactly: " + ", ".join(_BED_TILT_KEYS))
    tilt = {k: float(raw[k]) for k in _BED_TILT_KEYS}
    if not all(np.isfinite(v) and 0.0 <= v <= 90.0 for v in tilt.values()):
        raise ValueError("bed_tilt_deg values must be finite degrees in [0, 90]")
    if not (tilt["upright_enter"] < tilt["upright_exit"] <= tilt["reclined_exit"] < tilt["reclined_enter"]):
        raise ValueError(
            "bed_tilt_deg must satisfy upright_enter < upright_exit <= reclined_exit < reclined_enter"
        )
    return tilt


def drift_check(
    ankle_heights: list[float],
    tolerance_m: float = 0.10,
    *,
    baseline_m: float = 0.0,
) -> bool:
    """Is the calibration still trustworthy?

    While somebody walks in view, their ankles should read close to the floor.
    If that number wanders, the camera has moved and every metric height is
    now wrong -- silently, and in a way that still looks plausible on screen.

    This matters because the prototype mount is a flexible clamp, which will
    sag. Returns True while the readings look sane.
    """
    if not ankle_heights:
        return True
    mean = float(np.mean(ankle_heights))
    return abs(mean - float(baseline_m)) <= tolerance_m
