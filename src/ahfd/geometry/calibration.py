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

    @property
    def height_m(self) -> float:
        return self.ground.height_m


def _intrinsics_from(entry: dict) -> Intrinsics:
    width = int(entry["width"])
    height = int(entry["height"])

    if "fx" in entry:
        return Intrinsics(
            width=width,
            height=height,
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            cx=float(entry.get("cx", width / 2.0)),
            cy=float(entry.get("cy", height / 2.0)),
        )

    if "hfov_deg" in entry:
        return Intrinsics.from_hfov(
            width,
            height,
            hfov_deg=float(entry["hfov_deg"]),
            vfov_deg=(
                float(entry["vfov_deg"]) if entry.get("vfov_deg") is not None else None
            ),
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

    if "gravity" in camera:
        # Preferred on a D435i: tilt comes from the IMU, so it cannot go stale
        # when the mount sags.
        ground = GroundPlane.from_gravity(
            intrinsics,
            height_m=float(camera["height_m"]),
            gravity_cam=np.array([float(v) for v in camera["gravity"]]),
        )
    else:
        ground = GroundPlane(
            intrinsics=intrinsics,
            height_m=float(camera["height_m"]),
            pitch_deg=float(camera["pitch_deg"]),
            roll_deg=float(camera.get("roll_deg", 0.0)),
        )

    return Calibration(
        camera_id=str(data.get("camera_id", path.stem)),
        ground=ground,
        zones=ZoneMap.from_config(data.get("zones", [])),
        notes=str(data.get("notes", "")),
    )


def drift_check(ankle_heights: list[float], tolerance_m: float = 0.10) -> bool:
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
    return abs(mean) <= tolerance_m
