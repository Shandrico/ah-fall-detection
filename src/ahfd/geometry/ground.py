"""Metric geometry from a single RGB camera.

This module is the answer to the project's central measurement problem.

The ward needs one camera to watch three beds spread over 8.4 m. Depth cannot
help at that range -- stereo error grows with the square of distance, so a
D435i that is accurate to ~4 cm at 2 m is off by tens of centimetres at 8 m,
while the fall logic needs about +/-10 cm. Any feature measured in *pixels*
fails for a different reason: a pixel is worth more metres the further away it
is, so a fall at the far bed produces a fraction of the pixel velocity of the
identical fall at the near bed. Thresholds tuned on one bed then miss on
another.

The way out is standard and well documented: if the floor is a plane and you
know where the camera is relative to it, a single ray is enough. The camera
height resolves the scale ambiguity, and heights recovered this way are
accurate more or less regardless of range. Two facts make it cheap here:

* The floor genuinely is a plane, and its position is known -- the camera is
  mounted at a measured height.
* The D435i has an IMU. At rest an accelerometer measures gravity, which is
  the floor normal, so tilt comes for free and continuously. Only the height
  has to be entered by hand. That matters because the camera is on a flexible
  clamp that will sag: tilt drift self-corrects, and only the height is fixed.

Conventions
-----------
Camera frame is OpenCV: x right, y down, z forward.
World frame is floor-referenced: X right, Y forward (both in the floor plane),
Z up, origin directly beneath the camera. So Z is height above the floor in
metres, which is what every fall threshold is written in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ahfd.types import Intrinsics

# Camera-to-world basis for a perfectly level camera: camera z (forward) maps
# to world +Y, camera y (down) maps to world -Z, camera x (right) to world +X.
_LEVEL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _pitch_matrix(pitch_rad: float) -> np.ndarray:
    """Rotation about world X. Positive pitch tips the view downward."""
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )


def _roll_matrix(roll_rad: float) -> np.ndarray:
    """Rotation about the camera's own forward axis."""
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class GroundPlane:
    """Maps image points to floor-referenced metric coordinates.

    `height_m` is the lens height above the floor. `pitch_deg` is positive
    downward -- the ward design has the camera at 2.6 m looking down about
    18-21 degrees. `roll_deg` is rotation about the optical axis, normally
    near zero but not exactly zero on a clamp mount.
    """

    intrinsics: Intrinsics
    height_m: float
    pitch_deg: float
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.height_m <= 0.0:
            raise ValueError(
                "camera height must be positive, got " + repr(self.height_m)
            )
        # A camera pitched up, or exactly level, never intersects the floor
        # ahead of it, so no pixel has a floor position. Catch it here rather
        # than returning None for every pixel later.
        if self.pitch_deg <= 0.0:
            raise ValueError(
                "pitch_deg must be positive (camera looking down); got "
                + repr(self.pitch_deg)
                + ". A level or upward camera sees no floor."
            )

    # ---------------------------------------------------------------- build

    @classmethod
    def from_gravity(
        cls,
        intrinsics: Intrinsics,
        height_m: float,
        gravity_cam: np.ndarray,
    ) -> "GroundPlane":
        """Build from an IMU gravity vector measured in the camera frame.

        A stationary accelerometer reads the reaction to gravity, so the
        measured vector points along world *down* expressed in camera axes.
        For a level camera that is (0, 1, 0); tilting the camera down by theta
        rotates it to (sin(roll)cos(theta), cos(roll)cos(theta), sin(theta)),
        which inverts to give both angles directly.

        This is the reason to prefer the D435i over a plain webcam even when
        depth is unusable at range: orientation stops being a hand-measured
        constant that silently goes stale when the mount sags.
        """
        g = np.asarray(gravity_cam, dtype=float).reshape(3)
        norm = float(np.linalg.norm(g))
        if norm < 1e-6:
            raise ValueError("gravity vector is degenerate (near zero length)")
        g = g / norm

        pitch = math.degrees(math.asin(float(np.clip(g[2], -1.0, 1.0))))
        roll = math.degrees(math.atan2(float(g[0]), float(g[1])))
        return cls(
            intrinsics=intrinsics,
            height_m=height_m,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    # ------------------------------------------------------------- internals

    @property
    def rotation(self) -> np.ndarray:
        """Camera-to-world rotation."""
        return (
            _pitch_matrix(math.radians(self.pitch_deg))
            @ _LEVEL
            @ _roll_matrix(math.radians(self.roll_deg))
        )

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction in world coordinates for the pixel (u, v)."""
        k = self.intrinsics
        d_cam = np.array(
            [(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0],
            dtype=float,
        )
        d_world = self.rotation @ d_cam
        return d_world / float(np.linalg.norm(d_world))

    # -------------------------------------------------------------- mapping

    def pixel_to_plane(
        self, u: float, v: float, plane_z: float = 0.0
    ) -> tuple[float, float] | None:
        """Where the pixel's ray crosses the horizontal plane at height `plane_z`.

        The floor is the common case, but not the only one. A patient lying in
        bed is supported ~0.6 m up, and projecting them onto the *floor* puts
        them roughly 1.5 m beyond the bed at 6 m range -- far enough to fall
        outside the bed polygon entirely, which would defeat the zone that
        exists precisely to recognise them. Testing a bed at its own surface
        height fixes that.

        Returns None for a ray that never descends to the plane.
        """
        d = self.ray(u, v)
        if d[2] >= -1e-9:  # not heading downward
            return None
        t = (plane_z - self.height_m) / d[2]
        if t <= 0.0:  # plane is behind the camera, or above it
            return None
        return (float(t * d[0]), float(t * d[1]))

    def pixel_to_floor(self, u: float, v: float) -> tuple[float, float] | None:
        """Where the pixel's ray meets the floor, in metres.

        Returns None for any pixel at or above the horizon, where the ray
        never descends to the floor. Callers must handle that: it is not an
        error, it is what a ceiling or a far wall looks like.
        """
        return self.pixel_to_plane(u, v, 0.0)

    def joint_height(
        self, u: float, v: float, contact_xy: tuple[float, float]
    ) -> float | None:
        """Height above the floor of a joint, given the body's floor position.

        One ray cannot fix a point in 3D, so this adds the one extra
        constraint a standing person supplies: the body occupies the vertical
        line rising from its floor contact point. Intersecting the joint's ray
        with that line -- in the least-squares sense, since two equations
        constrain one unknown -- gives the height.

        The vertical-line assumption weakens as a person leaves upright, and
        that is acceptable here because of *which way* it fails: as somebody
        goes horizontal their joints move away from the contact line and the
        recovered heights collapse toward zero, which is precisely the signal
        a fall detector wants. It is a poor way to measure a person lying
        down, and a good way to notice that they are.

        Returns None if the geometry is degenerate (a ray pointing almost
        straight down, directly beneath the camera).
        """
        d = self.ray(u, v)
        x0, y0 = contact_xy

        # Least-squares s minimising ||(s*dx - x0, s*dy - y0)||.
        denom = d[0] * d[0] + d[1] * d[1]
        if denom < 1e-12:
            return None
        s = (d[0] * x0 + d[1] * y0) / denom
        if s <= 0.0:  # behind the camera
            return None
        return float(self.height_m + s * d[2])

    def floor_distance(self, contact_xy: tuple[float, float]) -> float:
        """Straight-line distance from the lens to a point on the floor."""
        x, y = contact_xy
        return float(math.sqrt(x * x + y * y + self.height_m * self.height_m))

    def metres_per_pixel(self, contact_xy: tuple[float, float]) -> float:
        """Approximate vertical metres spanned by one pixel at that position.

        Used for the scale-normalised apparent-height cross-check: measured
        pixel extent times this value is a metric extent, independent of range.
        A small-angle approximation about the optical axis, so it drifts at the
        frame edges -- adequate as a sanity check, not as a primary measurement.
        """
        return self.floor_distance(contact_xy) / self.intrinsics.fy

    def horizon_v(self) -> float:
        """Image row of the horizon. Everything above it has no floor point.

        Worth surfacing rather than leaving implicit: if the horizon sits high
        in frame the camera is nearly level and the far floor is compressed
        into very few rows, so distant floor positions become wildly sensitive
        to a one-pixel error in an ankle. That is a camera-placement problem
        made visible in a single number, before anyone mounts anything.

        Derivation: the horizon ray is the one whose world Z component is
        zero. For roll = 0 that reduces to (v - cy) / fy = -tan(pitch).
        """
        k = self.intrinsics
        return float(k.cy - k.fy * math.tan(math.radians(self.pitch_deg)))
