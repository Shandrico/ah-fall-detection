"""Ground-plane geometry tests.

These are the tests that matter most in the project, because every fall
threshold is expressed in metres above the floor and this module is what
produces that number. If it is wrong, every threshold is wrong in a way that
looks plausible on screen.

The approach throughout: construct a synthetic camera whose geometry is known
by hand, project a point of known height and distance *forward* into a pixel,
then check the module recovers the original metres. Round-tripping is the only
honest way to test this -- asserting against numbers the same code produced
would prove nothing.

Ward geometry used here is the real one from the Class C floor plan: camera at
2.6 m, pitched down about 20 degrees, beds out to roughly 7.4 m.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ahfd.geometry import GroundPlane
from ahfd.types import Intrinsics

# D435i colour stream at 1080p: ~69 deg horizontal, ~42.5 deg vertical.
WARD_INTRINSICS = Intrinsics.from_hfov(1920, 1080, hfov_deg=69.4, vfov_deg=42.5)
WARD_HEIGHT = 2.6
WARD_PITCH = 20.0


def ward_plane(pitch: float = WARD_PITCH, roll: float = 0.0) -> GroundPlane:
    return GroundPlane(
        intrinsics=WARD_INTRINSICS,
        height_m=WARD_HEIGHT,
        pitch_deg=pitch,
        roll_deg=roll,
    )


def project(plane: GroundPlane, point_world: np.ndarray) -> tuple[float, float]:
    """Project a world point (X, Y, Z) to a pixel. Inverse of `ray`.

    Written independently of the module under test so a shared sign error
    cannot cancel out.
    """
    rel = np.asarray(point_world, dtype=float) - np.array(
        [0.0, 0.0, plane.height_m]
    )
    d_cam = plane.rotation.T @ rel  # world -> camera
    if d_cam[2] <= 0:
        raise ValueError("point is behind the camera")
    k = plane.intrinsics
    return (
        k.cx + k.fx * d_cam[0] / d_cam[2],
        k.cy + k.fy * d_cam[1] / d_cam[2],
    )


class TestConstruction:
    def test_rejects_non_positive_height(self):
        with pytest.raises(ValueError, match="height must be positive"):
            GroundPlane(WARD_INTRINSICS, height_m=0.0, pitch_deg=20.0)

    def test_rejects_level_or_upward_camera(self):
        """A level camera never meets the floor, so every pixel would be None."""
        for pitch in (0.0, -5.0):
            with pytest.raises(ValueError, match="looking down"):
                GroundPlane(WARD_INTRINSICS, height_m=2.6, pitch_deg=pitch)

    def test_rotation_is_orthonormal(self):
        r = ward_plane().rotation
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(r)) == pytest.approx(1.0)


class TestFloorProjection:
    def test_round_trips_a_floor_point(self):
        plane = ward_plane()
        for truth in [(0.0, 3.0), (2.0, 5.0), (-1.5, 7.4), (0.5, 8.4)]:
            u, v = project(plane, np.array([truth[0], truth[1], 0.0]))
            got = plane.pixel_to_floor(u, v)
            assert got is not None
            np.testing.assert_allclose(got, truth, atol=1e-6)

    def test_optical_axis_lands_at_expected_distance(self):
        """Independent check: the centre pixel hits the floor at h/tan(pitch)."""
        plane = ward_plane()
        got = plane.pixel_to_floor(
            plane.intrinsics.cx, plane.intrinsics.cy
        )
        assert got is not None
        expected_y = WARD_HEIGHT / math.tan(math.radians(WARD_PITCH))
        assert got[0] == pytest.approx(0.0, abs=1e-9)
        assert got[1] == pytest.approx(expected_y, rel=1e-9)

    def test_above_horizon_returns_none(self):
        plane = ward_plane()
        horizon = plane.horizon_v()
        assert plane.pixel_to_floor(plane.intrinsics.cx, horizon - 10.0) is None
        assert plane.pixel_to_floor(plane.intrinsics.cx, 0.0) is None

    def test_just_below_horizon_is_very_far_away(self):
        """Sanity on the degenerate direction, and a placement warning."""
        plane = ward_plane()
        got = plane.pixel_to_floor(plane.intrinsics.cx, plane.horizon_v() + 1.0)
        assert got is not None
        assert got[1] > 50.0

    def test_projected_floor_converges_to_the_horizon(self):
        """The horizon is the limit as distance goes to infinity, not a row any
        finite point lands on. Floor points approach it from below, and the
        residual gap shrinks as fy * h / distance -- so assert the convergence
        rather than a tolerance at some arbitrary distance.
        """
        plane = ward_plane()
        horizon = plane.horizon_v()

        distances = (100.0, 1_000.0, 10_000.0)
        gaps = []
        for distance in distances:
            _, v = project(plane, np.array([0.0, distance, 0.0]))
            assert v > horizon, "floor must project below the horizon"
            gaps.append(v - horizon)

        assert gaps[0] > gaps[1] > gaps[2]

        # Exact closed form, from v(Y) = cy + fy * (c*h - s*Y) / (c*Y + s*h):
        #     gap(Y) = fy * h / (c^2 * Y + c * s * h)
        # The cos^2 term is not negligible -- at 20 degrees it is 0.88, so the
        # naive fy*h/Y is 13% wrong, which is exactly the kind of error this
        # test exists to catch.
        c = math.cos(math.radians(plane.pitch_deg))
        s = math.sin(math.radians(plane.pitch_deg))
        for distance, gap in zip(distances, gaps):
            predicted = plane.intrinsics.fy * plane.height_m / (
                c * c * distance + c * s * plane.height_m
            )
            assert gap == pytest.approx(predicted, rel=1e-6)

    def test_lower_in_frame_means_nearer(self):
        plane = ward_plane()
        near = plane.pixel_to_floor(plane.intrinsics.cx, 1000.0)
        far = plane.pixel_to_floor(plane.intrinsics.cx, 600.0)
        assert near is not None and far is not None
        assert near[1] < far[1]


class TestJointHeight:
    @pytest.mark.parametrize("height", [0.0, 0.15, 0.6, 0.95, 1.7])
    @pytest.mark.parametrize("distance", [2.0, 4.0, 7.4])
    def test_recovers_height_at_any_range(self, height, distance):
        """The property the whole design rests on: range independence.

        The same physical height must read the same in metres whether the
        person is at the near bed or the far one. This is what a pixel-based
        feature cannot do.
        """
        plane = ward_plane()
        contact = (0.0, distance)
        u, v = project(plane, np.array([contact[0], contact[1], height]))
        got = plane.joint_height(u, v, contact)
        assert got is not None
        assert got == pytest.approx(height, abs=1e-6)

    def test_recovers_height_off_axis(self):
        plane = ward_plane()
        contact = (2.5, 6.0)
        u, v = project(plane, np.array([contact[0], contact[1], 1.6]))
        got = plane.joint_height(u, v, contact)
        assert got is not None
        assert got == pytest.approx(1.6, abs=1e-6)

    def test_standing_person_reads_taller_than_fallen_one(self):
        plane = ward_plane()
        contact = (0.0, 6.0)

        head_standing = project(plane, np.array([0.0, 6.0, 1.65]))
        head_fallen = project(plane, np.array([0.0, 6.0, 0.18]))

        h_stand = plane.joint_height(*head_standing, contact)
        h_fall = plane.joint_height(*head_fallen, contact)
        assert h_stand is not None and h_fall is not None
        assert h_stand > 1.5
        assert h_fall < 0.3

    def test_the_same_pixel_means_different_heights_at_different_ranges(self):
        """Why the contact point is required rather than optional.

        One pixel is not enough information. Feeding the same image point two
        different floor positions must give two different heights -- if it did
        not, the contact point would be doing nothing.
        """
        plane = ward_plane()
        u, v = project(plane, np.array([0.0, 5.0, 1.0]))
        near = plane.joint_height(u, v, (0.0, 3.0))
        far = plane.joint_height(u, v, (0.0, 8.0))
        assert near is not None and far is not None
        assert abs(near - far) > 0.3

    def test_returns_none_behind_camera(self):
        plane = ward_plane()
        assert plane.joint_height(960.0, 1070.0, (0.0, -5.0)) is None


class TestGravity:
    @pytest.mark.parametrize("pitch", [5.0, 20.0, 35.0, 60.0])
    def test_recovers_pitch_from_gravity(self, pitch):
        """Round-trip through the IMU path with roll = 0."""
        g = np.array(
            [0.0, math.cos(math.radians(pitch)), math.sin(math.radians(pitch))]
        )
        plane = GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, g)
        assert plane.pitch_deg == pytest.approx(pitch, abs=1e-9)
        assert plane.roll_deg == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("tilt", [10.0, 24.75, 40.0])
    def test_downward_d435i_vector_gives_positive_downtilt(self, tilt):
        """Regression: a real D435i pointed down reads a NEGATIVE z component
        (opposite the level-camera model), which used to raise 'pitch must be
        positive' and crash `ahfd calibrate`. A downward mount must resolve to
        that downtilt, not an error."""
        g = np.array(
            [0.0, math.cos(math.radians(tilt)), -math.sin(math.radians(tilt))]
        )
        plane = GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, g)
        assert plane.pitch_deg == pytest.approx(tilt, abs=1e-6)
        assert plane.roll_deg == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("pitch,roll", [(20.0, 5.0), (30.0, -8.0)])
    def test_recovers_pitch_and_roll(self, pitch, roll):
        p, r = math.radians(pitch), math.radians(roll)
        g = np.array(
            [
                math.sin(r) * math.cos(p),
                math.cos(r) * math.cos(p),
                math.sin(p),
            ]
        )
        plane = GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, g)
        assert plane.pitch_deg == pytest.approx(pitch, abs=1e-9)
        assert plane.roll_deg == pytest.approx(roll, abs=1e-9)

    def test_magnitude_is_irrelevant(self):
        """Accelerometers report in m/s^2 or g; only the direction is used."""
        base = np.array([0.0, math.cos(math.radians(20.0)), math.sin(math.radians(20.0))])
        a = GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, base)
        b = GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, base * 9.81)
        assert a.pitch_deg == pytest.approx(b.pitch_deg)

    def test_rejects_degenerate_vector(self):
        with pytest.raises(ValueError, match="degenerate"):
            GroundPlane.from_gravity(WARD_INTRINSICS, 2.6, np.zeros(3))

    def test_gravity_derived_plane_measures_correctly(self):
        """End to end: IMU-derived orientation must give the same metres."""
        pitch = 20.0
        g = np.array(
            [0.0, math.cos(math.radians(pitch)), math.sin(math.radians(pitch))]
        )
        plane = GroundPlane.from_gravity(WARD_INTRINSICS, WARD_HEIGHT, g)
        contact = (0.0, 6.5)
        u, v = project(plane, np.array([contact[0], contact[1], 1.55]))
        assert plane.joint_height(u, v, contact) == pytest.approx(1.55, abs=1e-6)


class TestScaleAndSensitivity:
    def test_metres_per_pixel_grows_with_distance(self):
        plane = ward_plane()
        assert plane.metres_per_pixel((0.0, 8.0)) > plane.metres_per_pixel((0.0, 2.0))

    def test_person_pixel_height_matches_the_placement_analysis(self):
        """Cross-check against the number used to size the camera.

        A 1.7 m person at 8.4 m should occupy roughly 280 px at 1080p. This is
        the figure the 'must run 1080p, not 480p' conclusion rests on, so it is
        worth pinning in a test rather than trusting arithmetic in a document.
        """
        plane = ward_plane()
        feet = project(plane, np.array([0.0, 8.4, 0.0]))
        head = project(plane, np.array([0.0, 8.4, 1.7]))
        pixel_height = abs(feet[1] - head[1])
        assert 240.0 < pixel_height < 320.0

    def test_one_pixel_ankle_error_costs_less_near_than_far(self):
        """Quantifies the real cost of range, which FOV maths hides.

        Geometry stays exact at distance, but the *precision* does not: the
        same one-pixel keypoint error maps to a much larger floor error far
        away. This is the honest limit on the far bed, and it argues for
        1080p and for a camera that is not nearly level.
        """
        plane = ward_plane()

        def floor_error_at(distance: float) -> float:
            u, v = project(plane, np.array([0.0, distance, 0.0]))
            a = plane.pixel_to_floor(u, v)
            b = plane.pixel_to_floor(u, v - 1.0)
            assert a is not None and b is not None
            return abs(b[1] - a[1])

        near, far = floor_error_at(2.0), floor_error_at(8.4)
        assert far > near * 5.0
