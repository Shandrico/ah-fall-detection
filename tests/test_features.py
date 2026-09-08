"""Metric feature tests, driven by projected 3D bodies.

Unlike the state-machine tests -- which take features as given -- these build
an anatomically plausible body at a known place in the ward, project all 17
joints through the real camera model, and then check the extractor recovers
metric truth from the pixels.

This is where the central claim of `features/extractor.py` gets verified:
that **floor_spread separates upright from fallen**, and does so the opposite
way round to image-space intuition. A standing person's head ray, continued to
the floor, lands many metres past their feet. A fallen person's joints really
are on the floor, so they project to a patch about a body long.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ahfd.features import FeatureExtractor
from ahfd.geometry import GroundPlane
from ahfd.geometry.zones import Zone, ZoneMap, rectangle
from ahfd.types import NUM_KEYPOINTS, Intrinsics, PersonPose

WARD_INTRINSICS = Intrinsics.from_hfov(1920, 1080, hfov_deg=69.4, vfov_deg=42.5)
WARD = GroundPlane(WARD_INTRINSICS, height_m=2.6, pitch_deg=20.0)

# COCO-17 as (height above floor, lateral offset) for a 1.7 m adult standing.
BODY: tuple[tuple[float, float], ...] = (
    (1.62, 0.00),  # nose
    (1.64, 0.03),  # left eye
    (1.64, -0.03),  # right eye
    (1.62, 0.07),  # left ear
    (1.62, -0.07),  # right ear
    (1.40, 0.18),  # left shoulder
    (1.40, -0.18),  # right shoulder
    (1.10, 0.20),  # left elbow
    (1.10, -0.20),  # right elbow
    (0.85, 0.20),  # left wrist
    (0.85, -0.20),  # right wrist
    (0.95, 0.12),  # left hip
    (0.95, -0.12),  # right hip
    (0.50, 0.12),  # left knee
    (0.50, -0.12),  # right knee
    (0.08, 0.10),  # left ankle
    (0.08, -0.10),  # right ankle
)

# Torso centroid = mean of shoulders and hips = mean(1.40, 1.40, 0.95, 0.95).
TRUE_TORSO_H = 1.175


def project(plane: GroundPlane, point: np.ndarray) -> tuple[float, float]:
    rel = np.asarray(point, float) - np.array([0.0, 0.0, plane.height_m])
    d = plane.rotation.T @ rel
    k = plane.intrinsics
    return (k.cx + k.fx * d[0] / d[2], k.cy + k.fy * d[1] / d[2])


def standing(x: float, y: float, scale: float = 1.0) -> PersonPose:
    """An upright body at floor position (x, y). `scale` shrinks it (crouch)."""
    pts = []
    for height, lateral in BODY:
        pts.append(project(WARD, np.array([x + lateral, y, height * scale])))
    return PersonPose(
        keypoints=np.array(pts, dtype=np.float32),
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
        track_id=1,
    )


def lying(x: float, y: float, thickness: float = 0.12) -> PersonPose:
    """A body flat on the floor, feet at (x, y), head further from the camera.

    The body's long axis now runs along +Y, so what was height becomes
    distance. Every joint sits a body-thickness above the floor.
    """
    pts = []
    for height, lateral in BODY:
        along = height - 0.08  # ankles become the origin
        pts.append(project(WARD, np.array([x + lateral, y + along, thickness])))
    return PersonPose(
        keypoints=np.array(pts, dtype=np.float32),
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
        track_id=1,
    )


def extractor(zones: ZoneMap | None = None) -> FeatureExtractor:
    return FeatureExtractor(WARD, zones=zones)


class TestUprightMeasurement:
    @pytest.mark.parametrize("distance", [3.0, 5.0, 7.4])
    def test_torso_height_is_recovered_at_any_range(self, distance):
        """The property the design rests on, measured end to end from pixels."""
        f = extractor().extract(standing(0.0, distance), t=0.0)
        assert f is not None
        assert f.h_torso == pytest.approx(TRUE_TORSO_H, abs=0.05)

    @pytest.mark.parametrize("distance", [3.0, 6.0, 7.4])
    def test_contact_point_lands_at_the_feet(self, distance):
        """Checks the ankle-height correction.

        Without it the contact point sits systematically beyond the real feet
        -- +0.19 m at 6 m, worsening with range -- because an ankle keypoint
        is ~8 cm above the floor and its ray keeps descending past the true
        contact. Large enough to push a body across a bed-zone boundary, so
        the tolerance here is deliberately tight enough to catch a regression.
        """
        f = extractor().extract(standing(0.0, distance), t=0.0)
        assert f is not None
        assert f.contact_xy is not None
        assert f.contact_xy[1] == pytest.approx(distance, abs=0.05)

    def test_head_is_higher_than_torso(self):
        f = extractor().extract(standing(0.0, 5.0), t=0.0)
        assert f is not None
        assert f.h_head is not None and f.h_torso is not None
        assert f.h_head > f.h_torso

    def test_range_is_reported(self):
        f = extractor().extract(standing(0.0, 6.0), t=0.0)
        assert f is not None
        assert f.range_m == pytest.approx(math.hypot(6.0, 2.6), abs=0.2)


class TestFloorSpreadDiscriminates:
    """The core claim, and the reason heights alone are not enough."""

    @pytest.mark.parametrize("distance", [3.0, 5.0, 7.4])
    def test_upright_body_has_a_large_floor_spread(self, distance):
        f = extractor().extract(standing(0.0, distance), t=0.0)
        assert f is not None
        assert f.floor_spread > 4.0, (
            "an upright body's head ray must land far past its feet, got "
            + format(f.floor_spread, ".2f")
        )

    @pytest.mark.parametrize("distance", [3.0, 5.0, 7.0])
    def test_fallen_body_spread_is_about_a_body_length(self, distance):
        f = extractor().extract(lying(0.0, distance), t=0.0)
        assert f is not None
        assert 1.0 < f.floor_spread < 3.0, (
            "a body on the floor should span roughly its own length, got "
            + format(f.floor_spread, ".2f")
        )

    def test_the_two_cases_are_cleanly_separated(self):
        """No overlap across the working range -- which is what makes a fixed
        threshold band viable rather than a per-camera fudge."""
        up = [extractor().extract(standing(0.0, d), 0.0) for d in (3.0, 5.0, 7.4)]
        down = [extractor().extract(lying(0.0, d), 0.0) for d in (3.0, 5.0, 7.0)]

        worst_upright = min(f.floor_spread for f in up if f)
        worst_fallen = max(f.floor_spread for f in down if f)
        assert worst_upright > worst_fallen * 1.5

    def test_fallen_body_reads_lower_than_standing(self):
        """The height estimate is biased for a horizontal body, but it still
        moves decisively in the right direction -- which is what the velocity
        trigger needs."""
        up = extractor().extract(standing(0.0, 6.0), 0.0)
        down = extractor().extract(lying(0.0, 6.0), 0.0)
        assert up is not None and down is not None
        assert up.h_torso is not None and down.h_torso is not None
        assert down.h_torso < up.h_torso - 0.35


class TestKinematics:
    def test_falling_produces_negative_vertical_velocity(self):
        ex = extractor()
        v_last = 0.0
        # Collapse from full height to a crouch over 0.4 s at 15 fps.
        for i in range(6):
            t = i / 15.0
            scale = 1.0 - 0.75 * (i / 5.0)
            f = ex.extract(standing(0.0, 6.0, scale=scale), t=t)
            assert f is not None
            v_last = f.v_z
        assert v_last < -1.0, "expected a clear downward velocity, got " + str(v_last)

    def test_standing_still_has_near_zero_velocity(self):
        ex = extractor()
        f = None
        for i in range(10):
            f = ex.extract(standing(0.0, 6.0), t=i / 15.0)
        assert f is not None
        assert abs(f.v_z) < 0.05

    def test_motion_is_zero_when_stationary(self):
        ex = extractor()
        f = None
        for i in range(20):
            f = ex.extract(standing(0.0, 6.0), t=i / 15.0)
        assert f is not None
        assert f.motion < 0.05

    def test_walking_registers_motion(self):
        ex = extractor()
        f = None
        for i in range(20):
            f = ex.extract(standing(0.0, 6.0 - i * 0.06), t=i / 15.0)
        assert f is not None
        assert f.motion > 0.4


class TestZoneIntegration:
    def test_standing_in_a_bed_bay_is_not_supported_by_the_bed(self):
        """Standing beside a bed reports the zone but not support.

        The distinction matters: `zones` says where somebody is, while
        `supported_by_bed` says whether the bed is holding them up. Only the
        latter suppresses a fall, so a patient standing next to their bed is
        still protected.
        """
        bed = Zone(
            name="bed_2",
            kind="bed",
            polygon=rectangle((0.0, 6.0), length=2.0, width=1.0, angle_deg=45.0),
            top_m=0.6,
        )
        f = extractor(ZoneMap([bed])).extract(standing(0.0, 6.0), 0.0)
        assert f is not None
        assert "bed_2" in f.zones
        assert f.supported_by_bed is None
        assert f.bed_top_m is None

    def test_lying_in_a_bed_is_recognised_as_supported(self):
        """The dominant false positive, at the feature level.

        A body in bed sits ~0.6 m up, so its *floor* projection lands well
        beyond the bed. Support is therefore tested at the bed's own surface
        height -- without that, no patient in bed is ever inside their bed
        zone.
        """
        bed = Zone(
            name="bed_2",
            kind="bed",
            polygon=rectangle((0.0, 6.0), length=2.4, width=1.4, angle_deg=0.0),
            top_m=0.6,
        )
        # Lying on the bed surface rather than the floor.
        pts = []
        for height, lateral in BODY:
            pts.append(
                project(WARD, np.array([lateral, 6.0 + (height - 0.08), 0.60]))
            )
        in_bed = PersonPose(
            keypoints=np.array(pts, dtype=np.float32),
            scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
            score=1.0,
            track_id=1,
        )

        f = extractor(ZoneMap([bed])).extract(in_bed, 0.0)
        assert f is not None
        assert f.supported_by_bed == "bed_2"
        assert f.bed_top_m == 0.6

    def test_outside_all_zones_is_empty(self):
        bed = Zone(
            name="bed_2",
            kind="bed",
            polygon=rectangle((0.0, 2.0), length=2.0, width=1.0, angle_deg=0.0),
            top_m=0.6,
        )
        f = extractor(ZoneMap([bed])).extract(standing(0.0, 7.0), 0.0)
        assert f is not None
        assert f.zones == ()
        assert f.bed_top_m is None

    def test_excluded_zone_is_flagged(self):
        doorway = Zone(
            name="doorway",
            kind="exclude",
            polygon=rectangle((0.0, 6.0), length=3.0, width=3.0, angle_deg=0.0),
        )
        f = extractor(ZoneMap([doorway])).extract(standing(0.0, 6.0), 0.0)
        assert f is not None
        assert f.in_excluded_zone is True


class TestDegradedInput:
    def test_untracked_person_yields_nothing(self):
        person = standing(0.0, 5.0)
        untracked = PersonPose(
            keypoints=person.keypoints, scores=person.scores, score=1.0
        )
        assert extractor().extract(untracked, 0.0) is None

    def test_no_confident_keypoints_reports_missing_geometry(self):
        person = PersonPose(
            keypoints=np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
            scores=np.zeros(NUM_KEYPOINTS, dtype=np.float32),
            score=0.0,
            track_id=1,
        )
        f = extractor().extract(person, 0.0)
        assert f is not None
        assert f.has_geometry() is False
        assert f.n_valid_kp == 0

    def test_missing_ankles_falls_back_to_the_lowest_joint(self):
        """Ankles are often hidden behind a bed frame, so this path is common."""
        person = standing(0.0, 6.0)
        scores = person.scores.copy()
        scores[15] = scores[16] = 0.0  # both ankles gone
        degraded = PersonPose(
            keypoints=person.keypoints, scores=scores, score=1.0, track_id=1
        )
        f = extractor().extract(degraded, 0.0)
        assert f is not None
        assert f.has_geometry() is True
        # Knees become the contact, so the person reads as shorter -- a known
        # and acceptable bias, worth pinning so it cannot silently worsen.
        assert f.h_torso is not None
        assert 0.5 < f.h_torso < TRUE_TORSO_H

    def test_history_is_dropped_for_dead_tracks(self):
        ex = extractor()
        ex.extract(standing(0.0, 5.0), 0.0)
        ex.retain_only(set())
        f = ex.extract(standing(0.0, 5.0), 1.0)
        assert f is not None
        assert f.v_z == 0.0  # fresh history, no spurious velocity
