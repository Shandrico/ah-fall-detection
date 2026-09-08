"""Floor zone tests."""

from __future__ import annotations

import pytest

from ahfd.geometry.zones import Zone, ZoneMap, point_in_polygon, rectangle

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


class TestPointInPolygon:
    def test_inside(self):
        assert point_in_polygon((1.0, 1.0), SQUARE) is True

    @pytest.mark.parametrize(
        "point", [(-1.0, 1.0), (3.0, 1.0), (1.0, -1.0), (1.0, 3.0)]
    )
    def test_outside(self, point):
        assert point_in_polygon(point, SQUARE) is False

    def test_degenerate_polygon_contains_nothing(self):
        assert point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)]) is False

    def test_concave_polygon(self):
        """An L-shape: the notch must not count as inside.

        Real bed bays are not convex once a bedside cabinet is excluded, so
        this is not a hypothetical.
        """
        l_shape = [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ]
        assert point_in_polygon((0.5, 0.5), l_shape) is True
        assert point_in_polygon((0.5, 2.5), l_shape) is True
        assert point_in_polygon((2.5, 2.5), l_shape) is False  # the notch


class TestRectangle:
    def test_unrotated_rectangle_has_expected_extent(self):
        corners = rectangle((0.0, 0.0), length=2.0, width=1.0, angle_deg=0.0)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        assert min(xs) == pytest.approx(-1.0)
        assert max(xs) == pytest.approx(1.0)
        assert min(ys) == pytest.approx(-0.5)
        assert max(ys) == pytest.approx(0.5)

    def test_rotated_rectangle_still_contains_its_centre(self):
        """The ward's beds sit at ~45 degrees, which is the whole reason
        zones are floor polygons rather than image rectangles."""
        for angle in (0.0, 30.0, 45.0, 90.0, 135.0):
            corners = rectangle((4.0, 6.0), 2.0, 1.0, angle)
            assert point_in_polygon((4.0, 6.0), corners) is True

    def test_rotation_changes_which_points_are_inside(self):
        centre = (0.0, 0.0)
        flat = rectangle(centre, length=2.0, width=0.6, angle_deg=0.0)
        turned = rectangle(centre, length=2.0, width=0.6, angle_deg=90.0)

        probe = (0.8, 0.0)  # along the long axis when unrotated
        assert point_in_polygon(probe, flat) is True
        assert point_in_polygon(probe, turned) is False


class TestZone:
    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            Zone(name="bad", kind="floor", polygon=[(0.0, 0.0), (1.0, 1.0)])

    def test_bed_requires_a_surface_height(self):
        """Without top_m, lying in bed cannot be told from lying on the floor,
        which is the dominant false positive. Better to refuse than guess."""
        with pytest.raises(ValueError, match="needs top_m"):
            Zone(name="bed_1", kind="bed", polygon=SQUARE)

    def test_floor_zone_needs_no_height(self):
        Zone(name="walkway", kind="floor", polygon=SQUARE)  # must not raise

    def test_contains(self):
        zone = Zone(name="bed_1", kind="bed", polygon=SQUARE, top_m=0.6)
        assert zone.contains((1.0, 1.0)) is True
        assert zone.contains((5.0, 5.0)) is False


class TestZoneMap:
    def build(self) -> ZoneMap:
        return ZoneMap(
            [
                Zone("bed_1", "bed", rectangle((0.0, 3.0), 2.0, 1.0, 45.0), top_m=0.55),
                Zone("bed_2", "bed", rectangle((0.0, 6.0), 2.0, 1.0, 45.0), top_m=0.70),
                Zone("doorway", "exclude", rectangle((4.0, 9.0), 2.0, 2.0, 0.0)),
            ]
        )

    def test_finds_the_containing_zone(self):
        assert [z.name for z in self.build().at((0.0, 6.0))] == ["bed_2"]

    def test_outside_everything_is_empty(self):
        assert self.build().at((10.0, 10.0)) == []

    def test_none_position_is_empty(self):
        """No floor contact means no zone, and must not raise."""
        assert self.build().at(None) == []

    def test_first_of_kind(self):
        zones = self.build()
        bed = zones.first_of_kind((0.0, 3.0), "bed")
        assert bed is not None and bed.name == "bed_1"
        assert zones.first_of_kind((0.0, 3.0), "chair") is None

    def test_beds_keep_their_own_heights(self):
        """Ward beds are height-adjustable, so this is per-bed calibration,
        never one global constant."""
        zones = self.build()
        assert zones.first_of_kind((0.0, 3.0), "bed").top_m == 0.55
        assert zones.first_of_kind((0.0, 6.0), "bed").top_m == 0.70

    def test_exclusion(self):
        zones = self.build()
        assert zones.is_excluded((4.0, 9.0)) is True
        assert zones.is_excluded((0.0, 6.0)) is False
        assert zones.is_excluded(None) is False

    def test_from_config(self):
        zones = ZoneMap.from_config(
            [
                {
                    "name": "bed_3",
                    "kind": "bed",
                    "top_m": 0.65,
                    "polygon": [[0, 0], [2, 0], [2, 1], [0, 1]],
                },
                {
                    "name": "corridor",
                    "kind": "floor",
                    "polygon": [[2, 0], [4, 0], [4, 4], [2, 4]],
                },
            ]
        )
        assert len(zones.zones) == 2
        bed = zones.first_of_kind((1.0, 0.5), "bed")
        assert bed is not None and bed.top_m == 0.65
