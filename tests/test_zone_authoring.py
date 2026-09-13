"""Back-projection that powers `ahfd calibrate-zones` (the click-to-draw tool).

The interactive cv2 window can't be unit-tested, but the pixel -> floor-metres
conversion it relies on can, and that's the part that must be correct.
"""

from __future__ import annotations

import pytest

from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import Zone, polygon_from_pixels
from ahfd.types import Intrinsics


def _ground():
    # pitch 20 deg -> horizon sits in-frame (~v=43), so pixels above it miss the floor
    intr = Intrinsics(width=1920, height=1080, fx=1366.0, fy=1366.0, cx=960.0, cy=540.0)
    return GroundPlane(intr, height_m=2.5, pitch_deg=20.0)


def test_backprojects_clicks_to_a_floor_polygon():
    gp = _ground()
    pixels = [(700, 800), (1200, 800), (1200, 1000), (700, 1000)]
    poly = polygon_from_pixels(gp, pixels, plane_z=0.0)
    assert len(poly) == 4
    assert all(len(p) == 2 for p in poly)
    assert all(p[1] > 0 for p in poly)  # all in front of the camera
    # the result must form a usable bed zone
    Zone(name="bed_1", kind="bed", polygon=poly, top_m=0.4, risk_level="high")


def test_bed_height_plane_lands_nearer_than_the_floor():
    gp = _ground()
    px = [(960, 800)]
    floor_y = polygon_from_pixels(gp, px, plane_z=0.0)[0][1]
    bed_y = polygon_from_pixels(gp, px, plane_z=0.4)[0][1]
    # the same pixel projected at bed height is nearer the camera than at the floor
    assert bed_y < floor_y


def test_rejects_pixels_above_the_horizon():
    gp = _ground()
    with pytest.raises(ValueError):
        polygon_from_pixels(gp, [(960, 5), (900, 5), (1000, 8)], plane_z=0.0)
