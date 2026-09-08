"""Floor zones: beds, chairs, walkable floor, exclusions.

Zones are polygons in **floor coordinates (metres)**, not rectangles in image
space. Two reasons, both visible in the Class C ward plan:

* The beds sit at roughly 45 degrees in a herringbone layout. An axis-aligned
  image rectangle cannot describe an angled bed without swallowing half the
  corridor with it.
* Perspective means a fixed image rectangle covers a completely different
  patch of floor at the near bed than at the far one. A floor polygon is the
  same physical area wherever it appears in frame.

A bed also carries `top_m`, the height of its surface above the floor,
because that is what separates "lying in bed" from "lying on the floor". Real
ward beds are height-adjustable over roughly 0.4-0.8 m, so this is a measured
per-bed calibration value, never a global constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ZoneKind = Literal["bed", "chair", "floor", "exclude"]

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon test.

    Counts crossings of a ray heading in +X. Points exactly on an edge are not
    guaranteed either way, which is fine: zone boundaries are drawn by hand to
    within a few centimetres, so a body's floor position is never meaningfully
    "exactly on" one.
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Does the edge straddle the horizontal line through `point`?
        if (y1 > y) == (y2 > y):
            continue
        # X coordinate where the edge crosses that line.
        t = (y - y1) / (y2 - y1)
        if x < x1 + t * (x2 - x1):
            inside = not inside
    return inside


@dataclass(frozen=True)
class Zone:
    """A named region of floor."""

    name: str
    kind: ZoneKind
    polygon: list[Point]
    top_m: float | None = None  # bed/chair surface height above floor

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                "zone " + repr(self.name) + " needs at least 3 points, got "
                + str(len(self.polygon))
            )
        if self.kind in ("bed", "chair") and self.top_m is None:
            raise ValueError(
                "zone " + repr(self.name) + " of kind " + self.kind
                + " needs top_m (surface height above floor in metres); "
                "without it, lying in bed cannot be told from lying on the floor"
            )

    def contains(self, xy: Point) -> bool:
        return point_in_polygon(xy, self.polygon)


@dataclass
class ZoneMap:
    """All zones for one calibrated camera position."""

    zones: list[Zone] = field(default_factory=list)

    def at(self, xy: Point | None) -> list[Zone]:
        """Every zone containing the floor point. Empty if outside all of them."""
        if xy is None:
            return []
        return [z for z in self.zones if z.contains(xy)]

    def first_of_kind(self, xy: Point | None, kind: ZoneKind) -> Zone | None:
        for zone in self.at(xy):
            if zone.kind == kind:
                return zone
        return None

    def is_excluded(self, xy: Point | None) -> bool:
        """True inside a region we deliberately ignore.

        Doorways, corridors, a mirror, a window onto another ward -- places
        where a detection is either not our patient or not a person at all.
        """
        return any(z.kind == "exclude" for z in self.at(xy))

    @classmethod
    def from_config(cls, entries: list[dict]) -> "ZoneMap":
        """Build from the `zones:` list in a calibration YAML."""
        zones = []
        for entry in entries:
            zones.append(
                Zone(
                    name=str(entry["name"]),
                    kind=entry["kind"],
                    polygon=[(float(p[0]), float(p[1])) for p in entry["polygon"]],
                    top_m=(
                        float(entry["top_m"]) if entry.get("top_m") is not None else None
                    ),
                )
            )
        return cls(zones=zones)


def rectangle(
    centre: Point, length: float, width: float, angle_deg: float
) -> list[Point]:
    """Corners of a rotated rectangle on the floor, for authoring bed zones.

    The ward's beds are angled, so this is the convenient way to describe one:
    its centre, its footprint, and how far it is rotated.
    """
    import math

    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hl, hw = length / 2.0, width / 2.0

    corners = []
    for dx, dy in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
        corners.append(
            (
                centre[0] + dx * cos_a - dy * sin_a,
                centre[1] + dx * sin_a + dy * cos_a,
            )
        )
    return corners
