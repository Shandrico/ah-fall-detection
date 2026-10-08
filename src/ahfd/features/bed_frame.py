"""Bed-relative geometry for bed-exit detection.

The fall layer asks "is this body on the floor?". Bed exit asks a different
question -- "has the body *core* crossed the edge of its bed?" -- and that needs
the body measured in the bed's own frame, not the floor's.

The design rests on one deliberate choice: **the decision is made on the body
core (mid-hip and mid-shoulder), never on the limbs.** A patient reaching an arm
over the rail, or dangling a leg, has not left the bed, and a system that alarmed
on either would be switched off within a shift. So this module projects every
COCO-17 joint onto the bed's mattress plane, sorts them into core / lower-limb /
upper-limb groups, and reports the *core* distance to each bed edge as the thing
that decides an exit. Legs are reported separately as an earlier, weaker signal;
arms, hands and head are carried only as evidence a nurse can read.

Why the mattress plane and not the floor
-----------------------------------------
`GroundPlane.pixel_to_plane` already shows why: a body supported ~0.6 m above the
floor projects well past the bed when dropped onto the floor plane. Projecting to
the bed surface (`top_m`) puts the joints where the body actually is. The residual
error -- a shoulder is above the mattress, so its ray meets the plane a little
beyond the true point -- is real and grows with height above the surface, which is
exactly why the edge margins in `BedExitThresholds` are centimetres, not
millimetres, and why the decision leans on the hip (near the mattress) more than
the shoulder.

Everything here is pure geometry with no per-track memory: it turns one pose into
one `BedObservation`. The temporal logic -- binding, edge-locking, approach,
hysteresis -- lives in `detect/bed_exit.py`, so this stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import BedEdge, Zone, ZoneMap, point_in_polygon
from ahfd.pose.skeleton import ANKLES, HEAD, HIPS, KNEES, SHOULDERS
from ahfd.types import PersonPose

# COCO joint groups for the bed-exit decision. Core decides; legs inform early;
# everything else is evidence only and never moves the state machine.
CORE_HIP = HIPS            # (11, 12)
CORE_SHOULDER = SHOULDERS  # (5, 6)
LOWER_LIMB = KNEES + ANKLES  # (13, 14, 15, 16)
UPPER_AND_HEAD = (7, 8, 9, 10) + HEAD  # elbows, wrists, nose, eyes, ears

# Hill-Rom-style default guardrails, used when a bed zone lists no edges: a rail
# down each long side, open for the last stretch at the foot, and open ends.
DEFAULT_RAIL_GAP_FOOT_M = 0.48
DEFAULT_RAIL_HEIGHT_M = 0.38


def default_bed_edges(
    rail_gap_foot_m: float = DEFAULT_RAIL_GAP_FOOT_M,
    rail_height_m: float = DEFAULT_RAIL_HEIGHT_M,
) -> tuple[BedEdge, ...]:
    """The standard ward-bed guardrail layout as `BedEdge`s.

    Both long sides railed with an open gap at the foot; both ends open. This is
    what the dashboard zone picker attaches to a bed drawn on the RGB feed, so a
    bed authored by clicking corners is immediately usable for exit detection.
    """
    return (
        BedEdge("left", rail=True, rail_height_m=rail_height_m,
                rail_gap_foot_m=rail_gap_foot_m),
        BedEdge("right", rail=True, rail_height_m=rail_height_m,
                rail_gap_foot_m=rail_gap_foot_m),
        BedEdge("head", rail=False),
        BedEdge("foot", rail=False),
    )


@dataclass(frozen=True)
class BedFrame:
    """A bed's own coordinate frame, derived from its floor polygon.

    x runs along the long axis (head -> foot), y across it (one rail to the
    other), origin at the footprint centre. Distances are in metres.
    """

    origin: tuple[float, float]
    ex: tuple[float, float]  # unit long axis
    ey: tuple[float, float]  # unit short axis (ex rotated +90 deg)
    half_length: float
    half_width: float
    foot_at_far_end: bool
    edges: tuple[BedEdge, ...]

    @classmethod
    def from_zone(cls, zone: Zone) -> "BedFrame":
        poly = np.asarray(zone.polygon, dtype=float)
        origin = poly.mean(axis=0)

        # Long axis = direction of the longest polygon edge. For a rectangle the
        # two long sides are longest and parallel, so either gives the same axis.
        best_len, best_vec = -1.0, np.array([1.0, 0.0])
        n = len(poly)
        for i in range(n):
            v = poly[(i + 1) % n] - poly[i]
            length = float(np.hypot(v[0], v[1]))
            if length > best_len:
                best_len, best_vec = length, v
        ex = best_vec / (np.linalg.norm(best_vec) or 1.0)
        ey = np.array([-ex[1], ex[0]])  # +90 degrees

        # Extent by projecting every corner onto the axes.
        rel = poly - origin
        bx = rel @ ex
        by = rel @ ey
        half_length = float((bx.max() - bx.min()) / 2.0)
        half_width = float((by.max() - by.min()) / 2.0)
        # Recentre the origin on the bbox centre so bx/by are symmetric.
        centre_shift = ex * ((bx.max() + bx.min()) / 2.0) + ey * (
            (by.max() + by.min()) / 2.0
        )
        origin = origin + centre_shift

        edges = zone.edges or default_bed_edges()
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            ex=(float(ex[0]), float(ex[1])),
            ey=(float(ey[0]), float(ey[1])),
            half_length=half_length,
            half_width=half_width,
            foot_at_far_end=zone.foot_at_far_end,
            edges=edges,
        )

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Floor point (metres) -> bed-frame (bx along length, by across)."""
        dx, dy = x - self.origin[0], y - self.origin[1]
        bx = dx * self.ex[0] + dy * self.ex[1]
        by = dx * self.ey[0] + dy * self.ey[1]
        return bx, by

    def _edge(self, side: str) -> BedEdge | None:
        for e in self.edges:
            if e.side == side:
                return e
        return None

    def _foot_bx(self) -> float:
        """Long-axis coordinate of the foot end."""
        return self.half_length if self.foot_at_far_end else -self.half_length

    def signed_distances(self, bx: float, by: float) -> dict[str, float]:
        """Signed distance from a bed-frame point to each side, in metres.

        Positive is inside the bed, negative is outside. The four sides:
        left (+y), right (-y), head and foot (the two ends of the long axis).
        """
        foot = self._foot_bx()
        head = -foot
        return {
            "left": self.half_width - by,
            "right": by + self.half_width,
            "head": (bx - head) if foot > 0 else (head - bx),
            "foot": (foot - bx) if foot > 0 else (bx - foot),
        }

    def rail_protects(self, side: str, bx: float) -> bool:
        """Is the point at long-axis `bx` behind a raised rail on `side`?

        Only the long sides carry rails, and only outside the open foot gap.
        """
        edge = self._edge(side)
        if edge is None or not edge.rail or side in ("head", "foot"):
            return False
        if edge.rail_gap_foot_m <= 0.0:
            return True
        foot = self._foot_bx()
        # The gap is `rail_gap_foot_m` of length measured from the foot end.
        if foot > 0:
            return bx < foot - edge.rail_gap_foot_m
        return bx > foot + edge.rail_gap_foot_m


@dataclass(frozen=True)
class EdgeEvidence:
    """Per-side summary for one person in one frame."""

    side: str
    has_rail: bool
    d_core: float | None      # min signed distance of valid core joints (m)
    d_leg_min: float | None   # min signed distance of valid leg joints (m)
    legs_over: int            # valid leg joints with d < 0 (outside the line)


@dataclass(frozen=True)
class BedObservation:
    """One person measured against one bed in one frame. No history."""

    track_id: int
    t: float
    bed_name: str | None
    bed_risk: str
    on_bed: bool              # core currently over this bed's footprint
    observable: bool          # core/leg quorum met (see BedExitThresholds)
    n_core_valid: int
    n_leg_valid: int
    n_upper_valid: int
    edges: dict[str, EdgeEvidence]
    core_bed_xy: tuple[float, float] | None  # core centroid in bed frame, logged

    def edge(self, side: str) -> EdgeEvidence | None:
        return self.edges.get(side)


class BedFrameExtractor:
    """Turns a tracked pose into a `BedObservation` against its bed.

    Holds the calibrated ground plane and the zone map; no per-track state. The
    state machine owns the memory and asks this for one observation per frame.
    """

    def __init__(
        self,
        ground: GroundPlane,
        zones: ZoneMap,
        *,
        min_joint_conf: float = 0.30,
        min_core_valid: int = 1,
        min_total_valid: int = 4,
    ):
        self.ground = ground
        self.zones = zones
        self.min_joint_conf = min_joint_conf
        self.min_core_valid = min_core_valid
        self.min_total_valid = min_total_valid
        self._frames: dict[str, BedFrame] = {
            z.name: BedFrame.from_zone(z) for z in zones.beds()
        }

    def has_beds(self) -> bool:
        return bool(self._frames)

    def _project(
        self, person: PersonPose, idx: int, valid: np.ndarray, top_m: float
    ) -> tuple[float, float] | None:
        """Project one joint onto the mattress plane, in floor metres."""
        if not valid[idx]:
            return None
        return self.ground.pixel_to_plane(
            float(person.keypoints[idx, 0]),
            float(person.keypoints[idx, 1]),
            top_m,
        )

    def _mid(
        self, person: PersonPose, indices: tuple[int, ...], valid: np.ndarray, top_m: float
    ) -> tuple[float, float] | None:
        pts = [self._project(person, i, valid, top_m) for i in indices]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        return (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )

    def candidate_bed(self, person: PersonPose) -> Zone | None:
        """The bed whose footprint currently holds this person's hips.

        Used for binding. The mid-hip projected at mattress height is the single
        most reliable "in this bed" test -- more so than the floor contact, which
        for someone lying down lands well past the bed (see ground.py).
        """
        valid = person.valid_mask(self.min_joint_conf)
        for zone in self.zones.beds():
            hip = self._mid(person, CORE_HIP, valid, zone.top_m or 0.0)
            if hip is not None and point_in_polygon(hip, zone.polygon):
                return zone
        return None

    def observe(self, person: PersonPose, t: float, zone: Zone) -> BedObservation:
        """Measure this person against a specific bed."""
        frame = self._frames.get(zone.name)
        if frame is None:  # a zone added after construction; derive on demand
            frame = BedFrame.from_zone(zone)
            self._frames[zone.name] = frame
        top_m = zone.top_m or 0.0
        valid = person.valid_mask(self.min_joint_conf)

        def bed_pts(indices: tuple[int, ...]) -> list[tuple[float, float]]:
            out = []
            for i in indices:
                p = self._project(person, i, valid, top_m)
                if p is not None:
                    out.append(frame.to_frame(*p))
            return out

        core_pts = bed_pts(CORE_HIP + CORE_SHOULDER)
        leg_pts = bed_pts(LOWER_LIMB)
        n_upper = len(bed_pts(UPPER_AND_HEAD))

        core_centre = (
            (float(np.mean([p[0] for p in core_pts])),
             float(np.mean([p[1] for p in core_pts])))
            if core_pts else None
        )

        edges: dict[str, EdgeEvidence] = {}
        for side in ("left", "right", "head", "foot"):
            core_d = [frame.signed_distances(bx, by)[side] for bx, by in core_pts]
            leg_d = [frame.signed_distances(bx, by)[side] for bx, by in leg_pts]
            # Rail protection judged at the core's position along the bed.
            has_rail = (
                frame.rail_protects(side, core_centre[0])
                if core_centre is not None
                else frame.rail_protects(side, 0.0)
            )
            edges[side] = EdgeEvidence(
                side=side,
                has_rail=has_rail,
                d_core=min(core_d) if core_d else None,
                d_leg_min=min(leg_d) if leg_d else None,
                legs_over=sum(1 for d in leg_d if d < 0.0),
            )

        n_core = len(core_pts)
        n_leg = len(leg_pts)
        observable = n_core >= self.min_core_valid and (
            n_core + n_leg
        ) >= self.min_total_valid

        on_bed = core_centre is not None and point_in_polygon(
            # back to floor coords for the footprint test
            (
                frame.origin[0] + core_centre[0] * frame.ex[0] + core_centre[1] * frame.ey[0],
                frame.origin[1] + core_centre[0] * frame.ex[1] + core_centre[1] * frame.ey[1],
            ),
            zone.polygon,
        )

        return BedObservation(
            track_id=person.track_id if person.track_id is not None else -1,
            t=t,
            bed_name=zone.name,
            bed_risk=zone.risk_level,
            on_bed=on_bed,
            observable=observable,
            n_core_valid=n_core,
            n_leg_valid=n_leg,
            n_upper_valid=n_upper,
            edges=edges,
            core_bed_xy=core_centre,
        )
