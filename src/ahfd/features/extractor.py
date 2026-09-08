"""Metric features per tracked person.

Everything here is in metres, metres per second, or degrees. That is the whole
point: a threshold like 0.40 m means the same thing at the near bed and the
far one, so it is tuned once for a ward rather than once per camera.

A subtlety worth understanding, because it shapes the feature set
--------------------------------------------------------------------
`GroundPlane.joint_height` assumes the body stands on a vertical line above
its floor contact point. That is true when somebody is upright and false once
they are horizontal, and the error is not symmetric. Work an example: a person
lying with their head 1.7 m further from the camera than their feet has a head
at 0.15 m, but the vertical-line estimate reports roughly 0.69 m, because the
head's ray is being intersected with a line that the head is nowhere near.

So a naive "is every joint below 0.75 m" test does *not* reliably identify a
fallen person. Two consequences:

* Heights are used for **change** (the drop, `v_z`), where the bias largely
  cancels and the direction is right -- going horizontal always makes the
  estimate fall.
* Posture is decided by **`floor_spread`**, which is the complementary test
  and much stronger. Project every joint onto the floor as if it lay there. A
  fallen person genuinely is on the floor, so their projections spread over
  about a body length, ~1.5-2.5 m. A standing person's head ray, extended to
  the floor, lands metres beyond their feet -- around 11 m for someone at 6 m
  under this camera geometry -- or misses the floor entirely.

That inversion is the useful part: **upright bodies produce a large floor
spread, fallen bodies a small one.** It is the opposite of the image-space
intuition that a lying person "looks wider", and unlike that intuition it does
not depend on which way the person is facing or where they are in frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import ZoneMap
from ahfd.pose.skeleton import ANKLES, HEAD, HIPS, SHOULDERS
from ahfd.types import PersonPose

# Beyond this, a floor projection is treated as "does not meaningfully touch
# the floor" rather than a real position -- the ray is so shallow that a pixel
# of noise moves it tens of metres.
MAX_FLOOR_RANGE_M = 25.0

# Window used for the velocity fit. At 15 fps this is a third of a second,
# comfortably inside a fall's 0.3-0.8 s impact phase.
VELOCITY_WINDOW = 5

# Window for the stillness measure used to confirm a fall.
MOTION_WINDOW_S = 1.0

# An ankle *keypoint* is the joint, not the sole -- it sits roughly 8 cm above
# the floor on a standing adult. Projecting it straight onto the floor
# therefore lands systematically too far away: the ray keeps descending past
# the real contact point. At 6 m under a 2.6 m camera the error is +0.19 m,
# and it grows with range, so it is worth removing rather than tolerating --
# it is large enough to push a body across a bed-zone boundary.
#
# The correction is exact. Along a ray, height falls linearly with distance,
# so if the ray meets the floor at distance D, it passes through height h_a at
#     D * (1 - h_a / camera_height)
ANKLE_KEYPOINT_HEIGHT_M = 0.08

# Above this floor_spread a body is unambiguously upright, so it cannot be
# lying in a bed. Measured: a prone body spreads ~1.6 m at any range, an
# upright one 5 m at 4 m range and over 7 m at 6 m.
UPRIGHT_SPREAD_M = 4.0


@dataclass(frozen=True)
class Features:
    """One person, one frame, in metric units."""

    track_id: int
    t: float

    contact_xy: tuple[float, float] | None
    range_m: float | None  # lens-to-person distance

    h_torso: float | None  # shoulders+hips centroid, metres above floor
    h_head: float | None
    h_max: float | None  # highest joint
    h_min: float | None

    floor_spread: float  # metres; large when upright, ~body length when down
    v_z: float  # metres/second, negative is downward
    motion: float  # mean floor speed, metres/second

    n_valid_kp: int
    mean_conf: float
    zones: tuple[str, ...] = ()
    supported_by_bed: str | None = None  # name of the bed holding them up
    bed_top_m: float | None = None
    in_excluded_zone: bool = False

    @property
    def height_spread(self) -> float | None:
        if self.h_max is None or self.h_min is None:
            return None
        return self.h_max - self.h_min

    def has_geometry(self) -> bool:
        return self.contact_xy is not None and self.h_torso is not None


@dataclass
class _History:
    """Recent metric state for one track."""

    heights: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=VELOCITY_WINDOW)
    )
    positions: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=64)
    )


class FeatureExtractor:
    """Turns tracked 2D poses into metric features."""

    def __init__(
        self,
        ground: GroundPlane,
        zones: ZoneMap | None = None,
        min_keypoint_score: float = 0.3,
    ):
        self.ground = ground
        self.zones = zones or ZoneMap()
        self.min_keypoint_score = min_keypoint_score
        self._history: dict[int, _History] = {}

    # ------------------------------------------------------------- helpers

    def _mean_height(
        self,
        person: PersonPose,
        indices: tuple[int, ...],
        contact: tuple[float, float],
        valid: np.ndarray,
    ) -> float | None:
        values = []
        for i in indices:
            if not valid[i]:
                continue
            h = self.ground.joint_height(
                float(person.keypoints[i, 0]), float(person.keypoints[i, 1]), contact
            )
            if h is not None:
                values.append(h)
        return float(np.mean(values)) if values else None

    def _contact_point(
        self, person: PersonPose, valid: np.ndarray
    ) -> tuple[float, float] | None:
        """Estimate where this person meets the floor.

        Prefers the ankles, which are the actual contact for anyone upright.
        Falls back to the bottom-most confident joint, since for a fallen
        person every joint is near the floor and the lowest in frame is the
        closest thing to a contact point available.
        """
        candidates = [i for i in ANKLES if valid[i]]
        used_ankles = bool(candidates)
        if not candidates:
            confident = np.flatnonzero(valid)
            if confident.size == 0:
                return None
            # Largest v is lowest in the image.
            candidates = [int(confident[np.argmax(person.keypoints[confident, 1])])]

        # Ankles sit a little above the floor; correct for that (see
        # ANKLE_KEYPOINT_HEIGHT_M). On the fallback path the joint's height is
        # unknown -- it may be a knee or a shoulder -- so no correction is
        # applied and the contact point stays biased outward. That is the
        # honest option: a wrong correction would be worse than none.
        shrink = (
            1.0 - ANKLE_KEYPOINT_HEIGHT_M / self.ground.height_m
            if used_ankles
            else 1.0
        )

        points = []
        for i in candidates:
            xy = self.ground.pixel_to_floor(
                float(person.keypoints[i, 0]), float(person.keypoints[i, 1])
            )
            if xy is not None and math.hypot(*xy) <= MAX_FLOOR_RANGE_M:
                points.append((xy[0] * shrink, xy[1] * shrink))
        if not points:
            return None
        return (
            float(np.mean([p[0] for p in points])),
            float(np.mean([p[1] for p in points])),
        )

    def _supporting_bed(
        self, person: PersonPose, valid: np.ndarray, floor_spread: float
    ):
        """The bed holding this person up, if any.

        Not answerable from the floor contact point. Somebody lying in bed is
        supported ~0.6 m above the floor, so their floor projection lands well
        past the bed -- about 1.5 m beyond it at 6 m range -- and a floor
        polygon never contains them. The zone that exists to recognise
        patients in bed would miss every one of them.

        So each bed is tested at *its own* surface height.

        One ambiguity remains and is worth naming: somebody standing a little
        way behind a bed also projects into it at bed height, because the ray
        passes over the bed on its way to their feet. `floor_spread` separates
        the two -- an upright body spreads several metres, a body lying in bed
        spreads about its own length -- so an upright person is never treated
        as supported.
        """
        if floor_spread > UPRIGHT_SPREAD_M:
            return None

        candidates = [i for i in ANKLES if valid[i]]
        if not candidates:
            confident = np.flatnonzero(valid)
            if confident.size == 0:
                return None
            candidates = [int(confident[np.argmax(person.keypoints[confident, 1])])]

        for zone in self.zones.zones:
            if zone.kind not in ("bed", "chair") or zone.top_m is None:
                continue
            for i in candidates:
                xy = self.ground.pixel_to_plane(
                    float(person.keypoints[i, 0]),
                    float(person.keypoints[i, 1]),
                    zone.top_m,
                )
                if xy is not None and zone.contains(xy):
                    return zone
        return None

    def _floor_spread(self, person: PersonPose, valid: np.ndarray) -> float:
        """Spread of every joint's floor projection, in metres.

        Returns infinity if any confident joint's ray misses the floor or
        lands implausibly far away -- both mean "this body is not lying on the
        floor", which is exactly what the caller needs to know.
        """
        points: list[tuple[float, float]] = []
        for i in np.flatnonzero(valid):
            xy = self.ground.pixel_to_floor(
                float(person.keypoints[i, 0]), float(person.keypoints[i, 1])
            )
            if xy is None or math.hypot(*xy) > MAX_FLOOR_RANGE_M:
                return math.inf
            points.append(xy)

        if len(points) < 2:
            return math.inf

        arr = np.asarray(points)
        # Max pairwise distance; 17 points, so brute force is free.
        diffs = arr[:, None, :] - arr[None, :, :]
        return float(np.sqrt((diffs**2).sum(axis=-1)).max())

    @staticmethod
    def _slope(samples: deque[tuple[float, float]]) -> float:
        """Least-squares slope of height against time, metres per second.

        A least-squares fit over several frames rather than a first difference:
        differencing two noisy keypoint positions produces velocity noise of
        the same order as a real fall, which is what makes single-frame
        velocity triggers fire constantly.
        """
        if len(samples) < 3:
            return 0.0
        t = np.array([s[0] for s in samples], dtype=float)
        h = np.array([s[1] for s in samples], dtype=float)
        t = t - t.mean()
        denom = float((t * t).sum())
        if denom < 1e-9:
            return 0.0
        return float((t * (h - h.mean())).sum() / denom)

    def _motion(self, history: _History, now: float) -> float:
        """Mean floor speed over the recent window, metres per second."""
        recent = [p for p in history.positions if now - p[0] <= MOTION_WINDOW_S]
        if len(recent) < 2:
            return 0.0
        speeds = []
        for (t0, x0, y0), (t1, x1, y1) in zip(recent, recent[1:]):
            dt = t1 - t0
            if dt > 1e-6:
                speeds.append(math.hypot(x1 - x0, y1 - y0) / dt)
        return float(np.mean(speeds)) if speeds else 0.0

    # --------------------------------------------------------------- public

    def extract(self, person: PersonPose, t: float) -> Features | None:
        """Metric features for one tracked person. None if untracked."""
        if person.track_id is None:
            return None

        valid = person.valid_mask(self.min_keypoint_score)
        n_valid = int(np.count_nonzero(valid))
        mean_conf = float(person.scores[valid].mean()) if n_valid else 0.0

        history = self._history.setdefault(person.track_id, _History())
        contact = self._contact_point(person, valid) if n_valid else None

        if contact is None:
            return Features(
                track_id=person.track_id,
                t=t,
                contact_xy=None,
                range_m=None,
                h_torso=None,
                h_head=None,
                h_max=None,
                h_min=None,
                floor_spread=math.inf,
                v_z=0.0,
                motion=0.0,
                n_valid_kp=n_valid,
                mean_conf=mean_conf,
            )

        torso = self._mean_height(person, SHOULDERS + HIPS, contact, valid)
        head = self._mean_height(person, HEAD, contact, valid)

        all_heights = []
        for i in np.flatnonzero(valid):
            h = self.ground.joint_height(
                float(person.keypoints[i, 0]), float(person.keypoints[i, 1]), contact
            )
            if h is not None:
                all_heights.append(h)

        if torso is not None:
            history.heights.append((t, torso))
        history.positions.append((t, contact[0], contact[1]))

        floor_spread = self._floor_spread(person, valid)
        zones_here = self.zones.at(contact)

        # A supporting bed is found geometrically at bed height, not from the
        # floor contact -- see _supporting_bed. Fall back to a plain floor
        # lookup so an upright person standing in a bed bay still reports the
        # zone they are in.
        bed = self._supporting_bed(person, valid, floor_spread)
        if bed is not None and bed.name not in [z.name for z in zones_here]:
            zones_here = list(zones_here) + [bed]

        return Features(
            track_id=person.track_id,
            t=t,
            contact_xy=contact,
            range_m=self.ground.floor_distance(contact),
            h_torso=torso,
            h_head=head,
            h_max=max(all_heights) if all_heights else None,
            h_min=min(all_heights) if all_heights else None,
            floor_spread=floor_spread,
            v_z=self._slope(history.heights),
            motion=self._motion(history, t),
            n_valid_kp=n_valid,
            mean_conf=mean_conf,
            zones=tuple(z.name for z in zones_here),
            supported_by_bed=bed.name if bed else None,
            bed_top_m=bed.top_m if bed else None,
            in_excluded_zone=self.zones.is_excluded(contact),
        )

    def retain_only(self, live_ids: set[int]) -> None:
        for tid in list(self._history):
            if tid not in live_ids:
                del self._history[tid]
