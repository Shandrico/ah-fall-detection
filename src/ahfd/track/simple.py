"""Greedy IoU tracker.

Deliberately simple, and deliberately behind an interface. Stable identities
matter for two reasons that show up later:

* Smoothing and velocity are per-person, so mixing bodies corrupts both.
* A fall must not be triggered by an identity switch. The eventual rule is
  that a track needs a minimum age before it may trigger, which is only
  meaningful if ids are stable.

This is a placeholder for ByteTrack (via `supervision`, MIT). It handles the
easy cases -- people walking about, brief detection dropouts -- and will lose
identity through a long occlusion, which a ward has plenty of (curtains, bed
rails, IV poles). Swap it out before trusting any false-alarm number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ahfd.types import PersonPose, PoseFrame

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class Track:
    """One tracked identity."""

    track_id: int
    bbox: BBox
    age: int = 1  # frames in which this track was matched
    misses: int = 0  # consecutive frames unmatched
    first_t: float = 0.0
    last_t: float = 0.0

    def age_seconds(self) -> float:
        return max(0.0, self.last_t - self.first_t)


@dataclass
class SimpleTracker:
    """Assigns stable track ids to the people in successive PoseFrames."""

    iou_threshold: float = 0.3
    max_misses: int = 15  # ~1 s at 15 fps before an identity is retired
    min_keypoint_score: float = 0.3

    _tracks: list[Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def update(self, pose_frame: PoseFrame) -> PoseFrame:
        """Return the same frame with track ids attached."""
        # Detections we can actually match on: a person with too few confident
        # keypoints has no meaningful box.
        detections: list[tuple[int, BBox]] = []
        for i, person in enumerate(pose_frame.people):
            box = person.bbox(self.min_keypoint_score)
            if box is not None:
                detections.append((i, box))

        # Greedy matching, highest IoU first. Fine for a handful of people; a
        # proper assignment comes with the ByteTrack swap.
        candidates = [
            (iou(box, track.bbox), det_i, track_i)
            for det_i, box in detections
            for track_i, track in enumerate(self._tracks)
        ]
        candidates = [c for c in candidates if c[0] >= self.iou_threshold]
        candidates.sort(key=lambda c: c[0], reverse=True)

        det_to_track: dict[int, Track] = {}
        claimed_dets: set[int] = set()
        claimed_tracks: set[int] = set()
        for _, det_i, track_i in candidates:
            if det_i in claimed_dets or track_i in claimed_tracks:
                continue
            det_to_track[det_i] = self._tracks[track_i]
            claimed_dets.add(det_i)
            claimed_tracks.add(track_i)

        matched = {id(t) for t in det_to_track.values()}

        # Existing tracks that went unmatched this frame accumulate a miss.
        # Done before new tracks are added, so a fresh track is never penalised
        # for the frame it was born in.
        for track in self._tracks:
            if id(track) not in matched:
                track.misses += 1

        # Update matched tracks; open a new one for every unmatched detection.
        for det_i, box in detections:
            track = det_to_track.get(det_i)
            if track is None:
                track = Track(
                    track_id=self._next_id,
                    bbox=box,
                    first_t=pose_frame.t,
                    last_t=pose_frame.t,
                )
                self._next_id += 1
                self._tracks.append(track)
                det_to_track[det_i] = track
            else:
                track.bbox = box
                track.age += 1
                track.misses = 0
                track.last_t = pose_frame.t

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        people = tuple(
            person.with_track_id(det_to_track[i].track_id)
            if i in det_to_track
            else person
            for i, person in enumerate(pose_frame.people)
        )

        return PoseFrame(
            t=pose_frame.t,
            index=pose_frame.index,
            width=pose_frame.width,
            height=pose_frame.height,
            people=people,
        )

    @property
    def live_ids(self) -> set[int]:
        return {t.track_id for t in self._tracks}

    def track(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None
