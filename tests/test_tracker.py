"""Tracker behaviour, driven by synthetic poses.

No camera, no model, no video. Identity stability is a correctness property of
the decision layer, so it has to be testable from numbers alone -- which is
also what lets this work continue while the D435i is unplugged.
"""

from __future__ import annotations

import numpy as np

from ahfd.track import SimpleTracker, iou
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame


def person_at(x: float, y: float, w: float = 40.0, h: float = 160.0) -> PersonPose:
    """A synthetic upright person whose keypoints span the given box."""
    ys = np.linspace(y, y + h, NUM_KEYPOINTS, dtype=np.float32)
    xs = np.full(NUM_KEYPOINTS, x + w / 2.0, dtype=np.float32)
    # Spread the shoulders and hips so the box has real width.
    xs[5], xs[6] = x, x + w
    xs[11], xs[12] = x, x + w
    keypoints = np.stack([xs, ys], axis=1)
    scores = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    return PersonPose(keypoints=keypoints, scores=scores, score=1.0)


def frame(t: float, index: int, *people: PersonPose) -> PoseFrame:
    return PoseFrame(t=t, index=index, width=640, height=480, people=tuple(people))


class TestIou:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes sharing a 5x10 strip: 50 / 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50.0 / 150.0

    def test_degenerate_box_is_safe(self):
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestSimpleTracker:
    def test_assigns_an_id(self):
        tracker = SimpleTracker()
        out = tracker.update(frame(0.0, 0, person_at(100, 100)))
        assert out.people[0].track_id == 0

    def test_same_person_keeps_id_while_moving_slowly(self):
        tracker = SimpleTracker()
        ids = []
        for i in range(10):
            out = tracker.update(frame(i / 15.0, i, person_at(100 + i * 3, 100)))
            ids.append(out.people[0].track_id)
        assert len(set(ids)) == 1, "identity should be stable, got " + repr(ids)

    def test_two_people_get_distinct_ids(self):
        tracker = SimpleTracker()
        out = tracker.update(frame(0.0, 0, person_at(50, 100), person_at(400, 100)))
        assert {p.track_id for p in out.people} == {0, 1}

    def test_a_teleport_creates_a_new_identity(self):
        """A jump with no overlap must not silently inherit the old id."""
        tracker = SimpleTracker()
        first = tracker.update(frame(0.0, 0, person_at(50, 100)))
        second = tracker.update(frame(1 / 15.0, 1, person_at(500, 100)))
        assert first.people[0].track_id != second.people[0].track_id

    def test_track_is_retired_after_max_misses(self):
        tracker = SimpleTracker(max_misses=3)
        tracker.update(frame(0.0, 0, person_at(100, 100)))
        assert tracker.live_ids == {0}

        for i in range(1, 6):
            tracker.update(frame(i / 15.0, i))  # nobody in frame

        assert tracker.live_ids == set(), "stale track should have been dropped"

    def test_new_track_is_not_penalised_on_its_first_frame(self):
        """Regression: the age-out pass must run before new tracks are added."""
        tracker = SimpleTracker(max_misses=0)
        tracker.update(frame(0.0, 0, person_at(100, 100)))
        assert tracker.live_ids == {0}

    def test_age_accumulates_with_matches(self):
        tracker = SimpleTracker()
        for i in range(5):
            tracker.update(frame(i / 15.0, i, person_at(100, 100)))
        track = tracker.track(0)
        assert track is not None
        assert track.age == 5
        # Age in seconds is what the eventual min-track-age fall guard uses.
        assert track.age_seconds() > 0.0

    def test_low_confidence_person_gets_no_id(self):
        """Too few confident keypoints means no usable box, so no identity."""
        tracker = SimpleTracker(min_keypoint_score=0.5)
        weak = PersonPose(
            keypoints=np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
            scores=np.full(NUM_KEYPOINTS, 0.1, dtype=np.float32),
            score=0.1,
        )
        out = tracker.update(frame(0.0, 0, weak))
        assert out.people[0].track_id is None
        assert tracker.live_ids == set()
