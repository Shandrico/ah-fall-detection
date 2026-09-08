"""Renderer smoke tests.

Rendering is hard to assert pixel-by-pixel and not worth it; what matters is
that the renderers produce a correctly-shaped image and never crash on the
inputs they actually get -- including the metrics overlay, empty frames, and
untracked people. A crash here would take down the live view and the dashboard.
"""

from __future__ import annotations

import numpy as np

from ahfd.capture.base import Frame
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame
from ahfd.viz import render_overlay, render_skeleton


def person(track_id=1, x=320, y=240):
    kp = np.array(
        [[x + (i % 3) * 10, y + i * 8] for i in range(NUM_KEYPOINTS)],
        dtype=np.float32,
    )
    return PersonPose(
        keypoints=kp,
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
        track_id=track_id,
    )


def pose(*people):
    return PoseFrame(t=1.0, index=1, width=640, height=480, people=tuple(people))


METRICS = {
    1: {
        "state": "UPRIGHT",
        "h_torso": 1.15,
        "floor_spread": 6.2,
        "v_z": -0.03,
        "h_ankle_min": 0.06,
        "bed_risk": "high",
        "range_m": 5.6,
    }
}


class TestRenderSkeleton:
    def test_shape_matches_pose(self):
        canvas = render_skeleton(pose(person()))
        assert canvas.shape == (480, 640, 3)

    def test_with_metrics_does_not_crash(self):
        canvas = render_skeleton(pose(person()), metrics=METRICS, states={1: "UPRIGHT"})
        assert canvas.shape == (480, 640, 3)

    def test_empty_pose(self):
        canvas = render_skeleton(pose(), metrics={})
        assert canvas.shape == (480, 640, 3)

    def test_metrics_with_none_values(self):
        """Low-confidence tracks have None metrics; must not crash."""
        m = {1: {"state": "LOW_CONFIDENCE", "h_torso": None, "floor_spread": float("inf"),
                 "v_z": 0.0, "h_ankle_min": None, "bed_risk": None, "range_m": None}}
        canvas = render_skeleton(pose(person()), metrics=m)
        assert canvas.shape == (480, 640, 3)


class TestRenderOverlay:
    def frame(self):
        return Frame(index=1, t=1.0, bgr=np.zeros((480, 640, 3), dtype=np.uint8))

    def test_draws_on_the_frame(self):
        canvas = render_overlay(self.frame(), pose(person()), metrics=METRICS)
        assert canvas.shape == (480, 640, 3)

    def test_alert_banner_does_not_crash(self):
        canvas = render_overlay(
            self.frame(), pose(person()), alert="FALL CONFIRMED track 1"
        )
        assert canvas.shape == (480, 640, 3)

    def test_falls_back_to_black_when_no_rgb(self):
        """A depth-only frame has no bgr; overlay must still return an image."""
        depth_only = Frame(index=1, t=1.0, bgr=None)
        canvas = render_overlay(depth_only, pose(person()), metrics=METRICS)
        assert canvas.shape == (480, 640, 3)

    def test_untracked_person_is_skipped_in_metrics(self):
        untracked = PersonPose(
            keypoints=person().keypoints, scores=person().scores, score=1.0, track_id=None
        )
        canvas = render_overlay(self.frame(), pose(untracked), metrics=METRICS)
        assert canvas.shape == (480, 640, 3)
