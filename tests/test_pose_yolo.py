"""YOLO-pose backend tests.

Ultralytics is not installed by default (it is a benchmark-only, AGPL extra),
so these test the two things that do not need it: the result-conversion logic,
with plain numpy arrays standing in for a YOLO result, and the build_estimator
routing, with a stubbed `ultralytics` module. No torch, no model download.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ahfd.config import PoseConfig
from ahfd.pose.base import build_estimator
from ahfd.pose.yolo_pose import VALID_SIZES, _to_people
from ahfd.types import NUM_KEYPOINTS


def fake_result(n_people: int, seed: int = 0):
    """(keypoints, box_scores) shaped like an Ultralytics pose result."""
    rng = np.random.default_rng(seed)
    kps = rng.uniform(0, 640, size=(n_people, NUM_KEYPOINTS, 3)).astype(np.float32)
    kps[:, :, 2] = rng.uniform(0.4, 1.0, size=(n_people, NUM_KEYPOINTS))  # confidences
    box = rng.uniform(0.5, 1.0, size=n_people).astype(np.float32)
    return kps, box


class TestConversion:
    def test_one_person(self):
        kps, box = fake_result(1)
        people = _to_people(kps, box, min_score=0.3)
        assert len(people) == 1
        assert people[0].keypoints.shape == (NUM_KEYPOINTS, 2)
        assert people[0].scores.shape == (NUM_KEYPOINTS,)
        assert people[0].score == pytest.approx(float(box[0]))

    def test_multiple_people(self):
        kps, box = fake_result(4)
        assert len(_to_people(kps, box, 0.3)) == 4

    def test_no_people(self):
        empty = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
        assert _to_people(empty, np.zeros((0,)), 0.3) == ()

    def test_falls_back_to_mean_conf_without_box_scores(self):
        kps, _ = fake_result(1, seed=2)
        people = _to_people(kps, None, min_score=0.3)
        assert len(people) == 1
        assert 0.0 <= people[0].score <= 1.0

    def test_wrong_keypoint_count_is_rejected(self):
        """A non-COCO-17 model output must not silently produce garbage."""
        bad = np.zeros((1, 13, 3), dtype=np.float32)
        assert _to_people(bad, None, 0.3) == ()

    def test_xy_and_confidence_are_split_correctly(self):
        kps = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
        kps[0, 0] = [100.0, 200.0, 0.9]
        people = _to_people(kps, np.array([0.8]), 0.3)
        assert tuple(people[0].keypoints[0]) == (100.0, 200.0)
        assert people[0].scores[0] == pytest.approx(0.9)


class TestBuildEstimator:
    def test_routes_yolo_with_stubbed_ultralytics(self, monkeypatch):
        """build_estimator must construct the YOLO backend. ultralytics is
        stubbed so nothing downloads and no torch is needed."""
        created = {}

        class FakeYOLO:
            def __init__(self, name):
                created["name"] = name

        stub = types.ModuleType("ultralytics")
        stub.YOLO = FakeYOLO
        monkeypatch.setitem(sys.modules, "ultralytics", stub)

        est = build_estimator(PoseConfig(backend="yolo", model_size="s"))
        assert est.name == "yolo11s-pose"
        assert created["name"] == "yolo11s-pose.pt"

    def test_unknown_backend_lists_yolo(self):
        with pytest.raises(ValueError) as excinfo:
            build_estimator(PoseConfig(backend="nope"))
        assert "yolo" in str(excinfo.value)


class TestModelSize:
    def test_valid_sizes(self):
        assert VALID_SIZES == ("n", "s", "m", "l", "x")

    def test_rejects_bad_size(self, monkeypatch):
        import types as _t

        stub = _t.ModuleType("ultralytics")
        stub.YOLO = lambda name: None
        monkeypatch.setitem(sys.modules, "ultralytics", stub)
        from ahfd.pose.yolo_pose import YOLOPoseEstimator

        with pytest.raises(ValueError, match="model_size"):
            YOLOPoseEstimator(model_size="huge")
