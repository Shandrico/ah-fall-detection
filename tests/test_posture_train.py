"""Posture-classifier training pipeline -- camera-free, synthetic data.

The heavy build_dataset->features path is exercised by the real smoke test on
extracted clips; here we pin the pure logic (segment lookup, placeholder
skipping) and the scikit-learn training path on synthetic, separable features.
"""

from __future__ import annotations

import json

import pytest

# scikit-learn is an optional extra (`.[ml]`); skip this whole module cleanly if
# it isn't installed, so the default test run stays green without it.
pytest.importorskip("sklearn")

from ahfd.ml.posture import (
    FEATURES,
    _labelled_segments,
    _posture_at,
    train,
)


class TestSegmentLookup:
    def test_posture_at_inside_and_gap(self):
        segs = [
            {"start_s": 0.0, "end_s": 10.0, "posture": "upright"},
            {"start_s": 15.0, "end_s": 25.0, "posture": "sitting"},
        ]
        assert _posture_at(segs, 5.0) == "upright"
        assert _posture_at(segs, 20.0) == "sitting"
        assert _posture_at(segs, 12.0) is None  # transition gap -> excluded

    def test_labelled_segments_skips_placeholders(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(
            json.dumps(
                {
                    "clip_id": "p",
                    "duration_s": 30.0,
                    "segments": [
                        {"start_s": 0.0, "end_s": 0.0, "posture": "upright"},  # placeholder
                        {"start_s": 2.0, "end_s": 9.0, "posture": "sitting"},  # real
                    ],
                }
            ),
            encoding="utf-8",
        )
        segs = _labelled_segments(f)
        assert len(segs) == 1
        assert segs[0]["posture"] == "sitting"


def _row(h_torso, spread, rng=6.0):
    """A feature row shaped like `Features` fields."""
    return {
        "h_torso": h_torso,
        "h_head": h_torso + 0.3,
        "h_max": h_torso + 0.4,
        "h_min": 0.05,
        "h_ankle_min": 0.05,
        "floor_spread": spread,
        "range_m": rng,
    }


def _separable_dataset(n_per=12):
    """Three cleanly-separable postures, so a tree should nail them."""
    import random

    random.seed(0)
    rows, labels = [], []
    specs = {
        "upright": (1.05, 10.0),
        "sitting": (0.55, 6.0),
        "on_ground": (0.30, 1.6),
    }
    for posture, (h, s) in specs.items():
        for _ in range(n_per):
            rows.append(_row(h + random.uniform(-0.03, 0.03), s + random.uniform(-0.2, 0.2)))
            labels.append(posture)
    return rows, labels


class TestTrain:
    def test_trains_and_separates(self):
        rows, labels = _separable_dataset()
        result = train(rows, labels, seed=0)
        assert result.classes == ["on_ground", "sitting", "upright"]
        assert result.split_done is True
        assert result.n_test > 0
        assert result.accuracy >= 0.8  # clean separation -> high accuracy
        assert result.model is not None
        # importances cover the feature set and sum ~1
        assert [f for f, _ in result.importances]
        assert set(f for f, _ in result.importances) == set(FEATURES)
        assert abs(sum(v for _, v in result.importances) - 1.0) < 1e-6

    def test_tiny_data_trains_without_holdout(self):
        rows = [_row(1.0, 10.0), _row(0.5, 6.0)]
        labels = ["upright", "sitting"]
        result = train(rows, labels)
        assert result.split_done is False  # too few for a real test split
        assert result.n_test == 0
        assert result.model is not None

    def test_missing_optional_feature_is_imputed(self):
        rows, labels = _separable_dataset()
        for r in rows:  # drop an optional feature entirely
            r["h_ankle_min"] = None
        result = train(rows, labels, seed=0)  # must not crash on None -> imputed
        assert result.accuracy >= 0.8
