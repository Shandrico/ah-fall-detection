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


def _person_groups(labels):
    """Several scenario clips per person, with every class for both people."""
    return [f"synthetic_{label}_{i % 2 + 1:02}" for i, label in enumerate(labels)]


class TestTrain:
    def test_trains_and_separates(self):
        rows, labels = _separable_dataset()
        result = train(rows, labels, groups=_person_groups(labels), seed=0)
        assert result.classes == ["on_ground", "sitting", "upright"]
        assert result.split_done is True
        assert result.n_test > 0
        assert result.split_unit == "person"
        assert set(result.train_groups).isdisjoint(result.test_groups)
        assert set(result.train_groups + result.test_groups) == {"person_01", "person_02"}
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
        result = train(rows, labels, groups=_person_groups(labels), seed=0)
        assert result.accuracy >= 0.8

    def test_missing_groups_never_falls_back_to_random_frames(self):
        rows, labels = _separable_dataset()
        result = train(rows, labels)
        assert result.split_done is False
        assert result.n_train == len(rows)
        assert result.n_test == 0
        assert result.split_unit == "none"
        assert "no recording/person groups" in result.split_reason

    def test_multiple_clips_from_one_person_have_no_heldout_score(self):
        rows, labels = _separable_dataset()
        groups = [f"synthetic_{label}_03" for label in labels]
        result = train(rows, labels, groups=groups)
        assert result.split_done is False
        assert result.train_groups == ["person_03"]
        assert result.test_groups == []
        assert result.n_test == 0
        assert "at least two independent person groups" in result.split_reason

    def test_class_only_present_in_one_group_has_training_only_fallback(self):
        rows = [_row(1.0, 10.0)] * 4 + [_row(0.3, 1.6)] * 4
        labels = ["upright"] * 4 + ["on_ground"] * 4
        groups = ["fall_standing_01"] * 4 + ["fall_standing_02"] * 4
        result = train(rows, labels, groups=groups)
        assert result.split_done is False
        assert result.n_test == 0
        assert "no class-complete" in result.split_reason

    def test_tiny_four_class_dataset_splits_by_person_without_stratify_crash(self):
        rows = [_row(1.0, 10.0), _row(0.5, 6.0), _row(0.2, 2.0), _row(0.3, 1.6)] * 2
        labels = ["upright", "sitting", "in_bed", "on_ground"] * 2
        groups = ["clip_01"] * 4 + ["clip_02"] * 4
        result = train(rows, labels, groups=groups)
        assert result.split_done is True
        assert result.n_train == result.n_test == 4
        assert set(result.train_groups).isdisjoint(result.test_groups)
        assert len(result.confusion) == 4

    def test_score_fit_is_group_disjoint_and_saved_model_uses_every_row(self, monkeypatch):
        from sklearn.pipeline import Pipeline

        rows, labels = _separable_dataset()
        groups = _person_groups(labels)
        fitted_rows = []
        original_fit = Pipeline.fit

        def record_fit(self, X, y, **kwargs):
            fitted_rows.append(X[:, FEATURES.index("h_torso")].copy())
            return original_fit(self, X, y, **kwargs)

        monkeypatch.setattr(Pipeline, "fit", record_fit)
        result = train(rows, labels, groups=groups, seed=3)
        assert len(fitted_rows) == 2
        evaluation_indices = [
            i for i, group in enumerate(groups)
            if "person_" + group.rsplit("_", 1)[-1] in result.train_groups
        ]
        assert list(fitted_rows[0]) == [rows[i]["h_torso"] for i in evaluation_indices]
        assert list(fitted_rows[1]) == [row["h_torso"] for row in rows]
        assert result.model.named_steps["clf"].tree_.n_node_samples[0] == len(rows)

    def test_non_subject_group_names_remain_whole_recordings(self):
        rows, labels = _separable_dataset()
        groups = ["recording_A" if i % 2 else "recording_B" for i in range(len(rows))]
        result = train(rows, labels, groups=groups)
        assert result.split_done is True
        assert result.split_unit == "clip"
        assert set(result.train_groups).isdisjoint(result.test_groups)

    def test_split_is_reproducible(self):
        rows, labels = _separable_dataset()
        groups = [f"synthetic_{label}_{i % 3 + 1:02}" for i, label in enumerate(labels)]
        first = train(rows, labels, groups=groups, seed=5)
        second = train(rows, labels, groups=groups, seed=5)
        assert first.train_groups == second.train_groups
        assert first.test_groups == second.test_groups
        assert first.accuracy == second.accuracy

    def test_single_class_is_training_only_even_with_two_people(self):
        result = train([_row(1.0, 10.0)] * 4, ["upright"] * 4,
                       groups=["clip_01"] * 2 + ["clip_02"] * 2)
        assert result.split_done is False
        assert "at least two posture classes" in result.split_reason

    @pytest.mark.parametrize("rows,labels,groups,message", [
        ([], [], None, "nonempty"),
        ([_row(1.0, 10.0)], [], None, "matching lengths"),
        ([_row(1.0, 10.0)], ["upright"], [], "one entry per row"),
    ])
    def test_invalid_input_is_rejected(self, rows, labels, groups, message):
        with pytest.raises(ValueError, match=message):
            train(rows, labels, groups=groups)
