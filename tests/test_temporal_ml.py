"""Causality and group-leakage tests for the lightweight temporal baseline."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from ahfd.ml.temporal import (
    MODEL_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_ORDER_SHA256,
    TEMPORAL_SCHEMA_VERSION,
    CausalTemporalSummarizer,
    SampleMetadata,
    compare_grouped,
)


def feature(
    t: float,
    h_torso: float | None,
    *,
    track_id: int = 1,
    confidence: float = 0.9,
    supported: bool = True,
):
    """Small Features-shaped object; the summarizer intentionally duck-types."""
    return SimpleNamespace(
        track_id=track_id,
        t=t,
        contact_xy=(0.0, 4.0) if h_torso is not None else None,
        h_torso=h_torso,
        h_head=None if h_torso is None else h_torso + 0.3,
        floor_spread=1.6,
        v_z=0.0,
        motion=0.02,
        torso_tilt=70.0,
        bed_overlap=0.8,
        h_shoulder=None if h_torso is None else h_torso + 0.1,
        bed_edge_distance_m=0.4,
        mean_conf=confidence,
        n_valid_kp=15,
        supported_by_bed="bed_1" if supported else None,
    )


class TestCausalTemporalSummarizer:
    def test_episode_recline_resets_edge_progress_and_track_age_is_not_model_input(self):
        summarizer = CausalTemporalSummarizer()
        start = feature(0.0, 0.5)
        start.bed_edge_distance_m = 0.5
        summarizer.update(start)
        edge = feature(0.5, 0.7)
        edge.bed_edge_distance_m = 0.2
        moving = summarizer.update(edge)
        assert moving.values["cumulative_edge_progress_m"] == pytest.approx(0.3)
        reclined = feature(1.0, 0.5)
        reclined.bed_edge_distance_m = 0.4
        reset = summarizer.update(reclined, context={"episode_reclined": 1.0})
        assert reset.values["cumulative_edge_progress_m"] == pytest.approx(0.0)
        assert "track_age_s" in TEMPORAL_FEATURE_NAMES
        assert "track_age_s" not in MODEL_FEATURE_NAMES

    def test_schema_names_are_fixed_unique_and_include_missing_masks(self):
        assert len(TEMPORAL_FEATURE_NAMES) == len(set(TEMPORAL_FEATURE_NAMES))
        assert "h_torso__now" in TEMPORAL_FEATURE_NAMES
        assert "h_torso__missing_now" in TEMPORAL_FEATURE_NAMES
        assert "h_torso__slope_0p5s" in TEMPORAL_FEATURE_NAMES
        assert "h_torso__max_rise_8s" in TEMPORAL_FEATURE_NAMES
        assert "h_torso__valid_fraction_4s" in TEMPORAL_FEATURE_NAMES
        assert "bed_edge_distance_m__slope_2s" in TEMPORAL_FEATURE_NAMES
        assert "cusum_g__now" in TEMPORAL_FEATURE_NAMES
        assert "dwell_near_edge_s" in TEMPORAL_FEATURE_NAMES
        assert TEMPORAL_FEATURE_ORDER_SHA256 == hashlib.sha256(
            json.dumps(
                list(TEMPORAL_FEATURE_NAMES),
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()

    def test_known_trailing_window_statistics(self):
        summarizer = CausalTemporalSummarizer()
        summarizer.update(feature(0.0, 1.0))
        summarizer.update(feature(0.5, 2.0))
        row = summarizer.update(feature(1.0, 3.0)).values

        # The 0.5 s window contains only t=.5 and t=1.0.
        assert row["h_torso__mean_0p5s"] == pytest.approx(2.5)
        assert row["h_torso__delta_0p5s"] == pytest.approx(1.0)
        assert row["h_torso__slope_0p5s"] == pytest.approx(2.0)
        assert row["h_torso__max_rise_0p5s"] == pytest.approx(1.0)

        # The 2 s window contains the whole causal prefix.
        assert row["h_torso__mean_2s"] == pytest.approx(2.0)
        assert row["h_torso__delta_2s"] == pytest.approx(2.0)
        assert row["h_torso__slope_2s"] == pytest.approx(2.0)
        assert row["h_torso__valid_fraction_2s"] == 1.0

    def test_future_tail_cannot_change_prefix(self):
        prefix = [feature(0.0, 0.5), feature(0.2, 0.55), feature(0.4, 0.7)]
        a = CausalTemporalSummarizer()
        b = CausalTemporalSummarizer()
        prefix_a = [a.update(item) for item in prefix]
        prefix_b = [b.update(item) for item in prefix]

        # Two radically different futures are appended after the shared prefix.
        a.update(feature(0.6, 2.0))
        b.update(feature(0.6, 0.1))
        b.update(feature(0.8, 0.0))

        assert [row.values for row in prefix_a] == [row.values for row in prefix_b]
        assert prefix_a[-1].values["h_torso__now"] == 0.7

    def test_invalid_observation_is_missing_not_normal(self):
        summarizer = CausalTemporalSummarizer()
        summarizer.update(feature(0.0, 0.5), valid=True)
        row = summarizer.update(
            feature(0.1, 0.0, confidence=0.1, supported=False), valid=False
        ).values

        assert row["monitoring_valid_now"] == 0.0
        assert row["h_torso__now"] is None
        assert row["h_torso__missing_now"] == 1.0
        assert row["h_torso__since_valid_s"] == pytest.approx(0.1)
        assert row["h_torso__valid_fraction_0p5s"] == pytest.approx(0.5)
        # Quality remains visible, explaining why monitoring was unavailable.
        assert row["mean_conf__now"] == pytest.approx(0.1)
        assert row["bed_supported__now"] is None

    def test_long_gap_resets_instead_of_bridging_reassociation(self):
        summarizer = CausalTemporalSummarizer(max_gap_s=1.0)
        summarizer.update(feature(0.0, 0.4))
        summarizer.update(feature(0.5, 0.8))
        row = summarizer.update(feature(3.0, 1.2)).values

        assert row["track_age_s"] == 0.0
        assert row["h_torso__delta_8s"] == 0.0
        assert row["h_torso__max_8s"] == 1.2

    def test_one_missed_frame_breaks_all_temporal_continuity(self):
        summarizer = CausalTemporalSummarizer(max_gap_s=2.0)
        first = feature(0.0, 0.5)
        first.bed_edge_distance_m = 0.5
        second = feature(0.1, 0.7)
        second.bed_edge_distance_m = 0.2
        before = summarizer.update(first)
        before = summarizer.update(second)
        assert before.values["cumulative_edge_progress_m"] == pytest.approx(0.3)

        summarizer.mark_unobserved(1, 0.2)
        returned = feature(0.3, 1.0)
        returned.bed_edge_distance_m = 0.1
        after = summarizer.update(returned)
        assert after.values["track_age_s"] == 0.0
        assert after.values["cumulative_edge_progress_m"] == 0.0
        assert after.values["dwell_near_edge_s"] == 0.0
        assert after.values["h_torso__delta_8s"] == 0.0
        assert after.as_vector() == [
            after.values[name] for name in TEMPORAL_FEATURE_NAMES
        ]

    def test_reset_retain_and_out_of_order_guard(self):
        summarizer = CausalTemporalSummarizer()
        summarizer.update(feature(1.0, 0.5, track_id=1))
        summarizer.update(feature(1.0, 0.6, track_id=2))
        assert summarizer.live_ids == {1, 2}

        summarizer.retain_only({2})
        assert summarizer.live_ids == {2}
        reset_row = summarizer.update(feature(2.0, 0.7, track_id=1)).values
        assert reset_row["track_age_s"] == 0.0

        with pytest.raises(ValueError, match="backwards"):
            summarizer.update(feature(1.5, 0.8, track_id=1))

        summarizer.reset()
        assert summarizer.live_ids == set()

    def test_edge_dwell_progress_and_cusum_context_are_causal(self):
        summarizer = CausalTemporalSummarizer()
        first = feature(0.0, 0.5)
        first.bed_edge_distance_m = 0.30
        second = feature(0.5, 0.7)
        second.bed_edge_distance_m = 0.18
        row = summarizer.update(first, context={"cusum_g": 0.0})
        assert row.values["dwell_near_edge_s"] is None
        row = summarizer.update(
            second,
            context={"cusum_g": 7.0, "cusum_onset": 1.0, "edge_velocity_mps": -0.24},
        )
        assert row.values["dwell_near_edge_s"] == 0.0
        assert row.values["cumulative_edge_progress_m"] == pytest.approx(0.12)
        assert row.values["cusum_g__now"] == 7.0


pytest.importorskip("sklearn")


MODEL_FEATURES = ("h_torso__now", "h_torso__missing_now")


def comparison_data(n_subjects: int = 3):
    rows = []
    labels = []
    metadata = []
    for subject_index in range(n_subjects):
        subject = f"volunteer_{subject_index + 1:02}"
        offset = subject_index * 0.02
        for session in ("morning", "afternoon"):
            for label, value in (("rest", 0.2 + offset), ("exit", 1.8 + offset)):
                rows.append(
                    {
                        "h_torso__now": value,
                        "h_torso__missing_now": 0.0,
                    }
                )
                labels.append(label)
                metadata.append(
                    SampleMetadata(
                        subject_id=subject,
                        session_id=session,
                        clip_id=f"{subject}_{session}_{label}",
                    )
                )
    return rows, labels, metadata


class TestGroupedComparator:
    def test_subject_folds_are_disjoint_and_preprocessing_is_fold_local(self):
        rows, labels, metadata = comparison_data()
        result = compare_grouped(
            rows,
            labels,
            metadata,
            group_by="subject",
            feature_names=MODEL_FEATURES,
            min_samples_leaf=1,
        )

        assert len(result.folds) == 3
        tested = set()
        for fold in result.folds:
            assert set(fold.train_groups).isdisjoint(fold.test_groups)
            assert len(fold.test_groups) == 1
            tested.update(fold.test_groups)
        assert tested == {item.subject_id for item in metadata}
        assert {score.model_kind for score in result.scores} == {"logistic", "tree"}
        assert all(score.accuracy == pytest.approx(1.0) for score in result.scores)
        for score in result.scores:
            assert len(score.probabilities) == len(rows)
            assert all(len(item) == len(result.classes) for item in score.probabilities)
            assert score.macro_average_precision == pytest.approx(1.0)
            assert 0.0 <= score.brier_score < 0.25
            assert score.positive_class == "exit"
            assert score.positive_precision == pytest.approx(1.0)
            assert score.positive_recall == pytest.approx(1.0)
            assert score.false_positive_rate == pytest.approx(0.0)

        # The returned models are all-data deployment fits, clearly described;
        # the scores above came only from fresh per-fold pipelines.
        model = result.models["logistic"]
        assert model.metadata.schema_version == TEMPORAL_SCHEMA_VERSION
        assert model.metadata.group_by == "subject"
        assert model.metadata.n_samples == len(rows)
        assert model.metadata.feature_names == MODEL_FEATURES

    def test_session_group_is_subject_qualified(self):
        rows, labels, metadata = comparison_data(n_subjects=2)
        result = compare_grouped(
            rows,
            labels,
            metadata,
            group_by="session",
            feature_names=MODEL_FEATURES,
            min_samples_leaf=1,
        )
        assert len(result.folds) == 4
        groups = {group for fold in result.folds for group in fold.test_groups}
        assert groups == {
            "volunteer_01::morning",
            "volunteer_01::afternoon",
            "volunteer_02::morning",
            "volunteer_02::afternoon",
        }

    def test_explicit_metadata_is_required(self):
        with pytest.raises(ValueError, match="subject_id"):
            SampleMetadata(subject_id="", session_id="visit_01")
        with pytest.raises(ValueError, match="session_id"):
            SampleMetadata(subject_id="volunteer_01", session_id="")

    def test_fold_with_only_one_training_class_is_rejected(self):
        rows = [
            {"h_torso__now": 0.1, "h_torso__missing_now": 0.0},
            {"h_torso__now": 0.2, "h_torso__missing_now": 0.0},
            {"h_torso__now": 1.8, "h_torso__missing_now": 0.0},
            {"h_torso__now": 1.9, "h_torso__missing_now": 0.0},
        ]
        labels = ["rest", "rest", "exit", "exit"]
        metadata = [
            SampleMetadata("a", "one"),
            SampleMetadata("a", "one"),
            SampleMetadata("b", "one"),
            SampleMetadata("b", "one"),
        ]
        with pytest.raises(ValueError, match="fewer than two classes"):
            compare_grouped(rows, labels, metadata, feature_names=MODEL_FEATURES)

    def test_both_models_produce_reviewable_explanations(self):
        rows, labels, metadata = comparison_data()
        result = compare_grouped(
            rows,
            labels,
            metadata,
            feature_names=MODEL_FEATURES,
            min_samples_leaf=1,
        )
        candidate = {"h_torso__now": 1.9, "h_torso__missing_now": 0.0}

        logistic = result.models["logistic"].explain(candidate, top_k=2)
        assert logistic.predicted_class == "exit"
        assert 0.5 < logistic.probability <= 1.0
        assert logistic.contributions
        assert logistic.contributions[0].feature in MODEL_FEATURES
        assert logistic.metadata is result.models["logistic"].metadata

        tree = result.models["tree"].explain(candidate)
        assert tree.predicted_class == "exit"
        assert tree.decision_path
        assert tree.decision_path[0].feature in MODEL_FEATURES
