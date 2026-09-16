"""Deriving fall ground-truth from posture segments."""

from __future__ import annotations

import json

import pytest

from ahfd.annotate import derive_annotations, derive_falls, dump_annotation_json
from ahfd.eval import GroundTruth


def _seg(start, end, posture):
    return {"start_s": start, "end_s": end, "posture": posture}


class TestDeriveFalls:
    def test_upright_to_ground_is_a_fall(self):
        segs = [_seg(0, 10, "upright"), _seg(12, 20, "on_ground")]
        falls = derive_falls(segs)
        assert len(falls) == 1
        assert falls[0]["t_impact"] == 12.0  # on_ground start
        assert falls[0]["t_start"] == 10.0  # previous posture end

    def test_sitting_to_ground_is_a_fall(self):
        falls = derive_falls([_seg(0, 5, "sitting"), _seg(6, 9, "on_ground")])
        assert len(falls) == 1
        assert falls[0]["t_impact"] == 6.0

    def test_repeated_slumps_are_multiple_falls(self):
        segs = [
            _seg(0, 5, "sitting"), _seg(6, 8, "on_ground"),
            _seg(9, 12, "sitting"), _seg(13, 15, "on_ground"),
        ]
        assert len(derive_falls(segs)) == 2

    def test_in_bed_to_ground_is_not_counted(self):
        # lying-to-floor is a different event; the elevated set excludes in_bed
        assert derive_falls([_seg(0, 5, "in_bed"), _seg(6, 9, "on_ground")]) == []

    def test_pure_negative_has_no_falls(self):
        assert derive_falls([_seg(0, 5, "upright"), _seg(6, 9, "sitting")]) == []

    def test_leading_on_ground_is_not_a_fall(self):
        # already down at the start -- no preceding elevated posture
        assert derive_falls([_seg(0, 5, "on_ground"), _seg(6, 9, "upright")]) == []

    def test_out_of_order_segments_are_sorted(self):
        segs = [_seg(12, 20, "on_ground"), _seg(0, 10, "upright")]
        assert derive_falls(segs)[0]["t_impact"] == 12.0


class TestDumpAndLoad:
    def test_output_parses_as_groundtruth(self):
        falls = derive_falls([_seg(0, 10, "upright"), _seg(12, 20, "on_ground")])
        text = dump_annotation_json("clipX", 30.0, falls)
        gt = GroundTruth.from_dict(json.loads(text))
        assert gt.clip_id == "clipX"
        assert gt.duration_s == 30.0
        assert len(gt.falls) == 1
        assert gt.falls[0].t_impact == 12.0

    def test_negative_output_is_valid_and_empty(self):
        gt = GroundTruth.from_dict(json.loads(dump_annotation_json("c", 20.0, [])))
        assert gt.is_negative


def _write_posture(pdir, stem, duration, segments):
    pdir.mkdir(exist_ok=True)
    (pdir / (stem + ".json")).write_text(
        json.dumps({"clip_id": stem, "duration_s": duration, "segments": segments}),
        encoding="utf-8",
    )


class TestDeriveDirectory:
    @pytest.mark.parametrize("stem", ["neg_adl_x", "neg_postures_x", "bedexit_safe_x"])
    @pytest.mark.parametrize("elevated", ["upright", "sitting"])
    def test_labelled_negative_floor_activity_stays_negative(self, tmp_path, stem, elevated):
        pdir, adir = tmp_path / "postures", tmp_path / "annotations"
        segments = [_seg(0, 5, elevated), _seg(8, 15, "on_ground")]
        _write_posture(pdir, stem, 30.0, segments)
        adir.mkdir()
        # Re-deriving also repairs an annotation produced by the old ordering.
        (adir / (stem + ".json")).write_text(
            dump_annotation_json(stem, 30.0, derive_falls(segments)), encoding="utf-8"
        )

        written, unlabelled, pruned = derive_annotations(pdir, adir)

        assert written == [(stem, 0)]
        assert unlabelled == []
        assert pruned == []
        truth = GroundTruth.load(adir / (stem + ".json"))
        assert truth.is_negative
        assert truth.duration_s == 30.0
        # Keep the manual posture labels intact for classifier training.
        assert json.loads((pdir / (stem + ".json")).read_text())["segments"] == segments

    def test_labelled_fall_negative_and_unlabelled(self, tmp_path):
        pdir, adir = tmp_path / "postures", tmp_path / "annotations"
        _write_posture(pdir, "fall_x", 40.0, [_seg(0, 10, "upright"), _seg(12, 20, "on_ground")])
        # a negative clip with NO posture labels -> still a valid negative
        _write_posture(pdir, "neg_adl_x", 30.0, [_seg(0, 0, "upright")])
        # an unlabelled fall clip -> left out of the ground truth
        _write_posture(pdir, "fall_y", 25.0, [_seg(0, 0, "upright")])

        written, unlabelled, pruned = derive_annotations(pdir, adir)

        assert ("fall_x", 1) in written
        assert ("neg_adl_x", 0) in written  # negative written from duration alone
        assert unlabelled == ["fall_y"]
        assert pruned == []
        assert not (adir / "fall_y.json").exists()
        assert GroundTruth.load(adir / "neg_adl_x.json").is_negative

    def test_prune_removes_stale_placeholder(self, tmp_path):
        pdir, adir = tmp_path / "postures", tmp_path / "annotations"
        adir.mkdir()
        _write_posture(pdir, "fall_z", 25.0, [_seg(0, 0, "upright")])  # unlabelled
        # a stale placeholder annotation from the old generator
        (adir / "fall_z.json").write_text(
            json.dumps({"clip_id": "fall_z", "duration_s": 25.0, "falls": [{"t_impact": 0.0}]}),
            encoding="utf-8",
        )
        written, unlabelled, pruned = derive_annotations(pdir, adir, prune_placeholders=True)
        assert unlabelled == ["fall_z"]
        assert pruned == ["fall_z"]
        assert not (adir / "fall_z.json").exists()
