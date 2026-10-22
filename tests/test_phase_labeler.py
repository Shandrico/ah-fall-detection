"""Pure logic of the bed-exit phase labeller and script seeding (no display)."""

from __future__ import annotations

import json

from ahfd.annotate import (
    PHASE_CLASSES,
    PHASE_SCHEME,
    PILOT_PHASE_SCRIPT,
    dump_posture_json,
    load_existing_segments,
    seed_phase_drafts,
    seed_segments,
)
from ahfd.ml.posture import _posture_at


class TestPhaseScheme:
    def test_load_reads_phase_field_and_filters_bad_classes(self, tmp_path):
        f = tmp_path / "clip1_03.json"
        f.write_text(
            json.dumps(
                {
                    "clip_id": "clip1_03",
                    "duration_s": 60.0,
                    "segments": [
                        {"start_s": 0.0, "end_s": 0.0, "phase": "RECLINED"},  # placeholder
                        {"start_s": 2.0, "end_s": 9.0, "phase": "EDGE_SITTING"},  # real
                        {"start_s": 10.0, "end_s": 12.0, "phase": "NAPPING"},  # not a phase
                        {"start_s": 13.0, "end_s": 15.0, "posture": "sitting"},  # wrong field
                    ],
                }
            ),
            encoding="utf-8",
        )
        clip_id, segs = load_existing_segments(f, scheme=PHASE_SCHEME)
        assert clip_id == "clip1_03"
        assert segs == [{"start_s": 2.0, "end_s": 9.0, "phase": "EDGE_SITTING"}]

    def test_posture_file_is_empty_under_phase_scheme(self, tmp_path):
        """A 4-class posture file holds no phase segments (fields differ)."""
        f = tmp_path / "p.json"
        f.write_text(
            json.dumps(
                {"clip_id": "p", "duration_s": 30.0,
                 "segments": [{"start_s": 1.0, "end_s": 5.0, "posture": "sitting"}]}
            ),
            encoding="utf-8",
        )
        _cid, segs = load_existing_segments(f, scheme=PHASE_SCHEME)
        assert segs == []

    def test_dump_round_trips_under_phase_key(self, tmp_path):
        segs = [
            {"start_s": 10.0, "end_s": 20.0, "phase": "UPRIGHT_IN_BED"},
            {"start_s": 0.0, "end_s": 10.0, "phase": "RECLINED"},
        ]
        text = dump_posture_json("clip1_04", 20.0, segs, label_key="phase")
        f = tmp_path / "clip1_04.json"
        f.write_text(text, encoding="utf-8")
        _cid, back = load_existing_segments(f, scheme=PHASE_SCHEME)
        assert [s["phase"] for s in back] == ["RECLINED", "UPRIGHT_IN_BED"]  # sorted by start


class TestPhaseAt:
    def test_reads_phase_label_key(self):
        segs = [
            {"start_s": 0.0, "end_s": 15.0, "phase": "RECLINED"},
            {"start_s": 25.0, "end_s": 35.0, "phase": "EDGE_SITTING"},
        ]
        assert _posture_at(segs, 5.0, "phase") == "RECLINED"
        assert _posture_at(segs, 30.0, "phase") == "EDGE_SITTING"
        assert _posture_at(segs, 20.0, "phase") is None  # gap/transition


class TestScriptSeed:
    def test_every_scripted_phase_is_a_known_class(self):
        for idx in PILOT_PHASE_SCRIPT:
            for seg in seed_segments(idx):
                assert seg["phase"] in PHASE_CLASSES
                assert seg["end_s"] > seg["start_s"]

    def test_seed_drafts_writes_ten_then_skips_existing(self, tmp_path):
        written = seed_phase_drafts(tmp_path)
        assert len(written) == 10
        assert (tmp_path / "clip1_01.json").exists()
        # a draft is valid, loadable phase JSON
        _cid, segs = load_existing_segments(tmp_path / "clip1_03.json", scheme=PHASE_SCHEME)
        assert segs and segs[0]["phase"] == "RECLINED"
        # re-running does not clobber (returns nothing new)
        assert seed_phase_drafts(tmp_path) == []

    def test_overwrite_replaces(self, tmp_path):
        (tmp_path / "clip1_01.json").write_text("{}", encoding="utf-8")
        again = seed_phase_drafts(tmp_path, overwrite=True)
        assert len(again) == 10
        _cid, segs = load_existing_segments(tmp_path / "clip1_01.json", scheme=PHASE_SCHEME)
        assert segs  # the stub got replaced by a real draft
