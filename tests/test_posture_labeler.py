"""Pure logic of the posture labeller -- no display, no OpenCV needed."""

from __future__ import annotations

import json

from ahfd.annotate import dump_posture_json, load_existing_segments


class TestLoadExisting:
    def test_missing_file(self, tmp_path):
        clip_id, segs = load_existing_segments(tmp_path / "nope.json")
        assert clip_id is None
        assert segs == []

    def test_drops_placeholders_and_bad_classes(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text(
            json.dumps(
                {
                    "clip_id": "c",
                    "duration_s": 30.0,
                    "segments": [
                        {"start_s": 0.0, "end_s": 0.0, "posture": "upright"},  # placeholder
                        {"start_s": 2.0, "end_s": 9.0, "posture": "sitting"},  # real
                        {"start_s": 10.0, "end_s": 12.0, "posture": "crouching"},  # not a class
                    ],
                }
            ),
            encoding="utf-8",
        )
        clip_id, segs = load_existing_segments(f)
        assert clip_id == "c"
        assert segs == [{"start_s": 2.0, "end_s": 9.0, "posture": "sitting"}]

    def test_corrupt_json_is_tolerated(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{ this is not json", encoding="utf-8")
        assert load_existing_segments(f) == (None, [])


class TestDump:
    def test_sorts_and_is_valid_json(self):
        segs = [
            {"start_s": 20.0, "end_s": 25.0, "posture": "on_ground"},
            {"start_s": 5.0, "end_s": 9.0, "posture": "sitting"},
        ]
        text = dump_posture_json("clip9", 30.0, segs)
        data = json.loads(text)
        assert data["clip_id"] == "clip9"
        assert data["duration_s"] == 30.0
        # sorted by start time
        assert [s["start_s"] for s in data["segments"]] == [5.0, 20.0]
        assert data["segments"][0]["posture"] == "sitting"

    def test_empty_segments_is_valid(self):
        data = json.loads(dump_posture_json("c", 12.0, []))
        assert data["segments"] == []

    def test_one_segment_per_line(self):
        text = dump_posture_json("c", 12.0, [{"start_s": 1.0, "end_s": 2.0, "posture": "upright"}])
        seg_lines = [ln for ln in text.splitlines() if "start_s" in ln]
        assert len(seg_lines) == 1
