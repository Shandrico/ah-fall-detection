"""Bed-exit pilot phase labels -> posture training rows, with the holdout locked."""

from __future__ import annotations

from pathlib import Path

import pytest

from ahfd.ml.posture import (
    PHASE_TO_POSTURE,
    build_pilot_posture_dataset,
    pilot_locked_clips,
)

REPO = Path(__file__).resolve().parents[1]


def test_locked_clips_are_block_e():
    assert pilot_locked_clips() == {"clip1_09", "clip1_10"}


def test_transition_phases_are_not_postures():
    for phase in ("TORSO_RISING", "SHIFTING_TO_EDGE", "ATTEMPTING_STAND"):
        assert phase not in PHASE_TO_POSTURE
    assert set(PHASE_TO_POSTURE.values()) == {"in_bed", "sitting", "upright"}


@pytest.mark.skipif(
    not (REPO / "data/phases_d435f").is_dir() or not (REPO / "data/tracks_d435i").is_dir(),
    reason="pilot labels/tracks not in this checkout",
)
def test_pilot_dataset_never_reads_locked_takes(monkeypatch):
    from ahfd.ml import posture

    read = []
    real = posture.build_dataset

    def spy(labels_dir, tracks_dir, calib, **kw):
        out = real(labels_dir, tracks_dir, calib, **kw)
        read.extend(clip for clip, _n in out[3])
        return out

    monkeypatch.setattr(posture, "build_dataset", spy)
    rows, labels, groups, used, _skipped = build_pilot_posture_dataset(
        REPO / "data", REPO / "calib"
    )
    assert rows and len(rows) == len(labels) == len(groups)
    assert not {"clip1_09", "clip1_10"} & set(read)
    assert set(groups) <= {"block_A", "block_B", "block_C", "block_D"}
    assert set(labels) <= {"in_bed", "sitting", "upright"}
    cams = {clip.split("/")[0] for clip, _n in used}
    assert cams == {"d435i", "d435f"}
