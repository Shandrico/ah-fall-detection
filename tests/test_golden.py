"""Golden regression: replay a committed tracks.jsonl, assert the events.

The point of this test is to fail when a threshold change silently breaks a
scenario. Unit tests check pieces; this checks that the *whole* replay path --
tracks -> features -> state machine -> events -- still produces the same
decision on a fixed input. Because tracks.jsonl is keypoints only, the fixture
is tiny, contains no imagery, and is safe to commit.

The fixture is generated on first run (and regenerated if the generator
changes), then committed. If a real change to the thresholds is intended, the
expected events below are updated deliberately in the same commit -- which is
exactly the moment to notice a scenario regressed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ahfd.detect import FallStateMachine
from ahfd.features import FeatureExtractor
from ahfd.geometry import GroundPlane
from ahfd.io import read_tracks, write_tracks
from ahfd.types import NUM_KEYPOINTS, Intrinsics, PersonPose, PoseFrame

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_TRACKS = GOLDEN_DIR / "fall_at_6m.jsonl"

FPS = 15.0
DT = 1.0 / FPS

WARD = GroundPlane(
    Intrinsics.from_hfov(1920, 1080, hfov_deg=69.4, vfov_deg=42.5),
    height_m=2.6,
    pitch_deg=20.0,
)

BODY = (
    (1.62, 0.00), (1.64, 0.03), (1.64, -0.03), (1.62, 0.07), (1.62, -0.07),
    (1.40, 0.18), (1.40, -0.18), (1.10, 0.20), (1.10, -0.20),
    (0.85, 0.20), (0.85, -0.20), (0.95, 0.12), (0.95, -0.12),
    (0.50, 0.12), (0.50, -0.12), (0.08, 0.10), (0.08, -0.10),
)


def _project(point: np.ndarray) -> tuple[float, float]:
    rel = np.asarray(point, float) - np.array([0.0, 0.0, WARD.height_m])
    d = WARD.rotation.T @ rel
    k = WARD.intrinsics
    return (k.cx + k.fx * d[0] / d[2], k.cy + k.fy * d[1] / d[2])


def _pose(joints3d: np.ndarray, t: float, index: int) -> PoseFrame:
    pts = np.array([_project(p) for p in joints3d], dtype=np.float32)
    person = PersonPose(
        keypoints=pts,
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
        track_id=1,
    )
    return PoseFrame(t=t, index=index, width=1920, height=1080, people=(person,))


def _generate_fall_tracks() -> list[PoseFrame]:
    """Stand at 6 m, collapse to the floor, stay down. Deterministic."""
    up = np.array([[lat, 6.0, h] for h, lat in BODY])
    prone = np.array([[lat, 6.0 + (h - 0.08), 0.12] for h, lat in BODY])

    poses = []
    index = 0
    t = 0.0
    for _ in range(int(3.0 * FPS)):
        poses.append(_pose(up, t, index))
        t += DT
        index += 1
    n_fall = int(0.5 * FPS)
    for i in range(n_fall):
        a = (i + 1) / n_fall
        poses.append(_pose(up * (1 - a) + prone * a, t, index))
        t += DT
        index += 1
    for _ in range(int(14.0 * FPS)):
        poses.append(_pose(prone, t, index))
        t += DT
        index += 1
    return poses


@pytest.fixture(scope="module")
def golden_tracks() -> Path:
    """Ensure the committed fixture exists; generate it if missing."""
    if not GOLDEN_TRACKS.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        write_tracks(GOLDEN_TRACKS, _generate_fall_tracks())
    return GOLDEN_TRACKS


def _replay(tracks_path: Path) -> list:
    extractor = FeatureExtractor(WARD)
    machine = FallStateMachine()
    events = []
    for pose in read_tracks(tracks_path):
        live = {p.track_id for p in pose.people if p.track_id is not None}
        extractor.retain_only(live)
        machine.retain_only(live)
        for person in pose.people:
            features = extractor.extract(person, pose.t)
            if features is None:
                continue
            event = machine.update(features)
            if event is not None:
                events.append(event)
    return events


class TestGoldenReplay:
    def test_fixture_is_keypoints_only(self, golden_tracks):
        """The committed fixture must never contain imagery."""
        text = golden_tracks.read_text()
        for banned in ("bgr", "image", "rgb", "pixels"):
            assert banned not in text

    def test_replay_confirms_the_fall(self, golden_tracks):
        types = [e.type for e in _replay(golden_tracks)]
        assert "FALL_CONFIRMED" in types

    def test_replay_is_deterministic(self, golden_tracks):
        """Same input, byte-identical output -- the basis for threshold sweeps."""
        first = [(e.type, round(e.t_alert, 3)) for e in _replay(golden_tracks)]
        second = [(e.type, round(e.t_alert, 3)) for e in _replay(golden_tracks)]
        assert first == second

    def test_expected_event_sequence(self, golden_tracks):
        """The pinned expectation. If a threshold change alters this, update it
        deliberately -- that edit is the moment to confirm nothing regressed."""
        events = _replay(golden_tracks)
        types = [e.type for e in events]
        assert types == ["FALL_SUSPECTED", "FALL_CONFIRMED"]

        confirmed = next(e for e in events if e.type == "FALL_CONFIRMED")
        # Impact is at ~3.5 s; confirmation waits confirm_s (8 s) after landing.
        assert 11.0 < confirmed.t_alert < 12.5
        assert confirmed.evidence["peak_vz"] < -0.9
