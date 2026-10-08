"""Regression coverage for observation gaps in the ordinary live runner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from ahfd.capture import Frame, SourceMeta
from ahfd.cli import app
from ahfd.config import Config
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame


def test_run_marks_a_live_track_unobserved_after_a_missed_pose(monkeypatch):
    """A tracker-held identity must not look observed during a pose dropout."""

    class Source:
        meta = SourceMeta(uri="test://camera", width=64, height=48, fps=10.0)

        def __init__(self):
            self.closed = False

        def __iter__(self):
            for index, timestamp in enumerate((0.0, 0.1)):
                yield Frame(
                    index=index,
                    t=timestamp,
                    bgr=np.zeros((48, 64, 3), dtype=np.uint8),
                )

        def close(self):
            self.closed = True

    keypoints = np.column_stack(
        (
            np.linspace(10.0, 30.0, NUM_KEYPOINTS),
            np.linspace(5.0, 40.0, NUM_KEYPOINTS),
        )
    ).astype(np.float32)
    person = PersonPose(
        keypoints=keypoints,
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
    )

    class Estimator:
        name = "test-estimator"

        def estimate(self, frame):
            people = (person,) if frame.index == 0 else ()
            return PoseFrame(
                t=frame.t,
                index=frame.index,
                width=64,
                height=48,
                people=people,
            )

    feature = SimpleNamespace(
        h_torso=1.0,
        floor_spread=0.2,
        v_z=0.0,
        h_ankle_min=0.05,
        bed_risk="unknown",
        range_m=2.0,
    )

    class Extractor:
        def retain_only(self, _live_ids):
            pass

        def extract(self, _person, _timestamp):
            return feature

    class Machine:
        def __init__(self):
            self.unobserved_calls = []

        def retain_only(self, _live_ids):
            pass

        def state_of(self, _track_id):
            return "UPRIGHT"

        def update_all(self, _features):
            return ()

        def mark_frame_unobserved(self, timestamp, live_ids, observed_ids):
            self.unobserved_calls.append(
                (timestamp, set(live_ids), set(observed_ids))
            )

    class Sink:
        def emit(self, _event):
            pass

        def close(self):
            pass

    source = Source()
    machine = Machine()
    cfg = Config()
    cfg.detect.enabled = True
    cfg.smoothing.enabled = False
    cfg.view.mode = "none"
    cfg.calibration = "unused-test-calibration.yaml"

    import ahfd.capture
    import ahfd.cli
    import ahfd.pose

    monkeypatch.setattr(ahfd.cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr(ahfd.capture, "open_source", lambda _uri: source)
    monkeypatch.setattr(ahfd.pose, "build_estimator", lambda _cfg: Estimator())
    monkeypatch.setattr(
        ahfd.cli,
        "_build_detection",
        lambda _cfg, _path, _meta: (
            Extractor(),
            machine,
            Sink(),
            SimpleNamespace(
                camera_id="cam_test",
                height_m=2.5,
                ground=SimpleNamespace(pitch_deg=15.0),
                zones=SimpleNamespace(zones=()),
            ),
        ),
    )

    result = CliRunner().invoke(
        app,
        ["run", "--source", "test://camera", "--view", "none", "--max-frames", "2"],
    )

    assert result.exit_code == 0, result.output
    assert source.closed
    assert machine.unobserved_calls == [
        (0.0, {0}, {0}),
        (0.1, {0}, set()),
    ]
