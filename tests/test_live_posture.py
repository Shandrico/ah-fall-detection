"""The dashboard's display-only trained posture model (ml/live_posture.py)."""

import joblib
import pytest

from ahfd.ml import live_posture
from ahfd.ml.live_posture import LivePostureModel
from ahfd.ml.posture import FEATURES


class _Scripted:
    """Stand-in model: returns queued labels and records the rows it saw."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.seen = []

    def predict(self, x):
        self.seen.append(x)
        return [self.labels.pop(0)]


def _save(tmp_path, payload):
    path = tmp_path / "m.joblib"
    joblib.dump(payload, path)
    return path


@pytest.fixture
def fixed_row(monkeypatch):
    row = {name: None for name in FEATURES}
    row["h_torso"] = 1.2
    monkeypatch.setattr(live_posture, "features_row", lambda *a, **k: dict(row))
    return row


def test_majority_vote_resists_single_frame_flicker(tmp_path, fixed_row):
    labels = ["sitting", "sitting", "upright", "sitting", "upright", "upright", "upright"]
    model = LivePostureModel(_save(tmp_path, {"model": _Scripted(labels), "features": ["h_torso"]}),
                             vote_frames=5)
    out = [model.predict(1, None, None, None) for _ in labels]
    assert out == ["sitting", "sitting", "sitting", "sitting", "sitting", "upright", "upright"]


def test_feeds_only_the_saved_feature_columns_with_nan_for_missing(tmp_path, fixed_row):
    model = LivePostureModel(_save(tmp_path, {"model": _Scripted(["in_bed"]), "features": ["h_torso", "h_min"]}))
    assert model.predict(3, None, None, None) == "in_bed"
    x = model.model.seen[0]  # the unpickled copy
    assert x.shape == (1, 2) and x[0, 0] == 1.2 and x[0, 1] != x[0, 1]  # NaN


def test_tracks_vote_independently_and_dropped_tracks_are_forgotten(tmp_path, fixed_row):
    model = LivePostureModel(
        _save(tmp_path, {"model": _Scripted(["sitting", "upright", "upright"]), "features": ["h_torso"]})
    )
    assert model.predict(1, None, None, None) == "sitting"
    assert model.predict(2, None, None, None) == "upright"
    model.retain_only({2})
    assert model.predict(1, None, None, None) == "upright"  # fresh history for track 1


def test_bare_pipeline_uses_all_features_and_unknown_features_are_rejected(tmp_path):
    bare = LivePostureModel(_save(tmp_path, _Scripted([])))
    assert bare.features == list(FEATURES)
    with pytest.raises(ValueError, match="not_a_feature"):
        LivePostureModel(_save(tmp_path, {"model": _Scripted([]), "features": ["not_a_feature"]}))


def test_dashboard_runner_survives_a_missing_model(tmp_path):
    from ahfd.config import load_config
    from ahfd.dashboard.runner import PipelineRunner
    from ahfd.dashboard.state import DashboardState

    cfg = load_config(None)
    cfg.dashboard.posture_model = str(tmp_path / "missing.joblib")
    state = DashboardState()
    runner = PipelineRunner("seq://x", cfg, None, state, False)
    runner._load_posture_model()
    assert runner._posture_model is None
