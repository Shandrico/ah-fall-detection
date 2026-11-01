"""Run a trained posture model live, next to the rule-based states.

Display only: the label it produces is shown on the dashboard beside the rule
state so the two can be compared on a live camera. Alerts never read it.

The model file is what `~/.ahfd-work/save_models.py` / `ahfd train-posture`
save: either a bare scikit-learn pipeline (fed ``posture.FEATURES``) or a dict
``{"model": pipeline, "features": [...], ...}``. Features are built with the
same ``posture.features_row`` the model was trained on, so live numbers match
training numbers exactly.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, deque
from pathlib import Path

from ahfd.ml.posture import FEATURES, features_row


class LivePostureModel:
    """Per-track posture predictions with a short majority vote against flicker."""

    def __init__(self, path: str | Path, *, min_keypoint_score: float = 0.3, vote_frames: int = 9):
        import joblib

        if vote_frames < 1:
            raise ValueError("vote_frames must be at least 1")
        with warnings.catch_warnings():
            # Models pickled by a newer scikit-learn still load; the pilot
            # trees were checked to predict the same on this version.
            warnings.simplefilter("ignore")
            loaded = joblib.load(path)
        if isinstance(loaded, dict):
            self.model = loaded["model"]
            self.features = list(loaded.get("features") or FEATURES)
            self.trained_on = str(loaded.get("trained_on", ""))
        else:
            self.model = loaded
            self.features = list(FEATURES)
            self.trained_on = ""
        unknown = [name for name in self.features if name not in FEATURES]
        if unknown:
            raise ValueError("model expects features this build does not compute: " + ", ".join(unknown))
        self.path = Path(path)
        self.min_keypoint_score = min_keypoint_score
        self.vote_frames = vote_frames
        self._recent: dict[int, deque[str]] = {}

    def predict(self, track_id: int, features, person, ground) -> str:
        """The voted label for this track after adding this frame's prediction."""
        import numpy as np

        row = features_row(features, person, ground, self.min_keypoint_score)
        x = np.array(
            [[math.nan if row.get(name) is None else float(row[name]) for name in self.features]],
            dtype=float,
        )
        label = str(self.model.predict(x)[0])
        recent = self._recent.setdefault(track_id, deque(maxlen=self.vote_frames))
        recent.append(label)
        # Most common recent label; ties go to the newest of the tied labels.
        counts = Counter(recent)
        top = max(counts.values())
        return next(lbl for lbl in reversed(recent) if counts[lbl] == top)

    def retain_only(self, live_ids) -> None:
        live = set(live_ids)
        for track_id in list(self._recent):
            if track_id not in live:
                del self._recent[track_id]
