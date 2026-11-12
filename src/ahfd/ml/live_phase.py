"""Run the trained 7-phase bed-activity model live, next to the rule phase.

Display only, like ``live_posture``: the dashboard shows it beside the rule
engine's phase and alerts never read it. Features are built exactly as in
training (``ml.clip_review.predict_clip`` / ``phase_compare``): the per-frame
posture row plus causal over-time summaries fed by the bed engine's snapshot.
The model is view-specific (``phase_v3_side`` / ``phase_v3_along``), so each
camera in the picker names its own.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, deque
from pathlib import Path

from ahfd.ml.posture import features_row


CALM_PHASES = ("RECLINED", "UPRIGHT_IN_BED", "OUT_OF_BED")


class ConservativePhaseClassifier:
    """Names a getting-up phase only when the model is sure of it.

    Wraps a fitted scikit-learn classifier. When the most likely phase is one
    of the movement phases (torso rising, shifting, edge sitting, attempting to
    stand) but its probability is below ``threshold``, the likeliest calm phase
    (reclined, upright in bed, out of bed) is reported instead. This trades a
    little recall on short transitions for far fewer false "getting up" calls,
    which is what a ward screen needs. Pickled inside the phase-model file, so
    it must stay importable from this module.
    """

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = float(threshold)

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        import numpy as np

        proba = self.model.predict_proba(X)
        classes = [str(c) for c in self.model.classes_]
        calm = [i for i, c in enumerate(classes) if c in CALM_PHASES]
        out = []
        for p in proba:
            k = int(np.argmax(p))
            if classes[k] not in CALM_PHASES and p[k] < self.threshold and calm:
                k = max(calm, key=lambda i: p[i])
            out.append(classes[k])
        return np.array(out)


class LivePhaseModel:
    """Per-track 7-phase predictions with a short majority vote against flicker."""

    def __init__(self, path: str | Path, *, max_gap_s: float, near_edge_m: float,
                 min_keypoint_score: float = 0.3, vote_frames: int = 9):
        import joblib

        from ahfd.ml.temporal import CausalTemporalSummarizer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = joblib.load(path)
        if not isinstance(loaded, dict) or "model" not in loaded or "features" not in loaded:
            raise ValueError("phase model must be a dict with 'model' and 'features'")
        self.model = loaded["model"]
        self.features = list(loaded["features"])
        self.path = Path(path)
        self.min_keypoint_score = min_keypoint_score
        self.temporal = CausalTemporalSummarizer(max_gap_s=max_gap_s, near_edge_m=near_edge_m)
        self.vote_frames = vote_frames
        self._recent: dict[int, deque[str]] = {}

    def predict(self, track_id: int, features, person, ground, snap) -> str | None:
        """The voted phase for this track, or None while it has no geometry."""
        import numpy as np

        from ahfd.cli import _depth_provenance

        summary = self.temporal.update(features, valid=features.has_geometry(), context={
            "edge_velocity_mps": snap.edge_velocity_mps if snap else None,
            "cusum_z": snap.cusum_z if snap else None,
            "cusum_g": snap.cusum_g if snap else None,
            "cusum_onset": 1.0 if snap and "cusum_onset" in snap.reasons else 0.0,
            "cusum_armed": 1.0 if snap and snap.cusum_armed else 0.0,
            **_depth_provenance(person, min_score=self.min_keypoint_score),
            "imu_pitch_delta_deg": None,
            "imu_roll_delta_deg": None,
            "episode_reclined": 1.0 if snap and snap.phase == "RECLINED" else 0.0,
        })
        if not features.has_geometry():
            return None
        row = {**features_row(features, person, ground, self.min_keypoint_score), **summary.values}
        x = np.array([[math.nan if row.get(n) is None else float(row[n]) for n in self.features]], dtype=float)
        label = str(self.model.predict(x)[0])
        recent = self._recent.setdefault(track_id, deque(maxlen=self.vote_frames))
        recent.append(label)
        counts = Counter(recent)
        top = max(counts.values())
        return next(lbl for lbl in reversed(recent) if counts[lbl] == top)

    def mark_unobserved(self, track_id: int, t: float) -> None:
        self.temporal.mark_unobserved(track_id, t)

    def retain_only(self, live_ids) -> None:
        live = set(live_ids)
        self.temporal.retain_only(live)
        for track_id in list(self._recent):
            if track_id not in live:
                del self._recent[track_id]
