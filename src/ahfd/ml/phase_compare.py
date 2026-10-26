"""Bed-exit phase classifiers on the pilot: per-frame vs movement-over-time features.

Rebuilds, from the repository's phase labels and tracks, the comparison that
decides whether causal temporal summaries (``ml.temporal``) help recognise the
seven bed-exit phases beyond single-frame posture features. Each camera is
scored separately with leave-one-block-out folds (A-D: both takes of a block
held out together). Block E (takes 09/10) is never read.

Rows are taken from the person on the take's scripted bed, on the D435i's clock
for that take (the labels were marked there), at ``every``-th frame. The live
bed engine runs over every frame so its CUSUM/edge context matches what the
dashboard would have seen at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PHASES = (
    "RECLINED", "TORSO_RISING", "UPRIGHT_IN_BED", "SHIFTING_TO_EDGE",
    "EDGE_SITTING", "ATTEMPTING_STAND", "OUT_OF_BED",
)


@dataclass
class PhaseRows:
    frame: np.ndarray      # per-frame posture features (ml.posture.FEATURES)
    temporal: np.ndarray   # causal summaries (ml.temporal.MODEL_FEATURE_NAMES)
    labels: np.ndarray
    groups: np.ndarray     # "block_A" .. "block_D"


def dev_takes():
    """(stem, script entry) for every pilot take outside the locked block E."""
    from ahfd.annotate import PILOT_PHASE_SCRIPT

    return [
        ("clip1_%02d" % i, entry)
        for i, entry in sorted(PILOT_PHASE_SCRIPT.items())
        if entry["block"] != "E"
    ]


def build_rows(
    cam: str, cfg, *, data_dir=Path("data"), calib_dir=Path("calib"), every: int = 2, takes=None,
) -> PhaseRows:
    """Feature rows for one camera's dev takes (see the module docstring).

    ``takes`` overrides the (stem, script entry) list -- only for the one-off
    locked evaluation of block E once everything is frozen.
    """
    from ahfd.capture.base import SourceMeta
    from ahfd.cli import _build_detection, _depth_provenance
    from ahfd.io import read_tracks
    from ahfd.ml import posture as P
    from ahfd.ml.temporal import MODEL_FEATURE_NAMES, CausalTemporalSummarizer

    data_dir, calib_dir = Path(data_dir), Path(calib_dir)
    meta = SourceMeta(uri="tracks", width=1920, height=1080, fps=30.0)
    xf, xt, ys, gs = [], [], [], []
    for stem, entry in (dev_takes() if takes is None else takes):
        tracks = data_dir / ("tracks_" + cam) / (stem + ".jsonl")
        labels = data_dir / ("phases_" + cam) / (stem + ".json")
        if not tracks.exists() or not labels.exists():
            continue
        extractor, machine, _sink, calib = _build_detection(cfg, str(calib_dir / (cam + ".yaml")), meta)
        temporal = CausalTemporalSummarizer(
            max_gap_s=cfg.bed_activity.cusum.max_gap_s, near_edge_m=cfg.bed_activity.near_edge_enter_m,
        )
        segs = P._labelled_segments(labels)
        zone = P._resolve_bed_zone(calib, entry["bed"])
        target = P._dominant_bed_track(tracks, calib.ground, zone, 0.3)
        t0 = P._first_track_time(data_dir / "tracks_d435i" / (stem + ".jsonl"))
        for k, pose in enumerate(read_tracks(tracks)):
            ids = {p.track_id for p in pose.people if p.track_id is not None}
            extractor.retain_only(ids)
            machine.retain_only(ids)
            hit = None
            for person in pose.people:
                if person.track_id is None:
                    continue
                f = extractor.extract(person, pose.t)
                if f is None:
                    continue
                machine.update_all(f)
                if person.track_id == target:
                    hit = (person, f)
            if hit is None:
                if target in temporal.live_ids:
                    temporal.mark_unobserved(target, pose.t)
                continue
            person, f = hit
            snap = machine.bed_snapshot_of(target, pose.t)
            summary = temporal.update(f, valid=f.has_geometry(), context={
                "edge_velocity_mps": snap.edge_velocity_mps if snap else None,
                "cusum_z": snap.cusum_z if snap else None,
                "cusum_g": snap.cusum_g if snap else None,
                "cusum_onset": 1.0 if snap and "cusum_onset" in snap.reasons else 0.0,
                "cusum_armed": 1.0 if snap and snap.cusum_armed else 0.0,
                **_depth_provenance(person, min_score=cfg.pose.min_keypoint_score),
                "imu_pitch_delta_deg": None,
                "imu_roll_delta_deg": None,
                "episode_reclined": 1.0 if snap and snap.phase == "RECLINED" else 0.0,
            })
            if k % every:
                continue
            phase = P._posture_at(segs, pose.t - t0, "phase")
            if phase is None or not f.has_geometry():
                continue
            row = P.features_row(f, person, extractor.ground, cfg.pose.min_keypoint_score)
            xf.append([np.nan if row.get(n) is None else row[n] for n in P.FEATURES])
            xt.append([np.nan if summary.values.get(n) is None else summary.values[n] for n in MODEL_FEATURE_NAMES])
            ys.append(phase)
            gs.append("block_" + entry["block"])
    return PhaseRows(np.array(xf, float), np.array(xt, float), np.array(ys), np.array(gs))


def _models():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "tree": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=0)),
        ]),
        "logistic": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, class_weight="balanced", max_iter=400)),
        ]),
    }


def lobo_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make) -> dict:
    """Leave-one-group-out predictions -> accuracy, macro-F1 and per-phase F1."""
    import warnings

    from sklearn.metrics import accuracy_score, f1_score

    pred = np.empty_like(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lbfgs convergence chatter on small folds
        for g in sorted(set(groups)):
            test = groups == g
            pred[test] = make().fit(X[~test], y[~test]).predict(X[test])
    present = [p for p in PHASES if (y == p).any()]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
        "per_phase": {p: float(f1_score(y == p, pred == p)) for p in present},
        "frames": {p: int((y == p).sum()) for p in present},
    }


def compare(rows: PhaseRows) -> dict[str, dict]:
    """Score per-frame, over-time and combined features with each small model."""
    models = _models()
    sets = {
        "per-frame": rows.frame,
        "over-time": rows.temporal,
        "both": np.hstack([rows.frame, rows.temporal]),
    }
    out = {}
    for set_name, X in sets.items():
        for model_name, make in models.items():
            out[set_name + " / " + model_name] = lobo_scores(X, rows.labels, rows.groups, make)
    return out
