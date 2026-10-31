"""Per-frame rule and model predictions for one clip, next to its labels.

Keypoints only (a stored ``tracks.jsonl``), so it is privacy-safe and fast. It
reproduces the evaluation path of ``phase_compare.build_rows`` frame by frame:
the live rule engines (4-state posture from the fall machine, 7-phase bed
engine) and the trained models (posture tree, phase logistic on per-frame +
over-time features). ``debug.annotated_review`` draws the result on the video;
the same rows score a new test set.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Fall-machine states as 4-state posture labels (FALLING is on its way down).
RULE_POSTURE = {
    "UPRIGHT": "upright",
    "SITTING": "sitting",
    "IN_BED": "in_bed",
    "ON_GROUND": "on_ground",
    "FALLING": "on_ground",
}


@dataclass
class FramePrediction:
    """One frame of the clip's subject: what each source says."""

    t: float  # seconds since the clip's first frame
    track_id: int | None
    rule_posture: str | None = None
    rule_phase: str | None = None
    model_posture: str | None = None
    model_phase: str | None = None
    label_posture: str | None = None
    label_phase: str | None = None


@dataclass
class ClipPredictions:
    t0: float  # absolute time of the clip's first frame
    frames: dict[float, FramePrediction] = field(default_factory=dict)  # keyed by rounded absolute t
    subject: int | None = None

    def at(self, t_abs: float) -> FramePrediction | None:
        return self.frames.get(round(t_abs, 3))

    def agreement(self) -> dict[str, tuple[int, int]]:
        """(agreeing, labelled) frames for each prediction against the labels."""
        out = {}
        for name, label in (
            ("rule_posture", "label_posture"), ("model_posture", "label_posture"),
            ("rule_phase", "label_phase"), ("model_phase", "label_phase"),
        ):
            pairs = [(getattr(f, name), getattr(f, label)) for f in self.frames.values()]
            labelled = [(p, y) for p, y in pairs if y is not None]
            out[name] = (sum(p == y for p, y in labelled), len(labelled))
        return out


def _load_model(path):
    import joblib

    if path is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = joblib.load(path)
    if not isinstance(loaded, dict):
        from ahfd.ml.posture import FEATURES

        loaded = {"model": loaded, "features": list(FEATURES)}
    return loaded


def _row(values: dict, names) -> list[float]:
    return [math.nan if values.get(n) is None else float(values[n]) for n in names]


def subject_track(tracks: Path, calib, min_score: float = 0.3) -> int | None:
    """The track that occupies any calibrated bed for the most frames (the patient)."""
    from ahfd.ml import posture as P

    best, best_n = None, 0
    for zone in calib.zones.zones:
        if zone.kind != "bed":
            continue
        tid = P._dominant_bed_track(tracks, calib.ground, zone, min_score)
        if tid is None:
            continue
        n = sum(
            1 for pose in _read(tracks) for p in pose.people if p.track_id == tid
        )
        if n > best_n:
            best, best_n = tid, n
    return best


def _read(tracks):
    from ahfd.io import read_tracks

    return read_tracks(tracks)


def predict_clip(
    tracks: Path,
    calibration: Path,
    cfg,
    *,
    posture_model: Path | None = None,
    phase_model: Path | None = None,
    labels: Path | None = None,
    label_t0: float | None = None,
) -> ClipPredictions:
    """Run the rules and models over ``tracks``; attach ``labels`` if given.

    ``label_t0`` is the absolute time that label second 0 refers to; it
    defaults to this clip's first frame. Labels made on the other camera of a
    take need that camera's first-frame time (the pilot put the D435f labels on
    the D435i clock this way).
    """
    import numpy as np

    from ahfd.capture.base import SourceMeta
    from ahfd.cli import _build_detection, _depth_provenance
    from ahfd.ml import posture as P
    from ahfd.ml.temporal import CausalTemporalSummarizer

    tracks, calibration = Path(tracks), Path(calibration)
    meta = SourceMeta(uri="tracks", width=1920, height=1080, fps=30.0)
    extractor, machine, _sink, calib = _build_detection(cfg, str(calibration), meta)
    temporal = CausalTemporalSummarizer(
        max_gap_s=cfg.bed_activity.cusum.max_gap_s, near_edge_m=cfg.bed_activity.near_edge_enter_m,
    )
    posture = _load_model(posture_model)
    phase = _load_model(phase_model)
    segments = P._labelled_segments(Path(labels)) if labels is not None else []
    min_kp = cfg.pose.min_keypoint_score

    subject = subject_track(tracks, calib, 0.3)
    out = ClipPredictions(t0=math.nan, subject=subject)
    last_seen: dict[int, float] = {}
    for pose in _read(tracks):
        if math.isnan(out.t0):
            out.t0 = pose.t
        lt0 = out.t0 if label_t0 is None else label_t0
        for p in pose.people:
            if p.track_id is not None:
                last_seen[p.track_id] = pose.t
        # Identities live ~1 s after their last detection, as in the live tracker.
        live = {tid for tid, seen in last_seen.items() if pose.t - seen <= 1.0}
        extractor.retain_only(live)
        machine.retain_only(live)

        candidates = []
        for person in pose.people:
            if person.track_id is None:
                continue
            f = extractor.extract(person, pose.t)
            if f is None:
                continue
            machine.update_all(f)
            snap = machine.bed_snapshot_of(person.track_id, pose.t)
            summary = temporal.update(f, valid=f.has_geometry(), context={
                "edge_velocity_mps": snap.edge_velocity_mps if snap else None,
                "cusum_z": snap.cusum_z if snap else None,
                "cusum_g": snap.cusum_g if snap else None,
                "cusum_onset": 1.0 if snap and "cusum_onset" in snap.reasons else 0.0,
                "cusum_armed": 1.0 if snap and snap.cusum_armed else 0.0,
                **_depth_provenance(person, min_score=min_kp),
                "imu_pitch_delta_deg": None,
                "imu_roll_delta_deg": None,
                "episode_reclined": 1.0 if snap and snap.phase == "RECLINED" else 0.0,
            })
            candidates.append((person, f, snap, summary))

        if not candidates:
            continue
        # The bed's occupant when in view, else the most confident person.
        person, f, snap, summary = max(
            candidates, key=lambda c: (c[0].track_id == subject, c[0].score)
        )
        pred = FramePrediction(t=pose.t - out.t0, track_id=person.track_id)
        pred.rule_posture = RULE_POSTURE.get(machine.state_of(person.track_id))
        if snap is not None and snap.phase not in (None, "UNKNOWN"):
            pred.rule_phase = snap.phase
        if f.has_geometry():
            row = P.features_row(f, person, extractor.ground, min_kp)
            if posture is not None:
                x = np.array([_row(row, posture["features"])])
                pred.model_posture = str(posture["model"].predict(x)[0])
            if phase is not None:
                x = np.array([_row({**row, **summary.values}, phase["features"])])
                pred.model_phase = str(phase["model"].predict(x)[0])
        if segments:
            pred.label_phase = P._posture_at(segments, pose.t - lt0, "phase")
            pred.label_posture = P._posture_at(segments, pose.t - lt0, "posture") or (
                P.PHASE_TO_POSTURE.get(pred.label_phase) if pred.label_phase else None
            )
        out.frames[round(pose.t, 3)] = pred
    return out
