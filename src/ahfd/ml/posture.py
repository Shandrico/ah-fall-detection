"""Train an interpretable posture classifier from labelled clips.

This is the "optional re-ranker" path: a small scikit-learn model that learns
posture (upright / sitting / in_bed / on_ground) from the **same metric
features** the rule-based classifier uses -- torso height, floor-spread, etc.,
all range-invariant thanks to calibration. It is an add-on for study and
comparison, not a replacement for the rule-based state machine.

The design goal is that **retraining is trivial**: this reads labels + tracks +
calibration and produces a model. To retrain on new data you label more clips,
extract them, and re-run -- the code does not change, only the data grows.

Two stages, kept separate so each is testable on its own:

* ``build_dataset`` turns labelled posture segments + extracted tracks into a
  table of (metric features -> posture). Transitions (gaps between segments)
  are excluded, which is what makes the labels clean.
* ``train`` fits a small decision tree and reports accuracy, a confusion matrix,
  and per-feature importances -- the last of which answers "which joints/features
  actually separate the postures".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ahfd.geometry.depth_height import DEPTH_FEATURES, depth_features
from ahfd.ml.pose_features import JOINT_FEATURES, joint_features

_EPS = 1e-6

# Aggregate metric features pulled straight off `Features`. All in metres and
# range-invariant, so the same value means the same thing at any camera
# position. `range_m` (lens-to-person distance) is deliberately EXCLUDED: it is
# a *location* feature, not a posture one, and on single-session data a tree
# will cheat by learning "the bed is far, the chair is near" -- pure leakage
# that does not generalise. Posture must be decided by shape/height, never by
# where someone stood, exactly as the rule-based classifier does.
AGG_FEATURES = [
    "h_torso",  # shoulders+hips centroid height
    "h_max",  # highest joint
    "h_min",  # lowest joint
    "floor_spread",  # large upright, ~body length when down
    "vertical_extent",  # h_max - h_min, metres
    "compactness",  # vertical_extent / floor_spread -- tall-and-thin vs flat
]
# The four aggregate heights read directly as attributes of `Features`.
_AGG_FROM_FEATS = ["h_torso", "h_max", "h_min", "floor_spread"]

# The full input set: the aggregates above plus the per-joint heights, joint
# angles, and shape ratios from `pose_features`. This is the "use the whole
# skeleton, not just torso" change -- the trainer reports which of these
# actually separate the postures (see `train`).
#
# DEPTH_FEATURES are appended last: joint heights measured directly from depth
# (dh_*). They are None for any clip recorded without depth, so the trainer
# imputes them and RGB-only data behaves exactly as before -- but on a
# depth-recorded clip they add the measured-height signal that fixes the nadir
# case. This is what makes the model depth-ready without a second pipeline.
FEATURES = AGG_FEATURES + JOINT_FEATURES + DEPTH_FEATURES

# A row is kept only if at least these are present -- the rest are imputed.
REQUIRED = ["h_torso", "floor_spread"]


def _finite_or_none(v):
    """Coerce inf/nan/None to None so every stored feature is a real number or absent.

    `floor_spread` in particular is +inf when a ray misses the floor; that is a
    genuine "no measurement", not a huge value, so it must not reach the model
    as one. Everything non-finite is imputed downstream instead.
    """
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def features_row(feats, person, ground, min_keypoint_score: float = 0.3) -> dict:
    """The FEATURES-keyed dict for one person-frame.

    Shared by training (`build_dataset`) and export (`ahfd export-features`) so
    the numbers a model trains on are exactly the numbers you can inspect. All
    values are finite floats or None (non-finite coerced away).
    """
    row = {k: getattr(feats, k) for k in _AGG_FROM_FEATS}
    ve = feats.height_spread
    row["vertical_extent"] = ve
    row["compactness"] = (
        ve / (feats.floor_spread + _EPS)
        if ve is not None and math.isfinite(feats.floor_spread)
        else None
    )
    row.update(joint_features(person, ground, feats.contact_xy, min_keypoint_score))
    # Depth-measured joint heights, when the clip was recorded with depth
    # (person.heights). All None otherwise -> imputed, no effect on RGB clips.
    row.update(depth_features(getattr(person, "heights", None)))
    return {k: _finite_or_none(v) for k, v in row.items()}


def _posture_at(segments: list[dict], t: float) -> str | None:
    """The labelled posture at time ``t``, or None if ``t`` is in a gap.

    Gaps between segments are transitions -- deliberately unlabelled, so they
    are excluded from training rather than forced into a class.
    """
    for seg in segments:
        if seg["start_s"] <= t <= seg["end_s"]:
            return str(seg["posture"])
    return None


def _labelled_segments(path: Path) -> list[dict]:
    """Read a posture-label file's real segments (placeholders skipped)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        s
        for s in data.get("segments", [])
        if float(s.get("end_s", 0)) > float(s.get("start_s", 0))
    ]


def _main_person(pose):
    """The person to score in a frame: the highest-confidence tracked one.

    The clips are single-subject, so this is just "the person in view"; taking
    the most confident track guards against a spurious second detection.
    """
    people = [p for p in pose.people if p.track_id is not None]
    if not people:
        return None
    return max(people, key=lambda p: p.score)


@dataclass
class TrainResult:
    classes: list[str]
    n_total: int
    n_train: int
    n_test: int
    accuracy: float
    confusion: list[list[int]]
    report: str
    importances: list[tuple[str, float]]
    # Univariate separability: (feature, ANOVA F-score, mutual information),
    # ranked. Unlike the tree's importances this scores each feature on its own
    # and is stable on small data -- the honest answer to "which joint/angle
    # actually distinguishes the postures".
    separability: list[tuple[str, float, float]]
    # Per-class mean of each feature (nan-aware), so a difference is readable:
    # e.g. knee_angle ~95 deg sitting vs ~172 deg upright.
    class_means: dict[str, dict[str, float]]
    split_done: bool
    model: object = field(default=None, repr=False)


def build_dataset(labels_dir, tracks_dir, calib, min_keypoint_score: float = 0.3):
    """Build (rows, labels, groups, used, skipped) from labels + extracted tracks.

    ``rows`` is a list of {feature: value} dicts, ``labels`` the matching
    postures, and ``groups`` the clip each row came from -- so an evaluator can
    hold out a whole clip (frames within one recording are near-duplicates, and
    splitting them across train/test leaks). ``used`` is a list of
    (clip_id, n_frames) for coverage, ``skipped`` the label files that failed to
    parse. A clip contributes only if it has real segments *and* a matching
    ``<clip>.jsonl`` in ``tracks_dir``.
    """
    from ahfd.features import FeatureExtractor
    from ahfd.io import read_tracks

    labels_dir, tracks_dir = Path(labels_dir), Path(tracks_dir)
    rows: list[dict] = []
    labels: list[str] = []
    groups: list[str] = []
    used: list[tuple[str, int]] = []
    skipped: list[tuple[str, str]] = []  # (filename, why) -- e.g. a JSON typo

    for label_file in sorted(labels_dir.glob("*.json")):
        # A hand-edited label file with a JSON typo should not crash the whole
        # run -- skip it and report which one, so the rest still trains.
        try:
            segments = _labelled_segments(label_file)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            skipped.append((label_file.name, str(exc)))
            continue
        if not segments:
            continue
        track_path = tracks_dir / (label_file.stem + ".jsonl")
        if not track_path.exists():
            continue

        extractor = FeatureExtractor(
            calib.ground, zones=calib.zones, min_keypoint_score=min_keypoint_score
        )
        n_clip = 0
        for pose in read_tracks(track_path):
            posture = _posture_at(segments, pose.t)
            if posture is None:  # a transition/gap -- excluded
                continue
            person = _main_person(pose)
            if person is None:
                continue
            feats = extractor.extract(person, pose.t)
            if feats is None or not feats.has_geometry():
                continue

            row = features_row(
                feats, person, extractor.ground, min_keypoint_score
            )
            if any(row.get(k) is None for k in REQUIRED):
                continue
            rows.append(row)
            labels.append(posture)
            groups.append(label_file.stem)
            n_clip += 1
        used.append((label_file.stem, n_clip))

    return rows, labels, groups, used, skipped


def train(rows: list[dict], labels: list[str], *, test_size=0.3, max_depth=5, seed=0):
    """Fit a small decision tree on the feature table and report on it.

    Missing optional features are median-imputed; the tree is kept shallow so it
    stays interpretable and its ``feature_importances_`` are meaningful. When
    there are too few samples to hold out a stratified test set, it trains on
    everything and reports training-set accuracy -- flagged via ``split_done``.
    """
    import warnings

    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    X = np.array(
        [[np.nan if r.get(k) is None else float(r[k]) for k in FEATURES] for r in rows],
        dtype=float,
    )
    y = np.array(labels)
    classes = sorted(set(labels))

    def make_model():
        return Pipeline(
            [
                # keep_empty_features: a feature absent from every row (e.g. no
                # wrists ever visible) must stay a column so importances line up
                # with FEATURES; it is filled with 0 and simply carries no signal.
                (
                    "impute",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("clf", DecisionTreeClassifier(max_depth=max_depth, random_state=seed)),
            ]
        )

    # A stratified hold-out needs at least 2 of every class; otherwise the split
    # is not meaningful, so train on all and report on the training set.
    import numpy as _np

    counts = {c: int((_np.array(labels) == c).sum()) for c in classes}
    split_done = len(classes) >= 2 and min(counts.values()) >= 2 and len(rows) >= 8

    model = make_model()
    if split_done:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = float(accuracy_score(y_te, y_pred))
        report = classification_report(y_te, y_pred, zero_division=0)
        conf = confusion_matrix(y_te, y_pred, labels=classes)
        n_tr, n_te = len(y_tr), len(y_te)
    else:
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = float(accuracy_score(y, y_pred))
        report = classification_report(y, y_pred, zero_division=0)
        conf = confusion_matrix(y, y_pred, labels=classes)
        n_tr, n_te = len(y), 0

    tree = model.named_steps["clf"]
    importances = sorted(
        zip(FEATURES, (float(v) for v in tree.feature_importances_)),
        key=lambda kv: kv[1],
        reverse=True,
    )

    # Univariate separability, computed on the median-imputed table so every
    # feature is scored on the same rows. ANOVA F asks "do the class means
    # differ relative to the spread"; mutual information catches non-linear
    # splits an F-test misses. Constant columns give a degenerate F (nan) and a
    # divide-by-zero warning -- silence it and treat them as zero signal. Both
    # need at least two samples per class to be defined, so skip them on the
    # degenerate tiny-data path (where the number is meaningless anyway).
    separability: list[tuple[str, float, float]] = []
    if min(counts.values()) >= 2 and len(rows) >= 4:
        X_imp = SimpleImputer(
            strategy="median", keep_empty_features=True
        ).fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, _ = f_classif(X_imp, y)
            mi = mutual_info_classif(X_imp, y, random_state=seed)
        separability = sorted(
            (
                (feat, float(np.nan_to_num(f)), float(m))
                for feat, f, m in zip(FEATURES, f_scores, mi)
            ),
            key=lambda t: t[1],
            reverse=True,
        )

    # Per-class feature means, nan-aware so a missing value in a row does not
    # poison the average. All-nan (feature never seen for a class) reads as nan.
    class_means: dict[str, dict[str, float]] = {}
    for c in classes:
        rowsel = X[y == c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            means = np.nanmean(rowsel, axis=0)
        class_means[c] = {feat: float(m) for feat, m in zip(FEATURES, means)}

    return TrainResult(
        classes=classes,
        n_total=len(rows),
        n_train=n_tr,
        n_test=n_te,
        accuracy=acc,
        confusion=[[int(x) for x in row] for row in conf],
        report=report,
        importances=importances,
        separability=separability,
        class_means=class_means,
        split_done=split_done,
        model=model,
    )
