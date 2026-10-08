"""Causal temporal summaries and inexpensive learned comparators.

The live detector receives one :class:`~ahfd.features.Features` value at a
time.  This module deliberately keeps that streaming contract: a summary at
time ``t`` is made only from observations at or before ``t``.  There is no
centred smoothing, interpolation, or access to a completed clip.

The learned models are intentionally small.  Logistic regression provides a
linear, contribution-based explanation; a shallow decision tree provides the
literal threshold path.  Evaluation holds out whole, explicitly identified
subjects or sessions.  Frames from one recording are never randomly divided
between train and test.

scikit-learn is optional in AHFD, so it is imported only inside model methods.
Importing this module therefore remains safe in the core runtime.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence


TEMPORAL_SCHEMA_VERSION = "ahfd-temporal-v2"
WINDOWS_S: tuple[float, ...] = (0.5, 2.0, 4.0, 8.0)

# These are values already available on Features.  Location/range and patient
# risk are intentionally absent: the former is easy for a model to memorise,
# while the latter belongs in alert policy rather than behaviour recognition.
SIGNALS: tuple[str, ...] = (
    "h_torso",
    "h_shoulder",
    "h_head",
    "floor_spread",
    "v_z",
    "motion",
    "torso_tilt",
    "bed_overlap",
    "bed_support_fraction",
    "bed_edge_distance_m",
    "edge_velocity_mps",
    "cusum_z",
    "cusum_g",
    "cusum_onset",
    "cusum_armed",
    "joint_depth_valid_fraction",
    "shoulder_depth_available",
    "torso_depth_available",
    "imu_pitch_delta_deg",
    "imu_roll_delta_deg",
    "mean_conf",
    "n_valid_kp",
    "bed_supported",
)

_QUALITY_SIGNALS = frozenset(
    {
        "mean_conf",
        "n_valid_kp",
        "joint_depth_valid_fraction",
        "shoulder_depth_available",
        "torso_depth_available",
        "imu_pitch_delta_deg",
        "imu_roll_delta_deg",
    }
)
_WINDOW_STATS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "delta",
    "slope",
    "max_rise",
    "valid_fraction",
    "missing",
)


def _window_tag(seconds: float) -> str:
    return (str(seconds).replace(".", "p").rstrip("0").rstrip("p") + "s")


def _feature_names() -> tuple[str, ...]:
    names = [
        "monitoring_valid_now",
        "track_age_s",
        "monitoring_gap_s",
        "monitoring_gap_missing",
        "dwell_near_edge_s",
        "dwell_near_edge_missing",
        "cumulative_edge_progress_m",
        "time_since_support_lost_s",
        "time_since_support_lost_missing",
    ]
    for signal in SIGNALS:
        names.extend(
            (
                f"{signal}__now",
                f"{signal}__missing_now",
                f"{signal}__since_valid_s",
                f"{signal}__since_valid_missing",
            )
        )
        for window in WINDOWS_S:
            tag = _window_tag(window)
            names.extend(f"{signal}__{stat}_{tag}" for stat in _WINDOW_STATS)
    return tuple(names)


TEMPORAL_FEATURE_NAMES: tuple[str, ...] = _feature_names()
TEMPORAL_FEATURE_ORDER_SHA256 = hashlib.sha256(
    json.dumps(
        list(TEMPORAL_FEATURE_NAMES),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
# Absolute track age can encode a scripted study schedule rather than patient
# behaviour. Keep it in the recorded audit schema, but exclude it from learned
# comparators unless an investigator opts into a documented sensitivity run.
MODEL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in TEMPORAL_FEATURE_NAMES if name != "track_age_s"
)


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_geometry(features) -> bool:
    method = getattr(features, "has_geometry", None)
    if callable(method):
        return bool(method())
    return (
        getattr(features, "contact_xy", None) is not None
        and _finite(getattr(features, "h_torso", None)) is not None
    )


@dataclass(frozen=True)
class TemporalSummary:
    """One immutable-by-convention model row for one track and timestamp."""

    track_id: int
    t: float
    values: dict[str, float | None]
    schema_version: str = TEMPORAL_SCHEMA_VERSION

    def as_vector(
        self, feature_names: Sequence[str] = TEMPORAL_FEATURE_NAMES
    ) -> list[float | None]:
        return [self.values.get(name) for name in feature_names]


@dataclass
class _Observation:
    t: float
    values: dict[str, float | None]
    monitoring_valid: bool


@dataclass
class _TrackHistory:
    first_t: float
    last_t: float
    observations: deque[_Observation] = field(default_factory=deque)
    last_monitoring_valid_t: float | None = None
    last_signal_t: dict[str, float] = field(default_factory=dict)
    near_edge_since: float | None = None
    support_lost_since: float | None = None
    last_edge_distance: float | None = None
    cumulative_edge_progress_m: float = 0.0


class CausalTemporalSummarizer:
    """Build fixed causal summaries independently for every live track.

    ``valid`` is the caller's monitoring-quality decision.  If omitted, the
    least-assumptive geometry check is used.  Geometry signals are masked when
    monitoring is invalid; confidence and keypoint-count remain visible so a
    model can distinguish missing evidence from ordinary stillness.

    A long timestamp gap starts a fresh track history.  This prevents a newly
    associated person from inheriting stale movement from before an occlusion.
    """

    def __init__(
        self,
        windows_s: Sequence[float] = WINDOWS_S,
        *,
        max_gap_s: float = 2.0,
        near_edge_m: float = 0.22,
    ) -> None:
        windows = tuple(float(w) for w in windows_s)
        if windows != WINDOWS_S:
            raise ValueError(
                "the temporal feature schema has fixed windows "
                + repr(WINDOWS_S)
                + "; got "
                + repr(windows)
            )
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        self.windows_s = windows
        self.max_gap_s = float(max_gap_s)
        self.near_edge_m = float(near_edge_m)
        self._tracks: dict[int, _TrackHistory] = {}

    def reset(self, track_id: int | None = None) -> None:
        """Drop one track's history, or every history when ``track_id`` is None."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(int(track_id), None)

    def retain_only(self, live_ids: set[int]) -> None:
        """Forget histories whose tracker IDs are no longer live."""
        for track_id in list(self._tracks):
            if track_id not in live_ids:
                del self._tracks[track_id]

    def mark_unobserved(self, track_id: int, t: float) -> None:
        """Break every temporal dependency after a missed observation.

        A tracker can keep an ID alive while pose estimation misses a frame.
        Retaining windows in that situation would connect edge motion, dwell,
        and support state across time in which the person was not observable.
        The conservative first-study policy is therefore to start a fresh
        causal history when the association is seen again.
        """
        track_id = int(track_id)
        now = float(t)
        if not math.isfinite(now):
            raise ValueError("unobserved timestamp must be finite")
        state = self._tracks.get(track_id)
        if state is not None and now < state.last_t:
            raise ValueError(
                f"track {track_id} moved backwards in time: {now} < {state.last_t}"
            )
        self.reset(track_id)

    @property
    def live_ids(self) -> set[int]:
        return set(self._tracks)

    def update(
        self,
        features,
        *,
        valid: bool | None = None,
        context: Mapping[str, float | None] | None = None,
    ) -> TemporalSummary:
        """Consume one current ``Features`` object and return its causal row."""
        track_id = int(features.track_id)
        now = float(features.t)
        if not math.isfinite(now):
            raise ValueError("feature timestamp must be finite")

        state = self._tracks.get(track_id)
        if state is not None and now < state.last_t:
            raise ValueError(
                f"track {track_id} moved backwards in time: {now} < {state.last_t}"
            )
        if state is not None and now - state.last_t > self.max_gap_s:
            self.reset(track_id)
            state = None
        if state is None:
            state = _TrackHistory(first_t=now, last_t=now)
            self._tracks[track_id] = state

        monitoring_valid = _has_geometry(features) if valid is None else bool(valid)
        context_values = context or {}
        values = self._read_signals(features, monitoring_valid, context_values)
        observation = _Observation(now, values, monitoring_valid)
        state.observations.append(observation)
        state.last_t = now
        if monitoring_valid:
            state.last_monitoring_valid_t = now
        for signal, value in values.items():
            if value is not None:
                state.last_signal_t[signal] = now

        edge = values.get("bed_edge_distance_m")
        if _finite(context_values.get("episode_reclined")) == 1.0:
            state.cumulative_edge_progress_m = 0.0
            state.support_lost_since = None
        if edge is not None:
            if state.last_edge_distance is not None:
                # Only motion toward the boundary accumulates; retreat does not
                # erase evidence already observed in this association episode.
                state.cumulative_edge_progress_m += max(
                    0.0, state.last_edge_distance - edge
                )
            state.last_edge_distance = edge
        if monitoring_valid and edge is not None and edge <= self.near_edge_m:
            if state.near_edge_since is None:
                state.near_edge_since = now
        else:
            state.near_edge_since = None

        supported = values.get("bed_supported")
        if monitoring_valid and supported == 0.0:
            if state.support_lost_since is None:
                state.support_lost_since = now
        elif supported == 1.0:
            state.support_lost_since = None

        oldest = now - max(self.windows_s)
        while state.observations and state.observations[0].t < oldest:
            state.observations.popleft()

        row: dict[str, float | None] = {
            "monitoring_valid_now": float(monitoring_valid),
            "track_age_s": now - state.first_t,
            "monitoring_gap_s": (
                None
                if state.last_monitoring_valid_t is None
                else now - state.last_monitoring_valid_t
            ),
            "monitoring_gap_missing": float(state.last_monitoring_valid_t is None),
            "dwell_near_edge_s": (
                None if state.near_edge_since is None else now - state.near_edge_since
            ),
            "dwell_near_edge_missing": float(state.near_edge_since is None),
            "cumulative_edge_progress_m": state.cumulative_edge_progress_m,
            "time_since_support_lost_s": (
                None
                if state.support_lost_since is None
                else now - state.support_lost_since
            ),
            "time_since_support_lost_missing": float(
                state.support_lost_since is None
            ),
        }
        for signal in SIGNALS:
            current = values[signal]
            last_t = state.last_signal_t.get(signal)
            row[f"{signal}__now"] = current
            row[f"{signal}__missing_now"] = float(current is None)
            row[f"{signal}__since_valid_s"] = None if last_t is None else now - last_t
            row[f"{signal}__since_valid_missing"] = float(last_t is None)

            for window in self.windows_s:
                tag = _window_tag(window)
                observations = [
                    obs for obs in state.observations if now - obs.t <= window
                ]
                series = [
                    (obs.t, obs.values[signal])
                    for obs in observations
                    if obs.values[signal] is not None
                ]
                stats = self._summaries(series, len(observations))
                for stat, value in stats.items():
                    row[f"{signal}__{stat}_{tag}"] = value

        # Always return a fresh dict: later updates cannot mutate an earlier row.
        return TemporalSummary(track_id=track_id, t=now, values=dict(row))

    @staticmethod
    def _read_signals(
        features,
        monitoring_valid: bool,
        context: Mapping[str, float | None],
    ) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for signal in SIGNALS:
            if signal == "bed_supported":
                value = float(getattr(features, "supported_by_bed", None) is not None)
            elif signal == "bed_support_fraction":
                value = _finite(
                    getattr(
                        features,
                        "bed_support_fraction",
                        getattr(features, "bed_overlap", None),
                    )
                )
            elif signal in context:
                value = _finite(context.get(signal))
            else:
                value = _finite(getattr(features, signal, None))
            if not monitoring_valid and signal not in _QUALITY_SIGNALS:
                value = None
            values[signal] = value
        return values

    @staticmethod
    def _summaries(
        series: list[tuple[float, float | None]], total_count: int
    ) -> dict[str, float | None]:
        clean = [(t, float(v)) for t, v in series if v is not None]
        valid_fraction = len(clean) / total_count if total_count else 0.0
        if not clean:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "delta": None,
                "slope": None,
                "max_rise": None,
                "valid_fraction": valid_fraction,
                "missing": 1.0,
            }

        vals = [value for _, value in clean]
        mean = sum(vals) / len(vals)
        variance = sum((value - mean) ** 2 for value in vals) / len(vals)
        delta = vals[-1] - vals[0]

        slope = 0.0
        if len(clean) >= 2:
            times = [t for t, _ in clean]
            t_mean = sum(times) / len(times)
            denom = sum((t - t_mean) ** 2 for t in times)
            if denom > 1e-12:
                slope = sum(
                    (t - t_mean) * (value - mean) for t, value in clean
                ) / denom

        running_min = vals[0]
        max_rise = 0.0
        for value in vals[1:]:
            max_rise = max(max_rise, value - running_min)
            running_min = min(running_min, value)

        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(vals),
            "max": max(vals),
            "delta": delta,
            "slope": slope,
            "max_rise": max_rise,
            "valid_fraction": valid_fraction,
            "missing": 0.0,
        }


# ---------------------------------------------------------------------------
# Group-disjoint, explainable scikit-learn comparators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleMetadata:
    """Explicit grouping metadata; tracker IDs and filenames are not identity."""

    subject_id: str
    session_id: str
    clip_id: str = ""
    t: float | None = None

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be explicit and nonempty")
        if not self.session_id.strip():
            raise ValueError("session_id must be explicit and nonempty")

    def group(self, by: Literal["subject", "session"]) -> str:
        if by == "subject":
            return self.subject_id
        if by == "session":
            # Session names are often reused (e.g. "01") across subjects.
            return self.subject_id + "::" + self.session_id
        raise ValueError("group_by must be 'subject' or 'session'")


@dataclass(frozen=True)
class ModelMetadata:
    schema_version: str
    model_kind: str
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    group_by: str
    trained_groups: tuple[str, ...]
    n_samples: int
    random_seed: int
    task: str | None = None
    horizon_s: int | None = None
    input_manifest_sha256: tuple[str, ...] = ()
    dependency_versions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    raw_value: float | None
    model_value: float
    contribution: float


@dataclass(frozen=True)
class DecisionStep:
    feature: str
    raw_value: float | None
    model_value: float
    operator: str
    threshold: float


@dataclass(frozen=True)
class PredictionExplanation:
    model_kind: str
    predicted_class: str
    probability: float
    contributions: tuple[FeatureContribution, ...] = ()
    decision_path: tuple[DecisionStep, ...] = ()
    intercept: float | None = None
    metadata: ModelMetadata | None = None


def _row_values(row: TemporalSummary | Mapping[str, float | None]) -> Mapping:
    return row.values if isinstance(row, TemporalSummary) else row


def _matrix(rows, feature_names: Sequence[str]):
    import numpy as np

    return np.asarray(
        [
            [
                np.nan if _finite(_row_values(row).get(name)) is None
                else float(_row_values(row)[name])
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )


def _make_pipeline(kind: str, seed: int, max_depth: int, min_samples_leaf: int):
    # Lazy imports are required: sklearn is an optional dependency.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if kind == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "tree":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError("model kind must be 'logistic' or 'tree'")


class TemporalModel:
    """A fitted small model with stable feature order and reviewable evidence."""

    def __init__(self, pipeline, metadata: ModelMetadata):
        self.pipeline = pipeline
        self.metadata = metadata

    def predict(
        self, rows: Sequence[TemporalSummary | Mapping[str, float | None]]
    ) -> list[str]:
        X = _matrix(rows, self.metadata.feature_names)
        return [str(value) for value in self.pipeline.predict(X)]

    def predict_proba(
        self, row: TemporalSummary | Mapping[str, float | None]
    ) -> dict[str, float]:
        X = _matrix([row], self.metadata.feature_names)
        classes = self.pipeline.named_steps["clf"].classes_
        probabilities = self.pipeline.predict_proba(X)[0]
        return {str(c): float(p) for c, p in zip(classes, probabilities)}

    def explain(
        self,
        row: TemporalSummary | Mapping[str, float | None],
        *,
        top_k: int = 8,
    ) -> PredictionExplanation:
        """Explain one prediction using contributions or the traversed tree path."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        raw = _row_values(row)
        names = self.metadata.feature_names
        X = _matrix([row], names)
        imputed = self.pipeline.named_steps["impute"].transform(X)
        clf = self.pipeline.named_steps["clf"]
        predicted = str(self.pipeline.predict(X)[0])
        probabilities = self.predict_proba(row)

        if self.metadata.model_kind == "logistic":
            transformed = self.pipeline.named_steps["scale"].transform(imputed)[0]
            classes = [str(c) for c in clf.classes_]
            if len(classes) == 2:
                direction = 1.0 if predicted == classes[1] else -1.0
                coefficients = clf.coef_[0] * direction
                intercept = float(clf.intercept_[0] * direction)
            else:
                class_index = classes.index(predicted)
                coefficients = clf.coef_[class_index]
                intercept = float(clf.intercept_[class_index])
            contributions = coefficients * transformed
            order = sorted(
                range(len(names)), key=lambda i: abs(float(contributions[i])), reverse=True
            )[:top_k]
            details = tuple(
                FeatureContribution(
                    feature=names[i],
                    raw_value=_finite(raw.get(names[i])),
                    model_value=float(transformed[i]),
                    contribution=float(contributions[i]),
                )
                for i in order
            )
            return PredictionExplanation(
                model_kind="logistic",
                predicted_class=predicted,
                probability=probabilities[predicted],
                contributions=details,
                intercept=intercept,
                metadata=self.metadata,
            )

        tree = clf.tree_
        model_row = imputed[0]
        node = 0
        path: list[DecisionStep] = []
        while tree.children_left[node] != tree.children_right[node]:
            feature_index = int(tree.feature[node])
            threshold = float(tree.threshold[node])
            value = float(model_row[feature_index])
            go_left = value <= threshold
            path.append(
                DecisionStep(
                    feature=names[feature_index],
                    raw_value=_finite(raw.get(names[feature_index])),
                    model_value=value,
                    operator="<=" if go_left else ">",
                    threshold=threshold,
                )
            )
            node = int(tree.children_left[node] if go_left else tree.children_right[node])
        return PredictionExplanation(
            model_kind="tree",
            predicted_class=predicted,
            probability=probabilities[predicted],
            decision_path=tuple(path),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class FoldInfo:
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    n_train: int
    n_test: int


@dataclass(frozen=True)
class ComparatorScore:
    model_kind: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_group_accuracy: tuple[tuple[str, float], ...]
    predictions: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]
    per_class_average_precision: tuple[tuple[str, float], ...]
    macro_average_precision: float
    brier_score: float
    positive_class: str | None = None
    positive_precision: float | None = None
    positive_recall: float | None = None
    false_positive_rate: float | None = None


@dataclass
class ComparatorResult:
    classes: tuple[str, ...]
    group_by: str
    folds: tuple[FoldInfo, ...]
    scores: tuple[ComparatorScore, ...]
    models: dict[str, TemporalModel]


def compare_grouped(
    rows: Sequence[TemporalSummary | Mapping[str, float | None]],
    labels: Sequence[str],
    metadata: Sequence[SampleMetadata],
    *,
    group_by: Literal["subject", "session"] = "subject",
    model_kinds: Sequence[Literal["logistic", "tree"]] = ("logistic", "tree"),
    feature_names: Sequence[str] = MODEL_FEATURE_NAMES,
    seed: int = 0,
    max_depth: int = 4,
    min_samples_leaf: int = 2,
) -> ComparatorResult:
    """Leave one explicit subject/session out and compare small models.

    Every fold creates a fresh preprocessing pipeline, so median imputation and
    scaling are fitted on training groups only.  A separate final model is fit
    on all rows *after* out-of-fold metrics are complete; it is returned for
    deployment/explanation, never used to calculate those metrics.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import LeaveOneGroupOut

    if not rows or len(rows) != len(labels) or len(rows) != len(metadata):
        raise ValueError("rows, labels, and metadata must be nonempty and equal-length")
    if group_by not in ("subject", "session"):
        raise ValueError("group_by must be 'subject' or 'session'")
    kinds = tuple(model_kinds)
    if not kinds or any(kind not in ("logistic", "tree") for kind in kinds):
        raise ValueError("model_kinds must contain only 'logistic' and/or 'tree'")
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be nonempty and unique")

    y = np.asarray([str(label) for label in labels], dtype=object)
    classes = tuple(sorted(set(y)))
    class_index = {label: index for index, label in enumerate(classes)}
    if len(classes) < 2:
        raise ValueError("at least two target classes are required")
    groups = np.asarray([item.group(group_by) for item in metadata], dtype=object)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("at least two independent groups are required")
    X = _matrix(rows, names)

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))
    folds = tuple(
        FoldInfo(
            train_groups=tuple(sorted(set(groups[train_idx]))),
            test_groups=tuple(sorted(set(groups[test_idx]))),
            n_train=len(train_idx),
            n_test=len(test_idx),
        )
        for train_idx, test_idx in splits
    )
    for fold in folds:
        if set(fold.train_groups) & set(fold.test_groups):
            raise AssertionError("group leakage in comparator split")

    scores: list[ComparatorScore] = []
    models: dict[str, TemporalModel] = {}
    for kind in kinds:
        out_of_fold = np.empty(len(y), dtype=object)
        out_of_fold_probability = np.zeros((len(y), len(classes)), dtype=float)
        per_group: list[tuple[str, float]] = []
        for train_idx, test_idx in splits:
            train_classes = set(y[train_idx])
            if len(train_classes) < 2:
                held_out = sorted(set(groups[test_idx]))
                raise ValueError(
                    "training fold has fewer than two classes after holding out "
                    + ", ".join(held_out)
                )
            missing_classes = set(classes) - train_classes
            if missing_classes:
                held_out = sorted(set(groups[test_idx]))
                raise ValueError(
                    "training fold is missing target class(es) "
                    + ", ".join(sorted(missing_classes))
                    + " after holding out "
                    + ", ".join(held_out)
                )
            pipeline = _make_pipeline(kind, seed, max_depth, min_samples_leaf)
            pipeline.fit(X[train_idx], y[train_idx])
            prediction = pipeline.predict(X[test_idx])
            probability = pipeline.predict_proba(X[test_idx])
            out_of_fold[test_idx] = prediction
            for fold_column, label in enumerate(pipeline.named_steps["clf"].classes_):
                out_of_fold_probability[test_idx, class_index[str(label)]] = probability[
                    :, fold_column
                ]
            group = str(groups[test_idx][0])
            per_group.append(
                (group, float(accuracy_score(y[test_idx], prediction)))
            )

        one_hot = np.zeros_like(out_of_fold_probability)
        for row_index, label in enumerate(y):
            one_hot[row_index, class_index[str(label)]] = 1.0
        per_class_ap = tuple(
            (
                label,
                float(
                    average_precision_score(
                        one_hot[:, class_index[label]],
                        out_of_fold_probability[:, class_index[label]],
                    )
                ),
            )
            for label in classes
        )
        positive_class = "exit" if "exit" in classes and len(classes) == 2 else None
        positive_precision = positive_recall = false_positive_rate = None
        if positive_class is not None:
            truth_positive = y == positive_class
            predicted_positive = out_of_fold == positive_class
            positive_precision = float(
                precision_score(truth_positive, predicted_positive, zero_division=0)
            )
            positive_recall = float(
                recall_score(truth_positive, predicted_positive, zero_division=0)
            )
            negatives = ~truth_positive
            false_positive_rate = float(
                np.count_nonzero(predicted_positive & negatives)
                / max(1, np.count_nonzero(negatives))
            )
            brier = float(
                np.mean(
                    (
                        out_of_fold_probability[:, class_index[positive_class]]
                        - truth_positive.astype(float)
                    )
                    ** 2
                )
            )
        else:
            brier = float(np.mean(np.sum((out_of_fold_probability - one_hot) ** 2, axis=1)))

        scores.append(
            ComparatorScore(
                model_kind=kind,
                accuracy=float(accuracy_score(y, out_of_fold)),
                balanced_accuracy=float(balanced_accuracy_score(y, out_of_fold)),
                macro_f1=float(f1_score(y, out_of_fold, average="macro", zero_division=0)),
                per_group_accuracy=tuple(sorted(per_group)),
                predictions=tuple(str(value) for value in out_of_fold),
                probabilities=tuple(
                    tuple(float(value) for value in row)
                    for row in out_of_fold_probability
                ),
                per_class_average_precision=per_class_ap,
                macro_average_precision=float(
                    sum(value for _, value in per_class_ap) / len(per_class_ap)
                ),
                brier_score=brier,
                positive_class=positive_class,
                positive_precision=positive_precision,
                positive_recall=positive_recall,
                false_positive_rate=false_positive_rate,
            )
        )

        final_pipeline = _make_pipeline(kind, seed, max_depth, min_samples_leaf)
        final_pipeline.fit(X, y)
        model_metadata = ModelMetadata(
            schema_version=TEMPORAL_SCHEMA_VERSION,
            model_kind=kind,
            feature_names=names,
            classes=classes,
            group_by=group_by,
            trained_groups=tuple(unique_groups),
            n_samples=len(rows),
            random_seed=seed,
        )
        models[kind] = TemporalModel(final_pipeline, model_metadata)

    scores.sort(key=lambda score: (score.macro_f1, score.balanced_accuracy), reverse=True)
    return ComparatorResult(
        classes=classes,
        group_by=group_by,
        folds=folds,
        scores=tuple(scores),
        models=models,
    )


__all__ = [
    "TEMPORAL_SCHEMA_VERSION",
    "WINDOWS_S",
    "SIGNALS",
    "TEMPORAL_FEATURE_NAMES",
    "TEMPORAL_FEATURE_ORDER_SHA256",
    "MODEL_FEATURE_NAMES",
    "TemporalSummary",
    "CausalTemporalSummarizer",
    "SampleMetadata",
    "ModelMetadata",
    "FeatureContribution",
    "DecisionStep",
    "PredictionExplanation",
    "TemporalModel",
    "FoldInfo",
    "ComparatorScore",
    "ComparatorResult",
    "compare_grouped",
]
