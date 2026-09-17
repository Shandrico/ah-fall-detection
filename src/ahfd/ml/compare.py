"""Train the posture classifier several ways and rank them honestly.

Two questions this answers:

1. **Which approach wins?** A flat 4-class model, or the coarse-to-fine
   *cascade* (upright -> sitting-vs-down -> on_ground-vs-in_bed) where each
   node only uses the joints that actually decide it? We train both, plus a
   pure hand-threshold baseline, and rank them.

2. **How do they decide?** Three mechanisms sit side by side so the difference
   is visible:
   * a **decision tree** is literally a cascade of ``feature <= number`` tests
     -- the thresholds are *learned*, not hand-set (``ahfd train-posture`` can
     print them with ``export_text``);
   * a **linear/logistic** model has no per-feature threshold at all -- it takes
     a weighted sum of the features and picks the highest-scoring class;
   * a **hand-rule** cascade uses thresholds *we* set from the per-class means.

The evaluation is **leave-one-clip-out**, never a random frame split. Frames
inside one recording are near-duplicates; splitting them across train and test
lets the model recognise "this exact moment" and reports an accuracy it will
not reach on a new person. Holding out a whole clip is the weakest honest test
we can run today -- and it is still optimistic, because every clip so far is the
same person at the same camera. The real number needs several people, at which
point the grouping simply becomes leave-one-*person*-out and this code is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ahfd.ml.posture import FEATURES

# Column index of each feature in the FEATURES-ordered matrix.
_FIDX = {f: i for i, f in enumerate(FEATURES)}


def _cols(names: list[str]) -> list[int]:
    return [_FIDX[n] for n in names]


def rows_to_matrix(rows: list[dict]) -> np.ndarray:
    """(n, len(FEATURES)) float matrix, missing values as NaN."""
    return np.array(
        [[np.nan if r.get(k) is None else float(r[k]) for k in FEATURES] for r in rows],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# The cascade's per-node feature sets. Each node only sees the joints/shape
# features that separate *its* decision, which is the whole point of the
# hierarchy -- knee_angle belongs at the upright node and nowhere else.
# ---------------------------------------------------------------------------
UPRIGHT_FEATURES = [
    "knee_angle",
    "hip_angle",
    "torso_tilt",
    "bbox_aspect",
    "floor_spread",
    "vertical_extent",
]
SIT_VS_DOWN_FEATURES = [
    "bbox_aspect",
    "torso_tilt",
    "knee_above_ankle",
    "vertical_extent",
    "h_hip",
    "compactness",
]
# on_ground vs in_bed is barely separable from pose alone (both are flat and
# low) -- it is really a *location* question the bed zone answers. These are the
# best pose-only signals, and the confusion matrix is expected to show this node
# as the weak link until a bed zone is available.
GROUND_VS_BED_FEATURES = [
    "torso_tilt",
    "h_torso",
    "h_hip",
    "compactness",
    "vertical_extent",
]


class _Binary:
    """A binary node: a sklearn estimator, or a constant if a fold saw one class."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.const: bool | None = None

    def fit(self, X, y_bool):
        y_bool = np.asarray(y_bool, dtype=bool)
        if len(np.unique(y_bool)) < 2:
            self.const = bool(y_bool[0]) if len(y_bool) else False
            return self
        self.estimator.fit(X, y_bool)
        return self

    def predict(self, X) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const, dtype=bool)
        return self.estimator.predict(X).astype(bool)


class PostureCascade:
    """Coarse-to-fine posture classifier: three binary decisions in sequence.

    upright?  -> if not, sitting?  -> if not, on_ground vs in_bed.

    Each node is a fresh binary estimator from ``node_factory`` trained only on
    the rows that reach it and only on that node's feature columns. This keeps
    every decision an easy 2-way problem on strong features, which overfits far
    less than one 4-way tree on all features.
    """

    def __init__(self, node_factory):
        self._factory = node_factory
        self._colsA = _cols(UPRIGHT_FEATURES)
        self._colsB = _cols(SIT_VS_DOWN_FEATURES)
        self._colsC = _cols(GROUND_VS_BED_FEATURES)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = sorted(set(y))

        self._A = _Binary(self._factory()).fit(X[:, self._colsA], y == "upright")

        rest = y != "upright"
        self._B = _Binary(self._factory()).fit(
            X[rest][:, self._colsB], y[rest] == "sitting"
        )

        down = np.isin(y, ("on_ground", "in_bed"))
        self._C = _Binary(self._factory()).fit(
            X[down][:, self._colsC], y[down] == "on_ground"
        )
        return self

    def predict(self, X) -> np.ndarray:
        pred = np.empty(len(X), dtype=object)
        up = self._A.predict(X[:, self._colsA])
        pred[up] = "upright"

        rest_idx = np.flatnonzero(~up)
        if rest_idx.size:
            is_sit = self._B.predict(X[rest_idx][:, self._colsB])
            pred[rest_idx[is_sit]] = "sitting"
            down_idx = rest_idx[~is_sit]
            if down_idx.size:
                is_ground = self._C.predict(X[down_idx][:, self._colsC])
                pred[down_idx[is_ground]] = "on_ground"
                pred[down_idx[~is_ground]] = "in_bed"
        return pred


class RuleThresholds:
    """A learning-free cascade with thresholds read off the per-class means.

    Here to make "classify by threshold number" concrete and comparable: no
    training beyond memorising the column medians to fill gaps. If a learned
    model cannot beat this, the extra machinery is not earning its keep.
    """

    # Thresholds (metres / degrees / ratios) picked between the class means.
    UPRIGHT_KNEE = 150.0  # upright ~170 deg, others 105-140
    UPRIGHT_EXTENT = 1.30  # upright ~1.58 m tall, others 0.9-1.1
    SIT_ASPECT = 1.30  # sitting box ~1.6 tall, lying ~1.0 flat
    GROUND_TILT = 38.0  # on_ground torso ~46 deg flat, in_bed ~30 deg

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self._median = np.nanmedian(X, axis=0)
        return self

    def _fill(self, X):
        X = X.astype(float, copy=True)
        bad = np.isnan(X)
        if bad.any():
            X[bad] = np.take(self._median, np.where(bad)[1])
        return X

    def predict(self, X) -> np.ndarray:
        X = self._fill(X)
        knee = X[:, _FIDX["knee_angle"]]
        extent = X[:, _FIDX["vertical_extent"]]
        aspect = X[:, _FIDX["bbox_aspect"]]
        tilt = X[:, _FIDX["torso_tilt"]]

        pred = np.empty(len(X), dtype=object)
        up = (knee >= self.UPRIGHT_KNEE) & (extent >= self.UPRIGHT_EXTENT)
        pred[up] = "upright"
        rest = ~up
        sit = rest & (aspect >= self.SIT_ASPECT)
        pred[sit] = "sitting"
        down = rest & ~(aspect >= self.SIT_ASPECT)
        pred[down & (tilt >= self.GROUND_TILT)] = "on_ground"
        pred[down & ~(tilt >= self.GROUND_TILT)] = "in_bed"
        return pred


def _model_zoo(seed: int = 0):
    """Name -> factory. Each factory returns a fresh, unfitted classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    def pipe(clf, scale=False):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("clf", clf))
        return Pipeline(steps)

    def tree_node():
        return pipe(
            DecisionTreeClassifier(
                max_depth=4, class_weight="balanced", random_state=seed
            )
        )

    def logreg_node():
        return pipe(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True
        )

    return {
        # --- flat 4-class models: one decision over all features ---
        "flat_tree": lambda: pipe(
            DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=seed
            )
        ),
        "flat_forest": lambda: pipe(
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced_subsample", random_state=seed
            )
        ),
        "flat_logreg": lambda: pipe(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True
        ),
        "flat_naive_bayes": lambda: pipe(GaussianNB()),
        # --- hierarchical cascades: per-node binary decisions ---
        "cascade_tree": lambda: PostureCascade(tree_node),
        "cascade_logreg": lambda: PostureCascade(logreg_node),
        # --- learning-free threshold baseline ---
        "rule_thresholds": lambda: RuleThresholds(),
    }


@dataclass
class ModelScore:
    name: str
    balanced_accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: list[list[int]]
    per_clip_acc: list[tuple[str, float]]


@dataclass
class CompareResult:
    classes: list[str]
    scores: list[ModelScore]  # ranked best-first by macro_f1
    n_samples: int
    n_groups: int
    tree_rules: str = field(default="")  # export_text of a flat tree on all data


def compare(rows, labels, groups, *, seed: int = 0, holdout: str | None = None) -> CompareResult:
    """Evaluate every model in the zoo on held-out groups, ranked.

    Two protocols, depending on `holdout`:

    * ``holdout=None`` (default) -- leave-one-group-out cross-validation: every
      group takes a turn as the test set and the scores are pooled over all
      folds. The data-efficient estimate; every frame is tested exactly once.
    * ``holdout=<group>`` -- a single fixed split: train on every group *except*
      the named one, test on that one only. This is the plain train/test
      holdout ("train on persons 01+02, test on 03"); scores are computed on
      the held-out group's frames alone.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )
    from sklearn.model_selection import LeaveOneGroupOut

    X = rows_to_matrix(rows)
    y = np.asarray(labels)
    g = np.asarray(groups)
    classes = sorted(set(labels))

    if holdout is not None and holdout not in set(groups):
        raise ValueError(
            "holdout group " + repr(holdout) + " not found; groups are: "
            + ", ".join(sorted(set(map(str, groups))))
        )

    scores: list[ModelScore] = []
    for name, factory in _model_zoo(seed).items():
        if holdout is None:
            # Pooled out-of-fold predictions across every leave-one-group-out fold.
            logo = LeaveOneGroupOut()
            oof = np.empty(len(y), dtype=object)
            per_clip: list[tuple[str, float]] = []
            for tr, te in logo.split(X, y, g):
                model = factory().fit(X[tr], y[tr])
                pred = model.predict(X[te])
                oof[te] = pred
                per_clip.append((g[te][0], float((pred == y[te]).mean())))
            y_eval, pred_eval = y, oof
        else:
            # Single fixed split: train on the rest, score on the held-out group.
            te = np.where(g == holdout)[0]
            tr = np.where(g != holdout)[0]
            model = factory().fit(X[tr], y[tr])
            pred_eval = model.predict(X[te])
            y_eval = y[te]
            per_clip = [(str(holdout), float((pred_eval == y_eval).mean()))]

        scores.append(
            ModelScore(
                name=name,
                balanced_accuracy=float(balanced_accuracy_score(y_eval, pred_eval)),
                macro_f1=float(f1_score(y_eval, pred_eval, labels=classes, average="macro", zero_division=0)),
                per_class_recall={
                    c: float(r)
                    for c, r in zip(
                        classes,
                        recall_score(y_eval, pred_eval, labels=classes, average=None, zero_division=0),
                    )
                },
                confusion=[[int(x) for x in r] for r in confusion_matrix(y_eval, pred_eval, labels=classes)],
                per_clip_acc=sorted(per_clip),
            )
        )

    scores.sort(key=lambda s: s.macro_f1, reverse=True)

    # A flat tree fit on everything, exported as text, so the reader can see the
    # learned thresholds -- the literal answer to "how does it classify".
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier, export_text

    Ximp = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    tree = DecisionTreeClassifier(
        max_depth=3, class_weight="balanced", random_state=seed
    ).fit(Ximp, y)
    rules = export_text(tree, feature_names=list(FEATURES))

    return CompareResult(
        classes=classes,
        scores=scores,
        n_samples=len(y),
        n_groups=len(set(groups)),
        tree_rules=rules,
    )
