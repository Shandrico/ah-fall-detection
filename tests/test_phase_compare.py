"""Pilot phase comparison: holdout stays locked, scoring is group-disjoint."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from ahfd.ml.phase_compare import PHASES, PhaseRows, compare, dev_takes, lobo_scores


def test_dev_takes_never_include_block_e():
    stems = {stem for stem, _ in dev_takes()}
    assert stems == {"clip1_%02d" % i for i in range(1, 9)}


def test_lobo_never_trains_on_the_held_out_block():
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(set(X[:, 0]))
            return self

        def predict(self, X):
            return np.array(["RECLINED"] * len(X))

    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array(["RECLINED"] * 4)
    groups = np.array(["block_A", "block_A", "block_B", "block_B"])
    lobo_scores(X, y, groups, Spy)
    assert seen == [{1.0}, {0.0}]  # each fold trained only on the other block


def test_separable_phases_score_well():
    rng = np.random.default_rng(0)
    labels, frame, groups = [], [], []
    for b in "ABCD":
        for i, phase in enumerate(PHASES):
            for _ in range(20):
                labels.append(phase)
                frame.append([i + rng.normal(0, 0.05)])
                groups.append("block_" + b)
    frame = np.array(frame)
    rows = PhaseRows(frame, frame.copy(), np.array(labels), np.array(groups))
    result = compare(rows)
    assert set(result) == {"%s / %s" % (s, m) for s in ("per-frame", "over-time", "both") for m in ("tree", "logistic")}
    assert result["per-frame / tree"]["macro_f1"] > 0.95
