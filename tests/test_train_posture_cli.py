"""Training CLI preserves recording groups and labels evaluation honestly."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ahfd.cli import app
from ahfd.geometry import calibration
from ahfd.ml import posture


@pytest.mark.parametrize("held_out", [True, False])
def test_cli_passes_groups_and_reports_split_kind(monkeypatch, held_out):
    rows = [{"h_torso": 1.0}, {"h_torso": 0.5}]
    labels = ["upright", "sitting"]
    groups = ["fall_standing_03", "neg_fastsit_03"]
    monkeypatch.setattr(calibration, "load_calibration", lambda path: object())
    monkeypatch.setattr(
        posture,
        "build_dataset",
        lambda *args: (rows, labels, groups, [("fall_standing_03", 2)], []),
    )
    calls = []

    def fake_train(actual_rows, actual_labels, **kwargs):
        calls.append((actual_rows, actual_labels, kwargs))
        return SimpleNamespace(
            split_done=held_out,
            split_unit="person" if held_out else "none",
            train_groups=["person_03"],
            test_groups=["person_04"] if held_out else [],
            split_reason="need at least two independent groups",
            n_train=1 if held_out else 2,
            n_test=1 if held_out else 0,
            accuracy=0.75,
            classes=["sitting", "upright"],
            confusion=[[1, 0], [0, 1]],
            importances=[],
            separability=[],
            class_means={},
            report="synthetic report",
        )

    monkeypatch.setattr(posture, "train", fake_train)
    result = CliRunner().invoke(app, ["train-posture", "--calibration", "unused.yaml"])

    assert result.exit_code == 0, result.output
    assert calls == [(rows, labels, {"groups": groups, "max_depth": 5})]
    if held_out:
        assert "group-held-out split (person)" in result.output
        assert "held-out groups: person_04" in result.output
        assert "TEST accuracy: 0.750" in result.output
        assert "TRAINING accuracy:" not in result.output
    else:
        assert "need at least two independent groups" in result.output
        assert "TRAINING accuracy: 0.750" in result.output
        assert "TEST accuracy:" not in result.output
