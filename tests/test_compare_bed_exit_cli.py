"""The collected-session comparator completes its successful reporting path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ahfd.cli import _validate_comparator_output, app


class _Rows:
    labels = ("reclined", "exit")

    def __len__(self):
        return 2

    def compare_args(self):
        return ((), (), ())


def test_successful_comparison_reports_pair_encoded_group_macro(tmp_path, monkeypatch):
    sessions = []
    for index in range(2):
        session = tmp_path / ("ses_" + str(index))
        session.mkdir()
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "session_id": "ses_" + format(index, "016x"),
                    "inputs": {
                        "code_hash": "a" * 40,
                        "config_sha256": "b" * 64,
                        "calibration_sha256": "c" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        sessions.append(session)

    rows = _Rows()
    dataset = SimpleNamespace(
        session_ids=("ses_0000000000000000", "ses_0000000000000001"),
        participant_ids=("sub_0000000000000000", "sub_0000000000000001"),
        sample_hz=10.0,
        phase=rows,
        for_horizon=lambda _horizon: rows,
    )
    score = SimpleNamespace(
        model_kind="logistic",
        accuracy=0.75,
        balanced_accuracy=0.75,
        macro_f1=0.73,
        macro_average_precision=0.80,
        brier_score=0.20,
        per_class_average_precision=(("exit", 0.8), ("reclined", 0.8)),
        per_group_accuracy=(("sub_a", 0.5), ("sub_b", 1.0)),
        positive_class="exit",
        positive_precision=0.8,
        positive_recall=0.7,
        false_positive_rate=0.1,
    )
    result = SimpleNamespace(
        scores=(score,),
        classes=("exit", "reclined"),
        folds=(),
        models={},
    )

    import ahfd.ml.bed_dataset
    import ahfd.ml.temporal

    monkeypatch.setattr(
        ahfd.ml.bed_dataset, "load_bed_sessions", lambda _sessions: dataset
    )
    monkeypatch.setattr(
        ahfd.ml.temporal, "compare_grouped", lambda *_args, **_kwargs: result
    )

    invocation = ["compare-bed-exit", *(str(path) for path in sessions)]
    completed = CliRunner().invoke(app, invocation)

    assert completed.exit_code == 0, completed.output
    assert "macro-F1=0.730" in completed.output
    assert "Next gate:" in completed.output


def test_comparator_artifacts_require_matching_approved_external_ancestor(tmp_path):
    site_id = "site_0123abcd"
    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / ".ahfd-approved-output.json").write_text(
        json.dumps(
            {
                "schema": "ahfd.approved-output",
                "schema_version": 1,
                "site_id": site_id,
                "encrypted_storage_attested": True,
                "purpose": "research_shadow_collection",
            }
        ),
        encoding="utf-8",
    )
    session = approved / "ses_0123456789abcdef"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps({"site_id": site_id}), encoding="utf-8"
    )

    destination = approved / "models" / "cmp_001"
    _validate_comparator_output(destination, [session])
    assert not destination.exists()

    with pytest.raises(Exception, match="custodian-approved external root"):
        _validate_comparator_output(tmp_path / "unapproved" / "cmp_001", [session])
