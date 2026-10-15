"""Frozen splits, conservative duration accounting and training leakage guards."""
from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from ahfd.cli import app
from ahfd.collect.protocol import (
    ProtocolProgress, assignment, guard_development_inputs, load_plan, new_plan,
    plan_sha256, validate_assignment, validate_plan, validate_progress,
)
from ahfd.ml.temporal import SampleMetadata

SITE = "site_0123abcd"
SUBJECT = "sub_0123456789abcdef"


@pytest.fixture
def plan():
    return new_plan(SITE, SUBJECT, {"bed_a": "medium", "bed_b": "high"}, 40)


def write_manifest(tmp_path, protocol, name="session"):
    directory = tmp_path / name
    directory.mkdir()
    value = {"site_id": SITE, "participant_id": SUBJECT}
    if protocol is not None:
        value["collection_protocol"] = protocol
    (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return directory


@pytest.mark.parametrize("minutes,seconds", [(40, 240), (60, 360)])
def test_frozen_plan_balances_beds_and_partitions(minutes, seconds):
    plan = new_plan(SITE, SUBJECT, {"bed_a": "low", "bed_b": "high"}, minutes)
    assignments = [assignment(plan, b, bed) for b in "ABCDE" for bed in plan["beds"]]
    assert sum(p["target_usable_s"] for p in assignments) == minutes * 60
    assert sum(p["partition"] == "development" for p in assignments) == 8
    assert {p["target_usable_s"] for p in assignments} == {seconds}
    assert assignment(plan, "E", "bed_b")["partition"] == "validation"


@pytest.mark.parametrize("field,value", [
    ("total_minutes", True), ("total_minutes", 50),
    ("participant_id", "person name"), ("site_id", "ward number"),
    ("beds", {"bed_a": "high"}),
    ("beds", {"bed_a": "urgent", "bed_b": "high"}),
    ("scope", "patient_validation"), ("capture_policy", "raw"),
    ("blocks", {b: "development" for b in "ABCDE"}),
])
def test_invalid_plan_rejected(plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_plan_hash_is_canonical_and_assignment_detached(plan, tmp_path):
    reordered = dict(reversed(list(plan.items())))
    assert plan_sha256(plan) == plan_sha256(reordered)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8-sig")
    assert load_plan(path) == plan
    protocol = assignment(plan, "A", "bed_a")
    plan["beds"]["bed_a"] = "low"
    assert protocol["plan"]["beds"]["bed_a"] == "medium"
    assert protocol["plan_sha256"] != plan_sha256(plan)


@pytest.mark.parametrize("field,value", [
    ("partition", "development"), ("plan_sha256", "0" * 64),
    ("target_usable_s", 1), ("block", "F"), ("target_bed", "bed_c"),
])
def test_assignment_cannot_relabel_validation_or_duration(plan, field, value):
    protocol = assignment(plan, "E", "bed_a")
    protocol[field] = value
    with pytest.raises(ValueError):
        validate_assignment(protocol, site_id=SITE, participant_id=SUBJECT)


def test_assignment_rejects_other_subject_and_unknown_fields(plan):
    protocol = assignment(plan, "A", "bed_a")
    with pytest.raises(ValueError, match="site/participant"):
        validate_assignment(protocol, site_id=SITE, participant_id="sub_0000000000000000")
    protocol["note"] = "not part of schema"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_assignment(protocol, site_id=SITE, participant_id=SUBJECT)


def test_progress_never_backfills_warmup_gaps_or_reassociation():
    progress = ProtocolProgress(1.0)
    for t in (0.0, 0.1):
        progress.observe(t, eligible=True, association=(1, 1))
    assert progress.usable_s == 0  # Bed unverified.
    progress.target_bed_observed = True
    progress.observe(0.2, eligible=True, association=(1, 1))
    progress.observe(0.3, eligible=True, association=(1, 1))
    assert progress.usable_s == pytest.approx(0.1)
    progress.observe(0.4, eligible=False, association=(1, 1))
    progress.observe(0.5, eligible=True, association=(1, 1))
    progress.observe(1.5, eligible=True, association=(1, 1))
    progress.observe(1.6, eligible=True, association=(1, 2))
    assert progress.usable_s == pytest.approx(0.1)
    progress.observe(1.7, eligible=True, association=(1, 2))
    assert progress.usable_s == pytest.approx(0.2)
    progress.reset_binding()
    progress.observe(1.8, eligible=True, association=(2, 3))
    assert not progress.target_bed_observed
    assert progress.usable_s == pytest.approx(0.2)


def test_progress_ready_and_invalid_timestamps():
    progress = ProtocolProgress(0.3, target_bed_observed=True)
    for t in (0, 0.1, 0.2, 0.3):
        progress.observe(t, eligible=True, association=(1, 1))
    assert progress.ready
    assert validate_progress(progress.snapshot(), 0.3, complete=True)["usable_s"] == 0.3
    for t in (-1, 0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            progress.observe(t, eligible=True, association=(1, 1))
    for duration in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ProtocolProgress(duration)
    with pytest.raises(ValueError):
        validate_progress({"usable_s": True, "target_bed_observed": True}, 1)


def test_guard_rejects_e_before_loading_any_observation(plan, tmp_path, monkeypatch):
    directory = write_manifest(tmp_path, assignment(plan, "E", "bed_a"))
    import ahfd.ml.bed_dataset

    def forbidden(*args, **kwargs):
        pytest.fail("validation observations must not be opened by training")

    monkeypatch.setattr(ahfd.ml.bed_dataset, "load_bed_sessions", forbidden)
    result = CliRunner().invoke(app, ["compare-bed-exit", str(directory), "--group-by", "block"])
    assert result.exit_code != 0
    assert "Block E is locked validation" in result.output


def test_guard_requires_one_plan_block_grouping_and_no_legacy(plan, tmp_path):
    a = write_manifest(tmp_path, assignment(plan, "A", "bed_a"), "a")
    b = write_manifest(tmp_path, assignment(plan, "A", "bed_b"), "b")
    guard_development_inputs([a, b], "block")
    with pytest.raises(ValueError, match="group-by block"):
        guard_development_inputs([a, b], "session")
    legacy = write_manifest(tmp_path, None, "legacy")
    guard_development_inputs([legacy], "session")
    with pytest.raises(ValueError, match="legacy"):
        guard_development_inputs([a, legacy], "block")
    changed = copy.deepcopy(plan)
    changed["beds"]["bed_a"] = "low"
    c = write_manifest(tmp_path, assignment(changed, "B", "bed_a"), "changed")
    with pytest.raises(ValueError, match="unchanged collection plan"):
        guard_development_inputs([a, c], "block")


def test_block_group_ignores_session_and_keeps_both_beds_together():
    a = SampleMetadata(SUBJECT, "ses_a", t=0.0, block_id="plan_x::A")
    b = SampleMetadata(SUBJECT, "ses_b", t=1.0, block_id="plan_x::A")
    c = SampleMetadata(SUBJECT, "ses_c", t=1.0, block_id="plan_x::B")
    assert a.group("block") == b.group("block") != c.group("block")
    with pytest.raises(ValueError, match="block"):
        SampleMetadata(SUBJECT, "ses_legacy", t=0.0).group("block")


def test_comparator_holds_out_both_beds_of_each_block():
    from ahfd.ml.temporal import compare_grouped

    rows, labels, metadata = [], [], []
    for block in "ABCD":
        for bed in ("bed_a", "bed_b"):
            for label, value in (("rest", 0.0), ("exit", 1.0)):
                rows.append({"h_torso__now": value})
                labels.append(label)
                metadata.append(SampleMetadata(
                    SUBJECT, "ses_" + block + bed, block_id="plan_x::" + block,
                ))
    result = compare_grouped(
        rows, labels, metadata, group_by="block", model_kinds=("tree",),
        feature_names=("h_torso__now",), min_samples_leaf=1,
    )
    assert len(result.folds) == 4
    for fold in result.folds:
        assert len(fold.train_groups) == 3
        assert len(fold.test_groups) == 1
        assert set(fold.train_groups).isdisjoint(fold.test_groups)
    assert result.models["tree"].metadata.group_by == "block"


def test_plan_cli_only_writes_new_plan_to_approved_external_root(tmp_path):
    marker = {
        "schema": "ahfd.approved-output", "schema_version": 1, "site_id": SITE,
        "encrypted_storage_attested": True, "purpose": "research_shadow_collection",
    }
    args = ["collection-plan", "--out-root", str(tmp_path), "--site-id", SITE,
            "--participant-id", SUBJECT, "--bed-a", "bed_a", "--bed-b", "bed_b"]
    rejected = CliRunner().invoke(app, args)
    assert rejected.exit_code != 0
    assert not list(tmp_path.glob("plan_*.json"))
    (tmp_path / ".ahfd-approved-output.json").write_text(json.dumps(marker), encoding="utf-8")
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "240 labelled available seconds" in result.output
    paths = list(tmp_path.glob("plan_*.json"))
    assert len(paths) == 2  # Never overwrite the first plan.
    assert load_plan(paths[0])["participant_id"] == SUBJECT


def test_partial_protocol_cli_fails_before_config_or_camera(tmp_path):
    result = CliRunner().invoke(app, [
        "collect", "--out-root", str(tmp_path), "--site-id", SITE,
        "--config", str(tmp_path / "does-not-exist.yaml"), "--block", "A",
    ])
    assert result.exit_code != 0
    assert "must be supplied together" in result.output
