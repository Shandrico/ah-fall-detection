"""`ahfd copy-zones`: share bed zones across calibrations of one camera mount."""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from ahfd.cli import app

runner = CliRunner()

_ZONE = {
    "name": "bed_1",
    "kind": "bed",
    "top_m": 0.5,
    "risk_level": "high",
    "polygon": [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
}


def _calib(path, height_m, zones):
    path.write_text(
        yaml.safe_dump({"camera": {"height_m": height_m}, "zones": zones}),
        encoding="utf-8",
    )


def test_copies_zones_to_same_mount(tmp_path):
    src, tgt = tmp_path / "colour.yaml", tmp_path / "ir.yaml"
    _calib(src, 2.5, [_ZONE])
    _calib(tgt, 2.5, [])

    result = runner.invoke(app, ["copy-zones", "--from", str(src), "--to", str(tgt)])
    assert result.exit_code == 0, result.output

    out = yaml.safe_load(tgt.read_text(encoding="utf-8"))
    assert len(out["zones"]) == 1
    assert out["zones"][0]["name"] == "bed_1"
    assert out["zones"][0]["polygon"][0] == [0.0, 1.0]  # metres preserved


def test_replace_vs_append(tmp_path):
    src, tgt = tmp_path / "a.yaml", tmp_path / "b.yaml"
    other = dict(_ZONE, name="bed_2")
    _calib(src, 2.5, [_ZONE])
    _calib(tgt, 2.5, [other])

    runner.invoke(app, ["copy-zones", "--from", str(src), "--to", str(tgt)])
    assert [z["name"] for z in yaml.safe_load(tgt.read_text())["zones"]] == ["bed_1"]

    _calib(tgt, 2.5, [other])
    runner.invoke(app, ["copy-zones", "--from", str(src), "--to", str(tgt), "--append"])
    assert [z["name"] for z in yaml.safe_load(tgt.read_text())["zones"]] == ["bed_2", "bed_1"]


def test_skips_different_mount_height(tmp_path):
    src, tgt = tmp_path / "d435i.yaml", tmp_path / "webcam.yaml"
    _calib(src, 2.5, [_ZONE])
    _calib(tgt, 1.1, [])  # a different camera position entirely

    result = runner.invoke(app, ["copy-zones", "--from", str(src), "--to", str(tgt)])
    assert result.exit_code == 0
    assert "SKIP" in result.output
    assert yaml.safe_load(tgt.read_text())["zones"] == []  # untouched


def test_errors_when_source_has_no_zones(tmp_path):
    src, tgt = tmp_path / "a.yaml", tmp_path / "b.yaml"
    _calib(src, 2.5, [])
    _calib(tgt, 2.5, [])
    result = runner.invoke(app, ["copy-zones", "--from", str(src), "--to", str(tgt)])
    assert result.exit_code != 0
