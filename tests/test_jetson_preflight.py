"""Portable handoff checks must remain read-only and separate from runtime readiness."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jetson_preflight.py"
SPEC = importlib.util.spec_from_file_location("jetson_preflight", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def kit(tmp_path):
    folder = tmp_path / "handoff"
    folder.mkdir()
    artifact = folder / "source.txt"
    artifact.write_text("synthetic source", encoding="utf-8")
    database = tmp_path / "synthetic.db3"
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT)")
    con.execute("CREATE TABLE messages(topic_id INTEGER,timestamp INTEGER,data BLOB)")
    for tid, name in enumerate(("Color_0/image/data", "Depth_0/image/data"), 1):
        con.execute("INSERT INTO topics VALUES (?,?)", (tid, name))
        con.execute("INSERT INTO messages VALUES (?,1,X'00')", (tid,))
    con.commit()
    con.close()
    manifest = {
        "source_commit": "a" * 40,
        "handoff_artifacts": [{"path": "source.txt", "sha256": checker.sha256(artifact)}],
        "recordings": [{"path": "synthetic.db3", "bytes": database.stat().st_size}],
    }
    (folder / "transfer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, database


def test_file_checks_do_not_mutate_recordings_or_import_ml(kit, monkeypatch, capsys):
    folder, database = kit
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.setattr(checker.importlib, "import_module", lambda name: pytest.fail(name))
    assert checker.main(["--kit", str(folder), "--files-only"]) == 0
    assert "not runtime or clinical validation" in capsys.readouterr().out
    assert checker.sha256(database) == before


def test_tampered_source_is_rejected(kit, capsys):
    folder, _ = kit
    (folder / "source.txt").write_text("changed", encoding="utf-8")
    assert checker.main(["--kit", str(folder), "--files-only"]) == 1
    assert "hash mismatch" in capsys.readouterr().out


def test_recording_size_mismatch_is_rejected(kit, capsys):
    folder, _ = kit
    path = folder / "transfer_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["recordings"][0]["bytes"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert checker.main(["--kit", str(folder), "--files-only"]) == 1
    assert "size mismatch" in capsys.readouterr().out


def test_manifest_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        checker.under(tmp_path, "../outside.txt")
