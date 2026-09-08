"""The staged-session recorder must obey the privacy gate.

This is the one module allowed to write imagery, so its gate is the load-
bearing one. These tests confirm a recorder cannot even be *constructed* unless
all three consent switches agree -- the file is never opened otherwise.
"""

from __future__ import annotations

import pytest

from ahfd.privacy import ENV_VAR, PrivacyViolation


def _make(tmp_path, *, config_flag, cli_flag):
    from ahfd.debug import RawRecorder

    return RawRecorder(
        tmp_path / "session.mp4",
        width=64,
        height=48,
        fps=15.0,
        config_flag=config_flag,
        cli_flag=cli_flag,
    )


def test_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(PrivacyViolation):
        _make(tmp_path, config_flag=False, cli_flag=False)


def test_refused_with_only_two_switches(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    with pytest.raises(PrivacyViolation):
        _make(tmp_path, config_flag=True, cli_flag=False)


def test_no_file_created_when_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(PrivacyViolation):
        _make(tmp_path, config_flag=True, cli_flag=True)
    # The refusal happens before any file is opened.
    assert not (tmp_path / "session.mp4").exists()


def test_records_with_full_consent(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv(ENV_VAR, "1")
    rec = _make(tmp_path, config_flag=True, cli_flag=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    rec.write(frame)
    rec.write(frame)
    rec.close()
    assert rec.count == 2
    assert (tmp_path / "session.mp4").exists()
