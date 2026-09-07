"""The runtime raw-capture gate.

Three independent switches must agree before any RGB reaches disk. These tests
pin that down, because the failure mode being defended against is a flag left
enabled after a debugging session.
"""

from __future__ import annotations

import pytest

from ahfd.privacy import ENV_VAR, PrivacyViolation, raw_capture_allowed, require_raw_capture


def test_default_is_denied(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert raw_capture_allowed(config_flag=False, cli_flag=False) is False


@pytest.mark.parametrize(
    "config_flag,cli_flag,env",
    [
        (True, True, None),  # env missing
        (True, False, "1"),  # no cli acknowledgement
        (False, True, "1"),  # config says no
        (True, True, "0"),  # env explicitly off
        (True, True, "true"),  # only the exact string "1" counts
    ],
)
def test_any_single_switch_missing_denies(monkeypatch, config_flag, cli_flag, env):
    if env is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, env)
    assert raw_capture_allowed(config_flag=config_flag, cli_flag=cli_flag) is False


def test_all_three_switches_allow(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    assert raw_capture_allowed(config_flag=True, cli_flag=True) is True


def test_require_names_what_is_missing(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(PrivacyViolation) as excinfo:
        require_raw_capture(config_flag=False, cli_flag=False)

    message = str(excinfo.value)
    assert "allow_raw_capture" in message
    assert "--i-understand-raw-capture" in message
    assert ENV_VAR in message


def test_require_passes_when_authorised(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    require_raw_capture(config_flag=True, cli_flag=True)  # must not raise
