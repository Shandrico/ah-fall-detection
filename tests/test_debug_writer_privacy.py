"""Every raw/debug writer enforces consent at its own API boundary."""

from __future__ import annotations

import pytest

from ahfd.debug.bag_writer import record_bag
from ahfd.debug.color_export import export_color
from ahfd.debug.frame_cache import load_cache, save_cache
from ahfd.privacy import ENV_VAR, PrivacyViolation


@pytest.mark.parametrize(
    "writer, filename",
    [
        (lambda path: record_bag(path), "raw/session.bag"),
        (lambda path: export_color("source.bag", path), "raw/session.mp4"),
        (lambda path: save_cache("session", {"pixels": b"raw"}, root=path), "raw/cache"),
    ],
)
def test_raw_debug_writers_refuse_by_default_before_creating_paths(
    tmp_path, monkeypatch, writer, filename
):
    monkeypatch.delenv(ENV_VAR, raising=False)
    target = tmp_path / filename

    with pytest.raises(PrivacyViolation):
        writer(target)

    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "writer",
    [
        lambda path: record_bag(path, config_flag=True, cli_flag=False),
        lambda path: export_color(
            "source.bag", path, config_flag=True, cli_flag=False
        ),
        lambda path: save_cache(
            "session",
            {"pixels": b"raw"},
            root=path,
            config_flag=True,
            cli_flag=False,
        ),
    ],
)
def test_environment_and_config_are_not_enough(tmp_path, monkeypatch, writer):
    monkeypatch.setenv(ENV_VAR, "1")

    with pytest.raises(PrivacyViolation):
        writer(tmp_path / "raw")

    assert not (tmp_path / "raw").exists()


def test_frame_cache_round_trip_with_all_three_switches(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    payload = {"cache": [(b"rgb-jpeg", b"depth-jpeg")], "stats": {"n": 1}}

    path = save_cache(
        "session_01",
        payload,
        root=tmp_path,
        config_flag=True,
        cli_flag=True,
    )

    assert path == tmp_path / "session_01.pkl"
    assert load_cache("session_01", root=tmp_path) == payload
