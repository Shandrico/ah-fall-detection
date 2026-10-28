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


def test_color_export_keeps_recording_time_when_frames_were_dropped(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    import ahfd.capture.realsense as realsense
    from ahfd.capture.base import Frame, SourceMeta
    from ahfd.debug.color_export import export_color

    times = [0.0, 0.1, 0.2, 0.7, 0.8, 2.0]  # a 0.5 s and a 1.2 s hole, as in a write-limited recording

    class FakeBag:
        def __init__(self, path, with_depth=False):
            pass

        @property
        def meta(self):
            return SourceMeta(uri="bag://x", width=64, height=48, fps=30.0)

        def __iter__(self):
            for i, t in enumerate(times):
                yield Frame(index=i, t=100.0 + t, bgr=np.full((48, 64, 3), i * 40, np.uint8))

        def close(self):
            pass

    monkeypatch.setattr(realsense, "BagSource", FakeBag)
    monkeypatch.setenv("AHFD_ALLOW_RAW", "1")
    out = tmp_path / "clip.mp4"
    assert export_color("clip.db3", out, config_flag=True, cli_flag=True) == len(times)
    cap = cv2.VideoCapture(str(out))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert fps == 30.0
    assert abs(n / fps - 2.0) <= 2 / fps  # video time is recording time (2 s), not 6 frames
