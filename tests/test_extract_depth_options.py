"""Recorded depth extraction must preserve an explicitly requested range."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ahfd.cli import app
from ahfd.capture import realsense
from ahfd import pose


@pytest.mark.parametrize("source", ["bag://test.db3", "rs://"])
@pytest.mark.parametrize("extra,expected", [([], (6.0, 2)), (["--max-range", "10", "--smooth", "3"], (10.0, 3))])
def test_depth_options_reach_source(monkeypatch, tmp_path, source, extra, expected):
    calls = []

    class FakeSource:
        meta = SimpleNamespace(width=1920, height=1080, fps=30)

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def __iter__(self):
            return iter(())

        def close(self):
            pass

    monkeypatch.setattr(realsense, "BagSource", FakeSource)
    monkeypatch.setattr(realsense, "RealSenseSource", FakeSource)
    monkeypatch.setattr(pose, "build_estimator", lambda cfg: SimpleNamespace(name="fake"))
    result = CliRunner().invoke(app, ["extract", source, str(tmp_path / "tracks.jsonl"), "--depth", *extra])
    assert result.exit_code == 0, result.output
    # rs:// also carries the URI's own options (model=/serial=, emitter, ...) since
    # extract parses it like every other command; the requested depth settings win.
    wanted = {"with_depth": True, "max_range_m": expected[0], "spatial_magnitude": expected[1]}
    assert {k: calls[0][1].get(k) for k in wanted} == wanted


@pytest.mark.parametrize("extra", [["--max-range", "0"], ["--max-range", "nan"], ["--max-range", "inf"], ["--smooth", "0"], ["--smooth", "6"]])
def test_invalid_depth_options_refused(tmp_path, extra):
    result = CliRunner().invoke(app, ["extract", "bag://missing.db3", str(tmp_path / "tracks.jsonl"), "--depth", *extra])
    assert result.exit_code != 0
    assert not (tmp_path / "tracks.jsonl").exists()
