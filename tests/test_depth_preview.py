"""Live depth rendering, privacy, freshness and MJPEG transport regressions."""

import threading
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ahfd.config import Config
from ahfd.dashboard import DashboardServer, DashboardState
from ahfd.dashboard.runner import PipelineRunner
from ahfd.viz.depth_render import render_depth


def live_state():
    state = DashboardState()
    state.set_rgb(True)
    state.publish_status(0, "running")
    return state


def test_depth_fixed_scale_and_invalid_pixels():
    depth = np.array([[0, 100, 300, 1000, 6000, 6001, np.nan]], np.float32)
    result = render_depth(depth, 0.001)
    assert not result[0, [0, 1, 5, 6]].any()
    assert result[0, [2, 3, 4]].any(axis=1).all()
    assert np.array_equal(result[0, 3], render_depth(np.array([[1000]]), 0.001)[0, 0])
    assert not np.array_equal(result[0, 2], result[0, 4])


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_bad_scale_is_not_a_depth_image(scale):
    assert render_depth(np.ones((2, 2)), scale) is None


def test_no_depth_and_preview_size():
    assert render_depth(None, 0.001) is None
    assert render_depth(np.zeros((0, 0)), 0.001) is None
    assert render_depth(np.ones((1080, 1920)), 0.001).shape == (360, 640, 3)


def test_long_range_keeps_real_depth_beyond_six_metres():
    depth = np.array([[0, 300, 6000, 8000, 10000, 10001]], np.uint16)
    short = render_depth(depth, 0.001)
    long = render_depth(depth, 0.001, max_depth_m=10.0)
    assert not short[0, 3:].any()
    assert long[0, 1:5].any(axis=1).all()
    assert not long[0, [0, 5]].any()
    assert not np.array_equal(long[0, 2], long[0, 3])


@pytest.mark.parametrize('near,far', [(1, 1), (2, 1), (-1, 6), (0, float('nan'))])
def test_invalid_depth_preview_range_is_rejected(near, far):
    with pytest.raises(ValueError):
        render_depth(np.ones((2, 2)), 0.001, min_depth_m=near, max_depth_m=far)


def test_runner_uses_capture_range_for_preview_and_legend():
    state = live_state()
    runner = PipelineRunner(
        'rs://?depth=1&emitter=1&max_laser=1&max_range=10&spatial=3',
        Config(), None, state, True,
    )
    runner._configure_depth_preview()
    rt = state.snapshot()['runtime']
    assert rt['depth_min_m'] == 0.3
    assert rt['depth_max_m'] == 10.0
    assert rt['depth_spatial_passes'] == 3
    assert rt['depth_preview_hole_filled'] is False
    frame = SimpleNamespace(depth_raw=np.full((8, 8), 8000, np.uint16), depth_scale=0.001)
    runner._publish_depth_preview(frame, True)
    jpeg, _ = state.latest_depth_frame()
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).any()


def test_preview_needs_explicit_live_imagery_opt_in():
    state = DashboardState()
    state.publish_status(0, 'running')
    state.publish_depth_frame(b'private')
    assert state.latest_depth_frame()[0] != b'private'


def test_rgb_off_clears_depth_and_rejects_inflight_publication():
    state = live_state()
    state.publish_depth_frame(b'depth')
    _, before = state.latest_depth_frame()
    state.set_rgb(False)
    state.publish_depth_frame(b'inflight')
    safe, after = state.latest_depth_frame()
    assert after != before
    image = cv2.imdecode(np.frombuffer(safe, np.uint8), cv2.IMREAD_COLOR)
    assert not image.any()
    state.set_rgb(True)
    assert state.latest_depth_frame()[0] == safe  # no old image resurrection


def test_generation_clears_depth_and_rejects_retired_pipeline():
    state = live_state()
    state.publish_depth_frame(b'old')
    gen = state.begin_generation()
    state.publish_status(gen, 'running')
    state.publish_depth_frame(b'retired', gen=gen - 1)
    assert state.latest_depth_frame()[0] not in (b'old', b'retired')
    state.publish_depth_frame(b'new', gen=gen)
    assert state.latest_depth_frame()[0] == b'new'


def test_stale_preview_replaces_image_with_new_token(monkeypatch):
    state = live_state()
    monkeypatch.setattr('ahfd.dashboard.state.time.time', lambda: 10.0)
    state.publish_depth_frame(b'depth')
    _, before = state.latest_depth_frame()
    monkeypatch.setattr('ahfd.dashboard.state.time.time', lambda: 14.0)
    safe, after = state.latest_depth_frame()
    assert safe != b'depth' and after != before


@pytest.mark.parametrize('status', ['error', 'stopped', 'ended', 'switching'])
def test_not_running_does_not_show_last_depth(status):
    state = live_state()
    state.publish_depth_frame(b'depth')
    state.publish_status(0, status)
    assert state.latest_depth_frame()[0] != b'depth'


def test_runner_encodes_only_opted_in_and_throttles(monkeypatch):
    state = live_state()
    runner = PipelineRunner('rs://', Config(), None, state, True)
    frame = SimpleNamespace(depth_raw=np.full((8, 8), 1000, np.uint16), depth_scale=0.001)
    monkeypatch.setattr('ahfd.dashboard.runner.time.monotonic', lambda: 1.0)
    runner._publish_depth_preview(frame, False)
    assert state._depth_jpeg is None
    runner._publish_depth_preview(frame, True)
    jpeg, first = state.latest_depth_frame()
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).any()
    runner._publish_depth_preview(frame, True)
    assert state.latest_depth_frame()[1] == first
    monkeypatch.setattr('ahfd.dashboard.runner.time.monotonic', lambda: 1.2)
    frame.depth_raw = None
    runner._publish_depth_preview(frame, True)
    assert state.latest_depth_frame()[0] != jpeg


def test_depth_http_stream_and_immediate_privacy_replacement():
    state = live_state()
    jpeg = b'\xff\xd8' + b'preview' * 100 + b'\xff\xd9'
    state.publish_depth_frame(jpeg)
    server = DashboardServer(state, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def part(response):
        assert response.readline() == b'--ahfdframe\r\n'
        assert response.readline() == b'Content-Type: image/jpeg\r\n'
        length = int(response.readline().split(b':', 1)[1])
        assert response.readline() == b'\r\n'
        result = response.read(length)
        assert response.read(2) == b'\r\n'
        return result

    try:
        url = f'http://127.0.0.1:{server.server_port}/depth.mjpg'
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.headers['Cache-Control'] == 'no-store'
            assert part(response) == jpeg
            state.set_rgb(False)
            assert part(response) != jpeg
    finally:
        server.shutdown()
        server.server_close()


def test_clamp_paints_out_of_range_depth_in_end_colours():
    depth = np.array([[0, 500, 2000, 9000]], np.uint16)
    plain = render_depth(depth, 0.001, min_depth_m=1.0, max_depth_m=3.0)
    clamped = render_depth(depth, 0.001, min_depth_m=1.0, max_depth_m=3.0, clamp=True)
    assert not plain[0, 1].any() and not plain[0, 3].any()
    assert not clamped[0, 0].any()  # a hole is still black
    assert clamped[0, 1].any() and clamped[0, 3].any()


def test_shading_darkens_slopes_but_keeps_holes_black():
    yy, xx = np.mgrid[0:120, 0:160]
    flat = np.full((120, 160), 2000, np.uint16)
    ramp = (2000 + xx * 10).astype(np.uint16)
    ramp[:10, :10] = 0
    plain = render_depth(flat, 0.001, max_width=160)
    shaded_flat = render_depth(flat, 0.001, max_width=160, shade=True)
    shaded_ramp = render_depth(ramp, 0.001, max_width=160, shade=True)
    assert shaded_flat.shape == plain.shape
    assert not shaded_ramp[:10, :10].any()
    # A surface tilted away from the light is drawn darker than a flat one.
    assert shaded_ramp[60:, 40:].mean() < render_depth(ramp, 0.001, max_width=160)[60:, 40:].mean()


def test_legend_draws_a_bar_on_the_right():
    depth = np.full((360, 640), 3000, np.uint16)
    without = render_depth(depth, 0.001)
    with_legend = render_depth(depth, 0.001, legend=True)
    assert not np.array_equal(without[:, -60:], with_legend[:, -60:])
    assert np.array_equal(without[:, :400], with_legend[:, :400])


def test_scene_range_fits_the_occupied_band():
    from ahfd.viz.depth_render import scene_depth_range

    rng = np.random.default_rng(0)
    depth = rng.uniform(2200, 4800, size=(480, 640)).astype(np.uint16)
    lo, hi = scene_depth_range(depth, 0.001)
    assert 2.0 <= lo <= 2.25 and 4.75 <= hi <= 5.0
    assert scene_depth_range(np.zeros((480, 640), np.uint16), 0.001) is None


def test_switch_warning_clears_once_the_new_source_runs():
    state = DashboardState()
    gen = state.generation
    state.publish_status(gen, "switching", warning="the previous camera has not released yet")
    assert state.snapshot()["runtime"].get("warning")
    state.publish_status(gen, "running")
    assert "warning" not in state.snapshot()["runtime"]
