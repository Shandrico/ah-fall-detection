"""Showcase features: height-above-floor depth view, depth-mode switch, pages, conservative phase model."""

import json
import threading
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from ahfd.config import Config
from ahfd.dashboard import DashboardServer, DashboardState
from ahfd.dashboard.runner import PipelineRunner
from ahfd.geometry.ground import GroundPlane
from ahfd.ml.live_phase import ConservativePhaseClassifier
from ahfd.types import Intrinsics
from ahfd.viz.depth_render import render_height_map

K = Intrinsics(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
GROUND = GroundPlane(intrinsics=K, height_m=2.6, pitch_deg=30.0)


def plane_depth(height_m: float) -> np.ndarray:
    """Depth (m) of a horizontal surface ``height_m`` above the floor, per pixel."""
    v, u = np.mgrid[0:K.height, 0:K.width].astype(np.float64)
    r2 = np.asarray(GROUND.rotation)[2]
    dot = r2[0] * (u - K.cx) / K.fx + r2[1] * (v - K.cy) / K.fy + r2[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(dot < -1e-3, (height_m - GROUND.height_m) / dot, 0.0)
    d[(d < 0.2) | (d > 8)] = 0
    return d


def test_height_map_colours_by_height_not_distance():
    floor = plane_depth(0.0)
    bed = plane_depth(0.6)
    depth = floor.copy()
    depth[300:420, 200:440] = bed[300:420, 200:440]
    raw = (depth / 0.001).astype(np.uint16)
    img = render_height_map(raw, 0.001, K, GROUND, max_width=640, legend=False, shade=False)
    assert img.shape == (480, 640, 3)
    # Two floor pixels at very different distances get (almost) the same colour.
    near, far = img[470, 100].astype(int), img[250, 100].astype(int)
    assert np.abs(near - far).max() <= 12
    # The raised surface differs from the floor beside it.
    assert np.abs(img[360, 320].astype(int) - img[360, 150].astype(int)).max() > 40


def test_height_map_fills_speckle_but_needs_depth_and_calibration():
    raw = (plane_depth(0.0) / 0.001).astype(np.uint16)
    raw[::7, ::5] = 0  # sensor speckle
    img = render_height_map(raw, 0.001, K, GROUND, max_width=640, legend=False, shade=False)
    assert img[421, 100].any()  # a speckle hole is filled for display
    assert render_height_map(None, 0.001, K, GROUND) is None
    assert render_height_map(raw, 0.001, None, GROUND) is None
    assert render_height_map(raw, 0.001, K, None) is None


def test_height_map_draws_people_with_heights():
    raw = (plane_depth(0.0) / 0.001).astype(np.uint16)
    kp = np.stack([np.full(17, 320.0), np.linspace(120, 400, 17)], axis=1)
    person = SimpleNamespace(keypoints=kp, scores=np.ones(17), heights=np.linspace(1.6, 0.1, 17))
    plain = render_height_map(raw, 0.001, K, GROUND, max_width=640, legend=False)
    drawn = render_height_map(raw, 0.001, K, GROUND, max_width=640, legend=False, people=[person])
    assert not np.array_equal(plain, drawn)


def test_depth_mode_switch():
    state = DashboardState()
    assert state.depth_mode() == "height"
    assert state.set_depth_mode("distance") and state.depth_mode() == "distance"
    assert not state.set_depth_mode("rainbow") and state.depth_mode() == "distance"
    assert state.snapshot()["runtime"]["depth_mode"] == "distance"


def test_runner_renders_height_map_when_calibrated(monkeypatch):
    state = DashboardState()
    state.set_rgb(True)
    state.publish_status(0, "running")
    state.depth_viewer(+1)
    runner = PipelineRunner("rs://", Config(), None, state, True)
    frame = SimpleNamespace(depth_raw=(plane_depth(0.0) / 0.001).astype(np.uint16), depth_scale=0.001, intrinsics=K)
    runner._publish_depth_preview(frame, True, None, GROUND)
    jpeg, _ = state.latest_depth_frame()
    assert jpeg and jpeg[:2] == b"\xff\xd8"


def _serve(state):
    server = DashboardServer(state, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:%d" % server.server_address[1]


def test_showcase_page_and_depth_mode_endpoint():
    state = DashboardState()
    server, base = _serve(state)
    try:
        page = urllib.request.urlopen(base + "/showcase", timeout=5).read().decode()
        assert "How it works, live" in page and "/depth.mjpg" in page
        ward = urllib.request.urlopen(base + "/ward", timeout=5).read().decode()
        assert "Height map" in ward and "/showcase" in ward
        req = urllib.request.Request(base + "/api/depth-mode", data=json.dumps({"mode": "distance"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert json.load(urllib.request.urlopen(req, timeout=5))["mode"] == "distance"
        bad = urllib.request.Request(base + "/api/depth-mode", data=b'{"mode":"x"}',
                                     headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad, timeout=5)
    finally:
        server.shutdown()
        server.server_close()


class _Fixed:
    classes_ = np.array(["EDGE_SITTING", "OUT_OF_BED", "RECLINED", "UPRIGHT_IN_BED"])

    def __init__(self, rows):
        self.rows = np.array(rows, float)

    def predict_proba(self, X):
        return self.rows[: len(X)]


def test_conservative_classifier_needs_confidence_for_getting_up():
    model = ConservativePhaseClassifier(_Fixed([[0.45, 0.05, 0.30, 0.20], [0.80, 0.05, 0.10, 0.05]]), 0.5)
    assert list(model.predict(np.zeros((2, 3)))) == ["RECLINED", "EDGE_SITTING"]
    assert list(ConservativePhaseClassifier(_Fixed([[0.45, 0.05, 0.30, 0.20]]), 0.0).predict(np.zeros((1, 3)))) == ["EDGE_SITTING"]
