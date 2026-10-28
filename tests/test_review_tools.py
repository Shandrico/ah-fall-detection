"""Review-video export (raw-capture gated) and the localhost review server."""

from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from ahfd.debug.review_server import make_handler


# --- export-review: refuses before touching anything ----------------------

def test_export_review_refuses_without_all_privacy_switches(tmp_path, monkeypatch):
    from ahfd.debug.annotated_review import export_review
    from ahfd.privacy import PrivacyViolation

    monkeypatch.delenv("AHFD_ALLOW_RAW", raising=False)
    out = tmp_path / "review.mp4"
    with pytest.raises(PrivacyViolation):
        export_review(tmp_path / "clip.db3", out, cfg=None, calibration=tmp_path / "c.yaml",
                      config_flag=True, cli_flag=True)
    assert not out.exists() and not list(tmp_path.iterdir())


def test_export_review_never_overwrites(tmp_path, monkeypatch):
    from ahfd.debug.annotated_review import export_review

    monkeypatch.setenv("AHFD_ALLOW_RAW", "1")
    out = tmp_path / "review.mp4"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        export_review(tmp_path / "clip.db3", out, cfg=None, calibration=tmp_path / "c.yaml",
                      config_flag=True, cli_flag=True)
    assert out.read_bytes() == b"keep me"


def test_hud_draws_time_frame_and_phase():
    import cv2
    import numpy as np

    from ahfd.debug.annotated_review import _draw_hud

    canvas = np.zeros((1080, 1920, 3), np.uint8)
    _draw_hud(cv2, canvas, 75.25, 2257, {3: "EDGE_SITTING"})
    top_right = canvas[:120, 1200:]
    assert top_right.any()            # text was drawn ...
    assert not canvas[300:, :].any()  # ... only in the corner, nothing over the scene


# --- serve-review: Range, listing, refusals --------------------------------

@pytest.fixture
def server(tmp_path):
    (tmp_path / "a.mp4").write_bytes(bytes(range(256)) * 40)  # 10240 bytes
    (tmp_path / "b.partial.mp4").write_bytes(b"x")
    (tmp_path / "secret.txt").write_text("no")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def test_index_lists_only_finished_videos(server):
    resp, body = _get(server, "/")
    assert resp.status == 200
    assert b"a.mp4" in body and b"partial" not in body and b"secret" not in body


def test_full_and_range_requests(server):
    resp, body = _get(server, "/video/a.mp4")
    assert resp.status == 200 and len(body) == 10240
    assert resp.getheader("Accept-Ranges") == "bytes"
    resp, body = _get(server, "/video/a.mp4", {"Range": "bytes=10-19"})
    assert resp.status == 206 and body == bytes(range(10, 20))
    assert resp.getheader("Content-Range") == "bytes 10-19/10240"
    resp, body = _get(server, "/video/a.mp4", {"Range": "bytes=-16"})
    assert resp.status == 206 and len(body) == 16
    resp, _ = _get(server, "/video/a.mp4", {"Range": "bytes=20000-"})
    assert resp.status == 416


@pytest.mark.parametrize("path", [
    "/video/b.partial.mp4", "/video/secret.txt", "/video/..%2Fsecret.txt", "/etc/passwd", "/video/",
])
def test_refuses_anything_but_listed_videos(server, path):
    resp, _ = _get(server, path)
    assert resp.status == 404


# --- review-clip: labels / rules / models video ----------------------------

def test_labelled_review_refuses_without_all_privacy_switches(tmp_path, monkeypatch):
    from ahfd.debug.annotated_review import export_labelled_review
    from ahfd.privacy import PrivacyViolation

    monkeypatch.delenv("AHFD_ALLOW_RAW", raising=False)
    out = tmp_path / "review.mp4"
    with pytest.raises(PrivacyViolation):
        export_labelled_review(tmp_path / "clip.db3", tmp_path / "t.jsonl", out, cfg=None,
                               calibration=tmp_path / "c.yaml", config_flag=True, cli_flag=True)
    assert not list(tmp_path.iterdir())


def test_labelled_review_never_overwrites(tmp_path, monkeypatch):
    from ahfd.debug.annotated_review import export_labelled_review

    monkeypatch.setenv("AHFD_ALLOW_RAW", "1")
    out = tmp_path / "review.mp4"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        export_labelled_review(tmp_path / "clip.db3", tmp_path / "t.jsonl", out, cfg=None,
                               calibration=tmp_path / "c.yaml", config_flag=True, cli_flag=True)
    assert out.read_bytes() == b"keep me"


def test_bed_outlines_project_back_onto_the_bed_polygon():
    import numpy as np

    from ahfd.debug.annotated_review import _bed_outlines
    from ahfd.geometry.calibration import load_calibration

    calib = load_calibration("calib/d435f.yaml")
    beds = [z for z in calib.zones.zones if z.kind == "bed"]
    polys = _bed_outlines(np, calib)
    assert len(polys) == len(beds)
    for bed, poly in zip(beds, polys):
        for (u, v), (x, y) in zip(poly.reshape(-1, 2), bed.polygon):
            back = calib.ground.pixel_to_plane(float(u), float(v), bed.top_m)
            assert back is not None and abs(back[0] - x) < 0.02 and abs(back[1] - y) < 0.02


def test_table_marks_agreement_with_the_label():
    import cv2
    import numpy as np

    from ahfd.debug.annotated_review import _draw_table
    from ahfd.ml.clip_review import FramePrediction

    pred = FramePrediction(t=1.0, track_id=0, rule_phase="RECLINED", model_phase="TORSO_RISING",
                           label_phase="RECLINED")
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    _draw_table(cv2, canvas, pred, 1.0, 10)
    rules_row, model_row = canvas[130:160, 300:660], canvas[164:194, 300:660]
    green = lambda a: ((a[..., 1] > 150) & (a[..., 2] < 120)).sum()
    red = lambda a: ((a[..., 2] > 150) & (a[..., 1] < 120)).sum()
    assert green(rules_row) > red(rules_row)  # rules agree with the label
    assert red(model_row) > green(model_row)  # model does not
    assert not canvas[400:, :].any()          # the table stays in its corner


def test_clip_predictions_follow_labels_and_their_clock(tmp_path):
    import json

    from ahfd.config import load_config
    from ahfd.ml.clip_review import predict_clip

    tracks = tmp_path / "t.jsonl"
    with open("data/tracks_d435f/clip1_01.jsonl") as src:
        tracks.write_text("".join(line for _, line in zip(range(240), src)))
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps({"segments": [{"start_s": 0, "end_s": 3, "phase": "RECLINED"},
                                               {"start_s": 5, "end_s": 60, "phase": "OUT_OF_BED"}]}))
    cfg = load_config("configs/dashboard_jetson.yaml")
    cfg.alert.console, cfg.alert.jsonl_path = False, None

    preds = predict_clip(tracks, "calib/d435f.yaml", cfg, labels=labels)
    frames = sorted(preds.frames.values(), key=lambda f: f.t)
    assert len(frames) > 200 and frames[0].t == 0.0
    assert {f.label_phase for f in frames if f.t < 3} == {"RECLINED"}
    assert {f.label_posture for f in frames if f.t < 3} == {"in_bed"}
    assert all(f.label_phase is None for f in frames if 3.05 < f.t < 4.95)  # a gap stays unlabelled
    assert all(f.model_phase is None for f in frames)  # no model given
    agree, total = preds.agreement()["rule_phase"]
    assert total == sum(f.label_phase is not None for f in frames) and 0 <= agree <= total

    shifted = predict_clip(tracks, "calib/d435f.yaml", cfg, labels=labels, label_t0=preds.t0 - 2.0)
    early = [f for f in shifted.frames.values() if f.t < 1.0]
    assert early and all(f.label_phase is None or f.label_phase == "RECLINED" for f in early)
    assert all(f.label_phase is None for f in shifted.frames.values() if 1.05 < f.t < 2.95)
