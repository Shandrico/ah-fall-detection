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

    green = lambda a: ((a[..., 1] > 150) & (a[..., 2] < 120)).sum()
    red = lambda a: ((a[..., 2] > 150) & (a[..., 1] < 120)).sum()
    wrong = FramePrediction(t=1.0, track_id=0, rule_phase="RECLINED", model_phase="TORSO_RISING",
                            label_phase="RECLINED")
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    _draw_table(cv2, canvas, wrong, 1.0, 10)
    system_row = canvas[130:160, 300:660]  # the second row is the system (the trained model)
    assert red(system_row) > green(system_row)  # the model disagrees with the label
    assert not canvas[400:, :].any()          # the table stays in its corner
    right = FramePrediction(t=1.0, track_id=0, model_phase="RECLINED", label_phase="RECLINED")
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    _draw_table(cv2, canvas, right, 1.0, 10)
    assert green(canvas[130:160, 300:660]) > red(canvas[130:160, 300:660])


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


def test_alert_verdicts_follow_the_benchmark_windows():
    from ahfd.ml.clip_review import judge_alerts

    events = [
        ("POSSIBLE_FALL", 9.6, 1), ("FALL_SUSPECTED", 10.2, 1),  # one group, 0.4 s before the impact
        ("POSSIBLE_FALL", 60.0, 1),  # nothing near: false alarm
        ("FALL_SUSPECTED", 120.0, 1),  # a reviewer judged this episode risky (reviewed 8 s earlier)
        ("BED_EXIT", 31.0, 2), ("BED_EXIT", 90.0, 2),
        ("BED_EXIT_WARNING", 20.0, 2), ("BED_EXIT_WARNING", 150.0, 2),
    ]
    marks = judge_alerts(events, impacts=[10.0, 200.0], exits=[30.0], reviewed={112.0: "risky"})
    got = [(m.kind, round(m.t, 1), m.verdict) for m in marks]
    assert ("fall", 9.6, "TRUE") in got
    assert ("fall", 60.0, "FALSE ALARM") in got
    assert ("fall", 120.0, "RISKY") in got
    assert ("bed_exit", 31.0, "TRUE") in got and ("bed_exit", 90.0, "FALSE ALARM") in got
    assert ("warning", 20.0, "TRUE") in got and ("warning", 150.0, "NO EXIT") in got
    assert ("missed", 200.0, "MISSED") in got  # the second labelled fall had no alert
    assert len([m for m in marks if m.kind == "fall"]) == 3
