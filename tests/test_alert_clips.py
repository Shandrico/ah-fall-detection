"""Clips around alerts: only what the mode allows, and never lost or redirected quietly."""

import json
import time

import cv2
import numpy as np
import pytest

from ahfd.alert.sinks import JsonlSink
from ahfd.dashboard.state import HISTORY_S, DashboardState
from ahfd.debug.alert_clips import AlertClipRecorder, check_clip_settings
from ahfd.detect import Event
from ahfd.privacy import ENV_VAR, PrivacyViolation


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def jpeg(value=0):
    return cv2.imencode(".jpg", np.full((60, 80, 3), value, np.uint8))[1].tobytes()


def feed(rec, clock, seconds, fps=10, **kw):
    for _ in range(int(seconds * fps)):
        clock.t += 1 / fps
        rec.add_frame(jpeg(), **kw)


def make(tmp_path, **kw):
    clock = Clock()
    return AlertClipRecorder(tmp_path / "clips", clock=clock, pre_s=2, post_s=3, log=lambda m: None, **kw), clock


def alert(kind="FALL_SUSPECTED", zone="nearest"):
    return {"type": kind, "zone": zone, "clock": "14:03:22", "severity": 3, "track_id": 1, "evidence": {"h_torso": 0.2}}


def clips(tmp_path):
    return sorted((tmp_path / "clips").glob("*.mp4")) if (tmp_path / "clips").exists() else []


def test_nothing_reaches_the_disk_without_an_alert(tmp_path):
    rec, clock = make(tmp_path)
    feed(rec, clock, 30)
    rec.close()
    assert clips(tmp_path) == []


def test_keeps_the_seconds_before_and_after_an_alert(tmp_path):
    rec, clock = make(tmp_path)
    feed(rec, clock, 30)
    rec.on_alert(alert())
    feed(rec, clock, 10)
    rec.close()
    video = clips(tmp_path)
    assert len(video) == 1 and "FALL_SUSPECTED_nearest" in video[0].name
    side = json.loads(video[0].with_suffix(".json").read_text())
    assert side["picture"] == "stick figures only"
    assert side["alerts"][0]["type"] == "FALL_SUSPECTED"
    assert side["duration_s"] == pytest.approx(5, abs=0.4)
    assert side["seconds_before_first_alert"] == pytest.approx(2, abs=0.2)
    cap = cv2.VideoCapture(str(video[0]))
    ok, _ = cap.read()
    assert ok
    cap.release()


def test_alerts_close_together_share_one_clip_up_to_the_limit(tmp_path):
    rec, clock = make(tmp_path, max_s=6)
    feed(rec, clock, 3)
    rec.on_alert(alert("POSSIBLE_FALL"))
    feed(rec, clock, 1)
    rec.on_alert(alert("FALL_SUSPECTED"))
    feed(rec, clock, 10)
    rec.close()
    video = clips(tmp_path)
    assert len(video) == 1 and "POSSIBLE_FALL" in video[0].name
    side = json.loads(video[0].with_suffix(".json").read_text())
    assert [a["type"] for a in side["alerts"]] == ["POSSIBLE_FALL", "FALL_SUSPECTED"]


def test_a_camera_switch_does_not_lose_the_open_clip(tmp_path):
    rec, clock = make(tmp_path)
    feed(rec, clock, 3)
    rec.on_alert(alert())
    rec.close()
    assert len(clips(tmp_path)) == 1


def test_a_clip_is_saved_even_if_the_camera_dies_mid_incident(tmp_path):
    rec = AlertClipRecorder(tmp_path / "clips", post_s=0.1, log=lambda m: None)
    rec.add_frame(jpeg())
    rec.on_alert(alert())
    deadline = time.monotonic() + 15
    while not clips(tmp_path) and time.monotonic() < deadline:
        time.sleep(0.1)  # no more frames arrive; close() is never called
    assert len(clips(tmp_path)) == 1
    rec.close()


def test_a_skeleton_recorder_refuses_a_camera_frame(tmp_path):
    rec, _ = make(tmp_path)
    with pytest.raises(PrivacyViolation):
        rec.add_frame(jpeg(), is_rgb=True)


def test_switching_rgb_off_forgets_the_camera_frames(tmp_path):
    rec, clock = make(tmp_path, mode="rgb")
    feed(rec, clock, 1, is_rgb=True)
    rec.on_alert(alert())
    feed(rec, clock, 1, is_rgb=True)
    feed(rec, clock, 4)  # RGB off
    rec.close()
    side = json.loads(clips(tmp_path)[0].with_suffix(".json").read_text())
    assert side["picture"] == "stick figures only"


def test_skeleton_mode_needs_no_switches(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    check_clip_settings("skeleton", tmp_path, config_flag=False, cli_flag=False)


@pytest.mark.parametrize("config,cli,env", [(False, True, "1"), (True, False, "1"), (True, True, None)])
def test_rgb_mode_needs_every_switch(tmp_path, monkeypatch, config, cli, env):
    monkeypatch.delenv(ENV_VAR, raising=False)
    if env:
        monkeypatch.setenv(ENV_VAR, env)
    with pytest.raises(PrivacyViolation):
        check_clip_settings("rgb", tmp_path / "c", config_flag=config, cli_flag=cli)


def test_rgb_mode_refuses_the_repository_and_an_unmounted_drive(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    with pytest.raises(PrivacyViolation, match="repository"):
        check_clip_settings("rgb", "data/alert_clips", config_flag=True, cli_flag=True)
    with pytest.raises(PrivacyViolation, match="mounted"):
        check_clip_settings("rgb", tmp_path / "T9" / "clips", config_flag=True, cli_flag=True)
    check_clip_settings("rgb", tmp_path / "clips", config_flag=True, cli_flag=True)


def test_rgb_clip_is_dropped_if_the_drive_goes_away(tmp_path):
    clock = Clock()
    messages = []
    rec = AlertClipRecorder(tmp_path / "T9" / "clips", mode="rgb", clock=clock, pre_s=1, post_s=1, log=messages.append)
    feed(rec, clock, 1, is_rgb=True)
    rec.on_alert(alert())
    feed(rec, clock, 2, is_rgb=True)
    rec.close()
    assert not (tmp_path / "T9").exists() and "T9" in messages[0]


def test_a_low_disk_stops_saving_and_says_so(tmp_path):
    clock = Clock()
    messages = []
    rec = AlertClipRecorder(tmp_path / "clips", clock=clock, pre_s=1, post_s=1, min_free_gb=1e12, log=messages.append)
    feed(rec, clock, 1)
    rec.on_alert(alert())
    feed(rec, clock, 2)
    rec.close()
    assert "GB free" in messages[0] and not list((tmp_path / "clips").glob("*.mp4"))


def test_the_totals_cover_24_hours_and_survive_a_restart(tmp_path):
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(log)
    sink.emit(Event(type="POSSIBLE_FALL", track_id=1, t_trigger=1.0, t_alert=1.0))
    sink.close()
    old = {"type": "FALL_SUSPECTED", "wall_time": time.time() - HISTORY_S - 5}
    with log.open("a") as f:
        f.write(json.dumps(old) + "\n" + json.dumps({"type": "FALL_SUSPECTED"}) + "\n" + '{"torn')

    state = DashboardState()
    state.load_history(log)
    assert state.snapshot()["counts_24h"] == {"POSSIBLE_FALL": 1}
    state.publish_event({"event_id": "a", "type": "FALL_SUSPECTED", "alerting": True})
    assert state.snapshot()["counts_24h"] == {"POSSIBLE_FALL": 1, "FALL_SUSPECTED": 1}
    state._history[0] = (time.time() - HISTORY_S - 1, "POSSIBLE_FALL")
    assert state.snapshot()["counts_24h"] == {"FALL_SUSPECTED": 1}
    state.load_history(tmp_path / "missing.jsonl")


def _runner(recorder):
    from ahfd.config import Config
    from ahfd.dashboard.runner import PipelineRunner
    from ahfd.types import PoseFrame

    runner = PipelineRunner("x", Config(), None, DashboardState(), True)
    runner._clips = recorder
    return runner, PoseFrame(t=1.0, index=0, width=1920, height=1080, people=())


def test_camera_on_screen_still_saves_only_stick_figures(tmp_path):
    rec = AlertClipRecorder(tmp_path / "clips", log=lambda m: None)
    runner, pose = _runner(rec)
    camera_view = np.frombuffer(jpeg(255), np.uint8)
    runner._offer_clip_frame(pose, {}, 15.0, camera_view, True)
    saved = rec._ring[-1]
    assert saved[2] is False and saved[1] != camera_view.tobytes()
    image = cv2.imdecode(np.frombuffer(saved[1], np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image.shape[1] <= 960 and np.median(image) < 128  # blank canvas, not the white camera frame
    rec.close()


def test_rgb_recorder_keeps_the_camera_frame_only_when_rgb_is_shown(tmp_path):
    rec = AlertClipRecorder(tmp_path / "clips", mode="rgb", log=lambda m: None)
    runner, pose = _runner(rec)
    camera_view = np.frombuffer(jpeg(255), np.uint8)
    runner._offer_clip_frame(pose, {}, 15.0, camera_view, True)
    assert rec._ring[-1][2] is True
    runner._offer_clip_frame(pose, {}, 15.0, np.frombuffer(jpeg(0), np.uint8), False)
    assert [f[2] for f in rec._ring] == [False]
    rec.close()
