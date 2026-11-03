"""The nurses' page at /ward; the engineer dashboard at / keeps its controls."""

import threading
import typing
import urllib.request

from ahfd.dashboard.server import DashboardServer
from ahfd.dashboard.state import DashboardState
from ahfd.dashboard.ward_html import WARD_HTML
from ahfd.detect.events import SEVERITY
from ahfd.detect.state_machine import State


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def test_both_pages_are_served_and_stay_separate():
    srv = DashboardServer(DashboardState(), host="127.0.0.1", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:" + str(srv.server_address[1])
    try:
        engineer, ward = fetch(base + "/"), fetch(base + "/ward")
    finally:
        srv.shutdown()
    assert 'id="rescan"' in engineer and 'href="/ward"' in engineer
    assert 'id="rescan"' not in ward and 'href="/"' in ward


def test_all_clear_only_after_every_not_monitoring_case():
    assert WARD_HTML.count("head:'All clear'") == 1
    assert WARD_HTML.index("head:'Not monitoring'") < WARD_HTML.index("head:'All clear'")
    assert WARD_HTML.index("head:'Falls are not being monitored'") < WARD_HTML.index("head:'All clear'")
    assert "Not being monitored" in WARD_HTML and "Connection lost" in WARD_HTML


def test_every_alert_and_state_has_plain_words():
    for event_type in SEVERITY:
        assert event_type + ":" in WARD_HTML
    for state in typing.get_args(State):
        assert state + ":'" in WARD_HTML


def test_needs_nothing_from_outside():
    assert "http://" not in WARD_HTML and "https://" not in WARD_HTML


def test_alert_clips_review_lists_recent_skeleton_clips_only(tmp_path):
    import json
    import time as _t
    from datetime import datetime

    from ahfd.dashboard.alert_review import find_alert_clip, list_alert_clips

    def clip(stem, picture, age_h):
        (tmp_path / (stem + ".mp4")).write_bytes(b"\x00" * 10)
        started = datetime.fromtimestamp(_t.time() - age_h * 3600).isoformat(timespec="seconds")
        (tmp_path / (stem + ".json")).write_text(json.dumps({
            "video": stem + ".mp4", "picture": picture, "started": started, "duration_s": 25.0,
            "seconds_before_first_alert": 10.0, "alerts": [{"type": "BED_EXIT", "clock": "01:27:45", "zone": "nearest"}]}))

    clip("a_new", "stick figures only", 1)
    clip("b_rgb", "camera image", 1)
    clip("c_old", "stick figures only", 30)
    names = [c["name"] for c in list_alert_clips(tmp_path, allow_rgb=False)]
    assert names == ["a_new.mp4"]
    assert {c["name"] for c in list_alert_clips(tmp_path, allow_rgb=True)} == {"a_new.mp4", "b_rgb.mp4"}
    assert find_alert_clip(tmp_path, "b_rgb.mp4", allow_rgb=False) is None
    assert find_alert_clip(tmp_path, "../c_old.mp4", allow_rgb=True) is None
    assert list_alert_clips(tmp_path, allow_rgb=False)[0]["seek_s"] == 7.0


def test_ward_page_has_review_alerts_section():
    from ahfd.dashboard.ward_html import WARD_HTML

    assert 'id="review-box"' in WARD_HTML and "/api/alert_clips" in WARD_HTML
