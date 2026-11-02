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
