"""Dashboard tests: the state store and the stdlib server.

The server is exercised against a real socket on an ephemeral port, because the
things most worth testing here -- that a disconnect is noticed, that extra
viewers cost nothing -- are socket behaviours, not pure functions. The pipeline
thread is not started; state is pushed directly, so these stay fast and need no
camera or model.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from ahfd.dashboard import DashboardServer, DashboardState


def track(tid, state, **extra):
    return {"track_id": tid, "state": state, **extra}


class TestDashboardState:
    def test_starts_empty(self):
        s = DashboardState()
        jpeg, seq = s.latest_frame()
        assert jpeg is None
        snap = s.snapshot()
        assert snap["tracks"] == []
        assert snap["events"] == []
        assert snap["open_alerts"] == []
        assert snap["open_count"] == 0
        assert snap["people"] == 0

    def test_publish_frame_increments_seq(self):
        s = DashboardState()
        s.publish_frame(b"jpeg1", [track(1, "UPRIGHT")], fps=30.0)
        j, seq1 = s.latest_frame()
        assert j == b"jpeg1"
        s.publish_frame(b"jpeg2", [track(1, "UPRIGHT")], fps=29.0)
        _, seq2 = s.latest_frame()
        assert seq2 > seq1

    def test_tracks_and_people_appear_in_snapshot(self):
        s = DashboardState()
        s.publish_frame(
            b"x", [track(2, "ON_GROUND", height_m=0.2), track(5, "UPRIGHT")], fps=15.0
        )
        snap = s.snapshot()
        assert snap["people"] == 2
        states = {t["track_id"]: t["state"] for t in snap["tracks"]}
        assert states == {2: "ON_GROUND", 5: "UPRIGHT"}

    def test_events_newest_first(self):
        s = DashboardState()
        s.publish_event({"event_id": "a", "type": "BED_EXIT", "severity": 1})
        s.publish_event({"event_id": "b", "type": "FALL_CONFIRMED", "severity": 4})
        assert s.snapshot()["events"][0]["event_id"] == "b"

    def test_counts_accumulate_beyond_the_log_cap(self):
        """Session totals must survive the recent-events log scrolling past."""
        s = DashboardState(max_events=3)
        for i in range(10):
            s.publish_event({"event_id": f"f{i}", "type": "FALL_CONFIRMED", "severity": 4})
        s.publish_event({"event_id": "b", "type": "BED_EXIT", "severity": 1})
        snap = s.snapshot()
        assert snap["counts"]["fall_confirmed"] == 10  # counter, not the capped log
        assert snap["counts"]["bed_exit"] == 1
        assert len(snap["events"]) == 3

    def test_open_alerts_lists_only_unacked_alerting(self):
        s = DashboardState()
        s.publish_event({"event_id": "a", "type": "FALL_CONFIRMED", "severity": 4})
        s.publish_event({"event_id": "b", "type": "BED_EXIT", "severity": 1})
        snap = s.snapshot()
        assert snap["open_count"] == 1  # BED_EXIT is not an alerting type
        assert snap["open_alerts"][0]["event_id"] == "a"

    def test_acknowledge_clears_open_alert(self):
        s = DashboardState()
        s.publish_event({"event_id": "a", "type": "FALL_CONFIRMED", "severity": 4})
        assert s.snapshot()["open_count"] == 1
        s.acknowledge("a")
        assert s.snapshot()["open_count"] == 0
        assert s.snapshot()["events"][0]["acknowledged"] is True

    def test_unacknowledge_reopens(self):
        s = DashboardState()
        s.publish_event({"event_id": "a", "type": "PERSON_DOWN", "severity": 3})
        s.acknowledge("a")
        assert s.snapshot()["open_count"] == 0
        s.unacknowledge("a")
        assert s.snapshot()["open_count"] == 1

    def test_event_cap(self):
        s = DashboardState(max_events=5)
        for i in range(20):
            s.publish_event({"event_id": str(i), "type": "BED_EXIT", "severity": 1})
        assert len(s.snapshot()["events"]) == 5


@pytest.fixture
def server():
    state = DashboardState()
    srv = DashboardServer(state, host="127.0.0.1", port=0)  # port 0 = pick a free one
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield state, "http://127.0.0.1:" + str(port)
    srv.shutdown()
    srv.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return r.status, r.read()


class TestServer:
    def test_serves_the_page(self, server):
        _state, base = server
        status, body = _get(base + "/")
        assert status == 200
        assert b"Ward Fall Monitor" in body

    def test_state_endpoint_is_json(self, server):
        state, base = server
        state.publish_frame(b"x", [track(1, "UPRIGHT")], fps=20.0)
        status, body = _get(base + "/api/state")
        assert status == 200
        data = json.loads(body)
        assert data["fps"] == 20.0
        assert data["people"] == 1
        assert data["tracks"] == [{"track_id": 1, "state": "UPRIGHT"}]

    def test_ack_endpoint(self, server):
        state, base = server
        state.publish_event({"event_id": "e1", "type": "FALL_CONFIRMED", "severity": 4})
        req = urllib.request.Request(base + "/api/ack/e1", method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            assert r.status == 200
        assert state.snapshot()["open_count"] == 0

    def test_unknown_path_is_404(self, server):
        _state, base = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base + "/nope")
        assert exc.value.code == 404

    def test_stream_sends_frames_then_survives_disconnect(self, server):
        """Read one MJPEG frame, then drop the connection early.

        The server must not hang or leak when the client leaves mid-stream --
        the exact failure that overheated the reference build. Reaching the
        assertions (and the fixture tearing down cleanly) is the test.
        """
        state, base = server
        # A realistically sized frame, so one part alone has plenty of bytes to
        # read -- read() blocks until it has the requested count, and the stream
        # never EOFs, so the read must stay within a single published frame.
        fake_jpeg = b"\xff\xd8" + b"j" * 4000 + b"\xff\xd9"
        state.publish_frame(fake_jpeg, [track(1, "UPRIGHT")], fps=15.0)

        with urllib.request.urlopen(base + "/stream.mjpg", timeout=2.0) as r:
            assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
            chunk = r.read(200)  # boundary + part headers + into the payload
            assert chunk.startswith(b"--ahfdframe")
            assert b"image/jpeg" in chunk
            assert b"\xff\xd8" in chunk  # JPEG start-of-image marker
        # Closing the response drops the socket; the server thread should notice
        # on its next write and end. Give it a moment, then confirm the server
        # still answers other requests (i.e. it did not wedge).
        time.sleep(0.1)
        status, _ = _get(base + "/api/state")
        assert status == 200
