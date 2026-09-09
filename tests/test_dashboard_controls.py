"""Live camera / model switching from the dashboard page.

The seam that keeps these camera-free is `runner_factory` on the controller: a
FakeRunner records what it was constructed with and never opens anything. The
few tests that do drive a real PipelineRunner use `seq://` over generated PNGs
or a path that cannot open, so none of them touch a camera or download weights.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from ahfd.capture.devices import RealSenseDevice, RealSenseProbe, probe_realsense
from ahfd.config import Config, SourceOption
from ahfd.dashboard import DashboardController, DashboardServer, DashboardState
from ahfd.dashboard.controller import allowed_backends, build_source_options
from ahfd.dashboard.runner import PipelineRunner

NO_DEVICES = RealSenseProbe(installed=False)


def wait_for(pred, timeout: float = 5.0):
    """Switching is asynchronous -- poll rather than sleep a guessed interval."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def make_cfg(**dashboard) -> Config:
    cfg = Config()
    for key, value in dashboard.items():
        setattr(cfg.dashboard, key, value)
    return cfg


class FakeRunner:
    """Records its construction; never opens a device or loads a model."""

    def __init__(self, source_uri, cfg, calib_path, state, show_rgb, *, gen=0, log=None):
        self.source_uri = source_uri
        self.cfg = cfg
        self.calib_path = calib_path
        self.state = state
        self.show_rgb = show_rgb
        self.gen = gen
        self.started = False
        self.stopped = False
        self.alive = False
        self.block_stop: threading.Event | None = None
        self.stays_alive = False
        self.log = log if log is not None else []

    def start(self):
        self.started = True
        self.alive = True
        self.log.append(("start", self.source_uri, self.gen))

    def stop(self, timeout=5.0):
        if self.block_stop is not None:
            self.block_stop.wait(timeout=10.0)
        self.stopped = True
        self.alive = self.stays_alive
        self.log.append(("stop", self.source_uri, self.gen))


def controller(cfg=None, **kwargs):
    """A controller wired to FakeRunner, recording every start/stop in order."""
    calls: list = []
    made: list[FakeRunner] = []

    def factory(*args, **kw):
        runner = FakeRunner(*args, log=calls, **kw)
        made.append(runner)
        return runner

    ctl = DashboardController(
        cfg or make_cfg(),
        None,
        DashboardState(),
        runner_factory=factory,
        probe=lambda: NO_DEVICES,
        **kwargs,
    )
    ctl.calls = calls
    ctl.made = made
    return ctl


# --------------------------------------------------------------- state


class TestGenerationFence:
    def test_bare_state_reports_idle(self):
        rt = DashboardState().snapshot()["runtime"]
        assert rt["status"] == "idle"
        assert rt["error"] is None
        assert rt["switch_seq"] == 0

    def test_publish_status_reaches_the_snapshot(self):
        s = DashboardState()
        s.publish_status(0, "running", source="rs://", model="rtmo-s")
        rt = s.snapshot()["runtime"]
        assert rt["status"] == "running"
        assert rt["source"] == "rs://" and rt["model"] == "rtmo-s"

    def test_retired_runner_cannot_publish_frames(self):
        s = DashboardState()
        gen = s.begin_generation()
        s.publish_frame(b"new", [], 10.0, gen=gen)
        s.publish_frame(b"stale", [], 99.0, gen=gen - 1)
        assert s.latest_frame() == (b"new", 1)
        assert s.snapshot()["fps"] == 10.0

    def test_retired_runner_cannot_publish_events_or_status(self):
        s = DashboardState()
        gen = s.begin_generation()
        s.publish_status(gen, "running")
        s.publish_event({"event_id": "x", "type": "FALL_CONFIRMED", "severity": 4}, gen=gen - 1)
        s.publish_status(gen - 1, "error", error="boom")
        snap = s.snapshot()
        assert snap["events"] == []
        assert snap["runtime"]["status"] == "running"
        assert snap["runtime"]["error"] is None

    def test_begin_generation_clears_stale_metrics(self):
        s = DashboardState()
        s.publish_frame(b"f", [{"track_id": 1, "state": "UPRIGHT"}], 12.0)
        s.begin_generation(source="webcam://1")
        snap = s.snapshot()
        assert snap["people"] == 0 and snap["tracks"] == [] and snap["fps"] == 0.0
        # The last JPEG is deliberately kept -- see begin_generation's docstring.
        assert s.latest_frame()[0] == b"f"

    def test_existing_snapshot_keys_survive(self):
        """Regression pin: `runtime` is purely additive."""
        snap = DashboardState().snapshot()
        for key in (
            "fps", "uptime_s", "people", "tracks", "counts", "open_alerts",
            "open_count", "seconds_since_alert", "events",
        ):
            assert key in snap

    def test_update_runtime_leaves_the_status_alone(self):
        s = DashboardState()
        gen = s.begin_generation()
        s.publish_status(gen, "running", model="rtmo-s")
        s.update_runtime(gen, show_rgb=True)
        rt = s.snapshot()["runtime"]
        assert rt["status"] == "running" and rt["show_rgb"] is True


# ---------------------------------------------------------- controller


class TestSwitching:
    def test_start_spawns_one_runner(self):
        ctl = controller()
        ctl.start()
        assert len(ctl.made) == 1
        assert ctl.made[0].source_uri == "webcam://0" and ctl.made[0].started

    def test_switch_stops_the_old_before_starting_the_new(self):
        ctl = controller()
        ctl.start()
        code, _ = ctl.switch(source="file://clip.mp4")
        assert code == 202
        assert wait_for(lambda: len(ctl.made) == 2 and ctl.made[1].started)
        assert [c[0] for c in ctl.calls] == ["start", "stop", "start"]
        assert ctl.made[1].source_uri == "file://clip.mp4"

    def test_generation_increments_and_reaches_the_runner(self):
        ctl = controller()
        ctl.start()
        first = ctl.made[0].gen
        ctl.switch(backend="rtmpose")
        assert wait_for(lambda: len(ctl.made) == 2)
        assert ctl.made[1].gen == first + 1 == ctl.state.generation

    def test_backend_switch_copies_the_config(self):
        """The retiring runner is still reading cfg.pose on its own thread."""
        ctl = controller()
        ctl.start()
        ctl.switch(backend="rtmpose")
        assert wait_for(lambda: len(ctl.made) == 2)
        assert ctl.made[0].cfg.pose.backend == "rtmo"
        assert ctl.made[1].cfg.pose.backend == "rtmpose"

    def test_second_switch_is_rejected_not_queued(self):
        ctl = controller()
        ctl.start()
        gate = threading.Event()
        ctl.made[0].block_stop = gate
        assert ctl.switch(source="file://a.mp4")[0] == 202
        assert wait_for(lambda: ctl.state.snapshot()["runtime"]["status"] == "switching")
        code, body = ctl.switch(source="file://b.mp4")
        assert code == 409 and "in progress" in body["error"]
        gate.set()
        assert wait_for(lambda: len(ctl.made) == 2)
        assert ctl.made[1].source_uri == "file://a.mp4"

    def test_unknown_backend_is_refused(self):
        ctl = controller()
        ctl.start()
        code, body = ctl.switch(backend="nope")
        assert code == 400 and "unknown backend" in body["error"]
        assert len(ctl.made) == 1 and ctl.made[0].alive

    def test_backend_outside_the_allow_list_is_refused(self):
        ctl = controller(make_cfg(backends=["rtmo", "rtmpose"]))
        ctl.start()
        assert ctl.switch(backend="yolo")[0] == 400
        assert ctl.switch(backend="rtmpose")[0] == 202

    def test_custom_source_can_be_disallowed(self):
        cfg = make_cfg(
            allow_custom_source=False,
            sources=[SourceOption(label="Bed 3", uri="rs://")],
        )
        ctl = controller(cfg, source="rs://")
        ctl.start()
        code, body = ctl.switch(source="file://anything.mp4")
        assert code == 400 and "only offers the configured cameras" in body["error"]
        assert ctl.switch(source="rs://")[0] == 202

    def test_bare_path_is_refused_when_custom_is_allowed(self):
        ctl = controller()
        code, body = ctl.switch(source="C:/clips/fall.mp4")
        assert code == 400 and "unknown source scheme" in body["error"]

    def test_overlong_uri_is_refused(self):
        ctl = controller()
        assert ctl.switch(source="file://" + "a" * 600)[0] == 400

    def test_empty_switch_is_refused(self):
        assert controller().switch()[0] == 400

    def test_wedged_camera_warns_but_the_switch_proceeds(self):
        ctl = controller()
        ctl.start()
        ctl.made[0].stays_alive = True  # never lets go of the device
        ctl.switch(source="file://clip.mp4")
        assert wait_for(lambda: len(ctl.made) == 2 and ctl.made[1].started)
        assert "has not released yet" in ctl.state.snapshot()["runtime"]["warning"]
        # Fenced: whatever the wedged runner publishes is dropped.
        ctl.state.publish_frame(b"ghost", [], 5.0, gen=ctl.made[0].gen)
        assert ctl.state.latest_frame()[0] is None

    def test_stop_retires_the_runner(self):
        ctl = controller()
        ctl.start()
        ctl.stop()
        assert ctl.made[0].stopped


class TestRgbGate:
    def test_turning_rgb_on_needs_authorisation(self):
        ctl = controller()
        ctl.start()
        code, body = ctl.set_rgb(True)
        assert code == 403 and "not authorised" in body["error"]
        assert ctl.made[0].show_rgb is False
        assert len(ctl.made) == 1  # no restart

    def test_authorised_rgb_flips_in_place(self):
        ctl = controller(rgb_authorised=True)
        ctl.start()
        runner = ctl.made[0]
        code, body = ctl.set_rgb(True)
        assert code == 200 and body["show_rgb"] is True
        assert runner.show_rgb is True
        assert ctl.made[0] is runner  # same pipeline, no restart
        assert ctl.state.snapshot()["runtime"]["show_rgb"] is True

    def test_turning_rgb_off_never_needs_authorisation(self):
        ctl = controller()
        ctl.start()
        code, body = ctl.set_rgb(False)
        assert code == 200 and body["show_rgb"] is False

    def test_starting_with_rgb_authorises_it(self):
        ctl = controller(show_rgb=True)
        assert ctl.options()["allow_rgb"] is True

    def test_switch_routes_a_lone_rgb_flag_to_set_rgb(self):
        ctl = controller()
        ctl.start()
        assert ctl.switch(show_rgb=True)[0] == 403
        assert len(ctl.made) == 1


class TestOptions:
    def test_options_report_the_picker(self):
        cfg = make_cfg(
            sources=[SourceOption(label="Laptop", uri="webcam://0")],
            backends=["rtmo"],
            allow_custom_source=False,
        )
        opts = controller(cfg).options()
        assert opts["ok"] is True
        assert opts["sources"][0]["uri"] == "webcam://0"
        assert opts["backends"] == ["rtmo"]
        assert opts["allow_custom_source"] is False

    def test_running_source_is_always_listed(self):
        cfg = make_cfg(sources=[SourceOption(label="Laptop", uri="webcam://0")])
        opts = controller(cfg, source="file://clip.mp4").options()
        assert opts["sources"][0]["uri"] == "file://clip.mp4"


class TestSourceOptionBuilding:
    def test_realsense_is_appended_when_present(self):
        cfg = make_cfg(sources=[SourceOption(label="Laptop", uri="webcam://0")])
        probe = RealSenseProbe(
            installed=True, devices=(RealSenseDevice("D435i", "123", "3.2"),)
        )
        out = build_source_options(cfg, probe)
        assert [o["uri"] for o in out] == ["webcam://0", "rs://"]
        assert out[1]["label"] == "D435i" and out[1]["detected"] is True

    def test_realsense_is_not_duplicated(self):
        cfg = make_cfg(sources=[SourceOption(label="Bay A", uri="rs://")])
        probe = RealSenseProbe(installed=True, devices=(RealSenseDevice("D435i"),))
        out = build_source_options(cfg, probe)
        assert [o["uri"] for o in out] == ["rs://"]
        assert out[0]["label"] == "Bay A"

    def test_usb2_link_is_called_out(self):
        probe = RealSenseProbe(
            installed=True, devices=(RealSenseDevice("D435i", "1", "2.1"),)
        )
        assert "USB 2" in build_source_options(make_cfg(), probe)[0]["label"]

    def test_several_cameras_offer_one_entry(self):
        probe = RealSenseProbe(
            installed=True,
            devices=(RealSenseDevice("D435i", "1", "3.2"), RealSenseDevice("D455", "2", "3.2")),
        )
        out = build_source_options(make_cfg(), probe)
        assert len(out) == 1 and "first of 2" in out[0]["label"]

    def test_current_uri_is_prepended_when_unlisted(self):
        cfg = make_cfg(sources=[SourceOption(label="Laptop", uri="webcam://0")])
        out = build_source_options(cfg, NO_DEVICES, current="seq://clips/a")
        assert out[0]["uri"] == "seq://clips/a"

    def test_backend_allow_list_keeps_canonical_order(self):
        assert allowed_backends(make_cfg(backends=["yolo", "rtmo"])) == ["rtmo", "yolo"]
        assert allowed_backends(make_cfg()) == ["rtmo", "rtmpose", "yolo"]
        # An unknown name in the config cannot smuggle itself into the picker.
        assert allowed_backends(make_cfg(backends=["nope"])) == ["rtmo", "rtmpose", "yolo"]


# ------------------------------------------------ real runner, no camera


def image_sequence(tmp_path, n=2):
    for i in range(n):
        cv2.imwrite(str(tmp_path / ("f_" + str(i) + ".png")), np.zeros((48, 64, 3), np.uint8))
    return "seq://" + str(tmp_path)


def run_and_wait(uri, cfg, calib=None, timeout=10.0):
    state = DashboardState()
    runner = PipelineRunner(uri, cfg, calib, state, show_rgb=False)
    runner.start()
    wait_for(
        lambda: state.snapshot()["runtime"]["status"] in ("error", "ended", "stopped"),
        timeout=timeout,
    )
    runner.stop(timeout=2.0)
    return state.snapshot()["runtime"]


class TestRunnerErrorsSurface:
    def test_single_use(self):
        runner = PipelineRunner("seq://nowhere", Config(), None, DashboardState(), False)
        runner.start()
        with pytest.raises(RuntimeError, match="single-use"):
            runner.start()
        runner.stop(timeout=2.0)

    def test_unopenable_source_reports_an_error(self):
        rt = run_and_wait("file://does-not-exist.mp4", Config())
        assert rt["status"] == "error"
        assert "could not open" in rt["error"]

    def test_unknown_backend_reports_an_error(self, tmp_path):
        cfg = Config()
        cfg.pose.backend = "nope"  # raises on the name, before any model import
        rt = run_and_wait(image_sequence(tmp_path), cfg)
        assert rt["status"] == "error"
        assert "unknown pose backend" in rt["error"]

    def test_missing_calibration_reports_an_error(self, tmp_path):
        """Detection is wired before the model, so this needs no download."""
        cfg = Config()
        cfg.detect.enabled = True
        rt = run_and_wait(image_sequence(tmp_path), cfg)
        assert rt["status"] == "error"
        assert "calibration" in rt["error"]

    def test_source_that_runs_out_ends_rather_than_freezing(self, tmp_path, monkeypatch):
        class StubEstimator:
            name = "stub"

            def estimate(self, frame):
                from ahfd.types import PoseFrame

                h, w = frame.bgr.shape[:2]
                return PoseFrame(
                    t=frame.t, index=frame.index, width=w, height=h, people=()
                )

        monkeypatch.setattr("ahfd.pose.build_estimator", lambda cfg: StubEstimator())
        cfg = Config()
        cfg.detect.enabled = False
        rt = run_and_wait(image_sequence(tmp_path), cfg)
        assert rt["status"] == "ended"
        assert rt["model"] == "stub"


# -------------------------------------------------------------- server


class FakeController:
    def __init__(self, code=202, payload=None):
        self.code = code
        self.payload = payload or {"ok": True, "switching": True}
        self.seen: list[dict] = []

    def options(self):
        return {"ok": True, "sources": [{"label": "L", "uri": "webcam://0"}],
                "backends": ["rtmo"], "allow_custom_source": True, "allow_rgb": False}

    def switch(self, **kwargs):
        self.seen.append(kwargs)
        return self.code, self.payload


def serve(controller=None):
    state = DashboardState()
    srv = DashboardServer(state, host="127.0.0.1", port=0, controller=controller)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.05)
    return srv, state, "http://127.0.0.1:" + str(srv.server_address[1])


def request(url, data=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url, data=body, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"raw": raw}


class TestControlEndpoints:
    def test_without_a_controller_controls_are_unavailable(self):
        srv, _state, base = serve()
        try:
            assert request(base + "/api/options")[0] == 503
            assert request(base + "/api/switch", {"source": "rs://"})[0] == 503
            # The read-only dashboard still works.
            assert request(base + "/api/state")[0] == 200
        finally:
            srv.shutdown(); srv.server_close()

    def test_options_are_served(self):
        srv, _state, base = serve(FakeController())
        try:
            status, body = request(base + "/api/options")
            assert status == 200 and body["backends"] == ["rtmo"]
        finally:
            srv.shutdown(); srv.server_close()

    def test_body_is_forwarded_as_kwargs(self):
        ctl = FakeController()
        srv, _state, base = serve(ctl)
        try:
            status, body = request(
                base + "/api/switch",
                {"source": "  rs://  ", "backend": "", "show_rgb": "yes"},
            )
            assert status == 202 and body["ok"] is True
            # Whitespace trimmed, blanks and non-booleans become None.
            assert ctl.seen == [{"source": "rs://", "backend": None, "show_rgb": None}]
        finally:
            srv.shutdown(); srv.server_close()

    def test_the_controllers_status_reaches_the_client(self):
        ctl = FakeController(code=409, payload={"ok": False, "error": "busy"})
        srv, _state, base = serve(ctl)
        try:
            assert request(base + "/api/switch", {"source": "rs://"}) == (
                409, {"ok": False, "error": "busy"}
            )
        finally:
            srv.shutdown(); srv.server_close()

    def test_malformed_body_is_refused_without_wedging_the_server(self):
        ctl = FakeController()
        srv, _state, base = serve(ctl)
        try:
            req = urllib.request.Request(
                base + "/api/switch", data=b"not json", method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=2.0)
                raise AssertionError("expected 400")
            except urllib.error.HTTPError as e:
                assert e.code == 400
            assert request(base + "/api/switch", ["a", "list"])[0] == 400
            assert ctl.seen == []
            assert request(base + "/api/state")[0] == 200
        finally:
            srv.shutdown(); srv.server_close()

    def test_oversized_body_is_refused(self):
        ctl = FakeController()
        srv, _state, base = serve(ctl)
        try:
            assert request(base + "/api/switch", {"source": "x" * 9000})[0] == 400
            assert ctl.seen == []
        finally:
            srv.shutdown(); srv.server_close()

    def test_cross_origin_post_is_refused(self):
        ctl = FakeController()
        srv, state, base = serve(ctl)
        try:
            status, body = request(
                base + "/api/switch", {"source": "rs://"},
                headers={"Origin": "http://evil.example"},
            )
            assert status == 403 and "cross-origin" in body["error"]
            assert ctl.seen == []
            # The same protection now covers acknowledgement.
            state.publish_event({"event_id": "e1", "type": "FALL_CONFIRMED", "severity": 4})
            request(base + "/api/ack/e1", {}, headers={"Origin": "http://evil.example"})
            assert state.snapshot()["open_count"] == 1
        finally:
            srv.shutdown(); srv.server_close()


# -------------------------------------------------------------- devices


class TestProbe:
    def test_missing_pyrealsense_is_a_result_not_an_exception(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrealsense2", None)
        probe = probe_realsense()
        assert probe.installed is False and probe.devices == ()

    def test_devices_are_parsed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrealsense2", _fake_rs(
            [("D435i", "111", "3.2"), ("D455", "222", "2.1")]
        ))
        probe = probe_realsense()
        assert [d.name for d in probe.devices] == ["D435i", "D455"]
        assert probe.devices[0].usb2 is False and probe.devices[1].usb2 is True

    def test_enumeration_failure_is_reported(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrealsense2", _fake_rs(None, boom="driver gone"))
        probe = probe_realsense()
        assert probe.installed is True and probe.error == "driver gone"
        assert probe.devices == ()

    def test_a_device_that_will_not_answer_still_appears(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrealsense2", _fake_rs([None]))
        probe = probe_realsense()
        assert len(probe.devices) == 1
        assert probe.devices[0].name == "RealSense" and probe.devices[0].usb == ""


def _fake_rs(specs, boom=None):
    """A stand-in pyrealsense2. `specs` entries are (name, serial, usb) or None."""
    mod = types.ModuleType("pyrealsense2")
    mod.__version__ = "2.0.0-fake"
    mod.camera_info = types.SimpleNamespace(
        name="name", serial_number="serial_number", usb_type_descriptor="usb_type_descriptor"
    )

    class Device:
        def __init__(self, spec):
            self.spec = spec

        def get_info(self, field):
            if self.spec is None:
                raise RuntimeError("not supported")
            return dict(
                zip(("name", "serial_number", "usb_type_descriptor"), self.spec)
            )[field]

    class Context:
        def query_devices(self):
            if boom:
                raise RuntimeError(boom)
            return [Device(s) for s in specs]

    mod.context = Context
    return mod


# ------------------------------------------------------- posture reporting


def _posture_setup(tmp_path, n_frames=12):
    """A calibrated 640x480 ward and a stub that projects a standing adult.

    Mirrors tests/test_integration.py: a 3D body goes through the real camera
    model, the real FeatureExtractor and the real FallStateMachine. Only the
    pose model is stubbed, so this needs no camera and no download.
    """
    import numpy as np

    from ahfd.cli import _write_calibration_yaml
    from ahfd.geometry import GroundPlane
    from ahfd.types import NUM_KEYPOINTS, Intrinsics, PersonPose, PoseFrame

    intr = Intrinsics.from_hfov(640, 480, hfov_deg=69.4, vfov_deg=42.5)
    ground = GroundPlane(intr, height_m=2.6, pitch_deg=20.0)

    # (height above floor, lateral offset) for a 1.7 m adult, COCO-17 order.
    body = (
        (1.62, 0.00), (1.64, 0.03), (1.64, -0.03), (1.62, 0.07), (1.62, -0.07),
        (1.40, 0.18), (1.40, -0.18), (1.10, 0.20), (1.10, -0.20),
        (0.85, 0.20), (0.85, -0.20), (0.95, 0.12), (0.95, -0.12),
        (0.50, 0.12), (0.50, -0.12), (0.08, 0.10), (0.08, -0.10),
    )

    def project(point):
        rel = np.asarray(point, float) - np.array([0.0, 0.0, ground.height_m])
        d = ground.rotation.T @ rel
        return (intr.cx + intr.fx * d[0] / d[2], intr.cy + intr.fy * d[1] / d[2])

    standing = np.array([[lat, 5.0, h] for h, lat in body], dtype=float)
    pts = np.array([project(pt) for pt in standing], dtype=np.float32)

    class StandingStub:
        name = "standing-stub"

        def estimate(self, frame):
            person = PersonPose(
                keypoints=pts.copy(),
                scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
                score=1.0,
                track_id=None,
            )
            return PoseFrame(
                t=frame.t, index=frame.index, width=640, height=480, people=(person,)
            )

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(n_frames):
        cv2.imwrite(str(frames / ("f_%03d.png" % i)), np.zeros((480, 640, 3), np.uint8))

    calib = tmp_path / "calib.yaml"
    _write_calibration_yaml(calib, "test-cam", intr, 2.6, pitch_deg=20.0)
    return "seq://" + str(frames), str(calib), StandingStub()


class TestPostureReachesTheDashboard:
    def test_detection_on_reports_a_posture(self, tmp_path, monkeypatch):
        """The regression this guards: chips must not all read TRACKED."""
        uri, calib, stub = _posture_setup(tmp_path)
        monkeypatch.setattr("ahfd.pose.build_estimator", lambda cfg: stub)

        cfg = Config()
        cfg.detect.enabled = True
        state = DashboardState()
        runner = PipelineRunner(uri, cfg, calib, state, show_rgb=False)
        runner.start()
        wait_for(lambda: state.snapshot()["tracks"])
        tracks = state.snapshot()["tracks"]
        runner.stop(timeout=3.0)

        assert tracks, "no tracks published"
        assert tracks[0]["state"] == "UPRIGHT", tracks
        assert tracks[0]["height_m"] > 1.0, tracks  # metric, from the ground plane
        assert state.snapshot()["runtime"]["detect"] is True

    def test_detection_off_says_so(self, tmp_path, monkeypatch):
        """With detect off the chips read TRACKED -- the page must explain why."""
        uri, _calib, stub = _posture_setup(tmp_path)
        monkeypatch.setattr("ahfd.pose.build_estimator", lambda cfg: stub)

        cfg = Config()
        cfg.detect.enabled = False
        state = DashboardState()
        runner = PipelineRunner(uri, cfg, None, state, show_rgb=False)
        runner.start()
        wait_for(lambda: state.snapshot()["tracks"])
        snap = state.snapshot()
        runner.stop(timeout=3.0)

        assert snap["tracks"][0]["state"] == "TRACKED"
        assert snap["runtime"]["detect"] is False


def test_page_explains_missing_postures():
    from ahfd.dashboard.html import DASHBOARD_HTML

    assert "detect.enabled: false" in DASHBOARD_HTML
    assert "detection off" in DASHBOARD_HTML


# ------------------------------------------------------------ cli wiring


def _stub_serving(monkeypatch):
    """Ctrl+C the server the moment it would block, and let shutdown return.

    BaseServer.shutdown() waits on an event that only serve_forever() sets, so
    a stub that never enters the real loop has to close the socket itself or
    the command's finally block hangs forever.
    """
    monkeypatch.setattr(
        DashboardServer,
        "serve_forever",
        lambda self, *a, **k: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(DashboardServer, "shutdown", DashboardServer.server_close)


class TestDashboardCommand:
    """Actually execute `ahfd dashboard`.

    Every other test here reaches for DashboardController directly, so the
    command body itself -- which names it imports, which keywords it passes --
    was never run. A rename there is a NameError the moment a user types the
    command and nothing else would catch it.
    """

    def test_command_wires_controller_and_server(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from ahfd.cli import app

        started: list = []
        monkeypatch.setattr(PipelineRunner, "start", lambda self: started.append(self))
        monkeypatch.setattr(PipelineRunner, "stop", lambda self, timeout=5.0: None)
        _stub_serving(monkeypatch)

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source: \"seq://" + str(tmp_path).replace("\\", "/") + "\"\n"
            "detect:\n  enabled: false\n"
            "dashboard:\n  host: \"127.0.0.1\"\n  port: 0\n"
            "  sources:\n    - label: \"Clip\"\n      uri: \"webcam://0\"\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(app, ["dashboard", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "camera and pose model can be changed from the page" in result.output
        assert len(started) == 1  # exactly one pipeline, started through the controller

    def test_backend_override_reaches_the_pipeline(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from ahfd.cli import app

        made: list = []
        monkeypatch.setattr(
            PipelineRunner, "start", lambda self: made.append((self.source_uri, self.cfg))
        )
        monkeypatch.setattr(PipelineRunner, "stop", lambda self, timeout=5.0: None)
        _stub_serving(monkeypatch)

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "detect:\n  enabled: false\n"
            "dashboard:\n  host: \"127.0.0.1\"\n  port: 0\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            ["dashboard", "--config", str(cfg), "--source", "webcam://3",
             "--backend", "rtmpose"],
        )
        assert result.exit_code == 0, result.output
        assert made[0][0] == "webcam://3"
        assert made[0][1].pose.backend == "rtmpose"


# --------------------------------------------------------------- config


@pytest.mark.parametrize(
    "name",
    ["default.yaml", "dashboard_dev.yaml", "detect_dev.yaml", "laptop_gpu.yaml", "ward.yaml"],
)
def test_shipped_configs_load(name):
    """Nothing else in the suite loads the configs we actually ship."""
    from pathlib import Path

    from ahfd.config import load_config

    path = Path(__file__).resolve().parents[1] / "configs" / name
    cfg = load_config(path)
    for source in cfg.dashboard.sources:
        assert source.uri and source.label
