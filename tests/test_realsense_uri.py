"""Parsing of rs:// capture URIs into RealSenseSource kwargs (no hardware)."""

from __future__ import annotations

import pytest

from ahfd.capture.factory import open_source, parse_realsense_uri
from ahfd.capture.realsense import (
    BagSource,
    RealSenseVerificationError,
    _RealSenseBase,
    _apply_strict_depth_controls,
    _bind_config_to_device,
    _verify_active_usb3,
    _verify_active_device_serial,
)


def test_bag_meta_reports_whether_depth_was_enabled_without_opening_hardware():
    bag = object.__new__(BagSource)
    bag._path = "clip.bag"
    bag._color_size = (1920, 1080)
    bag._fps = 30
    bag._with_depth = False
    assert bag.meta.has_depth is False
    bag._with_depth = True
    assert bag.meta.has_depth is True


def _bag(path):
    bag = object.__new__(BagSource)
    bag._path = str(path)
    bag._color_size = (1920, 1080)
    bag._fps = 30
    bag._with_depth = False
    return bag


def test_bag_playback_skips_framesets_that_repeat_the_colour_frame():
    class Color:
        def __init__(self, number):
            self.number = number

        def __bool__(self):
            return True

        def get_frame_number(self):
            return self.number

        def get_timestamp(self):
            return self.number * 133.0  # the colour image's own clock

    class Frameset:
        def __init__(self, number, t_ms):
            self.color, self.t_ms = Color(number), t_ms

        def get_color_frame(self):
            return self.color

        def get_timestamp(self):
            return self.t_ms

    # A depth-only update re-emits colour frame 1, then colour 2 arrives twice;
    # the frameset stamps follow depth, not colour.
    queue = [Frameset(1, 120.0), Frameset(1, 150.0), Frameset(2, 250.0), Frameset(2, 280.0)]

    class Pipeline:
        def start(self, _config):
            class Profile:
                def get_device(self):
                    class Device:
                        def first_depth_sensor(self):
                            class Sensor:
                                def get_depth_scale(self):
                                    return 0.001
                            return Sensor()

                        def as_playback(self):
                            class Playback:
                                def set_real_time(self, _on):
                                    pass
                            return Playback()
                    return Device()
            return Profile()

        def try_wait_for_frames(self, *_a):
            return (True, queue.pop(0)) if queue else (False, None)

    class RS:
        class stream:
            color = "color"

        @staticmethod
        def align(_stream):
            return None

    bag = _bag("clip.db3")
    bag._rs, bag._pipeline, bag._config = RS, Pipeline(), None
    bag._assemble = lambda frames, index, t: (index, t)
    bag.close = lambda: None
    assert list(bag) == [(0, 0.133), (1, 0.266)]


def test_db3_recording_counts_colour_frames_and_is_seekable(tmp_path):
    import sqlite3

    db = tmp_path / "clip1_03.db3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE topics (id INTEGER, name TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER, topic_id INTEGER, timestamp INTEGER)")
    con.executemany("INSERT INTO topics VALUES (?, ?)", [
        (1, "/device_0/sensor_1/Color_0/image/data"),
        (2, "/device_0/sensor_0/Depth_0/image/data"),
    ])
    con.executemany("INSERT INTO messages VALUES (?, ?, ?)",
                    [(i, 1 + i % 2, i) for i in range(10)])
    con.commit()
    con.close()
    bag = _bag(db)
    assert bag.meta.frame_count == 5
    assert bag.seekable is True


def test_ros1_bag_and_unreadable_db3_are_not_seekable(tmp_path):
    assert _bag("clip.bag").meta.frame_count is None
    assert _bag("clip.bag").seekable is False
    broken = tmp_path / "broken.db3"
    broken.write_bytes(b"not sqlite")
    assert _bag(broken).seekable is False


class TestParseRealsenseUri:
    def test_plain_is_colour(self):
        kw = parse_realsense_uri("rs://")
        assert kw["infrared"] is False
        assert kw["emitter"] is None  # colour leaves the emitter at device default
        assert "color_size" not in kw and "ir_size" not in kw

    def test_ir_shorthand(self):
        kw = parse_realsense_uri("rs://ir")
        assert kw["infrared"] is True
        assert kw["emitter"] is False  # IR defaults the projector OFF (clean image)

    def test_ir_query_form(self):
        assert parse_realsense_uri("rs://?ir=1")["infrared"] is True
        assert parse_realsense_uri("rs://?ir=0")["infrared"] is False

    def test_emitter_override(self):
        assert parse_realsense_uri("rs://ir?emitter=1")["emitter"] is True
        assert parse_realsense_uri("rs://ir?emitter=off")["emitter"] is False
        # explicit emitter on a colour stream is honoured too
        assert parse_realsense_uri("rs://?emitter=1")["emitter"] is True

    def test_ir_index(self):
        assert parse_realsense_uri("rs://ir?ir_index=2")["ir_index"] == 2
        assert parse_realsense_uri("rs://")["ir_index"] == 1

    def test_resolution_from_query(self):
        kw = parse_realsense_uri("rs://ir?w=848&h=480")
        assert kw["ir_size"] == (848, 480)
        assert "color_size" not in kw

    def test_resolution_from_query_colour(self):
        kw = parse_realsense_uri("rs://?w=1280&h=720")
        assert kw["color_size"] == (1280, 720)

    def test_no_size_uses_device_default(self):
        # The RealSense ignores any generic capture size -- without an explicit
        # w/h it carries no size and RealSenseSource picks its own profile
        # (1920x1080 colour, 1280x720 IR) to match the calibration.
        assert "color_size" not in parse_realsense_uri("rs://")
        assert "ir_size" not in parse_realsense_uri("rs://ir")

    def test_depth_and_long_range_options(self):
        kw = parse_realsense_uri(
            "rs://?depth=1&max_laser=true&max_range=8.5&spatial=3"
        )
        assert kw["with_depth"] is True
        assert kw["max_laser"] is True
        assert kw["max_range_m"] == 8.5
        assert kw["spatial_magnitude"] == 3

    def test_depth_is_off_by_default(self):
        kw = parse_realsense_uri("rs://")
        assert kw["with_depth"] is False
        assert kw["max_laser"] is False

    @pytest.mark.parametrize(
        "uri",
        [
            "rs://patient_name?depth=1",
            "rs://?depth=1&ward=7",
            "rs://?depth=1&depth=0",
            "rs://?w=1280",
            "rs://?depth=1#patient",
            "rs://?depth=maybe",
        ],
    )
    def test_rejects_unrecognised_or_identifier_bearing_components(self, uri):
        with pytest.raises(ValueError):
            parse_realsense_uri(uri)


def test_live_source_receives_exact_preflight_device_serial(monkeypatch):
    captured = {}

    class Source:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import ahfd.capture.realsense

    monkeypatch.setattr(ahfd.capture.realsense, "RealSenseSource", Source)
    open_source(
        "rs://?depth=1&emitter=1&max_laser=1",
        device_serial="raw-device-serial",
        strict_depth_controls=True,
    )
    assert captured["device_serial"] == "raw-device-serial"
    assert captured["with_depth"] is True
    assert captured["emitter"] is True
    assert captured["max_laser"] is True
    assert captured["strict_depth_controls"] is True

    with pytest.raises(ValueError, match="only for a live rs://"):
        open_source("file://clip.mp4", device_serial="raw-device-serial")


def test_realsense_config_and_active_profile_are_both_identity_bound():
    configured = []
    config = type(
        "Config",
        (),
        {"enable_device": lambda _self, serial: configured.append(serial)},
    )()
    _bind_config_to_device(config, "expected-secret")
    assert configured == ["expected-secret"]

    rs = type(
        "RS",
        (),
        {"camera_info": type("Info", (), {"serial_number": object()})},
    )()
    matching = type(
        "Device",
        (),
        {"get_info": lambda _self, _field: "expected-secret"},
    )()
    _verify_active_device_serial(rs, matching, "expected-secret")

    other = type(
        "Device",
        (),
        {"get_info": lambda _self, _field: "different-secret"},
    )()
    with pytest.raises(RuntimeError, match="does not match") as error:
        _verify_active_device_serial(rs, other, "expected-secret")
    assert "expected-secret" not in str(error.value)
    assert "different-secret" not in str(error.value)


class _Options:
    emitter_enabled = "emitter_enabled"
    laser_power = "laser_power"


class _CameraInfo:
    serial_number = "serial_number"
    usb_type_descriptor = "usb_type_descriptor"


class _Rs:
    option = _Options()
    camera_info = _CameraInfo()


class _DepthSensor:
    def __init__(self, *, supported=True, emitter_readback=1.0, laser_readback=360.0):
        self.supported = supported
        self.values = {
            "emitter_enabled": float(emitter_readback),
            "laser_power": float(laser_readback),
        }
        self.set_calls = []

    def supports(self, _option):
        return self.supported

    def get_option_range(self, _option):
        return type("Range", (), {"max": 360.0})()

    def set_option(self, option, value):
        self.set_calls.append((option, value))

    def get_option(self, option):
        return self.values[option]


def test_strict_depth_controls_require_support_apply_and_verify_readback():
    sensor = _DepthSensor()
    result = _apply_strict_depth_controls(_Rs(), sensor)
    assert sensor.set_calls == [
        ("emitter_enabled", 1.0),
        ("laser_power", 360.0),
    ]
    assert result == {
        "emitter_enabled": True,
        "laser_at_max": True,
        "laser_power": 360.0,
        "laser_power_max": 360.0,
    }

    with pytest.raises(RuntimeError, match="must expose"):
        _apply_strict_depth_controls(_Rs(), _DepthSensor(supported=False))
    with pytest.raises(RuntimeError, match="emitter readback"):
        _apply_strict_depth_controls(
            _Rs(), _DepthSensor(emitter_readback=0.0)
        )
    with pytest.raises(RuntimeError, match="laser readback"):
        _apply_strict_depth_controls(
            _Rs(), _DepthSensor(laser_readback=300.0)
        )


def test_active_usb_readback_rejects_known_usb2_without_leaking_identity():
    class Device:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def supports(self, _field):
            return True

        def get_info(self, field):
            if field == "usb_type_descriptor":
                return self.descriptor
            return "secret-serial"

    assert _verify_active_usb3(_Rs(), Device("3.2")) == "3.2"
    with pytest.raises(RuntimeError, match="not USB 3") as error:
        _verify_active_usb3(_Rs(), Device("2.1"))
    assert "secret-serial" not in str(error.value)

    for descriptor in ("", "   "):
        with pytest.raises(RuntimeError, match="could not be verified"):
            _verify_active_usb3(_Rs(), Device(descriptor))

    unsupported = Device("3.2")
    unsupported.supports = lambda _field: False
    with pytest.raises(RuntimeError, match="could not be verified"):
        _verify_active_usb3(_Rs(), unsupported)

    no_support_method = type(
        "DeviceWithoutSupports",
        (),
        {"get_info": lambda _self, _field: "3.2"},
    )()
    with pytest.raises(RuntimeError, match="could not be verified"):
        _verify_active_usb3(_Rs(), no_support_method)


def test_strict_verification_failure_is_not_retried_or_hardware_reset():
    source = object.__new__(_RealSenseBase)
    calls = {"start": 0, "stop": 0, "reset": 0}

    class Pipeline:
        def stop(self):
            calls["stop"] += 1

    def fail_verification():
        calls["start"] += 1
        raise RealSenseVerificationError("strict controls unsupported")

    source._started = False
    source._verified_depth_controls = None
    source._rs = object()
    source._pipeline = Pipeline()
    source._start = fail_verification
    source._reset_device = lambda: calls.__setitem__("reset", calls["reset"] + 1)

    with pytest.raises(RealSenseVerificationError, match="strict controls unsupported"):
        source._start_with_retry()
    assert calls == {"start": 1, "stop": 1, "reset": 0}


class TestCameraSelection:
    def test_serial_and_model_are_parsed(self):
        assert parse_realsense_uri("rs://?serial=346122072607")["device_serial"] == "346122072607"
        assert parse_realsense_uri("rs://?depth=1&model=d435f")["device_model"] == "D435F"
        assert "device_serial" not in parse_realsense_uri("rs://")

    @pytest.mark.parametrize("uri", ["rs://?serial=1&model=d435f", "rs://?serial=12;rm", "rs://?model=d435 f"])
    def test_bad_selection_is_rejected(self, uri):
        with pytest.raises(ValueError):
            parse_realsense_uri(uri)

    def _probe(self, monkeypatch, names):
        from ahfd.capture import devices, factory

        fake = devices.RealSenseProbe(
            installed=True, version="test",
            devices=tuple(devices.RealSenseDevice(name=n, serial=s, usb="3.2") for n, s in names),
        )
        monkeypatch.setattr(devices, "probe_realsense", lambda: fake)
        return factory

    def test_model_resolves_to_that_cameras_serial(self, monkeypatch):
        factory = self._probe(monkeypatch, [("Intel RealSense D435I", "111"), ("Intel RealSense D435F", "222")])
        assert factory.resolve_realsense_model("D435F") == "222"
        assert factory.resolve_realsense_model("D435I") == "111"

    def test_model_missing_or_ambiguous_fails_loudly(self, monkeypatch):
        factory = self._probe(monkeypatch, [("Intel RealSense D435I", "111")])
        with pytest.raises(ValueError, match="found 0"):
            factory.resolve_realsense_model("D435F")
        factory = self._probe(monkeypatch, [("Intel RealSense D435I", "1"), ("Intel RealSense D435I", "2")])
        with pytest.raises(ValueError, match="found 2"):
            factory.resolve_realsense_model("D435I")


class TestOpenRealsense:
    def _fake(self, monkeypatch, names):
        from ahfd.capture import devices, factory, realsense

        fake = devices.RealSenseProbe(installed=True, version="t",
                                      devices=tuple(devices.RealSenseDevice(name=n, serial=s, usb="3.2") for n, s in names))
        monkeypatch.setattr(devices, "probe_realsense", lambda: fake)
        opened = {}
        monkeypatch.setattr(realsense, "RealSenseSource", lambda **kw: opened.update(kw) or kw)
        return factory, opened

    def test_model_and_overrides_reach_the_source(self, monkeypatch):
        factory, opened = self._fake(monkeypatch, [("Intel RealSense D435I", "111"), ("Intel RealSense D435F", "222")])
        factory.open_realsense("rs://?model=d435f&max_range=10", with_depth=True)
        assert opened["device_serial"] == "222" and opened["with_depth"] is True and opened["max_range_m"] == 10.0

    def test_explicit_serial_must_match_the_model(self, monkeypatch):
        """Regression (code review): model= is no longer ignored when a serial is passed."""
        factory, _ = self._fake(monkeypatch, [("Intel RealSense D435I", "111"), ("Intel RealSense D435F", "222")])
        with pytest.raises(ValueError, match="different camera"):
            factory.open_realsense("rs://?model=d435f", device_serial="111")


def test_filesystem_type_picks_the_deepest_mount(tmp_path):
    from ahfd.debug.bag_writer import filesystem_type

    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/nvme0n1p1 / ext4 rw 0 0\n"
        "/dev/sda1 /media/anormalm/T9 fuseblk rw 0 0\n"
    )
    assert filesystem_type("/media/anormalm/T9/ah_bags_d435f/x.db3", str(mounts)) == "fuseblk"
    assert filesystem_type("/media/anormalm/T9x/x.db3", str(mounts)) == "ext4"
    assert filesystem_type("/home/user/rec/x.db3", str(mounts)) == "ext4"


def test_recorded_rate_reads_colour_frames_and_span(tmp_path):
    import sqlite3

    from ahfd.debug.bag_writer import recorded_rate

    path = tmp_path / "clip.db3"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER)")
    con.executemany("INSERT INTO topics VALUES (?, ?)", [
        (1, "/device_0/sensor_1/Color_0/image/data"), (2, "/device_0/sensor_0/Depth_0/image/data")])
    con.executemany("INSERT INTO messages (topic_id, timestamp) VALUES (?, ?)",
                    [(1, int(i * 1e8)) for i in range(11)] + [(2, 5)])
    con.commit()
    con.close()
    frames, secs = recorded_rate(path)
    assert frames == 11 and abs(secs - 1.0) < 1e-9
    assert recorded_rate(tmp_path / "clip.bag") is None


def test_only_pilot_takes_are_seeded_from_the_script():
    from ahfd.cli import _clip_index

    assert _clip_index("clip1_03") == 3 and _clip_index("clip1_10") == 10
    # 27 Oct clips share the "_NN" suffix but are not pilot takes
    assert _clip_index("s2_01") is None and _clip_index("h1_01") is None
    assert _clip_index("clip2_test") is None


def test_record_dual_refuses_to_overwrite_an_existing_take(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ahfd.cli import app

    monkeypatch.setenv("AHFD_ALLOW_RAW", "1")
    monkeypatch.setenv("COLUMNS", "200")
    config = tmp_path / "raw.yaml"
    config.write_text("privacy:\n  allow_raw_capture: true\n")
    (tmp_path / "d435i").mkdir()
    (tmp_path / "d435f").mkdir()
    taken = tmp_path / "d435f" / "s1_01.db3"
    taken.write_bytes(b"irreplaceable take")
    (tmp_path / "d435i" / "s1_02.db3").write_bytes(b"also taken")
    result = CliRunner().invoke(app, [
        "record-dual", str(tmp_path / "d435i" / "s1_01.db3"), str(taken),
        "--config", str(config), "--i-understand-raw-capture",
    ])
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output and "s1_03" in result.output  # s1_02 is taken on one camera
    assert taken.read_bytes() == b"irreplaceable take"
    assert not (tmp_path / "d435i" / "s1_01.db3").exists()


def test_record_dual_refuses_the_same_file_for_both_cameras(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ahfd.cli import app

    monkeypatch.setenv("AHFD_ALLOW_RAW", "1")
    monkeypatch.setenv("COLUMNS", "200")
    config = tmp_path / "raw.yaml"
    config.write_text("privacy:\n  allow_raw_capture: true\n")
    out = tmp_path / "s1_01.db3"
    result = CliRunner().invoke(app, ["record-dual", str(out), str(out), "--config", str(config), "--i-understand-raw-capture"])
    assert result.exit_code != 0 and "same file" in result.output
