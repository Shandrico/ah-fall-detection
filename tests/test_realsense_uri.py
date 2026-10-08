"""Parsing of rs:// capture URIs into RealSenseSource kwargs (no hardware)."""

from __future__ import annotations

import pytest

from ahfd.capture.factory import open_source, parse_realsense_uri
from ahfd.capture.realsense import (
    RealSenseVerificationError,
    _RealSenseBase,
    _apply_strict_depth_controls,
    _bind_config_to_device,
    _verify_active_usb3,
    _verify_active_device_serial,
)


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
