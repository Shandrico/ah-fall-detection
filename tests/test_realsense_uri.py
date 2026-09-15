"""Parsing of rs:// capture URIs into RealSenseSource kwargs (no hardware)."""

from __future__ import annotations

from ahfd.capture.factory import parse_realsense_uri


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

    def test_resolution_fallback_args(self):
        kw = parse_realsense_uri("rs://", width=640, height=480)
        assert kw["color_size"] == (640, 480)
