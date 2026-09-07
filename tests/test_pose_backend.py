"""Pose backend construction and configuration guards.

These tests never load a model. Every check they exercise runs before the
import of rtmlib, so the suite stays fast and works offline -- which is the
point: a misconfiguration should be caught at construction, not after a 35 MB
download and a confusing tensor-shape error.
"""

from __future__ import annotations

import pytest

from ahfd.config import PoseConfig
from ahfd.pose.base import build_estimator
from ahfd.pose.rtmo import FIXED_INPUT_SIZE, MODEL_URLS, RTMOEstimator


class TestModelSize:
    def test_rejects_unknown_size(self):
        with pytest.raises(ValueError, match="model_size"):
            RTMOEstimator(model_size="xl")

    def test_known_sizes_are_the_three_published_ones(self):
        assert set(MODEL_URLS) == {"s", "m", "l"}

    def test_all_urls_point_at_the_official_mmpose_host(self):
        for url in MODEL_URLS.values():
            assert url.startswith("https://download.openmmlab.com/mmpose/")


class TestInputSizeGuard:
    """The published RTMO ONNX graphs are static 640x640.

    Regression test for a real trap: passing any other size passes every
    rtmlib check and then dies inside onnxruntime with "Got invalid dimensions
    for input", tens of seconds and one model download later.
    """

    @pytest.mark.parametrize("size", [(512, 512), (416, 416), (640, 480), (1280, 1280)])
    def test_rejects_non_native_input_size(self, size):
        with pytest.raises(ValueError, match="static"):
            RTMOEstimator(model_size="s", model_input_size=size)

    def test_error_names_the_real_fix(self):
        with pytest.raises(ValueError) as excinfo:
            RTMOEstimator(model_size="s", model_input_size=(416, 416))
        assert "Re-export" in str(excinfo.value)

    def test_native_size_is_640(self):
        assert FIXED_INPUT_SIZE == (640, 640)


class TestRuntimeGuard:
    def test_rejects_unknown_runtime(self):
        with pytest.raises(ValueError, match="unknown runtime"):
            RTMOEstimator(model_size="s", runtime="tensorrt")

    def test_rejects_device_the_runtime_cannot_serve(self):
        # openvino has no cuda device; onnxruntime has no npu.
        with pytest.raises(ValueError, match="does not support device"):
            RTMOEstimator(model_size="s", runtime="openvino", device="cuda")
        with pytest.raises(ValueError, match="does not support device"):
            RTMOEstimator(model_size="s", runtime="onnxruntime", device="npu")


class TestBuildEstimator:
    def test_rejects_unknown_backend(self):
        cfg = PoseConfig(backend="detectron2")
        with pytest.raises(ValueError, match="unknown pose backend"):
            build_estimator(cfg)

    def test_config_default_input_size_matches_the_model(self):
        """The shipped config must not be the thing that trips the guard."""
        assert tuple(PoseConfig().model_input_size) == FIXED_INPUT_SIZE

    def test_config_default_backend_is_rtmo(self):
        assert PoseConfig().backend == "rtmo"
