"""RTMPose backend construction guards.

Like the RTMO backend tests, these never load a model -- every check runs
before rtmlib is touched, so the suite stays offline and fast. What is being
verified is that a misconfiguration fails at construction with a clear message,
not after a model download.
"""

from __future__ import annotations

import pytest

from ahfd.config import PoseConfig
from ahfd.pose.base import build_estimator
from ahfd.pose.rtmpose import VALID_MODES, RTMPoseEstimator


class TestModeGuard:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            RTMPoseEstimator(mode="ludicrous")

    def test_valid_modes_are_the_three_rtmlib_presets(self):
        assert VALID_MODES == {"performance", "balanced", "lightweight"}


class TestRuntimeGuard:
    def test_rejects_unknown_runtime(self):
        with pytest.raises(ValueError, match="unknown runtime"):
            RTMPoseEstimator(mode="balanced", runtime="tensorrt")

    def test_rejects_device_the_runtime_cannot_serve(self):
        with pytest.raises(ValueError, match="does not support device"):
            RTMPoseEstimator(mode="balanced", runtime="openvino", device="cuda")


class TestBuildEstimator:
    def test_backend_rtmpose_is_recognised(self, monkeypatch):
        """build_estimator must route 'rtmpose' to the right class.

        rtmlib's Body is stubbed so nothing is downloaded and the test stays
        offline and instant. The stub records that it was called, which is the
        actual assertion: the dispatch reached RTMPose construction.
        """
        called = {}

        class FakeBody:
            def __init__(self, **kwargs):
                called.update(kwargs)

        import rtmlib

        monkeypatch.setattr(rtmlib, "Body", FakeBody)
        est = build_estimator(PoseConfig(backend="rtmpose", mode="balanced"))
        assert est.name == "rtmpose-balanced"
        assert called["mode"] == "balanced"

    def test_unknown_backend_still_rejected(self):
        with pytest.raises(ValueError, match="unknown pose backend"):
            build_estimator(PoseConfig(backend="posenet"))

    def test_both_backends_are_listed_in_the_error(self):
        with pytest.raises(ValueError) as excinfo:
            build_estimator(PoseConfig(backend="nope"))
        assert "rtmo" in str(excinfo.value)
        assert "rtmpose" in str(excinfo.value)
