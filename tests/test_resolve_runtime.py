"""`runtime: auto` picks a pose runtime that exists on this machine."""

import importlib.util

from ahfd.pose import resolve_runtime


def test_explicit_runtime_is_untouched():
    assert resolve_runtime("openvino", "npu") == ("openvino", "npu")
    assert resolve_runtime("onnxruntime", "cpu") == ("onnxruntime", "cpu")


def test_auto_without_openvino_uses_onnxruntime(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "openvino" else real(name, *a),
    )
    runtime, device = resolve_runtime("auto", "gpu")
    assert runtime == "onnxruntime" and device in ("cuda", "cpu")
