"""Config <-> thresholds consistency.

`DetectConfig.to_thresholds()` rebuilds a `FallThresholds` from the config, so
any field present in both must share a default -- otherwise the config path
silently overrides the code default and the two disagree. That exact drift
already bit once (down_h_torso was 0.90 in the thresholds but 0.60 in the
config, so every CLI run used the wrong value while the tests, which build
FallThresholds directly, passed). This test makes that impossible to
reintroduce.
"""

from __future__ import annotations

import dataclasses

from ahfd.config import DetectConfig
from ahfd.detect import FallThresholds


def test_shared_defaults_agree():
    cfg = DetectConfig()  # all defaults
    th_from_cfg = cfg.to_thresholds()
    th_direct = FallThresholds()  # code defaults

    for field in dataclasses.fields(FallThresholds):
        name = field.name
        assert getattr(th_from_cfg, name) == getattr(th_direct, name), (
            "default for " + name + " differs between DetectConfig ("
            + repr(getattr(th_from_cfg, name)) + ") and FallThresholds ("
            + repr(getattr(th_direct, name)) + ") -- they must match or the "
            "config path silently overrides the code default"
        )


def test_to_thresholds_drops_enabled():
    """`enabled` is config-only; it must not leak into FallThresholds."""
    th = DetectConfig(enabled=True).to_thresholds()
    assert not hasattr(th, "enabled")


def test_every_threshold_field_is_in_the_config():
    """A new FallThresholds knob must also be exposed in DetectConfig, or it
    can never be tuned from a YAML."""
    cfg_fields = set(DetectConfig.model_fields)
    for field in dataclasses.fields(FallThresholds):
        assert field.name in cfg_fields, (
            field.name + " is a FallThresholds knob with no DetectConfig field, "
            "so it cannot be set from config"
        )
