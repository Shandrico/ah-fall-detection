"""Tests for the causal CUSUM sit-up onset detector."""

import math

from ahfd.detect.cusum import CusumConfig, CusumOnset


def _series(values, dt=0.1, t0=0.0):
    return [(t0 + i * dt, v) for i, v in enumerate(values)]


def test_flat_baseline_never_triggers():
    # A resting patient with small sensor jitter must not fire.
    det = CusumOnset()
    noise = [0.60 + (0.015 if i % 2 else -0.015) for i in range(200)]
    out = det.run(_series(noise))
    assert not any(s.onset for s in out)
    assert out[-1].g < CusumConfig().threshold


def test_sustained_rise_triggers_once_and_only_after_onset():
    det = CusumOnset()
    rest = [0.60] * 30
    rise = [0.60 + 0.03 * i for i in range(1, 20)]  # slow, sustained upper-body rise
    hold = [0.90] * 30
    out = det.run(_series(rest + rise + hold))
    fires = [i for i, s in enumerate(out) if s.onset]
    assert len(fires) == 1              # latched: one onset per rise
    assert fires[0] >= len(rest)        # causal: never before the rise begins


def test_brief_reach_and_return_does_not_trigger():
    # Reaching for something and returning is the classic benign confounder.
    det = CusumOnset()
    out = det.run(_series([0.60] * 30 + [0.66, 0.67, 0.66] + [0.60] * 30))
    assert not any(s.onset for s in out)


def test_baseline_not_absorbed_during_a_real_rise():
    # A slow ramp must still fire even though the baseline updates during rest --
    # the rise itself must never be quietly folded into the baseline.
    det = CusumOnset()
    out = det.run(_series([0.60] * 20 + [0.60 + 0.02 * i for i in range(1, 40)]))
    assert any(s.onset for s in out)


def test_missing_samples_pause_without_crashing():
    det = CusumOnset()
    vals = [0.60] * 15 + [None, float("nan"), 0.61, None] + [0.60] * 10
    out = det.run(_series(vals))
    assert all(math.isfinite(s.g) for s in out)
    assert not any(s.onset for s in out)


def test_long_gap_resets_the_detector():
    det = CusumOnset()
    det.run(_series([0.60] * 15))
    assert det.armed
    # a valid sample far in the future = reassociation -> baseline dropped
    det.update(0.60, t=100.0)
    assert not det.armed


def test_arms_after_warmup_then_reset_disarms():
    det = CusumOnset()
    assert not det.armed
    det.run(_series([0.60] * (CusumConfig().warmup_samples + 2)))
    assert det.armed
    det.reset()
    assert not det.armed
