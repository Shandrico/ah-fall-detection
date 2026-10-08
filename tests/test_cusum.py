"""Tests for the causal, rest-gated CUSUM onset detector."""

import math

import pytest

from ahfd.detect.cusum import CusumConfig, CusumOnset


def _series(values, dt=0.1, t0=0.0):
    return [(t0 + i * dt, v) for i, v in enumerate(values)]


def _rest(det, values, dt=0.1, t0=0.0):
    return [
        det.update(value, t, baseline_eligible=True)
        for t, value in _series(values, dt=dt, t0=t0)
    ]


def test_baseline_requires_explicit_rest_eligibility():
    det = CusumOnset()
    det.run(_series([0.9] * 100))
    assert not det.armed

    _rest(det, [0.6] * CusumConfig().warmup_samples, t0=10.0)
    assert det.armed


def test_flat_baseline_never_triggers():
    det = CusumOnset()
    noise = [0.60 + (0.015 if i % 2 else -0.015) for i in range(200)]
    out = _rest(det, noise)
    assert not any(sample.onset for sample in out)
    assert out[-1].g < CusumConfig().threshold


def test_sustained_rise_triggers_once_and_only_after_rise():
    det = CusumOnset()
    rest = [0.60] * 30
    _rest(det, rest)
    rise = [0.60 + 0.03 * i for i in range(1, 20)]
    hold = [0.90] * 30
    out = det.run(_series(rise + hold, t0=3.0))
    fires = [i for i, sample in enumerate(out) if sample.onset]
    assert len(fires) == 1
    assert fires[0] < len(rise)


def test_brief_reach_and_return_does_not_trigger():
    det = CusumOnset()
    _rest(det, [0.60] * 30)
    out = det.run(_series([0.66, 0.67, 0.66] + [0.60] * 30, t0=3.0))
    assert not any(sample.onset for sample in out)


def test_elevated_hold_is_never_automatically_absorbed():
    det = CusumOnset()
    _rest(det, [0.60] * 30)
    samples = []
    for t, value in _series(
        [0.65, 0.72, 0.80, 0.90] + [0.90] * 100, t0=3.0
    ):
        # Deliberately keep eligibility true to prove the detector's fired latch
        # prevents an elevated stable signal from becoming a new rest baseline.
        samples.append(det.update(value, t, baseline_eligible=True))
    assert sum(sample.onset for sample in samples) == 1
    assert samples[-1].baseline_mean == pytest.approx(0.60, abs=0.03)
    assert samples[-1].g > CusumConfig().threshold


def test_missing_samples_pause_without_counting_as_rest():
    det = CusumOnset()
    _rest(det, [0.60] * 20)
    before = det.update(0.72, 2.0).g
    missing = det.update(None, 2.1, valid=False)
    after = det.update(0.72, 2.2)

    uninterrupted = CusumOnset()
    _rest(uninterrupted, [0.60] * 20)
    uninterrupted_before = uninterrupted.update(0.72, 2.0).g
    uninterrupted_after = uninterrupted.update(0.72, 2.1).g

    assert not missing.valid
    assert math.isfinite(missing.g)
    assert missing.g == before
    assert after.g > before
    # The resumed value contributes one observation, not the two elapsed
    # reference-rate steps that include the unobservable interval.
    assert after.g - before == pytest.approx(
        uninterrupted_after - uninterrupted_before
    )


def test_long_gap_resets_the_detector():
    det = CusumOnset()
    _rest(det, [0.60] * 15)
    assert det.armed
    det.update(None, t=100.0, valid=False)
    assert not det.armed


def test_reset_disarms():
    det = CusumOnset()
    _rest(det, [0.60] * (CusumConfig().warmup_samples + 2))
    assert det.armed
    det.reset()
    assert not det.armed


def _onset_time(hz):
    cfg = CusumConfig(warmup_samples=5, reference_hz=10.0)
    det = CusumOnset(cfg)
    dt = 1.0 / hz
    t = 0.0
    while t < 3.0:
        det.update(0.60, t, baseline_eligible=True)
        t += dt
    while t < 8.0:
        value = 0.60 + 0.08 * (t - 3.0)
        sample = det.update(value, t)
        if sample.onset:
            return t
        t += dt
    raise AssertionError("onset did not fire")


def test_elapsed_time_normalisation_reduces_frame_rate_dependence():
    assert abs(_onset_time(5.0) - _onset_time(15.0)) < 0.35
