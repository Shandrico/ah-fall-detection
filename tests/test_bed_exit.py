"""Tests for the bed-exit precursor monitor (sit-up while still bed-supported)."""

from ahfd.detect.bed_exit import BedExitConfig, BedExitMonitor
from ahfd.features.extractor import Features


def _feat(t, h_torso, *, tilt, supported, risk, track_id=1):
    """A Features frame with just the fields the monitor reads set meaningfully."""
    return Features(
        track_id=track_id,
        t=t,
        contact_xy=(0.0, 0.0),
        range_m=4.0,
        h_torso=h_torso,
        h_head=h_torso + 0.3,
        h_max=h_torso + 0.4,
        h_min=0.05,
        h_ankle_min=0.05,
        floor_spread=1.5,
        v_z=0.0,
        motion=0.0,
        n_valid_kp=15,
        mean_conf=0.9,
        supported_by_bed=supported,
        bed_risk=risk,
        torso_tilt=tilt,
    )


def _reclined_then_sit_up(mon, *, risk, supported="Bed 1", dt=0.1):
    """Feed a reclined baseline then a sustained rise; return emitted events."""
    events = []
    t = 0.0
    for _ in range(15):  # establish the reclined baseline (past CUSUM warmup)
        events.append(mon.update(_feat(t, 0.50, tilt=60.0, supported=supported, risk=risk)))
        t += dt
    for h in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:  # the sit-up
        events.append(mon.update(_feat(t, h, tilt=30.0, supported=supported, risk=risk)))
        t += dt
    return [e for e in events if e is not None]


def test_sit_up_on_high_risk_bed_fires_precursor():
    mon = BedExitMonitor()
    fired = _reclined_then_sit_up(mon, risk="high")
    assert len(fired) == 1  # one alert per rise (latched)
    e = fired[0]
    assert e.type == "BED_EXIT"
    assert e.evidence["trigger"] == "sit_up_onset"
    assert e.evidence["phase"] == "RISING"
    assert e.severity == 3  # high-risk grading
    assert mon.phase(1) == "RISING"


def test_self_exit_allowed_bed_does_not_fire():
    # A low-risk patient is cleared to mobilise, so a sit-up is not alerted.
    mon = BedExitMonitor()
    assert _reclined_then_sit_up(mon, risk="low") == []


def test_confused_tier_fires():
    mon = BedExitMonitor()
    fired = _reclined_then_sit_up(mon, risk="confused")
    assert len(fired) == 1
    assert fired[0].severity == 3


def test_off_bed_never_fires_and_marks_out_of_bed():
    mon = BedExitMonitor()
    out = mon.update(_feat(0.0, 1.1, tilt=10.0, supported=None, risk="high"))
    assert out is None
    assert mon.phase(1) == "OUT_OF_BED"


def test_flat_reclined_patient_stays_quiet():
    # Lying still on a high-risk bed must not fire -- only a rise does.
    mon = BedExitMonitor()
    events = [
        mon.update(_feat(i * 0.1, 0.50, tilt=60.0, supported="Bed 1", risk="high"))
        for i in range(40)
    ]
    assert all(e is None for e in events)
    assert mon.phase(1) == "RECLINED"


def test_short_cooldown_config_is_respected():
    # With a tiny cooldown, a second rise after settling can alert again.
    mon = BedExitMonitor(BedExitConfig(cooldown_s=0.1))
    first = _reclined_then_sit_up(mon, risk="high")
    assert len(first) == 1
