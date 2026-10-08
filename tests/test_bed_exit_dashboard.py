"""Dashboard integration for the bed-exit branch.

Two things to pin down: a high-risk bed exit must reach the triage queue (it is
graded by severity, not by being one of the hard-coded alert types), and a bed
drawn on the RGB feed must back-project through the live calibration and persist.
Both run camera-free -- the picker uses the FakeRunner seam and a temp
calibration, and the corners handed to it are real floor points projected to
pixels, so the round-trip is exercised for real.
"""

from __future__ import annotations

import yaml

from ahfd.config import Config, SourceOption
from ahfd.dashboard import DashboardController, DashboardState
from ahfd.geometry.calibration import load_calibration


class FakeRunner:
    def __init__(self, source_uri, cfg, calib_path, state, show_rgb, *, gen=0, log=None):
        self.source_uri = source_uri
        self.alive = False

    def start(self):
        self.alive = True

    def stop(self, timeout=5.0):
        self.alive = False


# --------------------------------------------------------------------------- #
# Severity-driven triage

def test_high_risk_bed_exit_becomes_open_alert():
    state = DashboardState()
    state.publish_event({
        "event_id": "a1", "type": "BED_EXIT_CONFIRMED", "severity": 3,
        "track_id": 1, "zone": "bed_1", "evidence": {},
    })
    snap = state.snapshot()
    assert snap["open_count"] == 1
    assert snap["counts"]["bed_exit_confirmed"] == 1


def test_low_risk_bed_exit_stays_in_the_log_only():
    state = DashboardState()
    for etype, sev in [("BED_EXIT_LIMB", 1), ("BED_EXIT_RISK", 2)]:
        state.publish_event({
            "event_id": etype, "type": etype, "severity": sev,
            "track_id": 1, "zone": "bed_1", "evidence": {},
        })
    snap = state.snapshot()
    assert snap["open_count"] == 0            # neither pages a nurse
    assert len(snap["events"]) == 2           # but both are logged


# --------------------------------------------------------------------------- #
# Drawing a bed zone on the RGB feed

def _write_calib(path):
    path.write_text(yaml.safe_dump({
        "camera_id": "dev",
        "camera": {"width": 1920, "height": 1080, "hfov_deg": 69.4, "vfov_deg": 42.5,
                   "height_m": 2.6, "pitch_deg": 20.0},
        "zones": [],
    }), encoding="utf-8")


def _controller(tmp_path):
    calib = tmp_path / "cam.yaml"
    _write_calib(calib)
    cfg = Config()
    cfg.source = "webcam://0"
    cfg.detect.enabled = True
    cfg.dashboard.sources = [
        SourceOption(label="dev", uri="webcam://0", calibration=str(calib))
    ]
    state = DashboardState()
    ctrl = DashboardController(
        cfg, str(calib), state,
        source="webcam://0",
        runner_factory=FakeRunner,
        probe=lambda: __import__("ahfd.capture.devices", fromlist=["RealSenseProbe"]).RealSenseProbe(installed=False),
    )
    return ctrl, calib


def test_add_bed_zone_backprojects_and_persists(tmp_path):
    ctrl, calib_path = _controller(tmp_path)
    ground = load_calibration(calib_path).ground

    # Four real floor corners at mattress height, projected to pixels -- exactly
    # what the browser would send after a click.
    top = 0.55
    corners_m = [(-0.5, 6.0), (0.6, 6.0), (0.6, 4.0), (-0.5, 4.0)]
    clicks = [ground.world_to_pixel(x, y, top) for (x, y) in corners_m]
    assert all(c is not None for c in clicks)

    code, result = ctrl.add_bed_zone(
        points=[list(c) for c in clicks], name="bed_1", top_m=top, risk_level="high"
    )
    assert code == 200 and result["ok"], result

    calib = load_calibration(calib_path)
    beds = calib.zones.beds()
    assert len(beds) == 1
    bed = beds[0]
    assert bed.name == "bed_1" and bed.risk_level == "high"
    assert len(bed.polygon) == 4
    assert bed.edges  # default Hill-Rom rails attached
    # The saved polygon matches the floor corners we started from.
    for (gx, gy), (px, py) in zip(corners_m, bed.polygon):
        assert abs(gx - px) < 0.05 and abs(gy - py) < 0.05


def test_add_bed_zone_rejects_bad_risk(tmp_path):
    ctrl, _ = _controller(tmp_path)
    code, result = ctrl.add_bed_zone(
        points=[[900, 600], [1000, 600], [1000, 700], [900, 700]],
        name="bad", top_m=0.55, risk_level="catastrophic",
    )
    assert code == 400 and not result["ok"]
