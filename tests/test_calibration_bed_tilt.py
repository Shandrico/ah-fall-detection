"""Per-camera bed tilt bands in the calibration (viewpoint-dependent)."""

import pytest
import yaml

from ahfd.geometry.calibration import load_calibration

CAMERA = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
          "height_m": 2.6, "pitch_deg": 30.0}
GOOD = {"reclined_enter": 26, "reclined_exit": 22, "upright_enter": 18, "upright_exit": 22}


def _write(tmp_path, **extra):
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump({"camera_id": "cam", "camera": CAMERA, **extra}))
    return path


def test_defaults_when_absent(tmp_path):
    assert load_calibration(_write(tmp_path)).bed_tilt_deg is None


def test_override_is_loaded(tmp_path):
    calib = load_calibration(_write(tmp_path, bed_tilt_deg=GOOD))
    assert calib.bed_tilt_deg == {k: float(v) for k, v in GOOD.items()}


@pytest.mark.parametrize("bad", [
    {"reclined_enter": 26, "reclined_exit": 22, "upright_enter": 18},          # missing key
    {**GOOD, "upright_exit": 30},                                                # upright band crosses reclined
    {**GOOD, "reclined_exit": 26},                                               # no reclined hysteresis
    {**GOOD, "reclined_enter": 120},                                             # not a tilt
])
def test_invalid_override_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        load_calibration(_write(tmp_path, bed_tilt_deg=bad))


def test_override_reaches_the_bed_machine(tmp_path):
    from ahfd.capture.base import SourceMeta
    from ahfd.cli import _build_detection
    from ahfd.config import Config

    cfg = Config()
    cfg.alert.console = False
    _ext, machine, _sink, _calib = _build_detection(
        cfg, str(_write(tmp_path, bed_tilt_deg=GOOD)), SourceMeta(uri="x", width=640, height=480, fps=30.0)
    )
    th = machine.bed.th
    assert (th.reclined_tilt_enter_deg, th.reclined_tilt_exit_deg,
            th.upright_tilt_enter_deg, th.upright_tilt_exit_deg) == (26, 22, 18, 22)
