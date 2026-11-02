"""Bed-edge warning: geometry, debounce, once per bout, re-arm (synthetic, no camera)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from ahfd.detect.edge_warning import FEATURES, EdgeWarning, raw_geometry
from ahfd.geometry.zones import Zone

FPS = 15.0
BED = Zone(name="bed1", kind="bed", polygon=[(0.0, 0.0), (0.9, 0.0), (0.9, 2.0), (0.0, 2.0)], top_m=0.5)


class FlatGround:
    """1 px = 1 cm on every plane."""

    def pixel_to_plane(self, u, v, h=0.0):
        return (u / 100.0, v / 100.0)


class ScoreModel:
    """Scores 'at the edge' from the ankle distance feature: feet outside the bed outline -> 0.95."""

    def predict_proba(self, X):
        ank = X[0, FEATURES.index("ank_d")]
        p = 0.95 if np.isfinite(ank) and ank < 0 else 0.05
        return np.array([[1 - p, p]])


def pose(hips_x, ankles_x, tid=1):
    k = np.zeros((17, 2))
    k[11], k[12] = (hips_x, 100), (hips_x, 110)
    k[15], k[16] = (ankles_x, 100), (ankles_x, 110)
    return NS(track_id=tid, keypoints=k, scores=np.ones(17))


def feats(t):
    return NS(t=t, knee_angle=95.0, torso_tilt=5.0, associated_bed="bed1", supported_by_bed="bed1")


def snap(phase="EDGE_SITTING"):
    return NS(phase=phase, bed_id="bed1", support="PARTIAL")


def det(**kw):
    return EdgeWarning(ground=FlatGround(), beds={"bed1": BED}, bundle={"model": ScoreModel(), "features": list(FEATURES)}, **kw)


def run(d, seq):
    return [e for p, f, s in seq if (e := d.update(p, f, s)) is not None]


def bout(t0, seconds, *, at_edge=True, phase="EDGE_SITTING"):
    # hips 10 cm inside the bed's right side; ankles 30 cm outside it (legs over the edge) or on the mattress
    ank = 120.0 if at_edge else 50.0
    return [(pose(80.0, ank), feats(t0 + i / FPS), snap(phase)) for i in range(int(seconds * FPS))]


def test_raw_geometry_signs():
    hip_d, ank_d, knee, tilt = raw_geometry(pose(80.0, 120.0), feats(0.0), FlatGround(), BED)
    assert hip_d == pytest.approx(0.10) and ank_d == pytest.approx(-0.30)
    assert (knee, tilt) == (95.0, 5.0)
    assert raw_geometry(pose(80.0, 120.0), feats(0.0), FlatGround(), None)[:2] == (None, None)


def test_sitting_in_bed_is_silent():
    assert run(det(), bout(0.0, 20.0, at_edge=False, phase="UPRIGHT_IN_BED")) == []


def test_fires_once_after_dwell():
    events = run(det(dwell_s=2.0), bout(0.0, 10.0))
    assert [e.type for e in events] == ["BED_EDGE_WARNING"]
    assert events[0].t_alert - events[0].t_trigger == pytest.approx(2.0, abs=0.1)
    assert events[0].zone == "bed1" and events[0].evidence["ank_d"] < 0


def test_short_edge_touch_does_not_fire():
    assert run(det(dwell_s=2.0), bout(0.0, 1.0)) == []


def test_rearms_after_lying_back_down():
    seq = bout(0.0, 4.0) + bout(4.0, 3.0, at_edge=False, phase="UPRIGHT_IN_BED") + bout(7.0, 4.0)
    assert len(run(det(rearm_s=5.0), seq)) == 1  # 3 s back in bed: still the same bout
    seq = bout(0.0, 4.0) + bout(4.0, 6.0, at_edge=False, phase="UPRIGHT_IN_BED") + bout(10.0, 4.0)
    assert len(run(det(rearm_s=5.0), seq)) == 2


def test_out_of_bed_never_warns():
    assert run(det(), bout(0.0, 10.0, phase="OUT_OF_BED")) == []


def test_rejects_model_with_other_features():
    with pytest.raises(ValueError):
        EdgeWarning(ground=FlatGround(), beds={}, bundle={"model": ScoreModel(), "features": ["x"]})
