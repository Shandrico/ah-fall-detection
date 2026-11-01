"""Pre-impact detector: balance features and the FALL_IMMINENT decision (synthetic, no camera)."""

from types import SimpleNamespace as NS

import numpy as np

from ahfd.detect.preimpact import PreImpactRule

FPS = 15.0


class FlatGround:
    """1 px = 1 cm on every plane: enough to move a body around the floor."""

    def pixel_to_plane(self, u, v, h=0.0):
        return (u / 100.0, v / 100.0)


def body(cx, lean_px=0.0, *, feet_x=None):
    """17 keypoints: hips at cx, shoulders `lean_px` sideways of them, ankles under feet_x."""
    k = np.zeros((17, 2))
    fx = cx if feet_x is None else feet_x
    k[5], k[6] = (cx - 20 + lean_px, 100), (cx + 20 + lean_px, 100)
    k[11], k[12] = (cx - 15, 200), (cx + 15, 200)
    k[15], k[16] = (fx - 15, 380), (fx + 15, 380)
    return k


def frame(t, kp, *, tid=1, h_torso=1.1, v_z=0.0, in_bed=False):
    person = NS(track_id=tid, keypoints=kp, scores=np.ones(17))
    f = NS(t=t, h_torso=h_torso, v_z=v_z, zones=(), supported_by_bed="b1" if in_bed else None,
           supported_by_surface=None, knee_angle=175.0, hip_angle=175.0, leg_tilt=3.0, bed_edge_distance_m=None)
    return person, f


def run(det, frames):
    return [e for p, f in frames if (e := det.update(p, f)) is not None]


def standing(t0, seconds, **kw):
    return [frame(t0 + i / FPS, body(300.0), **kw) for i in range(int(seconds * FPS))]


def tipping(t0, **kw):
    """Hips sliding 1.5 m/s off the feet while the trunk tips ~200 deg/s."""
    out = []
    for i in range(10):
        t = t0 + i / FPS
        out.append(frame(t, body(300.0 + 150.0 * (t - t0), lean_px=60.0 * i, feet_x=300.0), **kw))
    return out


def test_standing_still_is_silent():
    assert run(PreImpactRule(FlatGround()), standing(0.0, 10.0)) == []


def test_tipping_off_the_feet_fires_once():
    events = run(PreImpactRule(FlatGround()), standing(0.0, 2.0) + tipping(2.0))
    assert [e.type for e in events] == ["FALL_IMMINENT"]
    assert events[0].evidence["mos"] < 0.0 and events[0].evidence["tilt_rate"] > 80.0


def test_not_while_supported_by_a_bed():
    assert run(PreImpactRule(FlatGround()), standing(0.0, 2.0, in_bed=True) + tipping(2.0, in_bed=True)) == []


def test_not_while_getting_up_off_the_floor():
    frames = standing(0.0, 2.0, h_torso=0.2) + tipping(2.0)  # torso at floor level moments ago
    assert run(PreImpactRule(FlatGround()), frames) == []


def test_hidden_joints_give_no_features():
    det = PreImpactRule(FlatGround())
    p, f = frame(0.0, body(300.0))
    p.scores = np.zeros(17)
    assert det.update(p, f) is None and det.stability.update(p, f) is None
