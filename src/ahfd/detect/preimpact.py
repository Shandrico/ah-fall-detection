"""Pre-impact fall detection: flag a fall while the body is still going down.

The fall state machine waits for the body to land. This looks at balance
instead, using the biomechanics measure for "can no longer recover without a
step": Hof's extrapolated centre of mass (XCoM). The centre of mass plus its
velocity scaled by the pendulum eigenfrequency, compared with the base of
support under the feet. When the XCoM leaves the base of support AND the trunk
is rotating fast, the body is falling, typically 0.5-1 s before it lands.

Everything is colour-only geometry, so it behaves the same on any calibrated
camera: joint pixels are ray-cast to horizontal planes at nominal standing
heights (hips 0.90 m, shoulders 1.35 m), ankles to the floor. Depth heights are
not used -- on the D435i and the pilot recordings they are biased.

Two deciders share these features: a fixed rule (``PreImpactRule``) and a
trained classifier (``PreImpactModel``, a joblib dict from
``~/.ahfd-work/preimpact_train_*.py``). Both emit ``FALL_IMMINENT`` at most once
per ``refractory_s``. Not while the person is supported by a bed: lying down or
sitting back into bed is a large, fast, legitimate trunk rotation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ahfd.detect.events import Event

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16
HIP_H, SHOULDER_H = 0.90, 1.35  # nominal standing heights for the ray casts
COM_H = 0.7 * HIP_H + 0.3 * SHOULDER_H  # pendulum length for the XCoM
FOOT_MARGIN_M = 0.10  # base of support reaches past the ankle centres
G = 9.81

# The per-frame feature vector, in order. Window maxima/minima are causal.
FEATURES = (
    "mos",  # margin of stability, m: + inside the base of support, - outside
    "mos_min_05",  # lowest over the last 0.5 s
    "com_speed",  # horizontal CoM speed, m/s
    "tilt",  # trunk angle from image-vertical, deg
    "tilt_rate",  # trunk rotation speed, deg/s (0.3 s slope)
    "tilt_rate_max_05",
    "h_torso",  # monocular torso height, m (Features)
    "v_z",  # vertical torso velocity, m/s (Features)
    "v_z_min_05",
    "tilt_rate_signed",  # + trunk tipping over, - straightening up
    "h_torso_drop_05",  # torso height lost over the last 0.5 s, m
    "knee_angle",  # image degrees, 180 = straight (Features)
    "hip_angle",
    "leg_tilt",
    "bed_edge_distance_m",  # + inside the bed outline, - outside (Features)
)
# Getting up off the floor is a large trunk rotation with the XCoM outside the
# feet, but the fall's own alert is still open then: no new pre-impact alert
# within this long of the torso last being at floor level.
DOWN_H = 0.35
DOWN_HOLDOFF_S = 30.0


@dataclass
class _Track:
    com: deque = field(default_factory=deque)  # (t, xy)
    tilt: deque = field(default_factory=deque)  # (t, deg)
    recent: deque = field(default_factory=deque)  # (t, mos, tilt_rate, v_z, h_torso)
    streak: int = 0
    down_t: float | None = None  # last time the torso was at floor level
    last_alert_t: float | None = None


def _slope(samples) -> float | np.ndarray:
    ts = np.array([s[0] for s in samples], float)
    ys = np.array([s[1] for s in samples], float)
    ts = ts - ts.mean()
    denom = float((ts**2).sum())
    if denom <= 0.0:
        return np.zeros(ys.shape[1:]) if ys.ndim > 1 else 0.0
    return (ts.reshape(-1, *([1] * (ys.ndim - 1))) * (ys - ys.mean(0))).sum(0) / denom


class StabilityFeatures:
    """Per-track causal balance features from keypoints + the floor model."""

    def __init__(self, ground, *, min_score: float = 0.3, slope_window_s: float = 0.3, recent_s: float = 0.5):
        self.ground = ground
        self.min_score = min_score
        self.slope_window_s = slope_window_s
        self.recent_s = recent_s
        self._tracks: dict[int, _Track] = {}

    def retain_only(self, live_ids: set[int]) -> None:
        for tid in list(self._tracks):
            if tid not in live_ids:
                del self._tracks[tid]

    def _plane(self, kp, idx: int, h: float):
        u, v = kp[idx]
        xy = self.ground.pixel_to_plane(float(u), float(v), h)
        return None if xy is None else np.array(xy, float)

    def update(self, person, features) -> dict | None:
        """Feature dict for this frame, or None if the joints needed are not visible."""
        tid = person.track_id
        if tid is None or features is None:
            return None
        sc, kp = person.scores, person.keypoints
        if min(sc[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE)) < self.min_score:
            return None
        pts = [self._plane(kp, i, h) for i, h in
               ((L_HIP, HIP_H), (R_HIP, HIP_H), (L_SHOULDER, SHOULDER_H), (R_SHOULDER, SHOULDER_H))]
        feet = [self._plane(kp, i, 0.0) for i in (L_ANKLE, R_ANKLE)]
        if any(p is None for p in pts + feet):
            return None
        hip, sh = (pts[0] + pts[1]) / 2, (pts[2] + pts[3]) / 2
        com = 0.7 * hip + 0.3 * sh
        t = features.t
        tr = self._tracks.setdefault(tid, _Track())

        tr.com.append((t, com))
        sm, hm = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2, (kp[L_HIP] + kp[R_HIP]) / 2
        tilt = math.degrees(math.atan2(abs(float(sm[0] - hm[0])), float(hm[1] - sm[1])))
        tr.tilt.append((t, tilt))
        for q in (tr.com, tr.tilt):
            while q and t - q[0][0] > self.slope_window_s:
                q.popleft()
        if len(tr.com) < 3:
            return None
        v = np.asarray(_slope(tr.com))
        xcom = com + v / math.sqrt(G / COM_H)
        bos_c = (feet[0] + feet[1]) / 2
        bos_r = float(np.linalg.norm(feet[0] - feet[1])) / 2 + FOOT_MARGIN_M
        mos = bos_r - float(np.linalg.norm(xcom - bos_c))
        tilt_rate_signed = float(_slope(tr.tilt))
        tilt_rate = abs(tilt_rate_signed)
        if features.h_torso is not None and features.h_torso <= DOWN_H:
            tr.down_t = t

        tr.recent.append((t, mos, tilt_rate, features.v_z, features.h_torso))
        while tr.recent and t - tr.recent[0][0] > self.recent_s:
            tr.recent.popleft()
        return {
            "mos": mos,
            "mos_min_05": min(r[1] for r in tr.recent),
            "com_speed": float(np.linalg.norm(v)),
            "tilt": tilt,
            "tilt_rate": tilt_rate,
            "tilt_rate_max_05": max(r[2] for r in tr.recent),
            "h_torso": features.h_torso,
            "v_z": features.v_z,
            "v_z_min_05": min(r[3] for r in tr.recent),
            "tilt_rate_signed": tilt_rate_signed,
            "h_torso_drop_05": _drop([r[4] for r in tr.recent]),
            "knee_angle": getattr(features, "knee_angle", None),
            "hip_angle": getattr(features, "hip_angle", None),
            "leg_tilt": getattr(features, "leg_tilt", None),
            "bed_edge_distance_m": getattr(features, "bed_edge_distance_m", None),
        }

    def recently_down(self, track_id: int, t: float) -> bool:
        tr = self._tracks.get(track_id)
        return tr is not None and tr.down_t is not None and t - tr.down_t <= DOWN_HOLDOFF_S


def _drop(heights) -> float | None:
    hs = [h for h in heights if h is not None]
    return (max(hs) - hs[-1]) if hs else None


def _in_bed(features) -> bool:
    return (
        getattr(features, "supported_by_bed", None) is not None
        or getattr(features, "supported_by_surface", None) is not None
    )


class _Decider:
    """Shared debounce/refractory logic; subclasses say whether a frame is a hit."""

    def __init__(self, ground, *, frames: int = 2, refractory_s: float = 3.0):
        self.stability = StabilityFeatures(ground)
        self.frames = frames
        self.refractory_s = refractory_s
        self.last: dict[int, dict] = {}

    def retain_only(self, live_ids: set[int]) -> None:
        self.stability.retain_only(live_ids)
        self.last = {k: v for k, v in self.last.items() if k in live_ids}

    def score(self, x: dict) -> float:
        raise NotImplementedError

    threshold = 0.5

    def update(self, person, features) -> Event | None:
        x = self.stability.update(person, features)
        tid = person.track_id
        if x is None:
            return None
        tr = self.stability._tracks[tid]
        s = self.score(x)
        self.last[tid] = {**x, "score": s}
        hit = (
            s >= self.threshold
            and not _in_bed(features)
            and not self.stability.recently_down(tid, features.t)
        )
        tr.streak = tr.streak + 1 if hit else 0
        if tr.streak < self.frames:
            return None
        if tr.last_alert_t is not None and features.t - tr.last_alert_t < self.refractory_s:
            return None
        tr.last_alert_t = features.t
        return Event(
            type="FALL_IMMINENT",
            track_id=tid,
            t_trigger=features.t,
            t_alert=features.t,
            zone=features.zones[0] if features.zones else None,
            evidence={k: round(float(v), 3) for k, v in self.last[tid].items() if v is not None},
        )


class PreImpactRule(_Decider):
    """Trunk rotating faster than ``rot_deg_s`` while the XCoM is outside the base of support."""

    def __init__(self, ground, *, rot_deg_s: float = 80.0, mos_max: float = 0.0, **kw):
        super().__init__(ground, **kw)
        self.rot_deg_s, self.mos_max = rot_deg_s, mos_max

    def score(self, x: dict) -> float:
        return float(x["tilt_rate"] > self.rot_deg_s and x["mos"] < self.mos_max)


class PreImpactModel(_Decider):
    """A trained classifier over ``FEATURES`` (dict with ``model``, ``features``, ``threshold``)."""

    def __init__(self, ground, bundle: dict, **kw):
        super().__init__(ground, **kw)
        self.model = bundle["model"]
        self.features = list(bundle.get("features", FEATURES))
        self.threshold = float(bundle.get("threshold", 0.5))

    def score(self, x: dict) -> float:
        row = np.array([[np.nan if x.get(n) is None else x[n] for n in self.features]], float)
        return float(self.model.predict_proba(row)[0, 1])
