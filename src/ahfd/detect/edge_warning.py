"""Bed-edge warning: tell the nurse a patient is sitting at the bed edge, before they stand.

A bed exit alert fires once the patient is already up. Sitting at the edge with
the legs over the side comes typically 5-35 s earlier (development labels), so
a low-priority "check on them" notification at that point is the only bed
warning that gives a nurse time to walk over. It is a warning, not an alarm:
many patients sit at the edge and lie back down.

Signal ("legs over the edge"): per frame, colour-only geometry against the bed
the person is associated with --
  hip_d    hips ray-cast to the bed surface (top_m + 0.10 m), signed distance
           to the bed outline (+ inside), m
  ank_d    lowest ankle ray-cast to the floor, signed distance to the outline, m
  knee     knee angle, image degrees (180 = straight)
  tilt     trunk tilt from image vertical, degrees
plus 1 s causal medians of those four and the bed engine's support level.
A small boosted-tree classifier (``models/edge_<view>.joblib``, trained by
``~/.ahfd-work/edge_legs_train_*.py``) scores "at the edge" (labelled
SHIFTING_TO_EDGE / EDGE_SITTING / ATTEMPTING_STAND) against in-bed. A warning
needs ``dwell_s`` of continuous scores at or above the threshold, fires once,
and re-arms after ``rearm_s`` of not being at the edge.

It is only trained and measured for the side-on view; along the beds it is
much weaker (see docs/BENCHMARK.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ahfd.detect.events import Event
from ahfd.geometry.zones import signed_distance_to_polygon

L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 15, 16
HIP_ABOVE_TOP_M = 0.10
MEDIAN_S = 1.0
FEATURES = (
    "hip_d", "ank_d", "knee", "tilt",
    "hip_d_med", "ank_d_med", "knee_med", "tilt_med",
    "support",
)
_SUPPORT = {"SUPPORTED": 1.0, "PARTIAL": 0.5}
# Phases in which the person is not in a bed context (or no longer is).
_NO_BED = (None, "UNKNOWN", "OUT_OF_BED")


def raw_geometry(person, features, ground, bed, min_score: float = 0.3):
    """(hip_d, ank_d, knee, tilt) for one frame; None where a joint or the bed is missing."""
    hip_d = ank_d = None
    if bed is not None:
        sc, kp = person.scores, person.keypoints
        if min(sc[L_HIP], sc[R_HIP]) >= min_score:
            hm = (kp[L_HIP] + kp[R_HIP]) / 2
            xy = ground.pixel_to_plane(float(hm[0]), float(hm[1]), float(bed.top_m) + HIP_ABOVE_TOP_M)
            if xy is not None:
                hip_d = signed_distance_to_polygon(tuple(xy), bed.polygon)
        ds = []
        for a in (L_ANKLE, R_ANKLE):
            if sc[a] >= min_score:
                xy = ground.pixel_to_plane(float(kp[a][0]), float(kp[a][1]), 0.0)
                if xy is not None:
                    ds.append(signed_distance_to_polygon(tuple(xy), bed.polygon))
        ank_d = min(ds) if ds else None
    return hip_d, ank_d, features.knee_angle, features.torso_tilt


class EdgeVectorizer:
    """Turns per-frame raw geometry into the model's feature vector (causal 1 s medians)."""

    def __init__(self, median_s: float = MEDIAN_S):
        self.median_s = median_s
        self._hist: dict[int, deque] = {}

    def retain_only(self, live_ids: set[int]) -> None:
        for tid in list(self._hist):
            if tid not in live_ids:
                del self._hist[tid]

    def vector(self, track_id: int, t: float, raw, support: str | None) -> np.ndarray:
        h = self._hist.setdefault(track_id, deque())
        h.append((t, *raw))
        while t - h[0][0] > self.median_s:
            h.popleft()
        meds = []
        for k in range(1, 5):
            vals = [r[k] for r in h if r[k] is not None]
            meds.append(float(np.median(vals)) if vals else np.nan)
        now = [np.nan if v is None else float(v) for v in raw]
        return np.array(now + meds + [_SUPPORT.get(support or "", 0.0)], float)


@dataclass
class _Track:
    since: float | None = None
    calm_since: float | None = None
    armed: bool = True


@dataclass
class EdgeWarning:
    """Emit BED_EDGE_WARNING once per bout of sitting at the bed edge.

    ``update`` takes the pose, its Features and the bed engine's snapshot for
    the track (``DetectionEngine.bed_snapshot_of``), and the calibration's bed
    zones by name.
    """

    ground: object
    beds: dict
    bundle: dict
    dwell_s: float = 2.0
    rearm_s: float = 5.0
    threshold: float | None = None
    _vec: EdgeVectorizer = field(default_factory=EdgeVectorizer)
    _tracks: dict = field(default_factory=dict)
    last_score: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model = self.bundle["model"]
        if list(self.bundle.get("features", FEATURES)) != list(FEATURES):
            raise ValueError("edge model was trained on different features")
        if self.threshold is None:
            self.threshold = float(self.bundle.get("threshold", 0.7))

    def retain_only(self, live_ids: set[int]) -> None:
        self._vec.retain_only(live_ids)
        self._tracks = {k: v for k, v in self._tracks.items() if k in live_ids}
        self.last_score = {k: v for k, v in self.last_score.items() if k in live_ids}

    def update(self, person, features, snapshot) -> Event | None:
        tid = person.track_id
        if tid is None or features is None:
            return None
        t = features.t
        phase = getattr(snapshot, "phase", None)
        bed_name = getattr(snapshot, "bed_id", None) or features.associated_bed or features.supported_by_bed
        bed = self.beds.get(bed_name)
        raw = raw_geometry(person, features, self.ground, bed)
        x = self._vec.vector(tid, t, raw, getattr(snapshot, "support", None))
        in_bed = phase not in _NO_BED and bed is not None
        score = float(self.model.predict_proba(x.reshape(1, -1))[0, 1]) if in_bed else None
        if score is not None:
            self.last_score[tid] = score
        fired = self.decide(tid, t, score)
        if fired is None:
            return None
        return Event(
            type="BED_EDGE_WARNING",
            track_id=tid,
            t_trigger=fired,
            t_alert=t,
            zone=bed_name,
            evidence={
                "score": round(score, 3),
                "at_edge_s": round(t - fired, 1),
                **{k: round(float(v), 2) for k, v in zip(FEATURES, x) if np.isfinite(v)},
            },
        )

    def decide(self, track_id: int, t: float, score: float | None) -> float | None:
        """Debounce one frame's score (None = not in a bed context). Returns the bout start when it fires."""
        tr = self._tracks.setdefault(track_id, _Track())
        if score is None or score < self.threshold:
            tr.since = None
            tr.calm_since = t if tr.calm_since is None else tr.calm_since
            self._rearm(tr, t)
            return None
        tr.calm_since = None
        tr.since = t if tr.since is None else tr.since
        if not tr.armed or t - tr.since < self.dwell_s:
            return None
        tr.armed = False
        return tr.since

    def _rearm(self, tr: _Track, t: float) -> None:
        if not tr.armed and tr.calm_since is not None and t - tr.calm_since >= self.rearm_s:
            tr.armed = True
