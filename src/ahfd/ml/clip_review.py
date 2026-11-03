"""Per-frame rule and model predictions for one clip, next to its labels.

Keypoints only (a stored ``tracks.jsonl``), so it is privacy-safe and fast. It
reproduces the evaluation path of ``phase_compare.build_rows`` frame by frame:
the live rule engines (4-state posture from the fall machine, 7-phase bed
engine) and the trained models (posture tree, phase logistic on per-frame +
over-time features). ``debug.annotated_review`` draws the result on the video;
the same rows score a new test set.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Fall-machine states as 4-state posture labels (FALLING is on its way down).
RULE_POSTURE = {
    "UPRIGHT": "upright",
    "SITTING": "sitting",
    "IN_BED": "in_bed",
    "ON_GROUND": "on_ground",
    "FALLING": "on_ground",
}


# Bed phases that mean the patient is on the way out: the early, risk-level warning.
GETTING_UP_PHASES = frozenset({"TORSO_RISING", "SHIFTING_TO_EDGE", "EDGE_SITTING", "ATTEMPTING_STAND"})
FALL_ALERT_EVENTS = frozenset({"FALL_SUSPECTED", "PERSON_DOWN", "FALL_CONFIRMED", "NEAR_MISS"})
FALL_ALERT_SHOW_S = 5.0  # how long the video shows an alert after it fires
BED_EXIT_SHOW_S = 5.0  # how long the video shows a bed-exit alert


@dataclass
class FramePrediction:
    """One frame of the clip's subject: what each source says."""

    t: float  # seconds since the clip's first frame
    track_id: int | None
    rule_posture: str | None = None
    rule_phase: str | None = None
    model_posture: str | None = None
    model_phase: str | None = None
    label_posture: str | None = None
    label_phase: str | None = None
    # The most urgent thing the system is showing: FALL ALERT / BED EXIT / POSSIBLE FALL / GETTING UP.
    banner: str | None = None
    alert_type: str | None = None


# Alert events as the system raises them, grouped for the review video.
FALL_EVENTS = frozenset(FALL_ALERT_EVENTS | {"POSSIBLE_FALL"})
REVIEWED_EVENTS = FALL_EVENTS | {"BED_EXIT", "BED_EXIT_WARNING"}


@dataclass
class AlertMark:
    """One alert group (fall alerts <10 s apart are one group) or one missed labelled event, with a verdict.

    ``verdict``: TRUE, FALSE ALARM, RISKY (a reviewer judged the movement worth a look), EXTRA (a
    repeat near a real event), NO EXIT (an early warning not followed by an exit within 60 s), or
    MISSED (a labelled fall/exit with no alert). The windows are those of docs/BENCHMARK.md.
    """

    kind: str  # "fall", "bed_exit", "warning" or "missed"
    t: float  # seconds since the clip's first frame
    types: list[str]
    verdict: str
    note: str = ""


@dataclass
class ClipPredictions:
    t0: float  # absolute time of the clip's first frame
    frames: dict[float, FramePrediction] = field(default_factory=dict)  # keyed by rounded absolute t
    subject: int | None = None
    events: list[tuple[str, float, int]] = field(default_factory=list)  # (type, s since first frame, track)
    impacts: list[float] = field(default_factory=list)  # labelled fall impacts, s since first frame
    exits: list[float] = field(default_factory=list)  # labelled sustained exits (>= 3 s out of bed)
    marks: list[AlertMark] = field(default_factory=list)

    def at(self, t_abs: float) -> FramePrediction | None:
        return self.frames.get(round(t_abs, 3))

    def agreement(self) -> dict[str, tuple[int, int]]:
        """(agreeing, labelled) frames for each prediction against the labels."""
        out = {}
        for name, label in (
            ("rule_posture", "label_posture"), ("model_posture", "label_posture"),
            ("rule_phase", "label_phase"), ("model_phase", "label_phase"),
        ):
            pairs = [(getattr(f, name), getattr(f, label)) for f in self.frames.values()]
            labelled = [(p, y) for p, y in pairs if y is not None]
            out[name] = (sum(p == y for p, y in labelled), len(labelled))
        return out


def judge_alerts(events, impacts, exits, reviewed=None) -> list[AlertMark]:
    """Verdict for every alert group and every missed labelled event (benchmark windows).

    Fall alerts: groups of alerts less than 10 s apart; TRUE inside [impact - 3 s, impact + 30 s].
    A false group a reviewer judged risky (``reviewed``: {seconds: verdict}, matched to the episode:
    from 10 s before the group's first alert to 3 s after its last) shows as RISKY. Bed exit: TRUE inside [exit - 5, exit + 15]; inside [exit - 10, exit + 30]
    otherwise EXTRA; else FALSE ALARM. Early warning: TRUE when an exit follows within 60 s.
    """
    reviewed = reviewed or {}
    marks = []
    falls = sorted((t, k) for k, t, _ in events if k in FALL_EVENTS)
    groups: list[list[tuple[float, str]]] = []
    for t, k in falls:
        if groups and t - groups[-1][-1][0] <= 10.0:
            groups[-1].append((t, k))
        else:
            groups.append([(t, k)])
    for g in groups:
        t, types = g[0][0], [k for _, k in g]
        hit = [ti for ti in impacts if ti - 3.0 <= t <= ti + 30.0]
        if hit:
            marks.append(AlertMark("fall", t, types, "TRUE", "%+.1f s from the labelled impact" % (t - hit[0])))
            continue
        # The reviewer judged an alert *episode*; it may have started up to 10 s before this group
        # (alerts less than 10 s apart are one episode), e.g. when a later version drops its first alert.
        end = g[-1][0]
        verdict = next((v for rt, v in reviewed.items() if t - 10.0 <= rt <= end + 3.0), None)
        if verdict == "risky":
            marks.append(AlertMark("fall", t, types, "RISKY", "reviewed: risky movement, useful"))
        elif verdict == "real":
            marks.append(AlertMark("fall", t, types, "TRUE", "reviewed: a real fall the labels missed"))
        else:
            marks.append(AlertMark("fall", t, types, "FALSE ALARM", "no labelled fall within -3 / +30 s"))
    for t in sorted(t for k, t, _ in events if k == "BED_EXIT"):
        hit = [e for e in exits if e - 5.0 <= t <= e + 15.0]
        if hit:
            marks.append(AlertMark("bed_exit", t, ["BED_EXIT"], "TRUE", "%+.1f s from the labelled exit" % (t - hit[0])))
        elif any(e - 10.0 <= t <= e + 30.0 for e in exits):
            marks.append(AlertMark("bed_exit", t, ["BED_EXIT"], "EXTRA", "near a labelled exit"))
        else:
            marks.append(AlertMark("bed_exit", t, ["BED_EXIT"], "FALSE ALARM", "no labelled exit near"))
    for t in sorted(t for k, t, _ in events if k == "BED_EXIT_WARNING"):
        nxt = [e for e in exits if t <= e <= t + 60.0]
        if nxt:
            marks.append(AlertMark("warning", t, ["BED_EXIT_WARNING"], "TRUE", "%.1f s before the labelled exit" % (nxt[0] - t)))
        else:
            marks.append(AlertMark("warning", t, ["BED_EXIT_WARNING"], "NO EXIT", "no exit within 60 s"))
    for ti in impacts:
        if not any(ti - 2.5 <= t <= ti + 30.0 for t, _ in falls):
            marks.append(AlertMark("missed", ti, ["FALL"], "MISSED", "labelled fall, no fall alert"))
    for e in exits:
        if not any(e - 5.0 <= t <= e + 15.0 for k, t, _ in events if k == "BED_EXIT"):
            marks.append(AlertMark("missed", e, ["BED_EXIT"], "MISSED", "labelled exit, no bed-exit alert"))
    return sorted(marks, key=lambda m: m.t)


def _load_model(path):
    import joblib

    if path is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = joblib.load(path)
    if not isinstance(loaded, dict):
        from ahfd.ml.posture import FEATURES

        loaded = {"model": loaded, "features": list(FEATURES)}
    return loaded


def _row(values: dict, names) -> list[float]:
    return [math.nan if values.get(n) is None else float(values[n]) for n in names]


def subject_track(tracks: Path, calib, min_score: float = 0.3) -> int | None:
    """The track that occupies any calibrated bed for the most frames (the patient)."""
    from ahfd.ml import posture as P

    best, best_n = None, 0
    for zone in calib.zones.zones:
        if zone.kind != "bed":
            continue
        tid = P._dominant_bed_track(tracks, calib.ground, zone, min_score)
        if tid is None:
            continue
        n = sum(
            1 for pose in _read(tracks) for p in pose.people if p.track_id == tid
        )
        if n > best_n:
            best, best_n = tid, n
    return best


def _read(tracks):
    from ahfd.io import read_tracks

    return read_tracks(tracks)


def predict_clip(
    tracks: Path,
    calibration: Path,
    cfg,
    *,
    posture_model: Path | None = None,
    phase_model: Path | None = None,
    labels: Path | None = None,
    label_t0: float | None = None,
    posture_labels: Path | None = None,
    falls: Path | None = None,
    reviewed: dict[float, str] | None = None,
) -> ClipPredictions:
    """Run the rules and models over ``tracks``; attach ``labels`` if given.

    ``label_t0`` is the absolute time that label second 0 refers to; it
    defaults to this clip's first frame. Labels made on the other camera of a
    take need that camera's first-frame time (the pilot put the D435f labels on
    the D435i clock this way).

    ``posture_labels`` (4-state, on this clip's own clock) override postures
    derived from the phase labels. ``falls`` is a fall annotation JSON
    (``t_impact`` on this clip's own clock). With either, every alert gets a
    verdict in ``marks`` (see ``judge_alerts``); ``reviewed`` adds a
    reviewer's verdicts for false fall alerts.
    """
    import numpy as np

    from ahfd.capture.base import SourceMeta
    from ahfd.cli import _build_detection, _depth_provenance
    from ahfd.ml import posture as P
    from ahfd.ml.temporal import CausalTemporalSummarizer

    tracks, calibration = Path(tracks), Path(calibration)
    meta = SourceMeta(uri="tracks", width=1920, height=1080, fps=30.0)
    extractor, machine, _sink, calib = _build_detection(cfg, str(calibration), meta, emit_bed_early_warning=True)
    temporal = CausalTemporalSummarizer(
        max_gap_s=cfg.bed_activity.cusum.max_gap_s, near_edge_m=cfg.bed_activity.near_edge_enter_m,
    )
    posture = _load_model(posture_model)
    phase = _load_model(phase_model)
    segments = P._labelled_segments(Path(labels)) if labels is not None else []
    posture_segments = P._labelled_segments(Path(posture_labels)) if posture_labels is not None else []
    min_kp = cfg.pose.min_keypoint_score

    subject = subject_track(tracks, calib, 0.3)
    out = ClipPredictions(t0=math.nan, subject=subject)
    last_seen: dict[int, float] = {}
    last_alert: tuple[str, float] | None = None
    last_exit: float | None = None
    for pose in _read(tracks):
        if math.isnan(out.t0):
            out.t0 = pose.t
        lt0 = out.t0 if label_t0 is None else label_t0
        for p in pose.people:
            if p.track_id is not None:
                last_seen[p.track_id] = pose.t
        # Identities live ~1 s after their last detection, as in the live tracker.
        live = {tid for tid, seen in last_seen.items() if pose.t - seen <= 1.0}
        extractor.retain_only(live)
        machine.retain_only(live)

        candidates = []
        for person in pose.people:
            if person.track_id is None:
                continue
            f = extractor.extract(person, pose.t)
            if f is None:
                continue
            for event in machine.update_all(f):
                if event.type in REVIEWED_EVENTS:
                    out.events.append((event.type, event.t_alert - out.t0, event.track_id))
                if event.type in FALL_ALERT_EVENTS:
                    last_alert = (event.type, pose.t)
                elif event.type == "BED_EXIT":
                    last_exit = pose.t
            snap = machine.bed_snapshot_of(person.track_id, pose.t)
            summary = temporal.update(f, valid=f.has_geometry(), context={
                "edge_velocity_mps": snap.edge_velocity_mps if snap else None,
                "cusum_z": snap.cusum_z if snap else None,
                "cusum_g": snap.cusum_g if snap else None,
                "cusum_onset": 1.0 if snap and "cusum_onset" in snap.reasons else 0.0,
                "cusum_armed": 1.0 if snap and snap.cusum_armed else 0.0,
                **_depth_provenance(person, min_score=min_kp),
                "imu_pitch_delta_deg": None,
                "imu_roll_delta_deg": None,
                "episode_reclined": 1.0 if snap and snap.phase == "RECLINED" else 0.0,
            })
            candidates.append((person, f, snap, summary))

        if not candidates:
            continue
        # The bed's occupant when in view, else the most confident person.
        person, f, snap, summary = max(
            candidates, key=lambda c: (c[0].track_id == subject, c[0].score)
        )
        pred = FramePrediction(t=pose.t - out.t0, track_id=person.track_id)
        pred.rule_posture = RULE_POSTURE.get(machine.state_of(person.track_id))
        if snap is not None and snap.phase not in (None, "UNKNOWN"):
            pred.rule_phase = snap.phase
        if last_alert is not None and pose.t - last_alert[1] <= FALL_ALERT_SHOW_S:
            pred.banner, pred.alert_type = "FALL ALERT", last_alert[0]
        elif last_exit is not None and pose.t - last_exit <= BED_EXIT_SHOW_S:
            pred.banner = "BED EXIT"
        elif machine.possible_fall_of(person.track_id, pose.t):
            pred.banner = "POSSIBLE FALL"
        elif pred.rule_phase in GETTING_UP_PHASES and pred.rule_posture != "on_ground":
            pred.banner = "GETTING UP"
        if f.has_geometry():
            row = P.features_row(f, person, extractor.ground, min_kp)
            if posture is not None:
                x = np.array([_row(row, posture["features"])])
                pred.model_posture = str(posture["model"].predict(x)[0])
            if phase is not None:
                x = np.array([_row({**row, **summary.values}, phase["features"])])
                pred.model_phase = str(phase["model"].predict(x)[0])
        if segments:
            pred.label_phase = P._posture_at(segments, pose.t - lt0, "phase")
            pred.label_posture = P._posture_at(segments, pose.t - lt0, "posture") or (
                P.PHASE_TO_POSTURE.get(pred.label_phase) if pred.label_phase else None
            )
        if posture_segments:
            pred.label_posture = P._posture_at(posture_segments, pose.t - out.t0) or pred.label_posture
        out.frames[round(pose.t, 3)] = pred
    if labels is not None or falls is not None:
        import json

        lt0 = out.t0 if label_t0 is None else label_t0
        if labels is not None:
            segs = sorted(json.loads(Path(labels).read_text())["segments"], key=lambda x: x["start_s"])
            in_bed = {"RECLINED", "TORSO_RISING", "UPRIGHT_IN_BED", "SHIFTING_TO_EDGE", "EDGE_SITTING", "ATTEMPTING_STAND"}
            out.exits = [
                b["start_s"] + lt0 - out.t0 for a, b in zip(segs, segs[1:])
                if b.get("phase") == "OUT_OF_BED" and a.get("phase") in in_bed and b["end_s"] - b["start_s"] >= 3.0
            ]
        if falls is not None:
            out.impacts = [float(x["t_impact"]) for x in json.loads(Path(falls).read_text()).get("falls", [])]
        out.marks = judge_alerts(out.events, out.impacts, out.exits, reviewed)
    return out
