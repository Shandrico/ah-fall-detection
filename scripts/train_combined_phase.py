"""Train one 7-phase bed-exit model on every usable labelled clip (showcase model).

    python scripts/train_combined_phase.py --out models/phase_combined.joblib

Used:     pilot takes clip1_01-10 (15 Oct, both cameras, 30 fps tracks) and the
          29 Oct takes h*_02/03 and f* (both cameras, 15 fps tracks).
Excluded: 27 Oct takes (h*_01, s*_0x): tracks at 6-11 fps, too sparse for the
          over-time features; neg_f6/neg_f7: no phase labels; 4 Nov integration
          takes: no extracted tracks in data/.

Rows are built exactly as the live dashboard builds them (same feature
extractor, bed engine and causal temporal summaries), feeding frames at about
the Jetson's live rate (--hz, default 6) so the over-time features match.
Scored leave-one-group-out (each person/session held out in turn). The saved
model is fitted on everything and wrapped in ConservativePhaseClassifier, whose
threshold is chosen from the held-out predictions to keep false "getting up"
calls rare. Display only: the dashboard shows it, alerts never read it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ahfd.annotate import PILOT_PHASE_SCRIPT  # noqa: E402
from ahfd.capture.base import SourceMeta  # noqa: E402
from ahfd.cli import _build_detection, _depth_provenance  # noqa: E402
from ahfd.config import load_config  # noqa: E402
from ahfd.io import read_tracks  # noqa: E402
from ahfd.ml import posture as P  # noqa: E402
from ahfd.ml.live_phase import CALM_PHASES, ConservativePhaseClassifier  # noqa: E402
from ahfd.ml.phase_compare import PHASES  # noqa: E402
from ahfd.ml.temporal import MODEL_FEATURE_NAMES, CausalTemporalSummarizer  # noqa: E402

OCT29 = ["h1_02", "h1_03", "h2_02", "h2_03", "h3_02", "h3_03",
         "f1_01", "f1_02", "f3_01", "f3_02", "f4_01", "f4_02", "f5_01", "f5_02"]
FEATURE_NAMES = list(P.FEATURES) + list(MODEL_FEATURE_NAMES)


def clip_list():
    out = []
    for i, entry in sorted(PILOT_PHASE_SCRIPT.items()):
        stem = "clip1_%02d" % i
        for cam in ("d435i", "d435f"):
            out.append(dict(
                stem=stem, cam=cam, tracks=ROOT / f"data/tracks_{cam}/{stem}.jsonl",
                labels=ROOT / f"data/phases_{cam}/{stem}.json", calib=ROOT / f"calib/{cam}.yaml",
                clock=ROOT / f"data/tracks_d435i/{stem}.jsonl", bed=entry["bed"],
                group="pilot_block_" + entry["block"],
            ))
    for stem in OCT29:
        for cam in ("d435i", "d435f"):
            calib = ROOT / (f"calib/{cam}_20261029_f1_01.yaml" if stem == "f1_01" else f"calib/{cam}_20261029.yaml")
            out.append(dict(
                stem=stem, cam=cam, tracks=ROOT / f"data/tracks_{cam}_v3/{stem}.jsonl",
                labels=ROOT / f"data/phases_d435i_v2/{stem}.json", calib=calib,
                clock=ROOT / f"data/tracks_d435i_v3/{stem}.jsonl", bed=None,
                group="oct29_" + stem.split("_")[0],
            ))
    return out


def pick_target(c, calib, min_score):
    """The clip's patient: the person on the scripted bed, else on any bed, else the most-seen."""
    if c["bed"]:
        zone = P._resolve_bed_zone(calib, c["bed"])
        if zone is not None:
            tid = P._dominant_bed_track(c["tracks"], calib.ground, zone, min_score)
            if tid is not None:
                return tid
    best, best_n = None, 0
    for zone in [z for z in calib.zones.zones if z.kind == "bed"]:
        counts = Counter()
        for pose in read_tracks(c["tracks"]):
            for p in pose.people:
                if p.track_id is not None and P._person_bed_zone(p, calib.ground, (zone,), min_score) is not None:
                    counts[p.track_id] += 1
        if counts and counts.most_common(1)[0][1] > best_n:
            best, best_n = counts.most_common(1)[0]
    if best is not None:
        return best
    seen = Counter(p.track_id for pose in read_tracks(c["tracks"]) for p in pose.people if p.track_id is not None)
    return seen.most_common(1)[0][0] if seen else None


def rows_for(c, cfg, hz):
    meta = SourceMeta(uri="tracks", width=1920, height=1080, fps=30.0)
    extractor, machine, _sink, calib = _build_detection(cfg, str(c["calib"]), meta)
    temporal = CausalTemporalSummarizer(
        max_gap_s=cfg.bed_activity.cusum.max_gap_s, near_edge_m=cfg.bed_activity.near_edge_enter_m,
    )
    segs = P._labelled_segments(c["labels"])
    target = pick_target(c, calib, cfg.pose.min_keypoint_score)
    t0 = P._first_track_time(c["clock"]) if c["clock"].exists() else None
    if target is None or t0 is None:
        return [], [], "no target person" if target is None else "no clock"
    X, y, last = [], [], None
    for pose in read_tracks(c["tracks"]):
        if last is not None and pose.t - last < 1.0 / hz - 1e-6:
            continue                     # emulate the live frame rate
        last = pose.t
        ids = {p.track_id for p in pose.people if p.track_id is not None}
        extractor.retain_only(ids)
        machine.retain_only(ids)
        hit = None
        for person in pose.people:
            if person.track_id is None:
                continue
            f = extractor.extract(person, pose.t)
            if f is None:
                continue
            machine.update_all(f)
            if person.track_id == target:
                hit = (person, f)
        if hit is None:
            if target in temporal.live_ids:
                temporal.mark_unobserved(target, pose.t)
            continue
        person, f = hit
        snap = machine.bed_snapshot_of(target, pose.t)
        summary = temporal.update(f, valid=f.has_geometry(), context={
            "edge_velocity_mps": snap.edge_velocity_mps if snap else None,
            "cusum_z": snap.cusum_z if snap else None,
            "cusum_g": snap.cusum_g if snap else None,
            "cusum_onset": 1.0 if snap and "cusum_onset" in snap.reasons else 0.0,
            "cusum_armed": 1.0 if snap and snap.cusum_armed else 0.0,
            **_depth_provenance(person, min_score=cfg.pose.min_keypoint_score),
            "imu_pitch_delta_deg": None,
            "imu_roll_delta_deg": None,
            "episode_reclined": 1.0 if snap and snap.phase == "RECLINED" else 0.0,
        })
        phase = P._posture_at(segs, pose.t - t0, "phase")
        if phase is None or not f.has_geometry():
            continue
        row = {**P.features_row(f, person, extractor.ground, cfg.pose.min_keypoint_score), **summary.values}
        X.append([np.nan if row.get(n) is None else float(row[n]) for n in FEATURE_NAMES])
        y.append(phase)
    return X, y, None


def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000)),
        ]),
        "forest": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4,
                                             class_weight="balanced_subsample", n_jobs=-1, random_state=0)),
        ]),
    }


def oof_proba(make, X, y, groups, classes):
    proba = np.zeros((len(y), len(classes)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in sorted(set(groups)):
            test = groups == g
            m = make().fit(X[~test], y[~test])
            p = m.predict_proba(X[test])
            cols = [classes.index(str(c)) for c in m.classes_]
            proba[np.ix_(test, cols)] = p
    return proba


def decide(proba, classes, threshold):
    calm = [i for i, c in enumerate(classes) if c in CALM_PHASES]
    out = []
    for p in proba:
        k = int(np.argmax(p))
        if classes[k] not in CALM_PHASES and p[k] < threshold:
            k = max(calm, key=lambda i: p[i])
        out.append(classes[k])
    return np.array(out)


def vote(pred, clip_ids, n):
    """The dashboard's majority vote over the last n frames, per clip (ties -> most recent)."""
    from collections import deque

    out, recent, last_clip = [], deque(maxlen=n), None
    for p, c in zip(pred, clip_ids):
        if c != last_clip:
            recent.clear()
            last_clip = c
        recent.append(p)
        counts = Counter(recent)
        top = max(counts.values())
        out.append(next(lbl for lbl in reversed(recent) if counts[lbl] == top))
    return np.array(out)


def score(y, pred):
    from sklearn.metrics import accuracy_score, f1_score

    present = [p for p in PHASES if (y == p).any()]
    calm_truth = np.isin(y, CALM_PHASES)
    moving_pred = ~np.isin(pred, CALM_PHASES)
    moving_truth = ~calm_truth
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
        # How often a calm moment is shown as "getting up" -- the false positive we care about.
        "false_getting_up_rate": float((calm_truth & moving_pred).sum() / max(1, calm_truth.sum())),
        # How often a real getting-up moment is shown as some getting-up phase.
        "getting_up_recall": float((moving_truth & moving_pred).sum() / max(1, moving_truth.sum())),
        "per_phase_f1": {p: round(float(f1_score(y == p, pred == p)), 3) for p in present},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/dashboard_jetson.yaml"))
    ap.add_argument("--hz", type=float, default=6.0, help="live frame rate to emulate")
    ap.add_argument("--max-false-rate", type=float, default=0.02,
                    help="highest acceptable share of calm frames shown as getting up")
    ap.add_argument("--out", default=str(ROOT / "models/phase_combined.joblib"))
    ap.add_argument("--vote", type=int, default=9, help="majority-vote window used live (frames)")
    ap.add_argument("--cache", action="store_true", help="reuse the rows built on the last run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.alert.console = False
    cfg.alert.jsonl_path = None

    X, y, groups, used, clip_ids = [], [], [], [], []
    t_start = time.time()
    cache = Path(args.out).with_suffix(".rows.npz")
    if args.cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, y, groups, clip_ids, used = list(z["X"]), list(z["y"]), list(z["groups"]), list(z["clip_ids"]), list(z["used"])
        print("rows from cache " + str(cache))
    for c in ([] if args.cache and cache.exists() else clip_list()):
        if not c["tracks"].exists() or not c["labels"].exists():
            print("  skip %-8s %-6s missing %s" % (c["stem"], c["cam"], "tracks" if not c["tracks"].exists() else "labels"))
            continue
        xs, ys, why = rows_for(c, cfg, args.hz)
        if why:
            print("  skip %-8s %-6s %s" % (c["stem"], c["cam"], why))
            continue
        X += xs; y += ys; groups += [c["group"]] * len(ys); clip_ids += [len(used)] * len(ys)
        used.append({"clip": c["stem"], "camera": c["cam"], "group": c["group"], "rows": len(ys)})
        print("  used %-8s %-6s %4d rows  %s" % (c["stem"], c["cam"], len(ys), c["group"]))
    X, y, groups, clip_ids = np.array(X, float), np.array(y), np.array(groups), np.array(clip_ids)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, groups=groups, clip_ids=clip_ids, used=np.array(used, dtype=object))
    print("\n%d rows from %d clips, %d groups (%.0f s)" % (len(y), len(used), len(set(groups)), time.time() - t_start))
    print("rows per phase: " + ", ".join("%s=%d" % (p, (y == p).sum()) for p in PHASES))

    classes = sorted(set(y.tolist()))
    report = {"rows": int(len(y)), "clips": used, "groups": sorted(set(groups)), "emulated_hz": args.hz,
              "excluded": ["27 Oct h*_01, s*_0x: 6-11 fps tracks", "neg_f6/neg_f7: no phase labels",
                           "4 Nov integration takes: no tracks in data/"], "models": {}}
    best = None
    for name, make in make_models().items():
        proba = oof_proba(make, X, y, groups, classes)
        print("\n%s (leave one person/session out):" % name)
        print("  %-9s %6s %6s %10s %10s   | with the live %d-frame vote: %6s %10s %10s"
              % ("threshold", "acc", "mF1", "false-up", "up-recall", args.vote, "mF1", "false-up", "up-recall"))
        rows = {}
        for th in (0.0, 0.3, 0.4, 0.5, 0.6):
            raw = decide(proba, classes, th)
            s = score(y, raw)
            v = score(y, vote(raw, clip_ids, args.vote))
            s["voted"] = v
            rows[th] = s
            print("  %-9.1f %6.3f %6.3f %9.1f%% %9.1f%%   |                        %6.3f %9.1f%% %9.1f%%" % (
                th, s["accuracy"], s["macro_f1"], 100 * s["false_getting_up_rate"], 100 * s["getting_up_recall"],
                v["macro_f1"], 100 * v["false_getting_up_rate"], 100 * v["getting_up_recall"]))
        ok = [th for th, s in rows.items() if s["voted"]["false_getting_up_rate"] <= args.max_false_rate]
        meets = bool(ok)
        # Within the false-positive limit, the best macro-F1; a model that cannot
        # meet the limit only competes (on fewest false calls) if none can.
        th = (max(ok, key=lambda t: rows[t]["voted"]["macro_f1"]) if ok
              else min(rows, key=lambda t: rows[t]["voted"]["false_getting_up_rate"]))
        report["models"][name] = {"by_threshold": {str(k): v for k, v in rows.items()},
                                  "chosen_threshold": th, "meets_false_limit": meets}
        cand = (meets, rows[th]["voted"]["macro_f1"] if meets else -rows[th]["voted"]["false_getting_up_rate"], name, th)
        if best is None or cand > best:
            best = cand
    _, _, name, th = best
    print("\nchosen: %s, threshold %.1f -> %s" % (name, th, json.dumps(report["models"][name]["by_threshold"][str(th)])))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_models()[name]().fit(X, y)
    import joblib

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": ConservativePhaseClassifier(model, th), "features": FEATURE_NAMES,
                 "kind": name, "threshold": th, "trained_on": used, "emulated_hz": args.hz}, out)
    report["chosen"] = {"model": name, "threshold": th, "path": str(out)}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("saved " + str(out) + " (+ .json report)")


if __name__ == "__main__":
    main()
