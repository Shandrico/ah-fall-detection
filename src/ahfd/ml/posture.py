"""Train an interpretable posture classifier from labelled clips.

This is the "optional re-ranker" path: a small scikit-learn model that learns
posture (upright / sitting / in_bed / on_ground) from the **same metric
features** the rule-based classifier uses -- torso height, floor-spread, etc.,
all range-invariant thanks to calibration. It is an add-on for study and
comparison, not a replacement for the rule-based state machine.

The design goal is that **retraining is trivial**: this reads labels + tracks +
calibration and produces a model. To retrain on new data you label more clips,
extract them, and re-run -- the code does not change, only the data grows.

Two stages, kept separate so each is testable on its own:

* ``build_dataset`` turns labelled posture segments + extracted tracks into a
  table of (metric features -> posture). Transitions (gaps between segments)
  are excluded, which is what makes the labels clean.
* ``train`` fits a small decision tree and reports accuracy, a confusion matrix,
  and per-feature importances -- the last of which answers "which joints/features
  actually separate the postures".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ahfd.geometry.depth_height import DEPTH_FEATURES, depth_features
from ahfd.ml.pose_features import JOINT_FEATURES, joint_features

_EPS = 1e-6

# Aggregate metric features pulled straight off `Features`. All in metres and
# range-invariant, so the same value means the same thing at any camera
# position. `range_m` (lens-to-person distance) is deliberately EXCLUDED: it is
# a *location* feature, not a posture one, and on single-session data a tree
# will cheat by learning "the bed is far, the chair is near" -- pure leakage
# that does not generalise. Posture must be decided by shape/height, never by
# where someone stood, exactly as the rule-based classifier does.
AGG_FEATURES = [
    "h_torso",  # shoulders+hips centroid height
    "h_max",  # highest joint
    "h_min",  # lowest joint
    "floor_spread",  # large upright, ~body length when down
    "vertical_extent",  # h_max - h_min, metres
    "compactness",  # vertical_extent / floor_spread -- tall-and-thin vs flat
    "bed_overlap",  # fraction of body over a bed footprint: on_ground vs in_bed
]
# Aggregate features read directly as attributes of `Features`.
_AGG_FROM_FEATS = ["h_torso", "h_max", "h_min", "floor_spread", "bed_overlap"]

# The full input set: the aggregates above plus the per-joint heights, joint
# angles, and shape ratios from `pose_features`. This is the "use the whole
# skeleton, not just torso" change -- the trainer reports which of these
# actually separate the postures (see `train`).
#
# DEPTH_FEATURES are appended last: joint heights measured directly from depth
# (dh_*). They are None for any clip recorded without depth, so the trainer
# imputes them and RGB-only data behaves exactly as before -- but on a
# depth-recorded clip they add the measured-height signal that fixes the nadir
# case. This is what makes the model depth-ready without a second pipeline.
FEATURES = AGG_FEATURES + JOINT_FEATURES + DEPTH_FEATURES

# A row is kept only if at least these are present -- the rest are imputed.
REQUIRED = ["h_torso", "floor_spread"]

# Coarser label schemes for evaluating at the granularity a task actually needs.
# "bed-state" is the bed-exit target: it folds the two postures the shallow mount
# cannot separate (upright/on_ground) into a single OUT state, keeping the
# distinctions the ward cares about (lying vs sitting-up vs out of bed). Scoring
# here answers "how well does the system do at the job", separate from the
# harder-but-less-relevant 4-class posture number.
COLLAPSE_SCHEMES = {
    "bed-state": {
        "in_bed": "IN_BED",
        "sitting": "SITTING_UP",
        "upright": "OUT",
        "on_ground": "OUT",
    },
}


# Bed-exit activity phases -> the 4-class posture they unambiguously show, so
# the pilot's phase labels also train the posture model. The short transition
# phases (TORSO_RISING, SHIFTING_TO_EDGE, ATTEMPTING_STAND) are mid-movement,
# not a held posture, and are left out exactly like gaps between segments.
PHASE_TO_POSTURE = {
    "RECLINED": "in_bed",
    "UPRIGHT_IN_BED": "sitting",
    "EDGE_SITTING": "sitting",
    "OUT_OF_BED": "upright",
}

# Pilot cameras, in time-reference order: the phase labels were marked on the
# D435i recordings, so each take's D435i first frame is t=0 for both cameras.
PILOT_CAMERAS = ("d435i", "d435f")


def collapse_labels(labels, scheme: str):
    """Map fine posture labels to a coarser scheme (e.g. 'bed-state').

    Unknown labels pass through unchanged, so a partially-labelled set never
    raises -- it just isn't collapsed for those rows.
    """
    if scheme not in COLLAPSE_SCHEMES:
        raise ValueError(
            "unknown collapse scheme " + repr(scheme)
            + "; known: " + ", ".join(sorted(COLLAPSE_SCHEMES))
        )
    mapping = COLLAPSE_SCHEMES[scheme]
    return [mapping.get(lbl, lbl) for lbl in labels]


def _finite_or_none(v):
    """Coerce inf/nan/None to None so every stored feature is a real number or absent.

    `floor_spread` in particular is +inf when a ray misses the floor; that is a
    genuine "no measurement", not a huge value, so it must not reach the model
    as one. Everything non-finite is imputed downstream instead.
    """
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def features_row(feats, person, ground, min_keypoint_score: float = 0.3) -> dict:
    """The FEATURES-keyed dict for one person-frame.

    Shared by training (`build_dataset`) and export (`ahfd export-features`) so
    the numbers a model trains on are exactly the numbers you can inspect. All
    values are finite floats or None (non-finite coerced away).
    """
    row = {k: getattr(feats, k) for k in _AGG_FROM_FEATS}
    ve = feats.height_spread
    row["vertical_extent"] = ve
    row["compactness"] = (
        ve / (feats.floor_spread + _EPS)
        if ve is not None and math.isfinite(feats.floor_spread)
        else None
    )
    row.update(joint_features(person, ground, feats.contact_xy, min_keypoint_score))
    # Depth-measured joint heights, when the clip was recorded with depth
    # (person.heights). All None otherwise -> imputed, no effect on RGB clips.
    row.update(depth_features(getattr(person, "heights", None)))
    return {k: _finite_or_none(v) for k, v in row.items()}


def _posture_at(segments: list[dict], t: float, label_key: str = "posture") -> str | None:
    """The labelled class at time ``t``, or None if ``t`` is in a gap.

    Gaps between segments are transitions -- deliberately unlabelled, so they
    are excluded from training rather than forced into a class. ``label_key`` is
    the per-segment field: ``posture`` for the 4-class scheme, ``phase`` for the
    bed-exit activity phases.
    """
    for seg in segments:
        val = seg.get(label_key)
        if val is not None and seg["start_s"] <= t <= seg["end_s"]:
            return str(val)
    return None


def _labelled_segments(path: Path) -> list[dict]:
    """Read a posture-label file's real segments (placeholders skipped)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        s
        for s in data.get("segments", [])
        if float(s.get("end_s", 0)) > float(s.get("start_s", 0))
    ]


def _main_person(pose):
    """The person to score in a frame: the highest-confidence tracked one.

    For a single-subject clip this is just "the person in view"; taking the most
    confident track guards against a spurious second detection. When a clip has
    bystanders (other beds, staff walking past), this is WRONG -- it can pick a
    passer-by and flip between people frame to frame. Use ``_select_target_person``
    with the clip's target bed in that case.
    """
    people = [p for p in pose.people if p.track_id is not None]
    if not people:
        return None
    return max(people, key=lambda p: p.score)


# COCO-17 lower-body joints used to associate a person with a bed: the contact-
# ish joints (ankles, then hips) back-projected to a bed's surface height.
_ANKLES = (15, 16)
_HIPS = (11, 12)


def _person_bed_zone(person, ground, zones, min_score):
    """The bed/chair zone this person's lower body projects into, or None.

    Mirrors ``FeatureExtractor``'s association: each ankle/hip pixel is
    back-projected to a candidate bed's *surface* height and tested against that
    bed's polygon (a floor projection would miss someone lying ~0.5 m up). A
    bystander standing clear of the beds projects into none of them.
    """
    kps, sc = person.keypoints, person.scores
    for z in zones:
        if z.kind not in ("bed", "chair") or z.top_m is None:
            continue
        for i in _ANKLES + _HIPS:
            if sc[i] < min_score:
                continue
            xy = ground.pixel_to_plane(float(kps[i, 0]), float(kps[i, 1]), z.top_m)
            if xy is not None and z.contains(xy):
                return z
    return None


def _resolve_bed_zone(calib, bed_word):
    """The single bed zone whose name contains ``bed_word`` (e.g. 'middle'), else None.

    None when nothing matches, or when the word is ambiguous across several bed
    zones -- the caller then warns and falls back rather than guessing a bed.
    """
    word = bed_word.strip().lower()
    hits = [z for z in calib.zones.zones if z.kind == "bed" and word in z.name.strip().lower()]
    return hits[0] if len(hits) == 1 else None


def _dominant_bed_track(track_path, ground, target_zone, min_score):
    """The track_id that occupies ``target_zone`` in the most frames, or None.

    One pass over the clip. This is the clip's target-bed occupant -- the patient
    -- identified by where they lie/sit, so we can then follow that track id even
    after they leave the bed (when bed association no longer applies).
    """
    from collections import Counter

    from ahfd.io import read_tracks

    counts: "Counter" = Counter()
    for pose in read_tracks(track_path):
        for p in pose.people:
            if p.track_id is None:
                continue
            if _person_bed_zone(p, ground, (target_zone,), min_score) is not None:
                counts[p.track_id] += 1
    return counts.most_common(1)[0][0] if counts else None


def _select_target_person(pose, target_track, target_zone, ground, min_score):
    """The person to score in a multi-occupant frame: the target-bed patient.

    Prefer the locked ``target_track`` (so they are still chosen after they stand
    and walk off the bed), else the sole person currently in the target bed, else
    None -- a frame we cannot confidently attribute is dropped, never guessed, so
    a bystander's pose never enters the training rows.
    """
    people = [p for p in pose.people if p.track_id is not None]
    if not people:
        return None
    if target_track is not None:
        for p in people:
            if p.track_id == target_track:
                return p
    in_zone = [p for p in people if _person_bed_zone(p, ground, (target_zone,), min_score) is not None]
    return in_zone[0] if len(in_zone) == 1 else None


@dataclass
class TrainResult:
    classes: list[str]
    n_total: int
    n_train: int
    n_test: int
    accuracy: float
    confusion: list[list[int]]
    report: str
    importances: list[tuple[str, float]]
    # Univariate separability: (feature, ANOVA F-score, mutual information),
    # ranked. Unlike the tree's importances this scores each feature on its own
    # and is stable on small data -- the honest answer to "which joint/angle
    # actually distinguishes the postures".
    separability: list[tuple[str, float, float]]
    # Per-class mean of each feature (nan-aware), so a difference is readable:
    # e.g. knee_angle ~95 deg sitting vs ~172 deg upright.
    class_means: dict[str, dict[str, float]]
    split_done: bool
    split_unit: str = "none"
    train_groups: list[str] = field(default_factory=list)
    test_groups: list[str] = field(default_factory=list)
    split_reason: str = ""
    model: object = field(default=None, repr=False)


def build_dataset(
    labels_dir, tracks_dir, calib, min_keypoint_score: float = 0.3,
    label_key: str = "posture", target_beds: dict | None = None,
    *, label_map: dict | None = None, clip_t0: dict | None = None,
    group_map: dict | None = None, exclude=(),
):
    """Build (rows, labels, groups, used, skipped) from labels + extracted tracks.

    ``rows`` is a list of {feature: value} dicts, ``labels`` the matching
    classes, and ``groups`` the clip each row came from -- so an evaluator can
    hold out a whole clip (frames within one recording are near-duplicates, and
    splitting them across train/test leaks). ``used`` is a list of
    (clip_id, n_frames) for coverage, ``skipped`` the label files that failed to
    parse. A clip contributes only if it has real segments *and* a matching
    ``<clip>.jsonl`` in ``tracks_dir``.

    ``label_key`` is the per-segment field read as the class: ``posture`` (the
    default, 4-class) or ``phase`` (the bed-exit activity phases). The same
    metric features and clean-transition handling apply to both.

    ``target_beds`` maps a clip stem to its target bed name (e.g.
    ``{"clip1_04": "middle"}``). For a clip in the map, the row for each frame is
    taken from the person on that bed (followed by track id once identified), so
    bystanders on other beds or walking past never enter the data. Clips absent
    from the map, or when the map is None, fall back to the highest-confidence
    person -- correct only when the clip is single-subject.

    ``label_map`` translates each label (e.g. a phase to a posture); labels not
    in the map are excluded like a gap. ``clip_t0`` gives a clip's time zero
    (wall-clock) when its labels were marked on another recording of the same
    take -- otherwise its own first frame is t=0. ``group_map`` overrides a
    clip's evaluation group, and clips named in ``exclude`` are never read.
    """
    from ahfd.features import FeatureExtractor
    from ahfd.io import read_tracks

    labels_dir, tracks_dir = Path(labels_dir), Path(tracks_dir)
    rows: list[dict] = []
    labels: list[str] = []
    groups: list[str] = []
    used: list[tuple[str, int]] = []
    skipped: list[tuple[str, str]] = []  # (filename, why) -- e.g. a JSON typo

    for label_file in sorted(labels_dir.glob("*.json")):
        if label_file.stem in exclude:
            continue
        # A hand-edited label file with a JSON typo should not crash the whole
        # run -- skip it and report which one, so the rest still trains.
        try:
            segments = _labelled_segments(label_file)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            skipped.append((label_file.name, str(exc)))
            continue
        if not segments:
            continue
        track_path = tracks_dir / (label_file.stem + ".jsonl")
        if not track_path.exists():
            continue

        # Multi-occupant clips: pick the person on this clip's target bed and
        # follow their track, instead of the highest-confidence person (which
        # would mix in bystanders). Identify the target track once, up front.
        target_zone = None
        target_track = None
        if target_beds:
            bed_word = target_beds.get(label_file.stem)
            if bed_word:
                target_zone = _resolve_bed_zone(calib, bed_word)
                if target_zone is None:
                    skipped.append((
                        label_file.name,
                        "no single bed zone matching %r in the calibration -- "
                        "fell back to the highest-confidence person" % bed_word,
                    ))
                else:
                    target_track = _dominant_bed_track(
                        track_path, calib.ground, target_zone, min_keypoint_score
                    )

        extractor = FeatureExtractor(
            calib.ground, zones=calib.zones, min_keypoint_score=min_keypoint_score
        )
        n_clip = 0
        t0 = None
        for pose in read_tracks(track_path):
            # Normalise to a 0-based clip time. Label segments are in seconds
            # from the clip start, and mp4-extracted tracks already start at
            # t=0 -- but .bag/.db3 tracks carry wall-clock timestamps, so
            # offset by the first frame's time to make them line up.
            if t0 is None:
                t0 = (clip_t0 or {}).get(label_file.stem, pose.t)
            t = pose.t - t0
            posture = _posture_at(segments, t, label_key)
            if posture is not None and label_map is not None:
                posture = label_map.get(posture)
            if posture is None:  # a transition/gap -- excluded
                continue
            if target_zone is not None:
                person = _select_target_person(
                    pose, target_track, target_zone, calib.ground, min_keypoint_score
                )
            else:
                person = _main_person(pose)
            if person is None:
                continue
            feats = extractor.extract(person, t)
            if feats is None or not feats.has_geometry():
                continue

            row = features_row(
                feats, person, extractor.ground, min_keypoint_score
            )
            if any(row.get(k) is None for k in REQUIRED):
                continue
            rows.append(row)
            labels.append(posture)
            groups.append((group_map or {}).get(label_file.stem, label_file.stem))
            n_clip += 1
        used.append((label_file.stem, n_clip))

    return rows, labels, groups, used, skipped


def pilot_locked_clips(prefix: str = "clip1_") -> set[str]:
    """Clip stems of the pilot's locked validation takes (block E: 09 and 10).

    These must never be read for training or model selection; they are opened
    once, after everything is frozen.
    """
    from ahfd.annotate import PILOT_PHASE_SCRIPT

    return {
        "%s%02d" % (prefix, i)
        for i, entry in PILOT_PHASE_SCRIPT.items() if entry["block"] == "E"
    }


def _first_track_time(track_path: Path):
    from ahfd.io import read_tracks

    for pose in read_tracks(track_path):
        return pose.t
    return None


def build_pilot_posture_dataset(
    data_dir=Path("data"), calib_dir=Path("calib"), cameras=PILOT_CAMERAS,
    min_keypoint_score: float = 0.3,
):
    """Posture rows from the ten-take bed-exit pilot's phase labels, dev takes only.

    For each camera, reads ``data/phases_<cam>`` + ``data/tracks_<cam>`` with
    ``calib/<cam>.yaml``, maps phases to postures (``PHASE_TO_POSTURE``) and
    takes each frame from the person on the take's scripted bed. Both cameras
    of a take, and both takes of a block, share one evaluation group
    (``block_A`` .. ``block_D``) so neither view of a take can sit on the other
    side of a train/test split. Block E (takes 09/10) is never read.

    The labels were marked on the D435i recordings; the D435f started ~1 s
    later, so every camera is put on the D435i's clock for that take.
    """
    from ahfd.annotate import PILOT_PHASE_SCRIPT
    from ahfd.geometry.calibration import load_calibration

    data_dir, calib_dir = Path(data_dir), Path(calib_dir)
    locked = pilot_locked_clips()
    stems = {"clip1_%02d" % i: entry for i, entry in PILOT_PHASE_SCRIPT.items()}
    target_beds = {stem: entry["bed"] for stem, entry in stems.items()}
    group_map = {stem: "block_" + entry["block"] for stem, entry in stems.items()}

    reference = data_dir / ("tracks_" + cameras[0])
    clip_t0 = {}
    for stem in stems:
        path = reference / (stem + ".jsonl")
        if stem not in locked and path.exists():
            clip_t0[stem] = _first_track_time(path)

    rows, labels, groups, used, skipped = [], [], [], [], []
    for cam in cameras:
        calib = load_calibration(calib_dir / (cam + ".yaml"))
        r, lab, grp, u, sk = build_dataset(
            data_dir / ("phases_" + cam), data_dir / ("tracks_" + cam), calib,
            min_keypoint_score=min_keypoint_score, label_key="phase",
            target_beds=target_beds, label_map=PHASE_TO_POSTURE,
            clip_t0=clip_t0, group_map=group_map, exclude=locked,
        )
        rows += r
        labels += lab
        groups += grp
        used += [(cam + "/" + clip, n) for clip, n in u]
        skipped += [(cam + "/" + name, why) for name, why in sk]
    return rows, labels, groups, used, skipped


def _evaluation_groups(groups: list[str]) -> tuple[list[str], str]:
    """Use the repository's final numeric clip suffix as the person identity.

    For example, fall_standing_03 and neg_inbed_03 must stay together. Names
    without this convention retain their supplied recording-group identity.
    """
    result = []
    all_people = True
    for group in groups:
        head, separator, tail = str(group).rpartition("_")
        if separator and head and tail.isdigit():
            result.append("person_" + tail)
        else:
            result.append(str(group))
            all_people = False
    return result, "person" if all_people else "clip"


def _group_holdout(y, groups, test_size: float, seed: int):
    """Find a group-disjoint holdout containing every class on both sides.

    Exhaust small group sets so a valid split cannot be missed by chance. For
    larger datasets, try a bounded number of reproducible group splits.
    """
    from itertools import combinations

    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    unique = np.unique(groups)
    if len(unique) < 2:
        return None
    n_test = min(len(unique) - 1, max(1, math.ceil(test_size * len(unique))))
    classes = set(y)
    if math.comb(len(unique), n_test) <= 1024:
        choices = list(combinations(unique, n_test))
        np.random.RandomState(seed).shuffle(choices)
        candidates = (
            (np.flatnonzero(~mask), np.flatnonzero(mask))
            for mask in (np.isin(groups, choice) for choice in choices)
        )
    else:
        candidates = GroupShuffleSplit(
            n_splits=64, test_size=n_test, random_state=seed
        ).split(np.zeros((len(y), 1)), y, groups)
    for train_idx, test_idx in candidates:
        if set(y[train_idx]) == classes and set(y[test_idx]) == classes:
            return train_idx, test_idx
    return None


def train(
    rows: list[dict], labels: list[str], *, groups: list[str] | None = None,
    test_size=0.3, max_depth=5, seed=0,
):
    """Fit a small decision tree on the feature table and report on it.

    Missing optional features are median-imputed; the tree is kept shallow so it
    stays interpretable and its ``feature_importances_`` are meaningful. Score
    on whole held-out people (numeric clip suffixes), or recording groups for
    other names, never random frames. Both sides must contain every posture.
    Missing groups or insufficient coverage gives explicitly flagged training
    accuracy, not a pretend test score. ``test_size`` is the fraction of groups
    held out. The returned inference model is then fitted on ALL available rows;
    held-out metrics are calculated first with a separate training-only model.
    """
    import warnings

    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    if not rows or len(rows) != len(labels):
        raise ValueError("rows and labels must be nonempty and have matching lengths")
    if groups is not None and len(groups) != len(rows):
        raise ValueError("groups must have one entry per row")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction strictly between 0 and 1")

    X = np.array(
        [[np.nan if r.get(k) is None else float(r[k]) for k in FEATURES] for r in rows],
        dtype=float,
    )
    y = np.array(labels)
    classes = sorted(set(labels))

    def make_model():
        return Pipeline(
            [
                # keep_empty_features: a feature absent from every row (e.g. no
                # wrists ever visible) must stay a column so importances line up
                # with FEATURES; it is filled with 0 and simply carries no signal.
                (
                    "impute",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("clf", DecisionTreeClassifier(max_depth=max_depth, random_state=seed)),
            ]
        )

    counts = {c: int((y == c).sum()) for c in classes}
    split_unit = "none"
    group_ids = np.array([], dtype=str)
    split = None
    if groups is None:
        split_reason = "no recording/person groups supplied"
    else:
        evaluation_groups, split_unit = _evaluation_groups(groups)
        group_ids = np.asarray(evaluation_groups)
        if len(np.unique(group_ids)) < 2:
            split_reason = "need at least two independent " + split_unit + " groups"
        elif len(classes) < 2:
            split_reason = "need at least two posture classes for a held-out score"
        else:
            split = _group_holdout(y, group_ids, test_size, seed)
            split_reason = (
                "group-disjoint holdout with every posture represented"
                if split is not None else
                "no class-complete group-disjoint holdout found"
            )

    split_done = split is not None
    test_groups: list[str] = []
    if split is not None:
        train_idx, test_idx = split
        score_model = make_model().fit(X[train_idx], y[train_idx])
        y_pred = score_model.predict(X[test_idx])
        acc = float(accuracy_score(y[test_idx], y_pred))
        report = classification_report(y[test_idx], y_pred, zero_division=0)
        conf = confusion_matrix(y[test_idx], y_pred, labels=classes)
        n_tr, n_te = len(train_idx), len(test_idx)
        train_groups = sorted(set(group_ids[train_idx]))
        test_groups = sorted(set(group_ids[test_idx]))
    else:
        train_groups = sorted(set(group_ids))

    # Save/export the model trained on all available data, not the holdout fold.
    model = make_model().fit(X, y)
    if not split_done:
        y_pred = model.predict(X)
        acc = float(accuracy_score(y, y_pred))
        report = classification_report(y, y_pred, zero_division=0)
        conf = confusion_matrix(y, y_pred, labels=classes)
        n_tr, n_te = len(y), 0

    tree = model.named_steps["clf"]
    importances = sorted(
        zip(FEATURES, (float(v) for v in tree.feature_importances_)),
        key=lambda kv: kv[1],
        reverse=True,
    )

    # Univariate separability, computed on the median-imputed table so every
    # feature is scored on the same rows. ANOVA F asks "do the class means
    # differ relative to the spread"; mutual information catches non-linear
    # splits an F-test misses. Constant columns give a degenerate F (nan) and a
    # divide-by-zero warning -- silence it and treat them as zero signal. Both
    # need at least two samples per class to be defined, so skip them on the
    # degenerate tiny-data path (where the number is meaningless anyway).
    separability: list[tuple[str, float, float]] = []
    if len(classes) >= 2 and min(counts.values()) >= 2 and len(rows) >= 4:
        X_imp = SimpleImputer(
            strategy="median", keep_empty_features=True
        ).fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, _ = f_classif(X_imp, y)
            mi = mutual_info_classif(X_imp, y, random_state=seed)
        separability = sorted(
            (
                (feat, float(np.nan_to_num(f)), float(m))
                for feat, f, m in zip(FEATURES, f_scores, mi)
            ),
            key=lambda t: t[1],
            reverse=True,
        )

    # Per-class feature means, nan-aware so a missing value in a row does not
    # poison the average. All-nan (feature never seen for a class) reads as nan.
    class_means: dict[str, dict[str, float]] = {}
    for c in classes:
        rowsel = X[y == c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            means = np.nanmean(rowsel, axis=0)
        class_means[c] = {feat: float(m) for feat, m in zip(FEATURES, means)}

    return TrainResult(
        classes=classes,
        n_total=len(rows),
        n_train=n_tr,
        n_test=n_te,
        accuracy=acc,
        confusion=[[int(x) for x in row] for row in conf],
        report=report,
        importances=importances,
        separability=separability,
        class_means=class_means,
        split_done=split_done,
        split_unit=split_unit,
        train_groups=train_groups,
        test_groups=test_groups,
        split_reason=split_reason,
        model=model,
    )
