"""Derive fall ground-truth from posture labels.

A fall is just a posture transition: someone who was **upright or sitting** is
now **on the ground**, and the fall happened in between. So the posture
segments you mark for the classifier already contain the fall annotations --
this reads them out rather than making you label impact times a second time.

For each ``on_ground`` segment whose previous labelled posture was elevated:

* ``t_start``  = when the previous posture *ended* -- the fall begins here.
* ``t_impact`` = when the ``on_ground`` hold *starts* -- the body is down here.

The gap between them is the fall itself (the transition you deliberately left
unlabelled). ``t_impact`` is what the evaluator matches alerts against; the
30 s post-window easily covers the ~8 s the detector waits to confirm, so a
slightly late impact estimate costs nothing.

A clip whose postures never reach ``on_ground`` (or only from lying down) yields
no falls -- a valid negative, whose duration is still recorded because it is the
false-alarm denominator.
"""

from __future__ import annotations

import json
from pathlib import Path

from ahfd.annotate.posture_labeler import load_existing_segments

# Postures you can fall *from*. Lying (in_bed / on_ground) is excluded: you do
# not "fall" from already being down, and a bed-to-floor roll is a different
# event that the bed geometry, not this heuristic, should own.
ELEVATED_POSTURES: tuple[str, ...] = ("upright", "sitting")
FLOOR_POSTURE = "on_ground"

# Clip-name prefixes that ARE negatives by construction -- normal activity, safe
# bed exits. They need no posture labels to be a valid negative: an empty falls
# list plus the clip duration is the whole annotation, and that duration is the
# false-alarm denominator. This lets one `derive-falls` run keep every negative
# correct without hand-maintaining them.
NEGATIVE_PREFIXES: tuple[str, ...] = ("neg_", "bedexit_")


def derive_falls(
    segments: list[dict],
    *,
    elevated: tuple[str, ...] = ELEVATED_POSTURES,
    floor: str = FLOOR_POSTURE,
) -> list[dict]:
    """Fall records from posture segments: each elevated -> floor transition.

    ``segments`` are ``{start_s, end_s, posture}`` dicts; placeholders and
    out-of-order entries are handled by sorting and ignoring zero-length rows.
    """
    segs = sorted(
        (s for s in segments if float(s["end_s"]) > float(s["start_s"])),
        key=lambda s: float(s["start_s"]),
    )
    falls: list[dict] = []
    for i, seg in enumerate(segs):
        if seg["posture"] != floor or i == 0:
            continue
        prev = segs[i - 1]
        if prev["posture"] in elevated:
            falls.append(
                {
                    "t_impact": round(float(seg["start_s"]), 2),
                    "t_start": round(float(prev["end_s"]), 2),
                    "notes": "derived from postures: " + prev["posture"] + " -> " + floor,
                }
            )
    return falls


def dump_annotation_json(clip_id: str, duration_s: float, falls: list[dict]) -> str:
    """Format fall ground-truth as the project's compact one-per-line JSON."""
    lines = [
        '    { "t_impact": %s, "t_start": %s, "notes": %s }'
        % (_num(f["t_impact"]), _num(f["t_start"]), json.dumps(f.get("notes", "")))
        for f in falls
    ]
    body = ",\n".join(lines)
    return (
        "{\n"
        '  "clip_id": "' + clip_id + '",\n'
        '  "duration_s": ' + _num(duration_s) + ",\n"
        '  "falls": [\n'
        + (body + "\n" if body else "")
        + "  ]\n"
        "}\n"
    )


def _num(x: float) -> str:
    return format(round(float(x), 2), "g")


def _is_placeholder_annotation(path: Path) -> bool:
    """True if an annotation is an un-filled placeholder (fall(s) all at t=0)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    falls = data.get("falls", [])
    return bool(falls) and all(float(f.get("t_impact", 0.0)) == 0.0 for f in falls)


def derive_annotations(
    postures_dir: Path,
    annotations_dir: Path,
    *,
    elevated: tuple[str, ...] = ELEVATED_POSTURES,
    floor: str = FLOOR_POSTURE,
    prune_placeholders: bool = False,
) -> tuple[list[tuple[str, int]], list[str], list[str]]:
    """Regenerate the fall ground-truth from posture labels.

    Posture labels are the single source of truth. For each clip's posture file:

    * a negative-by-name clip (``neg_*`` / ``bedexit_*``) -> write an empty-falls
      negative from its duration, even if its posture labels include deliberate
      floor activity;
    * real segments in other clips -> derive falls (elevated -> floor transitions;
      may be empty);
    * no segments and a fall clip -> left *unlabelled* (we cannot invent impacts).

    Returns (written, unlabelled, pruned): ``written`` is (clip_id, n_falls) for
    every annotation written (negatives included, n_falls 0), ``unlabelled`` the
    fall clips still awaiting posture labels, ``pruned`` the stale placeholder
    annotations removed when ``prune_placeholders`` is set. Re-run after every
    labelling session; the fall ground-truth stays in sync.
    """
    postures_dir, annotations_dir = Path(postures_dir), Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []
    unlabelled: list[str] = []
    pruned: list[str] = []
    for path in sorted(postures_dir.glob("*.json")):
        clip_id, segs = load_existing_segments(path)
        stem = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        duration_s = float(data.get("duration_s", 0.0))
        out = annotations_dir / (stem + ".json")

        if stem.startswith(NEGATIVE_PREFIXES):
            # Posture training still uses these labels, but intentional floor
            # activity in a known negative must never hide a false alarm.
            falls = []
        elif segs:
            falls = derive_falls(segs, elevated=elevated, floor=floor)
        else:
            # An unlabelled fall clip: we must not fabricate impacts. Leave it
            # out of the ground truth (optionally removing a stale placeholder
            # so it can never be scored as garbage).
            unlabelled.append(stem)
            if prune_placeholders and out.exists() and _is_placeholder_annotation(out):
                out.unlink()
                pruned.append(stem)
            continue

        out.write_text(
            dump_annotation_json(clip_id or stem, duration_s, falls), encoding="utf-8"
        )
        written.append((stem, len(falls)))
    return written, unlabelled, pruned
