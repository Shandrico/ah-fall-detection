"""First-draft bed-exit phase labels from the frozen collection script.

The pilot was recorded to a fixed A-E / nearest-middle protocol (ten takes,
~60 s each). The script already says, roughly, which activity phase happens
when -- so it is a first draft of the labels, not a blank page. This module
turns that script into seed ``phase`` segment files, which the offline labeller
(:func:`ahfd.annotate.run_labeler` with ``PHASE_SCHEME``) then opens for
correction: you only nudge boundaries for the real recording delay instead of
marking every segment from scratch.

The seed uses the four primary phases the training scheme recommends starting
with -- RECLINED, UPRIGHT_IN_BED, EDGE_SITTING, OUT_OF_BED (docs/TRAINING_SCHEME
section 5.4) -- plus SHIFTING_TO_EDGE where the script moves to the edge without
sitting. The finer transition phases (TORSO_RISING, ATTEMPTING_STAND) are left
for the human to add by hand where wanted; they are deliberately not guessed.

A seed is intentionally approximate. ``seed_phase_drafts`` never overwrites a
file that already exists, so re-running it cannot clobber corrections you have
already made.
"""

from __future__ import annotations

from pathlib import Path

from ahfd.annotate.posture_labeler import PHASE_SCHEME, dump_posture_json

# One entry per take. ``block``/``bed`` are provenance for the operator; the
# ``segments`` are (start_s, end_s, phase) as scripted (nominal 0-60 s). The
# real takes drift by a few seconds -- that is exactly what you correct in the
# labeller. Phases must be members of PHASE_SCHEME.classes.
PILOT_PHASE_SCRIPT: dict[int, dict] = {
    1: {  # A nearest: reclined rest, roll, return, rest -- all in bed
        "block": "A", "bed": "nearest",
        "segments": [(0, 60, "RECLINED")],
    },
    2: {  # A middle: reclined, reach for phone, reposition, rest -- all in bed
        "block": "A", "bed": "middle",
        "segments": [(0, 60, "RECLINED")],
    },
    3: {  # B nearest: sit up, edge-sit, stand and leave, remain standing
        "block": "B", "bed": "nearest",
        "segments": [
            (0, 15, "RECLINED"),
            (15, 25, "UPRIGHT_IN_BED"),
            (25, 35, "EDGE_SITTING"),
            (35, 60, "OUT_OF_BED"),
        ],
    },
    4: {  # B middle: sit up, stay seated in bed, move to edge, stand and leave
        "block": "B", "bed": "middle",
        "segments": [
            (0, 10, "RECLINED"),
            (10, 40, "UPRIGHT_IN_BED"),
            (40, 50, "EDGE_SITTING"),
            (50, 60, "OUT_OF_BED"),
        ],
    },
    5: {  # C nearest: aborted exit -- edge-sit, move back onto bed, recline
        "block": "C", "bed": "nearest",
        "segments": [
            (0, 15, "RECLINED"),
            (15, 25, "UPRIGHT_IN_BED"),
            (25, 40, "EDGE_SITTING"),
            (40, 50, "UPRIGHT_IN_BED"),
            (50, 60, "RECLINED"),
        ],
    },
    6: {  # C middle: sit up, remain upright, recline, roll/reposition, rest
        "block": "C", "bed": "middle",
        "segments": [
            (0, 10, "RECLINED"),
            (10, 35, "UPRIGHT_IN_BED"),
            (35, 60, "RECLINED"),
        ],
    },
    7: {  # D nearest: sit up + edge + leave, walk to chair, sit, remain seated
        "block": "D", "bed": "nearest",
        "segments": [
            (0, 10, "RECLINED"),
            (10, 18, "UPRIGHT_IN_BED"),
            (18, 24, "EDGE_SITTING"),
            (24, 60, "OUT_OF_BED"),
        ],
    },
    8: {  # D middle: sit up + edge + leave, walk, bend to pick up, stand upright
        "block": "D", "bed": "middle",
        "segments": [
            (0, 15, "RECLINED"),
            (15, 23, "UPRIGHT_IN_BED"),
            (23, 30, "EDGE_SITTING"),
            (30, 60, "OUT_OF_BED"),
        ],
    },
    9: {  # E nearest (held out): rest, reach, sit up, pause upright, edge + leave
        "block": "E", "bed": "nearest",
        "segments": [
            (0, 25, "RECLINED"),
            (25, 45, "UPRIGHT_IN_BED"),
            (45, 52, "EDGE_SITTING"),
            (52, 60, "OUT_OF_BED"),
        ],
    },
    10: {  # E middle (held out): rest, reach, sit up, recline, rest/read in bed
        "block": "E", "bed": "middle",
        "segments": [
            (0, 20, "RECLINED"),
            (20, 30, "UPRIGHT_IN_BED"),
            (30, 60, "RECLINED"),
        ],
    },
}


def seed_segments(clip_index: int) -> list[dict]:
    """The scripted draft segments for one take, as labeller segment dicts.

    Raises ``KeyError`` for an unknown index and ``ValueError`` if the script
    ever drifts to a phase the scheme does not define (a guard for edits here).
    """
    entry = PILOT_PHASE_SCRIPT[clip_index]
    segs = []
    for start_s, end_s, phase in entry["segments"]:
        if phase not in PHASE_SCHEME.classes:
            raise ValueError("not a phase class: " + repr(phase))
        segs.append({"start_s": float(start_s), "end_s": float(end_s), "phase": phase})
    return segs


def seed_phase_drafts(
    out_dir,
    *,
    clips=range(1, 11),
    prefix: str = "clip1_",
    overwrite: bool = False,
) -> list[Path]:
    """Write seed ``<prefix><NN>.json`` phase drafts into ``out_dir``.

    Returns the paths actually written. An existing file is skipped unless
    ``overwrite`` is set, so corrections are never clobbered. The ``duration_s``
    recorded is the script's nominal length; the labeller replaces it with the
    real clip length on first save.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in clips:
        path = out_dir / ("%s%02d.json" % (prefix, i))
        if path.exists() and not overwrite:
            continue
        segs = seed_segments(i)
        duration = max((s["end_s"] for s in segs), default=0.0)
        clip_id = "%s%02d" % (prefix, i)
        path.write_text(
            dump_posture_json(clip_id, duration, segs, label_key="phase"),
            encoding="utf-8",
        )
        written.append(path)
    return written
