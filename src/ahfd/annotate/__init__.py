"""Interactive labelling helpers (posture segments, run on a laptop with a display).

These tools only ever *read* video and *display* it; the sole thing they write
is a small JSON of time-coded labels. No frame is ever saved -- the privacy
guard (tests/test_privacy.py) forbids it, and none is needed.
"""

from ahfd.annotate.falls import (
    derive_annotations,
    derive_falls,
    dump_annotation_json,
)
from ahfd.annotate.phase_script import (
    PILOT_PHASE_SCRIPT,
    seed_phase_drafts,
    seed_segments,
)
from ahfd.annotate.posture_labeler import (
    PHASE_CLASSES,
    PHASE_SCHEME,
    POSTURE_CLASSES,
    POSTURE_SCHEME,
    LabelScheme,
    dump_posture_json,
    load_existing_segments,
    run_labeler,
)

__all__ = [
    "LabelScheme",
    "PHASE_CLASSES",
    "PHASE_SCHEME",
    "PILOT_PHASE_SCRIPT",
    "POSTURE_CLASSES",
    "POSTURE_SCHEME",
    "derive_annotations",
    "derive_falls",
    "dump_annotation_json",
    "dump_posture_json",
    "load_existing_segments",
    "run_labeler",
    "seed_phase_drafts",
    "seed_segments",
]
