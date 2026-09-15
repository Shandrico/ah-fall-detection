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
from ahfd.annotate.posture_labeler import (
    POSTURE_CLASSES,
    dump_posture_json,
    load_existing_segments,
    run_labeler,
)

__all__ = [
    "POSTURE_CLASSES",
    "derive_annotations",
    "derive_falls",
    "dump_annotation_json",
    "dump_posture_json",
    "load_existing_segments",
    "run_labeler",
]
