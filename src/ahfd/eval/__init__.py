"""Event-level evaluation: recall, false alarms per camera-hour, latency."""

from ahfd.eval.annotations import (
    GroundTruth,
    PredictedEvent,
    TrueFall,
    load_events,
)
from ahfd.eval.metrics import Report, evaluate, match_clip
from ahfd.eval.report import render_markdown

__all__ = [
    "GroundTruth",
    "PredictedEvent",
    "TrueFall",
    "load_events",
    "Report",
    "evaluate",
    "match_clip",
    "render_markdown",
]
