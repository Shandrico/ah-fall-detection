"""Runtime per-bed alert policy for the dashboard.

The activity detector answers *what is happening*.  This module answers which
of those findings should interrupt staff for the selected bed.  Keeping that
choice outside calibration is important: geometry belongs to the camera/mount,
while a monitoring mode can change with the current care plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from ahfd.detect.events import Event

BedMode = Literal["low", "medium", "high"]
VALID_BED_MODES: tuple[BedMode, ...] = ("low", "medium", "high")
DEFAULT_BED_MODE: BedMode = "medium"


def mode_from_legacy_risk(risk: str | None) -> BedMode:
    """Choose an initial runtime mode without making calibration authoritative.

    Onsite calibrations deliberately store ``unknown`` and therefore start in
    medium mode.  Older development calibrations still get a unsurprising
    initial value, but a dashboard selection remains the runtime authority.
    """

    value = (risk or "unknown").strip().lower()
    if value in {"none", "low"}:
        return "low"
    if value == "high":
        return "high"
    return DEFAULT_BED_MODE


def apply_bed_mode(event: Event, mode: BedMode) -> Event | None:
    """Filter and grade one detector event for a bed's selected mode.

    Fall events never enter this function.  A medium bed receives only a
    completed out-of-bed event.  A high bed additionally receives the
    CUSUM-based early warning, which the state machine has already required to
    be corroborated by persistent edge/support evidence.
    """

    if mode not in VALID_BED_MODES:
        raise ValueError("unknown bed monitoring mode " + repr(mode))
    if event.type not in {"BED_EXIT", "BED_EXIT_WARNING"}:
        return event
    expected_trigger = {
        "BED_EXIT": "out_of_bed",
        "BED_EXIT_WARNING": "early_warning",
    }[event.type]
    if event.evidence.get("trigger") != expected_trigger:
        # Never reinterpret a legacy sit-up/in-bed-movement event as the new
        # temporal completed-exit or CUSUM warning contract.
        return None
    if mode == "low":
        return None
    if mode == "medium" and event.type != "BED_EXIT":
        return None
    severity = 3 if mode == "high" else 2
    evidence = {**event.evidence, "monitoring_mode": mode}
    return replace(event, severity_override=severity, evidence=evidence)


def is_dashboard_alert(event: Event) -> bool:
    """Whether an admitted event belongs in the acknowledgement queue."""

    return event.type in {
        "FALL_CONFIRMED",
        "PERSON_DOWN",
        "BED_EXIT",
        "BED_EXIT_WARNING",
    }
