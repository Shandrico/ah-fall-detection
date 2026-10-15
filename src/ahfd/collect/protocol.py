"""Frozen, single-volunteer A--E research plans; never a clinical care policy.

The plan contains pseudonyms and simulated bed modes, not action timestamps.
Observer labels remain independent of the script and of detector predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

PLAN_SCHEMA = "ahfd.collection.block-plan.v1"
BLOCKS = ("A", "B", "C", "D", "E")
MODES = ("low", "medium", "high")

# Point markers, not inferred interval labels or instructions to perform falls.
EXTRA_CONTEXT_KEYS = {
    ord("e"): "BED_EXIT_COMPLETED",
    ord("b"): "ABORTED_EXIT",
    ord("w"): "WARNING_WORTHY_ATTEMPT",
    ord("j"): "FALL_ONSET",
    ord("k"): "FLOOR_CONTACT",
    ord("l"): "RECOVERY",
    ord("o"): "INTENTIONAL_FLOOR_POSTURE",
}


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " has missing or unknown fields")


def _code(value, pattern, label):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError("invalid pseudonymous " + label)


def validate_plan(plan: dict) -> dict:
    _exact(plan, {
        "schema", "plan_id", "site_id", "participant_id", "total_minutes",
        "beds", "blocks", "scope", "capture_policy",
    }, "collection plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("unsupported collection plan schema")
    _code(plan["plan_id"], r"plan_[0-9a-f]{16}", "plan_id")
    _code(plan["site_id"], r"site_[0-9a-f]{8}", "site_id")
    _code(plan["participant_id"], r"sub_[0-9a-f]{16}", "participant_id")
    if type(plan["total_minutes"]) is not int or plan["total_minutes"] not in (40, 60):
        raise ValueError("total_minutes must be 40 or 60")
    if plan["scope"] != "same_volunteer_block_holdout":
        raise ValueError("plan must declare same-volunteer block holdout")
    if plan["capture_policy"] != "derived_only_shadow":
        raise ValueError("block collection is derived-only and shadow-only")
    beds = plan["beds"]
    if not isinstance(beds, dict) or len(beds) != 2:
        raise ValueError("plan requires exactly two distinct bed codes")
    for bed, mode in beds.items():
        _code(bed, r"bed_[a-z0-9]{1,20}", "bed code")
        if mode not in MODES:
            raise ValueError("simulated monitoring mode must be low, medium or high")
    expected = {block: "validation" if block == "E" else "development" for block in BLOCKS}
    if plan["blocks"] != expected:
        raise ValueError("frozen split must be A-D development and E validation")
    # Detached strict-JSON copy; callers cannot later mutate recorder provenance.
    return json.loads(json.dumps(plan, allow_nan=False))


def new_plan(site_id: str, participant_id: str, beds: dict, total_minutes: int) -> dict:
    return validate_plan({
        "schema": PLAN_SCHEMA,
        "plan_id": "plan_" + uuid.uuid4().hex[:16],
        "site_id": site_id,
        "participant_id": participant_id,
        "total_minutes": total_minutes,
        "beds": beds,
        "blocks": {b: "validation" if b == "E" else "development" for b in BLOCKS},
        "scope": "same_volunteer_block_holdout",
        "capture_policy": "derived_only_shadow",
    })


def plan_sha256(plan: dict) -> str:
    canonical = json.dumps(validate_plan(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_plan(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError("collection plan is too large")
    return validate_plan(json.loads(path.read_text(encoding="utf-8-sig")))


def assignment(plan: dict, block: str, target_bed: str) -> dict:
    plan = validate_plan(plan)
    if block not in BLOCKS or target_bed not in plan["beds"]:
        raise ValueError("block/target-bed must be declared in the frozen plan")
    return {
        "plan": plan,
        "plan_sha256": plan_sha256(plan),
        "block": block,
        "partition": plan["blocks"][block],
        "target_bed": target_bed,
        # Five blocks, two separately recorded beds per block.
        "target_usable_s": plan["total_minutes"] * 60 // 10,
    }


def validate_assignment(value: dict, *, site_id: str, participant_id: str | None) -> dict:
    _exact(value, {"plan", "plan_sha256", "block", "partition", "target_bed", "target_usable_s"}, "collection protocol")
    expected = assignment(value["plan"], value["block"], value["target_bed"])
    if value != expected or type(value["target_usable_s"]) is not int:
        raise ValueError("protocol differs from its frozen plan/hash/partition")
    if expected["plan"]["site_id"] != site_id or expected["plan"]["participant_id"] != participant_id:
        raise ValueError("plan site/participant does not match this session")
    return expected


def validate_progress(value: dict, target_s: float, *, complete: bool = False) -> dict:
    _exact(value, {"usable_s", "target_bed_observed"}, "protocol progress")
    seconds = value["usable_s"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("usable_s must be finite and non-negative")
    if type(value["target_bed_observed"]) is not bool:
        raise ValueError("target_bed_observed must be boolean")
    if complete and (seconds + 1e-6 < target_s or not value["target_bed_observed"]):
        raise ValueError("planned session has insufficient labelled available time or unverified target bed")
    return dict(value)


@dataclass
class ProtocolProgress:
    """Credit only adjacent eligible 10-Hz rows of the same association.

    Warmup, unlabelled time, unavailability, long gaps and reassociation are not
    backfilled. This is a collection-duration gate, NOT annotation certification.
    """
    target_s: float
    usable_s: float = 0.0
    target_bed_observed: bool = False
    _previous_t: float | None = None
    _previous_association: tuple | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_s) or self.target_s <= 0:
            raise ValueError("target duration must be finite and positive")

    def reset_binding(self) -> None:
        """Reverify the bed after an operator binds a new association."""
        self.target_bed_observed = False
        self._previous_association = None

    def observe(self, t: float, *, eligible: bool, association: tuple | None) -> None:
        if not math.isfinite(t) or t < 0 or (self._previous_t is not None and t < self._previous_t):
            raise ValueError("progress requires monotonic finite timestamps")
        current = association if eligible and self.target_bed_observed else None
        if current is not None and current == self._previous_association and self._previous_t is not None:
            dt = t - self._previous_t
            if dt <= 0.20 + 1e-9:
                self.usable_s += dt
        self._previous_t = t
        self._previous_association = current

    @property
    def ready(self) -> bool:
        return self.target_bed_observed and self.usable_s + 1e-6 >= self.target_s

    def snapshot(self) -> dict:
        return {"usable_s": round(self.usable_s, 6), "target_bed_observed": self.target_bed_observed}


def guard_development_inputs(session_dirs, group_by: str) -> None:
    """Fail BEFORE reading observations if a training command is handed E.

    Also keep both bed recordings (and repeats) of each block in one CV group.
    Explicit held-out evaluation must be a separate future command, not a flag
    that silently turns validation recordings into training samples.
    """
    protocols = []
    for directory in session_dirs:
        manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        protocol = manifest.get("collection_protocol")
        if protocol is not None:
            protocol = validate_assignment(protocol, site_id=manifest.get("site_id"), participant_id=manifest.get("participant_id"))
        protocols.append(protocol)
    planned = [p for p in protocols if p is not None]
    if not planned:
        return
    if len(planned) != len(protocols):
        raise ValueError("do not mix planned blocks with legacy/October-13 sessions")
    if len({p["plan_sha256"] for p in planned}) != 1:
        raise ValueError("planned sessions must share one unchanged collection plan")
    if any(p["partition"] != "development" for p in planned):
        raise ValueError("Block E is locked validation and cannot enter comparator training")
    if group_by != "block":
        raise ValueError("planned development sessions require --group-by block")
