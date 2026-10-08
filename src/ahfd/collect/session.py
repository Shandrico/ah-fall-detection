"""Durable, derived-only session records for an onsite shadow study.

``SessionRecorder`` is intentionally narrower than a general JSON logger.  It
accepts JSON-safe derived values only, rejects common imagery and direct-PII
field names, owns a new session directory, and leaves an ``incomplete``
manifest until the operator explicitly completes or aborts the session.

That narrow contract matters in a ward: a process crash must leave usable data
without falsely claiming a completed session, and adding a convenient field
must not silently turn a keypoint log into a raw-image recorder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

MANIFEST_SCHEMA = "ahfd.collection.session"
MANIFEST_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1

_STREAM_FILES = {
    "derived": "derived.jsonl",
    "telemetry": "telemetry.jsonl",
    "labels": "labels.jsonl",
}

# IDs are fixed-width random hex codes, not prose.  A permissive ``[a-z0-9]``
# suffix still admits names, ward labels and dates that happen to fit the
# length limit.  Exact widths also give operators one canonical format to
# validate before a collection begins.
_ID_RULES = {
    "session": re.compile(r"^ses_[0-9a-f]{16}$"),
    "site": re.compile(r"^site_[0-9a-f]{8}$"),
    "participant": re.compile(r"^sub_[0-9a-f]{16}$"),
}

_ID_EXAMPLES = {
    "session": "ses_0123456789abcdef",
    "site": "site_0123abcd",
    "participant": "sub_0123456789abcdef",
}

# Optional asset identifiers are opaque inventory codes.  They must never be
# repurposed as a filename or a free-text note; the content hashes below are
# the authoritative provenance.
_ASSET_ID_RULES = {
    "config": re.compile(r"^cfg_[0-9a-f]{16}$"),
    "calibration": re.compile(r"^cal_[0-9a-f]{16}$"),
}

_TEMPORAL_SCHEMA_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TEMPORAL_FEATURE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_TEMPORAL_FEATURES = 2048

# Keys are normalised by dropping punctuation and case.  Ban direct identifiers
# at every nesting level rather than trusting each caller to remember a list.
_PII_KEYS = {
    "name",
    "patientname",
    "participantname",
    "subjectname",
    "firstname",
    "lastname",
    "fullname",
    "patientid",
    "participantid",
    "subjectid",
    "medicalrecordnumber",
    "mrn",
    "nric",
    "nationalid",
    "dateofbirth",
    "dob",
    "email",
    "phone",
    "phonenumber",
    "address",
    "roomnumber",
    "wardnumber",
    "bednumber",
}

# Scalars such as ``depth_height_m`` and ``frame_index`` are allowed.  These
# names denote image payloads, which are outside the derived-only contract.
_IMAGERY_KEYS = {
    "bgr",
    "rgb",
    "image",
    "img",
    "pixels",
    "framedata",
    "depthimage",
    "depthraw",
    "jpeg",
    "jpg",
    "png",
    "bitmap",
}

_RESERVED_RECORD_KEYS = {"t_rel_s", "record_schema_version"}
_ABORT_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")

# Session records are not a general-purpose JSON channel.  Every mapping key
# emitted by the reviewed collector is enumerated here, including nested
# evidence and health fields.  This prevents a future call site from hiding an
# image under ``payload`` or free-text PII under ``note`` while still satisfying
# the JSON-only validator.
_ALLOWED_RECORD_KEYS = frozenset(
    {
        "kind",
        "frame_index",
        "source_t_s",
        "monitoring",
        "people",
        "events",
        "event_id",
        "status",
        "reasons",
        "details",
        "last_frame_t_rel_s",
        "track_id",
        "association_epoch",
        "keypoints_xy",
        "keypoint_scores",
        "joint_heights_m",
        "features",
        "fall_state",
        "bed_activity",
        "temporal_summary",
        "schema_version",
        "values",
        "contact_xy_m",
        "range_m",
        "h_torso_m",
        "h_shoulder_m",
        "h_head_m",
        "floor_spread_m",
        "vertical_velocity_mps",
        "motion_mps",
        "n_valid_keypoints",
        "mean_confidence",
        "zones",
        "associated_bed",
        "supported_by_bed",
        "bed_support_fraction",
        "bed_edge_distance_m",
        "torso_tilt_deg",
        "t",
        "phase",
        "phase_since",
        "support",
        "observation",
        "observation_since",
        "bed_id",
        "shoulder_elevation_m",
        "support_fraction",
        "edge_distance_m",
        "edge_velocity_mps",
        "cusum_armed",
        "cusum_z",
        "cusum_g",
        "baseline_mean",
        "baseline_std",
        "onset_t",
        "early_warning_candidate",
        "phase_elapsed_s",
        "observation_elapsed_s",
        "type",
        "severity",
        "zone",
        "trigger_t_s",
        "alert_t_s",
        "evidence",
        "trigger",
        "height_source",
        "bed_risk",
        "onset_to_alert_s",
        "support_fraction",
        "peak_vz",
        "h_before",
        "h_after",
        "h_torso",
        "floor_spread",
        "down_s",
        "motion",
        "n_valid_kp",
        "recovered_after_s",
        "no_impact_detected",
        "dwell_s",
        "seated_s",
        "previous",
        "current",
        "returned_track_ids",
        "new_track_ids",
        "association_epochs",
        "frames_seen",
        "effective_fps",
        "disk_free_bytes",
        "people",
        "frame_depth_valid_fraction",
        "target_joint_depth_fraction",
        "ankle_drift_samples",
        "ankle_drift_span_s",
        "ankle_height_mean_m",
        "calibration_check",
        "imu_orientation_check",
        "imu_pitch_delta_deg",
        "imu_roll_delta_deg",
        "condition",
        "check",
        "frame_age_s",
        "track_ids",
        "associations",
        "value",
    }
)

_RECORD_FIELDS = {
    "derived": {
        "frame_observation": frozenset(
            {"kind", "frame_index", "source_t_s", "monitoring", "people", "events"}
        )
    },
    "telemetry": {
        "monitoring_health_transition": frozenset(
            {"kind", "previous", "current", "reasons", "details"}
        ),
        "association_reset": frozenset(
            {"kind", "returned_track_ids", "new_track_ids", "association_epochs"}
        ),
        "shadow_event": frozenset(
            {
                "kind",
                "event_id",
                "type",
                "track_id",
                "association_epoch",
                "frame_index",
                "source_t_s",
                "severity",
                "zone",
                "trigger_t_s",
                "alert_t_s",
                "evidence",
            }
        ),
        "heartbeat": frozenset(
            {"kind", "frames_seen", "effective_fps", "disk_free_bytes", "monitoring"}
        ),
        "target_binding": frozenset(
            {"kind", "track_id", "association_epoch"}
        ),
    },
    "labels": {
        "phase_marker": frozenset(
            {"kind", "track_ids", "associations", "phase"}
        ),
        "context_marker": frozenset(
            {"kind", "track_ids", "associations", "value"}
        ),
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_pseudonymous_id(kind: str) -> str:
    """Return a random identifier with no person, bed, room, or date encoded."""
    if kind not in _ID_RULES:
        raise ValueError("identifier kind must be session, site, or participant")
    prefix = {"session": "ses", "site": "site", "participant": "sub"}[kind]
    # Site codes are usually assigned once by governance rather than generated,
    # but supporting generation keeps tests and dry-runs privacy-safe by default.
    size = {"session": 16, "site": 8, "participant": 16}[kind]
    return prefix + "_" + uuid.uuid4().hex[:size]


def _validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _ID_RULES[kind].fullmatch(value):
        raise ValueError(
            kind + "_id must be a fixed-length random-hex pseudonym like "
            + _ID_EXAMPLES[kind]
            + "; do not encode a name, MRN, room, bed, or date"
        )
    return value


def _validate_asset_id(value: str | None, kind: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _ASSET_ID_RULES[kind].fullmatch(value):
        prefix = "cfg" if kind == "config" else "cal"
        raise ValueError(
            kind
            + "_asset_id must be an opaque code like "
            + prefix
            + "_0123456789abcdef; filenames and free text are not allowed"
        )
    return value


def _temporal_schema_metadata(
    schema_version: str | None,
    feature_names: Sequence[str] | None,
) -> dict[str, Any] | None:
    """Validate and canonicalise one session-wide temporal feature schema."""
    if schema_version is None and feature_names is None:
        return None
    if schema_version is None or feature_names is None:
        raise ValueError(
            "temporal_schema_version and temporal_feature_names must be supplied together"
        )
    if (
        not isinstance(schema_version, str)
        or not _TEMPORAL_SCHEMA_VERSION.fullmatch(schema_version)
    ):
        raise ValueError(
            "temporal_schema_version must be a short controlled ASCII token"
        )
    if isinstance(feature_names, (str, bytes, bytearray)) or not isinstance(
        feature_names, Sequence
    ):
        raise TypeError("temporal_feature_names must be an ordered sequence of strings")

    names = list(feature_names)
    if not names or len(names) > _MAX_TEMPORAL_FEATURES:
        raise ValueError(
            "temporal_feature_names must contain 1-"
            + str(_MAX_TEMPORAL_FEATURES)
            + " names"
        )
    for name in names:
        if not isinstance(name, str) or not _TEMPORAL_FEATURE_NAME.fullmatch(name):
            raise ValueError(
                "each temporal feature name must match [a-z][a-z0-9_]{0,63}"
            )
    if len(set(names)) != len(names):
        raise ValueError("temporal_feature_names must be unique")

    # Compact JSON is an unambiguous canonical representation of the ordered
    # list (unlike delimiter joining when feature names evolve).
    canonical = json.dumps(
        names,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema_version": schema_version,
        "feature_names": names,
        "feature_order_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed so large JSONL files do not enter memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _validate_json_value(value: Any, *, path: str = "record") -> None:
    """Reject anything outside strict JSON and the derived-only field policy."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(path + " contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=path + "[" + str(index) + "]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(path + " contains a non-string JSON key")
            normal = _normalise_key(key)
            if normal in _PII_KEYS:
                raise ValueError(path + " contains forbidden direct-PII field " + repr(key))
            if normal in _IMAGERY_KEYS:
                raise ValueError(path + " contains forbidden image payload field " + repr(key))
            _validate_json_value(item, path=path + "." + key)
        return

    # This rejects bytes, bytearray, memoryview, numpy arrays/scalars, dataclass
    # Frames, Paths, tuples and arbitrary objects.  Callers must deliberately
    # reduce measurements to ordinary JSON lists and Python scalars first.
    raise TypeError(
        path + " must contain only JSON-safe dict/list/scalar values, got "
        + type(value).__module__ + "." + type(value).__qualname__
    )


def _validate_record_contract(stream: str, record: Mapping[str, Any]) -> None:
    """Enforce the reviewed, closed record vocabulary at every nesting level."""
    kind = record.get("kind")
    contract = _RECORD_FIELDS.get(stream, {}).get(kind)
    if contract is None:
        raise ValueError(stream + " record has an unsupported controlled kind")
    actual = frozenset(record)
    if actual != contract:
        missing = sorted(contract - actual)
        unknown = sorted(actual - contract)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ValueError(
            stream + " " + str(kind) + " record fields do not match contract: "
            + "; ".join(detail)
        )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key not in _ALLOWED_RECORD_KEYS:
                    raise ValueError(
                        path + " contains a field outside the derived-only contract: "
                        + repr(key)
                    )
                walk(item, path + "." + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + "[" + str(index) + "]")

    walk(record, "record")


class SessionRecorder:
    """Write one pseudonymous, derived-only collection session.

    Three append-only streams are created:

    * ``derived.jsonl`` for keypoints, confidence masks and scalar features;
    * ``telemetry.jsonl`` for availability/health transitions and heartbeats;
    * ``labels.jsonl`` for observer-entered phase markers.

    Every record is flushed immediately.  The manifest starts as ``incomplete``
    and changes only through :meth:`complete` or :meth:`abort`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        site_id: str,
        participant_id: str | None,
        code_hash: str,
        config_path: str | Path,
        calibration_path: str | Path,
        source: Mapping[str, Any],
        session_id: str | None = None,
        config_asset_id: str | None = None,
        calibration_asset_id: str | None = None,
        expected_config_sha256: str | None = None,
        expected_calibration_sha256: str | None = None,
        temporal_schema_version: str | None = None,
        temporal_feature_names: Sequence[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = _utc_now,
    ) -> None:
        self.session_id = _validate_id(
            session_id or new_pseudonymous_id("session"), "session"
        )
        self.site_id = _validate_id(site_id, "site")
        self.participant_id = (
            _validate_id(participant_id, "participant")
            if participant_id is not None
            else None
        )
        if not isinstance(code_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", code_hash):
            raise ValueError("code_hash must be a 7-64 character hexadecimal revision/hash")
        config_asset_id = _validate_asset_id(config_asset_id, "config")
        calibration_asset_id = _validate_asset_id(
            calibration_asset_id, "calibration"
        )
        temporal_schema = _temporal_schema_metadata(
            temporal_schema_version, temporal_feature_names
        )

        config_path = Path(config_path)
        calibration_path = Path(calibration_path)
        if not config_path.is_file():
            raise FileNotFoundError("config file not found: " + str(config_path))
        if not calibration_path.is_file():
            raise FileNotFoundError("calibration file not found: " + str(calibration_path))

        _validate_json_value(source, path="source")
        source_copy = dict(source)

        # Validate and hash everything before claiming the session directory. A
        # failed preflight must not leave what looks like an interrupted session.
        config_hash = sha256_file(config_path)
        calibration_hash = sha256_file(calibration_path)
        for label, expected, actual in (
            ("config", expected_config_sha256, config_hash),
            ("calibration", expected_calibration_sha256, calibration_hash),
        ):
            if expected is None:
                continue
            if not isinstance(expected, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", expected
            ):
                raise ValueError(label + " expected SHA-256 must be 64 hex characters")
            if actual != expected.lower():
                raise ValueError(
                    label + " changed after preflight; refusing mixed provenance"
                )

        self.root = Path(root)
        self.path = self.root / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                "session directory already exists; refusing to overwrite " + str(self.path)
            ) from exc

        self._clock = clock
        self._utc_now = utc_now
        self._started_tick = float(clock())
        self._last_t_rel = 0.0
        self._lock = threading.Lock()
        self._status = "incomplete"
        self._counts = {stream: 0 for stream in _STREAM_FILES}
        self._handles: dict[str, TextIO] = {}
        for stream, filename in _STREAM_FILES.items():
            self._handles[stream] = (self.path / filename).open(
                "x", encoding="utf-8", newline="\n"
            )

        inputs: dict[str, Any] = {
            "code_hash": code_hash.lower(),
            "config_sha256": config_hash,
            "calibration_sha256": calibration_hash,
        }
        if config_asset_id is not None:
            inputs["config_asset_id"] = config_asset_id
        if calibration_asset_id is not None:
            inputs["calibration_asset_id"] = calibration_asset_id

        self._manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "incomplete",
            "session_id": self.session_id,
            "site_id": self.site_id,
            "participant_id": self.participant_id,
            "purpose": "research_shadow_collection",
            "started_utc": utc_now(),
            "timebase": "monotonic_relative_seconds",
            "privacy": {
                "derived_only": True,
                "imagery_persisted": False,
                "dense_depth_persisted": False,
                "clinical_decisions_enabled": False,
            },
            "source": source_copy,
            # Hashes prove the exact inputs without persisting a workstation
            # path or a basename that may contain a patient/site identifier.
            "inputs": inputs,
            "streams": {
                stream: {
                    "file": filename,
                    "record_schema_version": RECORD_SCHEMA_VERSION,
                    "records": 0,
                    "sha256": None,
                }
                for stream, filename in _STREAM_FILES.items()
            },
        }
        if temporal_schema is not None:
            self._manifest["temporal_schema"] = temporal_schema
        self._write_manifest()

    @property
    def status(self) -> str:
        return self._status

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def elapsed_s(self) -> float:
        """Return finite non-negative time since this recorder's epoch.

        Reading elapsed time does not advance the cross-stream record-ordering
        watermark.  Callers can therefore use this clock for health events and
        then pass the value to a normal record write.
        """
        with self._lock:
            elapsed = float(self._clock()) - self._started_tick
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("recorder clock must yield finite non-negative elapsed time")
            return round(elapsed, 6)

    def _write_manifest(self) -> None:
        # Atomic replacement prevents a power loss during a status update from
        # leaving half a JSON document.  Text mode also keeps this module within
        # the repository's no-binary-image-writing privacy guard.
        target = self.manifest_path
        temporary = self.path / "manifest.json.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(self._manifest, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # On POSIX, syncing the directory makes the replacement itself durable.
        # Windows does not support opening a directory this way; the synced
        # temporary file still prevents a replaced manifest with partial bytes.
        if os.name != "nt":
            directory_fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def _relative_time(self, supplied: float | None) -> float:
        t_rel = (
            float(supplied)
            if supplied is not None
            else float(self._clock()) - self._started_tick
        )
        if not math.isfinite(t_rel) or t_rel < 0:
            raise ValueError("t_rel_s must be a finite, non-negative monotonic time")
        # One timebase across all streams makes labels and health intervals
        # directly joinable with derived observations.
        if t_rel + 1e-9 < self._last_t_rel:
            raise ValueError(
                "t_rel_s moved backwards: " + str(t_rel) + " < " + str(self._last_t_rel)
            )
        self._last_t_rel = max(self._last_t_rel, t_rel)
        return round(t_rel, 6)

    def _write_record(
        self, stream: str, record: Mapping[str, Any], *, t_rel_s: float | None = None
    ) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("record must be a JSON object (mapping)")
        if not record:
            raise ValueError("record must not be empty")
        overlap = _RESERVED_RECORD_KEYS.intersection(record)
        if overlap:
            raise ValueError("record uses reserved field(s): " + repr(sorted(overlap)))
        _validate_json_value(record)
        _validate_record_contract(stream, record)

        # Copy after validation so a caller cannot mutate the object while it is
        # being encoded under the lock.
        record_copy = dict(record)
        with self._lock:
            if self._status != "incomplete":
                raise RuntimeError("cannot write to a " + self._status + " session")
            t_rel = self._relative_time(t_rel_s)
            row = {
                "record_schema_version": RECORD_SCHEMA_VERSION,
                "t_rel_s": t_rel,
                "record": record_copy,
            }
            handle = self._handles[stream]
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            self._counts[stream] += 1
            # Labels, availability transitions and shadow events are sparse and
            # must survive an abrupt power loss once acknowledged. Derived rows
            # arrive at 10 Hz, so sync them once per second to bound loss without
            # stalling the camera on every observation.
            if stream != "derived" or self._counts[stream] % 10 == 0:
                os.fsync(handle.fileno())

    def write_derived(
        self, record: Mapping[str, Any], *, t_rel_s: float | None = None
    ) -> None:
        self._write_record("derived", record, t_rel_s=t_rel_s)

    def write_telemetry(
        self, record: Mapping[str, Any], *, t_rel_s: float | None = None
    ) -> None:
        self._write_record("telemetry", record, t_rel_s=t_rel_s)

    def write_label(
        self, record: Mapping[str, Any], *, t_rel_s: float | None = None
    ) -> None:
        self._write_record("labels", record, t_rel_s=t_rel_s)

    def _finish(self, status: str, *, abort_reason: str | None = None) -> Path:
        with self._lock:
            if self._status != "incomplete":
                raise RuntimeError("session is already " + self._status)
            if status not in ("complete", "aborted"):
                raise ValueError("final status must be complete or aborted")
            if status == "aborted":
                if not abort_reason or not _ABORT_CODE.fullmatch(abort_reason):
                    raise ValueError(
                        "abort_reason must be a controlled code such as OPERATOR_STOP"
                    )
            for handle in self._handles.values():
                if not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()

            duration = max(0.0, float(self._clock()) - self._started_tick)
            for stream, filename in _STREAM_FILES.items():
                path = self.path / filename
                self._manifest["streams"][stream].update(
                    {
                        "records": self._counts[stream],
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            self._status = status
            self._manifest["status"] = status
            self._manifest["ended_utc"] = self._utc_now()
            self._manifest["duration_s"] = round(duration, 6)
            self._manifest["counters"] = dict(self._counts)
            if abort_reason is not None:
                self._manifest["abort_reason"] = abort_reason
            self._write_manifest()
            return self.manifest_path

    def complete(self) -> Path:
        """Finalize hashes/counters and mark the session complete."""
        return self._finish("complete")

    def abort(self, reason: str = "OPERATOR_STOP") -> Path:
        """Finalize the usable prefix but mark it ineligible as a full run."""
        return self._finish("aborted", abort_reason=reason)

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._status != "incomplete":
            return
        if exc_type is None:
            self.complete()
        else:
            # Exception text can contain a patient name or local path; retain a
            # controlled class code only.  The console traceback is operational,
            # not part of the research dataset.
            safe_name = re.sub(r"[^A-Z0-9]", "_", exc_type.__name__.upper())
            self.abort(("EXCEPTION_" + safe_name)[:64])
