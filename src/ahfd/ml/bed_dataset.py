"""Load completed onsite sessions into causal bed-exit learning datasets.

The collection format stores already-computed ``temporal_summary`` rows.  This
loader never recomputes them from a completed clip and never interpolates from
the future.  It selects the latest prior observation on a global 10 Hz grid,
then joins controlled observer markers to create current-phase and anticipation
targets.

Identity comes only from the completed manifest's pseudonymous participant and
session fields.  Filenames, tracker IDs and association epochs are used solely
to join records within a session; none is treated as a person identity or an
evaluation group.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from array import array
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ahfd.ml.temporal import (
    TEMPORAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_ORDER_SHA256,
    TEMPORAL_SCHEMA_VERSION,
    SampleMetadata,
)


SAMPLE_HZ = 10.0
HORIZONS_S: tuple[int, ...] = (5, 10, 20)
EXIT_LABEL = "exit"
NO_EXIT_LABEL = "no_exit"

CONTROLLED_PHASES: frozenset[str] = frozenset(
    {
        "UNKNOWN",
        "RECLINED",
        "TORSO_RISING",
        "UPRIGHT_IN_BED",
        "SHIFTING_TO_EDGE",
        "EDGE_SITTING",
        "ATTEMPTING_STAND",
        "OUT_OF_BED",
    }
)
CONTROLLED_CONTEXTS: frozenset[str] = frozenset(
    {
        "RETURN_TO_RECLINE",
        "PAUSE",
        "FAST_TRANSITION",
        "SLIDE",
        "RAIL_CLIMB",
        "ASSISTED_TRANSFER",
        "STAFF_OCCLUSION",
        "BLANKET_OCCLUSION",
        "BED_ARTICULATION",
        "TRACK_ERROR",
        "OBSERVER_UNSURE",
    }
)
_CENSORING_CONTEXTS = frozenset(
    {
        "OBSERVER_UNSURE",
        "STAFF_OCCLUSION",
        "BLANKET_OCCLUSION",
        "TRACK_ERROR",
    }
)
_EPS = 1e-8
_SESSION_ID = re.compile(r"^ses_[0-9a-f]{16}$")
_PARTICIPANT_ID = re.compile(r"^sub_[0-9a-f]{16}$")
_SITE_ID = re.compile(r"^site_[0-9a-f]{8}$")
_CAMERA_ID = re.compile(r"^cam_[0-9a-f]{8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_HASH = re.compile(r"^[0-9a-f]{7,64}$")
_ASSET_IDS = {
    "config_asset_id": re.compile(r"^cfg_[0-9a-f]{16}$"),
    "calibration_asset_id": re.compile(r"^cal_[0-9a-f]{16}$"),
}
_STREAM_FILES = {
    "derived": "derived.jsonl",
    "telemetry": "telemetry.jsonl",
    "labels": "labels.jsonl",
}
_STREAM_ENTRY_FIELDS = frozenset(
    {"file", "record_schema_version", "records", "bytes", "sha256"}
)
_RECORD_FIELDS = {
    "derived": {
        "frame_observation": frozenset(
            {"kind", "frame_index", "source_t_s", "monitoring", "people", "events"}
        ),
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
                "trigger_t_s",
                "alert_t_s",
                "evidence",
                "zone",
            }
        ),
        "heartbeat": frozenset(
            {"kind", "frames_seen", "effective_fps", "disk_free_bytes", "monitoring"}
        ),
        "target_binding": frozenset({"kind", "track_id", "association_epoch"}),
    },
    "labels": {
        "phase_marker": frozenset({"kind", "track_ids", "associations", "phase"}),
        "context_marker": frozenset({"kind", "track_ids", "associations", "value"}),
    },
}
_REQUIRED_BINARY_MASK_FEATURES = frozenset(
    name for name in TEMPORAL_FEATURE_NAMES if "missing" in name
) | {"monitoring_valid_now"}
_OPTIONAL_BINARY_FEATURES = frozenset(
    {
        "cusum_onset__now",
        "cusum_armed__now",
        "shoulder_depth_available__now",
        "torso_depth_available__now",
        "bed_supported__now",
    }
)
_BINARY_MASK_FEATURES = (
    _REQUIRED_BINARY_MASK_FEATURES | _OPTIONAL_BINARY_FEATURES
)


@dataclass(frozen=True)
class SupervisedRows:
    """Rows, string targets and explicit grouping metadata for ``compare_grouped``."""

    rows: tuple[Mapping[str, float | None], ...] = ()
    labels: tuple[str, ...] = ()
    metadata: tuple[SampleMetadata, ...] = ()

    def __post_init__(self) -> None:
        if not (len(self.rows) == len(self.labels) == len(self.metadata)):
            raise ValueError("rows, labels, and metadata must have equal length")

    def __len__(self) -> int:
        return len(self.rows)

    def compare_args(self) -> tuple[tuple, tuple, tuple]:
        """Positional arguments accepted by :func:`compare_grouped`."""
        return self.rows, self.labels, self.metadata


@dataclass(frozen=True)
class BedDataset:
    """Phase and horizon-specific examples from one or more complete sessions."""

    session_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    phase: SupervisedRows
    anticipation: dict[int, SupervisedRows]
    sample_hz: float = SAMPLE_HZ

    def for_horizon(self, seconds: int) -> SupervisedRows:
        try:
            return self.anticipation[int(seconds)]
        except KeyError as exc:
            raise ValueError(
                "horizon must be one of " + repr(HORIZONS_S)
            ) from exc


@dataclass(frozen=True)
class _Observation:
    t: float
    track_id: int
    association_epoch: int
    available: bool
    values: Mapping[str, float | None]

    @property
    def association(self) -> tuple[int, int]:
        return self.track_id, self.association_epoch


_FEATURE_INDEX = {name: index for index, name in enumerate(TEMPORAL_FEATURE_NAMES)}


class _CompactTemporalRow(Mapping[str, float | None]):
    """Read-only feature mapping backed by one float32 vector.

    A Python dict with 929 repeated string keys can consume tens of kilobytes
    per 10 Hz row.  The manifest already fixes feature order, so retain one
    compact vector and expose the Mapping API expected by the comparator.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[float | None]) -> None:
        self._values = array(
            "f", (float("nan") if value is None else float(value) for value in values)
        )

    def __getitem__(self, name: str) -> float | None:
        try:
            value = self._values[_FEATURE_INDEX[name]]
        except KeyError:
            raise KeyError(name) from None
        return None if math.isnan(value) else float(value)

    def __iter__(self):
        return iter(TEMPORAL_FEATURE_NAMES)

    def __len__(self) -> int:
        return len(TEMPORAL_FEATURE_NAMES)


@dataclass(frozen=True)
class _PhaseMarker:
    t: float
    phase: str
    associations: tuple[tuple[int, int], ...] = ()
    legacy_track_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ContextMarker:
    t: float
    value: str
    associations: tuple[tuple[int, int], ...] = ()
    legacy_track_ids: tuple[int, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(name + " must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(name + " must be finite and non-negative")
    return result


def _exact_fields(value: Mapping, expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise ValueError(
        name
        + " fields do not match the collection contract: "
        + "; ".join(details)
    )


def _require_sha256(value, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(name + " must be a lowercase 64-hex SHA-256")
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read valid JSON from " + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError(str(path) + " must contain a JSON object")
    return value


def _stream_path(session_dir: Path, manifest: Mapping, stream: str) -> Path:
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping) or not isinstance(streams.get(stream), Mapping):
        raise ValueError("manifest has no " + stream + " stream")
    entry = streams[stream]
    filename = entry.get("file")
    if filename != _STREAM_FILES[stream]:
        raise ValueError(
            stream + " stream must use the controlled filename " + _STREAM_FILES[stream]
        )
    path = session_dir / filename
    if not path.is_file():
        raise ValueError("missing " + stream + " stream: " + str(path))
    expected_hash = entry.get("sha256")
    _require_sha256(expected_hash, stream + " stream sha256")
    if path.stat().st_size != entry.get("bytes"):
        raise ValueError(stream + " stream byte count does not match the manifest")
    if _sha256(path) != expected_hash:
        raise ValueError(stream + " stream SHA-256 does not match the manifest")
    return path


def _iter_stream(
    session_dir: Path,
    manifest: Mapping,
    stream: str,
    *,
    duration: float,
) -> Iterator[dict]:
    path = _stream_path(session_dir, manifest, stream)
    entry = manifest["streams"][stream]
    record_count = 0
    previous_t = -1.0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                try:
                    wrapper = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{stream} line {line_number} is not valid JSON"
                    ) from exc
                if not isinstance(wrapper, dict):
                    raise ValueError(f"{stream} line {line_number} must be an object")
                _exact_fields(
                    wrapper,
                    frozenset({"record_schema_version", "t_rel_s", "record"}),
                    f"{stream} line {line_number} wrapper",
                )
                if wrapper.get("record_schema_version") != 1:
                    raise ValueError(
                        f"{stream} line {line_number} has unsupported record schema"
                    )
                t = _finite_nonnegative(
                    wrapper.get("t_rel_s"), stream + " t_rel_s"
                )
                if t > duration + _EPS:
                    raise ValueError(
                        stream + " timestamp exceeds the manifest duration_s"
                    )
                if t + _EPS < previous_t:
                    raise ValueError(stream + " timestamps move backwards")
                previous_t = max(previous_t, t)
                record = wrapper.get("record")
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{stream} line {line_number} has no record object"
                    )
                kind = record.get("kind")
                expected_fields = _RECORD_FIELDS[stream].get(kind)
                if expected_fields is None:
                    raise ValueError(
                        f"{stream} line {line_number} has unsupported controlled record kind"
                    )
                _exact_fields(
                    record,
                    expected_fields,
                    f"{stream} line {line_number} {kind} record",
                )
                record_count += 1
                yield {"t": t, "record": record}
    except OSError as exc:
        raise ValueError("cannot read " + stream + " stream: " + str(path)) from exc

    declared = entry.get("records")
    if not isinstance(declared, int) or declared != record_count:
        raise ValueError(
            stream + " record count does not match manifest: "
            + repr(declared) + " != " + str(record_count)
        )


def _load_stream(
    session_dir: Path,
    manifest: Mapping,
    stream: str,
    *,
    duration: float,
) -> list[dict]:
    """Materialise small label streams; derived frames use ``_iter_stream``."""
    return list(_iter_stream(session_dir, manifest, stream, duration=duration))


def _verify_stream(
    session_dir: Path,
    manifest: Mapping,
    stream: str,
    *,
    duration: float,
) -> None:
    for _ in _iter_stream(session_dir, manifest, stream, duration=duration):
        pass


def _validate_manifest(session_dir: Path) -> tuple[dict, str, str, float]:
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("session has no manifest.json: " + str(session_dir))
    manifest = _load_json(manifest_path)
    _exact_fields(
        manifest,
        frozenset(
            {
                "schema",
                "schema_version",
                "status",
                "session_id",
                "site_id",
                "participant_id",
                "purpose",
                "started_utc",
                "ended_utc",
                "timebase",
                "duration_s",
                "privacy",
                "source",
                "inputs",
                "streams",
                "temporal_schema",
                "counters",
            }
        ),
        "manifest",
    )
    if (
        manifest.get("schema") != "ahfd.collection.session"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("unsupported onsite session manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("only sessions with manifest status 'complete' may be loaded")
    if manifest.get("purpose") != "research_shadow_collection":
        raise ValueError("manifest purpose must be research_shadow_collection")
    if manifest.get("timebase") != "monotonic_relative_seconds":
        raise ValueError("unsupported session timebase")
    for name in ("started_utc", "ended_utc"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ValueError("manifest " + name + " must be a nonempty timestamp")

    session_id = manifest.get("session_id")
    site_id = manifest.get("site_id")
    participant_id = manifest.get("participant_id")
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(
            "manifest session_id must be a pseudonymous code like "
            "ses_0123456789abcdef"
        )
    if session_dir.name != session_id:
        raise ValueError("session directory name does not match manifest session_id")
    if not isinstance(site_id, str) or not _SITE_ID.fullmatch(site_id):
        raise ValueError(
            "manifest site_id must be a pseudonymous code like site_0123abcd"
        )
    if not isinstance(participant_id, str) or not _PARTICIPANT_ID.fullmatch(
        participant_id
    ):
        raise ValueError(
            "manifest participant_id is required as a pseudonymous code like "
            "sub_0123456789abcdef for supervised learning; "
            "identity is never inferred from filenames or tracker IDs"
        )
    duration = _finite_nonnegative(manifest.get("duration_s"), "manifest duration_s")
    if duration <= 0:
        raise ValueError("manifest duration_s must be positive")

    privacy = manifest.get("privacy")
    expected_privacy = {
        "derived_only": True,
        "imagery_persisted": False,
        "dense_depth_persisted": False,
        "clinical_decisions_enabled": False,
    }
    if not isinstance(privacy, Mapping) or dict(privacy) != expected_privacy:
        raise ValueError(
            "manifest privacy must declare derived-only shadow data with no "
            "imagery/dense-depth persistence or clinical decisions"
        )

    source = manifest.get("source")
    source_fields = frozenset(
        {
            "kind",
            "width",
            "height",
            "fps",
            "depth_enabled",
            "max_laser",
            "emitter",
            "laser_power",
            "laser_power_max",
            "max_range_m",
            "spatial_magnitude",
            "pose_backend",
            "pose_model_size",
            "pose_runtime",
            "pose_device",
            "runtime_versions",
            "approval_sha256",
            "camera_id",
            "pose_model",
            "pose_model_sha256",
        }
    )
    if not isinstance(source, Mapping):
        raise ValueError("manifest source must be an object")
    _exact_fields(source, source_fields, "manifest source")
    if source.get("kind") != "intel_realsense_d435i":
        raise ValueError("manifest source kind must be intel_realsense_d435i")
    if source.get("depth_enabled") is not True:
        raise ValueError("manifest source must have depth_enabled true")
    if source.get("pose_backend") != "rtmo":
        raise ValueError("manifest source pose_backend must be rtmo")
    width = _positive_integer(source.get("width"), "manifest source width")
    height = _positive_integer(source.get("height"), "manifest source height")
    fps = _finite_nonnegative(source.get("fps"), "manifest source fps")
    if not (160 <= width <= 8192 and 120 <= height <= 8192):
        raise ValueError("manifest source camera dimensions are not plausible")
    if not (1.0 <= fps <= 240.0):
        raise ValueError("manifest source fps is not plausible")
    if not isinstance(source.get("camera_id"), str) or not _CAMERA_ID.fullmatch(
        source["camera_id"]
    ):
        raise ValueError("manifest source camera_id must be pseudonymous cam_ plus 8 hex")
    _require_sha256(source.get("approval_sha256"), "manifest source approval_sha256")
    _require_sha256(
        source.get("pose_model_sha256"), "manifest source pose_model_sha256"
    )
    if not isinstance(source.get("pose_model"), str) or not source[
        "pose_model"
    ].strip():
        raise ValueError("manifest source pose_model must be nonempty")
    for name in ("pose_model_size", "pose_runtime", "pose_device"):
        if not isinstance(source.get(name), str) or not source[name].strip():
            raise ValueError("manifest source " + name + " must be nonempty")
    if source.get("max_laser") is not True:
        raise ValueError("manifest source max_laser must be true")
    if source.get("emitter") is not True:
        raise ValueError("manifest source emitter must be true")
    laser_power = _finite_nonnegative(
        source.get("laser_power"), "manifest source laser_power"
    )
    laser_power_max = _finite_nonnegative(
        source.get("laser_power_max"), "manifest source laser_power_max"
    )
    laser_tolerance = max(0.01, abs(laser_power_max) * 1e-4)
    if laser_power_max <= 0.0 or abs(laser_power - laser_power_max) > laser_tolerance:
        raise ValueError(
            "manifest source laser power must confirm the positive sensor maximum"
        )
    max_range = _finite_nonnegative(
        source.get("max_range_m"), "manifest source max_range_m"
    )
    if not (0.1 <= max_range <= 20.0):
        raise ValueError("manifest source max_range_m is not plausible")
    spatial_magnitude = _positive_integer(
        source.get("spatial_magnitude"), "manifest source spatial_magnitude"
    )
    if spatial_magnitude > 5:
        raise ValueError("manifest source spatial_magnitude is not plausible")
    versions = source.get("runtime_versions")
    allowed_runtime_versions = {
        "numpy",
        "opencv-python",
        "rtmlib",
        "openvino",
        "onnxruntime",
    }
    if not isinstance(versions, Mapping) or not set(versions).issubset(
        allowed_runtime_versions
    ):
        raise ValueError("manifest source runtime_versions has unsupported entries")
    if any(
        not isinstance(value, str) or not value.strip() for value in versions.values()
    ):
        raise ValueError("manifest source runtime versions must be nonempty strings")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("manifest inputs must be an object")
    input_keys = set(inputs)
    required_inputs = {"code_hash", "config_sha256", "calibration_sha256"}
    allowed_inputs = required_inputs | set(_ASSET_IDS)
    if not required_inputs.issubset(input_keys) or not input_keys.issubset(
        allowed_inputs
    ):
        raise ValueError(
            "manifest inputs must contain only code_hash, config/calibration SHA-256, "
            "and optional opaque asset IDs"
        )
    if not isinstance(inputs.get("code_hash"), str) or not _CODE_HASH.fullmatch(
        inputs["code_hash"]
    ):
        raise ValueError("manifest inputs code_hash must be 7-64 lowercase hex")
    _require_sha256(inputs.get("config_sha256"), "manifest inputs config_sha256")
    _require_sha256(
        inputs.get("calibration_sha256"), "manifest inputs calibration_sha256"
    )
    for key, pattern in _ASSET_IDS.items():
        if key in inputs and (
            not isinstance(inputs[key], str) or not pattern.fullmatch(inputs[key])
        ):
            raise ValueError("manifest inputs " + key + " is not an opaque asset code")

    streams = manifest.get("streams")
    if not isinstance(streams, Mapping) or set(streams) != set(_STREAM_FILES):
        raise ValueError("manifest streams must declare derived, telemetry, and labels")
    counters = manifest.get("counters")
    if not isinstance(counters, Mapping) or set(counters) != set(_STREAM_FILES):
        raise ValueError("manifest counters must declare derived, telemetry, and labels")
    for stream, filename in _STREAM_FILES.items():
        entry = streams.get(stream)
        if not isinstance(entry, Mapping):
            raise ValueError("manifest " + stream + " stream must be an object")
        _exact_fields(entry, _STREAM_ENTRY_FIELDS, stream + " stream")
        if entry.get("file") != filename:
            raise ValueError(stream + " stream has the wrong controlled filename")
        if entry.get("record_schema_version") != 1:
            raise ValueError(stream + " stream record schema version is unsupported")
        records = _nonnegative_integer(
            entry.get("records"), stream + " stream records"
        )
        _nonnegative_integer(entry.get("bytes"), stream + " stream bytes")
        _require_sha256(entry.get("sha256"), stream + " stream sha256")
        if _nonnegative_integer(counters.get(stream), stream + " counter") != records:
            raise ValueError(stream + " counter does not match its stream record count")

    if not _compact_temporal_schema_is_valid(manifest):
        raise ValueError("manifest temporal_schema is required")
    return manifest, participant_id, session_id, duration


def _compact_temporal_schema_is_valid(manifest: Mapping) -> bool:
    """Validate the one session-wide ordering used by compact value vectors."""
    metadata = manifest.get("temporal_schema")
    if metadata is None:
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("manifest temporal_schema must be an object")
    _exact_fields(
        metadata,
        frozenset({"schema_version", "feature_names", "feature_order_sha256"}),
        "manifest temporal_schema",
    )
    if metadata.get("schema_version") != TEMPORAL_SCHEMA_VERSION:
        raise ValueError("manifest temporal schema version is incompatible")
    if metadata.get("feature_names") != list(TEMPORAL_FEATURE_NAMES):
        raise ValueError("manifest temporal feature order is incompatible")
    if metadata.get("feature_order_sha256") != TEMPORAL_FEATURE_ORDER_SHA256:
        raise ValueError("manifest temporal feature-order SHA-256 is incompatible")
    return True


def _normalise_summary(
    summary, line_context: str, *, compact_schema_valid: bool
) -> _CompactTemporalRow:
    if not isinstance(summary, Mapping):
        raise ValueError(line_context + " has no temporal_summary object")
    if summary.get("schema_version") != TEMPORAL_SCHEMA_VERSION:
        raise ValueError(line_context + " has incompatible temporal summary schema")
    raw = summary.get("values")
    if isinstance(raw, Mapping):
        expected = set(TEMPORAL_FEATURE_NAMES)
        actual = set(raw)
        unknown = actual - expected
        missing = expected - actual
        if unknown or missing:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError(
                line_context
                + " temporal summary feature keys do not exactly match the schema: "
                + "; ".join(details)
            )
        ordered_values = [raw[name] for name in TEMPORAL_FEATURE_NAMES]
    elif isinstance(raw, list):
        if not compact_schema_valid:
            raise ValueError(
                line_context
                + " compact temporal summary needs a validated manifest temporal_schema"
            )
        if len(raw) != len(TEMPORAL_FEATURE_NAMES):
            raise ValueError(
                line_context
                + " compact temporal summary must contain exactly "
                + str(len(TEMPORAL_FEATURE_NAMES))
                + " ordered values; got "
                + str(len(raw))
            )
        ordered_values = raw
    else:
        raise ValueError(
            line_context
            + " temporal summary values must be an object or ordered list"
        )
    clean: list[float | None] = []
    for name, value in zip(TEMPORAL_FEATURE_NAMES, ordered_values):
        if value is None:
            if name in _REQUIRED_BINARY_MASK_FEATURES:
                raise ValueError(
                    line_context + " binary mask " + name + " must be 0 or 1"
                )
            clean.append(None)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(line_context + " feature " + name + " is not numeric") from exc
        if not math.isfinite(number):
            raise ValueError(line_context + " feature " + name + " is non-finite")
        if name in _BINARY_MASK_FEATURES and number not in (0.0, 1.0):
            raise ValueError(
                line_context + " binary mask " + name + " must be 0 or 1"
            )
        clean.append(number)
    return _CompactTemporalRow(clean)


def _observations(
    records: Iterable[dict], *, compact_schema_valid: bool
) -> dict[tuple[int, int], list[_Observation]]:
    grouped: dict[tuple[int, int], list[_Observation]] = defaultdict(list)
    for frame_number, wrapper in enumerate(records, 1):
        t = wrapper["t"]
        record = wrapper["record"]
        if record.get("kind") != "frame_observation":
            raise ValueError("derived record must have kind 'frame_observation'")
        monitoring = record.get("monitoring")
        if not isinstance(monitoring, Mapping):
            raise ValueError("derived frame has no monitoring snapshot")
        status = monitoring.get("status")
        if status not in ("AVAILABLE", "DEGRADED", "UNAVAILABLE"):
            raise ValueError("derived frame has invalid monitoring status")
        people = record.get("people")
        if not isinstance(people, list):
            raise ValueError("derived frame people must be a list")
        seen: set[tuple[int, int]] = set()
        for person_number, person in enumerate(people, 1):
            if not isinstance(person, Mapping):
                raise ValueError("derived person must be an object")
            try:
                track_id = int(person["track_id"])
                epoch = int(person["association_epoch"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "derived person needs integer track_id and association_epoch"
                ) from exc
            key = (track_id, epoch)
            if key in seen:
                raise ValueError("duplicate association in one derived frame")
            seen.add(key)
            values = _normalise_summary(
                person.get("temporal_summary"),
                f"derived frame {frame_number} person {person_number}",
                compact_schema_valid=compact_schema_valid,
            )
            valid_now = values["monitoring_valid_now"] == 1.0
            grouped[key].append(
                _Observation(
                    t=t,
                    track_id=track_id,
                    association_epoch=epoch,
                    available=status == "AVAILABLE" and valid_now,
                    values=values,
                )
            )
    for values in grouped.values():
        values.sort(key=lambda item: item.t)
    return dict(grouped)


def _integer_identifier(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + " must be an integer")
    if value < 0:
        raise ValueError(name + " must be non-negative")
    return value


def _marker_targets(
    record: Mapping,
    marker_name: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Parse the same authoritative association scope for every marker kind."""
    raw_associations = record.get("associations")
    if not isinstance(raw_associations, list) or not raw_associations:
        raise ValueError(
            marker_name + " marker associations must be a nonempty list"
        )
    parsed: list[tuple[int, int]] = []
    for raw in raw_associations:
        if not isinstance(raw, Mapping):
            raise ValueError(marker_name + " marker association must be an object")
        _exact_fields(
            raw,
            frozenset({"track_id", "association_epoch"}),
            marker_name + " marker association",
        )
        track_id = _integer_identifier(
            raw["track_id"], marker_name + " marker association track_id"
        )
        epoch = _integer_identifier(
            raw["association_epoch"],
            marker_name + " marker association association_epoch",
        )
        parsed.append((track_id, epoch))
    associations = tuple(sorted(set(parsed)))
    if len(associations) != len(parsed):
        raise ValueError(marker_name + " marker contains a duplicate association")
    raw_ids = record.get("track_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(marker_name + " marker track_ids must be a nonempty list")
    parsed_ids = [
        _integer_identifier(value, marker_name + " marker track_id")
        for value in raw_ids
    ]
    track_ids = tuple(sorted(set(parsed_ids)))
    if len(track_ids) != len(parsed_ids):
        raise ValueError(marker_name + " marker contains a duplicate track_id")
    if set(track_ids) != {track_id for track_id, _ in associations}:
        raise ValueError(
            marker_name + " marker track_ids must match its explicit associations"
        )
    return associations, ()


def _label_markers(
    records: Sequence[dict],
) -> tuple[list[_PhaseMarker], list[_ContextMarker]]:
    phases: list[_PhaseMarker] = []
    contexts: list[_ContextMarker] = []
    for wrapper in records:
        record = wrapper["record"]
        kind = record.get("kind")
        if kind == "phase_marker":
            phase = record.get("phase")
            if not isinstance(phase, str) or phase not in CONTROLLED_PHASES:
                raise ValueError(
                    "unrecognised controlled phase marker: " + repr(phase)
                )
            associations, track_ids = _marker_targets(record, "phase")
            phases.append(
                _PhaseMarker(
                    t=wrapper["t"],
                    phase=str(phase),
                    associations=associations,
                    legacy_track_ids=track_ids,
                )
            )
        elif kind == "context_marker":
            value = record.get("value")
            if not isinstance(value, str) or value not in CONTROLLED_CONTEXTS:
                raise ValueError(
                    "unrecognised controlled context marker: " + repr(value)
                )
            associations, track_ids = _marker_targets(record, "context")
            contexts.append(
                _ContextMarker(
                    t=wrapper["t"],
                    value=str(value),
                    associations=associations,
                    legacy_track_ids=track_ids,
                )
            )
        else:
            raise ValueError(
                "label record kind must be phase_marker or context_marker"
            )
    return phases, contexts


def _assign_markers(
    observations: Mapping[tuple[int, int], Sequence[_Observation]],
    markers: Sequence[_PhaseMarker] | Sequence[_ContextMarker],
    *,
    marker_name: str,
) -> dict[tuple[int, int], list]:
    intervals: dict[int, list[tuple[float, float, tuple[int, int]]]] = defaultdict(list)
    for key, values in observations.items():
        intervals[key[0]].append((values[0].t, values[-1].t, key))

    assigned: dict[tuple[int, int], list] = defaultdict(list)
    for marker in markers:
        if marker.associations:
            for key in marker.associations:
                if key not in observations:
                    raise ValueError(
                        marker_name
                        + " marker targets an association absent from derived data: "
                        + repr(key)
                    )
                assigned[key].append(marker)
            continue

        for track_id in marker.legacy_track_ids:
            matches = [
                key
                for start, end, key in intervals.get(track_id, ())
                if start - _EPS <= marker.t <= end + _EPS
            ]
            if len(matches) != 1:
                raise ValueError(
                    marker_name
                    + " marker cannot be joined unambiguously to track "
                    + str(track_id) + " at t=" + str(marker.t)
                )
            assigned[matches[0]].append(marker)
    for values in assigned.values():
        values.sort(key=lambda item: item.t)
    return dict(assigned)


def _post_exit_intervals(markers: Sequence[_PhaseMarker], duration: float):
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for marker in markers:
        if marker.phase == "OUT_OF_BED" and start is None:
            start = marker.t
        elif marker.phase == "RECLINED" and start is not None:
            intervals.append((start, marker.t))
            start = None
    if start is not None:
        intervals.append((start, duration + 2 * _EPS))
    return intervals


def _unknown_intervals(markers: Sequence[_PhaseMarker], duration: float):
    """Intervals the observer explicitly marked unknown, independent of tracking."""
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for marker in markers:
        if marker.phase == "UNKNOWN" and start is None:
            start = marker.t
        elif marker.phase != "UNKNOWN" and start is not None:
            intervals.append((start, marker.t))
            start = None
    if start is not None:
        intervals.append((start, duration + 2 * _EPS))
    return intervals


def _context_censor_intervals(
    markers: Sequence[_ContextMarker], duration: float
) -> list[tuple[float, float]]:
    """Build per-value toggle intervals for target-visibility uncertainty."""
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for marker in markers:
        if marker.value not in _CENSORING_CONTEXTS:
            continue
        start = starts.pop(marker.value, None)
        if start is None:
            starts[marker.value] = marker.t
        else:
            intervals.append((start, marker.t))
    intervals.extend((start, duration + 2 * _EPS) for start in starts.values())
    return sorted(intervals)


def _inside_intervals(t: float, intervals: Sequence[tuple[float, float]]) -> bool:
    return any(start - _EPS <= t < end - _EPS for start, end in intervals)


def _future_overlaps_intervals(
    start: float,
    end: float,
    intervals: Sequence[tuple[float, float]],
) -> bool:
    """Return whether ``(start, end]`` intersects any half-open interval."""
    return any(
        interval_end > start + _EPS and interval_start <= end + _EPS
        for interval_start, interval_end in intervals
    )


def _index_epoch_times(
    observations: Mapping[tuple[int, int], Sequence[_Observation]],
) -> dict[int, dict[int, tuple[float, ...]]]:
    """Build one reusable time index for reassociation checks.

    Track IDs can be recycled, so every anticipation horizon must reject an
    interval in which another association epoch appears.  Building the other
    epoch's timestamp list inside every row/horizon check makes long sessions
    quadratic; build the immutable per-track/per-epoch index once instead.
    """
    indexed: dict[int, dict[int, tuple[float, ...]]] = defaultdict(dict)
    for (track_id, epoch), rows in observations.items():
        indexed[track_id][epoch] = tuple(item.t for item in rows)
    return {track_id: dict(epochs) for track_id, epochs in indexed.items()}


def _future_interval_is_observed(
    association: tuple[int, int],
    source_rows: Sequence[_Observation],
    source_times: Sequence[float],
    association_markers: Sequence[_PhaseMarker],
    association_marker_times: Sequence[float],
    epoch_times_by_track: Mapping[int, Mapping[int, Sequence[float]]],
    context_censor: Sequence[tuple[float, float]],
    *,
    start: float,
    end: float,
    max_hold_s: float,
) -> bool:
    """Whether ``(start, end]`` has a trustworthy target for one association.

    Target construction is deliberately stricter than row construction.  A
    negative label is safe only when the *same* association remains observable
    and labelled for the entire horizon; a positive needs that evidence through
    the known exit. A later frame cannot repair an invalid frame or a source
    gap that already exceeded the allowed causal hold.
    """
    source_index = bisect.bisect_right(source_times, start + _EPS) - 1
    if source_index < 0:
        return False
    previous = source_rows[source_index]
    if not previous.available or start - previous.t > max_hold_s + _EPS:
        return False

    # Bound both access and work to the evidence interval.  Slicing from the
    # current row to the end copied the entire remaining session for every
    # row/horizon even though the loop stopped after at most 20 seconds.
    evidence_stop = bisect.bisect_right(source_times, end + _EPS)
    for row_index in range(source_index + 1, evidence_stop):
        item = source_rows[row_index]
        if item.t <= start + _EPS:
            # Equal-time duplicate records are resolved by the latest record,
            # matching the causal row join.
            previous = item
            if not item.available:
                return False
            continue
        if item.t - previous.t > max_hold_s + _EPS or not item.available:
            return False
        previous = item
    if end - previous.t > max_hold_s + _EPS:
        return False

    if _future_overlaps_intervals(start, end, context_censor):
        return False

    # The phase at ``start`` was checked by the caller.  Any explicit UNKNOWN
    # transition inside the future interval makes the target censored even if
    # a known label resumes before the endpoint.
    first_marker = bisect.bisect_right(association_marker_times, start + _EPS)
    marker_stop = bisect.bisect_right(association_marker_times, end + _EPS)
    for marker_index in range(first_marker, marker_stop):
        marker = association_markers[marker_index]
        if marker.phase == "UNKNOWN":
            return False

    # Track numbers may be reused.  The appearance of another epoch for the
    # same track is reassociation, not evidence that the original person stayed
    # observable through the horizon.
    track_id, epoch = association
    for other_epoch, other_times in epoch_times_by_track.get(track_id, {}).items():
        if other_epoch == epoch:
            continue
        index = bisect.bisect_right(other_times, start + _EPS)
        if index < len(other_times) and other_times[index] <= end + _EPS:
            return False
    return True


def _metadata(participant: str, session: str, t: float) -> SampleMetadata:
    return SampleMetadata(
        subject_id=participant,
        session_id=session,
        clip_id=session,
        t=round(t, 6),
    )


def load_bed_session(
    session_dir: str | Path,
    *,
    max_hold_s: float = 0.20,
) -> BedDataset:
    """Load one completed session and produce phase/anticipation examples.

    Sampling is zero-order hold from the latest *past* source observation onto
    the global 10 Hz grid.  If that observation is older than ``max_hold_s``,
    the grid point is omitted rather than filled across a monitoring gap.
    """
    if not math.isfinite(max_hold_s) or max_hold_s <= 0:
        raise ValueError("max_hold_s must be finite and positive")
    path = Path(session_dir)
    manifest, participant, session, duration = _validate_manifest(path)
    # Telemetry carries monitoring gaps, association resets and the immediate
    # copy of each shadow event.  Labels/derived rows embed the state needed by
    # this loader, but a corrupted telemetry stream makes the completed session
    # unverifiable and must fail the dataset gate rather than be ignored.
    _verify_stream(path, manifest, "telemetry", duration=duration)
    labels = _load_stream(path, manifest, "labels", duration=duration)
    observations = _observations(
        _iter_stream(path, manifest, "derived", duration=duration),
        compact_schema_valid=_compact_temporal_schema_is_valid(manifest),
    )
    epoch_times_by_track = _index_epoch_times(observations)
    markers, context_markers = _label_markers(labels)
    assigned = _assign_markers(observations, markers, marker_name="phase")
    assigned_context = _assign_markers(
        observations, context_markers, marker_name="context"
    )
    if any(marker.value == "BED_ARTICULATION" for marker in context_markers):
        raise ValueError(
            "BED_ARTICULATION context invalidates the session for bed-exit learning"
        )

    phase_rows: list[Mapping[str, float | None]] = []
    phase_labels: list[str] = []
    phase_meta: list[SampleMetadata] = []
    horizon_rows: dict[int, list[Mapping[str, float | None]]] = {
        horizon: [] for horizon in HORIZONS_S
    }
    horizon_labels: dict[int, list[str]] = {horizon: [] for horizon in HORIZONS_S}
    horizon_meta: dict[int, list[SampleMetadata]] = {
        horizon: [] for horizon in HORIZONS_S
    }

    for association, source_rows in sorted(observations.items()):
        association_markers = assigned.get(association, [])
        if not association_markers:
            # A new/reassociated track must be explicitly relabelled.  Carrying
            # a phase across epochs could assign one person another's state.
            continue
        exit_times = [
            marker.t
            for marker in association_markers
            if marker.phase == "OUT_OF_BED"
        ]
        post_exit = _post_exit_intervals(association_markers, duration)
        unknown = _unknown_intervals(association_markers, duration)
        context_censor = _context_censor_intervals(
            assigned_context.get(association, []), duration
        )
        source_times = [item.t for item in source_rows]
        marker_times = [item.t for item in association_markers]
        first_tick = math.ceil((source_times[0] - _EPS) * SAMPLE_HZ)
        last_tick = math.floor((source_times[-1] + _EPS) * SAMPLE_HZ)

        for tick in range(first_tick, last_tick + 1):
            t = tick / SAMPLE_HZ
            source_index = bisect.bisect_right(source_times, t + _EPS) - 1
            if source_index < 0:
                continue
            source = source_rows[source_index]
            if t - source.t > max_hold_s + _EPS or not source.available:
                continue

            marker_index = bisect.bisect_right(marker_times, t + _EPS) - 1
            if marker_index < 0:
                continue
            phase = association_markers[marker_index].phase
            in_post_exit = _inside_intervals(t, post_exit)
            if phase == "UNKNOWN" or _inside_intervals(t, unknown):
                continue
            if _inside_intervals(t, context_censor):
                continue

            # Compact rows are immutable mappings. Reuse the same feature
            # object across phase/horizon tasks instead of copying 929 fields
            # up to four times for one timestamp.
            row = source.values
            meta = _metadata(participant, session, t)
            # OUT_OF_BED is a real observable phase for the phase classifier.
            # A contradictory later marker does not reopen a post-exit episode;
            # only an explicit RECLINED marker does that.
            if phase == "OUT_OF_BED" or not in_post_exit:
                phase_rows.append(row)
                phase_labels.append(phase)
                phase_meta.append(meta)

            # Anticipation is undefined once the target is already out of bed,
            # and remains suppressed throughout a post-exit interval.
            if phase == "OUT_OF_BED" or in_post_exit:
                continue

            next_index = bisect.bisect_right(exit_times, t + _EPS)
            next_exit = exit_times[next_index] if next_index < len(exit_times) else None
            for horizon in HORIZONS_S:
                target_t = t + horizon
                positive = next_exit is not None and next_exit <= target_t + _EPS
                # Positives need trustworthy evidence only through the known
                # exit. Negatives need the entire horizon; an early session or
                # association end is censoring, not evidence of no exit.
                evidence_end = next_exit if positive else target_t
                if not positive and target_t > duration + _EPS:
                    continue
                if not _future_interval_is_observed(
                    association,
                    source_rows,
                    source_times,
                    association_markers,
                    marker_times,
                    epoch_times_by_track,
                    context_censor,
                    start=t,
                    end=evidence_end,
                    max_hold_s=max_hold_s,
                ):
                    continue
                horizon_rows[horizon].append(row)
                horizon_labels[horizon].append(EXIT_LABEL if positive else NO_EXIT_LABEL)
                horizon_meta[horizon].append(meta)

    phase_set = SupervisedRows(
        rows=tuple(phase_rows),
        labels=tuple(phase_labels),
        metadata=tuple(phase_meta),
    )
    anticipation = {
        horizon: SupervisedRows(
            rows=tuple(horizon_rows[horizon]),
            labels=tuple(horizon_labels[horizon]),
            metadata=tuple(horizon_meta[horizon]),
        )
        for horizon in HORIZONS_S
    }
    return BedDataset(
        session_ids=(session,),
        participant_ids=(participant,),
        phase=phase_set,
        anticipation=anticipation,
    )


def _combine(parts: Sequence[SupervisedRows]) -> SupervisedRows:
    return SupervisedRows(
        rows=tuple(row for part in parts for row in part.rows),
        labels=tuple(label for part in parts for label in part.labels),
        metadata=tuple(item for part in parts for item in part.metadata),
    )


def load_bed_sessions(
    session_dirs: Iterable[str | Path],
    *,
    max_hold_s: float = 0.20,
) -> BedDataset:
    """Load and concatenate sessions without weakening their explicit groups."""
    datasets = [load_bed_session(path, max_hold_s=max_hold_s) for path in session_dirs]
    if not datasets:
        raise ValueError("at least one session directory is required")
    session_ids = tuple(session for data in datasets for session in data.session_ids)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("duplicate session_id in dataset input")
    participants = tuple(
        sorted({participant for data in datasets for participant in data.participant_ids})
    )
    return BedDataset(
        session_ids=session_ids,
        participant_ids=participants,
        phase=_combine([data.phase for data in datasets]),
        anticipation={
            horizon: _combine([data.anticipation[horizon] for data in datasets])
            for horizon in HORIZONS_S
        },
    )


__all__ = [
    "SAMPLE_HZ",
    "HORIZONS_S",
    "EXIT_LABEL",
    "NO_EXIT_LABEL",
    "CONTROLLED_PHASES",
    "CONTROLLED_CONTEXTS",
    "SupervisedRows",
    "BedDataset",
    "load_bed_session",
    "load_bed_sessions",
]
