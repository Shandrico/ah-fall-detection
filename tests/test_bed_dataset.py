"""Offline joining, censoring and causality tests for onsite bed datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ahfd.ml.bed_dataset as bed_dataset_module
from ahfd.collect import SessionRecorder
from ahfd.ml.bed_dataset import (
    EXIT_LABEL,
    HORIZONS_S,
    NO_EXIT_LABEL,
    SAMPLE_HZ,
    load_bed_session,
    load_bed_sessions,
)
from ahfd.ml.temporal import (
    TEMPORAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_ORDER_SHA256,
    TEMPORAL_SCHEMA_VERSION,
)


def wrapper(t, record):
    return {
        "record_schema_version": 1,
        "t_rel_s": round(float(t), 6),
        "record": record,
    }


def person(track_id, epoch, t, *, valid=True, feature_value=None):
    values = {
        name: (1.0 if "missing" in name else None)
        for name in TEMPORAL_FEATURE_NAMES
    }
    values.update(
        {
            "monitoring_valid_now": 1.0 if valid else 0.0,
            "h_torso__now": (
                round(t if feature_value is None else feature_value, 5)
                if valid
                else None
            ),
            "h_torso__missing_now": 0.0 if valid else 1.0,
        }
    )
    return {
        "track_id": track_id,
        "association_epoch": epoch,
        "temporal_summary": {
            "schema_version": TEMPORAL_SCHEMA_VERSION,
            "values": values,
        },
    }


def frame(t, people, *, status="AVAILABLE"):
    return wrapper(
        t,
        {
            "kind": "frame_observation",
            "frame_index": int(round(t * 20)),
            "source_t_s": float(t),
            "monitoring": {"status": status, "reasons": [], "details": {}},
            "people": people,
            "events": [],
        },
    )


def phase(t, value, track_ids=(1,)):
    ids = list(track_ids)
    return wrapper(
        t,
        {
            "kind": "phase_marker",
            "phase": value,
            "track_ids": ids,
            "associations": [
                {"track_id": track_id, "association_epoch": 1}
                for track_id in ids
            ],
        },
    )


def phase_associations(t, value, associations, *, track_ids=None):
    ids = (
        [track_id for track_id, _ in associations]
        if track_ids is None
        else list(track_ids)
    )
    record = {
        "kind": "phase_marker",
        "phase": value,
        "track_ids": ids,
        "associations": [
            {"track_id": track_id, "association_epoch": epoch}
            for track_id, epoch in associations
        ],
    }
    return wrapper(t, record)


def context(t, value, track_ids=(1,), *, associations=None):
    record = {"kind": "context_marker", "value": value}
    if associations is None:
        ids = list(track_ids)
        associations = [(track_id, 1) for track_id in ids]
    else:
        ids = (
            [track_id for track_id, _ in associations]
            if track_ids is None
            else list(track_ids)
        )
    record["track_ids"] = ids
    record["associations"] = [
        {"track_id": track_id, "association_epoch": epoch}
        for track_id, epoch in associations
    ]
    return wrapper(t, record)


def write_jsonl(path: Path, records):
    text = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records)
    path.write_text(text, encoding="utf-8")
    return {
        "file": path.name,
        "record_schema_version": 1,
        "records": len(records),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_session(
    root: Path,
    *,
    session_id="ses_0123456789abcdef",
    participant_id="sub_0123456789abcdef",
    duration=30.0,
    derived=None,
    labels=None,
    status="complete",
):
    directory = root / session_id
    directory.mkdir(parents=True)

    if derived is None:
        derived = []
        # Twenty source observations per second, deliberately offset from the
        # 10 Hz output grid. At grid t=.1 the causal source is .06, never .11.
        for index in range(600):
            t = round(0.01 + index * 0.05, 2)
            unavailable = abs(t - 8.96) < 1e-9
            derived.append(
                frame(
                    t,
                    [person(1, 1, t, valid=not unavailable)],
                    status="UNAVAILABLE" if unavailable else "AVAILABLE",
                )
            )
    if labels is None:
        labels = [
            phase(0.01, "RECLINED"),
            phase(2.01, "TORSO_RISING"),
            phase(4.01, "UPRIGHT_IN_BED"),
            phase(6.01, "SHIFTING_TO_EDGE"),
            phase(8.01, "EDGE_SITTING"),
            phase(10.01, "ATTEMPTING_STAND"),
            phase(12.01, "OUT_OF_BED"),
            # This marker must not make post-exit time trainable.
            phase(15.01, "SHIFTING_TO_EDGE"),
            phase(18.01, "RECLINED"),
            phase(23.01, "UNKNOWN"),
            phase(25.01, "RECLINED"),
        ]

    derived_entry = write_jsonl(directory / "derived.jsonl", derived)
    telemetry_entry = write_jsonl(directory / "telemetry.jsonl", [])
    labels_entry = write_jsonl(directory / "labels.jsonl", labels)
    manifest = {
        "schema": "ahfd.collection.session",
        "schema_version": 1,
        "status": status,
        "session_id": session_id,
        "site_id": "site_0123abcd",
        "participant_id": participant_id,
        "purpose": "research_shadow_collection",
        "started_utc": "2026-10-08T05:00:00Z",
        "ended_utc": "2026-10-08T05:00:30Z",
        "timebase": "monotonic_relative_seconds",
        "duration_s": duration,
        "privacy": {
            "derived_only": True,
            "imagery_persisted": False,
            "dense_depth_persisted": False,
            "clinical_decisions_enabled": False,
        },
        "source": {
            "kind": "intel_realsense_d435i",
            "width": 848,
            "height": 480,
            "fps": 30.0,
            "depth_enabled": True,
            "max_laser": True,
            "emitter": True,
            "laser_power": 360.0,
            "laser_power_max": 360.0,
            "max_range_m": 6.0,
            "spatial_magnitude": 2,
            "pose_backend": "rtmo",
            "pose_model_size": "s",
            "pose_runtime": "openvino",
            "pose_device": "gpu",
            "runtime_versions": {},
            "approval_sha256": "a" * 64,
            "camera_id": "cam_01234567",
            "pose_model": "rtmo-s",
            "pose_model_sha256": "b" * 64,
        },
        "inputs": {
            "code_hash": "c" * 40,
            "config_sha256": "d" * 64,
            "calibration_sha256": "e" * 64,
        },
        "temporal_schema": {
            "schema_version": TEMPORAL_SCHEMA_VERSION,
            "feature_names": list(TEMPORAL_FEATURE_NAMES),
            "feature_order_sha256": TEMPORAL_FEATURE_ORDER_SHA256,
        },
        "streams": {
            "derived": derived_entry,
            "telemetry": telemetry_entry,
            "labels": labels_entry,
        },
        "counters": {
            "derived": derived_entry["records"],
            "telemetry": telemetry_entry["records"],
            "labels": labels_entry["records"],
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def mutate_manifest(directory: Path, path, value):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def replace_stream(directory: Path, stream: str, records):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = write_jsonl(directory / (stream + ".jsonl"), records)
    manifest["streams"][stream] = entry
    manifest["counters"][stream] = entry["records"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def row_at(dataset, t):
    for row, metadata in zip(dataset.rows, dataset.metadata):
        if metadata.t == pytest.approx(t):
            return row
    raise AssertionError("no row at " + str(t))


def label_at(dataset, t):
    for label, metadata in zip(dataset.labels, dataset.metadata):
        if metadata.t == pytest.approx(t):
            return label
    raise AssertionError("no label at " + str(t))


def rows_at(dataset, t):
    return [
        (row, label)
        for row, label, metadata in zip(
            dataset.rows, dataset.labels, dataset.metadata
        )
        if metadata.t == pytest.approx(t)
    ]


def test_loads_current_session_recorder_output(tmp_path):
    config = tmp_path / "onsite.yaml"
    calibration = tmp_path / "calibration.yaml"
    config.write_text("privacy:\n  allow_raw_capture: false\n", encoding="utf-8")
    calibration.write_text("camera_id: cam_01234567\n", encoding="utf-8")
    now = [100.0]
    compact_person = person(1, 1, 0.0)
    values = compact_person["temporal_summary"]["values"]
    compact_person["temporal_summary"]["values"] = [
        values[name] for name in TEMPORAL_FEATURE_NAMES
    ]
    recorder = SessionRecorder(
        tmp_path / "sessions",
        site_id="site_0123abcd",
        participant_id="sub_0123456789abcdef",
        session_id="ses_0123456789abcdef",
        code_hash="c" * 40,
        config_path=config,
        calibration_path=calibration,
        source={
            "kind": "intel_realsense_d435i",
            "width": 848,
            "height": 480,
            "fps": 30.0,
            "depth_enabled": True,
            "max_laser": True,
            "emitter": True,
            "laser_power": 360.0,
            "laser_power_max": 360.0,
            "max_range_m": 6.0,
            "spatial_magnitude": 2,
            "pose_backend": "rtmo",
            "pose_model_size": "s",
            "pose_runtime": "openvino",
            "pose_device": "gpu",
            "runtime_versions": {},
            "approval_sha256": "a" * 64,
            "camera_id": "cam_01234567",
            "pose_model": "rtmo-s",
            "pose_model_sha256": "b" * 64,
        },
        temporal_schema_version=TEMPORAL_SCHEMA_VERSION,
        temporal_feature_names=TEMPORAL_FEATURE_NAMES,
        clock=lambda: now[0],
        utc_now=lambda: "2026-10-08T05:00:00Z",
    )
    recorder.write_derived(frame(0.0, [compact_person])["record"], t_rel_s=0.0)
    recorder.write_label(phase(0.0, "RECLINED")["record"], t_rel_s=0.0)
    recorder.write_telemetry(
        {
            "kind": "shadow_event",
            "event_id": "evt_0123456789abcdef",
            "type": "bed_exit",
            "track_id": 1,
            "association_epoch": 1,
            "frame_index": 0,
            "source_t_s": 0.0,
            "severity": 1,
            "trigger_t_s": 0.0,
            "alert_t_s": 0.0,
            "evidence": {},
            "zone": "bed_a",
        },
        t_rel_s=0.0,
    )
    now[0] += 1.0
    recorder.complete()

    dataset = load_bed_session(recorder.path)
    assert label_at(dataset.phase, 0.0) == "RECLINED"


class TestLoadSession:
    def test_samples_latest_past_summary_on_a_fixed_10hz_grid(self, tmp_path):
        directory = make_session(tmp_path)
        dataset = load_bed_session(directory)

        assert dataset.sample_hz == SAMPLE_HZ == 10.0
        assert row_at(dataset.phase, 0.1)["h_torso__now"] == pytest.approx(0.06)
        assert set(row_at(dataset.phase, 0.1)) == set(TEMPORAL_FEATURE_NAMES)
        assert all(
            abs((metadata.t or 0.0) * 10 - round((metadata.t or 0.0) * 10)) < 1e-8
            for metadata in dataset.phase.metadata
        )

    def test_current_phase_is_causal_and_controlled(self, tmp_path):
        dataset = load_bed_session(make_session(tmp_path))
        assert label_at(dataset.phase, 2.0) == "RECLINED"
        assert label_at(dataset.phase, 2.1) == "TORSO_RISING"
        assert label_at(dataset.phase, 8.1) == "EDGE_SITTING"
        assert label_at(dataset.phase, 10.1) == "ATTEMPTING_STAND"

    def test_includes_observable_out_but_excludes_invalid_and_post_exit_rows(
        self, tmp_path
    ):
        dataset = load_bed_session(make_session(tmp_path))
        times = {metadata.t for metadata in dataset.phase.metadata}

        assert 9.0 not in times  # unavailable source observation, not normal
        assert label_at(dataset.phase, 12.1) == "OUT_OF_BED"
        assert label_at(dataset.phase, 15.0) == "OUT_OF_BED"
        # A contradictory phase does not reopen the completed episode.
        assert not any(15.1 <= t <= 18.0 for t in times)
        assert 18.1 in times  # explicit return to RECLINED re-opens the episode
        assert not any(23.1 <= t <= 25.0 for t in times)  # UNKNOWN interval
        assert 25.1 in times
        assert "OUT_OF_BED" in dataset.phase.labels
        assert "UNKNOWN" not in dataset.phase.labels
        assert not any(
            12.1 <= (metadata.t or 0.0) <= 18.0
            for metadata in dataset.for_horizon(5).metadata
        )

    def test_horizon_targets_and_tail_censoring_are_per_horizon(self, tmp_path):
        # Use an intentionally continuous recording here.  The default fixture
        # contains an outage and UNKNOWN interval, both of which correctly
        # censor any anticipation horizon that crosses them.
        derived = [
            frame(i / 20, [person(1, 1, i / 20)]) for i in range(601)
        ]
        labels = [
            phase(0.0, "RECLINED"),
            phase(2.01, "TORSO_RISING"),
            phase(8.01, "EDGE_SITTING"),
            phase(10.01, "ATTEMPTING_STAND"),
            phase(12.01, "OUT_OF_BED"),
            phase(18.01, "RECLINED"),
        ]
        dataset = load_bed_session(
            make_session(tmp_path, derived=derived, labels=labels)
        )
        assert set(dataset.anticipation) == set(HORIZONS_S)

        five = dataset.for_horizon(5)
        assert label_at(five, 5.0) == NO_EXIT_LABEL
        assert label_at(five, 7.1) == EXIT_LABEL
        assert label_at(five, 12.0) == EXIT_LABEL

        for horizon in HORIZONS_S:
            target = dataset.for_horizon(horizon)
            assert all(
                label == EXIT_LABEL
                or (metadata.t or 0) + horizon <= 30.0 + 1e-8
                for label, metadata in zip(target.labels, target.metadata)
            )
        # A known exit remains positive even when its experimental horizon
        # extends beyond the completed session.
        assert label_at(dataset.for_horizon(20), 11.0) == EXIT_LABEL
        assert len(dataset.for_horizon(5)) > len(dataset.for_horizon(20))

    def test_positive_evidence_stops_at_exit_but_negative_needs_full_horizon(
        self, tmp_path
    ):
        positive_derived = [
            frame(
                i / 10,
                [person(1, 1, i / 10)],
                status="UNAVAILABLE" if i == 51 else "AVAILABLE",
            )
            for i in range(61)
        ]
        # Make the post-exit frame explicitly unusable. It must not erase a
        # positive whose evidence endpoint is the observed exit at t=5.
        positive_derived[51] = frame(
            5.1, [person(1, 1, 5.1, valid=False)], status="UNAVAILABLE"
        )
        positive = load_bed_session(
            make_session(
                tmp_path / "positive",
                session_id="ses_7777777777777777",
                duration=6.0,
                derived=positive_derived,
                labels=[phase(0.0, "RECLINED"), phase(5.0, "OUT_OF_BED")],
            )
        )
        assert label_at(positive.for_horizon(5), 4.0) == EXIT_LABEL
        assert 4.0 + 5 > 6.0  # retained despite the horizon crossing session end

        negative = load_bed_session(
            make_session(
                tmp_path / "negative",
                session_id="ses_8888888888888888",
                duration=6.0,
                derived=[
                    frame(i / 10, [person(1, 1, i / 10)])
                    for i in range(51)
                ],
                labels=[phase(0.0, "RECLINED")],
            )
        )
        negative_times = {
            item.t for item in negative.for_horizon(5).metadata
        }
        assert 0.0 in negative_times
        assert 1.0 not in negative_times

    def test_invalid_observation_before_exit_censors_positive(self, tmp_path):
        derived = [
            frame(
                i / 10,
                [person(1, 1, i / 10, valid=i != 49)],
                status="UNAVAILABLE" if i == 49 else "AVAILABLE",
            )
            for i in range(61)
        ]
        dataset = load_bed_session(
            make_session(
                tmp_path,
                duration=6.0,
                derived=derived,
                labels=[phase(0.0, "RECLINED"), phase(5.0, "OUT_OF_BED")],
            )
        )
        assert 4.0 not in {item.t for item in dataset.for_horizon(5).metadata}

    def test_future_outage_and_unknown_interval_censor_targets(self, tmp_path):
        dataset = load_bed_session(make_session(tmp_path))
        five_times = {metadata.t for metadata in dataset.for_horizon(5).metadata}

        assert 5.0 not in five_times  # future invalid observation at 8.96
        assert 20.0 not in five_times  # future UNKNOWN interval at 23.01
        assert 9.1 in five_times  # fully observed interval containing the exit

    def test_source_gap_beyond_hold_censors_the_whole_horizon(self, tmp_path):
        derived = [
            frame(i / 10, [person(1, 1, i / 10)])
            for i in range(101)
            if i not in (21, 22, 23)
        ]
        labels = [phase(0.0, "RECLINED"), phase(5.0, "OUT_OF_BED")]
        dataset = load_bed_session(
            make_session(
                tmp_path,
                duration=10.0,
                derived=derived,
                labels=labels,
            )
        )
        five_times = {metadata.t for metadata in dataset.for_horizon(5).metadata}
        assert 1.0 not in five_times  # 2.0 -> 2.4 exceeds the 0.2 s hold
        assert 2.4 in five_times

    def test_horizon_validation_never_slices_or_walks_the_session_tail(self):
        source_times = tuple(index / 10 for index in range(10_000))
        rows = [
            bed_dataset_module._Observation(
                t=t,
                track_id=1,
                association_epoch=1,
                available=True,
                values={},
            )
            for t in source_times
        ]

        class BoundedRows:
            def __len__(self):
                return len(rows)

            def __getitem__(self, index):
                if isinstance(index, slice):
                    raise AssertionError("horizon validation must not copy a tail slice")
                if index > 20:
                    raise AssertionError("horizon validation walked past its evidence end")
                return rows[index]

        markers = [
            bed_dataset_module._PhaseMarker(
                t=0.0,
                phase="RECLINED",
                associations=((1, 1),),
            )
        ]
        assert bed_dataset_module._future_interval_is_observed(
            (1, 1),
            BoundedRows(),
            source_times,
            markers,
            (0.0,),
            {1: {1: source_times}},
            (),
            start=0.0,
            end=1.0,
            max_hold_s=0.11,
        )
        assert not bed_dataset_module._future_interval_is_observed(
            (1, 1),
            BoundedRows(),
            source_times,
            markers,
            (0.0,),
            {1: {1: source_times, 2: (0.8,)}},
            (),
            start=0.0,
            end=1.0,
            max_hold_s=0.11,
        )

    def test_manifest_metadata_is_the_only_evaluation_identity(self, tmp_path):
        directory = make_session(
            tmp_path,
            session_id="ses_feedface1234abcd",
            participant_id="sub_deadbeefdeadbeef",
        )
        dataset = load_bed_session(directory)
        assert dataset.session_ids == ("ses_feedface1234abcd",)
        assert dataset.participant_ids == ("sub_deadbeefdeadbeef",)
        assert {
            (item.subject_id, item.session_id, item.clip_id)
            for item in dataset.phase.metadata
        } == {("sub_deadbeefdeadbeef", "ses_feedface1234abcd", "ses_feedface1234abcd")}


class TestSafetyAndAssociationBoundaries:
    def test_incomplete_session_and_missing_participant_are_rejected(self, tmp_path):
        incomplete = make_session(
            tmp_path / "one", session_id="ses_aaaaaaaaaaaaaaaa", status="incomplete"
        )
        with pytest.raises(ValueError, match="status 'complete'"):
            load_bed_session(incomplete)

        no_participant = make_session(
            tmp_path / "two", session_id="ses_bbbbbbbbbbbbbbbb", participant_id=None
        )
        with pytest.raises(ValueError, match="participant_id is required"):
            load_bed_session(no_participant)

    @pytest.mark.parametrize(
        "path,value,match",
        [
            (("purpose",), "clinical_alerting", "purpose"),
            (("privacy", "derived_only"), False, "privacy"),
            (("privacy", "imagery_persisted"), True, "privacy"),
            (("privacy", "dense_depth_persisted"), True, "privacy"),
            (("privacy", "clinical_decisions_enabled"), True, "privacy"),
            (("source", "kind"), "webcam", "source kind"),
            (("source", "depth_enabled"), False, "depth_enabled"),
            (("source", "pose_backend"), "rtmpose", "pose_backend"),
            (("source", "max_laser"), False, "max_laser"),
            (("source", "emitter"), False, "emitter"),
            (("source", "emitter"), "on", "emitter"),
            (("source", "laser_power"), float("nan"), "laser_power"),
            (("source", "laser_power_max"), 0.0, "laser power"),
            (("source", "laser_power"), 300.0, "laser power"),
            (("source", "width"), 0, "width"),
            (("source", "fps"), 0.0, "fps"),
            (("source", "camera_id"), "ward_bed_2", "camera_id"),
            (("source", "approval_sha256"), "a" * 63, "approval_sha256"),
            (("source", "pose_model_sha256"), "G" * 64, "pose_model_sha256"),
            (("inputs", "config_sha256"), "d" * 63, "config_sha256"),
            (("inputs", "calibration_sha256"), "E" * 64, "calibration_sha256"),
        ],
    )
    def test_manifest_rejects_nonstudy_or_unverifiable_provenance(
        self, tmp_path, path, value, match
    ):
        directory = make_session(
            tmp_path,
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED")],
        )
        mutate_manifest(directory, path, value)
        with pytest.raises(ValueError, match=match):
            load_bed_session(directory)

    @pytest.mark.parametrize(
        "mutation,match",
        [
            ("extra_manifest_field", "manifest fields"),
            ("unexpected_stream", "manifest streams"),
            ("stream_schema", "record schema"),
            ("counter_mismatch", "counter does not match"),
        ],
    )
    def test_manifest_and_stream_declarations_are_closed(
        self, tmp_path, mutation, match
    ):
        directory = make_session(
            tmp_path,
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED")],
        )
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "extra_manifest_field":
            manifest["free_text"] = "not part of the controlled format"
        elif mutation == "unexpected_stream":
            manifest["streams"]["video"] = dict(manifest["streams"]["derived"])
        elif mutation == "stream_schema":
            manifest["streams"]["derived"]["record_schema_version"] = 2
        else:
            manifest["counters"]["derived"] += 1
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=match):
            load_bed_session(directory)

    @pytest.mark.parametrize("stream", ["derived", "telemetry", "labels"])
    def test_stream_record_top_level_contract_is_closed(self, tmp_path, stream):
        directory = make_session(
            tmp_path,
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED")],
        )
        if stream == "derived":
            bad = frame(0.0, [person(1, 1, 0.0)])
        elif stream == "telemetry":
            bad = wrapper(
                0.0,
                {"kind": "target_binding", "track_id": 1, "association_epoch": 1},
            )
        else:
            bad = phase(0.0, "RECLINED")
        bad["record"]["uncontrolled"] = "field"
        replace_stream(directory, stream, [bad])
        with pytest.raises(ValueError, match="unknown uncontrolled"):
            load_bed_session(directory)

    def test_stream_hash_and_controlled_phase_are_validated(self, tmp_path):
        corrupt = make_session(tmp_path / "one", session_id="ses_cccccccccccccccc")
        with (corrupt / "derived.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with pytest.raises(ValueError, match="derived stream (byte count|SHA-256)"):
            load_bed_session(corrupt)

        bad_phase = make_session(
            tmp_path / "two",
            session_id="ses_dddddddddddddddd",
            labels=[phase(0.01, "SITTING")],
        )
        with pytest.raises(ValueError, match="controlled phase"):
            load_bed_session(bad_phase)

    def test_corrupted_telemetry_invalidates_completed_session(self, tmp_path):
        directory = make_session(
            tmp_path, session_id="ses_eeeeeeeeeeeeeeee"
        )
        with (directory / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")

        with pytest.raises(ValueError, match="telemetry stream (byte count|SHA-256)"):
            load_bed_session(directory)

    def test_reassociation_does_not_inherit_an_old_phase_label(self, tmp_path):
        derived = [frame(i / 10, [person(1, 1, i / 10)]) for i in range(21)]
        derived += [
            frame(3.0 + i / 10, [person(1, 2, 3.0 + i / 10)])
            for i in range(21)
        ]
        directory = make_session(
            tmp_path,
            duration=5.1,
            derived=derived,
            labels=[phase(0.0, "RECLINED")],
        )
        dataset = load_bed_session(directory)
        assert dataset.phase.metadata
        assert max(item.t for item in dataset.phase.metadata) == pytest.approx(2.0)

    def test_marker_without_explicit_associations_is_rejected(self, tmp_path):
        derived = [
            frame(
                i / 10,
                [person(1, 1, i / 10), person(1, 2, i / 10)],
            )
            for i in range(21)
        ]
        legacy_marker = wrapper(
            1.0,
            {"kind": "phase_marker", "phase": "RECLINED", "track_ids": [1]},
        )
        directory = make_session(
            tmp_path,
            duration=2.0,
            derived=derived,
            labels=[legacy_marker],
        )
        with pytest.raises(ValueError, match="missing associations"):
            load_bed_session(directory)

    def test_exact_summary_schema_and_binary_masks_are_required(self, tmp_path):
        missing_person = person(1, 1, 0.0)
        del missing_person["temporal_summary"]["values"]["track_age_s"]
        missing = make_session(
            tmp_path / "missing",
            session_id="ses_3333333333333333",
            duration=1.0,
            derived=[frame(0.0, [missing_person])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="exactly match.*missing track_age_s"):
            load_bed_session(missing)

        bad_mask_person = person(1, 1, 0.0)
        bad_mask_person["temporal_summary"]["values"][
            "h_head__missing_now"
        ] = 0.25
        bad_mask = make_session(
            tmp_path / "mask",
            session_id="ses_4444444444444444",
            duration=1.0,
            derived=[frame(0.0, [bad_mask_person])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="binary mask.*must be 0 or 1"):
            load_bed_session(bad_mask)

        bad_current_mask_person = person(1, 1, 0.0)
        bad_current_mask_person["temporal_summary"]["values"][
            "shoulder_depth_available__now"
        ] = 0.25
        bad_current_mask = make_session(
            tmp_path / "current-mask",
            session_id="ses_4545454545454545",
            duration=1.0,
            derived=[frame(0.0, [bad_current_mask_person])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="binary mask.*must be 0 or 1"):
            load_bed_session(bad_current_mask)

    def test_compact_summary_uses_fixed_feature_order_and_exact_length(
        self, tmp_path
    ):
        compact_person = person(1, 1, 0.0, feature_value=4.25)
        mapping = compact_person["temporal_summary"]["values"]
        compact_person["temporal_summary"]["values"] = [
            mapping[name] for name in TEMPORAL_FEATURE_NAMES
        ]
        compact = load_bed_session(
            make_session(
                tmp_path / "ok",
                session_id="ses_9999999999999999",
                duration=1.0,
                derived=[frame(0.0, [compact_person])],
                labels=[phase(0.0, "RECLINED")],
            )
        )
        assert row_at(compact.phase, 0.0)["h_torso__now"] == pytest.approx(4.25)

        short_person = person(1, 1, 0.0)
        mapping = short_person["temporal_summary"]["values"]
        short_person["temporal_summary"]["values"] = [
            mapping[name] for name in TEMPORAL_FEATURE_NAMES[:-1]
        ]
        short = make_session(
            tmp_path / "short",
            session_id="ses_abababababababab",
            duration=1.0,
            derived=[frame(0.0, [short_person])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="compact temporal summary.*exactly"):
            load_bed_session(short)

        bad_mask_person = person(1, 1, 0.0)
        mapping = bad_mask_person["temporal_summary"]["values"]
        compact_values = [mapping[name] for name in TEMPORAL_FEATURE_NAMES]
        compact_values[TEMPORAL_FEATURE_NAMES.index("monitoring_valid_now")] = 0.5
        bad_mask_person["temporal_summary"]["values"] = compact_values
        bad_mask = make_session(
            tmp_path / "compact_mask",
            session_id="ses_cdcdcdcdcdcdcdcd",
            duration=1.0,
            derived=[frame(0.0, [bad_mask_person])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="binary mask.*must be 0 or 1"):
            load_bed_session(bad_mask)

    def test_jsonl_streams_are_consumed_line_by_line(self, tmp_path, monkeypatch):
        directory = make_session(tmp_path)
        original = Path.read_text

        def guarded_read_text(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise AssertionError("JSONL must not be loaded with read_text")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", guarded_read_text)
        assert load_bed_session(directory).phase.rows

    def test_pseudonym_formats_and_manifest_duration_are_revalidated(self, tmp_path):
        bad_participant = make_session(
            tmp_path / "participant",
            session_id="ses_5555555555555555",
            participant_id="alice",
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="pseudonymous code"):
            load_bed_session(bad_participant)

        bad_session = make_session(
            tmp_path / "session",
            session_id="session_not_pseudonymous",
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="session_id must be a pseudonymous"):
            load_bed_session(bad_session)

        late_record = make_session(
            tmp_path / "late",
            session_id="ses_6666666666666666",
            duration=1.0,
            derived=[frame(1.01, [person(1, 1, 1.01)])],
            labels=[phase(0.0, "RECLINED")],
        )
        with pytest.raises(ValueError, match="exceeds the manifest duration"):
            load_bed_session(late_record)

    def test_empty_target_phase_marker_is_rejected(self, tmp_path):
        directory = make_session(
            tmp_path,
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "OUT_OF_BED", track_ids=())],
        )
        with pytest.raises(ValueError, match="nonempty list"):
            load_bed_session(directory)


class TestContextCensoring:
    @pytest.mark.parametrize(
        "value",
        [
            "OBSERVER_UNSURE",
            "STAFF_OCCLUSION",
            "BLANKET_OCCLUSION",
            "TRACK_ERROR",
        ],
    )
    def test_visibility_contexts_toggle_censor_intervals(self, tmp_path, value):
        derived = [
            frame(i / 10, [person(1, 1, i / 10)]) for i in range(101)
        ]
        dataset = load_bed_session(
            make_session(
                tmp_path,
                duration=10.0,
                derived=derived,
                labels=[
                    phase(0.0, "RECLINED"),
                    context(2.0, value),
                    context(4.0, value),
                ],
            )
        )
        phase_times = {item.t for item in dataset.phase.metadata}
        assert 1.9 in phase_times
        assert 2.0 not in phase_times
        assert 3.9 not in phase_times
        assert 4.0 in phase_times
        # Although the context toggles off before the endpoint, a negative
        # horizon crossing any censored interval is not trustworthy.
        assert 0.0 not in {item.t for item in dataset.for_horizon(5).metadata}

    def test_open_context_runs_to_session_end(self, tmp_path):
        derived = [
            frame(i / 10, [person(1, 1, i / 10)]) for i in range(101)
        ]
        dataset = load_bed_session(
            make_session(
                tmp_path,
                duration=10.0,
                derived=derived,
                labels=[
                    phase(0.0, "RECLINED"),
                    context(2.0, "OBSERVER_UNSURE"),
                ],
            )
        )
        phase_times = {item.t for item in dataset.phase.metadata}
        assert 1.9 in phase_times
        assert not any(t >= 2.0 for t in phase_times)

    @pytest.mark.parametrize(
        "value",
        [
            "RETURN_TO_RECLINE",
            "PAUSE",
            "FAST_TRANSITION",
            "SLIDE",
            "RAIL_CLIMB",
            "ASSISTED_TRANSFER",
        ],
    )
    def test_nonvisibility_context_remains_metadata_only(self, tmp_path, value):
        derived = [
            frame(i / 10, [person(1, 1, i / 10)]) for i in range(61)
        ]
        dataset = load_bed_session(
            make_session(
                tmp_path,
                duration=6.0,
                derived=derived,
                labels=[phase(0.0, "RECLINED"), context(2.0, value)],
            )
        )
        assert label_at(dataset.phase, 2.0) == "RECLINED"
        assert label_at(dataset.for_horizon(5), 0.0) == NO_EXIT_LABEL

    def test_positive_ignores_context_after_exit_but_not_before_exit(self, tmp_path):
        derived = [
            frame(i / 10, [person(1, 1, i / 10)]) for i in range(81)
        ]
        after = load_bed_session(
            make_session(
                tmp_path / "after",
                session_id="ses_1010101010101010",
                duration=8.0,
                derived=derived,
                labels=[
                    phase(0.0, "RECLINED"),
                    phase(5.0, "OUT_OF_BED"),
                    context(5.5, "TRACK_ERROR"),
                ],
            )
        )
        assert label_at(after.for_horizon(5), 4.0) == EXIT_LABEL

        before = load_bed_session(
            make_session(
                tmp_path / "before",
                session_id="ses_2020202020202020",
                duration=8.0,
                derived=derived,
                labels=[
                    phase(0.0, "RECLINED"),
                    context(4.5, "TRACK_ERROR"),
                    phase(5.0, "OUT_OF_BED"),
                ],
            )
        )
        assert 4.0 not in {item.t for item in before.for_horizon(5).metadata}

    def test_context_scope_isolated_by_exact_association(self, tmp_path):
        derived = [
            frame(
                i / 10,
                [
                    person(1, 1, i / 10, feature_value=10 + i / 10),
                    person(2, 7, i / 10, feature_value=20 + i / 10),
                ],
            )
            for i in range(51)
        ]
        labels = [
            phase_associations(0.0, "RECLINED", [(1, 1)]),
            phase_associations(0.0, "RECLINED", [(2, 7)]),
            context(1.0, "STAFF_OCCLUSION", track_ids=[2], associations=[(2, 7)]),
            context(3.0, "STAFF_OCCLUSION", track_ids=[2], associations=[(2, 7)]),
        ]
        dataset = load_bed_session(
            make_session(tmp_path, duration=5.0, derived=derived, labels=labels)
        )
        at_two = {
            round(row["h_torso__now"])
            for row, _ in rows_at(dataset.phase, 2.0)
        }
        assert at_two == {12}

    def test_context_marker_track_ids_must_match_associations(self, tmp_path):
        derived = [
            frame(i / 10, [person(1, 1, i / 10), person(1, 2, i / 10)])
            for i in range(21)
        ]
        directory = make_session(
            tmp_path,
            duration=2.0,
            derived=derived,
            labels=[
                phase_associations(0.0, "RECLINED", [(1, 1)]),
                phase_associations(0.0, "RECLINED", [(1, 2)]),
                context(1.0, "TRACK_ERROR", track_ids=[2], associations=[(1, 1)]),
            ],
        )
        with pytest.raises(ValueError, match="track_ids must match"):
            load_bed_session(directory)

    def test_bed_articulation_and_unknown_context_are_rejected(self, tmp_path):
        articulation = make_session(
            tmp_path / "articulation",
            session_id="ses_3030303030303030",
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED"), context(0.0, "BED_ARTICULATION")],
        )
        with pytest.raises(ValueError, match="BED_ARTICULATION.*invalidates"):
            load_bed_session(articulation)

        unknown = make_session(
            tmp_path / "unknown",
            session_id="ses_4040404040404040",
            duration=1.0,
            derived=[frame(0.0, [person(1, 1, 0.0)])],
            labels=[phase(0.0, "RECLINED"), context(0.0, "NOT_CONTROLLED")],
        )
        with pytest.raises(ValueError, match="controlled context"):
            load_bed_session(unknown)


def test_explicit_associations_isolate_people_labels_exits_and_censoring(tmp_path):
    derived = [
        frame(
            i / 10,
            [
                person(1, 1, i / 10, feature_value=10 + i / 10),
                person(2, 7, i / 10, feature_value=20 + i / 10),
            ],
        )
        for i in range(151)
    ]
    labels = [
        phase_associations(0.0, "RECLINED", [(1, 1)]),
        phase_associations(0.0, "RECLINED", [(2, 7)]),
        phase_associations(6.0, "OUT_OF_BED", [(2, 7)]),
        phase_associations(7.0, "SHIFTING_TO_EDGE", [(2, 7)]),
        phase_associations(10.0, "RECLINED", [(2, 7)]),
        phase_associations(11.0, "UNKNOWN", [(2, 7)]),
        phase_associations(12.0, "RECLINED", [(2, 7)]),
    ]
    dataset = load_bed_session(
        make_session(tmp_path, duration=15.0, derived=derived, labels=labels)
    )

    at_two = {
        round(row["h_torso__now"]): label
        for row, label in rows_at(dataset.for_horizon(5), 2.0)
    }
    assert at_two == {12: NO_EXIT_LABEL, 22: EXIT_LABEL}

    at_seven = {
        round(row["h_torso__now"]): label
        for row, label in rows_at(dataset.phase, 7.0)
    }
    assert at_seven == {17: "RECLINED"}  # other association remains usable
    at_unknown = {
        round(row["h_torso__now"]): label
        for row, label in rows_at(dataset.phase, 11.5)
    }
    assert at_unknown == {22: "RECLINED"}


def test_reassociation_censors_old_epoch_horizon_and_uses_new_epoch_exit(tmp_path):
    derived = [
        frame(i / 10, [person(1, 1, i / 10, feature_value=100 + i / 10)])
        for i in range(31)
    ]
    derived += [
        frame(
            3.1 + i / 10,
            [person(1, 2, 3.1 + i / 10, feature_value=200 + 3.1 + i / 10)],
        )
        for i in range(70)
    ]
    labels = [
        phase_associations(0.0, "RECLINED", [(1, 1)]),
        phase_associations(3.1, "RECLINED", [(1, 2)]),
        phase_associations(5.0, "OUT_OF_BED", [(1, 2)]),
    ]
    dataset = load_bed_session(
        make_session(tmp_path, duration=10.0, derived=derived, labels=labels)
    )

    old_values = {
        row["h_torso__now"]
        for row, _ in rows_at(dataset.for_horizon(5), 1.0)
    }
    assert not old_values
    new_rows = rows_at(dataset.for_horizon(5), 4.0)
    assert len(new_rows) == 1
    assert new_rows[0][0]["h_torso__now"] == pytest.approx(204.0)
    assert new_rows[0][1] == EXIT_LABEL


def test_combined_sessions_feed_compare_grouped_without_identity_inference(tmp_path):
    first = make_session(
        tmp_path / "first",
        session_id="ses_1111111111111111",
        participant_id="sub_1111111111111111",
    )
    second = make_session(
        tmp_path / "second",
        session_id="ses_2222222222222222",
        participant_id="sub_2222222222222222",
    )
    combined = load_bed_sessions([first, second])
    assert combined.session_ids == ("ses_1111111111111111", "ses_2222222222222222")
    assert combined.participant_ids == ("sub_1111111111111111", "sub_2222222222222222")

    pytest.importorskip("sklearn")
    from ahfd.ml.temporal import compare_grouped

    result = compare_grouped(
        *combined.for_horizon(5).compare_args(),
        group_by="subject",
        feature_names=("h_torso__now", "h_torso__missing_now"),
        min_samples_leaf=1,
    )
    assert len(result.folds) == 2
    assert all(set(fold.train_groups).isdisjoint(fold.test_groups) for fold in result.folds)
