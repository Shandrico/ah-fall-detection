"""The onsite session format is durable, pseudonymous and derived-only."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from ahfd.collect import SessionRecorder, new_pseudonymous_id, sha256_file
from ahfd.capture.base import Frame


class Clock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def inputs(tmp_path):
    config = tmp_path / "onsite.yaml"
    calibration = tmp_path / "camera.yaml"
    config.write_text("privacy:\n  allow_raw_capture: false\n", encoding="utf-8")
    calibration.write_text("camera_id: camera_a\n", encoding="utf-8")
    return config, calibration


def make_recorder(tmp_path, inputs, clock=None, **overrides):
    config, calibration = inputs
    args = {
        "site_id": "site_0123abcd",
        "participant_id": "sub_0123456789abcdef",
        "session_id": "ses_0123456789abcdef",
        "code_hash": "a1b2c3d4e5f6a7b8",
        "config_path": config,
        "calibration_path": calibration,
        "source": {
            "scheme": "rs",
            "camera_code": "cam_01",
            "width": 1920,
            "height": 1080,
        },
        "clock": clock or Clock(),
        "utc_now": lambda: "2026-10-08T05:00:00Z",
    }
    args.update(overrides)
    return SessionRecorder(tmp_path / "sessions", **args)


def load_manifest(recorder):
    return json.loads(recorder.manifest_path.read_text(encoding="utf-8"))


def frame_record(frame_index=1, *, people=None, events=None):
    return {
        "kind": "frame_observation",
        "frame_index": frame_index,
        "source_t_s": float(frame_index) / 10.0,
        "monitoring": {
            "status": "AVAILABLE",
            "reasons": [],
            "details": {},
            "last_frame_t_rel_s": float(frame_index) / 10.0,
        },
        "people": [] if people is None else people,
        "events": [] if events is None else events,
    }


def heartbeat_record():
    return {
        "kind": "heartbeat",
        "frames_seen": 1,
        "effective_fps": 10.0,
        "disk_free_bytes": 1_000_000_000,
        "monitoring": {
            "status": "AVAILABLE",
            "reasons": [],
            "details": {},
            "last_frame_t_rel_s": 0.1,
        },
    }


def phase_record():
    return {
        "kind": "phase_marker",
        "track_ids": [3],
        "associations": [{"track_id": 3, "association_epoch": 1}],
        "phase": "TORSO_RISING",
    }


def shadow_event_record():
    return {
        "event_id": "evt_0123456789abcdef",
        "type": "BED_EXIT",
        "track_id": 3,
        "association_epoch": 1,
        "frame_index": 10,
        "source_t_s": 1.0,
        "severity": 1,
        "zone": "bed_a",
        "trigger_t_s": 0.8,
        "alert_t_s": 1.0,
        "evidence": {"trigger": "out_of_bed"},
    }


class TestManifestLifecycle:
    def test_new_session_is_incomplete_and_privacy_explicit(self, tmp_path, inputs):
        rec = make_recorder(tmp_path, inputs)
        manifest = load_manifest(rec)
        assert manifest["schema_version"] == 1
        assert manifest["status"] == "incomplete"
        assert manifest["privacy"] == {
            "derived_only": True,
            "imagery_persisted": False,
            "dense_depth_persisted": False,
            "clinical_decisions_enabled": False,
        }
        assert manifest["timebase"] == "monotonic_relative_seconds"
        rec.abort("TEST_CLEANUP")

    def test_complete_writes_counts_hashes_and_duration(self, tmp_path, inputs):
        clock = Clock()
        rec = make_recorder(tmp_path, inputs, clock=clock)
        clock.advance(0.25)
        rec.write_derived(frame_record(7))
        clock.advance(0.1)
        rec.write_telemetry(heartbeat_record())
        clock.advance(0.15)
        rec.write_label(phase_record())
        clock.advance(1.0)
        rec.complete()

        manifest = load_manifest(rec)
        assert manifest["status"] == "complete"
        assert manifest["duration_s"] == pytest.approx(1.5)
        assert manifest["counters"] == {"derived": 1, "telemetry": 1, "labels": 1}
        for stream in ("derived", "telemetry", "labels"):
            entry = manifest["streams"][stream]
            path = rec.path / entry["file"]
            assert entry["records"] == 1
            assert entry["bytes"] == path.stat().st_size
            assert entry["sha256"] == sha256_file(path)
            assert len(entry["sha256"]) == 64

        assert manifest["inputs"]["config_sha256"] == hashlib.sha256(
            inputs[0].read_bytes()
        ).hexdigest()
        assert manifest["inputs"]["calibration_sha256"] == hashlib.sha256(
            inputs[1].read_bytes()
        ).hexdigest()
        assert set(manifest["inputs"]) == {
            "code_hash",
            "config_sha256",
            "calibration_sha256",
        }
        assert "config_file" not in manifest["inputs"]
        assert "calibration_file" not in manifest["inputs"]
        # A workstation path can expose a person's Windows account name.
        text = rec.manifest_path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text

    def test_optional_asset_ids_are_opaque_codes_not_filenames(self, tmp_path, inputs):
        rec = make_recorder(
            tmp_path,
            inputs,
            config_asset_id="cfg_0123456789abcdef",
            calibration_asset_id="cal_fedcba9876543210",
        )
        assert load_manifest(rec)["inputs"] == {
            "code_hash": "a1b2c3d4e5f6a7b8",
            "config_sha256": sha256_file(inputs[0]),
            "calibration_sha256": sha256_file(inputs[1]),
            "config_asset_id": "cfg_0123456789abcdef",
            "calibration_asset_id": "cal_fedcba9876543210",
        }
        rec.abort("TEST_CLEANUP")

    def test_expected_input_hashes_pin_preflight_bytes(self, tmp_path, inputs):
        config_hash = sha256_file(inputs[0])
        calibration_hash = sha256_file(inputs[1])
        rec = make_recorder(
            tmp_path / "good",
            inputs,
            expected_config_sha256=config_hash,
            expected_calibration_sha256=calibration_hash,
        )
        rec.abort("TEST_CLEANUP")

        with pytest.raises(ValueError, match="config changed after preflight"):
            make_recorder(
                tmp_path / "bad-config",
                inputs,
                expected_config_sha256="0" * 64,
                expected_calibration_sha256=calibration_hash,
            )
        with pytest.raises(ValueError, match="calibration changed after preflight"):
            make_recorder(
                tmp_path / "bad-calibration",
                inputs,
                expected_config_sha256=config_hash,
                expected_calibration_sha256="0" * 64,
            )

    def test_context_exception_marks_aborted_without_storing_exception_text(
        self, tmp_path, inputs
    ):
        with pytest.raises(RuntimeError, match="patient Alice"):
            with make_recorder(tmp_path, inputs) as rec:
                rec.write_derived(frame_record())
                raise RuntimeError("patient Alice was here")

        manifest = load_manifest(rec)
        assert manifest["status"] == "aborted"
        assert manifest["abort_reason"] == "EXCEPTION_RUNTIMEERROR"
        assert "Alice" not in rec.manifest_path.read_text(encoding="utf-8")
        assert manifest["streams"]["derived"]["records"] == 1

    def test_open_session_leaves_parseable_incomplete_manifest_and_flushed_prefix(
        self, tmp_path, inputs
    ):
        rec = make_recorder(tmp_path, inputs)
        rec.write_derived(frame_record(), t_rel_s=0.5)
        row = json.loads((rec.path / "derived.jsonl").read_text(encoding="utf-8"))
        assert row["record"]["frame_index"] == 1
        assert row["t_rel_s"] == 0.5
        assert load_manifest(rec)["status"] == "incomplete"
        rec.abort("TEST_CLEANUP")

    def test_shadow_event_zone_is_preserved_in_both_durable_copies(
        self, tmp_path, inputs
    ):
        rec = make_recorder(tmp_path, inputs)
        event = shadow_event_record()
        rec.write_telemetry({"kind": "shadow_event", **event}, t_rel_s=1.0)
        rec.write_derived(frame_record(10, events=[event]), t_rel_s=1.0)
        telemetry = json.loads(
            (rec.path / "telemetry.jsonl").read_text(encoding="utf-8")
        )
        derived = json.loads(
            (rec.path / "derived.jsonl").read_text(encoding="utf-8")
        )
        assert telemetry["record"]["zone"] == "bed_a"
        assert derived["record"]["events"][0]["zone"] == "bed_a"
        rec.abort("TEST_CLEANUP")


class TestSafetyBoundary:
    @pytest.mark.parametrize(
        "kind,value",
        [
            ("session", "Alice"),
            ("session", "ses_20261008"),
            ("session", "ses_0123456789abcdeg"),
            ("session", "ses_0123456789ABCDEF"),
            ("site", "ward_6"),
            ("site", "site_a1b2"),
            ("participant", "sub_bob"),
            ("participant", "sub_01234567"),
            ("participant", "MRN-12345"),
        ],
    )
    def test_ids_must_use_pseudonymous_code_shape(
        self, tmp_path, inputs, kind, value
    ):
        changes = {kind + "_id": value}
        with pytest.raises(ValueError, match="pseudonym"):
            make_recorder(tmp_path, inputs, **changes)

    def test_generated_ids_are_nonsemantic_codes(self):
        assert re.fullmatch(r"ses_[0-9a-f]{16}", new_pseudonymous_id("session"))
        assert re.fullmatch(r"site_[0-9a-f]{8}", new_pseudonymous_id("site"))
        assert re.fullmatch(r"sub_[0-9a-f]{16}", new_pseudonymous_id("participant"))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"config_asset_id": "onsite.yaml"},
            {"config_asset_id": "cfg_ward6"},
            {"calibration_asset_id": "cal_patient_alice"},
            {"calibration_asset_id": "cal_0123456789ABCDE"},
        ],
    )
    def test_asset_ids_reject_filenames_and_free_text(self, tmp_path, inputs, overrides):
        with pytest.raises(ValueError, match="opaque code"):
            make_recorder(tmp_path, inputs, **overrides)

    def test_existing_session_is_never_overwritten(self, tmp_path, inputs):
        first = make_recorder(tmp_path, inputs)
        first.abort("TEST_CLEANUP")
        before = first.manifest_path.read_bytes()
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            make_recorder(tmp_path, inputs)
        assert first.manifest_path.read_bytes() == before

    @pytest.mark.parametrize(
        "record",
        [
            {"bgr": [[[0, 0, 0]]]},
            {"payload": {"depth_raw": [[1000]]}},
            {"patient_name": "Alice"},
            {"metadata": {"mrn": "123"}},
            {"blob": b"pixels"},
            {"array": np.zeros((2, 2), dtype=np.uint8)},
            {"tuple": (1, 2)},
            {"bad": float("nan")},
        ],
    )
    def test_non_derived_or_non_json_records_are_rejected(
        self, tmp_path, inputs, record
    ):
        rec = make_recorder(tmp_path, inputs)
        with pytest.raises((TypeError, ValueError)):
            rec.write_derived(record)
        assert rec.counts["derived"] == 0
        rec.abort("TEST_CLEANUP")

    def test_frame_object_is_rejected(self, tmp_path, inputs):
        rec = make_recorder(tmp_path, inputs)
        frame = Frame(
            index=0,
            t=0.0,
            bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        with pytest.raises(TypeError, match="JSON-safe"):
            rec.write_derived({"frame": frame})
        rec.abort("TEST_CLEANUP")

    @pytest.mark.parametrize(
        "record",
        [
            frame_record(people=[{"payload": [[[0, 0, 0]]]}]),
            {
                **frame_record(),
                "monitoring": {
                    "status": "AVAILABLE",
                    "reasons": [],
                    "details": {"note": "Alice MRN 123"},
                    "last_frame_t_rel_s": 0.1,
                },
            },
        ],
    )
    def test_valid_record_kind_rejects_aliased_image_or_free_text_fields(
        self, tmp_path, inputs, record
    ):
        rec = make_recorder(tmp_path, inputs)
        with pytest.raises(ValueError, match="outside the derived-only contract"):
            rec.write_derived(record)
        rec.abort("TEST_CLEANUP")

    def test_time_cannot_move_backwards_across_streams(self, tmp_path, inputs):
        rec = make_recorder(tmp_path, inputs)
        rec.write_derived(frame_record(), t_rel_s=2.0)
        with pytest.raises(ValueError, match="backwards"):
            rec.write_label(phase_record(), t_rel_s=1.9)
        rec.abort("TEST_CLEANUP")

    def test_final_session_rejects_more_records(self, tmp_path, inputs):
        rec = make_recorder(tmp_path, inputs)
        rec.complete()
        with pytest.raises(RuntimeError, match="complete"):
            rec.write_telemetry(heartbeat_record())


class TestTemporalSchema:
    FEATURES = ("h_torso__now", "bed_edge_distance_m__mean_2s")

    def test_schema_records_order_once_and_rows_may_use_vectors(self, tmp_path, inputs):
        rec = make_recorder(
            tmp_path,
            inputs,
            temporal_schema_version="ahfd-temporal-v2",
            temporal_feature_names=self.FEATURES,
        )
        rec.write_derived(
            frame_record(
                people=[
                    {
                        "temporal_summary": {
                            "schema_version": "ahfd-temporal-v2",
                            "values": [0.83, None],
                        }
                    }
                ]
            ),
            t_rel_s=0.1,
        )
        manifest = load_manifest(rec)
        canonical = json.dumps(
            list(self.FEATURES), separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        assert manifest["temporal_schema"] == {
            "schema_version": "ahfd-temporal-v2",
            "feature_names": list(self.FEATURES),
            "feature_order_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        row = json.loads((rec.path / "derived.jsonl").read_text(encoding="utf-8"))
        assert row["record"]["people"][0]["temporal_summary"]["values"] == [
            0.83,
            None,
        ]
        rec.abort("TEST_CLEANUP")

    @pytest.mark.parametrize(
        "overrides,exception",
        [
            ({"temporal_schema_version": "ahfd-temporal-v2"}, ValueError),
            ({"temporal_feature_names": ("feature",)}, ValueError),
            (
                {
                    "temporal_schema_version": "ahfd temporal v2",
                    "temporal_feature_names": ("feature",),
                },
                ValueError,
            ),
            (
                {
                    "temporal_schema_version": "v2",
                    "temporal_feature_names": "not-a-sequence-of-names",
                },
                TypeError,
            ),
            (
                {
                    "temporal_schema_version": "v2",
                    "temporal_feature_names": (),
                },
                ValueError,
            ),
            (
                {
                    "temporal_schema_version": "v2",
                    "temporal_feature_names": ("same", "same"),
                },
                ValueError,
            ),
            (
                {
                    "temporal_schema_version": "v2",
                    "temporal_feature_names": ("patient name",),
                },
                ValueError,
            ),
            (
                {
                    "temporal_schema_version": "v2",
                    "temporal_feature_names": ("_private",),
                },
                ValueError,
            ),
        ],
    )
    def test_temporal_schema_is_all_or_nothing_and_controlled(
        self, tmp_path, inputs, overrides, exception
    ):
        with pytest.raises(exception):
            make_recorder(tmp_path, inputs, **overrides)


class TestRecorderClock:
    def test_elapsed_uses_recorder_epoch_without_advancing_record_watermark(
        self, tmp_path, inputs
    ):
        clock = Clock()
        rec = make_recorder(tmp_path, inputs, clock=clock)
        clock.advance(0.75)
        assert rec.elapsed_s() == pytest.approx(0.75)
        # elapsed_s is read-only: a valid earlier supplied record timestamp is
        # still accepted because no stream record has yet advanced the watermark.
        rec.write_telemetry(heartbeat_record(), t_rel_s=0.5)
        rec.abort("TEST_CLEANUP")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 99.0])
    def test_elapsed_rejects_invalid_or_backward_clock(self, tmp_path, inputs, bad):
        clock = Clock()
        rec = make_recorder(tmp_path, inputs, clock=clock)
        clock.value = bad
        with pytest.raises(ValueError, match="finite non-negative"):
            rec.elapsed_s()
        clock.value = 101.0
        rec.abort("TEST_CLEANUP")
