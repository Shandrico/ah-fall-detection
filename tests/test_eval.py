"""Evaluation harness tests.

The matching rule is where a metric silently lies, so most of these pin down
exactly what counts as a hit, a miss and a false alarm. The numbers a report
prints are only trustworthy if this matcher is.
"""

from __future__ import annotations

import json
import math

import pytest

from ahfd.eval import (
    GroundTruth,
    PredictedEvent,
    TrueFall,
    evaluate,
    load_events,
    match_clip,
    render_markdown,
)
from ahfd.eval.metrics import Report


def fall(t_impact: float) -> TrueFall:
    return TrueFall(t_impact=t_impact)


def alert(t_alert: float, type: str = "FALL_CONFIRMED", **evidence) -> PredictedEvent:
    return PredictedEvent(
        type=type,
        t_alert=t_alert,
        t_trigger=t_alert - 8.0,
        track_id=1,
        severity=4,
        evidence=evidence,
    )


class TestMatching:
    def test_alert_after_impact_within_window_is_a_hit(self):
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        # Confirmation pages ~8 s after impact -- must count as a hit.
        result = match_clip(truth, [alert(18.0)])
        assert result.n_tp == 1
        assert result.n_fp == 0
        assert result.n_miss == 0
        assert result.true_positives[0].latency_s == pytest.approx(8.0)

    def test_alert_slightly_before_impact_is_a_hit(self):
        """The fall was already in progress; an early alert is correct."""
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        result = match_clip(truth, [alert(9.0)])
        assert result.n_tp == 1

    def test_alert_far_after_impact_is_a_false_positive_and_a_miss(self):
        """Outside the window, one event becomes two problems: the fall is
        unmatched (miss) and the stray alert is unexplained (false positive)."""
        truth = GroundTruth("c1", duration_s=600.0, falls=(fall(10.0),))
        result = match_clip(truth, [alert(120.0)])
        assert result.n_tp == 0
        assert result.n_miss == 1
        assert result.n_fp == 1

    def test_no_alert_is_a_miss(self):
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        result = match_clip(truth, [])
        assert result.n_miss == 1
        assert result.n_tp == 0

    def test_alert_on_a_negative_clip_is_a_false_positive(self):
        truth = GroundTruth("c1", duration_s=1800.0, falls=())
        result = match_clip(truth, [alert(400.0)])
        assert result.n_fp == 1
        assert result.n_tp == 0

    def test_quiet_negative_clip_is_clean(self):
        truth = GroundTruth("c1", duration_s=1800.0, falls=())
        result = match_clip(truth, [])
        assert result.n_fp == 0 and result.n_tp == 0 and result.n_miss == 0

    def test_each_fall_claims_at_most_one_alert(self):
        """Two alerts near one fall: one hit, one false positive -- never two
        hits from a single fall."""
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        result = match_clip(truth, [alert(11.0), alert(13.0)])
        assert result.n_tp == 1
        assert result.n_fp == 1

    def test_each_alert_claims_at_most_one_fall(self):
        """Two falls close together, one alert: one hit, one miss."""
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0), fall(12.0)))
        result = match_clip(truth, [alert(11.0)])
        assert result.n_tp == 1
        assert result.n_miss == 1

    def test_nearest_alert_wins(self):
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        result = match_clip(truth, [alert(10.5), alert(25.0)])
        assert result.n_tp == 1
        assert result.true_positives[0].event.t_alert == 10.5
        assert result.n_fp == 1

    def test_informational_events_are_not_scored(self):
        """BED_EXIT and NEAR_MISS are not fall claims, so they are neither
        hits nor false positives."""
        truth = GroundTruth("c1", duration_s=1800.0, falls=())
        events = [
            alert(100.0, type="BED_EXIT"),
            alert(200.0, type="NEAR_MISS"),
        ]
        result = match_clip(truth, events)
        assert result.n_fp == 0

    def test_person_down_counts_as_an_alerting_event(self):
        truth = GroundTruth("c1", duration_s=60.0, falls=(fall(10.0),))
        result = match_clip(truth, [alert(30.0, type="PERSON_DOWN")])
        assert result.n_tp == 1


class TestReportMetrics:
    def build(self) -> Report:
        pairs = [
            # two clean detections
            (GroundTruth("c1", 60.0, (fall(10.0),)), [alert(16.0)]),
            (GroundTruth("c2", 60.0, (fall(20.0),)), [alert(27.0)]),
            # a miss
            (GroundTruth("c3", 60.0, (fall(30.0),)), []),
            # a negative clip with one false alarm, half an hour long
            (GroundTruth("c4", 1800.0, ()), [alert(500.0)]),
        ]
        return evaluate(pairs)

    def test_counts(self):
        r = self.build()
        assert r.n_tp == 2
        assert r.n_miss == 1
        assert r.n_fp == 1

    def test_recall(self):
        assert self.build().recall == pytest.approx(2 / 3)

    def test_precision(self):
        assert self.build().precision == pytest.approx(2 / 3)

    def test_false_alarms_per_hour(self):
        r = self.build()
        # total footage = 60 + 60 + 60 + 1800 = 1980 s = 0.55 h; 1 FP.
        assert r.false_alarms_per_hour == pytest.approx(1 / 0.55, rel=1e-3)

    def test_latency_stats(self):
        r = self.build()
        assert r.latency_median() == pytest.approx(6.5)  # median of 6 and 7
        assert r.latency_p90() == pytest.approx(7.0)

    def test_empty_report_is_nan_not_crash(self):
        r = Report()
        assert math.isnan(r.recall)
        assert math.isnan(r.false_alarms_per_hour)

    def test_perfect_run(self):
        pairs = [
            (GroundTruth("c1", 60.0, (fall(10.0),)), [alert(15.0)]),
            (GroundTruth("c2", 3600.0, ()), []),
        ]
        r = evaluate(pairs)
        assert r.recall == 1.0
        assert r.false_alarms_per_hour == 0.0


class TestAnnotationLoading:
    def test_negative_clip_needs_a_duration(self, tmp_path):
        """The false-alarm denominator: a clip with no duration is unusable."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"clip_id": "x", "falls": []}))
        with pytest.raises(ValueError, match="duration_s"):
            GroundTruth.load(path)

    def test_round_trip(self, tmp_path):
        path = tmp_path / "c1.json"
        path.write_text(
            json.dumps(
                {
                    "clip_id": "c1",
                    "duration_s": 42.0,
                    "falls": [{"t_impact": 12.0, "t_start": 11.5, "subject": "S1"}],
                }
            )
        )
        gt = GroundTruth.load(path)
        assert gt.clip_id == "c1"
        assert gt.duration_s == 42.0
        assert len(gt.falls) == 1
        assert gt.falls[0].t_impact == 12.0
        assert not gt.is_negative

    def test_events_round_trip_from_sink_format(self, tmp_path):
        """The event log the JsonlSink writes must read straight back."""
        from ahfd.alert import JsonlSink
        from ahfd.detect.events import Event

        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)
        sink.emit(
            Event(
                type="FALL_CONFIRMED",
                track_id=3,
                t_trigger=41.0,
                t_alert=49.0,
                zone="bed_2",
                evidence={"peak_vz": -1.4},
            )
        )
        sink.close()

        events = load_events(path)
        assert len(events) == 1
        assert events[0].type == "FALL_CONFIRMED"
        assert events[0].t_alert == 49.0
        assert events[0].evidence["peak_vz"] == -1.4


class TestReportRendering:
    def test_headline_number_is_present(self):
        pairs = [(GroundTruth("c1", 3600.0, ()), [alert(100.0)])]
        md = render_markdown(evaluate(pairs))
        assert "False alarms per camera-hour" in md
        assert "## Headline" in md

    def test_false_alarm_table_lists_the_offender(self):
        pairs = [(GroundTruth("c1", 3600.0, ()), [alert(100.0, peak_vz=-1.2)])]
        md = render_markdown(evaluate(pairs))
        assert "## False alarms" in md
        assert "peak_vz=-1.2" in md

    def test_missed_falls_are_listed(self):
        pairs = [(GroundTruth("c1", 60.0, (fall(10.0),)), [])]
        md = render_markdown(evaluate(pairs))
        assert "## Missed falls" in md

    def test_renders_without_crashing_on_empty(self):
        md = render_markdown(Report())
        assert "n/a" in md
