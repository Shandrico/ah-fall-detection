"""Event-level metrics.

The headline number is **false alarms per camera-hour**, and it is deliberately
first in every report, because it is the one number that decides whether nurses
keep the system switched on. A detector that catches every fall and also cries
wolf twice a shift will be unplugged by the end of the week, and its recall will
not save it.

Everything here is **event-level, not frame-level**. Frame-level accuracy is the
wrong unit and it flatters the system: a fall is a handful of frames against
hours of nothing, so a model can score 99.9% frame accuracy while missing every
fall. What a nurse experiences is events -- "it caught the fall" / "it paged me
for nothing" -- so that is what is measured.

The matching rule is stated explicitly rather than left implicit, because a
vague rule silently changes the numbers:

    A predicted alert is a true positive if it lands in the window
    [t_impact - PRE, t_impact + POST] of a real fall. Each real fall matches at
    most one alert (the nearest), and each alert matches at most one fall.
    Unmatched alerts are false positives. Unmatched falls are misses.

The window is asymmetric on purpose. An alert slightly *before* impact is fine
(the fall was already in progress). A long tail *after* is allowed because
confirmation intentionally waits ~8 s before paging -- so POST must exceed the
confirmation delay or every correct alert would score as both a miss and a
false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ahfd.eval.annotations import GroundTruth, PredictedEvent

# What counts as an alerting event. BED_EXIT and NEAR_MISS are informational --
# they are not claims that a fall happened, so they are excluded from fall
# precision/recall. Scoring them as false positives would punish the system for
# working as designed.
ALERTING_TYPES = frozenset({"FALL_CONFIRMED", "PERSON_DOWN"})

DEFAULT_PRE_S = 2.0
DEFAULT_POST_S = 30.0


@dataclass
class Match:
    fall_impact: float
    event: PredictedEvent
    latency_s: float


@dataclass
class ClipResult:
    clip_id: str
    duration_s: float
    true_positives: list[Match] = field(default_factory=list)
    false_positives: list[PredictedEvent] = field(default_factory=list)
    misses: list[float] = field(default_factory=list)  # unmatched impact times

    @property
    def n_tp(self) -> int:
        return len(self.true_positives)

    @property
    def n_fp(self) -> int:
        return len(self.false_positives)

    @property
    def n_miss(self) -> int:
        return len(self.misses)


def match_clip(
    truth: GroundTruth,
    events: list[PredictedEvent],
    pre_s: float = DEFAULT_PRE_S,
    post_s: float = DEFAULT_POST_S,
) -> ClipResult:
    """Match one clip's alerts against its ground truth."""
    alerts = sorted(
        (e for e in events if e.type in ALERTING_TYPES), key=lambda e: e.t_alert
    )

    result = ClipResult(clip_id=truth.clip_id, duration_s=truth.duration_s)
    claimed: set[int] = set()  # indices of alerts already used

    # Greedy, nearest alert to each fall. Falls are few, so this is fine and
    # avoids the ambiguity of letting one alert satisfy two falls.
    for fall in truth.falls:
        window = (fall.t_impact - pre_s, fall.t_impact + post_s)
        best_i = None
        best_dist = None
        for i, alert in enumerate(alerts):
            if i in claimed:
                continue
            if window[0] <= alert.t_alert <= window[1]:
                dist = abs(alert.t_alert - fall.t_impact)
                if best_dist is None or dist < best_dist:
                    best_dist, best_i = dist, i
        if best_i is None:
            result.misses.append(fall.t_impact)
        else:
            claimed.add(best_i)
            alert = alerts[best_i]
            result.true_positives.append(
                Match(
                    fall_impact=fall.t_impact,
                    event=alert,
                    latency_s=alert.t_alert - fall.t_impact,
                )
            )

    for i, alert in enumerate(alerts):
        if i not in claimed:
            result.false_positives.append(alert)

    return result


@dataclass
class Report:
    """Aggregate metrics over a set of clips."""

    results: list[ClipResult] = field(default_factory=list)

    @property
    def n_tp(self) -> int:
        return sum(r.n_tp for r in self.results)

    @property
    def n_fp(self) -> int:
        return sum(r.n_fp for r in self.results)

    @property
    def n_miss(self) -> int:
        return sum(r.n_miss for r in self.results)

    @property
    def total_hours(self) -> float:
        return sum(r.duration_s for r in self.results) / 3600.0

    @property
    def recall(self) -> float:
        denom = self.n_tp + self.n_miss
        return self.n_tp / denom if denom else float("nan")

    @property
    def precision(self) -> float:
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:  # nan-safe
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def false_alarms_per_hour(self) -> float:
        """The headline. False positives divided by total footage in hours."""
        return self.n_fp / self.total_hours if self.total_hours > 0 else float("nan")

    @property
    def latencies(self) -> list[float]:
        return [m.latency_s for r in self.results for m in r.true_positives]

    def latency_median(self) -> float:
        lat = sorted(self.latencies)
        if not lat:
            return float("nan")
        n = len(lat)
        return lat[n // 2] if n % 2 else (lat[n // 2 - 1] + lat[n // 2]) / 2

    def latency_p90(self) -> float:
        lat = sorted(self.latencies)
        if not lat:
            return float("nan")
        idx = min(len(lat) - 1, int(0.9 * len(lat)))
        return lat[idx]


def evaluate(
    pairs: list[tuple[GroundTruth, list[PredictedEvent]]],
    pre_s: float = DEFAULT_PRE_S,
    post_s: float = DEFAULT_POST_S,
) -> Report:
    """Match every (truth, events) pair and aggregate."""
    return Report(
        results=[match_clip(t, e, pre_s, post_s) for t, e in pairs]
    )
