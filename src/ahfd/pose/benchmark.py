"""Pose backend bake-off.

Runs any PoseEstimator over the same frames and records speed and detection
count, so RTMO and RTMPose (and any future backend) are compared on identical
input rather than on separate anecdotes. This is the harness behind the
report's "which pose model" section.

Speed is measured in steady state, after warm-up: the first few inferences pay
for lazy graph compilation and weight loading, and including them would slander
whichever backend happened to run first. The number that matters operationally
is the median, not the mean -- a single slow frame from a garbage-collection
pause should not dominate.

What this does *not* measure is accuracy. That needs labelled falls, which the
evaluation harness scores; speed and people-count are all that can be had from
unlabelled frames, and conflating the two is how misleading benchmarks happen.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from ahfd.capture.base import Frame
from ahfd.pose.base import PoseEstimator


@dataclass
class BenchResult:
    name: str
    n_frames: int
    median_ms: float
    p90_ms: float
    mean_people: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.median_ms if self.median_ms > 0 else float("inf")

    def line(self) -> str:
        return (
            self.name.ljust(20)
            + "median "
            + format(self.median_ms, "6.1f")
            + " ms  p90 "
            + format(self.p90_ms, "6.1f")
            + " ms  -> "
            + format(self.fps, "5.1f")
            + " fps   people/frame "
            + format(self.mean_people, ".2f")
        )


def benchmark(
    estimator: PoseEstimator,
    frames: list[Frame],
    warmup: int = 3,
) -> BenchResult:
    """Time an estimator over pre-captured frames."""
    if len(frames) <= warmup:
        raise ValueError(
            "need more than " + str(warmup) + " frames to benchmark, got "
            + str(len(frames))
        )

    for i in range(warmup):
        estimator.estimate(frames[i % len(frames)])

    times_ms: list[float] = []
    people: list[int] = []
    for frame in frames:
        t0 = time.perf_counter()
        result = estimator.estimate(frame)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        people.append(len(result.people))

    times_ms.sort()
    p90_idx = min(len(times_ms) - 1, int(0.9 * len(times_ms)))
    return BenchResult(
        name=estimator.name,
        n_frames=len(frames),
        median_ms=statistics.median(times_ms),
        p90_ms=times_ms[p90_idx],
        mean_people=statistics.mean(people) if people else 0.0,
    )
