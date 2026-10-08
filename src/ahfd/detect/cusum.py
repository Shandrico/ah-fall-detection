"""Causal, confidence-aware CUSUM change detection.

The detector intentionally knows nothing about beds or poses. Its caller must
say when a sample is eligible to teach the resting baseline. This is a safety
property: a patient who has risen and then holds still must not have that
continuing exit silently absorbed as a new baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CusumConfig:
    """Tunables for a one-sided upward CUSUM."""

    k: float = 0.5
    threshold: float = 6.0
    warmup_samples: int = 12
    baseline_alpha: float = 0.02
    rest_z: float = 2.0
    min_baseline_std: float = 0.03
    max_gap_s: float = 2.0
    reference_hz: float = 10.0
    max_step_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.warmup_samples < 2:
            raise ValueError("warmup_samples must be at least 2")
        if self.min_baseline_std <= 0 or self.reference_hz <= 0:
            raise ValueError("min_baseline_std and reference_hz must be positive")
        if self.max_gap_s <= 0 or self.max_step_scale <= 0:
            raise ValueError("max_gap_s and max_step_scale must be positive")


@dataclass(frozen=True)
class CusumSample:
    """Detector state after one observation; suitable for alert evidence."""

    t: float
    value: float
    z: float
    g: float
    baseline_mean: float
    baseline_std: float
    onset: bool
    armed: bool
    valid: bool = True
    baseline_eligible: bool = False


class CusumOnset:
    """Stateful, one-sided and strictly causal change detector.

    ``baseline_eligible`` is deliberately explicit. Set it only for a
    confidently observed stable-rest sample. Once armed, all valid samples are
    inspected, but only eligible ones can update or replace the baseline.
    Invalid samples pause the accumulator; a sufficiently long gap resets it.
    """

    def __init__(self, config: CusumConfig | None = None) -> None:
        self.cfg = config or CusumConfig()
        self.reset()

    def reset(self) -> None:
        """Forget the baseline and accumulator (track loss/reassociation)."""
        self._mean: float | None = None
        self._var = self.cfg.min_baseline_std**2
        self._warm: list[float] = []
        self._g = 0.0
        self._fired = False
        self._last_valid_t: float | None = None
        # A valid sample after an observation outage must not retrospectively
        # turn the unobserved interval into positive CUSUM evidence.  The
        # resumed sample still contributes one nominal reference-rate step.
        self._resume_after_invalid = False

    @property
    def armed(self) -> bool:
        return self._mean is not None

    @property
    def fired(self) -> bool:
        return self._fired

    def _sample(
        self,
        *,
        t: float,
        value: float,
        z: float = 0.0,
        onset: bool = False,
        valid: bool,
        baseline_eligible: bool,
    ) -> CusumSample:
        return CusumSample(
            t=t,
            value=value,
            z=z,
            g=self._g,
            baseline_mean=(self._mean if self._mean is not None else float("nan")),
            baseline_std=max(math.sqrt(self._var), self.cfg.min_baseline_std),
            onset=onset,
            armed=self.armed,
            valid=valid,
            baseline_eligible=baseline_eligible,
        )

    def update(
        self,
        value: float | None,
        t: float,
        *,
        baseline_eligible: bool = False,
        valid: bool = True,
    ) -> CusumSample:
        """Advance by one observation.

        ``valid=False`` (or a missing/non-finite value) contributes neither
        positive nor negative evidence. A gap longer than ``max_gap_s`` drops
        the baseline so evidence cannot bridge a lost/reassociated track.
        """
        finite = value is not None and math.isfinite(value)
        gap = (
            self._last_valid_t is not None
            and t - self._last_valid_t > self.cfg.max_gap_s
        )

        if not valid or not finite:
            if gap:
                self.reset()
            else:
                self._resume_after_invalid = True
            return self._sample(
                t=t,
                value=float("nan"),
                valid=False,
                baseline_eligible=False,
            )

        assert value is not None
        value = float(value)
        previous_t = None if self._resume_after_invalid else self._last_valid_t
        if gap:
            self.reset()
            previous_t = None
        self._resume_after_invalid = False
        self._last_valid_t = t

        # A baseline may only be seeded from an explicitly eligible rest run.
        if self._mean is None:
            if not baseline_eligible:
                self._warm.clear()
                return self._sample(
                    t=t,
                    value=value,
                    valid=True,
                    baseline_eligible=False,
                )
            self._warm.append(value)
            if len(self._warm) >= self.cfg.warmup_samples:
                mean = sum(self._warm) / len(self._warm)
                var = sum((v - mean) ** 2 for v in self._warm) / len(self._warm)
                self._mean = mean
                self._var = max(var, self.cfg.min_baseline_std**2)
                self._warm.clear()
            return self._sample(
                t=t,
                value=value,
                valid=True,
                baseline_eligible=True,
            )

        std = max(math.sqrt(self._var), self.cfg.min_baseline_std)
        z = (value - self._mean) / std

        # Accumulate per reference-time step, not per frame, so a threshold is
        # approximately comparable at different inference frame rates.
        dt = (1.0 / self.cfg.reference_hz) if previous_t is None else max(0.0, t - previous_t)
        scale = min(dt * self.cfg.reference_hz, self.cfg.max_step_scale)
        self._g = max(0.0, self._g + (z - self.cfg.k) * scale)

        onset = self._g > self.cfg.threshold and not self._fired
        if onset:
            self._fired = True

        if baseline_eligible and not self._fired:
            if abs(z) < self.cfg.rest_z and self._g <= self.cfg.k:
                a = self.cfg.baseline_alpha
                previous_mean = self._mean
                self._mean = (1.0 - a) * previous_mean + a * value
                self._var = max(
                    (1.0 - a) * self._var + a * (value - previous_mean) ** 2,
                    self.cfg.min_baseline_std**2,
                )
        return self._sample(
            t=t,
            value=value,
            z=z,
            onset=onset,
            valid=True,
            baseline_eligible=baseline_eligible,
        )

    def run(
        self,
        series: Iterable[tuple[float, float | None]],
        *,
        baseline_eligible: bool = False,
    ) -> list[CusumSample]:
        """Map ``(t, value)`` observations, primarily for offline tests."""
        return [
            self.update(value, t, baseline_eligible=baseline_eligible)
            for t, value in series
        ]
