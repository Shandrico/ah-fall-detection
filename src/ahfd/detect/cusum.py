"""Causal CUSUM onset detector for the earliest upper-body rise (sit-up).

Why CUSUM instead of a height threshold: the repo's elevation signal (``h_torso``
and friends) is *inferred* from 2D pose and, at the far/oblique ward mount, is
biased -- a reclined patient can read ~1.0-1.6 m instead of ~0.6 m (measured in
the D435f error analysis). An absolute threshold on such a signal is unreliable.
But the *change* when a patient begins to rise survives the bias: a constant
offset cancels once we measure departure from that patient's *own recent resting
baseline*. CUSUM (Page, 1954) accumulates small **sustained** departures, so it
flags the onset of a slow rise well before any single frame crosses a posture
boundary -- exactly the early bed-exit precursor the ward needs, and which
per-frame posture classification cannot provide at this mount.

Causal by construction: every update uses only current and past samples -- no
future frames, no centred smoothing -- so it is valid in a live anticipation
setting (a requirement the design review calls out explicitly).

This module is deliberately decoupled from ``Features``: it takes a plain
``(value, t)`` stream so it can be unit-tested without hardware and reused for
any elevation-like signal (shoulder, torso, head). Wiring it into the bed-exit
state machine -- feeding it ``h_torso`` while a patient is still bed-supported --
is the next step and belongs with the in-bed precursor owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CusumConfig:
    """Tunables. Validate on long *normal* recordings before trusting them."""

    k: float = 0.5
    """Slack, in baseline-std units. Discounts small baseline drift so noise does
    not accumulate; raise it to be less twitchy, at the cost of detection delay."""

    threshold: float = 6.0
    """Fire once the cumulative sum exceeds this. Trades detection delay against
    nuisance triggers -- the central speed/false-alarm knob."""

    warmup_samples: int = 12
    """Initial (assumed-rest) samples used to seed the baseline before the
    detector arms. Monitoring typically starts with a reclined patient."""

    baseline_alpha: float = 0.02
    """EWMA rate for slow baseline updates *during rest only* -- tracks legitimate
    drift (settling, a slowly raised backrest) without chasing a real rise."""

    rest_z: float = 2.0
    """|z| below this counts as "resting", a precondition for updating the
    baseline. A building rise has large z and so cannot be absorbed."""

    min_baseline_std: float = 0.03
    """Metres. Floor on the baseline std so a noise-free rest cannot make z blow
    up; also the ~sensor-noise scale of the elevation estimate."""

    max_gap_s: float = 2.0
    """A gap in valid samples longer than this is treated as reassociation (lost
    or reacquired track) and resets the detector rather than bridging it."""

    restabilize_samples: int = 20
    """Once the signal holds flat for this many samples it is treated as a new
    resting level: the baseline snaps to it and the detector re-arms. This is what
    lets one patient be tracked through several lie-down/sit-up cycles -- each rise
    from whatever rest level they settled at is caught, not just the first."""

    stable_band_m: float = 0.06
    """Metres. The recent window counts as "a stable rest" only if its spread is
    below this. A genuine rise spreads the window well past it, so a rise is never
    mistaken for rest and re-baselined away."""


@dataclass
class CusumSample:
    """The detector's state after one update -- attach to an alert as evidence."""

    t: float
    value: float
    z: float                 # standardised departure from baseline this step
    g: float                 # cumulative sum (the CUSUM statistic)
    baseline_mean: float
    baseline_std: float
    onset: bool              # True only on the step g first crosses threshold
    armed: bool              # baseline established (past warmup) and tracking


class CusumOnset:
    """Stateful, one-sided (upward) CUSUM change detector.

    Feed it one elevation sample per frame via :meth:`update`; it returns a
    :class:`CusumSample` whose ``onset`` is ``True`` on the single step the rise
    is confirmed. It re-arms automatically once the patient settles back to rest,
    so a later rise is caught too. Call :meth:`reset` on track reassociation.
    """

    def __init__(self, config: CusumConfig | None = None) -> None:
        self.cfg = config or CusumConfig()
        self.reset()

    def reset(self) -> None:
        """Forget the baseline and the accumulator (e.g. on reassociation)."""
        self._mean: float | None = None
        self._var: float = self.cfg.min_baseline_std ** 2
        self._warm: list[float] = []
        self._recent: list[float] = []
        self._g: float = 0.0
        self._fired: bool = False
        self._last_t: float | None = None

    @property
    def armed(self) -> bool:
        """True once the warmup baseline is established."""
        return self._mean is not None

    def update(self, value: float | None, t: float) -> CusumSample:
        """Advance the detector by one sample and return its state.

        ``value`` is the elevation signal (metres); pass ``None`` or a non-finite
        value for a frame with no confident estimate -- the detector pauses (holds
        its state) rather than treating missing evidence as stillness.
        """
        cfg = self.cfg
        gap = self._last_t is not None and (t - self._last_t) > cfg.max_gap_s

        # Missing evidence: pause. A long gap means the track was lost/reacquired,
        # so drop the stale baseline instead of bridging across the gap.
        if value is None or not math.isfinite(value):
            if gap:
                self.reset()
            return CusumSample(
                t, float("nan"), 0.0, self._g,
                self._mean if self._mean is not None else float("nan"),
                math.sqrt(self._var), False, self.armed,
            )

        if gap:
            self.reset()
        self._last_t = t

        # Warmup: seed the baseline from the initial (assumed-rest) samples.
        if self._mean is None:
            self._warm.append(value)
            if len(self._warm) >= cfg.warmup_samples:
                self._mean = sum(self._warm) / len(self._warm)
                mean = self._mean
                var = sum((v - mean) ** 2 for v in self._warm) / len(self._warm)
                self._var = max(var, cfg.min_baseline_std ** 2)
                self._warm.clear()
            return CusumSample(t, value, 0.0, 0.0, value, cfg.min_baseline_std, False, False)

        std = max(math.sqrt(self._var), cfg.min_baseline_std)
        z = (value - self._mean) / std
        self._g = max(0.0, self._g + z - cfg.k)

        onset = self._g > cfg.threshold and not self._fired
        if onset:
            self._fired = True

        self._recent.append(value)
        if len(self._recent) > cfg.restabilize_samples:
            self._recent.pop(0)
        settled = (
            len(self._recent) >= cfg.restabilize_samples
            and (max(self._recent) - min(self._recent)) < cfg.stable_band_m
        )

        if settled:
            # The patient is holding a stable level (possibly a *new* one, e.g.
            # they lay back down). Adopt it as the baseline, clear the accumulator
            # and re-arm, so the next rise -- from this level -- is caught.
            mean = sum(self._recent) / len(self._recent)
            var = sum((v - mean) ** 2 for v in self._recent) / len(self._recent)
            self._mean = mean
            self._var = max(var, cfg.min_baseline_std ** 2)
            self._g = 0.0
            self._fired = False
        elif abs(z) < cfg.rest_z and self._g <= cfg.k:
            # Near the current baseline but the window is not yet full: track slow
            # legitimate drift (settling, a slowly raised backrest) with an EWMA.
            a = cfg.baseline_alpha
            prev = self._mean
            self._mean = (1.0 - a) * prev + a * value
            self._var = max((1.0 - a) * self._var + a * (value - prev) ** 2,
                            cfg.min_baseline_std ** 2)
            self._fired = False

        return CusumSample(t, value, z, self._g, self._mean, std, onset, True)

    def run(self, series: list[tuple[float, float | None]]) -> list[CusumSample]:
        """Convenience for offline/testing: map a list of ``(t, value)`` samples."""
        return [self.update(v, t) for t, v in series]
