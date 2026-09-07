"""One-Euro keypoint smoothing.

This is load-bearing, not cosmetic. Raw keypoint jitter of only three or four
pixels turns into phantom vertical velocity once heights are differentiated,
and the fall trigger is a velocity threshold -- so unsmoothed keypoints
manufacture falls out of a person standing still.

A One-Euro filter is the right tool rather than a plain low-pass or a Kalman
filter: it adapts its cutoff to the observed speed, so it smooths hard while a
person is still but barely lags during the fast motion of an actual fall. A
fixed low-pass would have to choose between jitter and blunting the very
event we are trying to detect.

Reference: Casiez, Roussel & Vogel, "1 euro filter" (CHI 2012).
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter over an array of arbitrary shape."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            # Repeated or out-of-order timestamp: pass through untouched rather
            # than dividing by zero.
            return self._x_prev.copy()

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        # Vectorised alpha: each coordinate gets its own cutoff.
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """Per-track One-Euro smoothing of COCO-17 keypoints.

    State is keyed by track id, so identities that come and go do not inherit
    each other's history. Untracked people are passed through unchanged --
    smoothing without a stable identity would mix different bodies together,
    which is worse than not smoothing at all.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._filters: dict[int, OneEuroFilter] = {}

    def smooth(self, track_id: int | None, t: float, keypoints: np.ndarray) -> np.ndarray:
        if track_id is None:
            return keypoints
        f = self._filters.get(track_id)
        if f is None:
            f = OneEuroFilter(self._min_cutoff, self._beta, self._d_cutoff)
            self._filters[track_id] = f
        return f(t, keypoints)

    def forget(self, track_id: int) -> None:
        self._filters.pop(track_id, None)

    def retain_only(self, live_ids: set[int]) -> None:
        """Drop filter state for tracks that no longer exist."""
        for tid in list(self._filters):
            if tid not in live_ids:
                del self._filters[tid]
