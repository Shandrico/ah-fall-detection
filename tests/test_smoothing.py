"""One-Euro smoothing behaviour.

The property that matters is the trade-off: suppress jitter on a stationary
person, but do not blunt fast motion. A filter that failed the second half
would erase the very velocity signature the fall trigger depends on, so both
directions are asserted here.
"""

from __future__ import annotations

import numpy as np

from ahfd.pose import KeypointSmoother, OneEuroFilter
from ahfd.types import NUM_KEYPOINTS


class TestOneEuroFilter:
    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(f(0.0, x), x)

    def test_suppresses_jitter_around_a_constant(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
        truth = 100.0

        raw, smoothed = [], []
        for i in range(120):
            noisy = np.array([truth + rng.normal(0, 3.0)], dtype=np.float32)
            out = f(i / 30.0, noisy)
            if i > 30:  # let it settle
                raw.append(float(noisy[0]))
                smoothed.append(float(out[0]))

        assert np.std(smoothed) < np.std(raw) / 2.0, (
            "expected jitter to at least halve; raw std "
            + format(np.std(raw), ".2f")
            + " smoothed std "
            + format(np.std(smoothed), ".2f")
        )

    def test_tracks_fast_motion_without_excessive_lag(self):
        """A fall is fast. The adaptive cutoff must keep up with it."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        speed = 400.0  # pixels per second, a plausible falling torso
        last_out = 0.0
        for i in range(30):
            t = i / 30.0
            x = np.array([speed * t], dtype=np.float32)
            last_out = float(f(t, x)[0])

        expected = speed * (29 / 30.0)
        # Within 15% of truth after a second of constant fast motion.
        assert abs(last_out - expected) < 0.15 * expected, (
            "lagged too far behind: got "
            + format(last_out, ".1f")
            + " expected about "
            + format(expected, ".1f")
        )

    def test_non_advancing_timestamp_is_safe(self):
        """Duplicate timestamps happen on dropped frames; must not divide by zero."""
        f = OneEuroFilter()
        x = np.array([5.0], dtype=np.float32)
        f(1.0, x)
        out = f(1.0, np.array([99.0], dtype=np.float32))
        assert np.isfinite(out).all()

    def test_shape_is_preserved(self):
        f = OneEuroFilter()
        kp = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        f(0.0, kp)
        assert f(1 / 30.0, kp + 1.0).shape == (NUM_KEYPOINTS, 2)


class TestKeypointSmoother:
    def test_untracked_people_pass_through_unchanged(self):
        """Smoothing without an identity would blend different bodies."""
        s = KeypointSmoother()
        kp = np.full((NUM_KEYPOINTS, 2), 7.0, dtype=np.float32)
        np.testing.assert_allclose(s.smooth(None, 0.0, kp), kp)

    def test_state_is_per_track(self):
        s = KeypointSmoother()
        a = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        b = np.full((NUM_KEYPOINTS, 2), 500.0, dtype=np.float32)

        s.smooth(0, 0.0, a)
        s.smooth(1, 0.0, b)
        # Track 1 must not have been dragged toward track 0's position.
        out_b = s.smooth(1, 1 / 30.0, b)
        assert float(out_b[0, 0]) > 400.0

    def test_retain_only_drops_dead_tracks(self):
        s = KeypointSmoother()
        kp = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        s.smooth(0, 0.0, kp)
        s.smooth(1, 0.0, kp)

        s.retain_only({1})
        # Track 0 is gone, so its next sample is treated as a fresh start.
        fresh = np.full((NUM_KEYPOINTS, 2), 42.0, dtype=np.float32)
        np.testing.assert_allclose(s.smooth(0, 1 / 30.0, fresh), fresh)
