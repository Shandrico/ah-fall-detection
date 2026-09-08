"""End-to-end detection, from projected pixels to an emitted event.

The other test modules each verify one layer against known truth. This one
joins them: a 3D body moves in the ward, is projected through the real camera
model to pixels, and those pixels go through the actual FeatureExtractor and
the actual FallStateMachine. Nothing is stubbed and no feature is hand-fed.

That matters because the layers can each be right and still not compose. A
threshold in metres is only meaningful if the metres arriving from the
geometry land where the threshold expects them, and the only way to know is
to run the chain.

The camera is the real ward geometry: 2.6 m, 20 degrees down, beds at 6-7.5 m.
"""

from __future__ import annotations

import numpy as np
import pytest

from ahfd.alert import ConsoleSink, MultiSink
from ahfd.detect import FallStateMachine, FallThresholds
from ahfd.features import FeatureExtractor
from ahfd.geometry import GroundPlane
from ahfd.geometry.zones import Zone, ZoneMap, rectangle
from ahfd.types import NUM_KEYPOINTS, Intrinsics, PersonPose

FPS = 15.0
DT = 1.0 / FPS

WARD = GroundPlane(
    Intrinsics.from_hfov(1920, 1080, hfov_deg=69.4, vfov_deg=42.5),
    height_m=2.6,
    pitch_deg=20.0,
)

# (height above floor, lateral offset) for a 1.7 m adult, COCO-17 order.
BODY = (
    (1.62, 0.00), (1.64, 0.03), (1.64, -0.03), (1.62, 0.07), (1.62, -0.07),
    (1.40, 0.18), (1.40, -0.18), (1.10, 0.20), (1.10, -0.20),
    (0.85, 0.20), (0.85, -0.20), (0.95, 0.12), (0.95, -0.12),
    (0.50, 0.12), (0.50, -0.12), (0.08, 0.10), (0.08, -0.10),
)


def _project(point: np.ndarray) -> tuple[float, float]:
    rel = np.asarray(point, float) - np.array([0.0, 0.0, WARD.height_m])
    d = WARD.rotation.T @ rel
    k = WARD.intrinsics
    return (k.cx + k.fx * d[0] / d[2], k.cy + k.fy * d[1] / d[2])


def _joints_standing(x: float, y: float) -> np.ndarray:
    return np.array(
        [[x + lat, y, h] for h, lat in BODY],
        dtype=float,
    )


def _joints_lying(x: float, y: float, thickness: float = 0.12) -> np.ndarray:
    """Flat on the floor, feet at (x, y), body extending away from camera."""
    return np.array(
        [[x + lat, y + (h - 0.08), thickness] for h, lat in BODY],
        dtype=float,
    )


def _pose_from(joints3d: np.ndarray, track_id: int = 1) -> PersonPose:
    pts = np.array([_project(p) for p in joints3d], dtype=np.float32)
    return PersonPose(
        keypoints=pts,
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
        score=1.0,
        track_id=track_id,
    )


def fall_frames(
    x: float = 0.0,
    y: float = 6.0,
    stand_s: float = 3.0,
    fall_s: float = 0.5,
    down_s: float = 14.0,
) -> list[tuple[float, PersonPose]]:
    """Stand, collapse to the floor, then stay there.

    The collapse linearly interpolates 3D joint positions between the upright
    and prone bodies. Crude as biomechanics, but it produces the signature
    that matters: torso height dropping fast while the joints migrate onto
    the floor plane.
    """
    upright = _joints_standing(x, y)
    prone = _joints_lying(x, y)
    frames: list[tuple[float, PersonPose]] = []

    t = 0.0
    for _ in range(int(stand_s * FPS)):
        frames.append((t, _pose_from(upright)))
        t += DT

    n_fall = max(1, int(fall_s * FPS))
    for i in range(n_fall):
        alpha = (i + 1) / n_fall
        frames.append((t, _pose_from(upright * (1 - alpha) + prone * alpha)))
        t += DT

    for _ in range(int(down_s * FPS)):
        frames.append((t, _pose_from(prone)))
        t += DT

    return frames


def standing_frames(
    x: float = 0.0, y: float = 6.0, seconds: float = 30.0
) -> list[tuple[float, PersonPose]]:
    upright = _joints_standing(x, y)
    return [(i * DT, _pose_from(upright)) for i in range(int(seconds * FPS))]


def run_chain(
    frames: list[tuple[float, PersonPose]],
    zones: ZoneMap | None = None,
    thresholds: FallThresholds | None = None,
):
    extractor = FeatureExtractor(WARD, zones=zones)
    machine = FallStateMachine(thresholds)
    events = []
    for t, person in frames:
        features = extractor.extract(person, t)
        assert features is not None
        event = machine.update(features)
        if event is not None:
            events.append(event)
    return events, machine


class TestFallIsDetectedEndToEnd:
    def test_a_collapse_is_confirmed(self):
        events, _ = run_chain(fall_frames())
        types = [e.type for e in events]
        assert "FALL_CONFIRMED" in types, "expected a confirmed fall, got " + str(types)

    def test_evidence_reflects_real_geometry(self):
        """The numbers in the alert must come from the projection, not defaults."""
        events, _ = run_chain(fall_frames())
        confirmed = next(e for e in events if e.type == "FALL_CONFIRMED")

        assert confirmed.evidence["h_before"] > 1.0  # was standing
        assert confirmed.evidence["h_after"] < 0.7  # ended low
        assert confirmed.evidence["peak_vz"] < -0.9  # fell fast
        assert 1.0 < confirmed.evidence["floor_spread"] < 3.0  # a body long
        assert confirmed.evidence["range_m"] == pytest.approx(6.5, abs=1.0)

    def test_suspicion_precedes_confirmation(self):
        events, _ = run_chain(fall_frames())
        suspected = next(e for e in events if e.type == "FALL_SUSPECTED")
        confirmed = next(e for e in events if e.type == "FALL_CONFIRMED")
        assert suspected.t_alert < confirmed.t_alert

    @pytest.mark.parametrize("distance", [4.0, 6.0, 7.4])
    def test_detected_at_every_bed_distance(self, distance):
        """The payoff of metric features: one threshold set, all three beds.

        This is the test that pixel-based thresholds cannot pass. The same
        fall at 4 m and 7.4 m produces very different pixel velocities but
        the same metres per second.
        """
        events, _ = run_chain(fall_frames(y=distance))
        assert "FALL_CONFIRMED" in [e.type for e in events], (
            "fall missed at " + str(distance) + " m"
        )

    @pytest.mark.parametrize("lateral", [-2.5, 0.0, 2.5])
    def test_detected_across_the_frame(self, lateral):
        """Off-axis too, not just down the optical centre."""
        events, _ = run_chain(fall_frames(x=lateral, y=6.5))
        assert "FALL_CONFIRMED" in [e.type for e in events]


class TestNoFalseAlarmsEndToEnd:
    def test_standing_still_produces_nothing(self):
        events, machine = run_chain(standing_frames())
        assert events == []
        assert machine.state_of(1) == "UPRIGHT"

    def test_walking_towards_the_camera_produces_nothing(self):
        """The classic 2D failure: approaching the camera makes a bounding box
        wide and short, which aspect-ratio methods read as 'lying down'.
        Metric height is unmoved by it.
        """
        frames = []
        for i in range(int(20 * FPS)):
            y = 7.5 - i * 0.012  # walk from 7.5 m to ~4 m
            frames.append((i * DT, _pose_from(_joints_standing(0.0, y))))

        events, machine = run_chain(frames)
        assert events == []
        assert machine.state_of(1) == "UPRIGHT"

    def test_lying_in_bed_produces_nothing(self):
        """The dominant ward false positive, through the whole chain."""
        bed = Zone(
            name="bed_2",
            kind="bed",
            polygon=rectangle((0.0, 6.0), length=2.4, width=1.4, angle_deg=0.0),
            top_m=0.55,
        )
        # A body on the bed surface rather than the floor.
        prone_in_bed = _joints_lying(0.0, 6.0, thickness=0.60)
        frames = [(i * DT, _pose_from(prone_in_bed)) for i in range(int(40 * FPS))]

        events, _ = run_chain(frames, zones=ZoneMap([bed]))
        assert [e.type for e in events] == []


class TestAlertPlumbing:
    def test_events_reach_the_sinks(self, capsys):
        events, _ = run_chain(fall_frames())
        sink = MultiSink(ConsoleSink())
        for event in events:
            sink.emit(event)
        sink.close()

        out = capsys.readouterr().out
        assert "FALL_CONFIRMED" in out
        assert "peak_vz" in out, "the console line must carry its evidence"

    def test_a_broken_sink_does_not_silence_the_others(self, capsys):
        """A failed log write must not also take out the console."""

        class Broken:
            def emit(self, event):
                raise OSError("disk full")

            def close(self):
                pass

        events, _ = run_chain(fall_frames())
        sink = MultiSink(Broken(), ConsoleSink())
        for event in events:
            sink.emit(event)

        out = capsys.readouterr().out
        assert "FALL_CONFIRMED" in out
        assert "failed" in out
