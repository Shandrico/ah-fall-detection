"""Bed-exit decision tests.

Like the fall tests, these are synthetic sequences with explicit timestamps and
no camera -- but they run through the *real* geometry stack. A pose is built by
placing COCO joints at chosen positions in the bed's own frame and
inverse-projecting them to pixels, so the keypoints that reach the extractor are
exactly what the calibrated camera would have produced. That means the tests
exercise projection, bed-frame maths and the state machine together, which is
where the limbs-don't-trigger guarantee actually has to hold.

The negatives matter most: an arm over the rail, a leg dangling, a patient
sleeping near the rail -- every one of these must stay silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from ahfd.detect import BedExitStateMachine, BedExitThresholds
from ahfd.features import BedFrameExtractor
from ahfd.features.bed_frame import BedFrame, default_bed_edges
from ahfd.geometry.ground import GroundPlane
from ahfd.geometry.zones import Zone, ZoneMap, rectangle
from ahfd.types import Intrinsics, PersonPose

FPS = 20.0
DT = 1.0 / FPS

BED_CENTRE = (0.0, 6.0)
BED_LENGTH = 2.2
BED_WIDTH = 1.05
BED_TOP = 0.55
HALF_W = BED_WIDTH / 2.0  # 0.525


def make_ground() -> GroundPlane:
    return GroundPlane(
        Intrinsics.from_hfov(1920, 1080, 69.4, 42.5),
        height_m=2.6,
        pitch_deg=20.0,
    )


def make_bed(risk: str = "high") -> Zone:
    return Zone(
        name="bed_1",
        kind="bed",
        polygon=rectangle(BED_CENTRE, BED_LENGTH, BED_WIDTH, 0.0),
        top_m=BED_TOP,
        risk_level=risk,
        edges=default_bed_edges(rail_gap_foot_m=0.48),
    )


def make_stack(risk: str = "high", **th_kwargs):
    ground = make_ground()
    zone = make_bed(risk)
    zones = ZoneMap([zone])
    extractor = BedFrameExtractor(ground, zones)
    th = BedExitThresholds(
        bind_s=0.5,
        approach_s=0.4,
        legs_over_s=0.3,
        exit_confirm_s=0.3,
        abort_s=0.4,
        degraded_after_s=0.3,
        recover_s=0.3,
        risk_cooldown_s=0.0,
        **th_kwargs,
    )
    machine = BedExitStateMachine(th, extractor)
    frame = BedFrame.from_zone(zone)
    ground = extractor.ground
    return machine, frame, ground


# COCO-17 groups
_CORE = (5, 6, 11, 12)
_LEGS = (13, 14, 15, 16)
_ARMS = (7, 8, 9, 10, 0, 1, 2, 3, 4)


def make_pose(
    frame: BedFrame,
    ground: GroundPlane,
    *,
    track_id: int = 1,
    core_by: float = 0.0,
    leg_by: float | None = None,
    arm_by: float | None = None,
    core_bx: float = 0.0,
    conf: float = 0.9,
    core_valid: bool = True,
    legs_valid: bool = True,
) -> PersonPose:
    """A pose with core / legs / arms placed at given bed-frame y positions."""
    leg_by = core_by if leg_by is None else leg_by
    arm_by = core_by if arm_by is None else arm_by

    def place(bx: float, by: float):
        fx = frame.origin[0] + bx * frame.ex[0] + by * frame.ey[0]
        fy = frame.origin[1] + bx * frame.ex[1] + by * frame.ey[1]
        uv = ground.world_to_pixel(fx, fy, BED_TOP)
        assert uv is not None, "test point projects behind the camera"
        return uv

    kp = np.zeros((17, 2), dtype=np.float32)
    sc = np.full(17, conf, dtype=np.float32)
    for i in _CORE:
        kp[i] = place(core_bx, core_by)
        if not core_valid:
            sc[i] = 0.05
    for i in _LEGS:
        kp[i] = place(core_bx, leg_by)
        if not legs_valid:
            sc[i] = 0.05
    for i in _ARMS:
        kp[i] = place(core_bx, arm_by)
    return PersonPose(keypoints=kp, scores=sc, score=conf, track_id=track_id)


def drive(machine, frame, ground, segments, *, track_id=1, t0=0.0):
    """Run segments [(duration_s, kwargs), ...]; collect emitted events."""
    events = []
    t = t0
    for duration, kwargs in segments:
        n = max(1, int(round(duration * FPS)))
        for _ in range(n):
            pose = make_pose(frame, ground, track_id=track_id, **kwargs)
            ev = machine.update(pose, t)
            if ev is not None:
                events.append(ev)
            t += DT
    return events, t


# --------------------------------------------------------------------------- #

def test_projection_roundtrip():
    g = make_ground()
    for pt in [(-1.0, 6.0, 0.55), (0.5, 7.0, 0.0), (1.0, 5.5, 0.55)]:
        uv = g.world_to_pixel(*pt)
        back = g.pixel_to_plane(uv[0], uv[1], pt[2])
        assert back == pytest.approx((pt[0], pt[1]), abs=1e-4)


def test_bed_frame_signed_distances():
    frame = BedFrame.from_zone(make_bed())
    # Centre: positive (inside) to both long sides, ~half width.
    d = frame.signed_distances(0.0, 0.0)
    assert d["left"] == pytest.approx(HALF_W, abs=1e-6)
    assert d["right"] == pytest.approx(HALF_W, abs=1e-6)
    # Past the left rail line: negative (outside).
    assert frame.signed_distances(0.0, HALF_W + 0.2)["left"] < 0


def test_rail_gap_at_foot_is_open():
    frame = BedFrame.from_zone(make_bed())
    half_l = frame.half_length
    # Near the head end the long side is railed; in the foot gap it is open.
    assert frame.rail_protects("left", -half_l + 0.2) is True
    assert frame.rail_protects("left", half_l - 0.1) is False  # inside the 0.48 gap


def test_arm_over_rail_does_not_trigger():
    """The whole point: an arm far past the rail, core and legs inside -> silent."""
    machine, frame, ground = make_stack()
    events, _ = drive(machine, frame, ground, [
        (1.0, dict(core_by=0.0)),                 # settle / bind
        (5.0, dict(core_by=0.0, arm_by=HALF_W + 0.35)),  # arm way outside
    ])
    assert events == []
    assert machine.state_of(1) == "IN_BED_STABLE"


def test_leg_over_emits_limb_not_exit():
    """A leg over the rail is a low signal, never a confirmed exit."""
    machine, frame, ground = make_stack(risk="high")
    events, _ = drive(machine, frame, ground, [
        (1.0, dict(core_by=0.0)),
        (3.0, dict(core_by=0.0, leg_by=HALF_W + 0.25)),  # legs out, core in
    ])
    types = [e.type for e in events]
    assert "BED_EXIT_LIMB" in types
    assert "BED_EXIT_CONFIRMED" not in types
    limb = next(e for e in events if e.type == "BED_EXIT_LIMB")
    assert limb.severity <= 1  # low-priority, by design


def test_sleeping_near_rail_does_not_trigger():
    """Lying close to a rail with no movement is not an approach."""
    machine, frame, ground = make_stack()
    events, _ = drive(machine, frame, ground, [
        (8.0, dict(core_by=0.40)),  # d_core ~0.12 the whole time; baseline matches
    ])
    assert events == []


def test_progressive_exit_confirms():
    machine, frame, ground = make_stack(risk="medium")
    segs = [(1.5, dict(core_by=0.0))]  # bind + baseline
    # Walk the core out across the rail, legs leading.
    for by in [0.33, 0.40, 0.48, 0.55, 0.62, 0.70]:
        segs.append((0.3, dict(core_by=by, leg_by=by + 0.15)))
    segs.append((0.6, dict(core_by=0.75, leg_by=0.90)))
    events, _ = drive(machine, frame, ground, segs)
    types = [e.type for e in events]
    assert "BED_EXIT_RISK" in types
    assert "BED_EXIT_CONFIRMED" in types
    assert machine.state_of(1) == "EXITED"


def test_confirmed_is_higher_severity_than_risk():
    machine, frame, ground = make_stack(risk="medium")
    segs = [(1.5, dict(core_by=0.0))]
    for by in [0.33, 0.45, 0.60, 0.75]:
        segs.append((0.4, dict(core_by=by, leg_by=by + 0.15)))
    segs.append((0.6, dict(core_by=0.80, leg_by=0.95)))
    events, _ = drive(machine, frame, ground, segs)
    risk = next(e for e in events if e.type == "BED_EXIT_RISK")
    conf = next(e for e in events if e.type == "BED_EXIT_CONFIRMED")
    assert conf.severity > risk.severity


def test_rapid_roll_off_skips_stages():
    machine, frame, ground = make_stack()
    events, _ = drive(machine, frame, ground, [
        (1.5, dict(core_by=0.0)),
        (0.8, dict(core_by=0.85, leg_by=0.85)),  # straight out
    ])
    conf = [e for e in events if e.type == "BED_EXIT_CONFIRMED"]
    assert conf, "a fast roll-off must still confirm"
    assert conf[0].evidence.get("profile") == "rapid"


def test_degraded_not_safe():
    machine, frame, ground = make_stack()
    events, _ = drive(machine, frame, ground, [
        (1.5, dict(core_by=0.0)),
        (1.0, dict(core_by=0.0, core_valid=False, legs_valid=False)),
    ])
    types = [e.type for e in events]
    assert "BED_MONITORING_DEGRADED" in types
    assert machine.state_of(1) == "DEGRADED"


def test_other_track_does_not_drive_bed():
    """A second person crossing the line leaves the bound track's machine alone."""
    machine, frame, ground = make_stack()
    # Bind track 1 in bed.
    drive(machine, frame, ground, [(1.5, dict(core_by=0.0))], track_id=1)
    # Track 2 barrels across the same bed.
    drive(machine, frame, ground, [
        (2.0, dict(core_by=0.85, leg_by=0.85)),
    ], track_id=2, t0=2.0)
    assert machine.state_of(1) == "IN_BED_STABLE"


def test_reach_and_return_aborts():
    machine, frame, ground = make_stack()
    segs = [
        (1.5, dict(core_by=0.0)),
        (0.8, dict(core_by=0.38)),  # approach -> RISK
        (1.2, dict(core_by=0.0)),   # back inside, sustained -> ABORTED
    ]
    events, _ = drive(machine, frame, ground, segs)
    types = [e.type for e in events]
    assert "BED_EXIT_RISK" in types
    assert "BED_EXIT_ABORTED" in types
    assert "BED_EXIT_CONFIRMED" not in types
    assert machine.state_of(1) == "IN_BED_STABLE"
