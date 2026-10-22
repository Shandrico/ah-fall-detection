"""Target-bed person selection for multi-occupant clips (pure logic)."""

from __future__ import annotations

from ahfd.ml import posture


class FakePerson:
    def __init__(self, track_id):
        self.track_id = track_id
        self.keypoints = None
        self.scores = None
        self.score = 0.9


class FakePose:
    def __init__(self, people):
        self.people = people


def test_prefers_locked_target_track_even_off_the_bed(monkeypatch):
    """The locked track is returned regardless of zone -- follows them after exit."""
    pose = FakePose([FakePerson(5), FakePerson(1), FakePerson(9)])
    # Nobody is in the bed zone (all walking), but the locked track still wins.
    monkeypatch.setattr(posture, "_person_bed_zone", lambda p, g, zs, ms: None)
    got = posture._select_target_person(pose, 1, object(), None, 0.3)
    assert got.track_id == 1


def test_falls_back_to_sole_person_in_target_bed(monkeypatch):
    """When the locked track is absent, the one person in the bed is chosen."""
    pose = FakePose([FakePerson(5), FakePerson(1)])
    zone = object()
    monkeypatch.setattr(
        posture, "_person_bed_zone",
        lambda p, g, zs, ms: zone if p.track_id == 5 else None,
    )
    got = posture._select_target_person(pose, 99, zone, None, 0.3)  # 99 not present
    assert got.track_id == 5


def test_drops_frame_when_ambiguous(monkeypatch):
    """Two people in the bed and no locked track -> None (never guess)."""
    pose = FakePose([FakePerson(5), FakePerson(1)])
    monkeypatch.setattr(posture, "_person_bed_zone", lambda p, g, zs, ms: object())
    assert posture._select_target_person(pose, 99, object(), None, 0.3) is None


def test_drops_frame_when_nobody_attributable(monkeypatch):
    pose = FakePose([FakePerson(5), FakePerson(1)])
    monkeypatch.setattr(posture, "_person_bed_zone", lambda p, g, zs, ms: None)
    assert posture._select_target_person(pose, 99, object(), None, 0.3) is None


def test_no_tracked_people_returns_none():
    pose = FakePose([FakePerson(None)])
    assert posture._select_target_person(pose, 1, object(), None, 0.3) is None


class _Z:
    def __init__(self, name, kind="bed"):
        self.name = name
        self.kind = kind
        self.top_m = 0.5


class _Calib:
    def __init__(self, zones):
        class _ZM:
            pass
        zm = _ZM()
        zm.zones = zones
        self.zones = zm


def test_resolve_bed_zone_matches_one_name():
    calib = _Calib([_Z("nearest (15/10)"), _Z("\tmiddle (15/10)")])
    assert posture._resolve_bed_zone(calib, "middle").name.strip().startswith("middle")
    assert posture._resolve_bed_zone(calib, "nearest").name.strip().startswith("nearest")


def test_resolve_bed_zone_none_when_no_match_or_ambiguous():
    calib = _Calib([_Z("bed_a"), _Z("bed_b")])
    assert posture._resolve_bed_zone(calib, "middle") is None       # no match
    calib2 = _Calib([_Z("middle left"), _Z("middle right")])
    assert posture._resolve_bed_zone(calib2, "middle") is None      # ambiguous
