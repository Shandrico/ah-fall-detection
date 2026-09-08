"""tracks.jsonl round-trip tests.

The file is the pivot of the tuning workflow, so a value silently changing on
the way to disk and back would corrupt every downstream number. These pin the
round-trip to the precision the format rounds to.
"""

from __future__ import annotations

import numpy as np
import pytest

from ahfd.io import TracksWriter, read_tracks, tracks_meta, write_tracks
from ahfd.io.tracks_io import dict_to_pose, pose_to_dict
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame


def make_pose(t: float, index: int, n_people: int = 1) -> PoseFrame:
    people = []
    for pid in range(n_people):
        kp = np.random.default_rng(index * 10 + pid).uniform(
            0, 640, size=(NUM_KEYPOINTS, 2)
        ).astype(np.float32)
        sc = np.random.default_rng(pid).uniform(0.3, 1.0, size=NUM_KEYPOINTS).astype(
            np.float32
        )
        people.append(
            PersonPose(keypoints=kp, scores=sc, score=float(sc.mean()), track_id=pid)
        )
    return PoseFrame(t=t, index=index, width=640, height=480, people=tuple(people))


class TestRoundTrip:
    def test_single_frame_survives(self):
        pose = make_pose(1.5, 3)
        back = dict_to_pose(pose_to_dict(pose))
        assert back.t == pytest.approx(1.5)
        assert back.index == 3
        assert (back.width, back.height) == (640, 480)
        assert len(back.people) == 1

    def test_keypoints_survive_to_rounding_precision(self):
        pose = make_pose(0.0, 0)
        back = dict_to_pose(pose_to_dict(pose))
        # Format rounds coordinates to 2 dp; assert within that.
        np.testing.assert_allclose(
            back.people[0].keypoints, pose.people[0].keypoints, atol=0.01
        )

    def test_track_id_survives(self):
        pose = make_pose(0.0, 0, n_people=3)
        back = dict_to_pose(pose_to_dict(pose))
        assert [p.track_id for p in back.people] == [0, 1, 2]

    def test_empty_frame_survives(self):
        pose = PoseFrame(t=2.0, index=5, width=640, height=480, people=())
        back = dict_to_pose(pose_to_dict(pose))
        assert len(back.people) == 0
        assert back.t == 2.0

    def test_none_track_id_survives(self):
        """An untracked detection round-trips as null, not 0."""
        p = PersonPose(
            keypoints=np.zeros((NUM_KEYPOINTS, 2), np.float32),
            scores=np.ones(NUM_KEYPOINTS, np.float32),
            score=1.0,
            track_id=None,
        )
        pose = PoseFrame(t=0.0, index=0, width=640, height=480, people=(p,))
        back = dict_to_pose(pose_to_dict(pose))
        assert back.people[0].track_id is None


class TestFileIO:
    def test_write_then_read(self, tmp_path):
        path = tmp_path / "tracks.jsonl"
        poses = [make_pose(i / 15.0, i, n_people=2) for i in range(20)]
        n = write_tracks(path, poses)
        assert n == 20

        loaded = list(read_tracks(path))
        assert len(loaded) == 20
        assert [p.index for p in loaded] == list(range(20))
        assert all(len(p.people) == 2 for p in loaded)

    def test_writer_flushes_each_frame(self, tmp_path):
        """A half-written extraction should leave a usable prefix."""
        path = tmp_path / "tracks.jsonl"
        writer = TracksWriter(path)
        writer.write(make_pose(0.0, 0))
        writer.write(make_pose(1.0, 1))
        # Not closed yet -- but flushed, so readable.
        assert len(list(read_tracks(path))) == 2
        writer.close()

    def test_tracks_meta_reads_resolution_from_first_frame(self, tmp_path):
        path = tmp_path / "tracks.jsonl"
        write_tracks(path, [make_pose(0.0, 0)])
        assert tracks_meta(path) == (640, 480)

    def test_tracks_meta_on_empty_file_is_none(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert tracks_meta(path) is None

    def test_no_imagery_in_the_file(self, tmp_path):
        """The privacy property, checked at the format level: the serialised
        form is numbers only -- there is nowhere for a pixel to hide."""
        path = tmp_path / "tracks.jsonl"
        write_tracks(path, [make_pose(0.0, 0)])
        text = path.read_text()
        for banned in ("bgr", "image", "rgb", "pixels", "frame_data"):
            assert banned not in text
