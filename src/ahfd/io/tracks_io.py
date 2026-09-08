"""Read and write `tracks.jsonl` -- keypoints per frame, no imagery.

This format is the pivot of the tuning workflow. Pose is the slow part of the
pipeline (8-45 fps), and threshold tuning re-runs *detection* dozens of times.
Running pose once into this file, then replaying it, turns each tuning
iteration from minutes into milliseconds -- and the replay is deterministic,
so a threshold sweep is reproducible rather than at the mercy of frame timing.

Three further properties fall out of it being keypoints-only:

* **Privacy.** There are no pixels in it, so it is safe to commit, share, and
  keep. It is the same guarantee the whole system rests on, made portable.
* **Golden tests.** A small committed tracks file plus an expected event list
  catches a threshold change that silently breaks a scenario.
* **Demo safety net.** A canned clip replays as a skeleton with no camera and
  no model, so a USB fault cannot break the demo.

One object per line, one line per frame:

    {"t": 12.37, "index": 371, "width": 1920, "height": 1080,
     "people": [{"track_id": 3, "score": 0.91,
                 "keypoints": [[u, v], ...17], "scores": [c, ...17]}]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ahfd.types import PersonPose, PoseFrame


def pose_to_dict(pose: PoseFrame) -> dict:
    """One PoseFrame as a JSON-ready dict. Rounded to keep the file small."""
    return {
        "t": round(pose.t, 4),
        "index": pose.index,
        "width": pose.width,
        "height": pose.height,
        "people": [
            {
                "track_id": p.track_id,
                "score": round(float(p.score), 4),
                "keypoints": [
                    [round(float(x), 2), round(float(y), 2)] for x, y in p.keypoints
                ],
                "scores": [round(float(s), 4) for s in p.scores],
            }
            for p in pose.people
        ],
    }


def dict_to_pose(data: dict) -> PoseFrame:
    people = tuple(
        PersonPose(
            keypoints=np.asarray(p["keypoints"], dtype=np.float32).reshape(-1, 2),
            scores=np.asarray(p["scores"], dtype=np.float32).reshape(-1),
            score=float(p["score"]),
            track_id=p["track_id"],
        )
        for p in data.get("people", [])
    )
    return PoseFrame(
        t=float(data["t"]),
        index=int(data["index"]),
        width=int(data["width"]),
        height=int(data["height"]),
        people=people,
    )


class TracksWriter:
    """Append PoseFrames to a tracks.jsonl, one line each.

    Flushes per frame: an extraction killed halfway should leave a usable
    prefix, not an empty file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._n = 0

    def write(self, pose: PoseFrame) -> None:
        self._file.write(json.dumps(pose_to_dict(pose), separators=(",", ":")) + "\n")
        self._file.flush()
        self._n += 1

    @property
    def count(self) -> int:
        return self._n

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TracksWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_tracks(path: str | Path) -> Iterator[PoseFrame]:
    """Yield PoseFrames from a tracks.jsonl, in file order.

    A generator so a long recording streams rather than loading whole.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield dict_to_pose(json.loads(line))


def write_tracks(path: str | Path, poses: Iterable[PoseFrame]) -> int:
    """Write an iterable of PoseFrames; return how many were written."""
    with TracksWriter(path) as writer:
        for pose in poses:
            writer.write(pose)
        return writer.count


def tracks_meta(path: str | Path) -> tuple[int, int] | None:
    """(width, height) from the first frame, for the calibration check.

    None if the file is empty. Cheap: reads a single line.
    """
    for pose in read_tracks(path):
        return (pose.width, pose.height)
    return None
