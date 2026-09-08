"""RTMPose top-down pose backend.

This is the backend for the ward's real problem: a person at 8.4 m.

RTMO (the other backend) is one-stage -- it resizes the whole 1920x1080 frame
to 640x640 before doing anything, a 0.33x shrink that turns a 280-pixel-tall
distant person into ~94 pixels. Detecting a small person is easy; estimating
their *pose* at 94 px is not, and that is exactly the joint precision the fall
geometry depends on.

A top-down pipeline avoids the shrink. A detector finds each person in the full
frame, and RTMPose runs on that person's *crop* at native resolution. The
distant person is upscaled to the pose model's input rather than downscaled
away. The cost is that inference now scales with the number of people, since
the pose net runs once per person -- which is why RTMO is kept as the fast
near-range and development backend, and this is the accuracy option for range.

Both emit COCO-17, so which one runs is a config line and the bake-off compares
them on identical clips. rtmlib's `Body` solution wraps the YOLOX detector and
RTMPose together, so this is a thin adapter over it.

Weights are fetched and cached by rtmlib on first use, so the first run needs
network access.
"""

from __future__ import annotations

import numpy as np

from ahfd.capture.base import Frame
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame

# rtmlib's presets trade detector/pose size against speed. 'performance' uses
# the largest models -- the right default when the whole point is range, and
# the Jetson has the headroom.
VALID_MODES = frozenset({"performance", "balanced", "lightweight"})


class RTMPoseEstimator:
    """Top-down PoseEstimator backed by rtmlib's Body solution."""

    def __init__(
        self,
        mode: str = "performance",
        device: str = "cpu",
        runtime: str = "onnxruntime",
        min_score: float = 0.3,
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                "mode must be one of " + repr(sorted(VALID_MODES)) + ", got "
                + repr(mode)
            )

        from ahfd.pose.rtmo import SUPPORTED_RUNTIMES

        if runtime not in SUPPORTED_RUNTIMES:
            raise ValueError("unknown runtime " + repr(runtime))
        if device not in SUPPORTED_RUNTIMES[runtime]:
            raise ValueError(
                "runtime " + repr(runtime) + " does not support device " + repr(device)
            )

        from rtmlib import Body

        self._mode = mode
        self._min_score = min_score
        self._model = Body(mode=mode, backend=runtime, device=device)

    @property
    def name(self) -> str:
        return "rtmpose-" + self._mode

    def estimate(self, frame: Frame) -> PoseFrame:
        if frame.bgr is None:
            raise ValueError("RTMPose needs a colour frame")

        keypoints, scores = self._model(frame.bgr)
        height, width = frame.bgr.shape[:2]

        people: list[PersonPose] = []
        for kp, sc in zip(np.asarray(keypoints), np.asarray(scores)):
            kp = np.asarray(kp, dtype=np.float32).reshape(-1, 2)
            sc = np.asarray(sc, dtype=np.float32).reshape(-1)
            if kp.shape[0] != NUM_KEYPOINTS:
                continue
            confident = sc[sc >= self._min_score]
            person_score = (
                float(confident.mean()) if confident.size else float(sc.mean())
            )
            people.append(PersonPose(keypoints=kp, scores=sc, score=person_score))

        return PoseFrame(
            t=frame.t,
            index=frame.index,
            width=width,
            height=height,
            people=tuple(people),
        )
