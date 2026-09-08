"""The pose estimator contract.

This interface is the whole point of the pose bake-off. RTMO, YOLO-pose and
RTMPose each implement it, so choosing between them is a config line rather
than a rewrite -- and the licensing decision stays reversible, which matters
because Ultralytics is AGPL-3.0 and that is a genuine blocker for anything the
hospital might deploy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ahfd.capture.base import Frame
from ahfd.types import PoseFrame


@runtime_checkable
class PoseEstimator(Protocol):
    """Turns a Frame into coordinates, and keeps no pixels."""

    @property
    def name(self) -> str:
        """Identifier used in benchmark reports, e.g. 'rtmo-s'."""
        ...

    def estimate(self, frame: Frame) -> PoseFrame: ...


def build_estimator(cfg) -> PoseEstimator:
    """Construct the estimator named by `cfg.backend`.

    Imports are deferred so that a missing optional backend (torch, for the
    YOLO branch) does not break the default install.
    """
    backend = cfg.backend.lower()

    if backend == "rtmo":
        from ahfd.pose.rtmo import RTMOEstimator

        return RTMOEstimator(
            model_size=cfg.model_size,
            model_input_size=tuple(cfg.model_input_size),
            device=cfg.device,
            runtime=cfg.runtime,
            min_score=cfg.min_score,
        )

    if backend == "rtmpose":
        from ahfd.pose.rtmpose import RTMPoseEstimator

        return RTMPoseEstimator(
            mode=cfg.mode,
            device=cfg.device,
            runtime=cfg.runtime,
            min_score=cfg.min_score,
        )

    raise ValueError(
        "unknown pose backend: "
        + repr(cfg.backend)
        + " (available: rtmo, rtmpose)"
    )
