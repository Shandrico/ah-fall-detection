"""RTMO pose backend.

RTMO is a one-stage multi-person model, which is the reason it was chosen for a
ward: its inference cost is essentially flat in the number of people, whereas a
top-down model re-runs the pose network per person. A cubicle routinely holds
several patients plus staff plus visitors, so flat scaling is worth more here
than a point of COCO accuracy.

It is also Apache-2.0, unlike Ultralytics, whose AGPL-3.0 terms extend to
trained weights.

Weights are fetched and cached by rtmlib on first use, so the first run needs
network access.
"""

from __future__ import annotations

import numpy as np

from ahfd.capture.base import Frame
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"

# Trained on body7. 's' is the sensible CPU default; 'm'/'l' are for the Jetson
# once TensorRT is in play.
MODEL_URLS: dict[str, str] = {
    "s": _BASE + "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
    "m": _BASE + "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "l": _BASE + "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
}

# These exported ONNX graphs have a STATIC input shape. Handing rtmlib a
# different model_input_size does not resize anything -- it sails past every
# check and dies inside onnxruntime with "Got invalid dimensions for input",
# which is a miserable error to debug. So refuse it here, loudly, with the
# reason. Lowering resolution to buy speed means re-exporting the model, not
# changing a config value.
FIXED_INPUT_SIZE = (640, 640)

# Runtime/device combinations rtmlib actually knows how to build. Checked here
# so a typo fails at construction with a readable message instead of surfacing
# as a silent fall back to CPU -- which looks like "the model is just slow".
#
# Practical guidance:
#   onnxruntime + cpu    portable baseline, ~200 ms/frame on a laptop CPU
#   onnxruntime + cuda   the Jetson path, and also an RTX laptop -- same code
#   openvino    + gpu    Intel iGPU / Arc
#   openvino    + npu    Intel Core Ultra NPU
SUPPORTED_RUNTIMES: dict[str, frozenset[str]] = {
    "onnxruntime": frozenset({"cpu", "cuda", "rocm", "mps"}),
    "openvino": frozenset({"cpu", "gpu", "npu"}),
    "opencv": frozenset({"cpu", "cuda"}),
}


class RTMOEstimator:
    """PoseEstimator backed by rtmlib's RTMO."""

    def __init__(
        self,
        model_size: str = "s",
        model_input_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        runtime: str = "onnxruntime",
        min_score: float = 0.3,
    ):
        if model_size not in MODEL_URLS:
            raise ValueError(
                "model_size must be one of "
                + repr(sorted(MODEL_URLS))
                + ", got "
                + repr(model_size)
            )

        if tuple(model_input_size) != FIXED_INPUT_SIZE:
            raise ValueError(
                "the published RTMO ONNX models have a static "
                + str(FIXED_INPUT_SIZE[0])
                + "x"
                + str(FIXED_INPUT_SIZE[1])
                + " input; got "
                + repr(tuple(model_input_size))
                + ". Re-export the model to change it -- setting this value "
                "alone fails inside onnxruntime."
            )

        if runtime not in SUPPORTED_RUNTIMES:
            raise ValueError(
                "unknown runtime "
                + repr(runtime)
                + "; expected one of "
                + repr(sorted(SUPPORTED_RUNTIMES))
            )
        if device not in SUPPORTED_RUNTIMES[runtime]:
            raise ValueError(
                "runtime "
                + repr(runtime)
                + " does not support device "
                + repr(device)
                + "; expected one of "
                + repr(sorted(SUPPORTED_RUNTIMES[runtime]))
            )

        from rtmlib import RTMO

        self._model_size = model_size
        self._min_score = min_score
        self._model = RTMO(
            onnx_model=MODEL_URLS[model_size],
            model_input_size=model_input_size,
            backend=runtime,
            device=device,
            score_thr=min_score,
        )

    @property
    def name(self) -> str:
        return "rtmo-" + self._model_size

    def estimate(self, frame: Frame) -> PoseFrame:
        if frame.bgr is None:
            raise ValueError("RTMO needs a colour frame")

        keypoints, scores = self._model(frame.bgr)
        height, width = frame.bgr.shape[:2]

        people: list[PersonPose] = []
        for kp, sc in zip(np.asarray(keypoints), np.asarray(scores)):
            kp = np.asarray(kp, dtype=np.float32).reshape(-1, 2)
            sc = np.asarray(sc, dtype=np.float32).reshape(-1)
            if kp.shape[0] != NUM_KEYPOINTS:
                continue

            # Person-level confidence: the mean over joints the model is at all
            # sure about. A plain mean is dragged down by occluded limbs, which
            # would discard people who are partly behind a bed -- exactly the
            # cases that matter most.
            confident = sc[sc >= self._min_score]
            person_score = float(confident.mean()) if confident.size else float(sc.mean())

            people.append(PersonPose(keypoints=kp, scores=sc, score=person_score))

        return PoseFrame(
            t=frame.t,
            index=frame.index,
            width=width,
            height=height,
            people=tuple(people),
        )
