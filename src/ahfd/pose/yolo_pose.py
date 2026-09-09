"""Ultralytics YOLO-pose backend -- for the bake-off only.

This exists so the pose bake-off (`ahfd bench`) can put YOLO-pose next to RTMO
and RTMPose on identical frames and let the numbers decide. It is **not** a
deployment option: Ultralytics is AGPL-3.0, and that licence extends to trained
weights, so anything the hospital ships cannot depend on it. RTMO / RTMPose
(Apache-2.0) remain the deployable backends. Benchmarking against YOLO is a fair
and useful comparison; deploying it is the thing to avoid.

Like the other backends it is one-stage and multi-person, emits COCO-17, and
does its own detection per frame; tracking stays with the pipeline's tracker.

`ultralytics` (and the torch it pulls in) is an optional dependency, imported
lazily and only here, so the default install stays lean.
"""

from __future__ import annotations

import numpy as np

from ahfd.capture.base import Frame
from ahfd.types import NUM_KEYPOINTS, PersonPose, PoseFrame

# Model-size letters for the YOLO11 pose family, smallest to largest.
VALID_SIZES = ("n", "s", "m", "l", "x")


def _to_people(
    keypoints: np.ndarray, person_scores: np.ndarray | None, min_score: float
) -> tuple[PersonPose, ...]:
    """Convert a YOLO pose result to PersonPose objects.

    `keypoints` is (N, 17, 3) as (x, y, confidence); `person_scores` is (N,) box
    confidence, or None (then the mean joint confidence stands in). Factored out
    of the estimator so the conversion is unit-testable without torch or a model.
    """
    people: list[PersonPose] = []
    kp_all = np.asarray(keypoints, dtype=np.float32)
    if kp_all.ndim != 3 or kp_all.shape[1] != NUM_KEYPOINTS:
        return ()

    for i in range(kp_all.shape[0]):
        kp = kp_all[i, :, :2].astype(np.float32)
        sc = kp_all[i, :, 2].astype(np.float32)
        if person_scores is not None and i < len(person_scores):
            person_score = float(person_scores[i])
        else:
            confident = sc[sc >= min_score]
            person_score = float(confident.mean()) if confident.size else float(sc.mean())
        people.append(PersonPose(keypoints=kp, scores=sc, score=person_score))
    return tuple(people)


class YOLOPoseEstimator:
    """PoseEstimator backed by an Ultralytics YOLO pose model."""

    def __init__(self, model_size: str = "s", device: str = "cpu", min_score: float = 0.3):
        # Accept a bare size letter or a full model filename.
        if model_size in VALID_SIZES:
            model_name = "yolo11" + model_size + "-pose.pt"
        elif model_size.endswith(".pt"):
            model_name = model_size
        else:
            raise ValueError(
                "model_size must be one of " + repr(VALID_SIZES)
                + " or a .pt filename, got " + repr(model_size)
            )

        from ultralytics import YOLO

        # Ultralytics runs on torch, so it uses a torch device, not the
        # onnxruntime/openvino runtimes the other backends use. Map our "gpu"
        # to CUDA; on this laptop that needs a CUDA torch build (the RTX), else
        # it falls back to CPU.
        self._device = {"gpu": "cuda"}.get(device, device)
        self._min_score = min_score
        self._name = model_name[:-3] if model_name.endswith(".pt") else model_name
        self._model = YOLO(model_name)

    @property
    def name(self) -> str:
        return self._name

    def estimate(self, frame: Frame) -> PoseFrame:
        if frame.bgr is None:
            raise ValueError("YOLO-pose needs a colour frame")

        results = self._model.predict(frame.bgr, device=self._device, verbose=False)
        height, width = frame.bgr.shape[:2]

        people: tuple[PersonPose, ...] = ()
        if results:
            r = results[0]
            kp_obj = getattr(r, "keypoints", None)
            data = getattr(kp_obj, "data", None) if kp_obj is not None else None
            if data is not None and len(data) > 0:
                kps = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                boxes = getattr(r, "boxes", None)
                conf = getattr(boxes, "conf", None) if boxes is not None else None
                scores = (
                    conf.cpu().numpy() if (conf is not None and hasattr(conf, "cpu"))
                    else (np.asarray(conf) if conf is not None else None)
                )
                people = _to_people(kps, scores, self._min_score)

        return PoseFrame(
            t=frame.t, index=frame.index, width=width, height=height, people=people
        )
