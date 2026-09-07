# ah-fall-detection

Privacy-preserving vision-based patient fall detection for hospital wards.

CDE3301 capstone with **Alexandra Hospital** — project *Smart Ward Technologies
(Vision Systems)*, IS-307. Subsidised wards run roughly one nurse to 6–8 beds; this
system aims to augment that, not replace it.

## The privacy property

Pose estimation runs on RGB **in memory**, and only joint coordinates are ever
persisted. No video, no stills, no faces.

That claim is enforced rather than promised:

| Layer | Mechanism |
|---|---|
| Structural | [`types.py`](src/ahfd/types.py) has **no image field**. Downstream of the pose stage there is nowhere to put a frame. |
| Build-time | [`test_privacy.py`](tests/test_privacy.py) walks the AST of every module and fails if `imwrite` / `VideoWriter` / binary writes appear outside `ahfd.debug`. |
| Runtime | [`privacy.py`](src/ahfd/privacy.py) gates raw capture behind **three independent switches** — config, env var, and an explicit CLI acknowledgement. Any one missing denies. |
| Repo | [`.gitignore`](.gitignore) blocks every media extension, so a careless `git add -A` cannot leak footage. |

The default view renders **skeleton on black**. That is the privacy argument made
visible: what you see on screen is everything the system keeps.

## Status

Working end to end on a plain webcam — no depth camera required:

```
capture → pose (RTMO) → tracking → One-Euro smoothing → skeleton render
```

Not built yet: depth/floor calibration, bed zones, the fall state machine, alerting,
and the evaluation harness. See the plan for sequencing.

## Quick start

```bash
uv venv --python 3.10 .venv
.venv/Scripts/python.exe -m pip install -e .   # or: uv pip install -e .

ahfd info                        # versions + whether a RealSense is present
ahfd run                         # webcam -> skeleton on black; q to quit
ahfd run --source file://clip.mp4
pytest                           # 68 tests, no camera needed
```

The first `run` downloads RTMO weights (cached afterwards).

## Design decisions

**Pose model: RTMO via [`rtmlib`](https://github.com/Tau-J/rtmlib).** Three reasons:

1. **Apache-2.0.** Ultralytics YOLO-pose is AGPL-3.0, and those terms extend to
   *trained weights* — a genuine blocker for anything the hospital might deploy.
2. **Flat multi-person cost.** RTMO is one-stage, so inference time barely moves from
   1 to 10 people. A cubicle holds patients plus staff plus visitors; a top-down model
   re-runs the pose network per person.
3. **`rtmlib` sidesteps MMPose's decay.** MMPose the framework is effectively dormant,
   but `rtmlib` needs only numpy/opencv/onnxruntime and has a TensorRT backend. We take
   the weights, not the framework.

Pose sits behind a [`PoseEstimator`](src/ahfd/pose/base.py) protocol so the model is a
config line — the planned `pose/rtmo`, `pose/yolo`, `pose/rtmpose` branches share one
benchmark.

**Python 3.10, no conda.** JetPack ships TensorRT/CUDA bound to the *system* Python
3.10; a conda env cannot see those bindings. Dev uses `uv` + venv, deployment uses
system Python + `venv --system-site-packages`. Same tooling both ends.

## Hardware

Intel **RealSense D435i** + NVIDIA **Jetson Orin NX 16 GB**.

Notes that cost time if missed:

- **RGB FOV (~69°×42°) is much narrower than depth FOV (~87°×58°).** Pose runs on RGB,
  so RGB is the binding constraint: ~4.1 m width at 3 m range. **One camera cannot cover
  6–8 beds** — budget one per 1–2 beds.
- **USB 3 is mandatory.** Depth + colour at 30 fps needs ~345 Mbps; a USB 2 link
  (480 Mbps nominal) does not fit, and librealsense *silently* drops stream profiles
  rather than erroring. No passive extension cables. `ahfd info` warns if the link
  negotiated USB 2.
- **Orin NX has no eMMC** — an NVMe SSD is required.
- **Check carrier input voltage** before powering: the Orin dev-kit carrier expects 19 V.
- Colour is rolling shutter; depth and IR are global shutter.
- Night is the biggest open functional gap — RGB pose degrades in a dimmed ward, and the
  IR stream is the likely answer but is unvalidated.

## Layout

```
src/ahfd/
  types.py       coordinate-only data model (no image field, by design)
  privacy.py     triple-switch raw capture gate
  config.py      YAML config; no threshold is hard-coded
  capture/       Frame + FrameSource; webcam and video today, rs:// and bag:// to come
  pose/          PoseEstimator protocol, RTMO backend, COCO-17 skeleton, One-Euro smoothing
  track/         greedy IoU tracker (placeholder for ByteTrack)
  viz/           skeleton-on-black renderer
  cli.py         ahfd run / ahfd info
```

## Scope

**In:** fall detection core — capture, detect, alert stub.
**Out:** vitals and teleconsultation. They appear in the AH brief, but a later team
would attach them at the `alert/` sinks.
