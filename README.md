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

### Views and the dashboard

- `ahfd run` — skeleton on black (default, privacy-safe).
- `ahfd run --view overlay` — skeleton on the live video, for debugging pose
  quality. Displays RGB but never stores it.
- `ahfd dashboard` — a nurse-facing web page: metrics row (fps, people in view,
  open alerts, confirmed falls, bed exits, uptime), a live view with fullscreen,
  a triage queue of open alerts with acknowledge, per-person state chips, a
  severity-filtered event log with evidence, and an alert sound. Standard-library
  server, one pipeline thread, frames encoded once and shared — built this way on
  purpose to avoid the thread-leak that overheated an earlier FastAPI version.
  **Skeleton-only by default**; `--rgb` (or `dashboard.show_rgb`) shows video,
  which reverses the ward privacy stance and needs AH/DPO sign-off. Binds to
  localhost only by default.

  The camera and the pose model can be **changed from the page** — the picker
  is driven by `dashboard.sources`, plus a RealSense if one is attached. A
  switch swaps in a fresh pipeline; there is still exactly one. The RGB toggle
  is one-way: the page can turn video *off* at any time, but turning it *on*
  needs a process that was started with `--rgb`.

  It serves a web page — it opens **no window**. Run it, then open the printed
  `http://127.0.0.1:8000` in a browser. For a sharp 720p RGB view:
  `ahfd dashboard --config configs/dashboard_dev.yaml` (capture resolution and
  JPEG quality are set there; a bare webcam otherwise defaults to a soft 640×480).

## Status

Detecting falls end to end on a plain webcam — no depth camera required:

```
capture → pose → tracking → One-Euro smoothing
        → ground-plane geometry → metric features → fall state machine → alerts
                                                                → evaluation
```

Camera sources: webcam, video, image sequences (`seq://`, for public datasets),
plus live RealSense (`rs://`) and deterministic `.bag` replay — the RealSense
paths are written but only runnable once the D435i connects. Two pose backends
(RTMO, RTMPose) behind one interface, with a bake-off harness. A keypoint-only
`tracks.jsonl` extract/replay path for fast, deterministic threshold tuning, and
event-level evaluation with false-alarms-per-hour.

Remaining: real accuracy numbers (needs staged clips — see [docs/CLIPS.md](docs/CLIPS.md))
and depth-based calibration refinements.

## Recording and tuning against clips

The data-driven workflow — record, extract keypoints once, then tune and score
on the fast keypoint replay — is in **[docs/CLIPS.md](docs/CLIPS.md)**:

```bash
ahfd extract file://fall_01.mp4 data/tracks/fall_01.jsonl   # pose, once
ahfd replay data/tracks/fall_01.jsonl --calibration calib/ward6.yaml --view skeleton
ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 \
    --tracks data/tracks/ --annotations data/annotations/ --calibration calib/ward6.yaml
ahfd eval data/annotations/ data/events/ --out eval/report.md
```

`tracks.jsonl` is keypoints only — no imagery — so it is fast to replay, safe to
keep, and the basis of the committed golden regression test.

## Quick start

```bash
uv venv --python 3.10 .venv
uv pip install -e .
uv pip install openvino          # optional: ~3.3x faster on an Intel iGPU

ahfd info                        # versions + whether a RealSense is present
ahfd run                         # webcam -> skeleton on black; q to quit
ahfd run --view overlay          # skeleton on live video + live metric readout
ahfd calibrate cal.yaml --source rs:// --height 2.6   # calibrate a real camera
ahfd run --config configs/detect_dev.yaml   # full pipeline, detection on
ahfd dashboard --config configs/detect_dev.yaml   # nurse web dashboard
ahfd bench                       # pose backend bake-off (RTMO vs RTMPose)
ahfd eval <annotations/> <events/>   # recall, false alarms/hour, latency
pytest                           # 364 tests, no camera needed
```

The first `run` downloads pose weights (cached afterwards).

**Full how-to** — sources (webcam/RealSense), backends (RTMO/RTMPose/YOLO),
devices, calibration, tuning: **[docs/USAGE.md](docs/USAGE.md)**.

## Two pose backends

| Backend | How | Speed (iGPU, ~5 people) | Best for |
|---|---|---|---|
| RTMO | one-stage, whole frame → 640×640 | 22 ms / 45 fps | near range, dev, the demo |
| RTMPose | top-down: detect, then pose per crop | 129 ms / 8 fps | the far bed (8.4 m) |

RTMO shrinks the whole 1080p frame to 640×640, which turns a 280-pixel person at
8.4 m into ~94 px and loses the joint precision the geometry needs. RTMPose crops
each person and runs pose at native scale, at the cost of scaling with headcount.
Both emit COCO-17, so the choice is a config line and `ahfd bench` compares them
on identical frames.

## How the detection works

**A fall is a sequence, not a frame.** Treating it as a frame — "vertical speed
exceeded a threshold, therefore alert" — is the standard way these systems become
unusable, because a nurse sitting down quickly, a keypoint flicker or a tracker ID
swap each produce one bad frame.

| Phase | Test |
|---|---|
| **trigger** | torso drops fast, or drops far, quickly |
| **rest** | body is genuinely on the floor, not on a bed |
| **confirm** | stays down and still for ~8 s |

Only the third pages anybody. Something appears on screen at 1.5 s so the system
looks responsive, but the alert waits — buying a large false-alarm reduction for a
latency cost that doesn't matter clinically.

A second path catches what impact detection cannot: a frail patient sliding slowly
to the floor produces no velocity spike at all, so a track that simply *is* down,
outside a bed and still for long enough raises `PERSON_DOWN` regardless of how it
got there.

**Everything is in metres, which is the point.** One threshold set covers every
camera in a ward. The test suite verifies the same fall is detected at 4.0, 6.0 and
7.4 m with identical thresholds — something pixel-based thresholds cannot do, since
the same fall at the far bed produces a fraction of the pixel velocity.

### Two findings worth knowing

**`floor_spread` separates upright from fallen, and does it backwards from
intuition.** Project every joint onto the floor as if it lay there. A fallen person
really is on the floor, so their projections span about a body length. A standing
person's head ray, continued to the floor, lands *metres* past their feet. Measured
on projected bodies: a prone body gives 1.64 m at both 4 m and 6 m — identical —
while an upright one gives 5.19 m at 4 m and 7.54 m at 6 m.

**Height alone is not range-independent.** The vertical-line height estimate is
biased for a horizontal body, and the bias grows as people get closer: the same
prone body reads 0.47 m at 6 m but 0.62 m at 4 m. So heights are used for *change*
(the drop), and `floor_spread` decides posture.

## Calibration vs tuning

Different things, different frequency — and the metric design is what separates them.

| | Calibration | Tuning |
|---|---|---|
| What | Camera height, tilt, intrinsics, bed zones | The thresholds |
| Per what | **Every camera, every mount position** | **Once per ward** |
| File | `calib/*.yaml` | `configs/*.yaml` |

Calibration is mandatory and refused rather than guessed: without the camera's
height and tilt there is no way to compute a height in metres, and a default would
produce confident, meaningless alerts.

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
  geometry/      ground plane, floor zones, per-camera calibration
  features/      metric features: heights, vertical velocity, floor spread
  detect/        the fall state machine and its events
  alert/         console and JSONL sinks
  viz/           skeleton-on-black renderer
  cli.py         ahfd run / ahfd info
calib/           per-camera calibration (measured, one per mount position)
configs/         thresholds and runtime profiles (per ward, not per camera)
```

## Bed-exit prediction and graded risk

The project's focus is narrowing from "detect any fall" to **predicting unsafe
bed exits** — the largest *preventable* class of ward falls — because a fall's
impact is a fraction of a second (too fast for a nurse to reach), while a bed
exit unfolds over tens of seconds and is visible in advance.

The problem this raises: alerting on *every* bed exit is useless, because many
patients are cleared to mobilise on their own. The answer is a **graded response
keyed to per-bed fall risk**, not a binary alarm:

| Bed risk | Bed-exit response |
|---|---|
| none (cleared to self-mobilise) | dashboard status, no alarm |
| low | low-priority notice |
| medium / unknown | warning |
| high (should not exit unassisted) | alert — page a nurse |

Risk attaches to the **bed** (a zone attribute, `risk_level` in the calibration),
not to the patient — so it is identity-free, and it comes from the fall-risk
assessment nurses already do on admission (Morse / Hendrich), set once per
admission. This is what keeps alarm volume tolerable and sidesteps the privacy
concern of per-patient profiling. See `BED_EXIT_SEVERITY_BY_RISK` in
[detect/state_machine.py](src/ahfd/detect/state_machine.py).

Where the ward has a **bed pressure sensor** (binary on/off-bed), it is the
authoritative bed-exit trigger and the vision layer classifies the *safety* of
the exit and detects falls the sensor cannot see — a fusion planned once the
sensor interface is known.

## Scope

**In:** fall detection + bed-exit prediction — capture, detect, graded alert.
**Out:** vitals and (separately scoped) teleconsultation. Note AH also wants
**virtual nursing** — clinicians viewing patients live — which makes the RGB
dashboard a first-class, consented use rather than a privacy compromise.
