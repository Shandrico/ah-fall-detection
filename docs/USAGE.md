# Usage guide

How to run everything. For the data-tuning loop see [TUNING.md](TUNING.md); for
recording and evaluating clips see [CLIPS.md](CLIPS.md).

## 0. Every session: activate the environment

Open a terminal in the project folder and run this **once per terminal**:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt then shows `(.venv)`. If `ahfd` ever says "not recognized", you
opened a fresh terminal and forgot this step. git commands don't need it.

Check what's installed and whether a camera is present:

```powershell
ahfd info
```

## 1. The three things you choose when running

Almost every command is "pick a **source**, a **backend**, and a **device**."

### Source — where frames come from (`--source`)

| Source | Use |
|---|---|
| `webcam://0` | laptop webcam (default) |
| `rs://` | live RealSense D435i |
| `file://clip.mp4` | a recorded video |
| `seq://path/to/frames/` | a folder of images (public datasets) |
| `bag://recording.bag` | a recorded RealSense `.bag` |

These same URIs are what the dashboard's camera picker offers (§5).

### Backend — which pose model (`--backend`)

| Backend | What it is | When |
|---|---|---|
| `rtmo` | one-stage, fast (default) | near range, dev, the demo |
| `rtmpose` | top-down, better at distance | the far bed (~8 m) |
| `yolo` | Ultralytics YOLO-pose | **benchmark comparison only** (AGPL — don't deploy) |

Switch without editing anything: `--backend rtmo` / `rtmpose` / `yolo`.

### Device / runtime — what it runs on

| Goal | Flags | Notes |
|---|---|---|
| Portable default | `--runtime onnxruntime --device cpu` | works everywhere |
| Fast on this laptop (RTMO/RTMPose) | `--runtime openvino --device gpu` | Intel iGPU, ~60 fps, no CUDA needed |
| NVIDIA GPU | `--device cuda` | needs CUDA setup; the Jetson's path |

`gpu` = Intel iGPU (OpenVINO only). `cuda` = NVIDIA. YOLO runs on torch, so it
only does `cpu` or `cuda`, never the Intel iGPU.

## 2. Just look at the camera (no detection)

```powershell
# webcam, skeleton on black
ahfd run

# webcam, skeleton drawn on the video
ahfd run --view overlay

# RealSense, RGB + skeleton
ahfd run --source rs:// --view overlay

# try a different model
ahfd run --source rs:// --view overlay --backend rtmpose
```

No calibration needed for viewing — detection is off unless a config turns it
on. Press `q` in the window to quit.

## 3. Calibrate a camera (needed before fall detection)

Detection works in metres, so it needs to know the camera's height, tilt and
lens. Calibration measures those once per mounting position.

### Step 1 — aim the mount (RealSense only)

```powershell
ahfd level --source rs://
```

Live pitch/roll from the IMU. Tilt the camera until **pitch ≈ 20°** (your ward
downtilt) and **roll ≈ 0** (level). Ctrl+C to stop. Writes nothing.

### Step 2 — write the calibration

```powershell
# RealSense: tilt comes from the IMU, intrinsics from the device.
# You only supply the measured lens height (tape-measure it).
ahfd calibrate calib/ward.yaml --source rs:// --height 2.6

# Webcam / no IMU: give the angle and FOV yourself.
ahfd calibrate calib/desk.yaml --source webcam://0 --height 1.0 --pitch 15
```

`--height` is in metres and is the one thing no sensor gives you — measure it.
A measured `--pitch` overrides the IMU if you pass it.

### Step 3 — add bed zones

Open the calibration file and add the beds (floor coordinates in metres — X
along the wall, Y away from the camera, origin under the lens). See
`calib/example_ward6.yaml` for the format:

```yaml
zones:
  - name: bed_1
    kind: bed
    top_m: 0.55         # bed surface height above the floor (measure it)
    risk_level: high    # none | low | medium | high  (drives bed-exit urgency)
    polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
```

(Hand-writing metre coordinates is fiddly — a click-to-draw zone tool is
planned. Until then, derive corners from the floor plan or `example_ward6.yaml`.)

### Step 4 — sanity check

Run with the overlay and the metrics HUD on:

```powershell
ahfd run --source rs:// --calibration calib/ward.yaml --config configs/ward.yaml --view overlay
```

Walk around: a standing person's **ankle height should read ~0.05 m** (shown in
the corner) and walking toward the camera must stay `UPRIGHT`, not flip to
lying. If ankle height drifts, the calibration is off — recheck height/tilt.

## 4. Run fall detection

```powershell
ahfd run --source rs:// --calibration calib/ward.yaml --config configs/ward.yaml --view overlay
```

- `--config configs/ward.yaml` turns detection on and supplies the thresholds.
- `--calibration` must match the source resolution (the app refuses a mismatch).
- Alerts print to the console and, if `alert.jsonl_path` is set, to a log file.

## 5. The nurse dashboard

```powershell
ahfd dashboard --config configs/dashboard_dev.yaml
```

Then open **http://127.0.0.1:8000** in a browser (it opens no window itself).
Skeleton-only by default; add `--rgb` for live video (reverses the privacy
stance — needs sign-off).

### Changing the camera and the model from the page

`--source` and `--backend` set the **initial** camera and model. Both can then
be changed from the control bar at the top of the page, without restarting the
process — useful for the pose bake-off, and for a nurse who needs the other
camera and doesn't own the terminal.

The bar shows a status pill and, on the right, what is actually running
(`rtmo-s · 1280x720 · skeleton only`). While a switch is in flight the pill
reads `switching`/`starting` and the feed dims — a frozen last frame from the
old camera otherwise looks exactly like a live one. **The first switch to a
backend you have not used before downloads its weights (~35 MB), so `starting`
can persist for a while**; the hint line says so.

Notes:

- The camera list comes from `dashboard.sources` in the config. A connected
  RealSense is appended automatically. Webcam indices are deliberately *not*
  probed — scanning them is slow on Windows and can grab a device another
  program is using.
- The free-text box takes a full URI (`file://clip.mp4`, `seq://frames/`). A
  bare path is refused: it is indistinguishable from a typo.
- **RGB is one-way from the browser.** The button turns RGB *off* at any time;
  turning it *on* requires the process to have been started with `--rgb` (or
  `dashboard.show_rgb: true`), i.e. by someone who saw the AH/DPO warning.
- A failed switch (busy camera, bad URI, unknown backend) leaves the running
  pipeline alone and puts the reason in the hint line.

```yaml
dashboard:
  sources:
    - label: "Ward 6 -- bay A"
      uri: "rs://"
    - label: "Laptop webcam"
      uri: "webcam://0"
  backends: ["rtmo", "rtmpose"]   # empty offers all three
  allow_custom_source: false      # no free-text box on a ward
  switch_timeout_s: 5.0           # how long to wait for a camera to release
```

The page is unauthenticated by design and binds to localhost. If you set
`dashboard.host: "0.0.0.0"`, also set `allow_custom_source: false` — otherwise
anyone on the LAN can point the pipeline at any file on the box.

## 6. Benchmark the pose models

Compare all three on identical frames — this is how you decide which is best:

```powershell
# fair comparison: all on the same device (CPU is the common denominator)
ahfd bench --backends "rtmo,rtmpose,yolo" --device cpu --runtime onnxruntime
```

Prints speed (median ms, fps) and detections per frame for each. Accuracy
comparison needs labelled clips — see below.

## 7. Tune the thresholds (with data)

Summary: extract clips to keypoints once, then sweep thresholds fast.

```powershell
ahfd extract file://clips/fall_01.mp4 data/tracks/fall_01.jsonl
ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 --tracks data/tracks/ --annotations data/annotations/ --calibration calib/ward.yaml
ahfd eval data/annotations/ data/events/ --out eval/report.md
```

Full recipe, the tune/test split, and which knob fixes what: **[TUNING.md](TUNING.md)**
and **[CLIPS.md](CLIPS.md)**.

## Command reference

| Command | Purpose |
|---|---|
| `ahfd info` | versions + whether a camera is detected |
| `ahfd run` | live pipeline: capture → pose → (detect) → view |
| `ahfd level` | live IMU angle, for aiming the mount |
| `ahfd calibrate` | write a per-camera calibration |
| `ahfd dashboard` | nurse web dashboard |
| `ahfd bench` | pose backend bake-off |
| `ahfd extract` | clip → keypoint `tracks.jsonl` |
| `ahfd replay` | replay tracks through detection (fast, no camera) |
| `ahfd sweep` | tune one threshold against labelled clips |
| `ahfd eval` | score events vs ground truth |
| `ahfd record` | record raw video (consented staged sessions only) |

Add `--help` to any command for its options.

## Quick recipes

```powershell
# See the webcam
ahfd run --view overlay

# See the RealSense
ahfd run --source rs:// --view overlay

# Aim the RealSense mount
ahfd level --source rs://

# Calibrate the RealSense (after measuring height)
ahfd calibrate calib/ward.yaml --source rs:// --height 2.6

# Fall detection on the RealSense
ahfd run --source rs:// --calibration calib/ward.yaml --config configs/ward.yaml --view overlay

# Compare pose models
ahfd bench --backends "rtmo,rtmpose,yolo" --device cpu --runtime onnxruntime

# Run detection with a specific model
ahfd run --source rs:// --calibration calib/ward.yaml --config configs/ward.yaml --backend rtmpose
```
