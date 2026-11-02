# Benchmark: development clips (1 Nov 2026)

These numbers come from the committed code at `b6de1e8` plus the untracked `detect/preimpact.py`, run with `configs/dashboard_jetson.yaml`.

Only development clips were used:
- pilot takes 01-08;
- the 27 Oct students s1-s4;
- the 29 Oct fall student 01 (`f*_01`, `neg_f6_01`, `neg_f7_01`).

The staff clips, fall student 02 and pilot 09-10 were never read. They remain the locked tests.

These are staged falls by one healthy student on crash mats, not patient falls. The phase and fall labels are in whole seconds, so every time below is good to about ±1 s.

Scripts:
- `~/.ahfd-work/bench_falls_20261101.py`
- `~/.ahfd-work/bench_bedexit_20261101.py`

Each writes its output and a `.results.json` next to the script.

## 1. Fall detection

Clips:
- 5 falls per camera: f1, f3, f4, f5 (f5 contains 2 falls).
- `neg_f6` (deliberate lie-down) is reported separately.
- No-fall footage: about 26 min per camera (pilot 01-08, students s1-s4, kneel clip neg_f7).

Scoring:
- **Time** is the first alert relative to the labelled landing (negative = before landing).
- A fall counts as caught by an alert from 2.5 s before landing to 30 s after.
- A **false alarm** is any alert away from a labelled fall. Alerts less than 10 s apart count as one.
- The pre-impact models are scored on groups they were not trained on (leave-one-group-out cross-validation).

The alerts compared:
- **Floor alert** = FALL_SUSPECTED / FALL_CONFIRMED / PERSON_DOWN / NEAR_MISS.
- **Possible fall** = POSSIBLE_FALL (torso drops 0.15 m in 0.5 s).
- **Pre-impact** = FALL_IMMINENT from `preimpact.py`: rule = trunk rotation above 80 deg/s with the balance point outside the feet; model = boosted trees at threshold 0.9.

### D435f

| System | Falls caught | Caught before landing | Median time | neg_f6 | False alarms (per h) |
|---|---|---|---|---|---|
| Floor alert only | 5/5 | 0 | +1.25 s | +1.1 s | 6 (13.7) |
| Possible fall only | 4/5 | 4 | -0.70 s | -0.5 s | 9 (20.6) |
| **Current system** (floor + possible) | 5/5 | 4 | -0.62 s | -0.5 s | 13 (29.7) |
| Pre-impact rule only | 4/5 | 4 | -0.86 s | -0.6 s | **1 (2.3)** |
| Pre-impact model only | 4/5 | 4 | -0.83 s | -1.8 s | 4 (9.1) |
| Floor + pre-impact rule | 5/5 | 4 | -0.75 s | -0.6 s | 6 (13.7) |
| Floor + possible + pre-impact rule | 5/5 | **5** | **-0.97 s** | -0.6 s | 13 (29.7) |

### D435i

| System | Falls caught | Caught before landing | Median time | neg_f6 | False alarms (per h) |
|---|---|---|---|---|---|
| Floor alert only | 5/5 | 1 | +1.51 s | +2.6 s | **4 (9.6)** |
| Possible fall only | 4/5 | 4 | -0.69 s | -0.6 s | 25 (60.0) |
| **Current system** (floor + possible) | 5/5 | 4 | -0.44 s | -0.6 s | 25 (60.0) |
| Pre-impact rule only | 3/5 | 3 | -0.71 s | missed | 10 (24.0) |
| Pre-impact model only | 4/5 | 3 | -0.52 s | missed | 9 (21.6) |
| Floor + pre-impact model | 5/5 | 3 | -0.46 s | +2.6 s | 10 (24.0) |
| Floor + possible + pre-impact rule | 5/5 | 4 | -0.71 s | -0.6 s | 28 (67.2) |

Where the false alarms come from:
- **Floor alert:** a person lying in bed in pilot 03/06/08 on the D435f, read as on the floor (torso about 0.3 m with no bed support). This is a bug to fix. It also fires on the kneel in neg_f7, which is intended: being on the floor always alerts.
- **Possible fall:** sitting down and bending. These are the students s2/s4 sitting down from standing and the pilot bed-edge sit-downs, which account for most of the D435i's 25.
- **Pre-impact rule:** on the D435f, only the kneel. On the D435i, pilot sit-downs (the pilot's D435i geometry is poor).

## 2. Bed exit: frame-by-frame classification (accuracy and macro-F1)

These are the models from `~/.ahfd-work/train_20261031.py` (the other agent's v3 training, by viewpoint). Accuracy was added from the same cached rows by `~/.ahfd-work/bench_phase_20261101.py`; the macro-F1 values are identical to the training run.

- **Model** = trained logistic regression (7 phases) or tree (4 postures).
- **Rules** = the hand-written state machine.
- The **side** view is the D435f on the pilot and 27 Oct, and the D435i on 29 Oct.
- **Training people:** pilot volunteer, students 1-3, fall student 01.
- **Leave one person out:** each person is scored by a model trained on the other training people.
- **Staff test:** staff 01-03, never trained on. This test was already run once by the v3 training. It is reported here, not tuned on.

### 7-phase activity

The 7 phases: RECLINED, TORSO_RISING, UPRIGHT_IN_BED, SHIFTING_TO_EDGE, EDGE_SITTING, ATTEMPTING_STAND, OUT_OF_BED.

| View | Test | Frames | Model accuracy | Model macro-F1 | Rules accuracy | Rules macro-F1 |
|---|---|---|---|---|---|---|
| side | leave out pilot | 6187 | 0.42 | 0.34 | 0.64 | 0.35 |
| side | leave out student 1 | 1949 | 0.63 | 0.55 | 0.59 | 0.40 |
| side | leave out student 2 | 1383 | 0.77 | 0.70 | 0.32 | 0.21 |
| side | leave out student 3 | 1860 | 0.79 | 0.72 | 0.36 | 0.21 |
| side | leave out fall student 01 | 1774 | 0.64 | 0.39 | 0.17 | 0.18 |
| **side** | **staff test (3 people)** | 5235 | **0.70** | **0.56** | 0.42 | 0.30 |
| along | leave out pilot | 6140 | 0.75 | 0.51 | 0.57 | 0.33 |
| along | leave out student 1 | 2200 | 0.70 | 0.63 | 0.31 | 0.20 |
| along | leave out student 2 | 1852 | 0.79 | 0.70 | 0.27 | 0.22 |
| along | leave out student 3 | 2036 | 0.83 | 0.76 | 0.27 | 0.24 |
| along | leave out fall student 01 | 1963 | 0.51 | 0.45 | 0.35 | 0.25 |
| **along** | **staff test (3 people)** | 4651 | **0.66** | **0.54** | 0.41 | 0.28 |

Staff test per staff member (model accuracy / macro-F1):
- side: 0.83 / 0.81, 0.68 / 0.48, 0.69 / 0.58;
- along: 0.66 / 0.62, 0.60 / 0.41, 0.73 / 0.54.

The old pilot-only model scored macro-F1 0.40 (side) and 0.31 (along) on the same staff frames.

Per-phase F1 on the staff test:

| View | RECLINED | TORSO_RISING | UPRIGHT_IN_BED | SHIFTING_TO_EDGE | EDGE_SITTING | ATTEMPTING_STAND | OUT_OF_BED |
|---|---|---|---|---|---|---|---|
| side | 0.80 | 0.48 | 0.67 | 0.30 | 0.59 | 0.14 | 0.92 |
| along | 0.62 | 0.38 | 0.29 | 0.37 | 0.51 | 0.60 | 0.98 |

### 4-state posture (in bed / sitting / upright / on the ground), both cameras together

| Test | Frames | Model accuracy | Model macro-F1 | Rules accuracy | Rules macro-F1 |
|---|---|---|---|---|---|
| leave out pilot | 11108 | 0.90 | 0.89 | 0.71 | 0.55 |
| leave out student 1 | 3578 | 0.99 | 0.99 | 0.40 | 0.44 |
| leave out student 2 | 2527 | 0.99 | 0.99 | 0.59 | 0.52 |
| leave out student 3 | 3066 | 0.95 | 0.90 | 0.62 | 0.51 |
| leave out fall student 01 | 4875 | 0.46 | 0.53 | 0.85 | 0.69 |
| **staff test (3 people)** | 8225 | **0.92** | **0.94** | 0.54 | 0.46 |

- When fall student 01 is left out, the training set has no on-the-ground frames at all, which is why the model loses to the rules there. The deployed model is trained on fall student 01, so this does not apply to it.
- The old pilot+Lifan posture model scored macro-F1 0.82 on the staff frames.
- Accuracy is higher than macro-F1 where long, easy phases dominate (RECLINED, OUT_OF_BED). Macro-F1 weights the short transition phases equally, so it is the fairer number.

## 3. Bed exit: exit events

Scoring:
- A **labelled exit** is the start of OUT_OF_BED after a bed phase.
- Exits are split into **sustained** (out of bed for 3 s or more) and **brief stands** (stood for 1-2 s and sat back down: s2_02 ×3, s2_03 ×1).
- **Latency** = BED_EXIT time − labelled exit. An exit counts as caught by a BED_EXIT from 5 s before it to 15 s after.
- A **false exit** is a BED_EXIT away from every labelled exit.
- **Early warning** = BED_EXIT_WARNING with `emit_early_warning` on (High mode). Lead = how long before the exit the first warning came. Turning warnings on did not change any exit.

| Camera | Clips | Sustained exits caught | Median latency | Brief stands caught | False exits | Exits warned | Median warning lead | Warnings with no exit after |
|---|---|---|---|---|---|---|---|---|
| D435f | pilot | 4/4 | +3.5 s | - | 0 | 2/4 | 13.7 s | 2 |
| D435f | students 27 Oct | 5/7 | +3.9 s | 0/4 (one later exit) | 0 | 11/11 | 35 s | 9 |
| D435f | fall student 01 | 0/1 | - | - | 1 (f4) | 0/1 | - | 0 |
| **D435f** | **all** | **9/12** | **+3.9 s** | 0/4 | **1** | 13/16 | 28 s | 11 |
| D435i | pilot | 4/4 | +1.3 s | - | 0 | 0/4 | - | 0 |
| D435i | students 27 Oct | 6/7 | +2.0 s | 0/4 (one later exit) | 0 | 0/11 | - | 0 |
| D435i | fall student 01 | 1/1 | +2.5 s | - | 0 | 1/1 | 26.5 s | 0 |
| **D435i** | **all** | **11/12** | **+1.9 s** | 0/4 | **0** | 1/16 | - | 0 |

Missed sustained exits:
- D435f: s1_01 (64 s, 4 s out of bed), s4_03 and f1_01.
- D435i: s1_01.

Notes:
- **Brief stands are never reported as exits**, because BED_EXIT needs sustained standing. For a high-risk patient a 1-2 s stand-up is itself the risky moment, so it should become its own "stood up" alert, or count under the early warning.
- **The early warning only fires on the side-on view.** That is the D435f on 27 Oct and the D435i on 29 Oct; the cameras swapped. When it fires, it comes when the person sits up, typically 15-45 s before the exit, and also on sit-ups that never become exits (s1, s3).
- The 7-phase classifier and posture results are in the other agent's `~/.ahfd-work/train_20261031.out` (staff test: phase macro-F1 0.56 side / 0.54 along, posture 0.94).

### 3b. Bed-edge warning (2 Nov, `detect/edge_warning.py`, not yet wired into the dashboard)

This is a low-priority "check on the patient" notification, sent when they sit at the bed edge with their legs over the side. It is not an alarm.

How it works:
- A small boosted-tree model scores each frame "at the edge" against "in bed". "At the edge" is the labelled SHIFTING_TO_EDGE, EDGE_SITTING or ATTEMPTING_STAND.
- Its inputs:
  - where the hips sit on the bed outline (`hip_d`);
  - where the feet land on the floor relative to the outline (`ank_d`);
  - knee angle and trunk tilt;
  - their 1 s medians;
  - how much the bed supports the body.
- It warns after 2 s of scores at or above 0.7, once per bout. It re-arms after 5 s back in bed.

Scoring:
- Each person is scored by a model trained on the other people (leave one person out). This uses the same development clips as section 3: 12 sustained exits per view, about 26 min.
- **Lead** = how long before the person left the bed the first warning came.

| View | Frame accuracy | Frame F1 (at the edge) | Exits warned | Median lead (minimum) | At the edge, no exit followed | Up to 10 s before the labelled edge sitting | Not at the edge |
|---|---|---|---|---|---|---|---|
| **side** | **0.93** | **0.85** | **11/12** | **12.9 s** (0.4 s) | 7 | 3 | **2** |
| along | 0.73 | 0.50 | 11/12 | 17.5 s (7.3 s) | 8 | 2 | 11 |

What the columns mean:
- **"At the edge, no exit followed"** is the intended use: the patient sat at the edge and the nurse gets a chance to check.
- **"Up to 10 s before the labelled edge sitting"** means the warning came slightly before the labelled edge sitting started, as the person was getting into position.
- **"Not at the edge"** is the only true false-alarm column.

Compared with the current BED_EXIT, which comes 2-4 s *after* standing, this moves the first bed notification to typically about 13 s *before* the exit.

Use it on the side-on view only. Along the beds, the feet are hidden behind the bed and it gives 11 false warnings. The rule-based `EDGE_SITTING` phase was tried first and rejected: it fired on sitting up in the middle of the bed (about 11-18 false warnings per view).

Scripts:
- `~/.ahfd-work/edge_legs_scan_20261102.py` builds the features.
- `~/.ahfd-work/edge_legs_train_20261102.py` scores the warning and saves `models/edge_{side,along}.joblib` (git-ignored).

## 4. Takeaways

- **Falls:** the current system catches 5/5 on both cameras, 4 of them before landing. The cost is false alarms: about 30 per hour of active movement on the D435f and 60 on the D435i, nearly all from POSSIBLE_FALL on sit-downs.
- **Pre-impact rule:** on the D435f it gives about the same warning time (-0.86 s) with 1 false alarm instead of 9. Replacing POSSIBLE_FALL with it, keeping the floor alert as the safety net, gives 5/5 caught, 4 before landing, and 6 false alarms (13.7 per hour) on the D435f.
- **Bed exit:** sustained exits are reliable on both cameras: 20/24, no false exits except one on the D435f, typically 2-4 s after the person is up. Brief stand-ups are not caught.
- **Bed-exit classification:** the trained 7-phase model beats the rules on unseen staff. Side view: accuracy 0.70 vs 0.42, macro-F1 0.56 vs 0.30. Along view: 0.66 vs 0.41, macro-F1 0.54 vs 0.28. The short transition phases (SHIFTING_TO_EDGE, ATTEMPTING_STAND, TORSO_RISING) stay weak, at F1 0.14-0.60, because there is little labelled data for them. The posture model reaches accuracy 0.92 and macro-F1 0.94 on unseen staff.

## 5. Additional metrics (2 Nov)

Scripts and outputs are in `~/.ahfd-work/`:
- `bench_metrics_cls_20261102.*` for the classifiers;
- `bench_falls_20261102.out` for the falls;
- `edge_legs_train_20261102.py` for the edge warning.

### Classifiers on the staff test (3 unseen staff)

| Model | Accuracy | Macro-F1 | Balanced accuracy | Cohen's kappa | Per staff member: accuracy, mean ± SD |
|---|---|---|---|---|---|
| Posture (trained) | 0.922 | 0.936 | 0.928 | 0.928 | 0.937 ± 0.070 |
| Posture (rules) | 0.538 | 0.464 | 0.556 | 0.357 | - |
| 7-phase, side (trained) | 0.704 | 0.557 | 0.587 | 0.631 | 0.734 ± 0.079 |
| 7-phase, side (rules) | 0.423 | 0.297 | 0.318 | 0.313 | - |
| 7-phase, along (trained) | 0.662 | 0.536 | 0.538 | 0.564 | 0.664 ± 0.066 |
| 7-phase, along (rules) | 0.411 | 0.284 | 0.283 | 0.330 | - |

Posture per class, precision / recall:
- in bed: 0.87 / 1.00;
- sitting: 0.97 / 0.88;
- upright: 1.00 / 0.91.

There are no on-the-ground frames in the staff test, so posture is scored over 3 classes. **256 of 3274 standing staff frames (7.8%) were called on_ground**, a false floor reading. The model is display-only today; fix this before alerts read it.

7-phase side view, precision / recall per phase:

| Phase | Precision | Recall |
|---|---|---|
| RECLINED | 0.92 | 0.71 |
| TORSO_RISING | 0.38 | 0.66 |
| UPRIGHT_IN_BED | 0.77 | 0.60 |
| SHIFTING_TO_EDGE | 0.22 | 0.47 |
| EDGE_SITTING | 0.51 | 0.69 |
| ATTEMPTING_STAND | 0.29 | 0.09 |
| OUT_OF_BED | 0.94 | 0.90 |

Most errors are between neighbouring phases (see the confusion matrices in the `.out` file).

Edge-sitting model (leave one person out, at the edge vs in bed):

| View | ROC-AUC | PR-AUC (base rate 0.23) | Precision | Recall | Balanced accuracy | Kappa | Per-person ROC-AUC |
|---|---|---|---|---|---|---|---|
| side | 0.978 | 0.945 | 0.82 | 0.89 | 0.916 | 0.810 | 0.966 ± 0.045 |
| along | 0.729 | 0.550 | 0.44 | 0.57 | 0.675 | 0.318 | 0.840 ± 0.158 |

### Alerts (development clips, events)

Definitions:
- **Sensitivity** = labelled events alerted.
- **Precision** = alert episodes (alerts less than 10 s apart count as one) that belong to a real event.

| Alert | Camera | Sensitivity | Precision | False alarms per hour | Timing, median (range) |
|---|---|---|---|---|---|
| Fall: floor alert only | D435f / D435i | 5/5 / 5/5 | 0.60 / 0.67 | 13.7 / 9.6 | +1.25 s / +1.51 s |
| Fall: current system (floor + possible fall) | D435f / D435i | 5/5 / 5/5 | 0.41 / 0.24 | 29.7 / 60.0 | -0.62 s / -0.44 s |
| Fall: pre-impact rule only | D435f | 4/5 | **0.83** | **2.3** | -0.86 s (-1.07 to -0.43) |
| Fall: floor + pre-impact rule | D435f | 5/5 | 0.60 | 13.7 | -0.75 s |
| Bed exit (sustained) | both | 20/24 (0.83) | 0.95 | 1.1 | +2.6 s (+0.1 to +11.5) |
| Bed-edge warning | side view | 11/12 exits warned | 0.90 (19/21 at or near the edge) | about 4.6 | -12.9 s before exit (0.4 to 42) |

Frame-level ROC-AUC of the pre-impact scores (out-of-bed frames, falling = 1 s before to 0.3 s after impact):
- D435f: boosted model 0.889, logistic 0.814, rule 0.762.
- D435i: boosted 0.974, logistic 0.977, rule 0.685.

The PR-AUC is low (0.17-0.48) because falling frames are only 0.3-2.6% of all frames. The event-level numbers above are the ones to quote.

### Jetson load

Headless pipeline: RTMO-s on CUDA, tracking, fall and bed engines. Input was a 1080p D435f recording, 793 frames, measured with `tegrastats` at 1 s intervals.

| | Idle (system services) | Pipeline running |
|---|---|---|
| Throughput | - | **24.9 fps** headless, from a recording with no depth, no camera and no drawing. This is an upper limit, not the live speed; see section 6. |
| GPU (GR3D) | 0% | 73% mean, 98% peak |
| CPU (8 cores, mean) | 3% | 25% |
| RAM | 3.5 GB | 4.4 GB of 15.7 GB (+0.9 GB) |
| Board power (VDD_IN) | 6.8 W | **16.4 W** mean, 18.2 W peak |
| Junction temperature | 63 C | 66 C mean, 68 C max (fan control on, no throttling) |

## 6. Live speed and latency (2 Nov)

Measured on the real dashboard pipeline with the D435f connected, timed from outside the code. Scripts and JSON results are in `~/.ahfd-work/`: `bench_runtime_20261102*`, `bench_stream_20261102.py`, `bench_models_20261102.py`, `bench_fps_effect_20261102.py`.

**The live room was empty in every live run.** One tracked person added about 14 ms per frame on a recording, so live speed with a patient in view may be slightly lower. That has not been measured.

Recorded-clip runs used dev clip `f3_01`. Nothing was written to the real event log.

### Live results

**FPS** counts frames fully processed per second. **Camera-to-decision** is the camera's frame timestamp to the end of processing.

| Setup | FPS | Camera frames skipped | Camera-to-decision, median (p90) | Stream |
|---|---|---|---|---|
| **Depth + camera view (today's default, nothing else running)** | **4.93** | 84% | **257 ms** (280 ms) | 0.8 Mbit/s |
| Depth + stick-figure view (other work running) | 5.87 | 80% | 225 ms (246 ms) | 0.4 Mbit/s |
| No depth + camera view (nothing else running) | 15.79 | 47% | 128 ms (134 ms) | 2.6 Mbit/s |
| No depth + camera view (other work running) | 18.38 | 38% | 112 ms (131 ms) | 3.3 Mbit/s |
| Depth with lighter filter (`spatial=1`) | 5.13 | 83% | 249 ms (273 ms) | 0.8 Mbit/s |

- **Depth is the main cost, and other work on the Jetson is not.** With no other process running, the default still reached 4.93 FPS (4.98 with other work running).
- The two no-depth runs differ by about 15%, so treat no-depth speed as 16-18 FPS.
- The earlier "about 14.5 fps, 18.6 with a 960 px stream" figures match the no-depth speed. They do not describe today's default.
- **The FPS shown on the dashboard is too high** (12.7 when the real rate is 4.93), because it leaves out the camera read, depth alignment, drawing and encoding. Do not quote it.

### Time per frame, default setup (median)

| Stage | Time |
|---|---|
| Depth alignment and filtering | 77 ms |
| Pose (RTMO-s on CUDA) | 73 ms |
| Depth preview | 35 ms |
| JPEG encoding | 8 ms |
| Depth heights on joints | 5 ms |
| Alert clip buffer | 5 ms |
| Drawing and resizing | 4 ms |

The frames are about 200 ms apart. The stages add up to about 207 ms because each median comes from separate frames.

### Pose model alone

| Device | Median | p90 | Pose-only FPS |
|---|---|---|---|
| CUDA | 34.9 ms | 41.4 ms | 28.7 |
| CPU | 159 ms | 164 ms | 6.3 |

- CUDA is 4.6 times faster than the CPU.
- Pose took 73 ms inside the depth pipeline and 49 ms in the no-depth run. The cause of the slowdown has not been established.
- Trained models are cheap: 0.5 ms (posture tree) and 0.6 ms (7-phase logistic) for one prediction.

### Resources and stability

| Measure | Result |
|---|---|
| 10-minute run, default setup | 4.85-4.93 FPS each minute, no slowdown |
| Board power | 10.5 W at the default setup, against 6.8 W idle (section 5). The 16.4 W in section 5 is the headless pipeline running flat out. |
| Peak junction temperature | 66.7 C over the 10 minutes |
| CPU / GPU | 16% mean over 8 cores / 38-39% |
| Process memory | About 1.2 GB (1222 MB rising to 1264 MB in 10 minutes; too short to rule out a slow leak) |
| Start to first frame | 3.4 s (model load 0.5 s) |
| Viewers | 3 browsers at once each got 5.0 FPS at 0.88 Mbit/s, the same as 1 |
| Status request / ward page load | 1.5 ms / 1.2 ms on localhost |
| 4 people (synthetic 2x2 mosaic, no depth, no camera) | 89 ms per frame, against 54 ms with nobody detected |

### Alert delay to the screen

This is calculated from measured parts, not measured end to end. Camera to decision takes about 0.26 s, and the page checks for alerts once a second, adding 0 to 1 s. That gives **about 0.3 to 1.3 s**, plus any network delay.

**The fall and exit timings in sections 1, 3 and 5 do not include this delay.** They use recording time. A fall detected 0.6 s before landing can therefore reach the nurse's screen after the landing. Write "detected before landing", not "nurse alerted before landing".

### Does 5 FPS still detect?

Stored tracks were replayed with frames dropped to 5 FPS. This is a **simulation**: tracking and smoothing still had every frame when the tracks were made, so it may be optimistic. It has not been tested with a live fall. The comparison values come from section 1 and 3 scoring; the D435f rerun at the recorded rate gave 14 false alarms (30.7/h) against the 13 (29.7/h) in section 1, so small count differences are scoring noise.

| Measure | D435f: recorded rate → 5 FPS | D435i: section 1 and 3 → 5 FPS |
|---|---|---|
| Falls caught | 5/5 → 5/5 | 5/5 → 5/5 |
| Detected before landing | 4 → 4 | 4 → 4 |
| Median fall timing | -0.62 s → -0.55 s | -0.44 s → -0.40 s |
| False alarms per hour | 30.7 → 24.1 | 60.0 → 45.0 |
| Bed exits caught, including brief stands | 10/16 → 10/16 | 12/16 → 12/16 |
| Median exit timing | +3.9 s → +4.0 s | +1.9 s → +1.9 s |
| False exits | 1 → 1 | 0 → 0 |

On these dev clips, dropping to 5 FPS changed detection very little.
