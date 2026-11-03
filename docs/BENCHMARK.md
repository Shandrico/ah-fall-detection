# Benchmark: development clips (1 Nov 2026)

These numbers come from the committed code at `b6de1e8` plus the untracked `detect/preimpact.py`, run with `configs/dashboard_jetson.yaml`.

Only development clips were used:
- pilot takes 01-08;
- the 27 Oct students s1-s4;
- the 29 Oct fall student 01 (`f*_01`, `neg_f6_01`, `neg_f7_01`).

The staff clips, fall student 02 and pilot 09-10 were not read for sections 1 and 3. Fall student 02 and the staff events have since been used once, as the locked test in section 7.

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

Leave one person out, averaged per person (5 training people, from the tables in section 2, sample SD):

| Model | Accuracy, mean ± SD | Macro-F1, mean ± SD | Lowest person (accuracy) |
|---|---|---|---|
| Posture | 0.86 ± 0.23 | 0.86 ± 0.19 | fall student 01: 0.46 |
| 7-phase, side | 0.65 ± 0.15 | 0.54 ± 0.17 | pilot volunteer: 0.42 |
| 7-phase, along | 0.72 ± 0.13 | 0.61 ± 0.13 | fall student 01: 0.51 |

No person is removed from these averages. Without fall student 01, posture is 0.96 ± 0.04 accuracy and 0.94 ± 0.06 macro-F1 across 4 people. That person's 0.46 happens because leaving them out removes every on-the-ground frame from training, so always state this exclusion and its reason when quoting 0.96.

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

## 7. Locked test (2 Nov, run once)

Code frozen at commit `51c2673`. The rules were written down before any locked clip was scored (`~/.ahfd-work/locked_test_20261102/PREREGISTERED.md`) and use the same scoring windows as sections 1, 3 and 3b. Nothing is to be tuned on these results. **These clips are now used. Any further tuning needs new recordings to test on.**

Clips:
- fall student 02: 4 falls, plus a deliberate lie-down and a kneel clip, seen by both cameras (about 3 min per camera);
- 3 staff members, about 10 min per camera, no falls. Their 7 sustained bed exits come from staff 2 and 3 only: staff 1's clips have no labelled exit.

Replayed from stored tracks at the recorded frame rate, not live.

### Falls

**Precision** counts each caught fall as one true alert. **False alarms** cover the fall-student clips and the staff clips together (about 13 min per camera).

| System | Camera | Falls caught | Detected before landing | Median timing | Precision | False alarms (per h) | Dev false alarms per h |
|---|---|---|---|---|---|---|---|
| **Current (floor + possible fall)** | D435f | **4/4** | **4** | **-0.58 s** | 0.40 | 6 (28.1) | 29.7 |
| **Current (floor + possible fall)** | D435i | **4/4** | **4** | **-0.53 s** | 0.44 | 5 (23.9) | 60.0 |
| Floor alert only | D435f | 4/4 | 1 | +0.06 s | 0.57 | 3 (14.0) | 13.7 |
| Floor alert only | D435i | 4/4 | 0 | +0.27 s | 0.50 | 4 (19.1) | 9.6 |
| Pre-impact rule only | D435f | 4/4 | 4 | -0.45 s | 0.50 | 4 (18.7) | **2.3** |
| Pre-impact rule only | D435i | **1/4** | 1 | -0.45 s | 0.20 | 4 (19.1) | 24.0 |
| Pre-impact model only | D435f | 4/4 | 4 | -0.70 s | 0.50 | 4 (18.7) | 9.1 |
| Pre-impact model only | D435i | 4/4 | 2 | +0.08 s | 0.50 | 4 (19.1) | 21.6 |

- **The current system held up:** every fall was caught on both cameras, all before the labelled landing, with false alarm rates no worse than on the development clips.
- **The pre-impact rule's low false alarm rate did not hold.** The D435f went from 2.3 to 18.7 per hour, based on only 4 alarms. It also caught just 1 of 4 falls on the D435i.
- The deliberate lie-down raised a floor alert at +0.81 s and +0.94 s, as intended. Each kneel clip raised 1 false alarm, included above.
- Floor alert timing is earlier than on the development clips (+1.25 to +1.51 s). The D435i impact labels on these clips are in whole seconds and every label is good to about ±1 s, so do not read much into differences under 1 s.

### Bed exit (sustained, at least 3 s)

| People | Camera | Exits caught | Median timing | False exits |
|---|---|---|---|---|
| **staff 2 and 3** | D435f | 5/7 | +0.4 s | 0 |
| **staff 2 and 3** | D435i | 6/7 | +0.15 s | 0 |
| **staff 2 and 3, both cameras added** | | **11/14 (79%)** | **+0.4 s** | **0** |
| Fall student 02 | both | 2/2 | 0 s | 0 |

- Missed: `h2_01` on both cameras and `h3_01` on the D435f.
- There were no brief stand-ups in these clips, so that weakness was not tested.

### Bed-edge warning (not connected to the dashboard)

| View | Exits warned | Median lead (minimum) | Warnings | At the edge, no exit followed | Not at the edge | Dev result (warned, lead, not at edge) |
|---|---|---|---|---|---|---|
| side | **5/8** | 6.2 s (1.5 s) | 14 | 2 | **5** | 11/12, 12.9 s, 2 |
| along | 8/8 | 7.7 s (1.0 s) | 16 | 2 | 4 | 11/12, 17.5 s, 11 |

Clearly weaker than on the development clips, and the earlier advice to use the side view only is not supported by this test.

### Posture on a person with real on-the-ground frames (fall student 02, 4212 frames)

| Method | Accuracy | Macro-F1 | On the ground: precision / recall |
|---|---|---|---|
| Trained tree | 0.950 | 0.915 | 1.00 / 0.95 |
| Rules | 0.883 | 0.541 | 1.00 / 0.96 |

83% of these frames are on the ground, so macro-F1 is the fairer number. This fills the gap in section 5, where the staff test has no on-the-ground frames.

### Without the 27 Oct staff clips (`h*_01`), added after the test

These clips recorded at only 6-9 FPS, with gaps of up to 4.2 s, because `record-dual` wrote straight to the T9 (found 28 Oct). The other staff clips run at about 15 FPS with no gap over 0.2 s. This exclusion was decided after the results were seen, so report it next to the full result above, never instead of it.

| Measure | D435f | D435i |
|---|---|---|
| Fall detection rate | 4/4 | 4/4 |
| Fall false-alarm groups | 3 in 9.6 min (about 19/h) | 3 in 9.5 min (about 19/h) |
| Fall precision | 0.57 | 0.57 |
| Bed exits caught (staff 2 and 3) | **4/4**, median +0.4 s | **4/4**, median -0.65 s |
| False bed exits | 0 | 0 |
| Bed-edge warning, side / along view | 2/5 warned / 5/5 warned | |

All 3 missed exits in the full result were in the `h*_01` clips, and each happened at an ID change (traced in `~/.ahfd-work/locked_test_20261102/trace_missed_exits.py`). Without them only 4 exits remain, so 4/4 is a very small sample.

### Human review of every alert (Shandrico, 2 Nov)

Every alert in the clips above (`h*_01` excluded) was watched on video and given a verdict on the review page (`~/.ahfd-work/locked_test_20261102/assess_server.py`; verdicts in `assessments.json`, reviewer 1).

A fall alert counted false by the strict windows could be marked:
- **false**: no risk;
- **risky**: useful, because the person was bending, kneeling or near the floor;
- **real**: a fall the labels missed.

| Measure | D435f | D435i |
|---|---|---|
| Alerts reviewed | 13/13 | 13/13 |
| Labelled falls with an alert: verdict "correct" | 5/5 (4 falls + deliberate lie-down) | 5/5 |
| Strict false alarms: verdict "risky" (useful) | 3/3 (staff h3_02, h3_03 bending; kneel clip neg_f7_02) | 3/3 (same moments) |
| Strict false alarms: verdict "false" (no risk) | **0** | **0** |
| Fall precision, strict / by review | 0.57 / **1.00** | 0.57 / **1.00** |
| Bed-exit alerts: verdict "correct" | 5/5 | 5/5 |
| False bed exits, strict / by review | 0 / 0 | 0 / 0 |

- By the reviewer's judgement, every fall alert on these clips pointed at a real fall or a genuinely risky movement near the floor. No alert fired on a person who was safe. This matches the policy that a fall-risk patient bending to the floor is worth a look.
- The strict numbers stay the primary result. The review is one person's judgement, made by a team member who knew which clips were falls.
- **Not reviewed:** the `h*_01` clips (recorded at 6-9 FPS), whose alerts count as false in the full result. These are h2_01 on the D435f (2 possible-fall alerts) and h3_01 on both cameras (1 on the D435f, 2 on the D435i).

### Limits

- Falls are from one healthy student on crash mats, and 4/4 is still consistent with a true rate as low as about 40% (95% confidence).
- The bed boxes for 29 Oct were drawn on the view shared with the staff clips, so the staff bed-exit test is not fully separate on geometry. The staff phase labels were also scored once before (section 2), without tuning.
- All timings exclude the 0.3-1.3 s screen delay in section 6.

## 8. Fewer false alarms: possible-fall alert needs the trunk tipping (3 Nov)

**Policy (reviewer, 2 Nov):** an alert on a genuinely risky movement (bending, kneeling, getting low near the floor) is useful, not a false positive. The target is 2-5 false alarms an hour, accepting later detection.

**Change:** the POSSIBLE_FALL *alert* now also needs the trunk to tip over. The trunk's image tilt must turn at 80 deg/s or more within 0.5 s of the torso drop:
- `possible_rot_deg_s` = 80;
- `possible_rot_window_s` = 0.5;
- 0 turns the gate off.

Sitting down drops the torso as far as a fall does, but keeps it upright. The on-screen flag and the floor alerts are unchanged.

**Tuned on development clips only.** The scripts are in `~/.ahfd-work/fp_tune_20261103/`.

Counting:
- The kneel / pick-up clip (`neg_f7_01`) counts as risky.
- Every other alert away from a fall counts as false. That is conservative: pilot block D (takes 07/08) had some bending in its script, but it was never reviewed.

Threshold sweep:
- 15 drop-threshold settings, from 0.15 to 0.5 m over 0.5 or 0.8 s. Raising the drop threshold lost the early warning before it removed sit-downs.
- The balance-based rule from `preimpact.py`, with and without agreement from the possible-fall alert.
- A rotation gate at 40-130 deg/s.
- A fast-v_z bypass, rejected because it added pilot false alarms.
- A minimum number of frames before the floor alert, rejected because it delayed the slow-slump fall from +2.0 s to +3.7 s.

### Development clips (pilot clips 01-08 excluded, about 19 min per camera)

**Decision (3 Nov):** the pilot recordings `clip1_01`-`clip1_08` are no longer used for fall evaluation. Their depth heights are known to be wrong (handoff notes, "Depth"): a lying torso reads 0.2-1.1 m and jumps. Stray short-lived detections also appear beside the bed. The files are kept, not deleted. The dev set is now:
- fall student 01 (`f*_01`, `neg_f6_01`, `neg_f7_01`);
- the 27 Oct students s1-s4.

| Camera | System | Falls caught | Before landing | Median timing | False alarms (per h) | Risky alerts (kneel clip) |
|---|---|---|---|---|---|---|
| D435f | before (possible-fall alert ungated) | 5/5 | 4 | -0.62 s | 5 (15.3) | 3 |
| D435f | **after (gate at 80 deg/s)** | 5/5 | **4** | **-0.62 s** | **0 (0.0)** | 2 |
| D435i | before | 5/5 | 4 | -0.44 s | 4 (13.0) | 3 |
| D435i | **after** | 5/5 | **4** | **-0.44 s** | **0 (0.0)** | 1 |

With the pilot clips included (27 min per camera), for reference only:
- D435f: 21.5 per hour before, 8.6 after.
- D435i: 47.2 per hour before, 15.7 after.

All of the remaining false alarms come from those pilot clips.

### Post-hoc check on the locked clips (already used on 2 Nov, so not an independent test)

| Camera | | Falls caught | Before landing | Median timing | Risky (reviewed) | Other alert groups (per h, 12.6 min) |
|---|---|---|---|---|---|---|
| D435f | before | 4/4 | 4 | -0.58 s | 3 | 3 (14.2) |
| D435f | after | 4/4 | **4** | -0.48 s | 3 | **1 (4.7)** |
| D435i | before | 4/4 | 4 | -0.53 s | 3 | 3 (14.3) |
| D435i | after | 4/4 | **1** | +0.17 s | 3 | **1 (4.8)** |

- The floor alert in h3_02 at about 34 s belongs to the bending episode reviewed as risky. It now starts its own group because the possible-fall alert at 27 s is gone, so it is counted as risky here.
- The one remaining alert on each camera is in staff clip `h3_01` from 27 Oct, which was recorded at 6-9 FPS. It has not been reviewed.

**The cost the policy accepted:**
- On the locked D435i clips, 3 of the 4 falls lost their before-landing alert. They are still caught at or just after landing (0 to +0.8 s).
- On the D435f, nothing was lost.

**Known blind spot:** a fall straight towards or away from the camera barely changes the image tilt. It gets no possible-fall alert, only the floor alert after landing; the golden synthetic fall is one of these.

## 9. Bed exit: track ID handover (3 Nov)

**Change:** the tracker often gives a patient a new ID as they leave the bed, because pose drops out at the edge for a second or two. The new ID had never been seen in the bed, so no exit was reported. Now a new track that first appears within 4 s of a lost in-bed track, and within 1 m of where that track was last seen, takes over its bed history (`BedExitThresholds.handoff_s`, `handoff_m`).

This was found by tracing the 3 missed staff exits in section 7 (`~/.ahfd-work/locked_test_20261102/trace_missed_exits.py`): all three happened at an ID change. Scored with `~/.ahfd-work/handoff_eval_20261103.py`, sustained exits (at least 3 s):

| Clips | Camera | Without handover | With handover | False exits (without / with) |
|---|---|---|---|---|
| Development (pilot, students, fall student 01) | D435f | 9/12 | 9/12 | 1 / 1 |
| Development | D435i | 11/12 | 11/12 | 0 / 0 |
| Staff 2 and 3 (locked, used) | D435f | 4/4 | 4/4 | 0 / 0 |
| Staff 2 and 3 (locked, used) | D435i | 4/4 | 4/4 | 0 / 0 |
| Staff 1-3, 27 Oct `h*_01` (6-9 FPS) | D435f | 1/3 | 1/3 | 0 / 0 |
| Staff 1-3, 27 Oct `h*_01` (6-9 FPS) | D435i | 2/3 | **3/3** | 0 / 0 |

Updated staff result (all 7 exits): **D435f 5/7 (71%), D435i 7/7 (100%)**, no false exits.

- The handover recovered 1 of the 3 misses (`h2_01` on the D435i) and added no false exits.
- The two D435f misses remain. In `h2_01` the new track stood inside the 27 Oct bed outline, which was never redrawn for that mount. In `h3_01` no new track appeared close enough.
- It was designed after reading the locked clips, so the improvement is not an independent result. A visitor appearing beside the bed just as the patient's pose drops out could take over the history; this is untested.

### Full bed-exit metrics, with the handover (3 Nov)

From `~/.ahfd-work/exit_metrics_20261103/` (`results.json`), committed code at `9e0c40e`. Event windows as in sections 3 and 7. Frame scores cover labelled frames of the person the review tool follows, and missing or UNKNOWN rule phases count as wrong.

**Events (sustained exits, at least 3 s):**

| Clips | Camera | Exits caught (sensitivity) | Precision | False exits (per h) | Timing, median (range) |
|---|---|---|---|---|---|
| Development | D435f | 9/12 (0.75) | 0.90 | 1 (2.3) | +3.9 s (+1.8 to +11.5) |
| Development | D435i | 11/12 (0.92) | 1.00 | 0 (0.0) | +1.9 s (+0.1 to +9.3) |
| **Staff, all 3 (locked, used)** | **D435f** | **5/7 (0.71)** | **1.00** | **0 (0.0)** | **+0.4 s (-0.3 to +3.4)** |
| **Staff, all 3 (locked, used)** | **D435i** | **7/7 (1.00)** | **1.00** | **0 (0.0)** | **+0.2 s (-3.5 to +3.2)** |
| Staff 2-3 only (no `h*_01`) | D435f | 4/4 | 1.00 | 0 | +0.4 s |
| Staff 2-3 only (no `h*_01`) | D435i | 4/4 | 1.00 | 0 | -0.7 s |

Without the handover, the staff D435i result is 6/7 (0.86). Every other row is the same with or without it. Brief stand-ups (1-2 s) are still caught 1 of 4 on the development clips; the staff clips have none.

**Frames (7 phases), staff clips, all 3 staff:**

| Camera | Rules: accuracy / macro-F1 / balanced accuracy | Phase model: accuracy / macro-F1 / balanced accuracy |
|---|---|---|
| D435f | 0.41 / 0.30 / 0.31 | 0.69 / 0.57 / 0.56 |
| D435i | 0.43 / 0.33 / 0.34 | 0.66 / 0.51 / 0.53 |

- The handover changes the rule frame scores by under 0.04 (largest on the D435i `h*_01` clips: accuracy 0.31 to 0.35).
- Development-clip rule scores: D435f 0.57 / 0.35 / 0.35, D435i 0.42 / 0.28 / 0.30. The model is not scored there, because those clips are its training data.
- These model numbers are grouped by camera and follow one person per frame, so they differ slightly from section 5 (grouped by view): side 0.70 / 0.56, along 0.66 / 0.54. Both show the model well ahead of the rules.

## 10. Alerts from the trained models instead of the rules (3 Nov)

**Question:** the posture and 7-phase models label frames far better than the rules (section 5), so would alerts based on their probability be better?

**Test:** the deployed models (never trained on these clips), turned into alerts by rules fixed before scoring (`~/.ahfd-work/model_alerts_20261103/RULES.md`):
- **Fall:** the posture model's P(on_ground) averaged over 0.5 s at or above a threshold, once per 10 s.
- **Bed exit:** at least 2 s in a bed phase, then P(OUT_OF_BED) averaged over 0.5 s at or above a threshold.
- Thresholds 0.5 (the chosen default), 0.7 and 0.9.

Clips: fall student 02 and staff 2-3, the locked clips already used on 2 Nov, so this is a post-hoc comparison. The rules are the committed code at `19d2a3a`, including the tipping gate (section 8) and the handover (section 9). Results: `~/.ahfd-work/model_alerts_20261103/results.txt`.

| Alert | Camera | Rules | Model at 0.5 | Model at 0.7 | Model at 0.9 |
|---|---|---|---|---|---|
| Falls caught (before landing) | D435f | 4/4 (4) | 4/4 (4) | 4/4 (3) | 4/4 (3) |
| Fall precision (false groups) | D435f | **0.57 (3)** | 0.44 (5) | 0.44 (5) | 0.44 (5) |
| Falls caught (before landing) | D435i | 4/4 (1) | 4/4 (2) | 4/4 (1) | 4/4 (1) |
| Fall precision (false groups) | D435i | **0.57 (3)** | 0.44 (5) | 0.50 (4) | 0.50 (4) |
| Bed exits caught, false | D435f | **5/5, 0** | 4/5, 0 | 4/5, 0 | 4/5, 0 |
| Bed exits caught, false | D435i | **5/5, 0** | 4/5, 1 | 4/5, 0 | 4/5, 0 |

- Including the 27 Oct `h*_01` clips, the exit model also gives 3 false exits on the D435f; the rules give none.
- Keeping only rule alerts the model agrees with (P(on_ground) at least 0.5 within -1 to +3 s) changed nothing: the false alarms are kneeling and bending, where the body really is low and the model is as sure as the rules.
- **Decision: alerts stay rule-based; the models stay on screen as labels.** The models win at labelling each frame, which is dominated by long easy stretches (lying, standing), not at the 1-2 s moments that decide an alert.

## 11. Early warning lead time (High mode, 3 Nov)

`BED_EXIT_WARNING`, which the dashboard admits only for beds set to High. **Lead** = how long before the event the first warning came, within the 60 s before leaving the bed. Two events are measured:
- **leaving the bed:** the start of OUT_OF_BED;
- **the stand attempt:** the start of the labelled ATTEMPTING_STAND before it.

Sustained exits only. Script and output: `~/.ahfd-work/early_warning_lead_20261103.*`. Staff clips are the used locked clips, scored with the committed code.

| Clips | View | Exits warned | Before leaving the bed, median (range) | Before the stand attempt, median (range) | Warnings with no exit within 60 s (per h) |
|---|---|---|---|---|---|
| Pilot (D435f) | side | 2/4 | 13.6 s (7.5-19.8) | 12.1 s (5.5-18.8) | 2 (14.4) |
| Students 27 Oct (D435f) | side | 7/7 | 42.5 s (16.1-57.7) | 39.5 s (15.1-55.7) | 9 (34.9) |
| Fall student 01 (D435i) | side | 1/1 | 26.5 s | 24.5 s | 0 |
| **Staff 27 Oct, `h*_01` (D435f)** | side | **2/3** | **24.7 s (10.7-38.7)** | n/a (1 exit has a labelled stand) | 2 (37.7) |
| **Staff 29 Oct, `h*_02/03` (D435i)** | side | **2/4** | **11.9 s (6.2-17.5)** | **10.4 s (4.2-16.5)** | 3 (27.7) |
| All along-the-beds clips | along | **0/19** | - | - | 3 |

- **Side view only.** From along the beds it warned before none of 19 exits.
- On the staff clips, which none of the rules were tuned on, it warned before **4 of 7** exits from the side, typically **10-25 s** before leaving the bed.
- The warning comes when the person sits up and moves toward the edge. That is why it also fires on sit-ups that do not become exits: 14-38 per hour of this active footage. Whether nurses count those as useful "go and check" prompts needs their input, as with the kneeling alerts in section 7.
- The stand-attempt lead is only 1-3 s shorter than the bed-leaving lead, because the labelled stand attempt is short.

## 12. Full re-run on the current code (3 Nov)

All numbers in this section come from committed code `9e0c40e`, which includes the trunk-tipping gate (section 8) and the bed-exit ID handover (section 9).

**Pilot clips 01-08 are not used for any evaluation** (decision of 3 Nov; their depth heights are faulty). They are still in the training data.

Scripts in `~/.ahfd-work/`:
- `bench_dev_20261103.*`: development events;
- `locked_posthoc_20261103/`: the locked clips re-run, a post-hoc check because they were used on 2 Nov;
- `bench_full_cls_20261103.*`: classifiers;
- `phase_*_20261103.*`: phase models;
- `edge_nopilot_20261103.*`: edge warning;
- `bench_runtime_20261103/`: speed.

Times are on the recording clock. "−" means before the labelled impact or exit. Labels are good to about ±1 s, and the alert reaches the screen 0.3-1.3 s later (section 6).

### 12a. Falls, fall by fall (current system = floor alerts + possible-fall alert with the tipping gate)

| Clip set | Fall | D435f | D435i |
|---|---|---|---|
| Development | f1_01 (stand attempt at the edge) | −0.62 s | −0.44 s |
| Development | f3_01, 1st fall | +2.00 s | +1.51 s |
| Development | f3_01, 2nd fall | −0.77 s | −0.66 s |
| Development | f4_01 (fall, quick recovery) | −1.42 s | −1.66 s |
| Development | f5_01 | −0.50 s | −0.40 s |
| Locked (post-hoc) | f1_02 | −0.46 s | −0.45 s |
| Locked (post-hoc) | f3_02 | −0.40 s | +0.33 s |
| Locked (post-hoc) | f4_02 | −1.03 s | +0.80 s |
| Locked (post-hoc) | f5_02 | −0.50 s | 0.00 s |

| Clip set | Camera | Falls caught | Before landing | Median (range) | Precision, strict / by review | False alarms per hour, strict / by review |
|---|---|---|---|---|---|---|
| Development (19.6 / 19.0 min) | D435f | 5/5 | 4 | −0.62 s (−1.42 to +2.00) | 1.00 / 1.00 | 0 / 0 |
| Development | D435i | 5/5 | 4 | −0.44 s (−1.66 to +1.51) | 1.00 / 1.00 | 0 / 0 |
| Locked, post-hoc (12.8 / 12.5 min) | D435f | 4/4 | 4 | −0.48 s (−1.03 to −0.40) | 0.50 / 0.80 | 18.7 / 4.7 |
| Locked, post-hoc | D435i | 4/4 | 1 | +0.17 s (−0.45 to +0.80) | 0.50 / 0.80 | 19.1 / 4.8 |

How the counts are made:
- On development clips, kneel-clip alerts count as risky (useful), not false.
- On the locked clips, "strict" counts every alert away from a fall as false.
- "By review" uses the 2 Nov verdicts: the kneel and the staff bending in h3_02 and h3_03 were judged risky. That leaves one unreviewed alert per camera, in the 6-9 FPS clip `h3_01`.
- The deliberate lie-down (`neg_f6`) raised a floor alert at +0.8 to +1.1 s in every run, as intended.

### 12b. Bed exits (sustained, at least 3 s)

| Clip set | Camera | Exits caught | Median delay (range) | Precision | False exits per hour |
|---|---|---|---|---|---|
| Development | D435f | 5/8 (0.62) | +3.9 s (+2.7 to +11.5) | 0.83 | 3.3 |
| Development | D435i | 7/8 (0.88) | +2.0 s (+0.2 to +9.3) | 1.00 | 0 |
| Staff, locked (post-hoc) | D435f | 5/7 (0.71) | +0.4 s (−0.3 to +3.4) | 1.00 | 0 |
| Staff, locked (post-hoc) | D435i | **7/7 (1.00)** | +0.2 s (−3.5 to +3.2) | 1.00 | 0 |
| Fall student 02, locked (post-hoc) | both | 2/2 | −0.2 to +0.2 s | 1.00 | 0 |

**Bed-edge warning**, development clips without the pilot, scored leave-one-person-out:

| View | Exits warned | Median lead before leaving the bed (minimum) | Frame ROC-AUC | Warning precision |
|---|---|---|---|---|
| side | 7/8 | 35.9 s (0.4 s) | 0.97 | 0.93 |
| along | 8/8 | 18.6 s (11.3 s) | 0.93 | 0.76 |

On the locked staff clips (section 7) it warned 5/8 (side) and 8/8 (along), with leads of 6-8 s.

### 12c. Recognition on the 3 unseen staff, and the new 7-phase model

| Model | Accuracy | Macro-F1 | Balanced accuracy | Cohen's kappa | 4-stage accuracy / macro-F1 | Time per prediction |
|---|---|---|---|---|---|---|
| Posture tree (4 classes) | 0.922 | 0.936 | 0.928 | 0.928 | - | 0.5 ms |
| 7-phase, side: v3 logistic | 0.704 | 0.557 | 0.587 | 0.631 | 0.821 / 0.783 | 0.6 ms |
| **7-phase, side: v4 random forest** | **0.811** | **0.621** | 0.617 | **0.755** | **0.857 / 0.809** | 4.5 ms |
| 7-phase, along: v3 logistic | 0.662 | 0.536 | 0.538 | 0.564 | 0.761 / 0.692 | 0.6 ms |
| **7-phase, along: v4 random forest** | **0.718** | **0.599** | 0.579 | **0.633** | **0.786 / 0.740** | 4.5 ms |
| Rules, 7-phase (side / along) | 0.423 / 0.412 | 0.297 / 0.285 | 0.318 / 0.284 | 0.313 / 0.329 | - | - |

- The 4 stages are: lying or sitting up / sitting upright in bed / at the edge (shifting, sitting, stand attempt) / out of bed. The same predictions are simply grouped.
- **How v4 was chosen:**
  - Leave one person out on the 5 training people only, with the staff not used. Mean macro-F1 was: side, logistic 0.54, gradient boosting 0.67, random forest 0.69; along, logistic 0.61, gradient boosting 0.66, random forest 0.65.
  - The forest was then cut from 300 to 60 trees for speed (18.8 ms → 4.5 ms; gradient boosting spiked to 80-135 ms on the Jetson).
  - The lighter variants were compared on the staff, so the staff number for v4 is slightly optimistic.
- Smoothing the predictions over time (a 0.5-1 s causal average, or a causal hidden Markov model) did not help, so the errors are real confusions between neighbouring phases, not flicker.
- **Learning curve** (v3, mean over subsets of the training people), macro-F1 with 1/2/3/4/5 people:
  - side 0.42 / 0.47 / 0.52 / 0.54 / 0.56, levelling off;
  - along 0.42 / 0.45 / 0.49 / 0.52 / 0.54, still rising.
- The weakest phases with v4: SHIFTING_TO_EDGE (F1 0.23 on both views) and ATTEMPTING_STAND on the side view (0.03, only 201 frames).
- `configs/dashboard_jetson.yaml` now shows the v4 models. Alerts never read them (section 10).

### 12d. Speed and resources (live D435f, empty room, 90 s per run; pose model alone on 120 frames)

| Setup | FPS | Camera-to-decision, median | Board power | Junction temperature |
|---|---|---|---|---|
| Depth on (default dashboard) | **5.3** | 240 ms | 10.3 W | 67 C |
| Colour only (no depth) | **16.9** | 116 ms | 12.6 W | 68 C |
| Recorded clip, person in view, depth | 15.3 | - (no camera wait) | 12.1 W | 68 C |
| Pose model alone, CUDA | 29.1 | 34 ms per frame (p99 58 ms) | - | - |
| Pose model alone, CPU | 6.2 | 162 ms per frame | - | - |

- Time per frame with depth on (median):
  - camera read 86 ms;
  - depth alignment 86 ms;
  - pose 77 ms;
  - depth heights 8 ms;
  - JPEG 8 ms;
  - alert clip buffer 5 ms.
- On the recorded clip, the processing time per frame was 60 / 72 / 77 ms with 0 / 1 / 2 people in view.
- Process memory was about 1.2 GB, and the first frame arrived 3.5 s after start.
- The trained models cost 0.5 ms (posture) and 4.5 ms (v4 phase) per prediction.
- The FPS shown on the page now matches the real rate (5.36 shown against 5.32 measured).

### 12e. Bed exit: can it go higher?

- **Exit alerts (rules) are close to what these recordings allow.**
  - On the staff clips: D435i 7/7, D435f 5/7, no false exits.
  - The two D435f misses are a 27 Oct clip whose bed outline was never redrawn for that mount (`h2_01`) and a track that disappeared (`h3_01`). Both are recording or calibration problems, not the logic.
  - Section 10 found that alerts driven by the models do worse than the rules.
- **7-phase labelling had algorithm headroom, now taken.** The random forest adds about 6-11 points of accuracy and 0.06 of macro-F1 on unseen staff.
- **What is left is mainly data.**
  - The along view is still improving with every person added.
  - The short phases (shifting to the edge, stand attempt) have only 200-320 frames each on the staff test.
  - The next gains come from more bed-exit recordings, especially of those transitions, and from a side-on camera.

## 12. Early warning: warn on the start of standing up, not on every sit-up (3 Nov)

**Problem:** the early warning in section 11 fired on 14-38 no-exit sit-ups an hour, with precision about 0.5. Stricter versions of the same sit-up trigger (longer hold, higher CUSUM threshold, stronger rise, must be at the edge) cut warned exits as fast as false warnings; precision stayed near 0.5 (`~/.ahfd-work/ew_tune_20261103/`).

**Exploration (development clips only):** 19 trigger ideas on recorded per-frame engine state, both views (`~/.ahfd-work/ew_explore_20261103/`):
- sitting at the edge for 1-8 s;
- moving toward the edge for 2-5 s;
- the stand-attempt phase for 0.3-2 s;
- the torso rising at the edge, with or without the knee opening;
- combinations.

Sitting at the edge for a few seconds helped a little from the side (2 s: precision 0.62, 18 per hour) but failed along the beds. The stand-attempt phase alone almost never fired before an exit.

**Chosen trigger (`early_warning_trigger: rise_at_edge`, now the default):** at the bed edge, the torso rises 0.10 m **and** the knee opens 20 deg within 1 s, which is the start of standing up. It fires once, then re-arms after 5 s back in bed away from the edge. The old trigger stays available as `early_warning_trigger: sit_up`.

| Clips | View | Trigger | Exits warned | Before the stand attempt (median) | No-exit warnings (per h) | Precision |
|---|---|---|---|---|---|---|
| Development | side | sit-up (old) | 10/12 | 28.3 s | 10 (22.7) | 0.50 |
| Development | side | **rise at edge** | 8/12 | 12.4 s | **6 (13.6)** | 0.60 |
| Development | along | sit-up (old) | 0/12 | - | 0 | - |
| Development | along | **rise at edge** | **12/12** | 21.5 s | 5 (11.6) | 0.80 |
| Staff, all 3 (locked, used) | side | sit-up (old) | 4/7 | 16.5 s | 4 (24.8) | 0.50 |
| Staff, all 3 (locked, used) | side | **rise at edge** | 4/7 | 5.2 s (7.1 s before leaving the bed) | **1 (6.2)** | **0.80** |
| Staff, all 3 (locked, used) | along | sit-up (old) | 0/7 | - | 3 (18.7) | 0.00 |
| Staff, all 3 (locked, used) | along | **rise at edge** | 4/7 | -1.3 s (0.2 s before leaving the bed) | 1 (6.2) | 0.80 |

- **Trade:** about four times fewer false warnings on the staff clips, and the same number of exits warned from the side. The cost is lead time: about 5-9 s before standing from the side instead of about 15 s.
- From along the beds it now catches exits, but only at the moment of standing, so it adds little over the bed-exit alert there. Use a side-on camera for the early warning.
- **Not an independent result:** the trigger and thresholds were chosen on the development clips (19 candidates), and the staff clips were already used on 2 Nov. The built-in version reproduces the exploration exactly (development side 8/12, 13.6 per hour; staff side 4/7, 1 no-exit warning).
