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

## 4. Takeaways

- **Falls:** the current system catches 5/5 on both cameras, 4 of them before landing. The cost is false alarms: about 30 per hour of active movement on the D435f and 60 on the D435i, nearly all from POSSIBLE_FALL on sit-downs.
- **Pre-impact rule:** on the D435f it gives about the same warning time (-0.86 s) with 1 false alarm instead of 9. Replacing POSSIBLE_FALL with it, keeping the floor alert as the safety net, gives 5/5 caught, 4 before landing, and 6 false alarms (13.7 per hour) on the D435f.
- **Bed exit:** sustained exits are reliable on both cameras: 20/24, no false exits except one on the D435f, typically 2-4 s after the person is up. Brief stand-ups are not caught.
- **Bed-exit classification:** the trained 7-phase model beats the rules on unseen staff. Side view: accuracy 0.70 vs 0.42, macro-F1 0.56 vs 0.30. Along view: 0.66 vs 0.41, macro-F1 0.54 vs 0.28. The short transition phases (SHIFTING_TO_EDGE, ATTEMPTING_STAND, TORSO_RISING) stay weak, at F1 0.14-0.60, because there is little labelled data for them. The posture model reaches accuracy 0.92 and macro-F1 0.94 on unseen staff.
