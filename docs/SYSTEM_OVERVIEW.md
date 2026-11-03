# How the system works: an overview (4 Nov 2026)

Privacy-preserving fall and bed-exit detection (CDE3301 capstone with Alexandra Hospital).

The system has three layers:
1. **Deep learning** turns each video frame into a stick figure.
2. **Rules and thresholds** turn the stick figure into **alerts**.
3. **Small machine-learning models** (a random forest and a decision tree) label what the patient is doing, **on screen only**.

Current numbers are in `docs/BENCHMARK.md`, section 12.

---

## 1. The pipeline

```
Camera (Intel RealSense D435i / D435f, mounted high, looking down at the beds)
   │  colour image 1920x1080 (+ optional depth)
   ▼
1. Pose model: RTMO-s (deep learning, pretrained)   -> 17 body keypoints per person
   │  the video is discarded here; only keypoints are kept
   ▼
2. Tracker + smoothing                              -> the same ID for each person over time
   ▼
3. Feature extraction (geometry + calibration)      -> heights, speeds, angles, bed position
   ▼
   ├──► 4a. Rule engines (state machines)           -> ALERTS
   │        - fall detector
   │        - bed-exit detector
   │
   └──► 4b. Trained models (machine learning)       -> LABELS ON SCREEN
            - 4-state posture (decision tree)
            - 7-phase bed activity (random forest)
   ▼
5. Dashboard + ward page: alerts, stick figures, labels; alert clips are stick figures only
```

Hardware: NVIDIA Jetson Orin NX 16 GB. Everything runs on the device, with no cloud.

---

## 2. Each step in detail

### Step 1: pose estimation (deep learning, not trained by us)
- **RTMO-s** is a neural network trained by its authors on large public datasets. We use it as it is.
- For every person it finds **17 keypoints** (nose, shoulders, elbows, wrists, hips, knees, ankles and so on), each with a confidence score.
- It takes about 35 ms per frame on the Jetson GPU.
- It was chosen in a bake-off against RTMPose and YOLO26s:
  - RTMPose missed reclined people.
  - YOLO26s was slower, lost the patient's ID at stand-up, and is AGPL-licensed.
- **Privacy:** after this step only keypoints are kept. Storing video needs three separate switches, and is used only for consented review videos, which stay on the external SSD.

### Step 2: tracking and smoothing (simple algorithms)
- A tracker keeps the same ID on a person from frame to frame and survives gaps of up to 1 s.
- A One-Euro filter smooths keypoint jitter.

### Step 3: feature extraction (geometry, no learning)
Each camera has a **calibration** file holding:
- its height, tilt and lens values;
- each bed drawn as a box on the floor, with its height.

With that, the keypoints are turned into real-world measurements:
- **torso height** above the floor (m), and its **vertical speed** (m/s);
- **trunk angle** (0° upright, 90° lying) and **trunk rotation speed** (°/s);
- **knee, hip and leg angles** (straight legs mean standing);
- **floor spread**: how stretched out on the floor the body is;
- **bed support** (is the bed holding the body up?) and **distance to the bed edge**.

### Step 4a: detection with rules and thresholds (what raises alarms)
Each detector is a **state machine**: a set of states (upright, falling, on the ground, …) with threshold rules for moving between them.

**Fall detector**

| Alert | Rule |
|---|---|
| **POSSIBLE FALL** (earliest) | torso drops ≥ 0.15 m within 0.5 s **and** the trunk tips over at ≥ 80°/s |
| **FALL SUSPECTED** | lands on the floor after a fast drop (0.3 s after impact), or reads as on the floor for 1.5 s with no impact |
| **FALL CONFIRMED** | still down and not moving 8 s after landing |
| **PERSON DOWN** | on the floor for 20 s (slow slump) |
| **NEAR MISS** | got back up. Still reported: a fall-risk patient falling is never acceptable |

The fast-drop trigger fires at a vertical speed of −0.9 m/s or less for 3 frames, or a 0.45 m drop within 0.8 s.

**Bed-exit detector**
- Follows the patient through the 7 bed phases using rules: trunk angle, distance to the bed edge, bed support, and a change detector (CUSUM).
- **BED EXIT:** the patient is standing and no longer supported by the bed, after having been in bed for at least 2 s.
- **EARLY WARNING** (High-risk beds only): at the bed edge, the torso rises 0.10 m and the knee opens 20° within 1 s, which is the start of standing up.
- **ID handover:** if the tracker gives the patient a new ID at the bed edge, the bed history is passed to the new ID.
- **Per-bed modes:**
  - Low: falls only.
  - Medium: falls and bed exits.
  - High: falls, bed exits and early warnings.

**Why rules decide the alerts:**
- Every alert can be explained to a nurse, for example "torso dropped 0.9 m in 0.6 s, on the floor beside Bed 2".
- They need very little fall data to work.
- When tested, alerts driven by the trained models were worse than the rules (benchmark section 10).

**How the thresholds were set:** by sweeping values on the development clips and picking the best trade-off between early detection and false alarms. For example, the 80°/s trunk-tipping value came from testing 40-130°/s, and it removed the false alarms caused by people sitting down. All thresholds live in YAML config files, not in the code.

### Step 4b: classification with machine learning (labels on screen)

| Model | Algorithm | Predicts | Inputs | Accuracy on unseen staff |
|---|---|---|---|---|
| **Posture** | Decision tree (depth 5) | in bed / sitting / standing / on the ground | 30 per-frame features | 91% (macro-F1 0.93) |
| **7-phase bed activity** | **Random forest** (60 trees), one model per camera view (side / along the beds) | lying → sitting up → sitting upright → moving to edge → sitting at edge → trying to stand → out of bed | 958 features: per-frame values plus summaries over the last few seconds | side 81%, along the beds 69% |

These models are **display only**. They label what the patient is doing on the dashboard and never trigger alerts.

**Built and measured, but not connected to the dashboard:**
- **Edge-sitting warning:** gradient-boosted trees that spot "legs over the bed edge" to give an early "go and check" notice.
- **Pre-impact fall detector:** a balance rule (Hof's margin of stability) plus a gradient-boosted model. It aims to flag a fall up to about 1 s before landing.

---

## 3. How the models were trained

1. **Recording:** volunteers acted out scenarios in the hospital lab, filmed by both cameras: lying, sitting up, getting out of bed, staged falls onto crash mats, and look-alikes such as kneeling and lying down on purpose.
2. **Labelling:** each clip was labelled by hand with the labelling tool:
   - the 7 bed phases;
   - the 4 postures;
   - the moment of each fall impact.
3. **Features:** computed from the stored keypoints using only past frames, exactly as the live system would.
4. **Training data:**
   - pilot volunteer (8 clips);
   - 3 students (27 Oct);
   - fall student 01 (29 Oct).
5. **Choosing the algorithm:** "leave one person out". The model is trained on all people but one, tested on the one left out, and this is repeated for each person. This picked the random forest over logistic regression and gradient boosting. Its size was then cut to 60 trees, so it predicts in 4.5 ms per frame.
6. **Test (never trained on):**
   - staff 2 and 3 for bed activity and bed exits;
   - fall student 02 for falls.
7. **Tools:** Python and scikit-learn on the Jetson. Models are saved as `.joblib` files in `models/`.

Which recordings are used for evaluation:
- The **pilot clips 01-08** are not used for evaluation, because their depth data is faulty. They are still used for training.
- The test clips were used once, as a locked test on 2 Nov. Later numbers on them are post-hoc checks, not independent tests.

---

## 4. How it is evaluated

- **Recognition (frame by frame):**
  - accuracy, macro-F1, balanced accuracy, Cohen's kappa;
  - per-class precision and recall, and confusion matrices.
- **Alerts (events):**
  - sensitivity (events caught);
  - precision (alerts that were real);
  - false alarms per hour;
  - timing before or after the event.
  - Each fall alert counts if it comes between 2.5 s before and 30 s after the labelled impact. Each bed exit counts if it comes between 5 s before and 15 s after the labelled exit.
  - Alerts less than 10 s apart count as one.
  - Kneeling and bending low count as useful, not false: a team member reviewed every test alert by hand and judged these worth a nurse's look.
- **Speed:** FPS, camera-to-decision delay, power, temperature.

---

## 5. Current results (summary of BENCHMARK.md section 12)

**Fall detection**

| | D435f | D435i |
|---|---|---|
| Development falls (5) | 5/5 caught, 4 before landing, typically 0.6 s before; 0 false alarms per hour | 5/5, 4 before landing, 0.4 s before; 0 per hour |
| Test falls (4), post-hoc | 4/4, 4 before landing, 0.5 s before | 4/4, 1 before landing, 0.2 s after |
| Test false alarms | **0** (its only alerts away from a fall were kneeling and bending near the floor, which count as correct alerts) | **0** (same) |

**Bed exit**

| | D435f | D435i |
|---|---|---|
| Development exits | 5/8, about 4 s after leaving the bed | 7/8, about 2 s after |
| Staff test exits (post-hoc) | **4/4**, no false exits | **4/4**, no false exits |

**Recognition (2 unseen staff)**

| Model | Accuracy | Macro-F1 | Kappa |
|---|---|---|---|
| Posture | 91.2% | 0.926 | 0.87 |
| 7-phase, side view | 80.8% | 0.606 | 0.75 |
| 7-phase, along the beds | 68.8% | 0.519 | 0.59 |

**Speed:**
- 5.3 FPS with depth on, 16.9 FPS colour only;
- about 240 ms from camera to decision;
- about 10 W of power, 1.2 GB of memory.

**Limits:**
- The staged falls come from healthy students on crash mats, not patients.
- There are few people in the data: 1 fall student per split and 2 test staff.
- Labels are in whole seconds, so all timings are accurate to about ±1 s.
- About 3 hours of normal activity would be needed to show 1 false alarm per hour or less.

---

## 6. In one sentence

**Deep learning sees the person, geometry measures the body, rules decide the alerts, and small ML models (random forest, decision tree) label the activity.**

The rules handle the safety-critical part because they are explainable and need little data. The ML handles the fine-grained labelling, where it clearly beats the hand-written rules: about 69-81% against 41% for the 7 phases.

---

## 7. Where things are in the repository

| What | Where |
|---|---|
| Pose, tracking, features | `src/ahfd/pose/`, `src/ahfd/track/`, `src/ahfd/features/extractor.py` |
| Fall detector (rules) | `src/ahfd/detect/state_machine.py` |
| Bed-exit detector (rules) | `src/ahfd/detect/bed_exit.py` |
| Prototypes | `src/ahfd/detect/preimpact.py`, `src/ahfd/detect/edge_warning.py` |
| Trained-model training code | `src/ahfd/ml/` |
| Thresholds and settings | `configs/*.yaml` |
| Camera calibrations and bed boxes | `calib/*.yaml` |
| Labels and keypoint tracks | `data/` |
| All results | `docs/BENCHMARK.md` |
