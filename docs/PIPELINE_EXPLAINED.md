# The fall and bed-exit detection system, explained from zero (9 Nov 2026)

This document is for a reader with **no background**: no machine learning, no computer vision, no knowledge of the project. Every term is defined the first time it appears, and there is a glossary at the end (section 12).

It explains:
- what each step of the system does and how it works inside;
- what each alternative was and why we chose what we use;
- the history: what we tried, what failed, and how each problem was fixed;
- how every number in the report is measured.

All numbers come from `docs/BENCHMARK.md` (section 12 holds the current results) or from scripts named next to them.

---

## 0. The problem in one page

**Who it is for:** patients at Alexandra Hospital who are at risk of falling.

**The two events nurses care about:**
1. **A fall.** The patient ends up on the floor. A fallen elderly patient who lies there for a long time can be seriously harmed, so speed matters.
2. **A bed exit.** A high-risk patient getting out of bed alone. Most in-hospital falls happen around getting out of bed, so knowing early lets a nurse get there first.

**Constraints:**
- **Privacy.** We may not keep video of patients. The system must work from a camera but throw the images away immediately.
- **Runs on a small computer next to the bed.** An NVIDIA Jetson Orin NX, about the size of a paperback book. There is no cloud and no internet needed.
- **Few false alarms.** If it cries wolf, nurses switch it off. This is called *alarm fatigue*.
- **Little data.** We cannot film real patients falling. We have staged falls by healthy students onto crash mats, and scripted bed exits.

**The answer in one sentence:**
> A pretrained neural network turns each video frame into a stick figure, and the video is discarded. Geometry turns the stick figure into real measurements (heights in metres, speeds, angles). **Hand-written rules** decide when to raise an alert. **Small trained models** (a decision tree and a random forest) label what the patient is doing, on screen only.

---

## 1. The pipeline at a glance

```
Camera (Intel RealSense D435i or D435f, mounted high, looking down at the beds)
   │  colour image 1920x1080, ~15 times a second, plus a depth image
   ▼
Step 1  Pose estimation: RTMO-s neural network          -> 17 body points per person ("keypoints")
   │    (the image is thrown away here)
   ▼
Step 2  Tracking + smoothing                             -> same ID for the same person over time
   ▼
Step 3  Feature extraction (geometry, calibration, depth)-> heights, speeds, angles, bed position
   ▼
   ├──► Step 4a  Rule engines (state machines)          -> ALERTS (fall, bed exit, early warning)
   └──► Step 4b  Trained models (tree, random forest)   -> LABELS on screen (posture, 7 phases)
   ▼
Step 5  Dashboard and ward page                          -> nurse sees the alert, a stick-figure clip, the bed
```

The sections below go through each step and answer, for each: what is it, how does it work, why this one, and what else we tried.

---

## 2. The camera

### What it is
An **Intel RealSense D435** is a camera that gives two pictures at once:
- a normal **colour image** (RGB = red, green, blue);
- a **depth image**: for every pixel, how far away that point is, in millimetres.

It measures depth with **stereo vision**, two lenses a few centimetres apart, like human eyes. A nearby object appears shifted between the two images more than a far one, and that shift gives the distance. A built-in infrared dot projector paints a pattern on blank surfaces so there is always something to match.

We have two:
- **D435i:** has an IMU, a motion sensor.
- **D435f:** has a filter that cuts infrared light.

Both are mounted high (about 2.6 m) and tilted down (about 30°).

### Two viewpoints
- **Side view:** the camera looks across the bed, from the patient's side.
- **Along view:** the camera looks along the bed, from the head or foot end.

The two cameras swapped positions on 29 Oct, so we always report by **view**, not camera name, when the view matters.

| Recording | Side view | Along view |
|---|---|---|
| Pilot (15 Oct), 27 Oct students | D435f | D435i |
| 29 Oct (falls, staff) | D435i | D435f |

**Finding:** the side view is better for labelling the bed phases (80.8% vs 68.8% accuracy) and for the early warning. Bed-exit alerts tie (6/8 each on the development clips).
- **Why:** from the side, the legs coming over the bed edge are visible.
- From along the bed, the legs are hidden behind the bed and a lying body is foreshortened: it looks short and steep.

### Calibration: turning pixels into metres
A camera only gives pixels. To say "the torso is 0.3 m above the floor" we need to know where the camera is. That is the **calibration** file (`calib/*.yaml`), which holds:
- **lens values** (focal length, image centre): how pixels map to directions;
- **mount height and tilt**: where the floor is relative to the camera;
- **bed outlines**: each bed drawn as a box on the floor, plus its mattress height (about 0.6 m).

With these, a pixel can be turned into a ray from the camera. Where that ray meets the floor or the bed gives a real position. This is called *projective geometry*.

### Frame rate
**Frame rate (FPS, frames per second)** is how many images are processed each second.
- The camera itself delivers up to 30 FPS.
- Live, with depth turned on, the whole system processes about **5 FPS**: aligning the depth image to the colour image and the pose network each take about 80 ms.
- Colour only, it processes about **17 FPS**.

On 6 Nov we found the possible-fall alert could never fire at 5 FPS (section 6.4) and fixed it.

---

## 3. Step 1: pose estimation (deep learning)

### What is "pose estimation"?
Given an image, find each person and the positions of their joints. The output is **17 keypoints** per person:
- nose, eyes, ears;
- shoulders, elbows, wrists;
- hips, knees, ankles.

Each keypoint has a **confidence score** from 0 to 1. Joined up, they make a stick figure.

### What is "deep learning"? What is a neural network?
- A **neural network** is a very large mathematical function with millions of adjustable numbers called *weights*. It takes the image's pixel values in and produces numbers out, here keypoint positions.
- **Training** means showing it hundreds of thousands of images where humans have marked the joints. After each one the weights are nudged so its answer is a little closer to the human's. This nudging is called *gradient descent*.
- **Deep** means it has many layers stacked: early layers detect edges, middle layers detect limbs, late layers detect whole people.
- Deep learning is the only practical way to find body joints in raw images. Writing rules for "what an elbow looks like" by hand is hopeless.

**We did not train the pose network.** We use **RTMO-s** exactly as its authors (OpenMMLab) published it. They trained it on large public datasets (COCO and others, together called "body7"). This is called using a *pretrained model*. We could not have trained a better one: that needs huge labelled datasets and weeks of GPU time.

### RTMO-s, and what we compared it with
| Model | What it is | How it works |
|---|---|---|
| **RTMO-s** (chosen) | "Real-Time Multi-person One-stage", small version | Finds all people and all their joints **in one pass** over the image (one-stage). |
| RTMPose-lightweight | "Real-Time Multi-person Pose", top-down | **Two stages:** first a person detector draws a box around each person, then a second network finds joints inside each box. |
| YOLO26s-pose | The YOLO family ("You Only Look Once"), pose version | One-stage, like RTMO. |

**Bake-off (6 Nov, the same 8 pilot clips and the same rules for all three):**

| Measure (D435f / D435i) | RTMO-s | RTMPose | YOLO26s |
|---|---|---|---|
| Patient tracked, all frames | **94% / 91%** | 35% / 48% | 67% / 70% |
| Patient tracked **while lying in bed** | **89% / 84%** | 2% / 30% | 78% / 71% |
| Patient tracked once **out of bed** | **100% / 100%** | 83% / 50% | 5% / 45% |
| Correct bed phase, all frames | **60% / 44%** | 17% / 32% | 46% / 44% |
| Bed exits caught, false exits | 4/4, 0 | 4/4, 0 | 4/4, 0 |
| Speed on the Jetson | 34–43 ms per frame | (not measured) | 1.6× slower than RTMO-s |
| Licence | Apache-2.0 (free for any use) | Apache-2.0 | AGPL-3.0 (forces anything built on it to be open-sourced) |

**Why RTMO-s:**
- **RTMPose fails on reclined people.** Its first-stage person detector was trained mostly on upright people and does not draw a box around someone lying under a blanket. With no box, the second stage never runs. A bed monitor that cannot see a lying patient is useless.
- **YOLO26s loses the patient at stand-up.** Once the patient is out of bed, it keeps them on the same track in only 5–45% of frames. It is also slower, and its licence is restrictive.
- **RTMO-s keeps the patient through every phase**, is fast, and has a permissive licence.
- **Honest note:** all three caught all 4 bed exits. The difference is in continuous tracking (needed for labelling and falls), not in the exit alert.
- *Script: `~/.ahfd-work/pose_bakeoff_20261106/bakeoff.py`. One volunteer, 8 short clips, so this is a development comparison, not a validation.*

**TensorRT** is NVIDIA's tool to make networks run faster. We tried it through onnxruntime, but it fell back to the CPU or crashed, so we use plain CUDA (the GPU).

### Privacy happens here
- After this step the image is discarded. Only the 17 points per person are kept: no face, no clothes, no skin.
- Saving any video requires **three separate switches**: a config setting, an environment variable, and a command-line flag. It is used only for consented staged volunteers, and kept on an external drive.
- The alert clips the nurse sees are **stick-figure animations**, not video.

---

## 4. Step 2: tracking and smoothing

### Tracking
The pose network looks at each frame independently. It does not know that the person in this frame is the same as the person in the last one. **Tracking** links them so each person gets a stable **ID**, such as "track 3".

How our `SimpleTracker` works:
1. Draw a box around each person's keypoints.
2. Match new boxes to existing tracks by **IoU** (intersection over union): how much two boxes overlap, from 0 (none) to 1 (identical). A match needs IoU ≥ 0.3.
3. Leftover people are matched to leftover tracks if their box centres are close.
4. A track survives for **1 second** without being seen before it is dropped.
   - Originally it was 15 frames, which at 30 FPS is only 0.5 s. That was too short: brief pose dropouts gave patients new IDs.

**Why IDs matter:** the bed-exit detector remembers that "track 3 was in bed". If the patient gets a new ID as they stand up, the new ID was never in bed, and the exit is missed.
- **Fix (3 Nov), "ID handover":** a new track that appears within **4 s** and **1 m** of a lost in-bed track inherits its bed history.
- Found by tracing missed staff exits; it changed no result on our clips but protects against pose dropouts at the bed edge.

### Smoothing: the One Euro filter
Keypoints jitter by a few pixels from frame to frame even when a person is still. Speed is the change in position over time, so jitter makes speeds noisy.

The **One Euro filter** (Casiez et al., CHI 2012) is a smoothing filter that adapts to speed:
- When a joint moves **slowly**, it smooths heavily, which removes jitter.
- When a joint moves **fast**, it smooths lightly, which avoids lag. A fall must not be delayed by the filter.

It has three settings, kept at the published defaults:
- `min_cutoff` 1.0: smoothing when still;
- `beta` 0.007: how quickly it lightens with speed;
- `d_cutoff` 1.0: smoothing of the speed estimate itself.

**Why not a simple moving average:** a moving average smooths everything equally, so it would either leave jitter or delay falls.

---

## 5. Step 3: feature extraction (geometry, no learning)

A **feature** is a single number that describes something useful about the person at this moment. We compute them with geometry, not learning. The main ones:

| Feature | Meaning | Used for |
|---|---|---|
| `h_torso` | Height of the torso centre (middle of shoulders and hips) above the floor, in metres | Standing ≈ 1.0–1.3 m; sitting lower; on the floor ≈ 0.1–0.4 m |
| `v_z` | Vertical speed of the torso, m/s (negative = going down) | Detects a fast drop |
| trunk tilt | Angle of the shoulder-to-hip line: 0° upright, 90° lying | Lying vs sitting vs standing |
| trunk rotation speed | How fast that angle changes, degrees per second | Tipping over (a fall) vs sitting down (stays upright) |
| knee, hip and leg angles | Joint angles from three keypoints | Straight legs = standing; bent = sitting |
| `floor_spread` | How stretched out along the floor the body is | Lying on the floor is long and flat |
| bed support | How much of the body (hips especially) is over the bed outline | In bed vs off the bed |
| edge distance | Distance from the body to the nearest bed edge, in metres | Moving towards the edge |

### Where do heights come from? Colour geometry vs depth
There are two ways to get a joint's height:
1. **Colour-only geometry:** assume the feet are on the floor and use the calibration to place the person in 3D. This works for standing people but fails for someone in bed, whose feet are not on the floor.
2. **Depth:** the depth image gives the real distance to each joint, so its 3D position, and so its height, can be computed directly.

**What depth did for us (development clips, same tracks with and without depth):**
- Standing is recognised as upright **100%** of the time.
- No more false "on the ground" readings.
- Bed-phase accuracy improves by **1–8 percentage points**.
- **The cost:** 6–7% of sitting-on-the-bed-edge frames read as standing, because a seated torso is 1.1–1.3 m above the floor, close to standing height.
- **Trouble we found:** on the pilot recordings the depth heights were geometrically inconsistent (a standing torso at 1.8 m, a lying torso below the mattress).
  - We traced this to the depth camera's own pose estimate (its "floor fit") differing from the colour calibration by about 4°.
  - It was fixed for later sessions by storing a separate `depth_ground` pose in each calibration.
  - The pilot clips are therefore excluded from fall evaluation, but still used for training.

**Why not depth only, with no colour?** The pose network needs the colour image to find joints. Depth gives *where* each joint is in 3D once we know *which pixel* is the elbow.

---

## 6. Step 4a: the rule engines that raise alerts

### What is a "rule engine" or "state machine"?
A **state machine** is a diagram of states connected by arrows. Each arrow has a condition. For example: "UPRIGHT → FALLING if the torso drops faster than 0.9 m/s for 3 frames". The system is always in exactly one state per person, and alerts are raised on certain transitions.

The conditions use **thresholds**: fixed numbers like 0.9 m/s or 80°/s, chosen from data (section 6.3).

### Why rules and not a trained model for alerts?
1. **Explainable.** Every alert can be explained to a nurse: "torso dropped 0.9 m in 0.6 s, on the floor beside Bed 2". A neural network or forest cannot say why.
2. **Need very little fall data.** We have 9 staged falls in total. You cannot train a reliable learned fall detector on 9 examples, but a rule built from physics ("a fall is a fast drop that ends low") works from the start.
3. **We tested the alternative, and it was worse** (3 Nov, BENCHMARK section 10). We turned the trained models' probabilities into alerts and compared on the same clips:

   | | Rules | Trained model as alarm |
   |---|---|---|
   | Falls caught, D435f / D435i | 4/4 / 4/4 | 4/4 / 4/4 |
   | False fall alarms (no risk) | **0 / 0** | 2 / 1–2 |
   | Bed exits caught | **5/5 / 5/5** | 4/5 / 4/5 |

   The models are better at labelling *every frame* (which is dominated by long, easy stretches like lying), but worse at the critical 1–2 seconds that decide an alarm.

### 6.1 The fall detector
| Alert | When it fires | Purpose |
|---|---|---|
| **POSSIBLE FALL** (earliest) | The torso drops ≥ 0.15 m within 0.5 s **and** the trunk tips over at ≥ 80°/s | Warn *as the person is going down*, before they land |
| **FALL SUSPECTED** | Lands on the floor after a fast drop (0.3 s after impact), or reads as on the floor for 1.5 s with no impact | A person on the floor |
| **FALL CONFIRMED** | Still down and not moving 8 s after landing | Escalation |
| **PERSON DOWN** | On the floor for 20 s (a slow slump with no impact) | Catches slow slides to the floor |
| **NEAR MISS** | Fell and got back up | Still reported: a fall-risk patient falling is never acceptable |

The fast-drop trigger fires at a vertical speed of −0.9 m/s or faster for 3 frames, or a 0.45 m drop within 0.8 s.

**Detection, not prediction.** Nothing here predicts a fall before it starts. POSSIBLE FALL means the body has already started going down. The only earlier signal is the bed-exit early warning, which flags the risky moment before a possible fall.

### 6.2 The journey: early alerts, too many false alarms, and how we fixed it
This is the story the user asked about: how we got an alert *before* landing, why it first raised many false alarms, and how we brought them down.

**Stage 1: alert only after landing.**
- The original design (8 Sep) alerted only after the person had been down and still for **8 s** (FALL CONFIRMED).
- Our staged falls stayed down only 3–7 s, so recall could not even be measured.
- **Fixes:**
  - one "not down" frame no longer cancels a fall;
  - FALL SUSPECTED now alerts 0.3 s after impact, or after 1.5 s down with no impact (policy, 31 Oct–1 Nov).

**Stage 2: add an early alert (POSSIBLE FALL, 1 Nov).**
- **Rule:** the torso drops 0.15 m in 0.5 s from standing height.
- **Result on the development clips:** 4 of 5 falls flagged before landing, typically **0.4–0.6 s before**, up to 1.6 s.
- **Problem:** many false alarms. People **sitting down** drop their torso just as fast as a falling person. We measured about 13–15 false alarms per hour of active movement (pilot clips excluded), and 30–60 per hour with the faulty pilot clips included.

**Stage 3: what we tried to remove the false alarms (3 Nov, development clips only).** Each idea was tested by replaying the stored keypoints with the setting changed and counting falls caught and false alarms. This is called a **threshold sweep**.

| Idea | Settings tested | Outcome |
|---|---|---|
| Raise the drop threshold | 15 settings, 0.15–0.5 m over 0.5 or 0.8 s | Rejected: it lost the early warning before it removed the sit-downs |
| A balance-based rule (Hof's "margin of stability": is the body's balance point outside the feet?) | with and without the possible-fall agreeing | Good on the D435f but caught only 1 of 4 falls on the D435i in the locked test; kept as a prototype only |
| A trained "pre-impact" model (gradient-boosted trees) | thresholds | Similar timing, but more false alarms than the rules; prototype only |
| Bypass for very fast vertical speed | - | Rejected: added false alarms |
| Require several frames before the floor alert | - | Rejected: delayed the slow-slump fall from +2.0 s to +3.7 s |
| **Trunk-tipping gate** | rotation 40–130°/s within 0.5 s of the drop | **Chosen at 80°/s** |

**Why the tipping gate works:** sitting down lowers the torso, but the trunk **stays upright**. A fall **tips the trunk over**. Requiring the trunk angle to turn at ≥ 80°/s separates the two. 80°/s was the value in the sweep that removed the sit-down false alarms without losing early detection on the development clips.

**Result:**

| | Before the gate | After the gate (80°/s) |
|---|---|---|
| Falls caught (development) | 5/5 per camera | 5/5 per camera |
| Caught before landing | 4 per camera | 4 per camera (same) |
| False alarms per hour (development, pilot excluded) | 15.3 / 13.0 | **0 / 0** |
| Locked clips, post-hoc | 0 | **0**; but on the D435i 3 of 4 falls lost their before-landing alert (still caught at 0 to +0.8 s) |

**Is this hyperparameter tuning?** Yes, in the simple sense.
- A **hyperparameter** is a setting chosen by the designer, not learned automatically: here a threshold such as 80°/s.
- We chose it by a **grid sweep**: try a list of values, score each, and pick the best trade-off.
- We did this **only on development clips**, never on the test clips, so that the test results stay honest.

All thresholds live in YAML config files, not in the code:

| Threshold | Value | How it was set |
|---|---|---|
| Possible-fall drop | 0.15 m in 0.5 s | Sweep (1 Nov) |
| Trunk-tipping gate | 80°/s within 0.5 s | Sweep 40–130°/s (3 Nov) |
| Fast drop (FALL SUSPECTED path) | v_z ≤ −0.9 m/s for 3 frames, or 0.45 m in 0.8 s | Original design (8 Sep), hand-chosen from physics |
| Floor alert without impact | 1.5 s | Policy: a fall-risk patient on the floor must alert quickly |
| FALL CONFIRMED | 8 s still | Original design, a clinical choice |
| PERSON DOWN | 20 s | Original design |
| Bed exit: far outside the bed | 0.30 m | Hand-set on early clips (26 Oct) |
| Bed exit: must have been in bed | 2 s | Hand-set (26 Oct) |
| Early warning: rise at edge | 0.10 m torso rise + 20° knee opening within 1 s | Chosen among 19 candidate triggers (3 Nov) |
| One Euro smoothing | 1.0 / 0.007 / 1.0 | Published defaults |

**Known blind spot:** a fall straight towards or away from the camera hardly changes the trunk angle *in the image*, so it gets no early alert. It is still caught by the floor alert after landing.

### 6.3 The bed-exit detector
It follows the patient through **7 bed phases**:

| Phase | Meaning |
|---|---|
| RECLINED | Lying in bed |
| TORSO_RISING | Sitting up |
| UPRIGHT_IN_BED | Sitting up in bed |
| SHIFTING_TO_EDGE | Moving towards the bed edge |
| EDGE_SITTING | Sitting on the edge, legs over |
| ATTEMPTING_STAND | Starting to stand |
| OUT_OF_BED | Standing or walking away |

It uses trunk angle, edge distance, bed support, and a **CUSUM** change detector.
- **CUSUM** (cumulative sum) adds up small, consistent changes, such as the body creeping towards the edge, until the total crosses a threshold.
- That catches slow, steady movement that no single frame shows.

**Alerts:**
- **BED EXIT:** the patient is standing (straight legs, upright trunk) and no longer supported by the bed, having been in bed for at least 2 s.
- **EARLY WARNING** (High-risk beds only): at the bed edge, the torso rises 0.10 m **and** the knee opens 20° within 1 s, which is the start of standing up.

**History of the early warning:**
1. The first version warned on sitting up. It came 15–45 s early but also fired on every sit-up that did not become an exit: about 28–35 per hour, with precision about 0.5.
2. We then tried 19 trigger ideas and chose "rise at edge". It gives about three times fewer false warnings, at the cost of a shorter lead (about 7 s instead of about 10 s before standing, from the side).

**Per-bed modes:**
- Low: falls only.
- Medium: falls and bed exits.
- High: falls, bed exits and early warnings.

These match how nurses grade a patient's fall risk.

---

## 7. Step 4b: trained models for on-screen labels

These models **only label** what the patient is doing on the dashboard. They never raise alarms (section 6 explains why).

### 7.1 What is machine learning here?
Machine learning means the computer finds the rules itself from **labelled examples**.
- We give it rows of features (the numbers from step 3) and, for each row, the correct answer (the **label**, typed in by a person watching the video). It learns a rule that maps features to labels.
- This is called **supervised learning** and **classification**: the answer is one of a fixed set of classes.

### 7.2 Decision tree: the posture model
A **decision tree** is a flowchart of yes/no questions learned from data, for example:
```
Is torso height < 0.45 m?
 ├─ yes: is the body over the bed?  ── yes: IN BED
 │                                   └─ no:  ON THE GROUND
 └─ no:  is the knee angle > 155°?   ── yes: UPRIGHT
                                     └─ no:  SITTING
```

**How it learns:**
- At each step it tries every feature and every cut-off, and picks the one question that best separates the classes. "Best" is measured by *Gini impurity*: how mixed the classes are on each side.
- Then it repeats on each side, until a maximum depth.
- Our tree has **depth 5**, so at most 5 questions. That keeps it readable and stops it from memorising the training data (*overfitting*).
- Missing values (a joint not seen) are filled with the median of that feature (*median imputation*).

**Our posture model:**
- **Four classes:** in bed / sitting / upright / on the ground.
- **Inputs:** about 30 per-frame features.
- **Speed:** 0.5 ms per prediction.

**Results on 2 unseen staff:**
- 91.2% accuracy, macro-F1 0.926.
- On fall student 02: 95.0% accuracy, and the on-the-ground class is found with precision 1.00 and recall 0.95.
- The hand-written rules score 54.9% accuracy, macro-F1 0.466.

### 7.3 Random forest: the 7-phase model
A **random forest** is **many decision trees that vote**.
- Each tree is trained on a **random sample of the training rows** (*bootstrap sampling*), and at each question it may only choose among a **random subset of the features**.
- So every tree is a bit different and makes different mistakes. When they vote, the mistakes cancel out, and the forest is much more accurate and stable than any single tree.

**Our settings:**
- **60 trees.**
- `min_samples_leaf` = 5: a final answer must rest on at least 5 training examples, which stops trees memorising single frames.
- `class_weight` = balanced: rare phases such as ATTEMPTING_STAND count more during training, so the model does not just predict the common phases.
- Missing values are filled with the median.

**Its inputs: 958 features.**
- The per-frame features, plus **summaries over the last 0.5, 2, 4 and 8 seconds**: average, minimum, maximum, trend, and so on.
- These *temporal features* give the model a sense of motion. "Sitting" and "about to stand" look the same in a single frame, but differ in how they have been changing.
- **Only past frames** are used (they are *causal*), exactly as the live system would see them.

**One model per view** (side, along), because a lying body looks completely different from each.

### 7.4 Why a random forest, and not deep learning?
| Reason | Explanation |
|---|---|
| **Data size** | We have tens of thousands of frames, but from only **5 training people** and short transitions (a stand attempt lasts 1–2 s). Deep networks need far more varied examples. With little data they memorise the people they saw (*overfitting*). Forests work well on small tabular data. |
| **Input type** | Our inputs are already meaningful numbers (heights, angles), a table, not images. On tabular data, tree ensembles routinely match or beat deep networks. |
| **Speed** | The forest predicts in 4.5 ms on the Jetson CPU. The GPU is busy with pose. |
| **Explainability** | Trees can show which features mattered. |

**We compared algorithms fairly with "leave one person out"** (section 9.3), on the 5 training people only:

| Algorithm | Side view, macro-F1 | Along view, macro-F1 |
|---|---|---|
| Logistic regression (a linear model: a weighted sum of features) | 0.54 | 0.61 |
| Gradient boosting (trees built one after another, each fixing the last one's errors) | 0.67 | **0.66** |
| **Random forest** | **0.69** | 0.65 |

- The forest won from the side and tied along the beds.
- Gradient boosting was unpredictable in speed on the Jetson (80–135 ms spikes).
- We cut the forest from **300 trees to 60**: 18.8 ms → 4.5 ms per prediction, with nearly the same accuracy.

**Results on 2 unseen staff:**

| | Side view | Along view |
|---|---|---|
| Random forest (v4, current) | **80.8%**, macro-F1 0.606 | **68.8%**, macro-F1 0.519 |
| Logistic regression (v3, previous) | 68.8%, macro-F1 0.522 | 66.3%, macro-F1 0.507 |
| Hand-written rules | 41.1%, macro-F1 0.271 | 41.0%, macro-F1 0.267 |

**What still limits it: data.**
- Short phases (SHIFTING_TO_EDGE, ATTEMPTING_STAND) have F1 0.0–0.2 from the side, because there are only 180–260 frames of them.
- **Learning curve:** going from 1 to 5 training people, macro-F1 went 0.41 → 0.52 from the side, levelling off, and 0.37 → 0.51 along, still rising. More people should help the along view.
- Smoothing the predictions over time did not help, so the errors are genuine confusions between neighbouring phases, not flicker.

### 7.5 Prototypes built but not connected
- **Bed-edge warning:** gradient-boosted trees that spot "legs over the bed edge" and give a gentle "go and check" notice, typically about 13–36 s before the exit (development clips). It was weaker on the staff test.
- **Pre-impact fall detector:** Hof's margin-of-stability rule plus a gradient-boosted model. Not reliable on the D435i: in the locked test the rule caught only 1 of 4 falls.

---

## 8. Step 5: what the nurse sees
- **Dashboard:** live stick figures, bed outlines, labels, and the alert list.
- **Ward page (`/ward`):** one tile per bed, coloured by urgency (urgent / check) and bed mode (low / medium / high).
- **Alert clip:** a stick-figure replay of the seconds around the alert, never video.
- **Speed:** an alert reaches the screen about 0.3–1.3 s after the event; camera to decision takes about 240 ms with depth on.

---

## 9. How everything is measured (definitions)

### 9.1 Two kinds of evaluation
- **Frame by frame (recognition):** for each video frame, is the label the model shows the same as the human label? This is used for posture and the 7 phases.
- **Events (alerts):** for each real fall or exit, did an alert come in time? And for each alert, was there a real event? This is used for falls, exits and warnings.

### 9.2 Frame-level measures
Take one class, say "on the ground". Every frame falls in one of four boxes:

| | Model says "on the ground" | Model says something else |
|---|---|---|
| **Really on the ground** | **TP**, true positive | **FN**, false negative (missed) |
| **Really something else** | **FP**, false positive (false alarm) | **TN**, true negative |

| Measure | Formula | Plain meaning |
|---|---|---|
| **Accuracy** | correct frames ÷ all frames | How often it is right overall |
| **Precision** | TP ÷ (TP + FP) | When it says "on the ground", how often is that true? |
| **Recall** (= sensitivity) | TP ÷ (TP + FN) | Of all real on-the-ground frames, how many did it find? |
| **F1** | 2 × precision × recall ÷ (precision + recall) | One number balancing the two. It is high only if both are high. |
| **Macro-F1** | the **average of F1 over all classes**, each class counting equally | Fair when classes are unbalanced (see below) |
| **Balanced accuracy** | average of recall over all classes | Same idea as macro-F1, using recall only |
| **Cohen's kappa** | (accuracy − chance accuracy) ÷ (1 − chance accuracy) | Agreement beyond luck: 0 = no better than guessing, 1 = perfect |
| **Confusion matrix** | table of true class vs predicted class | Shows *which* classes get mixed up |

**Why macro-F1 matters here.**
- Most frames are long easy phases: lying, or out of bed.
- A model that always answered "lying" could get high *accuracy* while being useless at the short, important phases like ATTEMPTING_STAND.
- Macro-F1 gives every phase equal weight, so it exposes that. That is why we quote both.

**Worked example.** Suppose there are 3 classes with F1 values 0.95 (lying), 0.90 (out of bed) and 0.10 (stand attempt).
- Macro-F1 = (0.95 + 0.90 + 0.10) ÷ 3 = **0.65**.
- If 95% of frames are lying or out of bed, accuracy is about 0.93. That looks great but hides the 0.10.

### 9.3 How we keep tests honest
- **Never test on what you trained or tuned on.** Otherwise the score flatters the model, like marking an exam you wrote the answers to.
- **Leave one person out (LOPO):**
  - With 5 training people: train on 4, test on the 5th; repeat for each person; average.
  - This measures how well the model works **on a new person**, which is what matters in a hospital.
  - Frames from one person are never split between training and testing. Neighbouring frames are nearly identical, so splitting them would leak answers.
- **Locked test:**
  - Some people (staff 2 and 3, fall student 02) were set aside before any tuning.
  - They were scored **once**, on 2 Nov, with the code frozen and the scoring rules written down beforehand.
  - Later numbers on them are labelled "post-hoc": an honest re-check, not an independent test.
- **Development clips:** the clips we are allowed to tune on (fall student 01, students 1–4).

### 9.4 Event-level measures (alerts)
- **Sensitivity:** real events that got an alert, for example falls caught 5/5.
- **Precision:** alerts that belonged to a real event. Alerts less than 10 s apart count as one.
- **False alarms per hour:** alerts with no real event ÷ hours of footage.
- **Timing:** alert time minus the labelled event time. Negative means **before** the event.

**Windows** (how close an alert must be to count):
- A fall counts if alerted from 2.5 s before to 30 s after the labelled impact.
- A bed exit counts from 5 s before to 15 s after.

**Policy:** an alert on kneeling or bending low near the floor counts as a **correct alert** (an at-risk movement for a fall-risk patient), not a false alarm. A team member reviewed every test alert on video.

**Accuracy of the labels:** labels are in whole seconds, so all timings are good to about ±1 s.

### 9.5 The current headline results
**Falls (current system, floor alerts plus the possible-fall alert with the tipping gate):**

| | D435f | D435i |
|---|---|---|
| Development (5 falls) | 5/5, 4 before landing, typically 0.6 s before | 5/5, 4 before landing, 0.4 s before |
| Locked, post-hoc (4 falls) | 4/4, all 4 before landing | 4/4, 1 before landing |
| False alarms | **0** on development and locked clips | **0** |

**Bed exits:** staff test 4/4 on both cameras, with no false exits. Development clips: 5/8 and 7/8 by camera, or 6/8 and 6/8 by view.

**Recognition on unseen staff:** posture 91%; 7 phases 81% from the side, 69% along the beds.

**Speed:** about 5 FPS live with depth, 17 FPS colour only. About 10 W of power and 1.2 GB of memory.

### 9.6 What these numbers do *not* show (be upfront)
- The falls are staged by **healthy students on crash mats**, not patients.
- There are few people: 1 fall student per split, 2 test staff, 1 pilot volunteer.
- **"0 false alarms"** was measured over only 10–20 min of footage per camera. Statistically, the true rate could still be up to about 9–19 per hour. About 3 hours of normal activity would be needed to show 1 per hour or less.
- **"4/4"** is consistent with a true catch rate as low as about 40% (95% confidence), because 4 is a tiny sample.
- Low light was never tested, and no infrared was recorded.

---

## 10. The ToF sensor and fusion (in progress, 4 Nov recordings)
The second team's **ToF sensor** (time-of-flight) is mounted above the bed, looking straight down.
- It measures distance by timing how long light takes to bounce back, at 240×180 pixels, 10 times a second.
- It sees the mattress surface and whether a body is on it, but **not who** it is. That gives strong privacy.

On 4 Nov both systems recorded the same bed exits and falls on a shared clock (9 takes).

**The comparison being prepared:**
- ToF alone;
- camera alone (D435i, colour + depth);
- ToF + one camera together.

The rules for combining them were fixed before looking at results:
- The ToF decides whether the bed is occupied.
- The camera gives the detail in bed.
- An exit alert comes from whichever system fires first.
- A camera fall alert counts only when the ToF does not see the person in bed.

Results need the hand labels of the 4 Nov takes. Script: `~/.ahfd-work/recsync_20261104/fuse_score.py`.

---

## 11. The journey in short (timeline)
| Date | What happened |
|---|---|
| 8 Sep | Original design: pose → rules; fall alert after 8 s down and still |
| 15 Oct | Pilot recording: one volunteer, two beds, 10 takes, both cameras |
| 22–24 Oct | Pose bake-off: RTMO-s chosen. Depth found geometrically inconsistent on the pilot; the floor fit separated from the colour calibration |
| 25 Oct | First locked test (pilot takes 09–10), used once |
| 27 Oct | 4 students + 1 staff, bed exits. Recorded at only 6–11 FPS because it wrote straight to the external drive; fixed by recording to the internal disk |
| 28–29 Oct | Bed-engine fixes. Fall session (staged falls, 2 fall students) and staff session. The cameras swapped positions |
| 31 Oct | FALL SUSPECTED alerts quickly (a fall-risk patient on the floor is never acceptable); models trained by view |
| 1 Nov | POSSIBLE FALL early alert added: 4/5 falls before landing, but 13–15 false alarms per hour (more with the faulty pilot clips) |
| 2 Nov | Locked test run once: all falls caught, 0 false alarms, 8/8 staff exits |
| 3 Nov | Trunk-tipping gate: false alarms → 0 on development clips. ID handover. Rise-at-edge early warning. Random forest v4 for phases (+12 points from the side). Model-driven alerts tested and rejected |
| 4 Nov | ToF + camera recorded together for integration |
| 6 Nov | Live-rate bug fixed: the possible-fall alert could not fire at 5 FPS (the rotation window held too few frames). Re-run: falls 5/5 at 5 FPS, 0 false alarms. Bake-off re-scored |

---

## 12. Glossary
| Term | Meaning |
|---|---|
| **Alert / alarm** | A notification to the nurse |
| **Algorithm** | A step-by-step method a computer follows |
| **Bake-off** | A side-by-side comparison of options on the same data |
| **Calibration** | Measured facts about a camera's lens, height, tilt and bed positions, used to turn pixels into metres |
| **Causal** | Using only the past, never the future: what a live system can do |
| **Class** | One of the possible answers of a classifier, such as "sitting" |
| **Classifier** | A model whose output is a class |
| **CUDA / GPU** | The graphics processor, and NVIDIA's way of running maths on it fast |
| **CUSUM** | Cumulative sum: adds small consistent changes until they cross a threshold |
| **Decision tree** | A learned flowchart of yes/no questions |
| **Deep learning** | Neural networks with many layers, trained on large datasets |
| **Depth image** | Per-pixel distance from the camera |
| **Development clips** | Recordings we are allowed to tune on |
| **Event** | Something that happens at one moment: a fall, an exit |
| **F1 / macro-F1** | See section 9.2 |
| **Feature** | A number describing the person at a moment, such as torso height |
| **FPS** | Frames per second |
| **Gradient boosting** | Trees built one after another, each correcting the previous ones' errors |
| **Hyperparameter** | A setting chosen by the designer (threshold, number of trees), not learned |
| **ID / track** | The number that follows one person over time |
| **Imputation** | Filling in missing values (we use the median) |
| **IoU** | Intersection over union: the overlap of two boxes, 0 to 1 |
| **Jetson Orin NX** | The small NVIDIA computer that runs everything beside the bed |
| **Keypoint** | One body joint position found by the pose model |
| **Label** | The correct answer, typed in by a person watching the video |
| **Leave one person out** | Train on all people but one, test on that one, repeat for each |
| **Locked test** | Data set aside and used once, at the end, for honest numbers |
| **Logistic regression** | A linear model: a weighted sum of features turned into a probability |
| **Neural network** | A large adjustable function trained from examples |
| **One Euro filter** | A smoothing filter that smooths more when still and less when moving fast |
| **Overfitting** | Memorising the training examples instead of learning the general pattern |
| **Pose estimation** | Finding body joints in an image |
| **Post-hoc** | Checked after the fact, on data already seen, so not independent |
| **Precision / recall** | See section 9.2 |
| **Pretrained** | Trained by someone else on a large dataset; used as is |
| **Random forest** | Many decision trees trained on random samples of rows and features, voting together |
| **RealSense D435** | The Intel colour + depth camera we use |
| **RTMO-s** | The pose network we use (one-stage, small, fast, Apache-2.0) |
| **Rule / threshold** | A hand-written condition with a fixed number, such as "drop faster than 0.9 m/s" |
| **Sensitivity** | Share of real events that got an alert (= recall for events) |
| **State machine** | States connected by conditional transitions; raises alerts on certain transitions |
| **Sweep** | Trying many values of a threshold and scoring each |
| **ToF** | Time-of-flight depth sensor (the other team's overhead sensor) |
| **Tracking** | Linking detections of the same person across frames |
