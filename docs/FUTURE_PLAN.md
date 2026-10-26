# Future plan: from a one-volunteer pilot to a system that generalises

Status, 24 October 2026. Everything below is based on the ten-take, one-volunteer
bed-exit pilot (takes 01–08 used for development, 09–10 still locked) and the
benchmarks run on the Jetson. These are engineering numbers from staged data,
not clinical validation.

## 1. Where the pilot stands

| Area | Result on takes 01–08 | Verdict |
|---|---|---|
| Pose model | RTMO-s: 43 ms/frame in the pipeline, ~14.5 fps at 1080p (~18.6 fps with a 960 px browser stream). Never loses the patient's ID at stand-up | Keep RTMO-s |
| Alternatives | RTMPose-lightweight misses people lying in bed (2–30% found); YOLO26s is 1.6× slower, loses IDs at stand-up, and is AGPL-licensed | Rejected |
| 4-state posture (dashboard rules) | Standing → UPRIGHT 94–97% after the knee-angle fix (was 56–64%) | Good |
| Posture classifier (tree) | macro-F1 ≈ 0.96 on held-out blocks | Good, but one person only |
| Bed exit (completed) | Oct 15: 4 of 4 exits caught on both cameras, 0.1–7 s after the labelled moment, 0 false. Lifan Oct 13: 12 of 12 clips correct after the 25 Oct standing fix (it had 4 false/early alarms before) | Good |
| 7 bed-exit phases | Rule engine about 50% frame accuracy on both cameras (D435i was 19% before its tilt fix). Best learned model: macro-F1 0.59 (D435f) / 0.51 (D435i) | Weak on short phases |
| Early warning | Fires before 2 of 4 exits, on the D435f only | Weak |
| Camera position | For one bed, the D435f position is better (lower body visible 91% vs 76%, posture F1 0.952 vs 0.925). For 3 beds, a side-on view like the D435i's should be better (untested) | See §4 |
| Depth | Fixed 25 Oct with `depth_ground` (§5). On identical tracks, corrected depth gives standing → UPRIGHT **100%** (RGB 80–94%), no false ON_GROUND for standing (RGB 3–7%), bed phases +1–8 points, phase classifier +0.03–0.08. Costs: 6–7% of bed-edge sitting frames shown UPRIGHT; the posture tree is better without depth | Use it, with §6 caveats |

**Main limitation:** everything comes from **one person**, 8 short takes, two beds and
one room. Every number above is a same-volunteer block holdout, so it may not hold for
a new person.

## 1c. Locked test on takes 09–10 (run once, 25 Oct; everything frozen, trained on 01–08)

| | D435f RGB / +corrected depth | D435i RGB / +corrected depth |
|---|---|---|
| Exits (09: labelled 57.0 s; 10: none) | caught +2.8 s, no false / same | caught −0.2 s, no false / same |
| Lying shown IN_BED | 100% / 100% | 100% / 100% |
| Sitting in bed shown IN_BED | 95% / 95% | 95% / 95% |
| Standing shown UPRIGHT | **48% / 100%** | 100% / 100% |
| Standing shown ON_GROUND | 0% / 0% | 0% / 0% |
| Rule engine 7-phase accuracy | 30.8% / 32.1% | 33.5% / 48.4% |
| Phase classifier macro-F1 | **0.619** / 0.512 | **0.703** / 0.524 |
| Posture tree macro-F1 | 0.997 / 0.971 (**1.000 with Lifan's clips**) | 0.856 / **0.988** |

Exits and the main posture states hold up on unseen takes; short bed phases are weaker
than on development; depth fixes standing in the rules but its classifier effect is mixed
(2 takes only, so per-classifier gaps of a few points are noise). **09–10 are now used:
any further tuning needs fresh recordings to test against.**

## 1a. Handoff checklist (T9 `AHFD_JETSON_HANDOFF_20261022/README.txt`), status 25 Oct

| Step | Status |
|---|---|
| Preflight (`jetson_preflight.py`) | ✅ File checks pass: handoff artefacts by SHA-256, all 20 recordings by size and topics (re-run 25 Oct) |
| Runtime: Python 3.10, onnxruntime-gpu with CUDA, RTMO-S weights in the rtmlib cache | ✅ |
| Code from `lifan-latesst` / bundle, at least the handoff revision | ✅ Merged into `shandrico-dev` |
| a. Replay one clip from each camera | ✅ Dashboard and `export-review` play `.db3` recordings with seeking |
| b. Per-camera geometry, alignment, phase/exit timestamps | ✅ Calibrations and bed outlines for both cameras; D435f labels put on the D435i clock (~1.1 s offset); depth geometry fixed with `depth_ground` (§5) |
| c. Stream recordings into compact features | ✅ RGB tracks (Codex), RGB+depth tracks with the calibration geometry (25 Oct, takes 01–08 both cameras) |
| d. Train/tune on takes 01–08 only, both views and paired takes grouped | ✅ Posture, phase and exit work all use leave-one-block-out (A–D); 09–10 never read |
| e. Freeze, then evaluate 09–10 once | ⏳ **Your decision.** Everything is ready; it only waits for "frozen" |
| f. Live shadow mode | ⏳ Needs the cameras on USB-A ports |
| Raw-file SHA-256 hashes ("NOT computed" in the handoff) | ✅ All 20 main recordings hashed on the Jetson, 25 Oct: `AHFD_JETSON_HANDOFF_20261022/raw_sha256_jetson_20261025.txt`. This is a baseline from now on; it can't prove they match the Windows originals, which were never hashed |
| Keep the Oct 13 dataset separate | ✅ Used only as an extra, separately reported experiment (§1b) |

## 1b. Lifan's Oct 13 D435f pilot (same mount, 12 reviewed clips)

- **Posture:** adding it to training lifts the Oct 15 held-out macro-F1 from 0.952 to
  **0.984**. A model trained on Oct 15 scores 0.971 on Lifan's session; the reverse is
  only 0.781 (his clips have less variety in edge sitting and standing).
- **Bed exit:** it exposed **4 bad exit alarms** that the Oct 15 takes never showed
  (two walk-pasts of an empty bed, an edge approach and return, and one exit flagged
  10 s early while still sitting). The cause: losing "body over the mattress" was
  treated as an exit even when seated, and a passer-by standing in front of the bed
  overlapped its outline in the image. **Fixed:** a completed exit now requires
  standing (upright torso, near-vertical legs, knees not clearly bent; leg direction
  is a new image feature), standing bodies don't count as bed-supported, and support
  must last 2 s before a track counts as having been in bed. Result: Lifan 12/12
  correct; Oct 15 still 8/8 exits caught with unchanged timings and no false alarms.

## 1d. Code review fixes (25 Oct, all with regression tests)
- **Fall logic:** a fall whose trigger fired while the torso was still above 0.70 m
  was being discarded on the next frame (no alert for 20 s in simulation). The
  "never went low" check now waits for the 0.8 s drop window.
- **Bed exit:** exits that don't end standing (into a bedside chair, a slide to the
  floor) now count once the hips are more than 0.3 m beyond the bed edge; the leg
  angle is measured against the room's true vertical at that point in the image
  (verticals lean ~15° at the frame edges on these mounts); the 2 s "in bed" timer
  restarts after a low-confidence gap.
- **Tracker:** the proximity re-match only applies to a box of similar size and shape,
  so a nurse beside a lying patient can't inherit the patient's ID.
- **Cameras / extract:** `model=` is checked against an explicit serial;
  `extract --depth rs://` honours all URI options; a bad calibration no longer leaves
  the camera running; `--calibration` without `--depth` warns. New
  `ahfd reground-tracks` converts older depth tracks to `depth_ground`.
- After the fixes: Oct 15 exits 8/8 with unchanged timings; Lifan 12/12; 1,121 tests pass.

## 2. Data that would help most

### 2.1 Short transition phases (most urgent)
Labelled time today (takes 01–08): RECLINED 248 s, OUT_OF_BED 75 s,
UPRIGHT_IN_BED 60 s, EDGE_SITTING 26 s, SHIFTING_TO_EDGE 25 s,
**TORSO_RISING 9 s, ATTEMPTING_STAND 7 s.**

The rule engine never catches TORSO_RISING (0 of 5), and no learned model got
above F1 ≈ 0.4 on the short phases. That's a data problem, not a modelling problem.

Target: **at least 5 minutes of labelled time per phase**. For a 2-second sit-up,
that's about 150 sit-ups and about 150 stand attempts. Spread over the cohort below,
that's about 10–15 of each per person per session. Recording them is cheap: they
take seconds each.

### 2.2 How many people, and who
Aim for **12 or more volunteers** (minimum 8) so that leave-one-person-out results
mean something:

- **Height:** spread from about **150 cm to 190 cm** (at least 3 people under 160 cm
  and 3 over 180 cm). Joint heights, bed overlap and torso height all scale with
  stature, and the knee and torso angles depend on limb proportions.
- **Build:** slim to heavy. Box and joint confidence change with body shape and
  loose clothing.
- **Sex and age:** both sexes, and older adults if any can safely take part. Older
  people sit up and stand more slowly and with more pauses, which changes exactly
  the phases we can't detect yet.
- **Mobility style:** normal, slow and careful, using hands on the rail, rolling to
  the side before sitting up, and aborted attempts.
- **Two sessions per person** on different days, so day-to-day changes (clothing,
  lighting, bed position) are also tested.

### 2.3 Conditions to cover
- Both beds and **all bed positions** to be used in deployment, plus a **third bed**
  for the one-camera-for-3-beds setup.
- **Blankets** (thin and thick), pillows, and the bed head raised or flat.
- **Lighting:** day, evening, and night or low light (and IR mode if it will be used).
- **Clothing:** hospital gown, loose clothes, dark and light colours.
- **Other people:** staff walking past, standing at the bedside and leaning over the
  bed, plus a visitor on a chair. This needs a separately approved multi-person
  protocol; the current collector stops when an extra person appears.
- **Roll-overs and long still periods.** Take clip1_01 showed the patient vanishing
  from the pose model for about 7 s during a roll, and the tracker can't bridge that.
- **Staged falls only under a separate approved protocol** (trained consenting
  volunteers, mats, supervision). Without them we can measure fall false alarms but
  not fall recall.

### 2.4 Labelling
- Keep the 7-phase scheme and the point markers (`e`, `b`, `w`, ...) in
  `docs/TRAINING_SCHEME.md`.
- **Label each camera on its own timeline.** Today the D435f labels are D435i labels
  shifted by about 1.1 s, which blurs 1–2 s phases.
- Have a second person independently label about 10% of takes, and report agreement,
  especially phase boundaries.
- Budget: about 5–10 minutes of labelling per minute of video. The `ahfd label-phases`
  seed drafts help.

## 3. How to measure generalisation honestly
1. **Leave-one-person-out** cross-validation on the development people: train on all
   but one, test on the one left out, and rotate. Both cameras and all sessions of a
   person stay in the same fold.
2. **A locked test group of 2–3 people** who are never used for any tuning, opened
   once at the end, like takes 09–10 today.
3. Report per-person results (worst person, not just the average), per height range,
   and per condition (blanket, low light, ...).
4. Keep reporting **exit events** (caught/missed, latency, false exits per hour) next
   to frame accuracy. Events are what nurses experience.
5. Once the current rules and models are frozen, run the pilot's own locked test on
   **takes 09–10**, before collecting new data, to get the honest pilot number.

## 4. Camera setup
- **One camera per bed:** use the **D435f position** (side-on, near), the best pilot
  result.
- **One camera for 3 beds:** a side-on, slightly higher and central mount, so all
  three beds are at a similar distance (~4.5 m) and none hides behind another. The
  D435's ~69° colour view covers ~6.2 m at 4.5 m; 3 beds need ~5.5 m, which fits but
  only just. **Record a 3-bed test first** to check coverage, occlusion and ID
  switches.
- Per-camera bed-angle settings now exist (`bed_tilt_deg` in the calibration file):
  a camera looking along the beds needs lower reclined/upright angles. Re-derive them
  for each new mount from a few labelled takes.

## 5. Depth: fixed in software, verify at every session
**Root cause (found 25 Oct):** the depth stream and the colour calibration disagree
about the camera's tilt by about **6° (D435f) and 3–4° (D435i)**. Depth heights are
measured points, so they need the depth stream's own pose. Computed with the colour
calibration's 30°, they came out 0.3–0.5 m too low, worse the farther the bed: standing
ankles at −0.2 m, a lying torso *below* the mattress. Fitting the floor in depth
(`ahfd estimate-ground`) gives a consistent pose on every take (D435f 2.58 m / 24.1° /
roll 1.2°, D435i 2.69 m / 26.4° / roll 0.8°). With it, depth heights are physical:
standing ankles 0.10–0.14 m, hips 0.86–0.96 m, shoulders 1.30–1.41 m, lying shoulders
+0.20–0.27 m above the mattress.

The colour calibration is kept as it is: its bed outlines and camera-only heights are
self-consistent with the image, and replacing it with the depth pose makes the beds
3.8 m long. The fix is a separate **`depth_ground`** block in each calibration file,
used only to turn depth into joint heights (live dashboard, collection, and
`extract --depth --calibration`). Existing depth tracks can be converted exactly
with `geometry.depth_height.reground_heights`, without re-reading the recordings.

At every session:
1. Run `ahfd estimate-ground` on the first recording and update `depth_ground` if it
   moved by more than ~1°.
2. Record **10 s of one standing person of known height** and check that ankles read
   about 0.1 m and the head reads their real height.
3. Write down the mount **height (tape)** and **tilt (phone inclinometer)**.

**Mounting rule:** don't point a camera straight down the length of a bed from its
foot. A body lying along the line of sight looks upright with straight, vertical legs
in the image, which the image-angle rules can't tell from standing.

### 5b. Dense-depth bed occupancy (prototype, 25 Oct, dev takes 01–08)
Computed live from the depth image inside each bed outline; only a few numbers per bed
per frame are kept (privacy-safe).
- **Whole-area height above the mattress:** doesn't work. An *empty* bed already has
  rails, pillows and blankets as tall as a lying body (top 10% at 0.18–0.30 m).
- **Per-pixel change vs an empty-bed reference:** separates occupied from empty on the
  **D435i** (median share of bed pixels raised >5 cm: lying 0.26, sitting 0.33, empty
  0.08) but barely on the D435f (0.09 vs 0.04). On the D435f it still sees the patient
  in **99%** of the frames where the pose model lost them (D435i 30%). Empty-bed frames
  are noisy (up to 0.20–0.42) because bedding moved between takes and the reference
  came from another take.
- **Classifiers:** no gain from adding either version (posture and phase F1 unchanged
  or slightly lower).
- **To make it usable:** take the empty-bed reference at the start of each session and
  re-take it whenever the bed is confirmed empty; ignore slow changes (bedding being
  moved); use it as a separate "bed occupied" signal to bridge pose dropouts, not as a
  classifier feature.

### 5c. Depth in the final report
Include it: (1) the depth/colour tilt mismatch and the `depth_ground` fix as a
methodological result; (2) depth as an improvement to the **rule-based** system (locked
test: D435f standing 48% → 100%, D435i bed phases 33.5% → 48.4%, no false ON_GROUND,
exits unchanged); (3) learned classifiers reported **RGB-only**, stating that depth gave
no consistent gain with one volunteer; (4) bed occupancy as a preliminary result.

## 5a. Fall detection status
- **Data with falls:** only Shandrico's 6 September D435f clips (10 staged falls,
  older mount, unverified 2.5 m / ~15° calibration). The volunteer stays down for
  **3–7 s** before getting up, below the 8 s stillness needed to page, so none can be
  confirmed by design. Fall **recall cannot be measured** from them. On the floor, the
  posture rules say ON_GROUND only 31–59% of the time, depending on the assumed tilt.
- **Bugs fixed 25 Oct:** (1) a fall attempt's height history wasn't cleared on reset,
  so one dip re-fired every frame (up to 55 NEAR_MISS events in a clip with no fall);
  (2) a NEAR_MISS also started the 60 s paging cooldown, which would have **hidden a
  real fall within a minute of a stumble**; (3) a trigger whose torso never went low
  is now discarded instead of reported as a near-miss. Near-misses in no-fall clips:
  34 → 1 and 55 → 0.
- **Needed:** staged falls (approved protocol, mats, trained volunteers) that stay on
  the floor **≥ 10–15 s**, on a current, verified mount, with a few slower "slumps"
  and a few recoveries, plus fall-like movements that aren't falls (picking things
  up, kneeling). Label onset, floor contact and recovery.
- **Known limitation:** someone standing between the camera and a bed overlaps the
  bed in the image and can read as "on the bed", which blocks the fall trigger. An
  image-only fix (tried tonight) made a patient lying along the line of sight look
  standing, causing false PERSON_DOWN alarms, so it was reverted. It needs a 3D cue;
  the corrected depth heights are the candidate (§5).

## 6. Models: what to train next
0. **RGB vs RGB + corrected depth, same tracks (takes 01–08):**

   | | D435f RGB / +depth | D435i RGB / +depth |
   |---|---|---|
   | Dashboard: standing shown UPRIGHT | 80% / **100%** | 94% / **100%** |
   | Dashboard: standing shown ON_GROUND | 7% / **0%** | 3% / **0%** |
   | Dashboard: bed-edge sitting shown UPRIGHT/ON_GROUND | **1%** / 6% | **0%** / 7% |
   | Bed engine 7-phase accuracy | 47.8% / **49.2%** | 43.2% / **50.8%** |
   | Exits | 4/4, 0 false / same | 4/4, 0 false / same |
   | Phase classifier macro-F1 | 0.547 / **0.579** | 0.500 / **0.579** |
   | Posture tree macro-F1 | **0.969** / 0.945 | **0.939** / 0.873 |

   So: **use corrected depth for the live rules and the phase model; train the posture
   classifier on RGB features only.** The remaining depth error is someone sitting on
   the bed edge with their feet down: their torso (~1.1–1.3 m) is above the seated
   height band (0.70–1.00 m). Widening the band or trusting bent knees made walking
   people flicker to SITTING, so it's left as is.
1. **Now:** keep the rules for posture and exits. As a shadow comparison, run the
   balanced **logistic regression on single-frame + over-time features** (best phase
   model: macro-F1 0.59 on the D435f). Re-run `ahfd compare-phases` after each new
   batch of labels.
2. **After about 1 hour of labelled transitions from 8+ people:** retune the bed engine
   thresholds per mount, and retrain the posture and phase models with
   leave-one-person-out validation.
3. **After several hours from 12+ people:** a small temporal network (TCN or GRU over
   about 2–4 s of keypoints) is worth trying for TORSO_RISING and early warning. Not
   before: with today's 7–9 s per short phase it would just memorise.
4. **Tracker:** the simple tracker now retires identities by time (1 s) and re-attaches
   nearby detections. A proper tracker (for example ByteTrack) and "same person
   reappearing in the same bed" logic are worth doing once multi-person recordings
   exist to test them safely.

## 7. Requirements
- **Hardware:** Jetson Orin NX 16 GB (MAXN, adaptive clocks; don't lock clocks, as that
  caused over-current throttling), D435f per bed or one D435i/D435f for a 3-bed view,
  **USB 3 Type-A ports** (the developer kit's USB-C port doesn't run the camera), a
  fast SSD for recordings (each 1-minute dual-camera take is ~7–13 GB per camera).
- **Software:** RTMO-s on onnxruntime-gpu (CUDA). The dashboard's browser stream is
  set to 960 px for remote viewing over the slow relay. Select cameras with
  `rs://...&model=d435i|d435f` when both are plugged in.
- **Process and approvals:** consent forms covering RGB recording; AH/DPO sign-off for
  any RGB display or saved review video; a separate protocol for multi-person and
  staged-fall sessions; one approved storage location for raw recordings. Review
  videos and the raw-capture config stay on the T9 and must never go into the
  repository.
- **People time:** per session, about 1 hour of recording per volunteer plus about
  5–10× the recorded minutes in labelling time.

## 8. Next steps, in order
1. Run the locked test on takes 09–10 with the current frozen rules (pilot number).
2. Run a 3-bed test recording to settle the camera position.
3. Session-start depth/geometry checks (§5), then re-record.
4. Collect the cohort in §2 (many short sit-ups and stand attempts), labelled per
   camera.
5. Leave-one-person-out evaluation; then decide on temporal models.
