# Fall detection: recording, labelling and training plan

Written 28 Oct 2026. Builds on `docs/NEXT_RECORDING_PROTOCOL.md` section 4 and the bed-clip
pipeline that worked on 27-28 Oct (record -> extract -> label -> train -> held-out test).

## 0. Why falls need their own session

- The only fall data so far is the 6 September D435f clips: 10 staged falls, an old mount, and the
  volunteer back up after 3-7 s. The detector only pages after **8 s down and still**, so none of
  them can be confirmed. Fall recall has never been measured.
- The posture model has no `on_ground` class yet: the v2 tree knows in_bed / sitting / upright only.
- Every bed clip (pilot, S1-S4, H1-H3) is a **negative**: no fall. Together they give the
  false-alarm denominator (about 45 min so far).

## 1. Before the session (must do)

| Step | Who | Why |
|---|---|---|
| Fix the fall state machine so one "not down" frame no longer cancels a fall | Claude (code) | On the Sept clips the longest unbroken "down" run was 0-5 s, so an 8 s confirmation was impossible even for a long fall (see the 27 Oct review). Without this fix, recall is ~0 whatever is recorded. |
| Make "on the floor" robust to one stray joint (`floor_spread`) | Claude (code) | The main reason frames fail the down test. |
| Recording test: 30 s at `--fps 15` to the **internal disk** and check the printed saved fps | User + Claude | 27 Oct was 6-11 fps because it wrote to the T9; falls last under 1 s, so frame rate matters more than for bed exits. |
| Approval, consent, crash mats, a spotter and a supervisor present | User / supervisor | Staged falls only by consenting students. Never patients or staff. |
| Don't move the cameras after the setup check; if they move, run `estimate-ground` and redraw the bed boxes | User | Moved cameras broke the D435i bed boxes on 27 Oct. |

Recording command (internal disk, then copy to the T9 afterwards):
```
AHFD_ALLOW_RAW=1 ahfd record-dual ~/recordings/d435i/f1_01.db3 ~/recordings/d435f/f1_01.db3 \
  --serial-a 346122072607 --serial-b 406122070025 --fps 15 \
  --config configs/raw_capture.yaml --i-understand-raw-capture
```
At 15 fps both cameras use about 10 GB per minute. The internal disk has about 260 GB free, so
copy to the T9 after every student (`cp ~/recordings/d435f/*.db3 /media/anormalm/T9/ah_bags_d435f/`, same for d435i).

## 2. Recording script (per student, about 6 min recorded, about 20 min with resets)

The mat is where the fall lands. Lying still for **20+ s** matters: confirmation needs 8 s of
stillness and the slow-slump path needs 20 s. Both cameras if there is space, otherwise D435f only.

| Clip | Length | 0-12 s | Event | Down | End |
|---|---|---|---|---|---|
| `f1` Fall during a stand attempt at the bed edge | 60 s | sit at the edge | 12-16: start to stand, lose balance, land on the mat | 16-40 lie still | 40-60 get up, sit on the bed |
| `f2` Trip while walking near the bed | 60 s | stand, walk | 12-14: trip forward onto the mat | 14-38 lie still | 38-60 get up |
| `f3` Slow slump (no impact) from sitting on the edge | 60 s | sit at the edge | 12-20: slide down slowly to the floor | 20-50 lie still (30 s, for PERSON_DOWN) | 50-60 get up |
| `f4` Near miss | 30 s | sit at the edge | 12-16: fall onto the mat | 16-20 down | 20-30 get up quickly (must NOT page) |
| `neg_f5` Look-alike: lie down on the floor on purpose | 30 s | stand | 8-14: lie down slowly on the mat | 14-24 lie still | 24-30 get up |
| `neg_f6` Look-alike: pick something up / kneel / tie a shoe | 40 s | stand | 5-35: pick up an object from the floor twice, kneel 5 s, crouch 5 s | - | stand |

**Out of scope: rolling or sliding out of a lying position.** We assume the bed rails are up and prevent
it; the bed-exit side (sit up -> edge -> stand) covers the route a patient actually takes out of bed.
State this assumption in the report. (`NEXT_RECORDING_PROTOCOL.md` still lists an F2 slide-out clip;
skip it.)

File names: `f1_01.db3` = clip f1, student 01 (same name in both camera folders). **Look-alikes must
start with `neg_`**: `derive-falls` then treats them as no-fall clips. Otherwise an intentional lie-down
counts as a fall.

**How many:** at least **5 students x 2 impact falls (f1-f2) = 10 falls** plus 5 slumps (f3), 5 near
misses (f4) and 10 look-alikes. 8+ students would give 16+ impact falls plus 8 slumps (24+ down events), which is when a learned fall model becomes worth
trying. Choose **one student as the locked test before recording**; nobody looks at their clips
until everything is frozen.

## 3. Extraction (Claude)

Same as 27 Oct: `~/.ahfd-work/extract_session_*.sh`-style batch, `ahfd extract --depth
--calibration calib/<cam>_<session>.yaml`, derived tracks to the T9 and copied to `data/tracks_*_falls/`.
Check that every track file has one row per colour frame.

## 4. Labelling (user): 4-state posture, not 7-phase

```
AHFD_ALLOW_RAW=1 ahfd label-postures /media/anormalm/T9/ah_bags_d435f/f1_01.db3 \
  --out data/postures_falls/f1_01.json --config configs/raw_capture.yaml --i-understand-raw-capture
```
Keys 1 upright, 2 sitting, 3 in_bed, **4 on_ground**. Label the hold before the fall (upright or
sitting), then the **on_ground** hold, and leave the falling movement itself as a gap.
`ahfd derive-falls` then reads the fall start (the end of the posture before it) and the impact (the
start of on_ground) from the labels, so no separate impact marking is needed. Label one camera; Claude
shifts the labels to the other camera's clock.

## 5. Training and tuning (Claude)

1. **Posture model with `on_ground`:** retrain the tree on bed clips + fall clips (4 classes).
   Test: the locked student, plus leave-one-person-out over the others.
2. **Fall state machine thresholds**, using the dev students only:
   ```
   ahfd derive-falls --postures data/postures_falls --out data/annotations_falls
   ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 --tracks data/tracks_d435f_falls/ \
       --annotations data/annotations_falls/ --calibration calib/d435f_<session>.yaml
   ```
   Also sweep `drop_trigger`, `confirm_s` (8 s is a clinical choice; report the curve) and
   `down_spread`. Choose the operating point from recall vs false alarms per hour, where the
   false alarms come from **all** the no-fall clips (pilot, S1-S4, H1-H3, look-alikes).
3. **Optional, with 24+ falls:** a small learned "is this person down" confirmation, using the same
   movement-history features as the 7-phase model, as a check beside the rules (never replacing
   them for paging at first).

## 6. What to report

| Metric | Target to aim for | On |
|---|---|---|
| Fall recall (FALL_CONFIRMED or PERSON_DOWN within 30 s of impact) | as close to 100% as possible | locked student |
| Time from impact to alert | about 8-10 s (the confirmation wait) | locked student |
| False alarms per hour | as low as possible; report the number | all no-fall clips + look-alikes |
| Near misses (f4) paged | 0 | all students |
| Slow slumps (f3) caught by PERSON_DOWN | report | all students |

Say "staged falls by N healthy student volunteers on crash mats", never patient falls.
