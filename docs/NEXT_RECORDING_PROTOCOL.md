# Recording guide (compact): students and hospital staff

Written 26 Oct 2026. Sized for **one week** of labelling and training and the **T9's
1.4 TB of free space**. Same timed format as the pilot's 10-take choreography
(`src/ahfd/annotate/phase_script.py`): every clip has the cue to say and the phase we
expect, so labels can be pre-filled.

## 0. Budget

| | Size |
|---|---|
| D435f recording | ≈ 6.9 GB per minute |
| D435i recording | ≈ 12.9 GB per minute |
| Both cameras | ≈ 20 GB per minute |
| Free on the T9 (26 Oct) | 1.4 TB ≈ 70 min of both-camera recording |

| Group | Clips | Minutes | Storage |
|---|---|---|---|
| 5 students, bed clips (both cameras) | S1–S4 | 5.5 each = 27.5 | ≈ 550 GB |
| 5 students, falls (**D435f only**) | F1–F4 | 4 each = 20 | ≈ 140 GB |
| 4 staff, bed clips (both cameras) | H1–H3 | 4 each = 16 | ≈ 320 GB |
| Setup checks (both cameras, 20 s per session) | — | ~3 | ≈ 60 GB |
| **Total** | | **≈ 66 min** | **≈ 1.07 TB** (leaves ~0.3 TB spare) |

**Before recording, check free space** (`df -h /media/anormalm/T9`). Stop if it falls
below 100 GB. Don't record to the Jetson's own disk.

**What this buys** (vs the pilot's 5 sit-ups and 4 stand attempts):

| | Students | Staff | Total |
|---|---|---|---|
| Sitting up (TORSO_RISING) | ~40 (8 each) | ~24 (6 each) | **~64** |
| Stand attempts (ATTEMPTING_STAND) | ~25 (5 each) | ~12 (3 each) | **~37** |
| Completed exits | ~10 (2 each) | ~8 (2 each) | **~18** |
| Staged falls / near-miss / look-alike | 15 / 5 / 5 | — | |

## 1. Groups and split
- **Students (5):** 4 for development, **1 locked** (chosen before recording) to test falls.
- **Hospital staff (~4):** **all locked**: the honest test on new people. Nobody looks at
  their clips until rules and models are frozen.
- **Beds:** students 1, 3, 5 use the **nearest** bed; students 2, 4 the **middle** bed;
  staff alternate. Each person uses one bed only.

**File names:** `s02_S1_nearest.db3` (student 02, clip S1, nearest bed),
`h01_H2_middle.db3`, `s02_F1.db3`. Both camera folders use the same name. Keep a sheet
with each ID's height and build only, no names.

**Cueing:** read the cue at the start time; the volunteer moves naturally. Times are
targets; the labels are corrected afterwards.

## 2. Setup (every session, 5 min)
1. Note camera height (tape) and tilt (phone inclinometer).
2. Record **10 s empty room**, then **10 s of one person of known height standing** in
   front of the bed; run `ahfd estimate-ground` and update `depth_ground` if the tilt
   moved more than ~1°.
3. Check the bed outlines still match (`ahfd calibrate-zones` only if something moved).

## 3. Students: bed clips (both cameras, 5.5 min)

### S1 — Sit-ups (90 s)
Five cycles of 16 s after a 10 s settle: **0 s** "sit up" → hold to **+8 s** → **+8 s**
"lie back" → still to **+16 s**.

| Time (s) | Cue | Variation | Expected |
|---|---|---|---|
| 0–10 | "lie still" | | RECLINED |
| 10–26 | rep 1 | slow (3–4 s to rise) | RECLINED → TORSO_RISING → UPRIGHT_IN_BED → RECLINED |
| 26–42 | rep 2 | half sit-up: rise halfway, lie back | RECLINED → TORSO_RISING → RECLINED |
| 42–58 | rep 3 | roll to your side first, push up | full cycle |
| 58–74 | rep 4 | brisk (~1 s) | full cycle |
| 74–90 | rep 5 | normal, use the rail | full cycle, end lying |

### S2 — Stand attempts and exit (120 s)
| Time (s) | Cue | Expected |
|---|---|---|
| 0–8 | "lie still" | RECLINED |
| 8–12 | "sit up" | TORSO_RISING |
| 12–18 | "stay sitting" | UPRIGHT_IN_BED |
| 18–22 | "move to the edge" | SHIFTING_TO_EDGE |
| 22–28 | "feet on the floor, sit" | EDGE_SITTING |
| 28–40 | attempt 1: "start to stand" (28–31), "sit back down" (31–34), rest | ATTEMPTING_STAND → EDGE_SITTING (`b`) |
| 40–52 | attempt 2: slower, hands on the mattress | same (`b`) |
| 52–64 | attempt 3: rise almost fully, then sit | same (`b`) |
| 64–67 | "stand up fully" | ATTEMPTING_STAND |
| 67–85 | "walk 2–3 m away, stop" | OUT_OF_BED (`e`) |
| 85–98 | "come back, sit on the edge" | EDGE_SITTING |
| 98–105 | "legs into bed" | SHIFTING_TO_EDGE |
| 105–120 | "lie down" | RECLINED |

### S3 — Everyday, not an exit (60 s): no standing
| Time (s) | Cue | Expected |
|---|---|---|
| 0–10 | "lie, use your phone" | RECLINED |
| 10–18 | "roll onto your side, then back" | RECLINED |
| 18–24 | "reach to the bedside table" | RECLINED |
| 24–28 | "sit up" | TORSO_RISING |
| 28–38 | "stay sitting, drink / look around" | UPRIGHT_IN_BED |
| 38–42 | "swing your legs over the edge" | SHIFTING_TO_EDGE |
| 42–50 | "sit at the edge — **don't stand**" | EDGE_SITTING |
| 50–54 | "legs back in" | SHIFTING_TO_EDGE |
| 54–60 | "lie back" | RECLINED |

### S4 — Slow exit, like an older patient (60 s)
| Time (s) | Cue | Expected |
|---|---|---|
| 0–6 | "lie still" | RECLINED |
| 6–11 | "sit up slowly" | TORSO_RISING |
| 11–18 | "pause sitting" | UPRIGHT_IN_BED |
| 18–24 | "move to the edge slowly" | SHIFTING_TO_EDGE |
| 24–32 | "sit, hold the rail, pause" | EDGE_SITTING |
| 32–37 | "stand up slowly" | ATTEMPTING_STAND |
| 37–50 | "walk slowly to the chair, stop" | OUT_OF_BED (`e`) |
| 50–60 | "come back and sit on the bed" | EDGE_SITTING |

## 4. Students: falls (approval required; D435f only; ~10 min with resets)
Consenting students only, **crash mats** where the falls land, a spotter, a supervisor
present. Anyone may stop at any time. Never ask patients or staff to fall. Record on the
**D435f only** to halve storage.

| Clip | Length | 0–12 s | Fall | Down | End |
|---|---|---|---|---|---|
| F1 Fall during a stand attempt at the bed edge | 60 s | sit at the edge | 12–16: start to stand, lose balance, land on the mat | **16–40 lie still** | 40–60 get up, sit on the bed |
| F2 Slide / roll out of bed | 60 s | lie near the edge | 12–20: slide slowly off onto the mat | **20–42 lie still** | 42–60 get up |
| F3 Trip while walking near the bed | 60 s | stand (0–6), walk (6–12) | 12–14: trip forward onto the mat | **14–38 lie still** | 38–60 get up |
| F4 Near miss (as F1, quick recovery) | 30 s | sit at the edge | 12–16: fall onto the mat | 16–20 down | **20–30 get up quickly**, sit |
| F5 Look-alike: lie down on the floor on purpose | 30 s | stand | 8–14: lie down slowly on the mat (`o`) | 14–24 lie still | 24–30 get up |

That's 3 falls + 1 near-miss + 1 look-alike per student (4 min of recording). Labels:
`j` fall onset, `k` floor contact, `l` recovery, `o` intentional floor posture, and
`on_ground` posture while down. Lying still for **20+ s** matters: the system confirms
a fall after 8 s of stillness, and the September falls recovered too fast to test it.

## 5. Hospital staff: gentle bed clips (both cameras, 4 min, no falls)
| Clip | Length | Script | Expected |
|---|---|---|---|
| H1 Sit-ups | 70 s | 0–10 lie still · then 4 cycles of 15 s: "sit up" (normal pace), stay ~7 s, "lie back" | 4 sit-ups |
| H2 Edge, one stand attempt, exit | 90 s | 0–8 lie · 8–12 sit up · 12–18 sit · 18–22 to the edge · 22–30 sit at the edge · 30–34 start to stand, sit back down (`b`) · 34–42 sit · 42–46 stand · 46–62 walk 2–3 m away, stop (`e`) · 62–75 come back, sit · 75–90 legs in, lie down | 1 attempt + 1 exit |
| H3 Everyday, then an exit at their own pace | 80 s | 0–15 lie, phone · 15–20 sit up · 20–35 sit, drink · 35–45 legs over the edge, no standing · 45–50 legs back · then "whenever you're ready, get up and walk to the chair as you normally would" | 1 exit, natural |

Optional, with multi-person approval: a staff member plays a **nurse** (walks past,
stands at the bedside, leans over the bed) while a student lies in bed: 60 s, both
cameras, the best test of passer-by false alarms. Record with `record-dual` (`collect`
stops when a second person appears).

## 6. Labelling in one week
Total ≈ 66 min of recording. To keep labelling to about 6–10 hours:
1. **Label the D435f fully.** For the D435i, copy the labels shifted to the D435i's
   clock (the pipeline already aligns the two cameras by their recording timestamps), then
   spot-check the short phases.
2. **Order:** S1 and S2 first (most of the missing sit-ups and stand attempts), then S4, the
   falls, S3, staff last (they're the locked test, labelled but not looked at in tuning).
3. Leave unclear moments unlabelled rather than guessing.

## 7. Session timing
- **Student:** setup 5 min · S1–S4 about 10 min with resets · falls about 10 min (can be a
  different day). About 25 minutes per student.
- **Staff:** setup (shared) · H1–H3 about 8 min with resets.
- After the first 2 students are labelled, run `ahfd compare-phases` and the exit and
  posture checks, to confirm the new clips actually help before recording the rest.

## 8. If more space or time turns up
In order of value: S1 on the other bed (more sit-ups from a second view), F1–F3 at a
second mat position (beside the bed, partly hidden), more staff, the nurse clip.
