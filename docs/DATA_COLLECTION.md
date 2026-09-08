# Data collection guide

## First, an important correction: this is tuning data, not training data

The fall detector has **no trained model** — the state machine's numbers are
thresholds you set, not weights learned from data. So classic *overfitting* and
*underfitting* do not apply to the core system. Nothing is being fitted to your
clips.

What your clips are actually for:

1. **Tuning** — choosing the thresholds (fall velocity, confirm time, spread
   bands) by sweeping them against labelled clips.
2. **Measuring** — the honest false-alarm rate and recall for the report.

The analogue of overfitting here is **tuning on the same clips you then report
numbers on** — that flatters the result the same way. The fix is the same as in
ML: split your clips into two sets.

- **Tune set** (~70%): sweep thresholds against these.
- **Held-out test set** (~30%): touch these *only once*, at the end, to report
  final numbers. Never tune against them.

Keep falls and negatives in *both* sets. (If you later add the optional
gradient-boosted re-ranker, then real train/val/test applies to that — but not
to what exists today.)

## What to record — coverage, not volume

Because you are measuring a false-alarm rate, **negatives matter more than
falls, and you need a lot of negative time.** To claim "under one false alarm
per 8-hour shift" with any confidence you need on the order of **20-30
continuous camera-hours of ordinary activity**, not a handful of clips.

### Negatives (record the most of these — they are safe and free)

The ones that look like falls and must NOT trigger:

- Lying in bed: on the back, on the side, face-down, under a blanket
- Sitting on the edge of the bed
- A second person bending over the bed (nurse/visitor) — the hardest one
- Two people at one bed
- Making the bed / reaching across it
- Picking something off the floor, kneeling to tie a shoe
- Sitting down on a chair, and on the floor deliberately
- Walking toward and away from the camera (the 2D-aspect-ratio trap)
- Reaching up to a shelf, stretching

Vary: lighting, clothing, blanket colours, how many people, where in the frame.
Variety in the negatives is what drives the false-alarm rate down honestly.

### Falls (onto the mattress or a crash mat, with a spotter)

- Roll out of bed
- Slide off the bed edge to the floor
- Slow slump down a wall (tests the no-impact path)
- Trip forward / backward / sideways
- Collapse then get straight back up (tests the near-miss cancel)
- A fall partly hidden by the bed or a drawn curtain

Aim for ~40-60 staged falls across those types. More types matter more than
more repetitions of one type.

### Ratio

Roughly 1 fall to 10+ of ordinary activity by count, and far more by *time*.
A ward is almost entirely non-falls, so your data should be too — otherwise the
measured false-alarm rate will look better than reality.

## Two ways to record

### A. External (simplest, recommended) — phone, OBS, any recorder

Record to an `.mp4`, drop it in, extract keypoints:

```bash
ahfd extract file://clips/roll_out_of_bed_01.mp4 data/tracks/roll_out_of_bed_01.jsonl
```

Nothing in the system touches raw capture; you manage the videos yourself.

### B. Built-in recorder — `ahfd record`

For when you want the system to capture directly (e.g. from the D435i later):

```bash
ahfd record data/raw/fall_01.mp4 --source webcam://0 --i-understand-raw-capture
```

- It records **on start** and shows a live window with a red REC banner burned
  into every frame — recording is never invisible.
- It stops when you **press `q`**, or after `--seconds N` if you set it.
- The `--i-understand-raw-capture` flag is required: this is the one tool that
  writes video to disk, so it will not run by accident.
- Then extract and **delete the video**, keeping only the keypoints:

```bash
ahfd extract file://data/raw/fall_01.mp4 data/tracks/fall_01.jsonl
del data\raw\fall_01.mp4
```

Either way you end up with `data/tracks/<clip>.jsonl` — keypoints only, no
imagery — which is what everything downstream uses.

## Then: annotate, tune, evaluate

For each clip, write `data/annotations/<clip>.json` saying what happened (fall
impact time, or empty for a negative — but always the duration). Full schema and
the tune → evaluate steps are in [CLIPS.md](CLIPS.md).

## Suggested naming

`<scenario>_<subject>_<take>.mp4`, e.g. `bededge_slide_A_02.mp4`,
`adl_bendover_B_01.mp4`. Prefix negatives with `adl_` (activities of daily
living) so tune/test splitting and per-scenario breakdowns are easy.
