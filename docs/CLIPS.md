# Working with clips: record → extract → annotate → tune → evaluate

This is the workflow for turning staged footage into real accuracy numbers. It
needs no depth camera — a webcam or phone video is enough to start.

The shape of it:

```
 record  ──▶  extract  ──▶  tracks.jsonl  ──▶  replay / sweep / eval
(imagery)     (once)       (keypoints only)     (fast, deterministic, no imagery)
```

Pose is the slow step, so it runs **once** per clip into `tracks.jsonl`. Every
later step works on those keypoints — fast, repeatable, and with no pixels on
disk, so the keypoint files are safe to keep and commit.

## 1. Record

Any of these produce a clip the pipeline can read:

- **Phone / OBS / screen recorder** → an `.mp4`. Simplest, and nothing in this
  project has to touch raw capture. Recommended.
- **A directory of image frames** (this is how public datasets ship) → read
  with `seq://path/to/frames`.
- **The built-in recorder** (`ahfd.debug.RawRecorder`) for consented staged
  sessions. This is gated behind three switches (config + `AHFD_ALLOW_RAW=1` +
  an explicit flag) and burns a banner into every frame. Extract from it, then
  **delete the video** and keep only the tracks.

Record the negatives too, and record more of them than falls. A 30-minute clip
of nobody falling is the denominator for the false-alarm rate, and it is where
most tuning value is. Suggested list: lying in bed (supine, prone, on side,
under a blanket), sitting on the bed edge, someone bending over the bed, two
people at one bed, making the bed, picking something off the floor. Falls go
onto a mattress or crash mat with a spotter — roll out of bed, slide off the
edge, slow slump down a wall, and one where the person gets straight back up
(that tests the cancel logic).

## 2. Extract keypoints

```bash
ahfd extract file://clips/fall_01.mp4 data/tracks/fall_01.jsonl
ahfd extract seq://datasets/urfd/fall-01/ data/tracks/urfd_fall_01.jsonl
```

Run once per clip. This is the only step that reads imagery; the pixels are
discarded and only keypoints are written.

## 3. Annotate

For each clip, write a small JSON file saying what actually happened. A fall
clip:

```json
{
  "clip_id": "fall_01",
  "duration_s": 22.5,
  "falls": [
    { "t_impact": 12.0, "t_start": 11.4, "subject": "volunteer_A",
      "notes": "roll out of bed onto mat" }
  ]
}
```

A negative clip — note the empty list, and that the duration is still required
(it is the false-alarm denominator):

```json
{ "clip_id": "adl_bed_01", "duration_s": 1800.0, "falls": [] }
```

`t_impact` is the reference instant — when the body reaches the floor. Put these
in `data/annotations/<clip_id>.json`.

## 4. Replay (see it, or produce events)

```bash
# Watch it play back as a skeleton -- no camera, no model, runs fast.
ahfd replay data/tracks/fall_01.jsonl --calibration calib/ward6.yaml --view skeleton

# Produce an events log for evaluation (set alert.jsonl_path in the config).
ahfd replay data/tracks/fall_01.jsonl --calibration calib/ward6.yaml --config configs/ward.yaml
```

The `--view skeleton` form is also the demo safety net: a canned clip stands in
for the live camera, so a USB fault cannot break a stakeholder demo.

## 5. Tune

```bash
ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 \
    --tracks data/tracks/ --annotations data/annotations/ \
    --calibration calib/ward6.yaml
```

This replays every clip at each threshold value and prints the recall vs
false-alarms-per-hour curve. Pick the operating point off that curve. Because
it runs on keypoints, the whole sweep takes seconds. Good thresholds to sweep:
`vz_trigger`, `confirm_s`, `down_spread`.

## 6. Evaluate

```bash
ahfd eval data/annotations/ data/events/ --out eval/report.md
```

Produces the report with **false alarms per camera-hour** first, then recall,
precision, latency, and a per-clip and per-false-alarm breakdown. That report
is the numbers section for the CDE3301 write-up.

## Notes

- **Calibration must match the clip's resolution.** Extract and replay both
  refuse a calibration whose resolution differs from the footage — intrinsics
  are per-resolution, and a mismatch produces plausible-but-wrong metres.
- **`data/tracks/` and `data/events/` are git-ignored.** They are your data,
  not source. The one committed tracks file is `tests/golden/`, which is a
  regression fixture.
- **Thresholds are metric, so they are per-ward, not per-camera.** Tune once
  against clips from one camera; the values carry to the others in that ward.
