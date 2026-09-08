# Tuning the hyperparameters

The detection knobs live in `configs/ward.yaml`. This is how to set them
*with data* rather than by guessing — and why guessing is the wrong instinct.

## The one rule: tune on data, measure on held-out data

You have two things to protect against:

- **Missing falls** (low recall) — the system fails its job.
- **False alarms** (low precision) — nurses stop trusting it and switch it off.

These trade off: loosen a threshold and you catch more falls but cry wolf more.
There is no single "correct" value — there is an operating point *you choose* on
that trade-off, and the only honest way to choose it is against labelled clips.

Split your clips into a **tune set** (~70%) and a **held-out test set** (~30%).
Tune against the tune set; touch the test set once, at the end, for the numbers
you report. Tuning on the clips you then report on flatters the result — it's the
same mistake as training on the test set.

## The mental model for each knob

Every threshold pushes recall and false-alarms in opposite directions. Before
sweeping, know which way:

| Knob | Loosen it (→) | Effect |
|---|---|---|
| `vz_trigger` | toward 0 (e.g. -0.9 → -0.6) | triggers on gentler drops: **more falls caught, more false alarms** |
| `drop_trigger` | smaller (0.45 → 0.30) | same direction |
| `confirm_s` | shorter (8 → 4) | alerts sooner, but more brief lie-downs become alarms |
| `confirm_motion_max` | larger | a fidgeting fallen person still confirms: more recall, more FPs |
| `down_spread` | wider band | more postures count as "down": more recall, more FPs |
| `min_valid_kp` / `min_mean_conf` | lower | acts on poorer pose: more coverage, more noise |
| `slow_down_s` | shorter | catches slow slumps sooner, but a patient sitting on the floor alarms faster |
| `bed_exit_s` | shorter | earlier bed-exit warning, more nuisance triggers |
| `cooldown_s` | shorter | more repeat alerts per incident |

Tighten (opposite direction) and every effect reverses: fewer false alarms,
more misses.

## The workflow

### 1. Get labelled clips into keypoints

```bash
ahfd extract file://clips/fall_01.mp4 data/tracks/fall_01.jsonl
# ...one per clip. Annotate each in data/annotations/<clip>.json (see docs/CLIPS.md).
```

Do this once. Everything below replays the keypoints, so it's fast and
repeatable — no re-running pose.

### 2. Sweep one knob at a time

```bash
ahfd sweep detect.vz_trigger --range -1.5:-0.5:0.1 \
    --tracks data/tracks/ --annotations data/annotations/ \
    --calibration calib/ward.yaml
```

This replays every clip at each value and prints a row per value:

```
vz_trigger  |  recall  |  FA/hour  |  latency(s)
   -1.500   |   0.62   |    0.05   |    8.1
   -1.100   |   0.88   |    0.14   |    8.0
   -0.900   |   0.94   |    0.28   |    8.0
   -0.700   |   0.97   |    0.9    |    8.1
```

Read it as a curve: as you loosen, recall climbs and so do false alarms. **Pick
the point where recall is acceptable and FA/hour is still below what the ward
tolerates** (ask AH for that number — see the AH questions doc). Above, ~-1.0
buys 0.88 recall at 0.14 FA/hour; -0.7 gains a little recall for a big jump in
false alarms — a bad trade.

### 3. Set it, then sweep the next

Put the chosen value in `configs/ward.yaml`, then sweep the next knob. Sweep the
high-impact ones first — `vz_trigger`, `confirm_s`, `down_spread`, `bed_exit_s` —
the rest usually need little movement from the defaults.

Knobs interact, so after setting several, do one more pass over the two or three
that matter most. You are not searching for a global optimum; you are finding a
defensible operating point.

### 4. Report on the held-out set

```bash
ahfd replay <each test clip> ... --config configs/ward.yaml   # writes events
ahfd eval data/annotations_test/ data/events_test/ --out eval/report.md
```

Those are the numbers for the report: recall, **false alarms per camera-hour**,
and latency, on clips you never tuned against.

## What NOT to tune

- **Calibration** (height, pitch, zones) — those are *measured*, not tuned. A
  wrong height doesn't get fixed by moving a threshold; it corrupts every metric.
  If detections look wrong everywhere, check calibration first (the live metrics
  HUD's ankle-height readout should sit ~0.05 m).
- **`down_h_torso`** — leave it near 0.90. It's a loose sanity guard; the real
  posture test is `down_spread`. Tightening it silently loses falls at the near
  bed (this bit us once — see tests/test_config.py).
- **Smoothing** — the defaults are standard One-Euro values; only touch them if
  keypoints are visibly jittery (raise `beta`) or laggy (raise `min_cutoff`).

## A worked example

Suppose staged clips show 2 of 20 falls missed and 3 false alarms over 5 hours:

1. Missed falls → sweep `vz_trigger` and `drop_trigger` looser; check recall rises.
2. If the misses were slow slumps (no velocity spike), they won't move with
   `vz_trigger` — shorten `slow_down_s` instead.
3. False alarms → open the `ahfd eval` report's false-alarm table. If they're all
   "person sat on the floor", that's the `slow_down_s` path; if they're "nurse bent
   over bed", the posture bands need attention. **The report tells you which knob**,
   which is why you read the per-scenario breakdown, not just the F1.
4. Re-sweep, re-measure on the held-out set, done.
