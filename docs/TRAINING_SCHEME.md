# Next collection and training scheme

Status: research plan and collection support, not a clinically validated system.
This plan keeps Low / Medium / High and uses CUSUM warnings only for High.
The unit of collection is one consenting volunteer on one calibrated bed per
recording. Nearest and middle beds are recorded separately.

## 1. What is trained, calibrated, or configured

| Component | Work in this experiment |
|---|---|
| Camera, floor and two bed zones | Measure and calibrate manually; verify aligned metric depth and bed association. Do not learn geometry from activity labels. |
| Bed/occupant monitoring mode | Manually configure Low / Medium / High. Volunteer modes are simulated policy metadata, not medical risk assessments. |
| RTMO pose estimator | Keep the pretrained model fixed; record its version/hash. |
| Tracking, smoothing, depth features | Keep reproducible and causal; reset on gaps/reassociation. Missing depth remains missing. |
| Fall state machine | Tune separately against labelled falls and difficult normal activities. Negative-only recordings cannot measure fall sensitivity. |
| Bed-activity state machine | Tune transitions, corroborating cues, entry/exit hysteresis and dwell on development blocks. Bed support remains separate from activity phase. |
| CUSUM | Calibrate stable-rest baseline; compare onset/dwell settings on development blocks. It detects change, not exit intent by itself. |
| Logistic / small tree comparator | Train on causal temporal summaries for activity phase and future exit. Compare pose-only with pose + depth. |
| TCN / GRU / ST-GCN | Deferred until the simple baselines and additional independent data justify them. |

Risk is an alert policy, **not a behaviour-classification target or an input
that lets the model memorise who should alert**. Real patient risk assignment
belongs to authorised clinical staff. Test all three policy settings against
the same held-out observations; do not triple-count those observations as data.

| Mode | Intended research policy |
|---|---|
| Low | Fall detection. Bed activity may remain a dashboard observation. |
| Medium | Fall detection and completed bed exit; temporal activity observations. No CUSUM movement warning. |
| High | Medium plus corroborated early movement/attempt warnings using CUSUM and temporal evidence. Sitting up alone is not automatically a page. |

The collection command remains **shadow-only**: no clinical pages or live early
warnings are enabled by a plan. Simulated modes are stored separately from
geometry, and do not overwrite actual care settings.

## 2. Freeze the 80/20 split before recording

Choose **40 or 60 usable minutes**, not 40/60 wall-clock minutes. Warmup,
unlabelled time, unavailability and long sampling gaps do not count toward the
collector's completion gate. Label review may exclude further time afterward;
the manifest duration gate is not proof of accurate annotation.

| Block | Partition | 40-minute option | 60-minute option |
|---|---|---|---|
| A | Development | 4 min nearest + 4 min middle | 6 min nearest + 6 min middle |
| B | Development | 4 + 4 min | 6 + 6 min |
| C | Development | 4 + 4 min | 6 + 6 min |
| D | Development | 4 + 4 min | 6 + 6 min |
| E | Locked validation | 4 + 4 min | 6 + 6 min |

This is ten recordings: 32/8 or 48/12 minutes. Each block includes both exits
and non-exits. Randomise bed order and vary timing/speed without using a fixed
"exit at second 20" script. Collect E as a fresh block with the same core task
coverage, not a wholly different class distribution.

Use A-D for model/threshold selection with leave-one-block-out development
folds; **both beds and any repeats of a block stay in the same fold**. Fit
imputation/scaling and any feature selection on the training fold only. Then
refit the chosen settings on all A-D and evaluate E once. Do not split frames
or overlapping windows randomly. Do not tune after seeing E and still call E
held out; another independent block would then be needed.

One volunteer is acceptable for a prototype, but the result must be called
**same-volunteer block holdout**, not unseen-patient validation. Not doing
person identification does not remove person-specific movement correlations.
Keep October 13 data in a separate development/regression experiment, outside
this new 80/20 comparison. Report counts/durations after exclusions; do not move
E samples into A-D to recover an exact ratio.

## 3. Collection content: new variations, not repeated timed scripts

For each 8-minute paired block, aim for about five minutes of ordinary activity
and three minutes of varied transitions. For 12-minute blocks, extend natural
rest and variation rather than accelerating the actions. Where comfortable,
aim for two ordinary exits per bed per block; never rush to meet a quota.

| Block | Emphasis, in addition to normal rest and ordinary exits |
|---|---|
| A | Natural resting, reading/phone use, reaching, rolling, sitting up, reclining. |
| B | Different comfortable speeds; long/short pauses; sitting upright without exiting. |
| C | Edge sitting, aborted standing/exit, return to recline, repositioning, leaving and returning. |
| D | Safe fall lookalikes: bending, picking up an object, sitting in a chair, controlled squatting if comfortable. Approved clothing/lighting variation. |
| E | Fresh natural mixture of the above on both beds, after the protocol is frozen. No model-feedback-driven performance. |

Keep the existing fixed-angle setup only after checking it again. Manually draw
actual bed support outlines, not expanded polygons that hide true exits. Any
tolerance/hysteresis belongs in a separately measured rule. Record both bed
heights, camera height/pitch/roll, depth coverage, usable range and calibration
hashes in approved records. Recheck geometry after camera or bed movement;
articulation currently aborts a collection and requires recalibration.

No blanket is required for this collection; report blanket occlusion as
untested. Staff occlusion and simultaneous occupants need a separately approved
multi-person collection design: **the present collector aborts on extra people**.
Do not bypass that guard to collect two occupied beds at once.

Do not ask patients to fall, slide out, or climb rails. Staged positive falls
require a separate approved protocol, trained consenting volunteers, supervision
and appropriate landing equipment. If unavailable, collect normal lookalikes
and report fall false alarms only; do not claim fall recall. A point-marker key
is not permission to perform an activity.

## 4. Human labels and calibration records

Keep these separate:

1. **Geometry:** calibrated bed polygons, floor, camera/depth calibration.
2. **Association/policy:** the selected volunteer, current bed and simulated
   mode. No names, diagnoses or hospital bed numbers in research files.
3. **Observed activity:** actual phase changes and event times, not planned
   script times or the model's predictions.

Phase keys are `1` RECLINED, `2` TORSO_RISING, `3` UPRIGHT_IN_BED,
`4` SHIFTING_TO_EDGE, `5` EDGE_SITTING, `6` ATTEMPTING_STAND,
`7` OUT_OF_BED, `0` UNKNOWN. Phases need not occur in order and may be skipped.
Bed support is an independent measured attribute. Agree the completed-exit
definition before labelling: the transfer off the bed is complete, not merely
an image point crossing a polygon.

Additional **point** markers:

| Key | Marker |
|---|---|
| e | BED_EXIT_COMPLETED |
| b | ABORTED_EXIT |
| w | WARNING_WORTHY_ATTEMPT |
| j | FALL_ONSET |
| k | FLOOR_CONTACT |
| l | RECOVERY |
| o | INTENTIONAL_FLOOR_POSTURE |

These markers complement phase intervals; they do not automatically create
training intervals, infer intent, or replace the existing phase-based exit
target. Review event boundaries and uncertainty before using them for event
evaluation. Fall onset/contact/recovery can delimit a reviewed fall event.
Use `u` (observer unsure), `t` (track error), or the relevant occlusion marker
when unobservable; usable-time credit pauses until a new valid phase marker.

An aborted stand can still warrant an assistance warning. Keep **actual future
exit**, **warning-worthy attempt** and **fall** as distinct targets. Do not mark
every non-completed exit as a false High-risk warning without clinical review.

## 5. Training and rule experiments

1. Audit every manifest, stream hash, approval, calibration and association.
   Select one accepted recording per block/bed before training; retain failures
   and retry reasons separately. Review labels independently of predictions.
2. Derive features causally from current/past samples only. No centred smoothing,
   future interpolation, full-clip statistics, or windows crossing recording,
   gap or association boundaries. Preserve confidence/missingness masks.
3. First compact comparator: current pose and 0.5/2-second summaries; then add
   aligned relative depth. Use floor-relative height only with verified
   calibration. Compare pose-only vs pose+depth rather than assuming depth helps.
   The governed collector also saves its existing 0.5/2/4/8-second summaries;
   compact-feature ablations must be explicit, versioned projections/replays.
4. Proposed small learned matrix: logistic/tree x pose/pose+depth x
   phase/exit-in-5s/exit-in-10s/exit-in-20s = **16 candidates**. Start the compact
   phase experiment with RECLINED, UPRIGHT_IN_BED, EDGE_SITTING and OUT_OF_BED;
   retain intermediate labels for temporal rules and later experiments. The
   existing comparator currently uses all eligible observed phases.
5. Future labels mean an exit in `(t, t+H]`, while inputs end at `t`. Censor an
   unknown future, recording tail, identity break or unavailable interval;
   unknown is not a negative. Report the remaining support for each horizon.
6. Fit preprocessing and class/record weighting on development folds only.
   Select settings using block-level results and event review, then freeze.
   Model scores are not automatically calibrated clinical probabilities.

### CUSUM and fall rules

Begin each bed take with 30-60 seconds of confidently stable reclined rest as
an engineering baseline opportunity, with no forced immobility if uncomfortable.
Estimate/update the baseline only during confidently stable rest, never absorb
a continuing rise into it. Invalid observations or reassociation pause/reset
accumulation; availability must remain explicit.

Compare the existing CUSUM threshold `6` / corroboration dwell `0.60 s` against
`6` / `0.40 s` in offline shadow experiments first. These are proposed test
settings, not validated clinical limits. A short dwell must still require
confidence and edge-directed/support-loss/stand-attempt evidence; CUSUM alone
is not a High-risk alert. Measure both nuisance warnings and detection delay.
Do not change the Low/Medium policy to emit CUSUM warnings.

Tune fall thresholds separately, with labelled positive events if available
and difficult non-fall examples. Do not OR the learned score and CUSUM together
without evaluating the resulting combined event policy. This collection update
does not change any detector threshold or activate a trained model.

## 6. Validation and exit criteria

Report by bed and overall: phase confusion/support; completed exits detected
and missed; warning-worthy attempts; duplicate alerts; first-alert lead time;
fall sensitivity only when positive falls exist; false alert episodes per
**available monitoring hour**; and total unavailable time/reasons. Audit wrong
bed/association and Low/Medium CUSUM suppression. Include uncertainty and sample
counts; eight or twelve validation minutes cannot establish ward reliability.

Before opening E, freeze model, feature order, preprocessing, rule settings,
alert deduplication, event-matching tolerance and clinical warning definition.
The chosen anticipation horizon must eventually reflect measured nursing
response time; 5/10/20 seconds are experimental comparisons only.

After E, use a supervised live shadow session with fresh actions and hardware
checks. No deployment or paging gate is passed merely by completing this plan.

## 7. Collection commands and what is implemented

Use the existing [onsite SOP](ONSITE_COLLECTION.md) for approvals, approved
external storage, preflight and calibration. The examples below use placeholder
paths and pseudonyms. The two bed codes must exactly match the approved
calibration; the plan is not a replacement for that approval.

```powershell
ahfd collection-plan --out-root D:\approved-study --site-id site_0123abcd --participant-id sub_0123456789abcdef --bed-a bed_a --bed-b bed_b --mode-a medium --mode-b high --minutes 40
```

Keep the printed `plan_<random>.json` and hash outside Git. Do not edit it
mid-study. Pass that same file to all ten recordings:

```powershell
ahfd collect --config configs/onsite_collection.yaml --calibration D:\approved-study\camera.yaml --approval D:\approved-study\approval.json --out-root D:\approved-study --site-id site_0123abcd --participant-id sub_0123456789abcdef --plan D:\approved-study\plan_<random>.json --block A --target-bed bed_a
```

Repeat A for bed_b, then B/C/D/E for both beds. Bind the volunteer with `[` or
`]`, then mark actual phase changes. The selected person must be observed
supported by the target bed; a different bed aborts. A new binding must verify
bed support again. The overlay shows block, partition, simulated mode and
usable seconds. Collection stops at 240 usable seconds (40-minute plan) or
360 (60-minute plan). Early `q`, a wall-clock/frame limit, or `Esc` cannot
produce a complete planned recording before the gate is satisfied. A take can
run longer than its target; the operator must stop if tired or unsafe.

Each session embeds the frozen plan, its canonical SHA-256, block/partition,
target bed, target duration and achieved progress in its manifest. Hashes detect
inconsistent plans, not malicious tampering or incorrect human annotations.

For development only, list the accepted A-D session directories explicitly:

```powershell
ahfd compare-bed-exit D:\approved-study\ses_<A_nearest> D:\approved-study\ses_<A_middle> D:\approved-study\ses_<B_nearest> D:\approved-study\ses_<B_middle> --group-by block
```

The short example illustrates arguments; use all eight accepted A-D sessions
for the planned experiment. The command rejects E **before reading observation
streams**, rejects mixing planned/legacy data or different plan hashes, and
requires block grouping. It gives development, frame-level comparator scores;
it is **not** the final locked-E event evaluation. A fixed-holdout evaluator,
the compact 16-candidate ablation and a final event report remain subsequent
implementation work. Ordinary legacy collections remain supported.

### Storage and raw recording boundary

This updated `collect` command saves derived measurements, **not RGB or dense
depth video**. It neither starts nor changes the separate raw recorder. If
approved RGB+depth recording is required, use the separate raw-recording SOP,
check synchronisation and shared session mapping, and budget storage first.
Colourised depth previews are not metric training depth.

The earlier uncompressed setup was estimated at roughly 6.3 GB/minute:
approximately 252 GB for 40 minutes or 378 GB for 60, before working copies.
Measure the actual recorder's bytes/minute in an approved short pilot and allow
headroom; do not assume those estimates apply to another stream profile. Derived
recordings are smaller, but still require preflight free-space checks.

Keep plans, raw/derived recordings, real calibrations, participant linkage,
labels, fitted models and study reports on approved restricted storage, not
GitHub. Git contains this scheme, schemas, code and synthetic tests only.

## 8. Implementation order after collection support

1. Confirm storage and geometry, freeze plan and event definitions.
2. Collect ten takes, review labels/quality and freeze the accepted manifest.
3. Run A-D block-held-out comparisons; implement the compact ablations and
   event-level rule/policy evaluation; freeze the selected pipeline.
4. Evaluate E once, document limitations, then conduct fresh live shadow tests.

Method references: [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),
[probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
and [CUSUM principles](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm).
