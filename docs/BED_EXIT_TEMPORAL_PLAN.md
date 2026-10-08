# Bed-exit temporal implementation and next-week plan

## Status and scope

This is an engineering research build for causal, early bed-activity analysis.
It is not clinically validated and must run in shadow mode during the first
onsite collection. It does not replace observation, direct care, a nurse-call
system, or the hospital's existing fall precautions.

The implementation now has three comparable layers over the same causal input:

1. an explainable hierarchical `BedExitStateMachine`;
2. a rest-gated, one-sided CUSUM movement-onset signal; and
3. fold-safe logistic-regression and shallow-tree comparators over causal
   trailing summaries.

The first neural model is deliberately deferred. A causal TCN is only worth
training after multiple participants and complete, quality-masked temporal
labels exist. The derived session format and 0.5/2/4/8-second summaries are
already suitable as its future input contract.

## Runtime design

Fall state and bed activity are independent. The dashboard can therefore show
`SITTING` as the fall/posture state while the bed layer says
`SHIFTING_TO_EDGE`, and losing visibility does not overwrite the last activity
phase with normality.

The bed layer stores three separate dimensions:

- activity phase: `UNKNOWN`, `RECLINED`, `TORSO_RISING`,
  `UPRIGHT_IN_BED`, `SHIFTING_TO_EDGE`, `EDGE_SITTING`,
  `ATTEMPTING_STAND`, `OUT_OF_BED`;
- physical support: `UNKNOWN`, `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`; and
- observation: `VALID`, `LOW_CONFIDENCE`, `MONITORING_UNAVAILABLE`.

All cues are evaluated every frame, so transitions may be skipped. A rapid
exit can move from reclined directly to attempting stand or out of bed. A slow
slide can move through edge-directed motion without a torso-rise onset. A
return to sustained, supported recline resets the episode. Entry and exit
bands differ, and phase candidates have measured dwell times; missing frames
pause or reset evidence rather than count as persistence.

The provisional edge cue is the signed distance from the shoulder/hip anchor,
projected onto the associated bed surface, to the nearest polygon boundary.
Positive means inside and negative means outside. At the hospital, mark the
true egress edge in a later calibration revision if the nearest-edge assumption
is not valid for the bed/rail arrangement.

## CUSUM contract

For shoulder elevation `h(t)` the implementation uses:

```text
z(t) = (h(t) - baseline_mean) / max(baseline_std, epsilon)
g(t) = max(0, g(t-1) + (z(t) - k) * elapsed_time_scale)
```

Baseline samples are accepted only while the activity owner sees confident,
bed-supported, stable recline. Initial frames are not assumed to be rest. An
invalid sample pauses the detector; a long gap/reassociation resets temporal
evidence. Once onset fires, a flat elevated plateau cannot become a new
baseline. Only a confirmed return-to-recline episode reset re-arms it.

CUSUM is evidence of sustained change, not a prediction of intent. The default
early-warning policy remains off. A shadow warning candidate requires rise
evidence plus corroboration such as persistent edge progress, partial support,
or a later activity phase. Sitting upright without lateral progression stays a
dashboard observation for nurse review.

## Learned comparator

`CausalTemporalSummarizer` emits current values, missingness masks, change,
slope, min/max/range-like statistics, maximum rise, and valid fraction over
trailing 0.5, 2, 4 and 8 second windows. It also includes near-edge dwell,
cumulative edge progress (reset on confirmed recline), support-loss time, edge
velocity, CUSUM values, sparse-depth availability masks and live IMU orientation
deltas. Absolute track age is retained for audit but excluded from the default
learned feature set because it can encode scripted session order.
All windows end at the current sample; there is no centred smoothing, future
interpolation, bidirectional recurrence, or full-clip feature.

The first comparator uses:

- median imputation fitted inside each training fold;
- scaled, class-balanced logistic regression; and
- a shallow, class-balanced decision tree.

Every prediction can expose either signed logistic contributions or the tree's
decision path. Patient risk is not a model input; an approved care policy can
act on model output later. Evaluation holds out complete participants, or
complete sessions for development only. Track IDs and filename suffixes are
never treated as identities. `ahfd compare-bed-exit` verifies completed
derived-only session hashes, samples by past-only zero-order hold at 10 Hz,
censors monitoring gaps and reassociations, and reports participant-held-out
out-of-fold probabilities. Phase and 5/10/20-second average precision, Brier,
precision/recall and false-positive metrics are frame-level development
baselines, not event-level clinical evidence.
The collector processes every frame but stores model rows at most 10 Hz as
an ordered numeric vector. The feature order and its SHA-256 appear once in the
manifest; this removes repeated JSON keys and keeps the first-study files
manageable. Only the explicitly bound participant association is persisted.

## Onsite sequence for next week

### Before the hospital

1. Obtain written research/privacy approval for the exact participant scope,
   derived fields, encrypted destination, access list, retention and deletion.
2. Freeze a reviewed commit and environment; run the full test suite.
3. Create an external calibration for the exact D435i, resolution and mount.
   Use a pseudonymous `cam_<code>`, retain the automatically hashed device
   identity, measure the standing two-ankle baseline with
   `ahfd measure-ankle-baseline`, remove all test/dated/duplicate zones, set bed
   risk to `unknown`, measure current mattress height, complete depth/IMU drift
   checks, then set `verified_for_onsite: true`. Have the custodian create a
   fresh external approval record bound to the final config, calibration and
   reviewed pose-model SHA-256 values, and
   provision the encrypted output-volume marker described in the SOP.
4. Run an empty-room and consenting-staff dry run with
   `configs/onsite_collection.yaml`. Confirm no `.mp4`, `.bag`, `.db3`, image or
   frame cache appears.
5. Verify explicit target binding, track-loss re-binding, second-person abort,
   and empty-room person abort. Then unplug/cover/move the camera and verify explicit unavailable/calibration
   handling. Verify `Esc` produces an `aborted` requested-stop session while
   `q` completes only a normally finished protocol block. Confirm a crash
   leaves an `incomplete` or `aborted` manifest.

### Collection order

Collect coverage before volume:

1. long ordinary negative activity and empty-bed time;
2. recline, pause, sit-up-and-return and edge-approach-and-retreat;
3. normal unassisted transfers, when approved;
4. fast transitions, slow slides and rail-supported paths with consenting
   staff/volunteers and the hospital's safety procedure; and
5. single-person confounders: blankets, curtains, mobility aids, lighting
   change and partial occlusion. Test bed articulation only as an abort and
   recalibration drill; do not treat the resulting session as training data.

The first-week collector is intentionally restricted to one explicitly bound
person and aborts when a second pose appears. Assisted transfers, staff-at-bed
confounders and any other multi-person scene require a separately approved
protocol and a future collector mode with explicit consent scope; do not try to
capture them with this build.

Do not ask patients to stage falls or exits. If patient observation is not
explicitly approved, use consenting staff/volunteers only.

The observer marks actual visible transitions and context; phases need not be
complete or sequential. A second observer should review a sample of sessions.
Uncertain and unavailable intervals stay masked, never converted to negatives.

### Daily closeout

Verify manifest completion, counters and hashes; reconcile controlled markers;
quarantine incomplete sessions; transfer to the hospital-managed encrypted
store; verify hashes; apply the approved deletion rule; and review coverage,
reassociation, depth loss and calibration drift before collecting more.

## Evaluation gates

Freeze participant-held-out test data before tuning. For hand rules, CUSUM and
each learned comparator, report on identical eligible time:

- monitoring coverage and unavailable/degraded time by reason;
- event-level exit sensitivity and missed exits;
- nuisance/false alerts per available monitored hour;
- warning time per hour;
- first raw threshold crossing and final operational alert time;
- median/IQR lead time to observed out-of-bed and the proportions with at
  least 5, 10 and 20 seconds of warning;
- phase macro-F1 and balanced accuracy;
- horizon AUPRC, Brier/calibration and event-level coalesced performance; and
- results by participant, session, mount/day and major confounder.

No clinical paging should be enabled from this study merely because an offline
metric looks promising. Nurse review, a defined response workflow, prospective
validation and hospital approval are separate go/no-go gates.
