# Bed-exit detection: specification

Status: draft for implementation. Save as `docs/BED_EXIT_SPEC.md`.

This spec adds an **independent bed-exit warning branch** next to the existing
fall detector. It is grounded in `src/ahfd/detect/state_machine.py` as it stands
today. Statements marked **[verified]** come from that file. Statements marked
**[assumed]** depend on code I have not seen (`features/extractor.py`,
`detect/events.py`, `geometry/`, `config.py`, the CLI) and must be checked
against the repo before building on them.

All numeric values are **untuned starting points** for staged-clip tuning, in the
same spirit as `FallThresholds`. None is a validated clinical setting.

---

## 1. Purpose and scope

**Goal.** Recognise that a patient is leaving, or is preparing to leave, the bed
early enough to support a nurse intervention, and grade the response by the
bed's fall-risk level.

**In scope**

- A new `BedExitStateMachine` that runs in parallel with `FallStateMachine`.
- Bed-relative geometry that models the bed's two side guardrails and the
  unprotected head and foot ends.
- Body-core evidence that ignores arms, hands and head for the exit decision.
- Graded events using the existing `BED_EXIT_SEVERITY_BY_RISK` mapping.
- Explicit handling of poor observation (degraded or occluded) instead of
  treating missing evidence as safe.

**Out of scope for v1**

- Learned models (causal TCN, GRU, ST-GCN). These come after labelled precursor
  sequences exist; this spec produces the rule-based baseline they must beat.
- Inferring intent. The system measures a motion signature and reports it. It
  does not decide whether an exit was deliberate or accidental.
- Assisted-transfer mode and nurse-presence suppression (see section 12).
- Raw or depth-image playback. Privacy rules below still apply.

**Non-goals.** This is not a claim to predict or prevent all falls. Sitting up is
an observable action, not proof that a fall will follow.

---

## 2. What the current code does, and the gaps

**[verified]** from `state_machine.py`:

| Behaviour | Where | Consequence for bed exit |
|---|---|---|
| `_on_bed(f)` returns `f.supported_by_bed is not None` | `_on_bed` | Bed support is a single boolean decided upstream in `FeatureExtractor._supporting_bed`. |
| When `on_bed` is true, the track becomes `IN_BED`, and `sitting_since` and `bed_exit_emitted` are reset | start of `_steady_state` | A patient sitting up **while still bed-supported never enters any precursor path**. This is the main gap. |
| `BED_EXIT` is emitted only from `_enter_sitting`, when the track is `SITTING`, `near_bed` is true and seated for `bed_exit_s` (3.0 s) | `_enter_sitting` | The event means "sat near a bed for 3 s", not "left the bed". |
| `near_bed = any("bed" in z.lower() for z in f.zones)` | `_enter_sitting` | Any seated track near any zone whose name contains "bed" qualifies: visitors, staff, a patient in the next bed. No association with an occupied bed. |
| Trigger detection requires `not on_bed` | `update` | A fall that starts from a bed-supported pose is not triggered as an impact fall. The slow-slump path (`PERSON_DOWN`) still applies once the track is down outside a bed. |
| Quality gate: `n_valid_kp < min_valid_kp (8)`, `mean_conf < min_mean_conf (0.40)` or no geometry gives `LOW_CONFIDENCE` and returns `None` | `update` | Floor-geometry gate shared by all decisions. Bed-relative evidence needs its own, upper-body- and lower-limb-specific gate. |
| `f.zones[0]` is used as the event zone | `update` | Event zone is whichever zone is listed first, not necessarily the occupied bed. |
| `update(f)` takes the timestamp from `f.t` and holds no wall clock | class docstring | The new machine must follow the same rule, so it is testable from synthetic sequences. |
| No trained parameters; every threshold lives in a frozen dataclass populated from config | `FallThresholds` | The new thresholds follow the same pattern. |

**Interaction hazard to design for.** When a patient steps out and the track is no
longer bed-supported, the existing machine may classify them `SITTING` or
`UPRIGHT` next to a zone named "bed" and emit the legacy `BED_EXIT`. Without
coordination, one real exit would produce two bed-exit events from two machines
(see section 8.4).

---

## 3. Design principles

1. **Separate machine, parallel.** `BedExitStateMachine` is a new class. It must
   not alter `FallStateMachine` behaviour, thresholds or cooldown. A low-priority
   bed-exit event must never block a later urgent fall alert.
2. **Body core decides; limbs inform.** Arm or hand crossings must not trigger
   anything. Leg crossings count as a distinct, earlier stage.
3. **Unknown is not safe.** If the joints needed for a decision are not
   observable, the machine reports `DEGRADED`. It never infers "still in bed" or
   "exited" from missing data.
4. **Metres, not pixels.** All geometry is in the bed frame in metres, so one
   threshold set covers every camera, consistent with the rest of the project.
5. **Explainable.** No learned parameters. Every event carries the evidence that
   produced it.
6. **Privacy preserved.** No image field, no image writes. `tests/test_privacy.py`
   must pass unchanged.
7. **Testable offline.** Timestamp is an argument; no camera, video or pose model
   is needed to test the machine.

---

## 4. Architecture

```
capture -> pose -> tracking -> smoothing -> geometry -> FeatureExtractor
                                                          |
                          +-------------------------------+--------------+
                          |                                              |
                  FallStateMachine (unchanged)         BedFrameExtractor (new)
                          |                                              |
                          |                                   BedExitStateMachine (new)
                          |                                              |
                          +------------------ events ---------------------+
                                              |
                                       alert sinks / dashboard
```

### 4.1 New and changed files

| File | Change |
|---|---|
| `src/ahfd/features/bed_frame.py` | **New.** Converts a track's joints to the bed frame and computes signed edge distances. |
| `src/ahfd/detect/bed_exit.py` | **New.** `BedExitThresholds`, `BedExitState`, `BedExitStateMachine`. |
| `src/ahfd/detect/events.py` | **Changed** [assumed]. Register `BED_EXIT_RISK`, `BED_EXIT_CONFIRMED`, `BED_EXIT_ABORTED`, `BED_MONITORING_DEGRADED`. |
| `src/ahfd/geometry/` | **Changed** [assumed]. Extend the zone schema with `edges` and rail dimensions. |
| `src/ahfd/config.py` | **Changed** [assumed]. Load a `bed_exit:` section into `BedExitThresholds`. |
| `src/ahfd/cli.py` and the pipeline loop | **Changed** [assumed]. Run both machines per frame and merge events. |
| `configs/ward.yaml`, `configs/detect_dev.yaml` | **Changed.** Add a `bed_exit:` section. |
| `tests/test_bed_exit.py` | **New.** Synthetic-sequence tests (section 11). |
| `src/ahfd/detect/state_machine.py` | **Minimal, optional.** Only the legacy `BED_EXIT` dedupe hook in section 8.4. |

### 4.2 Required upstream data [assumed]

The `Features` object that `FallStateMachine` consumes exposes `track_id`, `t`,
`h_torso`, `v_z`, `floor_spread`, `motion`, `torso_tilt`, `n_valid_kp`,
`mean_conf`, `zones`, `bed_risk`, `supported_by_bed`, `range_m`,
`in_excluded_zone` and `has_geometry()`. **[verified]** by use in
`state_machine.py`.

It is **[assumed]** not to expose per-joint positions or per-joint confidences.
The new machine needs both. Two options; pick after reading `extractor.py`:

- **A (preferred):** `BedFrameExtractor` reads the smoothed keypoints and
  per-joint confidences directly from the same stage that feeds
  `FeatureExtractor`, and outputs a `BedFrameFeatures` object keyed by
  `track_id` and timestamp.
- **B:** extend `Features` with a joint-position dict. Larger blast radius for
  the fall machine's tests; avoid unless A is impractical.

---

## 5. Bed-frame geometry

### 5.1 Frame definition

For each bed zone, derive a bed frame from its calibrated floor polygon:

- origin: polygon centre
- x axis: along the bed's long axis (head to foot)
- y axis: across the bed (left rail to right rail)
- z: up, with the mattress surface at `top_m`

For a rectangular bed, use the polygon's longest edge for x. If the polygon is
not a rectangle, fit the minimum-area rectangle and warn.

### 5.2 Edges and guardrails

The patient crosses a rail to exit sideways. The head and foot ends have no
rails and are separate exit paths that must be monitored.

```yaml
zones:
  - name: bed_1
    kind: bed
    top_m: 0.55
    risk_level: high            # none | low | medium | high
    polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    edges:
      - {side: left,  rail: true,  rail_height_m: TBD, rail_thickness_m: TBD, rail_length_m: TBD}
      - {side: right, rail: true,  rail_height_m: TBD, rail_thickness_m: TBD, rail_length_m: TBD}
      - {side: foot,  rail: false}
      - {side: head,  rail: false}
```

**Guardrail dimensions to supply (not provided in the material this spec was
written from):**

| Quantity | Value | Used for |
|---|---|---|
| Rails per long side (1 or 2 segments) | TBD | Whether the rail line has a gap a patient can use |
| `rail_height_m` above mattress | TBD | Sanity check that a core joint over the line is physically plausible |
| `rail_thickness_m` | TBD | Width of the rail band between "inside" and "outside" |
| `rail_length_m` and offset from head end | TBD | Limits the rail line to its real extent; beyond it the edge is open |
| Whether rails can be lowered | TBD | Per-episode `rail:` flag; a lowered rail is not a barrier |

Rail extent matters: a rail that covers only part of the bed leaves an open
section of that side. Model each rail as a segment on the edge, and treat the
rest of that side as `rail: false`.

### 5.3 Signed distance

For each edge `e` and joint `j`, compute `d_e(j)`: the signed distance, in
metres, from the joint's bed-frame position to the edge's **rail line** (the
outer face of the rail band for railed edges, the mattress edge otherwise).
Positive is inside the bed, negative is outside.

**Projection.** Project joints onto the bed-surface plane at `top_m`, not the
floor plane. **[verified]** (via the `_on_bed` docstring) that floor-contact
heights are wrong for an elevated body; the same reasoning applies here.
Projection error grows for joints well above the mattress (sitting patient) or
below it (legs over the edge), so edge margins must exceed that error. When valid
aligned depth exists, deproject instead and drop the error term. The thresholds
below assume the plane projection.

---

## 6. Evidence model

### 6.1 Joint groups

| Group | Joints (COCO-17) | Role |
|---|---|---|
| Core | mid-hip (from L/R hip), mid-shoulder (from L/R shoulder) | Decides "body is out" |
| Lower limb | L/R knee, L/R ankle | Decides "legs are over" |
| Upper limb, head | wrists, elbows, nose, ears, eyes | Logged in evidence only. **Never decisive.** |

A mid-hip needs at least one valid hip; a mid-shoulder needs at least one valid
shoulder.

### 6.2 Per-frame evidence

Per tracked patient, per frame, for the governing edge:

- `d_core`: the smaller signed distance of the valid core points
- `d_legs`: signed distances of valid lower-limb joints
- `n_core_valid`, `n_leg_valid`
- `legs_over`: count of valid lower-limb joints with `d < 0`
- `core_speed_toward_edge`: rate of decrease of `d_core` over 1 s (m/s)
- `observable`: true if quorum is met (section 9)

### 6.3 Baseline

While `IN_BED_STABLE`, maintain a rolling median of `d_core` over
`baseline_window_s`. A patient who sleeps near a rail is not approaching it. An
approach is measured as **displacement toward the edge relative to the baseline**,
not as proximity alone. Freeze the baseline when leaving `IN_BED_STABLE` and
restore it on return.

### 6.4 Edge selection

Select the governing edge as the one with the smallest `d_core` when leaving
`IN_BED_STABLE`. **Lock it** until `ABORTED` or `EXITED`, so a patient moving
along the bed does not flip the decision between edges.

---

## 7. State machine

### 7.1 States

```python
BedExitState = Literal[
    "NOT_IN_BED",       # no bed episode bound to this track
    "IN_BED_STABLE",    # bed-bound, no approach evidence
    "EDGE_APPROACH",    # core moving toward the locked edge
    "LEGS_OVER",        # lower limbs beyond the rail line
    "CORE_CROSSING",    # mid-hip at or past the rail line
    "EXITED",           # mid-hip beyond the line, sustained
    "DEGRADED",         # insufficient observable evidence
]
```

`ABORTED` is an event, not a resting state: the machine emits it and returns to
`IN_BED_STABLE`.

### 7.2 Transitions

Skipping forward is allowed (a fast roll-off can go `IN_BED_STABLE` straight to
`CORE_CROSSING`). Returning backward is allowed through the abort rule.

| From | To | Condition (sustained per `evidence_window_s` / `evidence_fraction`) |
|---|---|---|
| `NOT_IN_BED` | `IN_BED_STABLE` | Track bound to a bed episode (section 10) |
| `IN_BED_STABLE` | `EDGE_APPROACH` | `d_core <= edge_near_m` **and** displacement toward edge from baseline `>= approach_delta_m`, held for `approach_s` |
| `IN_BED_STABLE`, `EDGE_APPROACH` | `LEGS_OVER` | `legs_over >= legs_quorum` for `legs_over_s` |
| any earlier | `CORE_CROSSING` | `d_core <= crossing_margin_m` for `evidence_window_s` |
| `CORE_CROSSING` | `EXITED` | `d_core <= -exit_margin_m` for `exit_confirm_s` |
| `EDGE_APPROACH`, `LEGS_OVER`, `CORE_CROSSING` | `IN_BED_STABLE` (emit `BED_EXIT_ABORTED`) | `d_core >= return_margin_m` for `abort_s` and `legs_over == 0` |
| any state | `DEGRADED` | Observation quorum lost for `degraded_after_s` |
| `DEGRADED` | previous state or re-evaluate | Quorum regained for `recover_s`; do not carry forward counters across the gap |

**Hysteresis.** Entry margins and return margins differ (`exit_margin_m` vs
`return_margin_m`), so a joint jittering on the line does not flicker the state.

**Persistence.** "Sustained" means at least `evidence_fraction` of valid frames
within `evidence_window_s`, not N consecutive frames, so single bad frames do not
reset progress. Measure and report the delay these windows add to the first
alert.

### 7.3 What does not trigger

- A wrist, elbow or hand beyond the line. Upper-limb joints are not in
  `d_core` or `legs_over`.
- A patient lying near a rail with no displacement from baseline.
- Reaching toward the edge and returning inside `abort_s` without legs or core
  crossing.
- Another person crossing the line (section 10).

### 7.4 Motion signature (not intent)

When emitting `BED_EXIT_RISK` or `BED_EXIT_CONFIRMED`, attach a `profile`:

| Profile | Signature |
|---|---|
| `progressive` | Passed `EDGE_APPROACH` and `LEGS_OVER` before core crossing; core speed toward edge below `rapid_core_speed` |
| `rapid` | Skipped stages, or core speed toward the edge `>= rapid_core_speed` |

This is a descriptive label for nurse review. It must not be presented as intent.
`rapid` events are also a prompt for the fall branch to be watched, but the fall
machine decides falls independently.

---

## 8. Events and severity

### 8.1 Event types

| Event | When | Severity |
|---|---|---|
| `BED_EXIT_RISK` | Entering `EDGE_APPROACH` (sustained) or `LEGS_OVER`, first time per episode | `BED_EXIT_SEVERITY_BY_RISK[risk]` |
| `BED_EXIT_CONFIRMED` | Entering `EXITED` | `BED_EXIT_SEVERITY_BY_RISK[risk]` |
| `BED_EXIT_ABORTED` | Abort rule fires | 0 (informational) |
| `BED_MONITORING_DEGRADED` | Entering `DEGRADED` while a bed episode is active | 1 (notice); policy for escalation is an open question (section 12) |

`risk = f.bed_risk or "unknown"`, as in the existing code. **[verified]** the
mapping is `none 0, low 1, medium 2, high 3, unknown 2`.

Pass the value through `Event.severity_override`, as the existing `BED_EXIT` does.
**[verified]** field name from `_enter_sitting`.

### 8.2 Evidence payload

Every event carries, rounded as the existing events do:

```python
evidence = {
    "bed_risk": risk,
    "edge": "left",              # which side
    "edge_has_rail": True,
    "d_core_m": -0.04,
    "legs_over": 2,
    "n_core_valid": 2,
    "n_leg_valid": 3,
    "baseline_d_core_m": 0.38,
    "approach_s": 3.2,
    "profile": "progressive",
    "observable_fraction": 0.92, # of the window preceding the event
}
```

### 8.3 Cooldown

Per-track, per-event-type cooldown, held in the new machine only. It must not
read or write `FallStateMachine` cooldown state. `BED_EXIT_RISK` repeats at most
once per `risk_cooldown_s` unless the state fell back to `IN_BED_STABLE` through
`ABORTED`.

### 8.4 De-duplicating the legacy `BED_EXIT`

The legacy event fires from `FallStateMachine._enter_sitting`. After the new
machine ships, a real exit can emit both. Options, in order of preference:

1. **Flag in config:** `detect.legacy_bed_exit: true|false`. Default `true` until
   the new machine is validated on staged clips, then `false`. The legacy path
   stays available for comparison runs.
2. **Merge layer:** drop a legacy `BED_EXIT` if the new machine emitted
   `BED_EXIT_RISK` or `BED_EXIT_CONFIRMED` for the same bed within `dedupe_s`.

Implement option 1 first. It is a small change guarded in `_enter_sitting`
(skip emission when the flag is false) and does not touch fall logic.

---

## 9. Observation quality

The fall machine's gate **[verified]** is `n_valid_kp >= 8` and `mean_conf >= 0.40`
plus floor geometry. The bed machine uses its own:

| Parameter | Start | Meaning |
|---|---|---|
| `min_core_valid` | 1 of 2 | At least one valid mid-hip or mid-shoulder |
| `min_total_valid` | 4 of 6 | Valid joints across core plus lower limb |
| `min_joint_conf` | 0.30 | Per-joint confidence to count as valid |
| `degraded_after_s` | 2.0 | Quorum lost this long before `DEGRADED` |
| `recover_s` | 1.0 | Quorum regained this long before leaving `DEGRADED` |

While `DEGRADED`: do not advance or retreat the state machine, do not count the
time toward "stable" or "exited", emit `BED_MONITORING_DEGRADED` once, and surface
the condition on the dashboard. Report unobservable time in evaluation so a good
false-alert rate cannot hide poor coverage, as the fall evaluator already does.

Expected degradation sources: blankets over the legs, rails confusing the pose
model (phantom or displaced joints), curtains, night lighting. Weight lower-limb
evidence accordingly: require `legs_quorum` valid joints, not a single joint.

---

## 10. Binding the machine to the patient

The legacy code treats any seated track near any "bed" zone as a candidate
**[verified]**. The new machine needs an explicit association:

- A track is **bound to a bed episode** after it has been `supported_by_bed`
  for `bind_s` (start 20 s) at one bed. `FeatureExtractor._supporting_bed`
  already identifies the supporting bed [assumed to return an identifier].
- Only the bound track drives that bed's machine. Other tracks near the bed
  (nurse leaning over a rail, visitor sitting on the edge) do not.
- Unbind when the track disappears for longer than `unbind_s` or the bed is
  reassigned. Call a retention hook equivalent to
  `FallStateMachine.retain_only(live_ids)` so state does not leak across
  tracks. **[verified]** the existing hook.
- **Tracker ID swaps.** The tracker is a greedy IoU placeholder. After an ID
  change, the new track must re-bind rather than inherit; the machine resets and
  reports `DEGRADED` for the gap instead of carrying a stale `EXITED`.
- A bed episode (admission, transfer) carries `risk_level` and any supervision
  permission. The calibration currently attaches risk to the bed zone
  **[verified]**; per-episode context is a later extension (section 12).

---

## 11. Tests

New file `tests/test_bed_exit.py`. Follow the project's style: synthetic feature
sequences with explicit timestamps, no camera or pose model. Add a builder such
as `make_bed_features(t, joints, conf)` that produces `BedFrameFeatures` from
bed-frame joint positions.

### 11.1 Behaviour tests

| Test | Sequence | Expected |
|---|---|---|
| `test_arm_over_rail_does_not_trigger` | Wrist and elbow at `d = -0.3` for 10 s, core and legs inside | Stays `IN_BED_STABLE`, no events |
| `test_sleeping_near_rail_does_not_trigger` | Core at `d = 0.10` for 60 s, no displacement | No events (baseline already near the edge) |
| `test_reach_and_return_aborts_or_ignores` | Core moves to `d = 0.15`, returns within `abort_s`, legs stay inside | No `BED_EXIT_RISK`, or `RISK` then `ABORTED` if approach sustained |
| `test_progressive_exit_confirms` | Approach, legs over, hip crosses, hip beyond margin | `BED_EXIT_RISK` then `BED_EXIT_CONFIRMED`, `profile = progressive` |
| `test_rapid_roll_off_skips_stages` | Core goes from inside to `d = -0.2` in under 1 s | `BED_EXIT_CONFIRMED` with `profile = rapid` |
| `test_foot_end_exit_detected` | Exit across the unrailed foot edge | Detected |
| `test_open_rail_section_detected` | Exit through the part of a side with no rail | Treated as `rail: false` |
| `test_lowered_rail_flag` | `rail: false` set for a side | Mattress edge governs; exit detected |
| `test_degraded_not_safe` | Core and leg joints drop below `min_joint_conf` mid-approach | `DEGRADED`, `BED_MONITORING_DEGRADED`, no `EXITED`, no return to stable |
| `test_degraded_recovery_does_not_inherit_counters` | Gap then recovery | Counters restart |
| `test_other_track_does_not_drive_bed` | Second track crosses the line | Bound patient's machine unchanged |
| `test_id_swap_resets` | Track id changes mid-approach | Machine reset, degraded gap, no stale `EXITED` |
| `test_edge_lock_prevents_flipping` | Patient moves along the bed during approach | Governing edge unchanged until abort or exit |
| `test_hysteresis_no_flicker` | Core jitters around `exit_margin_m` | At most one transition |
| `test_range_independence` | Same exit at 4.0, 6.0 and 7.4 m | Same events, same thresholds (matches the fall suite's property) |

### 11.2 Interaction tests

| Test | Expected |
|---|---|
| `test_fall_branch_unaffected` | Existing fall tests pass unchanged |
| `test_bed_exit_event_does_not_block_fall_alert` | `BED_EXIT_RISK` followed by a fall still emits `FALL_CONFIRMED` |
| `test_legacy_bed_exit_flag` | With the flag off, the legacy `BED_EXIT` is not emitted; on, it is |
| `test_privacy_unchanged` | `tests/test_privacy.py` passes |

### 11.3 Acceptance for v1

- All the above pass.
- The full existing suite passes unchanged (the README states 364 tests).
- On staged clips, arm-only crossings produce zero `BED_EXIT_RISK` events.

---

## 12. Open questions and decisions needed

1. **Guardrail dimensions** (section 5.2): all values TBD in this spec.
2. **Which ward.** The interview notes alternate between Ward 10 and Ward 11.
   Confirm before fixing the deployment protocol.
3. **Critical transition.** Agree with AH, before seeing model output, which
   landmark counts as the exit (for example mid-hip crossing the rail line, or
   initiation of standing). Lead time is measured to it.
4. **Escalation for `CONFIRMED`.** Should `BED_EXIT_CONFIRMED` page at a higher
   level than `BED_EXIT_RISK` for the same bed (for example `min(3, sev + 1)`
   when `sev >= 2`)? This spec uses the same mapping for both.
5. **Degraded policy.** Should prolonged degraded monitoring of a high-risk bed
   escalate to a nurse? Needs a ward fallback procedure.
6. **Assisted-transfer mode.** A nurse-confirmed, time-limited suppression is a
   later feature. Presence of another person in view must not suppress warnings.
7. **Rail state.** Is rail up or down known to the system, or set by a nurse per
   episode?
8. **Bed articulation.** A raised backrest changes torso geometry. v1 uses core
   and leg joints relative to the rail line, which is less sensitive, but
   upper-body-rise cues (v2) need a section-aware bed model.
9. **Extractor access.** Does the extractor give per-joint positions and
   confidences (section 4.2)?
10. **Tracker.** Is ByteTrack replacing the greedy IoU tracker before pilot?
    ID swaps matter more here than for falls.
11. **Clinical landmarks and labelling.** Two annotators on a subset, with
    adjudication; keep uncertainty labels.

---

## 13. Parameters

`BedExitThresholds` is a frozen dataclass, like `FallThresholds`, populated from
the `bed_exit:` config section. Units are metres, seconds and metres per second.

```yaml
bed_exit:
  enabled: true

  # evidence windows
  evidence_window_s: 1.0
  evidence_fraction: 0.70
  baseline_window_s: 30.0

  # approach
  edge_near_m: 0.20
  approach_delta_m: 0.15
  approach_s: 2.0

  # legs
  legs_quorum: 2              # of valid knee/ankle joints beyond the line
  legs_over_s: 1.0

  # crossing and exit
  crossing_margin_m: 0.0
  exit_margin_m: 0.10
  exit_confirm_s: 1.0

  # abort (hysteresis: larger than the exit margin it cancels)
  return_margin_m: 0.05
  abort_s: 5.0

  # profile
  rapid_core_speed: 0.50      # m/s toward the edge

  # observation quality
  min_core_valid: 1
  min_total_valid: 4
  min_joint_conf: 0.30
  degraded_after_s: 2.0
  recover_s: 1.0

  # binding
  bind_s: 20.0
  unbind_s: 5.0

  # hygiene
  risk_cooldown_s: 30.0
  dedupe_s: 10.0

  # joint weights for lower-limb evidence (blanket occlusion)
  joint_weights:
    knee: 0.5
    ankle: 0.5

detect:
  legacy_bed_exit: true       # set false after validation
```

All values are starting points to tune with `ahfd sweep` against labelled clips.
The sweep command must accept `bed_exit.*` keys; check `cli.py` and extend it if
it only recognises `detect.*`.

---

## 14. Evaluation and tuning

**Annotation labels** (added to the existing schema): `edge_approach`,
`legs_over`, `core_crossing`, `exited`, `aborted`, plus observation context
(blanket coverage, lighting, rail up/down, camera view).

**Required negative clips**, because they decide alert burden: arm reaches over
the rail, adjusting a blanket, sitting up to eat, talking to a visitor, nurse
leaning over the rail, rolling to the edge and back, routine repositioning,
patient permitted to self-mobilise.

**Metrics**

| Metric | Report |
|---|---|
| Lead time | `t(critical transition) - t(alert)`, median and lower percentile, plus a count of alerts arriving after the transition |
| Timely recall | Fraction of exits warned before the transition, at a nurse-agreed minimum lead |
| Alert burden | False alerts per occupied bed-hour and per shift |
| Coverage | Unobservable fraction; events during unobservable time counted in the denominator |

Split recordings by person and session. Freeze thresholds before the held-out
evaluation. Compare against the legacy `BED_EXIT` on identical clips, at the same
false-alert budget.

---

## 15. Dashboard

- Per-person chip showing the bed-exit state (`IN_BED_STABLE`, `EDGE_APPROACH`,
  `LEGS_OVER`, `CORE_CROSSING`, `EXITED`, `DEGRADED`) beside the existing posture
  chip.
- Events in the triage queue and event log with the evidence payload.
- `DEGRADED` shown explicitly as "monitoring unavailable", never as "no activity".
- Skeleton-only by default. Nothing here requires RGB.

---

## 16. Privacy constraints (non-negotiable)

- Do not modify `types.py`. It has no image field by design.
- No `imwrite`, `VideoWriter` or binary writes outside `ahfd.debug`.
- Bed episode and risk linkage to an occupied bed are not anonymous; keep them in
  local, access-controlled storage and agree retention with the hospital.
- `tests/test_privacy.py` must pass.

---

## 17. Build order

Each step ends with `pytest` green.

1. `features/bed_frame.py`: bed frame, edge schema, signed distances. Tests for
   geometry (known points, rail extent, foot and head edges).
2. Per-frame evidence and observation gate. Tests for joint groups, quorum and
   the arm-only case.
3. `BedExitStateMachine` states and transitions, using the section 11 tests.
4. Events, severity, evidence payloads, per-event cooldown.
5. Config loading, `bed_exit:` sections, and the `detect.legacy_bed_exit` flag.
6. Pipeline wiring: run both machines per frame, merge events.
7. Dashboard chip and degraded display.
8. Annotation labels and `ahfd sweep` support for `bed_exit.*`; tune on staged
   clips.

### Brief for a coding agent

> Read `docs/BED_EXIT_SPEC.md`, `README.md`, `docs/USAGE.md`,
> `src/ahfd/detect/state_machine.py`, `src/ahfd/features/`, and
> `src/ahfd/geometry/`. Implement the steps in section 17 one at a time. Do not
> change `FallStateMachine` behaviour or thresholds. Do not edit `types.py`.
> Never write images or video. Run `pytest` after each step and show the result.
> Mark anything in the spec tagged [assumed] that turns out to be wrong, and stop
> to ask before working around it.
