# Bed-exit detection

Predicting that a patient is **leaving the bed** — early enough for a nurse to
reach them — is the highest-value thing this system can do, because a bed exit
unfolds over tens of seconds while a fall's impact is over in a fraction of one.
This document describes the bed-exit branch that was built alongside the fall
detector: how it uses the pose model, how the geometry works, how it grades its
response, and how it shows up on the dashboard (including drawing the bed zone on
the live RGB feed).

The companion design note is [`BED_EXIT_SPEC.md`](BED_EXIT_SPEC.md); this file is
the as-built record.

## The one guarantee that matters

> **A bed exit is the body *core* leaving the bed. A hand over the rail or a leg
> dangling off the side is not a bed exit, and never raises the exit alarm.**

Everything below exists to keep that promise. The failure mode that gets these
systems switched off is alarming when a patient reaches for a water cup, so the
decision is made on the **mid-hip and mid-shoulder** and nothing else. Legs are a
separate, earlier, low-priority signal; arms and head are recorded as evidence
but can never move the decision.

## How the pose model is used to model a bed exit

The pipeline is unchanged up to the pose stage: `capture → pose (RTMO/RTMPose) →
tracking → One-Euro smoothing`. Every backend emits the same **COCO-17**
keypoints in pixels. The bed-exit branch takes those smoothed keypoints and turns
them into a bed-relative measurement in four steps.

1. **Group the joints by what they can decide.** From the 17 COCO joints:

   | Group | COCO joints | Role in the decision |
   |---|---|---|
   | **Core** | mid-hip (11, 12), mid-shoulder (5, 6) | **Decides the exit** |
   | **Lower limb** | knees (13, 14), ankles (15, 16) | Early, low-priority "legs over" signal |
   | **Upper limb + head** | elbows, wrists, nose, eyes, ears | Evidence only — **never decisive** |

   This grouping is the whole mechanism behind the guarantee. An arm beyond the
   rail contributes nothing to `d_core` or the leg count, so it cannot trigger
   anything.

2. **Project each joint onto the bed's mattress plane.** A single camera ray does
   not fix a 3-D point, but the bed surface is a known horizontal plane at height
   `top_m`. Intersecting each joint's ray with that plane
   (`GroundPlane.pixel_to_plane`) gives the joint's position on the mattress in
   floor metres. Projecting onto the *floor* instead would be wrong: a body
   supported ~0.6 m up projects well past the bed (the same reason the fall
   detector tests beds at their own height, see
   [`features/extractor.py`](../src/ahfd/features/extractor.py)). The residual
   error — a shoulder sits above the mattress, so its ray lands slightly beyond
   the true point — is why the edge margins are centimetres, not millimetres, and
   why the hip (near the mattress) carries more weight than the shoulder.

3. **Measure in the bed's own frame.** Each bed has a coordinate frame derived
   from its calibrated polygon: `x` along the long axis (head→foot), `y` across
   (rail to rail), origin at the centre. A joint's bed-frame position gives a
   **signed distance to each of the four edges** — positive inside, negative
   outside. `d_core` is the signed distance of the core to the governing edge.

4. **Decide over time, not per frame.** The signed distances feed a state machine
   that tracks an approach, a crossing, and a sustained exit, with hysteresis and
   an "unknown is not safe" degraded state. Nothing is decided from a single
   frame, exactly as with falls.

Because every distance is in **metres**, one threshold set covers every camera in
a ward — the same property the rest of the project relies on.

## The bed we modelled

A standard Hill-Rom-style ward bed, from the measurements provided:

- Footprint **2.2 m × 1.05 m**.
- Height-adjustable **0.48 m – 0.90 m** (this is the per-bed `top_m`).
- **Side guardrails run the full length except a 0.48 m gap at the foot**; the
  head and foot ends have no rail.

The rail gap matters: a patient can slide out through the open foot section
without ever crossing a rail, so that stretch of each long side is modelled as
rail-free. Rail presence is metadata on the event (did the core cross a railed
section or an open one?); it does not change the geometric decision.

## Architecture

```
capture → pose → tracking → smoothing ─┬─────────────────────────────┐
                                       │                             │
                              FeatureExtractor              BedFrameExtractor   (new)
                                       │                             │
                              FallStateMachine            BedExitStateMachine   (new)
                              (unchanged)                            │
                                       └──────── events ─────────────┘
                                                   │
                                        alert sinks / dashboard
```

The two machines run **in parallel and share no state**. A quiet bed-exit notice
can never delay or suppress an urgent fall alert, and the fall machine's
behaviour and thresholds are untouched.

### Files

**New**

| File | What it does |
|---|---|
| [`features/bed_frame.py`](../src/ahfd/features/bed_frame.py) | `BedFrame` (bed coordinate frame + signed edge distances + rail gap), `BedFrameExtractor` (joints → `BedObservation`), joint groups, default Hill-Rom rails. Pure geometry, no memory. |
| [`detect/bed_exit.py`](../src/ahfd/detect/bed_exit.py) | `BedExitThresholds`, `BedExitState`, `BedExitStateMachine` — binding, edge-locking, the approach/legs/crossing/exit/abort transitions, grading, degraded handling. |
| [`tests/test_bed_exit.py`](../tests/test_bed_exit.py) | Synthetic sequences driven through the *real* geometry (poses placed at known bed-frame positions and inverse-projected to pixels). |
| [`tests/test_bed_exit_dashboard.py`](../tests/test_bed_exit_dashboard.py) | Severity-based triage, and the RGB zone picker's back-projection + persistence. |

**Changed**

| File | Change |
|---|---|
| [`geometry/ground.py`](../src/ahfd/geometry/ground.py) | `world_to_pixel` — the inverse projection used to draw a fixed bed outline on the frame. |
| [`geometry/zones.py`](../src/ahfd/geometry/zones.py) | `BedEdge` + `edges`/`foot_at_far_end` on `Zone`; `ZoneMap.beds()` / `by_name()`. |
| [`detect/events.py`](../src/ahfd/detect/events.py) | New event types + severities (below). |
| [`detect/state_machine.py`](../src/ahfd/detect/state_machine.py) | Legacy `BED_EXIT` now behind `legacy_bed_exit` (default on, so old tests are unchanged). |
| [`config.py`](../src/ahfd/config.py) | `BedExitConfig` (`bed_exit:` section) + `detect.legacy_bed_exit`. |
| [`cli.py`](../src/ahfd/cli.py) | `_build_detection` builds the bed machine; `run` and `replay` drive both machines. |
| [`dashboard/runner.py`](../src/ahfd/dashboard/runner.py) | Runs the bed machine, publishes its events, adds a `bed_state` chip, draws the fixed bed overlay. |
| [`dashboard/state.py`](../src/ahfd/dashboard/state.py) | Triage is now severity-driven (a high-risk exit pages); new counters. |
| [`dashboard/controller.py`](../src/ahfd/dashboard/controller.py) | `add_bed_zone` — back-project clicked corners and persist. |
| [`dashboard/server.py`](../src/ahfd/dashboard/server.py) | `POST /api/bed_zone`. |
| [`dashboard/html.py`](../src/ahfd/dashboard/html.py) | Bed-state chips, the "Define bed zone" picker, severity/metric wiring. |
| [`viz/skeleton_render.py`](../src/ahfd/viz/skeleton_render.py), [`viz/overlay.py`](../src/ahfd/viz/overlay.py) | `draw_bed_zones`, threaded through both renderers. |
| `calib/example_ward6.yaml`, `configs/dashboard.yaml`, `configs/detect_dev.yaml` | Example rails; `bed_exit:` sections; legacy precursor off. |

## The state machine

```
NOT_IN_BED → IN_BED_STABLE → EDGE_APPROACH → LEGS_OVER → CORE_CROSSING → EXITED
                   ↑  ↓ (abort)                                    
                DEGRADED  (observation lost; "unknown is not safe")
```

- **Binding.** A track becomes *bound* to a bed after its core sits in that bed's
  footprint for `bind_s`. Only the bound track drives that bed's machine, so a
  visitor leaning over the rail or a patient in the next bed does nothing. The
  bound bed is remembered by name, which is what lets the machine keep watching
  *after* the body leaves the footprint — the moment the exit happens.
- **Baseline.** While stable, a rolling median of the core's distance to the
  nearest edge is kept. Approach is measured as movement *relative to that
  baseline*, so a patient who simply sleeps near a rail is not "approaching".
- **Edge lock.** The governing edge is locked when the approach begins, so a
  patient shifting along the bed does not flip the decision between edges.
- **Hysteresis.** Exit and return use different margins, so a joint jittering on
  the line does not flicker the state.
- **Sustained, not consecutive.** A transition needs the condition to hold over a
  fraction of the *observable* frames in a short window, so one bad frame neither
  resets progress nor fakes it.
- **Degraded.** If the core/leg quorum is lost (blankets, curtains, night
  lighting, the rail confusing the pose model) the machine goes `DEGRADED`, emits
  `BED_MONITORING_DEGRADED` once, and stops advancing. It never reads missing
  evidence as "still in bed".

## Events and grading

Urgency is graded by the **bed's `risk_level`** (set once per admission from the
Morse/Hendrich assessment), via `BED_EXIT_SEVERITY_BY_RISK`
(`none 0, low 1, medium 2, high 3, unknown 2`). Severity 3+ pages a nurse (joins
the triage queue); below that it is a dashboard status / log entry.

| Event | Fires when | Severity |
|---|---|---|
| `BED_EXIT_LIMB` | A leg crosses the rail line (core still inside) | **low** — `min(1, risk)`. The "just in case" heads-up. |
| `BED_EXIT_RISK` | Core approaching, or beginning to cross | `risk` level of the bed |
| `BED_EXIT_CONFIRMED` | Core beyond the line, sustained | **one notch above risk** — `min(3, risk+1)` (and still 0 for a bed cleared to self-mobilise) |
| `BED_EXIT_ABORTED` | Core came back inside — a reach-and-return | 0 (informational) |
| `BED_MONITORING_DEGRADED` | Can't observe the bed well enough | 1 (notice) |

Two decisions you asked for are baked in:

- **Confirmed outranks risk.** For a `medium` bed, risk warns at 2 and a
  confirmed exit pages at 3. (For a `high` bed both are already at the ceiling of
  3 — a high-risk patient merely reaching the edge is itself worth a page.)
- **A limb crossing is its own low signal**, never mistaken for an exit.

Every event carries an evidence payload a nurse can read: which edge, whether it
was railed, `d_core`, the leg count, the baseline, the observable fraction, and a
descriptive `profile` (`progressive` vs `rapid` — a label for review, **not** a
claim about intent).

## Dashboard

- **A second chip** beside the posture chip shows the bed-exit state
  (`IN_BED STABLE → EDGE APPROACH → LEGS OVER → CORE CROSSING → EXITED`, or
  `DEGRADED`), colour-graded calm→amber→red.
- **Triage and the event log** show bed-exit events with their evidence; a
  high-risk confirmed exit rings the alert sound and sits in the queue with an
  Acknowledge button, exactly like a fall.
- **The bed is drawn on the feed, fixed** (see below), and its outline turns
  amber as the core approaches and red as it crosses — so what fired the alert is
  visible on screen. This works in both the skeleton-only and RGB views; nothing
  here needs RGB, and no imagery is ever stored.

### Drawing the bed zone on the RGB frame (how, and why it stays fixed)

The bed is authored once as a **floor polygon in metres**. To draw it on the
video, each corner (at mattress height `top_m`) is projected back to a pixel with
the new `GroundPlane.world_to_pixel`, which is the exact inverse of the
ray-casting the detector already uses:

```
world point (X, Y, Z)  --world_to_pixel-->  pixel (u, v)
```

Because the camera is **calibrated and physically fixed**, the same world corner
lands on the same pixel every frame, so the outline is locked to the bed and does
not track or drift with people moving through it. The drawing
(`viz.skeleton_render.draw_bed_zones`) paints the footprint, the rail segments
(thicker, and visibly absent across the foot gap), and the bed name + risk. It is
pure geometry on the canvas — it never reads or writes pixels of the person.

### Choosing a bed zone on the RGB feed

The dashboard has a **"Define bed zone"** button. Click it, then click the four
bed corners on the video (head-first: head-left, head-right, foot-right,
foot-left), and give the bed a name, height and risk level. The browser sends the
four clicks as **source-resolution pixels**; the server
(`DashboardController.add_bed_zone`) intersects each click's ray with the mattress
plane at the given height — the same back-projection `ahfd calibrate-zones` uses —
to recover the corners in floor metres, attaches the default Hill-Rom rails,
writes the zone into the camera's calibration file, and respawns the pipeline so
detection picks it up immediately. A bed drawn on the page and one authored at the
CLI are therefore identical, and the zone persists across restarts.

> The picker needs a calibrated camera (height + tilt), because without the
> calibration there is no way to turn a click into metres. An uncalibrated camera
> is refused with that explanation rather than guessing.

## Configuration vs calibration

The project's split holds: **metric thresholds describe a ward, physical facts
describe a bed.**

- **`configs/*.yaml` → `bed_exit:`** — the decision thresholds (margins, windows,
  quorums, cooldowns). One set per ward. Tunable, untuned starting points.
- **`calib/*.yaml` → `zones:`** — each bed's polygon, `top_m`, `risk_level`, and
  `edges` (rails). Per bed, per mount. The zone picker writes here.

The legacy "seated near a bed" precursor is kept behind `detect.legacy_bed_exit`
(default `true` for backward compatibility) and turned **off** in the shipped
dashboard/dev configs, so one real exit yields one set of events.

## Testing

`tests/test_bed_exit.py` builds synthetic poses by placing COCO joints at chosen
bed-frame positions and inverse-projecting them to pixels, so projection,
bed-frame maths and the state machine are all exercised together. The negatives
are the point:

- `test_arm_over_rail_does_not_trigger` — an arm far past the rail, silent.
- `test_leg_over_emits_limb_not_exit` — a leg over is a low signal, never a
  confirmed exit.
- `test_sleeping_near_rail_does_not_trigger` — lying by a rail is not an approach.
- plus progressive/rapid exits, abort, degraded, range-independence by
  construction, and other-track isolation.

The full suite (fall logic, privacy AST check, dashboard) passes unchanged.

## Limitations and follow-ups

- **Thresholds are untuned.** The numbers are starting points for
  `ahfd sweep` against staged clips; bed-exit annotation labels and lead-time
  metrics (spec §14) are the next piece.
- **Replay** runs the bed machine for inspection, but `ahfd sweep` still tunes
  `detect.*` only; a bed-exit sweep needs the annotation/metric work above.
- **Rail state** (up/down per episode) is modelled in the schema but not yet fed
  from a nurse action; lowering a rail is currently a calibration edit.
- **Bed articulation** (a raised backrest) changes torso geometry; v1 leans on the
  hip and legs, which is less sensitive to it. A section-aware bed model is a v2.
- **Tracker ID swaps** reset the machine to a degraded gap rather than inheriting
  a stale state; a stronger tracker (ByteTrack) would reduce these.
