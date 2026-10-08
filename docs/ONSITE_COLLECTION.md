# Onsite hospital collection SOP

## Scope and non-negotiable boundary

This procedure is for a **research, shadow-mode study** of early bed-exit
activity. The prototype does not direct care, page staff, replace observation,
or certify that a patient is safe. A staff member follows the hospital's normal
care plan regardless of what the software displays.

The default onsite dataset is derived-only: keypoints, per-joint confidence and
depth-height scalars, bed-relative features, state explanations, observer
markers, and monitoring-health records. It contains no RGB image, dense depth
image, video, audio, or face crop.

Do not collect patient data until the hospital's research/governance owner and
data-protection owner have approved, in writing:

- who may participate and how consent or the applicable research authority is
  documented;
- the exact fields collected, storage location, access list, retention period,
  deletion process, and incident contact;
- whether staff, visitors, or neighbouring beds may enter the field of view;
- whether any live view is permitted; and
- the distinction between research output and the clinical alarm system.

If that approval is not complete, collect only staged activities by consenting
staff or volunteers. Never ask a patient to stage a fall. Staged falls require
the hospital's own safety protocol, spotter, and approved landing equipment.

Raw `.mp4`, `.bag`, `.db3`, exported colour, and classify-view frame caches are
outside this SOP. They contain imagery. Do not use the raw-recording, export, or
frame-cache commands in an occupied ward. A separately approved staged-volunteer
study must use a separate procedure, encrypted hospital storage, an obvious
recording indicator, time-limited capture, and verified deletion.

## Roles for each session

Name these people in the hospital's private study log, not in filenames or AHFD
records:

1. **Clinical/safety lead** — controls the activity and may stop it at any time.
2. **System operator** — runs preflight, starts/stops collection, and watches
   monitoring health and disk state.
3. **Observer/annotator** — enters ground-truth phase and context markers. The
   operator may fill this role only for a simple dry run.
4. **Data custodian** — controls the encrypted destination, consent linkage,
   access log, retention, and deletion evidence.

The software session uses only random codes such as `ses_…`, `sub_…`, and
`site_…`. The consent-to-subject-code link remains with the hospital custodian
in a separate system. Do not put names, initials, MRNs, dates of birth, room or
bed numbers, admission dates, diagnoses, or exact ward names in filenames,
labels, notes, zone names, or source metadata. Track IDs are ephemeral and must
not be linked across sessions.

Keypoints and movement traces are still sensitive health/behavioural data.
Derived-only does not mean public or anonymous. Store sessions only on the
approved encrypted hospital volume. **Never commit or upload onsite data to
GitHub**, including `manifest.json`, JSONL streams, screenshots, logs, or
calibrations that reveal ward layout.

## What one session writes

Each session owns a new, non-overwriting directory:

```text
ses_<random>/
  manifest.json
  derived.jsonl
  telemetry.jsonl
  labels.jsonl
```

`manifest.json` is schema version 1. It starts with status `incomplete`. A clean
stop changes it to `complete`; an operator or system stop changes it to
`aborted`. A process or power loss therefore cannot masquerade as a complete
run. The final manifest contains record counts, byte counts, and SHA-256 hashes
for all three streams, plus hashes of the code revision, config, and calibration.

Every JSONL row uses seconds relative to the session's monotonic clock. The
restricted manifest supplies the UTC session anchor; model-development copies
should retain relative time and remove/coarsen the anchor according to the data
agreement. Rows are flushed as they are written, so an aborted session leaves a
parseable prefix.

### Derived stream

Allowed examples:

- frame index and source-relative timestamp;
- ephemeral track ID and association epoch;
- 2D keypoints, per-joint confidence, and explicit visibility mask;
- per-joint height scalars sampled from depth, with missing-value mask;
- bed support, support confidence, signed edge distance, edge-directed motion,
  posture summaries, and CUSUM/state-machine explanations; and
- shadow predictions and their causal evidence.

Disallowed examples include `Frame` objects, NumPy arrays, bytes, base64 images,
RGB/BGR pixels, dense depth, JPEG/PNG data, or direct identifiers. Convert
approved arrays to ordinary JSON lists of numeric scalars before writing.

### Telemetry stream

At minimum, record:

- session start/end and periodic heartbeat;
- `AVAILABLE`, `DEGRADED`, and `UNAVAILABLE` transitions;
- camera/frame age and effective pipeline FPS;
- pose confidence/visibility and depth-validity coverage summaries;
- track reassociation and model/backend changes;
- calibration/drift checks; and
- camera disconnect, source end, disk error, or operator abort.

`LOW_CONFIDENCE`, `DEPTH_MISSING`, `TRACK_REASSOCIATED`, and `NO_FRAMES` are
missingness evidence. They must never be relabelled as normal behaviour.

### Observer-label stream

Use a controlled vocabulary rather than free-text patient details. Mark the
entry/exit of observable phases:

- `RECLINED`
- `TORSO_RISING`
- `UPRIGHT_IN_BED`
- `SHIFTING_TO_EDGE`
- `EDGE_SITTING`
- `ATTEMPTING_STAND`
- `OUT_OF_BED`

Also mark `RETURN_TO_RECLINE`, `PAUSE`, `FAST_TRANSITION`, `SLIDE`,
`RAIL_CLIMB`, `ASSISTED_TRANSFER`, `STAFF_OCCLUSION`, `BLANKET_OCCLUSION`,
`BED_ARTICULATION`, `TRACK_ERROR`, and `OBSERVER_UNSURE` where applicable. Do
not force an activity through every phase. An observer marker is evidence of
what was seen, not a diagnosis of intent.

## Preparation before the onsite day

1. Freeze the approved protocol, field list, participant scope, retention, and
   abort/incident process. Confirm the encrypted destination is writable and
   has sufficient free space.
2. Pin a reviewed Git commit and dependency environment. Record the commit hash;
   do not update code during a collection session.
3. Run the complete automated test suite. Run `ahfd info` and record the camera
   model, coded serial/device identity, firmware, USB link mode, stream profile,
   pose backend and model hash in the private equipment log.
4. Use an onsite config with raw capture, RGB display, RGB enablement, custom
   source entry, network binding, and frame caching disabled. Ensure
   `AHFD_ALLOW_RAW` is absent. The dashboard, if used, binds to localhost and
   stays skeleton-only.
5. Prepare only fixed-width random hexadecimal codes: `site_` plus 8 hex,
   `sub_` plus 16 hex, `ses_` plus 16 hex, and `cam_` plus 8 hex. Never derive
   a code from initials, MRN, bed/room, or date.
6. Print the one-page operator checklist and the hospital incident contact. Test
   the stop control without a participant present.

## Camera and calibration preflight

Calibration describes a particular camera, resolution, lens position and mount.
It becomes invalid when the camera, mount, stream profile, or bed geometry moves.

For every physical mount:

1. Secure the camera and cable. Use the approved USB-3 port and strain relief;
   no unapproved hub or passive extension.
2. Verify the exact camera code/serial and capture resolution match the approved
   calibration. `ahfd calibrate` stores only a SHA-256 of the factory serial;
   collection refuses another unit, multiple RealSense devices, a non-D435i,
   or a USB-2 link. A different camera or resolution needs a new calibration.
3. Measure lens height. Measure pitch/roll using the D435i IMU or the approved
   floor-plane procedure. Record the result and calibration SHA-256.
4. Author clean, uniquely coded bed polygons for the actual view. Remove test,
   duplicate, dated, cancelled, or overlapping zones. Measure each current bed
   surface height. Any bed articulation or height change aborts the session;
   recalibrate and start a new session before collecting more data.
5. Keep physical geometry separate from the patient's care/risk profile. The
   latter is access-controlled runtime information with an effective time, not
   a value committed in the calibration.
6. Check depth fill/noise at the head, shoulders, hips and edge of every covered
   bed. Check for projector interference if more than one depth camera operates.
   The onsite source enables the emitter and requests maximum laser power; the
   collector must set both controls and read them back from the active D435i.
   Missing controls or a mismatched readback is a preflight failure, not a
   warning to override.
7. With exactly one consenting staff member standing naturally, run
   `ahfd measure-ankle-baseline E:\AHFD_CONFIG\cam_01234567.yaml`. This writes
   one robust two-ankle scalar and resets the verification latch to false. Then
   verify walking/bed association. A calibration or IMU-orientation warning
   blocks collection; it is not an informational banner to ignore.
8. Run a ten-minute empty/staff dry run. Confirm no raw media or frame cache was
   created, all JSONL files grow, timestamps are monotonic, and the health state
   becomes unavailable when the camera is unplugged, covered, or moved.
9. After the final calibration edit, model installation and same-day physical
   check, have the data custodian create an external
   `ahfd.calibration.approval` JSON record containing the site code,
   `research_shadow_collection` purpose, exact onsite-config, calibration and
   reviewed pose-model SHA-256 values, and timezone-qualified `approved_utc`.
   Collection requires this record to be no more than 12 hours old and refuses
   it after any config or calibration byte changes.

Do not use the existing development calibrations without repeating this
preflight at the hospital. A note or filename is not evidence that a mount is
still calibrated.

## Start-of-session checklist

The operator and safety lead both confirm:

- approved participant scope/consent and normal clinical safeguards remain in
  force;
- field of view and neighbouring-bed/visitor handling match the protocol;
- correct random site/participant/session codes, approved code revision, config
  and calibration hashes;
- raw capture and RGB/cache paths are disabled;
- camera identity, resolution, mount, bed height/zones, depth coverage, clock,
  disk space and power are acceptable;
- observer controls and controlled vocabulary work;
- system displays **research shadow mode — not for clinical decisions**; and
- the operator can stop immediately without losing the flushed prefix.

Start the derived collection tool. Inspect the new manifest before activity:
it must say `incomplete`, `derived_only: true`, `imagery_persisted: false`, and
`clinical_decisions_enabled: false`. Start a private hospital study-log entry
that maps the random session code to the approved consent/protocol record.

After completing the physical checklist for the exact mount, use a random
camera code such as `cam_01234567`. For this two-bed deployment, create exactly
two measured, non-overlapping bed zones named `bed_a` and `bed_b`; use the same
pseudonymous scheme for any other zones (for example `floor_1`). Keep every
bed's committed `risk_level: unknown`, confirm the
hashed device identity and measured `ankle_height_baseline_m`, and set
`verified_for_onsite: true` in that calibration.
That latch is deliberately written as `false` by `ahfd calibrate`; software
cannot certify a tape measurement or a human inspection.

The separate approval file is governance-controlled and contains no approver
name. For example (replace the hash and timestamp after the final edit):

```json
{
  "schema": "ahfd.calibration.approval",
  "schema_version": 2,
  "site_id": "site_0123abcd",
  "purpose": "research_shadow_collection",
  "config_sha256": "<64 lowercase hex characters>",
  "calibration_sha256": "<64 lowercase hex characters>",
  "pose_model_sha256": "<64 lowercase hex characters>",
  "approved_utc": "2026-10-08T01:30:00+00:00"
}
```

Changing the config, calibration or pose-model artifact after approval
invalidates this record. Generate all three SHA-256 values with the hospital's
approved integrity tool; do not add a software command that lets an operator
self-approve the file during collection.

Start a derived-only session on the approved encrypted volume:

The custodian must first provision `E:\AHFD_APPROVED\.ahfd-approved-output.json`
with exactly:

```json
{"schema":"ahfd.approved-output","schema_version":1,"site_id":"site_0123abcd","encrypted_storage_attested":true,"purpose":"research_shadow_collection"}
```

This is an attestation and path allowlist, not encryption software; the
custodian must separately verify the volume's encryption and access controls.

```powershell
ahfd collect `
  --out-root E:\AHFD_APPROVED `
  --site-id site_0123abcd `
  --participant-id sub_0123456789abcdef `
  --config configs/onsite_collection.yaml `
  --calibration E:\AHFD_CONFIG\cam_01234567.yaml `
  --approval E:\AHFD_CONFIG\cam_01234567.approval.json
```

The command refuses a non-D435i/USB-3 or ambiguous device, every `--source`
override, and any approved config URI without explicit `depth=1`, `emitter=1`
and `max_laser=1`. It also refuses unverified/dirty calibration geometry,
missing/stale external approval, an
unapproved output volume marker, a calibration/output path inside Git, a
dirty code revision, RGB-enabled config, unmanaged alert sinks, an armed raw
capture switch, enabled early-warning paging, or a missing mount calibration.
It also verifies emitter and maximum-laser readback before opening the session
and records the numeric laser readback in provenance. It checks live intrinsics,
IMU pitch/roll, standing ankle drift, depth coverage, disk space and frame age.
It shows a skeleton-only health banner.
Observer keys are `1` reclined, `2` torso rising, `3` upright in bed, `4`
shifting, `5` edge sitting, `6` attempting stand, `7` out of bed and `0`
unknown. Before any participant skeleton is written, use `[` / `]` to bind the
single visible, consented participant. There is no automatic selection. Track
loss clears the binding and requires an explicit re-bind. A second detected
person aborts a participant session before their body data is written; any
person aborts an empty-room session. Context keys are `x` return,
`p` pause, `f` fast transition, `z` slide, `r` rail climb, `i` assisted transfer,
`c` staff occlusion, `v` blanket occlusion, `t` track error and `u` observer
unsure; press the same key again to end those censoring intervals. `a` records
bed articulation and immediately aborts for recalibration. Press `q` for a
planned protocol completion only. Press `Esc` when a participant, clinician or
operator requests a stop, consent is withdrawn, or a safety concern arises;
that preserves the flushed prefix with status `aborted`, never `complete`.
For a participant run, normal completion also requires at least one persisted
`AVAILABLE` 10 Hz row after a non-`UNKNOWN` phase marker for that same track and
association epoch, plus a valid IMU orientation preflight. Otherwise `q` or an
automatic session limit preserves the prefix as `PROTOCOL_INCOMPLETE` rather
than claiming a usable training session. `Ctrl+C` is also an abort. Keep the private activity
schedule beside the hospital study log, not in free-text software labels.

This first-week build is a single-person collector. Although the controlled
vocabulary reserves `ASSISTED_TRANSFER` and `STAFF_OCCLUSION` for compatible
future datasets, do not attempt a two-person assisted-transfer or staff-at-bed
session: this command will fail closed when a second pose appears. Such scenes
need a separately approved protocol and explicit multi-person consent handling.

## During collection

- The clinical/safety lead controls all movement. Continue normal care and stop
  whenever the participant, clinician, observer, or operator asks.
- The observer marks phase transitions and confounders as they occur. Do not
  invent an unobserved phase and do not silently fill occluded intervals.
- The operator watches frame age, FPS, pose visibility, depth validity,
  calibration status, disk state, and session status. A healthy frame with no
  person is available empty-bed time; a missing/low-confidence frame is not.
- Treat CUSUM and state-machine outputs as shadow research signals. Do not tune
  thresholds during a session and do not tell clinical staff that an alert
  predicts an exit.
- Record changes to rails, bedding, lighting, curtains,
  mobility aids, staff assistance and camera obstruction using controlled
  markers. Bed articulation is an abort/recalibration condition, not an ordinary
  confounder marker. Avoid patient-specific free text.

### Abort conditions

Stop and mark the session `aborted` when any of these occurs:

- participant, clinician, observer, or operator requests a stop;
- a person enters view outside the approved participant scope;
- raw imagery/cache is unexpectedly created or RGB becomes enabled;
- wrong session/participant code, camera, config, calibration, resolution or
  stream is discovered;
- camera disconnect, stale frames, persistent low-confidence/depth loss, track
  instability, calibration drift or mount/bed movement invalidates observation;
- disk write/hash failure, insufficient space, clock anomaly, power/network or
  security incident; or
- the safety lead determines normal care could be affected.

Do not restart into the same session directory. Correct the cause, create a new
random session, and cross-reference the two codes only in the private study log.
Do not delete the aborted derived prefix until the custodian applies the
approved retention/incident decision.

Use `Esc` for every requested/safety stop above. Use `q` only after the planned
protocol block has ended normally and the safety lead confirms completion.

## End-of-session and daily closeout

1. Stop activity, then stop collection cleanly. Verify the manifest says
   `complete` (or the correct controlled abort reason), has end time/duration,
   counters, non-null stream hashes, and no raw-imagery flag.
2. Reconcile observer markers with the private activity schedule. Mark uncertain
   or unavailable intervals; never turn them into negative examples.
3. Copy the untouched session directory to the approved encrypted destination.
   Verify SHA-256 hashes after transfer before removing an approved temporary.
4. Record transfer, access and deletion evidence in the hospital's private log.
   Do not use personal cloud drives, email, chat attachments, removable media,
   or GitHub.
5. Review coverage and health before the next participant. Fix recurring camera,
   pose, depth, tracker or observer problems before collecting more volume.

## Dataset partitioning and reporting

Assign train/tune/test partitions by participant, never by random frames. Where
sample size permits, hold out a collection day, room/mount, or site as well.
Freeze the test partition before threshold tuning or model selection. Repeated
frames from one session, activity or participant must stay in one partition.

Report at least:

- consented session count and monitored hours by scenario;
- available, degraded and unavailable time, with reason breakdown;
- phase-label coverage and observer-uncertain time;
- bed-exit event recall and false alerts per **available monitored hour**;
- first-warning lead time to observed `OUT_OF_BED`, plus median and tail latency;
- missed exits, fast exits, return-to-recline cases and staff/blanket/rail/bed
  articulation confounders;
- performance by participant-held-out and mount/day-held-out split; and
- state-machine/CUSUM/learned-comparator results on exactly the same causal
  inputs and partitions.

Always publish raw counts and denominators. A low false-alert rate obtained by
excluding low-confidence or unavailable time is not a successful result; report
coverage beside accuracy.

After at least two participants have complete, hash-valid sessions, run the
small learned baseline only on the approved encrypted study volume:

```powershell
ahfd compare-bed-exit `
  E:\AHFD_APPROVED\ses_0123456789abcdef `
  E:\AHFD_APPROVED\ses_abcdef0123456789 `
  --group-by subject `
  --out-dir E:\AHFD_APPROVED\models\cmp_001
```

The command uses association-local labels, censors unavailable/unknown future
intervals, holds out whole participants, and reports out-of-fold phase and
5/10/20-second frame metrics. It never enables live inference. Event
coalescing, false alerts per available hour and lead-time analysis remain a
separate required gate. The output directory must be a new child of the same
custodian-approved external volume (or another volume with the same approved
output marker); reports and model files must never be written into Git.
