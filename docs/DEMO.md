# Live demo runbook (living-lab ward, 2 beds)

Every command to run, in order. **Everything runs on the Jetson itself**, with a monitor, keyboard and
mouse plugged in, and the pages open in the Jetson's own web browser. No laptop or SSH is needed.
(Running it remotely from a laptop is in the appendix at the end.)

Open a terminal on the Jetson desktop (Ctrl+Alt+T). The repo is `~/projects/ah-fall-detection`; the
venv activates by itself when you `cd` into it.

What the demo shows: falls and bed exits on one screen, from skeletons only. Alerts come from the
rules tested in the locked test (docs/BENCHMARK.md section 7). The trained posture and 7-phase models
are shown next to each person for comparison; they never raise alerts.

---

## 0. Things to bring

- **A monitor (HDMI or DisplayPort), USB keyboard and mouse** for the Jetson. For the audience, a
  large screen or TV connected to the Jetson works best.
- Power for the Jetson and the screen.
- Tape measure, and a phone with an inclinometer app (for the camera tilt if the fit looks wrong).
- Crash mat for the staged fall.
- Optional: a laptop, only if you want to watch remotely (appendix).
- The T9 drive is **not** needed for the demo. Only the engineer page's "Review videos" link uses it.

---

## 1. The day before: set up and calibrate (about 20 min)

### 1.1 Mount and connect the cameras

Since 29 Oct the **D435i looks at the beds from the side** and the **D435f looks along them**. Keep
that: each camera's 7-phase model matches that view. If you swap them, use the other `--view` below.

```bash
lsusb | grep -i realsense
```

Expect one `Intel Corp. RealSense D435` line per camera. If a camera is missing, re-plug it into a
blue USB 3 port.

### 1.2 Stop the screen from going blank

The Jetson blanks or locks its screen after a few minutes, which would hide the demo. In
**Settings > Power** set **Blank Screen** to **Never**, and in **Settings > Privacy > Screen Lock**
turn automatic lock off. Do this once.

### 1.3 Get the latest code

```bash
cd ~/projects/ah-fall-detection
git pull
git status --short          # only configs/raw_capture.yaml should be listed (untracked)
```

Make sure no dashboard is still running (it holds the camera):

```bash
ss -ltn | grep ':8000 '     # no output = nothing running
```

If something is listed, go to its terminal and press Ctrl+C.

### 1.4 Measure

Write these down:

- Lens height above the floor, in metres (tape from the floor to the camera's lens).
- Mattress length, in metres.
- Mattress top height above the floor, in metres (usually about 0.60).

### 1.5 Calibrate each camera

Run this on the Jetson's own screen: a window opens for clicking the beds.

D435f (along the beds, no tilt sensor: its tilt is fitted from the bed length):

```bash
python scripts/quick_calibrate.py --camera d435f --view along --height 2.60 --bed-length 2.10
```

D435i (side view, the tilt comes from its IMU):

```bash
python scripts/quick_calibrate.py --camera d435i --view side --height 2.60
```

Replace 2.60 and 2.10 with your measurements. What happens:

1. It samples the camera for about 2 s and prints the depth floor fit. Keep the floor in view and
   people out of the way.
   - `floor fit rejected` or `does not seem to see the room` means the camera is covered or facing a
     wall. Fix the view, or pass a measured tilt with `--pitch 25` (phone inclinometer on the camera).
2. A window opens with the live view. Press **SPACE** to freeze it.
3. Click the **4 mattress corners** of the first bed, press **f**. In the terminal answer:
   name `nearest`, kind `bed`, top_m your mattress height, risk_level press Enter.
4. Do the same for the second bed, name `middle`. Then press **q** to close.
5. It prints the fitted tilt and the bed lengths (they should be close to your measured length),
   writes `calib/<camera>_<date>.yaml`, and points `configs/dashboard_jetson.yaml` at it.

Two non-overlapping beds are required: with any other number the bed levels (Low/Medium/High) stay
off and no bed-exit alert appears.

### 1.6 Check it live (20 s per camera)

```bash
python scripts/quick_calibrate.py --camera d435f --check
python scripts/quick_calibrate.py --camera d435i --check
```

During the 20 s: stand beside a bed for 10 s, then lie on it. Each second it prints a line like
`person 3: torso 1.12 m, UPRIGHT, bed -, UNKNOWN`.

| You are | Should read |
|---|---|
| Standing | torso 0.9-1.3 m, `UPRIGHT` |
| Lying on the nearest bed | `bed nearest`, `IN_BED`, phase `RECLINED` |

If standing reads above 1.4 m or below 0.7 m, the height or tilt is off: run 1.5 again with
`--pitch` from the phone inclinometer.

### 1.7 Save the calibration

```bash
git add calib/ configs/dashboard_jetson.yaml
git commit -m "Demo calibration $(date +%d\ %b)"
git push
```

### 1.8 Full rehearsal

Run sections 2 and 3 once end to end, with a teammate as the patient.

---

## 1.9 Bed Monitor app (no browser, no "localhost")

Once per machine, create the desktop icon:

```bash
.venv/bin/python scripts/showcase_app.py --install-shortcut      # Jetson
.venv\Scripts\python.exe scripts\showcase_app.py --install-shortcut   # laptop (PowerShell)
```

Then double-click **Bed Monitor**: it starts the dashboard, waits for the camera, and opens the
showcase full screen in its own window (no address bar or tabs). **Alt+F4 closes it and stops the
camera.** Right-click the icon (Jetson) or use the second icon (laptop) for the nurse board. From
a terminal: `python scripts/showcase_app.py --page ward --window`. If the camera is unplugged it
says so instead of opening. Log: `data/logs/showcase_app.log` (dashboard output in
`data/logs/dashboard.log`).

## 1.10 Showcase mode (open day, one D435i)

```bash
ahfd dashboard --config configs/showcase.yaml          # Jetson (CUDA)
ahfd dashboard --config configs/showcase_laptop.yaml   # laptop (OpenVINO)
```

Open **http://localhost:8000/showcase** on the audience screen (F11). It shows the camera and the
**height map** side by side (every pixel coloured by its height above the floor: floor blue,
mattress green, a standing head red, with each person's head and hip height), each bed's progress
through the seven bed-exit phases (rules, with the learned model's call as a blue tag), a live
head/hip height chart, the pipeline with live values, and every alert as it happens. The nurse board
(/ward) can switch its picture between Camera and Height map.

The phase model is `models/phase_combined.joblib` (git-ignored): one random forest on every usable
labelled clip, both cameras and both views, with a confidence threshold so a calm patient is
rarely shown as getting up. Rebuild it with `python scripts/train_combined_phase.py`
(report next to it as `.json`). Display only: alerts still come from the rules.

The height map costs about 40 ms per refresh (5 Hz) while a page shows it, so the live rate drops
slightly; it is only drawn while someone is looking at it, and needs the camera image on.

## 2. Demo day: start everything (5 min)

### 2.1 Start the dashboard

In a terminal on the Jetson:

```bash
cd ~/projects/ah-fall-detection
ahfd dashboard
```

It starts on the camera you calibrated last and prints the page addresses (`http://127.0.0.1:8000`,
which is the same as `localhost:8000`). Leave this terminal open (minimise it). To use the other
camera, pick it in the camera dropdown on the engineer page.

If you pulled new code, restart the dashboard (Ctrl+C, then `ahfd dashboard`): a running dashboard
keeps the code it started with.

### 2.2 Open the pages (Jetson browser)

Open the web browser on the Jetson (Chromium or Firefox from the dock or the applications menu) and go to:

- **Nurse page:** http://localhost:8000/ward (beds, alerts, last 24 hours). Show this one to the audience.
- **Engineer page:** http://localhost:8000/ (camera picker, live view, every number). Keep it on a second tab.

Press **F11** for full screen on the audience screen; F11 again to leave it.

Keep the browser to these two tabs: the browser shares the Jetson's processor with the detector, and
extra tabs (or the depth preview) slow the live view.

**Camera image or stick figures:** `configs/dashboard_jetson.yaml` has `show_rgb: true`, so both pages
show the real camera image with the skeleton on top. The engineer page's **RGB** button switches to
stick figures only (the nurse page follows within a second). Show the camera image only with your
volunteers' consent; say that a real ward would run on stick figures.

### 2.3 Check before the audience arrives

On the engineer page:

1. The live view moves and the FPS reads about 5.
2. Both beds are listed. Set each bed's level to **Medium** (bed-exit alerts) or **High** (adds the
   early warning). On **Low**, no bed-exit alert appears.
3. Lie on a bed for a moment: the person shows the bed name, `IN_BED`, and the model's
   `in_bed / RECLINED`.
4. Acknowledge any alerts left over from the rehearsal, so the ward page starts clear.
5. On the nurse page, **Review alerts** lists the clips saved in the last 24 hours; clear old rehearsal
   clips first if you want an empty list (`mkdir -p data/alert_clips_old && mv data/alert_clips/* data/alert_clips_old/`).

---

## 3. The demo, step by step

| Step | Patient does | Expect on screen |
|---|---|---|
| 1 | Lie on the nearest bed for 10 s | Bed occupied; rules `IN_BED`, model `in_bed / RECLINED` |
| 2 | Sit up, slide to the edge, sit there for a few seconds | `GETTING UP`, then `EDGE_SITTING` (no warning yet: sitting is not enough) |
| 2b | (High bed) Start to stand up from the edge | Early warning "Patient may be getting up", about 5-9 s before fully up |
| 3 | Stand up, walk away from the bed, stay up 5 s | **BED EXIT** alert about 1-4 s after standing |
| 4 | (Presenter) acknowledge it on the ward page | It moves to the shift log |
| 5 | Kneel on the mat, as if picking something up | Floor alert. Say beforehand: any body near the floor alerts, by design |
| 6 | Staged fall onto the crash mat, stay down | **FALL ALERT** from about 0.5 s before to 0.3 s after landing; escalates while down |
| 7 | Get up | The open alert is marked "person is up again" |

Tips:

- Do one thing at a time and pause 5 s between steps.
- Keep anyone else out of the camera's view of the beds: one person per bed.
- Stand up fully for step 3. A 1-2 s half-stand is not reported as an exit.

---

## 4. After the demo

Stop the dashboard: press **Ctrl+C** in its terminal (2.1).

Each alert's clip (stick figures, 10 s before to 15 s after) is in **Review alerts** on the nurse
page: click one to play it there. This works while the dashboard runs. With the dashboard stopped,
play them with:

```bash
ahfd serve-review data/alert_clips
```

and open http://localhost:8001 in the Jetson's browser. Ctrl+C in that terminal stops it.

The demo's alerts also count in the ward page's "Last 24 hours" totals until tomorrow.

---

## 5. If something goes wrong

| Problem | Cause | Fix |
|---|---|---|
| Browser says "refused to connect" | Dashboard not running, or it stopped with an error | Look at the terminal from 2.1; start `ahfd dashboard` again |
| Screen went black or locked mid-demo | Power settings | Move the mouse; set Blank Screen to Never (1.2) |
| New features (labels, Review alerts) missing | Dashboard started before the latest `git pull` | Ctrl+C and start `ahfd dashboard` again; refresh with Ctrl+F5 |
| No picture on the monitor | Monitor plugged in after boot, or wrong input | Plug the monitor in before powering the Jetson; check the screen's input source |
| Page loads but no camera feed | Dashboard opened a camera that is not plugged in | Pick the connected camera in the dropdown, or check `lsusb` (1.1) |
| `no frames` / camera busy error | Another dashboard or script holds the camera | `ss -ltn \| grep ':8000 '`, stop that process, start again |
| Bed-exit never alerts | Bed level on Low, or not exactly two bed outlines | Set Medium/High (2.3); redo 1.5 if beds are missing |
| Lying on the bed shows `ON_GROUND` | Height or tilt off, or bed outline in the wrong place | Redo 1.5, check with 1.6 |
| Standing reads far over 1.4 m | Tilt wrong | Redo 1.5 with a measured `--pitch` |
| Alerts while just walking around | Possible-fall alert on fast sit-downs/bends | Expected sometimes; explain it as the known false-alarm source |

---

## 6. Known limits, to mention if asked

- **Speed:** about 5 FPS with depth (measured with nobody in view), so movement on screen looks
  jerky. Closing the depth preview tab helps a little.
- **Bending and kneeling raise a fall alert.** Deliberate: a high-risk patient near the floor is worth checking.
- **One person per bed.** A second person at the bed can be taken for the patient.
- **ID changes:** if tracking breaks as the patient leaves, a new track within 4 s and 1 m takes over
  the bed history (added 3 Nov). It added no false exits on the development or test clips, and is
  not yet tested on new recordings.
- **Brief stand-ups** (1-2 s, then sitting back) are not reported as exits.
- **Blankets** are untested.
- Results so far are from staged falls by one healthy student and bed exits by a few volunteers,
  not patients.

---

## 7. Measuring false alarms per 24 hours (target: 5 or fewer)

The recorded clips are too short and too busy to measure this (about 10 min per camera, mostly exits
and falls). Measure it live instead:

1. Calibrate (section 1) and start the dashboard (section 2.1). Set the beds' levels as they would be
   used (High turns the early warning on). Note the start time.
2. Leave it running for 24 h (or several nights) with normal activity: people resting, reading,
   getting up to walk around, visitors. Keep the Jetson powered and the T9 not needed.
3. Count the alerts:

   ```bash
   python scripts/alarm_audit.py --since "2026-11-04 08:00" --hours 24
   ```

   It groups alerts into alarms (same person, less than 10 s apart), prints them per type and per
   24 h, and writes `data/alarm_review.csv`.
4. Watch each alarm's clip (nurse page **Review alerts**, or `ahfd serve-review data/alert_clips`, section 4) and type `real`, `risky`
   or `false` in the csv's verdict column. Run the script again: it prints false alarms per 24 h.

If the false alarms come mainly from one alert type, that type is what to tune next, on this new
footage, keeping a later recording as the test.

---

## Appendix: running it remotely from a laptop (optional)

Use this only if nobody is at the Jetson, for example to check it from home over Tailscale. The
dashboard and its pages only listen on the Jetson itself, so the laptop reaches them through an SSH
tunnel.

1. Start the dashboard on the Jetson (section 2.1). From the laptop, you can first SSH in:
   `ssh anormalm@100.125.187.13`, then `cd ~/projects/ah-fall-detection && ahfd dashboard`.
2. Open a **new PowerShell window on the laptop** (not inside the SSH session; the prompt must read
   `PS C:\...`):

   ```powershell
   ssh -N -L 9000:127.0.0.1:8000 -L 9001:127.0.0.1:8001 anormalm@100.125.187.13
   ```

   It shows nothing and keeps running; that is correct. Leave it open.
3. In the laptop browser: nurse page http://localhost:9000/ward, engineer page http://localhost:9000/,
   alert clips (if `ahfd serve-review` is running) http://localhost:9001.
4. Close the PowerShell window when done.

Notes:
- The Jetson's Tailscale address is 100.125.187.13 (check with `tailscale ip -4` on the Jetson).
- Over Tailscale the connection is relayed and slow (about 4-7 Mbit/s), so the live view may lag.
  The ward page is light and stays responsive.
- For a remote desktop of the Jetson (for example to calibrate), connect Windows Remote Desktop
  (mstsc) to 100.125.187.13 and log in with Xorg.

