# Recording day handover: 29 Oct 2026

Everything the person running today's session needs, start to finish.
**Staff (2 volunteers): bed clips h1-h3. Students: fall clips f1-f6.**
Raw video stays on the Jetson and the T9 only. Never copy `.db3` files to a laptop, phone or cloud drive.

---

## 0. Before anyone arrives

### 0.1 Where to run the commands
`record-dual` opens a **live preview window**, so run it on the **Jetson's own screen** or through
**Remote Desktop** (Windows: `mstsc` -> `100.125.187.13`, log in with **Xorg**, your Jetson username and password).
A plain SSH window cannot show the preview and the recording will fail to start.

Open a terminal there and go to the project:
```
cd ~/projects/ah-fall-detection
```

### 0.2 Plug in and mount the T9
1. Plug the T9 into a **USB-A** port on the Jetson.
2. Check whether it is already mounted (**never mount it twice**):
   ```
   mountpoint /media/anormalm/T9
   ```
   - "is a mountpoint" -> it is already mounted, skip step 3.
   - "is not a mountpoint" -> continue.
3. Find the drive and mount it:
   ```
   lsblk -o NAME,SIZE,FSTYPE,LABEL | grep -i T9      # normally sda1, 1.8T, exfat, T9
   sudo mount.exfat-fuse -o uid=1001,gid=1002,umask=022 /dev/sda1 /media/anormalm/T9
   ```
   (Use the name `lsblk` shows if it is not `sda1`.)
4. Check free space on both disks:
   ```
   df -h ~ /media/anormalm/T9
   ```
   Internal disk: about 260 GB free = about 25 min of both-camera recording at 15 fps. **Stop and copy
   to the T9 if the internal disk drops below 50 GB.**

> **Record to the internal disk, never straight to the T9.** Writing to the T9 from the Jetson only
> manages 6-11 fps (what went wrong on 27 Oct). Copy to the T9 after each person (step 4).

### 0.3 Cameras
1. Both RealSense cameras in **USB-A** ports (the developer kit's USB-C port does not run them).
2. Check that both are seen:
   ```
   ahfd info
   ```
   You should see the D435i (serial ending **2607**) and the D435f (serial ending **0025**).
3. **Do not move the cameras or the beds once you start.** If anything is bumped, redo step 0.5
   and write it down in the log (section 5).

### 0.4 Folders on the internal disk
```
mkdir -p ~/recordings/d435i ~/recordings/d435f
```

### 0.5 Setup check (once at the start, and again if a camera moves)
Two short clips, both cameras. Change only the clip name each time.
```
AHFD_ALLOW_RAW=1 ahfd record-dual ~/recordings/d435i/setup_empty_01.db3 ~/recordings/d435f/setup_empty_01.db3 \
  --serial-a 346122072607 --serial-b 406122070025 --fps 15 --seconds 10 \
  --config configs/raw_capture.yaml --i-understand-raw-capture
```
- `setup_empty_01`: **empty room**, nobody in view, beds made, 10 s.
- `setup_stand_01`: **one person of known height standing still** in front of the nearest bed, 10 s.
  Write their height in the log.

After each recording the terminal prints the saved frame rate, for example
`s1_01.db3: 150 colour frames in 10.0 s = 15.0 fps`.
**If it says less than 12 fps or shows a yellow WARNING, stop and check** (internal disk? USB-A ports?).

---

## 1. How to record a clip (same command every time)

```
AHFD_ALLOW_RAW=1 ahfd record-dual ~/recordings/d435i/<NAME>.db3 ~/recordings/d435f/<NAME>.db3 \
  --serial-a 346122072607 --serial-b 406122070025 --fps 15 --seconds <LENGTH> \
  --config configs/raw_capture.yaml --i-understand-raw-capture
```
- Replace `<NAME>` (same in both places) and `<LENGTH>` (seconds; the table gives it). Press **q** in
  the preview window to stop early.
- **Names:** `<clip>_<person>_<bed>`, for example `h1_02_nearest`, `f1_04`. People are numbered: staff
  `02` and `03` (staff `01` was recorded on 27 Oct); students start at `04` (students 01-03 were 27 Oct).
- If you reuse a name by mistake, the recorder **refuses** and suggests the next free name. Nothing
  is overwritten.
- Read each cue at the time shown; the volunteer moves naturally. Times are targets; the labels are
  corrected later.

---

## 2. Staff: bed clips (2 volunteers, both beds)

Each staff member records **h1, h2 and h3 on the nearest bed, then again on the middle bed**
(6 clips each, about 15 min with resets). If time is short, do one bed per person (staff 02 nearest,
staff 03 middle). **No falls for staff.**

### h1: Sit-ups (75 s), e.g. `h1_02_nearest`
| Time (s) | Cue |
|---|---|
| 0-10 | "lie still" |
| 10-25 | "sit up" (normal pace), stay sitting about 7 s, "lie back" |
| 25-40 | "sit up slowly", pause, "lie back" |
| 40-55 | "roll onto your side first, then push up", "lie back" |
| 55-75 | "sit up using the rail", stay, "lie back", lie still to the end |

### h2: Full cycle, lie -> out of bed -> back to lying (100 s), e.g. `h2_02_nearest`
| Time (s) | Cue |
|---|---|
| 0-8 | "lie still" |
| 8-14 | "sit up", stay sitting |
| 14-20 | "move to the edge of the bed" |
| 20-28 | "feet on the floor, sit on the edge" |
| 28-34 | "start to stand... sit back down" (one stand attempt) |
| 34-40 | "stand up" |
| 40-58 | "walk 2-3 m away and stop" |
| 58-70 | "come back, sit on the edge" |
| 70-78 | "legs back into the bed" |
| 78-100 | "lie down", lie still to the end |

### h3: Free (120 s), e.g. `h3_02_nearest`
"For two minutes, do whatever you would normally do in a hospital bed, at your own pace." For
example: use the phone, roll over, reach for the table, sit up, sit on the edge, get up and walk to
the chair, come back. **No falls, and nothing unsafe.** Note in the log what they actually did.

---

## 3. Students: falls (approval required)

**Safety first:** consenting students only, **crash mats** where the falls land, a **spotter** next to
the mat, a **supervisor** present. Anyone may stop at any time. The mat must be in view of **both
cameras**. Each student records f1-f6 once (about 6 min recorded, about 20 min with resets).

**Before recording, choose one student as the locked test** and write it in the log. Nobody looks
at that student's clips until everything is trained.

Lying still for the full "down" time matters: the system only confirms a fall after **8 s** down and
still, and the "person down" (slow slump) alarm needs about **20 s**.

| Clip | Length | 0-12 s | Event | Down (lie still) | End |
|---|---|---|---|---|---|
| `f1` Fall during a stand attempt at the bed edge | 60 s | sit at the edge | 12-16: start to stand, lose balance, land on the mat | 16-40 | 40-60 get up, sit on the bed |
| `f2` Trip while walking near the bed | 60 s | stand, walk | 12-14: trip forward onto the mat | 14-38 | 38-60 get up |
| `f3` Slow slump (no impact) from sitting on the edge | 60 s | sit at the edge | 12-20: slide down slowly to the floor | 20-50 (30 s) | 50-60 get up |
| `f4` Near miss | 30 s | sit at the edge | 12-16: fall onto the mat | 16-20 only | 20-30 get up quickly |
| `neg_f5` Look-alike: lie down on the floor on purpose | 30 s | stand | 8-14: lie down slowly on the mat | 14-24 | 24-30 get up |
| `neg_f6` Look-alike: pick up / kneel / crouch | 40 s | stand | 5-35: pick an object up from the floor twice, kneel 5 s, crouch 5 s | - | stand |

Names: `f1_04`, `f2_04`, ..., `neg_f5_04`, `neg_f6_04` (student 04); the next student is `_05`, and so on.
**Keep the `neg_` prefix** on f5 and f6: it marks them as "not a fall" for training.

**Not recorded: rolling or sliding out of bed while lying.** We assume the bed rails are up and
prevent it.

---

## 4. After each person: copy to the T9

```
cp -n ~/recordings/d435i/*.db3 /media/anormalm/T9/ah_bags_d435i/
cp -n ~/recordings/d435f/*.db3 /media/anormalm/T9/ah_bags_d435f/
ls -la ~/recordings/d435i ~/recordings/d435f /media/anormalm/T9/ah_bags_d435i /media/anormalm/T9/ah_bags_d435f | tail -40
```
- `cp -n` never overwrites anything already on the T9.
- Check that the copies on the T9 are **the same size** as the originals.
- **Don't delete anything from the internal disk today.** Claude checks the copies (sizes and frame
  counts) before the internal copies are removed.

---

## 5. Log sheet (fill in as you go; no names, IDs only)

| Clip name | Person ID | Staff / student | Height (cm) | Bed | Time | Saved fps (printed) | Notes (what happened, anything odd) |
|---|---|---|---|---|---|---|---|
| setup_empty_01 | - | - | - | - | | | |
| setup_stand_01 | | | | nearest | | | |
| | | | | | | | |

Also note: **which student is the locked test**, and whether any camera or bed was moved.

---

## 6. End of the day: unmount the T9

1. Make sure nothing is still copying or recording.
2. Stop the review-video server if it is running (it keeps the T9 busy):
   ```
   pkill -f "ahfd serve-review"
   ```
3. Unmount:
   ```
   sync
   sudo umount /media/anormalm/T9
   ```
   - No message = success. You can unplug the T9 now.
   - "target is busy": close any terminal or file window that is inside `/media/anormalm/T9`
     (`cd ~` first), then run `sudo umount /media/anormalm/T9` again.
4. Leave the cameras plugged in.

---

## 7. What happens next (Claude)

Check every recording (fps, frame counts, copies), refit the floor from `setup_empty_01`, extract
the keypoints, then label: **staff clips with the 7-phase labeller, student fall clips with the
4-state posture labeller** (key 4 = on_ground). See `docs/FALL_PLAN.md` for the fall training plan.
