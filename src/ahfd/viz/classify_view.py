"""Side-by-side viewer: RGB-model vs RGB+depth-model posture, on one recording.

Replays a depth recording and, per frame, runs the pose estimator, measures the
joint heights from depth, builds the posture features, and predicts with TWO
models trained on the labelled depth clips:

* an **RGB-only** model (the dh_* depth features masked out), drawn on the colour
  pane, and
* an **RGB+depth** model (all features), drawn on the depth pane.

The clip's own person is held out of BOTH models' training, so the calls you see
are honest (cross-person), and the ground-truth posture is shown when a label
exists -- so you can watch exactly which frames depth gets right that RGB misses
(the sitting / nadir cases the numbers flagged).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ahfd.pose.skeleton import EDGES


def _person_of(stem: str) -> str:
    tail = stem.rsplit("_", 1)[-1]
    return "person_" + tail if tail.isdigit() else "person_?"


def _train_two_models(labels_dir, tracks_dir, calib, exclude_person, exclude_stem=None):
    """Fit (rgb_only, rgb_depth) models, holding the viewed clip out of training.

    Holds out the whole person when there is more than one (a real cross-person
    test); for a single-subject set, holds out just the viewed CLIP so the calls
    are still on unseen frames, not the ones it trained on.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ahfd.ml.compare import _FIDX, rows_to_matrix
    from ahfd.ml.posture import DEPTH_FEATURES, build_dataset

    rows, labels, groups, used, _ = build_dataset(labels_dir, tracks_dir, calib)
    keep = [i for i, g in enumerate(groups) if _person_of(g) != exclude_person]
    if not keep and exclude_stem is not None:  # single subject: hold out the clip
        keep = [i for i, g in enumerate(groups) if g != exclude_stem]
    if not keep:  # last resort -> all clips (optimistic)
        keep = list(range(len(rows)))
    X = rows_to_matrix([rows[i] for i in keep])
    y = np.array([labels[i] for i in keep])

    def _pipe():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

    depth_cols = [_FIDX[f] for f in DEPTH_FEATURES if f in _FIDX]
    rgb_depth = _pipe().fit(X, y)
    Xr = X.copy()
    Xr[:, depth_cols] = np.nan
    rgb_only = _pipe().fit(Xr, y)
    n_train = len({g for i, g in enumerate(groups) if i in set(keep)})
    return rgb_only, rgb_depth, depth_cols, n_train


def _colorize_depth(depth_m, dmin, dmax):
    import cv2

    norm = np.clip((depth_m - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[depth_m <= 0] = 0  # holes black
    return col


def _draw_skeleton(img, kp, sc, thr=0.3):
    import cv2

    for a, b in EDGES:
        if sc[a] >= thr and sc[b] >= thr:
            cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                     (255, 255, 255), 2, cv2.LINE_AA)
    for i in range(len(kp)):
        if sc[i] >= thr:
            p = (int(kp[i][0]), int(kp[i][1]))
            cv2.circle(img, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p, 3, (60, 220, 60), -1, cv2.LINE_AA)


_GREEN, _RED, _GREY, _WHITE, _CYAN = (60, 200, 60), (50, 50, 225), (150, 150, 150), (240, 240, 240), (0, 255, 255)


def _verdict_color(pred, truth):
    if truth is None or pred is None:
        return _GREY
    return _GREEN if pred == truth else _RED


def _render_dashboard(cv2, np, rgb, dvis, clip, truth, t, pred_rgb, pred_depth, stats, timeline, pane_w=760, pos=None):
    """Compose the two panes + header + live stats + timeline into one canvas."""
    ph = max(1, int(rgb.shape[0] * pane_w / rgb.shape[1]))
    rp = cv2.resize(rgb, (pane_w, ph))
    dp = cv2.resize(dvis, (pane_w, ph))
    head, stat, w2 = 48, 78, pane_w * 2
    cv = np.full((head + ph + stat, w2, 3), 26, np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX

    # --- header: clip, truth, time ---
    cv2.putText(cv, "clip: " + clip, (14, 32), F, 0.8, _WHITE, 2, cv2.LINE_AA)
    if truth is not None:
        tt = "TRUTH: " + truth.upper()
        (tw, _), _ = cv2.getTextSize(tt, F, 0.8, 2)
        cv2.putText(cv, tt, (w2 // 2 - tw // 2, 32), F, 0.8, _CYAN, 2, cv2.LINE_AA)
    cv2.putText(cv, "t=%.1fs" % t, (w2 - 150, 32), F, 0.75, _WHITE, 2, cv2.LINE_AA)

    # --- panes ---
    cv[head:head + ph, 0:pane_w] = rp
    cv[head:head + ph, pane_w:w2] = dp
    cv2.line(cv, (pane_w, head), (pane_w, head + ph), (26, 26, 26), 3)

    def _pane(x0, title, pred):
        cv2.rectangle(cv, (x0, head), (x0 + pane_w, head + 30), (0, 0, 0), -1)
        cv2.putText(cv, title, (x0 + 12, head + 22), F, 0.62, _WHITE, 2, cv2.LINE_AA)
        badge = (pred or "--").upper()
        col = _verdict_color(pred, truth)
        (bw, bh), _ = cv2.getTextSize(badge, F, 1.1, 3)
        bx, by = x0 + (pane_w - bw) // 2, head + ph - 22
        cv2.rectangle(cv, (bx - 16, by - bh - 14), (bx + bw + 16, by + 12), (0, 0, 0), -1)
        cv2.putText(cv, badge, (bx, by), F, 1.1, col, 3, cv2.LINE_AA)

    _pane(0, "RGB MODEL", pred_rgb)
    _pane(pane_w, "RGB + DEPTH MODEL", pred_depth)

    # --- stats bar ---
    y = head + ph
    def _pct(a, b):
        return (100.0 * a / b) if b else 0.0
    s = stats
    line = "agreement %.0f%%    RGB acc %.0f%%    RGB+Depth acc %.0f%%    frames %d" % (
        _pct(s["agree"], s["n"]), _pct(s["rgb_ok"], s["truth_n"]),
        _pct(s["dep_ok"], s["truth_n"]), s["n"])
    cv2.putText(cv, line, (14, y + 26), F, 0.68, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(cv, "timeline:  green = depth fixes a wrong RGB call    red = depth wrong",
                (14, y + 70), F, 0.5, _GREY, 1, cv2.LINE_AA)

    # --- timeline strip (last w2 frames): highlight where depth beats RGB ---
    ty0, ty1 = y + 36, y + 52
    n = len(timeline)
    if n:
        for i, (rc, dc) in enumerate(timeline):
            x = int(i * (w2 - 1) / max(n - 1, 1))
            if rc is False and dc is True:
                col = (60, 230, 60)      # the win: RGB wrong, depth right
            elif dc is False:
                col = (50, 50, 225)      # depth wrong
            elif dc is None:
                col = (55, 55, 55)       # no ground truth
            else:
                col = (90, 90, 90)       # both fine
            cv2.line(cv, (x, ty0), (x, ty1), col, 1)
        if pos is not None:  # playhead marker
            px = int(pos * (w2 - 1))
            cv2.line(cv, (px, ty0 - 5), (px, ty1 + 5), (255, 255, 255), 2)
    return cv


def _play(cv2, cache, stats, timeline, stem, pane_w, win):
    """Scrub the cached frames: trackbar seek, space=play/pause, a/d step."""
    n = len(cache)
    print("player: space=play/pause  a/d=step  [ ]=jump 30  q=quit  (%d frames)" % n)
    state = {"idx": 0, "play": True, "prog": False}

    def _on_track(v):
        if state["prog"]:
            return
        state["idx"] = v
        state["play"] = False  # dragging the bar pauses

    cv2.createTrackbar("frame", win, 0, n - 1, _on_track)

    while True:
        idx = max(0, min(n - 1, state["idx"]))
        rjpg, djpg, pr, pd, truth, t = cache[idx]
        rp = cv2.imdecode(rjpg, cv2.IMREAD_COLOR)
        dp = cv2.imdecode(djpg, cv2.IMREAD_COLOR)
        canvas = _render_dashboard(cv2, np, rp, dp, stem, truth, t, pr, pd,
                                   stats, timeline, pane_w=pane_w, pos=idx / max(n - 1, 1))
        tag = "PLAY " if state["play"] else "PAUSE"
        cv2.putText(canvas, tag + "  %d/%d" % (idx + 1, n), (canvas.shape[1] - 260, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if state["play"] else _WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, "SPACE pause/play   a/d step frame   [ ] jump 30   drag bar seek   q quit",
                    (14, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(25 if state["play"] else 40) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            state["play"] = not state["play"]
        elif key in (ord("a"), ord(",")):
            state["idx"], state["play"] = max(0, idx - 1), False
        elif key in (ord("d"), ord(".")):
            state["idx"], state["play"] = min(n - 1, idx + 1), False
        elif key == ord("["):
            state["idx"], state["play"] = max(0, idx - 30), False
        elif key == ord("]"):
            state["idx"], state["play"] = min(n - 1, idx + 30), False

        if state["play"]:
            nxt = idx + 1
            if nxt >= n:
                state["play"] = False  # stop at the end
            else:
                state["idx"] = nxt
                state["prog"] = True
                cv2.setTrackbarPos("frame", win, nxt)
                state["prog"] = False

    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


def run_classify_viewer(
    source: str,
    *,
    calibration,
    height_m: float = 2.5,
    labels_dir: str = "data/postures_depth",
    tracks_dir: str = "data/tracks_depth",
    backend: str = "rtmo",
    runtime: str = "openvino",
    device: str = "gpu",
    dmin: float = 2.5,
    dmax: float = 5.5,
    max_width: int = 1600,
    rebuild: bool = False,
    pitch: float | None = None,
) -> None:
    import cv2

    from ahfd.capture.realsense import BagSource
    from ahfd.config import PoseConfig
    from ahfd.features import FeatureExtractor
    from ahfd.geometry.calibration import load_calibration
    from ahfd.geometry.depth_height import keypoint_heights_from_depth
    from ahfd.geometry.ground import GroundPlane
    from ahfd.ml.compare import _FIDX
    from ahfd.ml.posture import FEATURES, _labelled_segments, _posture_at, features_row
    from ahfd.pose import build_estimator

    src_path = source[len("bag://"):] if source.startswith("bag://") else source
    stem = Path(src_path).stem
    person = _person_of(stem)
    pane_w = min(760, max_width // 2)
    win = "ahfd classify: RGB vs RGB+depth"

    # Re-opening a clip: replay the saved run instantly -- no pose, no .db3 read.
    from ahfd.debug.frame_cache import load_cache, save_cache

    payload = None if rebuild else load_cache(stem)
    if payload is not None:
        print("loaded cached run for %s (%d frames). Pass --rebuild to refresh." % (stem, len(payload["cache"])))
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        _play(cv2, payload["cache"], payload["stats"], payload["timeline"], stem, pane_w, win)
        return

    calib = load_calibration(Path(calibration))
    print("training RGB-only and RGB+depth models (holding out " + person + ") ...")
    rgb_only, rgb_depth, depth_cols, n_train = _train_two_models(
        labels_dir, tracks_dir, calib, person, exclude_stem=stem
    )
    print("trained on " + str(n_train) + " held-out clip(s).")

    # Ground truth for this clip, if it was labelled (times are 0-based seconds).
    seg_path = Path(labels_dir) / (stem + ".json")
    segments = _labelled_segments(seg_path) if seg_path.exists() else []

    print("loading pose model (" + backend + " / " + runtime + ") ...")
    estimator = build_estimator(PoseConfig(backend=backend, runtime=runtime, device=device))
    extractor = FeatureExtractor(calib.ground, zones=calib.zones, min_keypoint_score=0.3)

    feat_idx = list(FEATURES)
    pane_w = min(760, max_width // 2)

    def _area(p):
        m = p.scores >= 0.3
        return float(np.ptp(p.keypoints[m][:, 0]) * np.ptp(p.keypoints[m][:, 1])) if m.any() else 0.0

    # ---- pass 1: run pose + both models on every frame, cache the result ----
    # Panes are JPEG-encoded so a whole ~2 min clip fits in a few hundred MB and
    # the player can seek instantly (a .db3 cannot random-seek smoothly).
    src = BagSource(src_path, with_depth=True)
    cache = []  # (rgb_jpg, depth_jpg, pred_rgb, pred_depth, truth, t)
    stats = {"n": 0, "truth_n": 0, "rgb_ok": 0, "dep_ok": 0, "agree": 0}
    timeline = []
    t0 = None
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 82]
    win = "ahfd classify: RGB vs RGB+depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("pre-processing (pose + both models) -- watch live; scrubbing turns on when done ...")
    aborted = False
    try:
        for k, frame in enumerate(src):
            if frame.bgr is None or frame.depth_raw is None:
                continue
            if t0 is None:
                t0 = frame.t
            t = frame.t - t0
            rgb = frame.bgr.copy()
            depth_m = frame.depth_raw.astype(np.float32) * float(frame.depth_scale)
            dvis = _colorize_depth(depth_m, dmin, dmax)

            pred_rgb = pred_depth = None
            people = estimator.estimate(frame).people
            if people:
                p = max(people, key=_area)
                _draw_skeleton(rgb, p.keypoints, p.scores)
                _draw_skeleton(dvis, p.keypoints, p.scores)
                gp = None
                if frame.intrinsics is not None:
                    try:
                        if frame.gravity is not None:
                            gp = GroundPlane.from_gravity(frame.intrinsics, height_m, np.asarray(frame.gravity))
                        elif pitch is not None:  # D435f (no IMU): fixed mount tilt
                            gp = GroundPlane(intrinsics=frame.intrinsics, height_m=height_m,
                                             pitch_deg=float(pitch), roll_deg=0.0)
                    except Exception:
                        gp = None
                if gp is not None:
                    h = keypoint_heights_from_depth(p.keypoints, p.scores, depth_m, frame.intrinsics, gp)
                    p = p.with_track_id(0).with_heights(h)
                    feats = extractor.extract(p, t)
                    if feats is not None and feats.has_geometry():
                        row = features_row(feats, p, extractor.ground, 0.3)
                        vec = np.array([[np.nan if row.get(kk) is None else float(row[kk]) for kk in feat_idx]])
                        vmask = vec.copy()
                        vmask[:, depth_cols] = np.nan
                        pred_rgb = str(rgb_only.predict(vmask)[0])
                        pred_depth = str(rgb_depth.predict(vec)[0])

            truth = _posture_at(segments, t) if segments else None
            rc = None if (truth is None or pred_rgb is None) else (pred_rgb == truth)
            dc = None if (truth is None or pred_depth is None) else (pred_depth == truth)
            if pred_rgb is not None and pred_depth is not None:
                stats["n"] += 1
                stats["agree"] += int(pred_rgb == pred_depth)
                if truth is not None:
                    stats["truth_n"] += 1
                    stats["rgb_ok"] += int(pred_rgb == truth)
                    stats["dep_ok"] += int(pred_depth == truth)
            timeline.append((rc, dc))

            rp = cv2.resize(rgb, (pane_w, int(rgb.shape[0] * pane_w / rgb.shape[1])))
            dp = cv2.resize(dvis, (pane_w, int(dvis.shape[0] * pane_w / dvis.shape[1])))
            cache.append((cv2.imencode(".jpg", rp, jpg)[1], cv2.imencode(".jpg", dp, jpg)[1],
                          pred_rgb, pred_depth, truth, t))

            # Show it live as the cache builds -- no blank wait, and 'q' jumps
            # straight to scrubbing with whatever has been processed so far.
            live = _render_dashboard(cv2, np, rp, dp, stem, truth, t, pred_rgb, pred_depth,
                                     stats, timeline, pane_w=pane_w)
            cv2.putText(live, "BUILDING SCRUBBER  %d frames  (q = start now)" % (k + 1),
                        (live.shape[1] - 480, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, live)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                aborted = True
                break
            if k % 30 == 0:
                print("  %d frames..." % k, end="\r")
    except Exception as exc:  # mid-stream end/disconnect -> stop pre-processing
        print("\npre-process stopped: " + str(exc))
    finally:
        try:
            src.close()
        except Exception:
            pass

    if not cache:
        print("no frames decoded -- nothing to play.")
        return
    save_cache(stem, {"cache": cache, "stats": stats, "timeline": timeline})
    _play(cv2, cache, stats, timeline, stem, pane_w, win)
