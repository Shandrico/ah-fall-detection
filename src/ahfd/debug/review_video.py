"""Opt-in local review proxies from already-recorded RealSense files.

Raw RGB stays under the caller's approved raw directory. Every encoded frame
has an original recording timestamp; the CFR player clock is NOT ground truth.
No pose labels, model predictions or live camera are used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ahfd.config import load_config
from ahfd.privacy import require_raw_capture


def export_clip(manifest_path: Path, raw_root: Path, output: Path, *, preview_hz: float = 10.,
                config_flag: bool = False, cli_flag: bool = False) -> dict:
    require_raw_capture(config_flag=config_flag, cli_flag=cli_flag)
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    import math
    if not math.isfinite(preview_hz) or not 1 <= preview_hz <= 30:
        raise ValueError('Preview rate must be between 1 and 30 Hz')
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError('Extraction must be complete')
    raw = Path(meta['source']).resolve()
    raw.relative_to(raw_root.resolve())
    if raw.stat().st_size != meta['raw_bytes']:
        raise ValueError('Recording size changed')
    output.resolve().relative_to(raw_root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    stem = raw.stem
    movie = output / (stem + '.webm')
    mapping = output / (stem + '.json')
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if movie.exists() or mapping.exists():
        if movie.exists() and mapping.exists():
            old = json.loads(mapping.read_text())
            if old.get('status') == 'complete' and old.get('extraction_sha256') == manifest_hash and old.get('fps') == preview_hz:
                return old
        raise ValueError('Existing incomplete/different proxy; inspect before replacing: ' + stem)
    partial = output / (stem + '.partial.webm')
    if partial.exists():
        raise ValueError('Partial proxy already exists: ' + str(partial))
    if shutil.disk_usage(output).free < 2_000_000_000:
        raise RuntimeError('Less than 2 GB free; no export started')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(raw), False)
    writer = None
    times = []
    last_progress = -1
    started = False
    next_t = 0.
    try:
        profile = pipeline.start(config)
        started = True
        profile.get_device().as_playback().set_real_time(False)
        while True:
            ok, frames = pipeline.try_wait_for_frames(5000)
            if not ok:
                break
            color = frames.get_color_frame()
            if not color or not frames.get_depth_frame():
                continue
            t = color.get_timestamp() / 1000 - meta['first_source_t_s']
            if t < -1e-4 or (times and t <= times[-1]):
                continue
            if t > meta['duration_s'] + 1e-4:
                break
            if t + 1e-6 < next_t:
                continue
            next_t = (math.floor((t + 1e-6) * preview_hz) + 1) / preview_hz
            if not times and t > .2:
                raise RuntimeError('Proxy start does not match extraction timebase')
            bgr = np.asanyarray(color.get_data())
            fmt = color.profile.format()
            if fmt == rs.format.rgb8:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            elif fmt != rs.format.bgr8:
                raise RuntimeError('Unsupported RGB pixel format: ' + str(fmt))
            if writer is None:
                width = 960
                height = round(bgr.shape[0] * width / bgr.shape[1] / 2) * 2
                writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*'VP80'), preview_hz, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('VP8 WebM encoder unavailable')
            writer.write(cv2.resize(bgr, (width, height)))
            times.append(round(float(t), 6))
            if int(t) // 10 != last_progress:
                last_progress = int(t) // 10
                print(f'clip {meta["clip_number"]}: {t:.1f}s, {len(times)} RGB frames', flush=True)
            if len(times) % 300 == 0 and shutil.disk_usage(output).free < 1_000_000_000:
                raise RuntimeError('Low disk space; partial export retained')
    finally:
        if writer is not None:
            writer.release()
        if started:
            pipeline.stop()
    if not times or times[-1] < meta['duration_s'] - .2:
        raise RuntimeError('Short proxy; incomplete file retained')
    check = cv2.VideoCapture(str(partial))
    encoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = check.read()
    check.release()
    if not ok or encoded != len(times):
        raise RuntimeError('Encoded proxy verification failed')
    partial.rename(movie)
    result = dict(schema='ahfd.review_video.v1', status='complete', clip_number=meta['clip_number'],
                  bed_id=meta['bed_id'], duration_s=meta['duration_s'], fps=preview_hz, frames=len(times),
                  frame_times_s=times, first_source_t_s=meta['first_source_t_s'],
                  extraction_sha256=manifest_hash, video=movie.name,
                  video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
                  sampling='existing recorded frames at preview rate, no interpolation',
                  source_gap_max_s=max(b-a for a,b in zip(times,times[1:])))
    with mapping.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('READY', meta['clip_number'], movie, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--i-understand-raw-capture', action='store_true')
    parser.add_argument('--clips', type=int, nargs='+', default=[5,11,1,2,3,4,6,7,8,9,10,12])
    parser.add_argument('--preview-hz', type=float, default=10.)
    args = parser.parse_args()
    config = load_config(args.config)
    require_raw_capture(config_flag=config.privacy.allow_raw_capture, cli_flag=args.i_understand_raw_capture)
    manifests = {json.loads(p.read_text())['clip_number']: p for p in args.data.glob('*.extraction.json')}
    for clip in args.clips:
        export_clip(manifests[clip], args.raw_root, args.output, preview_hz=args.preview_hz,
                    config_flag=config.privacy.allow_raw_capture, cli_flag=args.i_understand_raw_capture)


if __name__ == '__main__':
    main()
