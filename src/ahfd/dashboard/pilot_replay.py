"""Read-only loopback viewer for saved pilot predictions and consented video.

Run with --data, --models and --videos pointing to the external study volume.
No live camera, retraining, annotation mutation, or network publication.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from ahfd.eval.pilot_replay import compare_reference, simulate

VARIANTS = ('rgb_logistic', 'rgb_depth_logistic', 'rgb_tree', 'rgb_depth_tree')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayLibrary:
    def __init__(self, data: Path, models: Path, videos: Path):
        self.data, self.models, self.videos = data.resolve(), models.resolve(), videos.resolve()
        report_path = models / 'training_report.json'
        self.report = json.loads(report_path.read_text())
        if self.report['status'] != 'complete' or self.report['live_inference_enabled']:
            raise ValueError('Expected complete offline pilot report')
        labels = data / 'reviewed_labels.json'
        if sha(labels) != self.report['review_sha256']:
            raise ValueError('Reference labels changed since training')
        self.reference = json.loads(labels.read_text())['clips']
        self.clips, self.predictions = {}, {}
        inputs = {i['stem']: i for i in self.report['inputs']}
        for manifest in data.glob('*.extraction.json'):
            meta = json.loads(manifest.read_text())
            stem = manifest.name.removesuffix('.extraction.json')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', stem):
                raise ValueError('Unsafe clip identifier')
            if meta['status'] != 'complete' or sha(manifest) != inputs[stem]['extraction_sha256']:
                raise ValueError('Extraction provenance mismatch')
            tracks = data / 'tracks' / (stem + '.jsonl')
            if sha(tracks) != inputs[stem]['tracks_sha256']:
                raise ValueError('Tracks changed since training')
            self.clips[meta['clip_number']] = {**meta, 'stem': stem, 'hash': sha(manifest),
                                               'tracks': [json.loads(l) for l in tracks.read_text().splitlines()]}
        predictions_path = models / 'replay_predictions.jsonl'
        self.predictions_hash = sha(predictions_path)
        self.report_hash = sha(report_path)
        for line in predictions_path.read_text().splitlines():
            row = json.loads(line)
            clip = row['clip_number']
            if row['stem'] != self.clips[clip]['stem']:
                raise ValueError('Prediction clip mismatch')
            key = round(row['t'] * 10000)
            bucket = self.predictions.setdefault(clip, {})
            if key in bucket:
                raise ValueError('Duplicate prediction time')
            bucket[key] = row

    def media(self, clip):
        meta = self.clips[clip]
        path = self.videos / (meta['stem'] + '.json')
        movie = self.videos / (meta['stem'] + '.webm')
        if not path.exists() or not movie.exists():
            return None
        mapping = json.loads(path.read_text())
        if (mapping['status'] != 'complete' or mapping['extraction_sha256'] != meta['hash']
                or mapping['video'] != movie.name or mapping['frames'] != len(mapping['frame_times_s'])
                or any(b <= a for a,b in zip(mapping['frame_times_s'], mapping['frame_times_s'][1:]))):
            raise ValueError('Video timestamp/provenance mismatch')
        return mapping

    def catalog(self):
        return {'scope': 'TRAINING_DATA_REPLAY_NOT_VALIDATION', 'live': False,
                'clips': [{'id': n, 'bed': m['bed_id'], 'name': m['stem'].split('_2026')[0].replace('_',' '),
                           'duration_s': m['duration_s'], 'video_ready': self.media(n) is not None}
                          for n,m in sorted(self.clips.items())]}

    @lru_cache(maxsize=144)
    def replay(self, clip: int, variant: str = 'rgb_depth_logistic', horizon: int = 10):
        if variant not in VARIANTS or horizon not in (5,10,20):
            raise ValueError('Unsupported model/horizon')
        meta = self.clips[clip]
        rows = []
        for frame in meta['tracks']:
            t = frame['t']
            pred = self.predictions.get(clip, {}).get(round(t*10000))
            single = len(frame['people']) == 1
            person = frame['people'][0] if single else None
            available = bool(t >= 1 and pred and person and person['track_id'] is not None and person['score'] > 0)
            score = pred['predictions'][f'exit_{horizon}s_{variant}']['probabilities']['1'] if available else None
            phase = pred['predictions']['phase_' + variant]['label'] if available else None
            rows.append({'t': t, 'available': available, 'score': score, 'phase': phase,
                         'track_id': person['track_id'] if available else None,
                         'reason': '' if available else 'STARTUP' if t < 1 else 'MULTIPLE_PEOPLE' if len(frame['people']) > 1 else 'NO_USABLE_POSE',
                         'keypoints': person['keypoints'] if available else None,
                         'scores': person['scores'] if available else None})
        # Source tracks round times to 4 decimals; clamp only last-rounding drift.
        duration = max(meta['duration_s'], rows[-1]['t'])
        result = simulate(rows, duration)
        ref = self.reference[str(clip)]
        offset = ref.get('event_review_first_source_t_s',meta['first_source_t_s']) - meta['first_source_t_s']
        exits = [[lo+offset,hi+offset] for lo,hi in ref['exit_intervals']]
        return {**result, 'clip': clip, 'bed': meta['bed_id'], 'duration_s': duration,
                'variant': variant, 'horizon_s': horizon, 'width': rows and meta['intrinsics']['width'],
                'height': meta['intrinsics']['height'], 'reference_segments': ref['segments'],
                'reference_exits': exits, 'reference_notes': ref['notes'],
                'reference_quality': 'approximate sampled review; not independent adjudication',
                'comparison': compare_reference(result['events'], exits, horizon),
                'scope': 'TRAINING_DATA_REPLAY_NOT_VALIDATION', 'assignment': 'known from clip metadata; not evaluated',
                'calibration_verified': False, 'live': False}

    def write_summary(self, destination: Path):
        summary = {'scope': 'TRAINING_DATA_REPLAY_NOT_VALIDATION', 'live': False,
                   'probabilities_calibrated': False, 'source_predictions_sha256': self.predictions_hash,
                   'source_training_report_sha256': self.report_hash,
                   'limitations': ['Same recordings used to train these models.',
                                   'Demonstration warning policy, not clinically selected thresholds.',
                                   'Known clip/bed assignment; no live identity evaluation.',
                                   'No validated floor heights, fall detector, or wall-clock inference latency.'],
                   'clips': []}
        for clip in sorted(self.clips):
            r = self.replay(clip)
            summary['clips'].append({k:v for k,v in r.items() if k not in ('rows','reference_segments')})
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, allow_nan=False)


def byte_range(value: str | None, size: int):
    if value is None:
        return 0, size-1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
    if not match or not any(match.groups()):
        raise ValueError('Invalid byte range')
    start, end = match.groups()
    if not start:
        suffix = int(end)
        if suffix <= 0:
            raise ValueError('Invalid suffix')
        return max(0,size-suffix), size-1, True
    lo, hi = int(start), min(int(end), size-1) if end else size-1
    if lo >= size or hi < lo:
        raise ValueError('Unsatisfiable range')
    return lo, hi, True


def make_handler(library):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body, mime='application/json', extra=None):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.headers_common()
            for key,value in (extra or {}).items():
                self.send_header(key,value)
            self.end_headers()
            self.wfile.write(body)

        def headers_common(self):
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Cross-Origin-Resource-Policy','same-origin')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'none'")

        def do_GET(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.reply(403,b'{"error":"Loopback same-origin access only"}')
                return
            try:
                url = urlsplit(self.path)
                params = parse_qs(url.query)
                if url.path == '/':
                    from ahfd.dashboard.pilot_replay_html import HTML
                    self.reply(200, HTML.encode(), 'text/html; charset=utf-8')
                elif url.path == '/api/clips':
                    self.reply(200,json.dumps(library.catalog()).encode())
                elif url.path == '/api/replay':
                    clip = int(params['clip'][0])
                    result = dict(library.replay(clip,params.get('model',['rgb_depth_logistic'])[0],int(params.get('horizon',['10'])[0])))
                    result['media'] = library.media(clip)
                    self.reply(200,json.dumps(result,allow_nan=False).encode())
                elif url.path == '/video':
                    clip = int(params['clip'][0])
                    if library.media(clip) is None:
                        self.reply(404,b'{"error":"Video still preparing"}')
                        return
                    path = library.videos / (library.clips[clip]['stem'] + '.webm')
                    size = path.stat().st_size
                    try:
                        lo,hi,partial = byte_range(self.headers.get('Range'),size)
                    except ValueError:
                        self.reply(416,b'',extra={'Content-Range':f'bytes */{size}'})
                        return
                    self.send_response(206 if partial else 200)
                    self.send_header('Content-Type','video/webm')
                    self.send_header('Accept-Ranges','bytes')
                    self.send_header('Content-Length',str(hi-lo+1))
                    if partial:
                        self.send_header('Content-Range',f'bytes {lo}-{hi}/{size}')
                    self.headers_common()
                    self.end_headers()
                    with path.open('rb') as f:
                        f.seek(lo)
                        left = hi-lo+1
                        while left:
                            chunk = f.read(min(left,256*1024))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            left -= len(chunk)
                else:
                    self.reply(404,b'{"error":"Not found"}')
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                pass
            except (KeyError,ValueError,TypeError):
                self.reply(400,b'{"error":"Invalid clip, model, or data"}')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','models','videos'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8003)
    parser.add_argument('--summary',type=Path)
    args = parser.parse_args()
    library = ReplayLibrary(args.data,args.models,args.videos)
    if args.summary:
        library.write_summary(args.summary)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(library))
    print(f'Local replay: http://127.0.0.1:{server.server_port}/',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
