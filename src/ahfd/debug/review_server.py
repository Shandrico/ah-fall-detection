"""Serve rendered review videos to a browser, with seeking (HTTP Range).

Lists the ``.mp4`` files in one directory and plays them in a ``<video>``
element. The browser buffers ahead, so a 1080p review plays smoothly even over
a slow link once a little has loaded. Binds to localhost by default -- reach it
through an SSH tunnel, exactly like the dashboard. Serves only ``.mp4`` files
directly inside the directory; no paths, no listing outside it, no uploads.
"""

from __future__ import annotations

import html
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

_CHUNK = 1 << 20


def _page(videos: list[str]) -> bytes:
    items = "".join(
        '<li><a href="#" data-v="%s">%s</a></li>' % (html.escape(quote(v)), html.escape(v))
        for v in videos
    ) or "<li>no .mp4 files yet</li>"
    return ("""<!doctype html><meta charset="utf-8"><title>AHFD review videos</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{margin:0;padding:16px;font:14px system-ui,sans-serif;background:#111;color:#eee}
 a{color:#7cc4ff} ul{padding-left:18px} video{width:100%%;max-height:80vh;background:#000}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0} .t{font-variant-numeric:tabular-nums;color:#aaa}
</style>
<h2>Review videos</h2><ul>%s</ul>
<video id="v" controls preload="auto"></video>
<div class="bar"><span id="name"></span><span class="t" id="t"></span>
 <label>speed <select id="sp"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label>
 <span class="t">keys: space play/pause, &larr;/&rarr; 1 s, , . one frame</span></div>
<script>
const v=document.getElementById('v'), t=document.getElementById('t');
document.querySelectorAll('a[data-v]').forEach(a=>a.onclick=e=>{e.preventDefault();
  v.src='video/'+a.dataset.v; document.getElementById('name').textContent=decodeURIComponent(a.dataset.v); v.play();});
document.getElementById('sp').onchange=e=>v.playbackRate=parseFloat(e.target.value);
v.ontimeupdate=()=>{const s=v.currentTime,m=Math.floor(s/60);t.textContent=m+':'+(s-60*m).toFixed(1).padStart(4,'0')+'  (frame '+Math.round(s*30)+')';};
document.onkeydown=e=>{if(!v.src)return;const f=1/30;
  if(e.key===' '){e.preventDefault();v.paused?v.play():v.pause();}
  else if(e.key==='ArrowLeft')v.currentTime-=1; else if(e.key==='ArrowRight')v.currentTime+=1;
  else if(e.key===','){v.pause();v.currentTime-=f;} else if(e.key==='.'){v.pause();v.currentTime+=f;}};
</script>""" % items).encode()


def make_handler(root: Path):
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet; the terminal is for the user
            pass

        def _videos(self) -> list[str]:
            return sorted(
                p.name for p in root.glob("*.mp4") if not p.name.endswith(".partial.mp4")
            )

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path in ("/", "/index.html"):
                body = _page(self._videos())
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.path.startswith("/video/"):
                self.send_error(404)
                return
            name = unquote(self.path[len("/video/"):])
            if name not in self._videos():  # only listed files, never a path
                self.send_error(404)
                return
            path = root / name
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:  # suffix range: the last N bytes
                    start = max(0, size - int(match.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                with path.open("rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(_CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser seeks by dropping the old request

    return Handler


def serve(root: Path, host: str = "127.0.0.1", port: int = 8001) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(Path(root)))
    try:
        server.serve_forever()
    finally:
        server.server_close()
