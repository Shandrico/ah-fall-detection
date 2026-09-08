"""Standard-library web server for the dashboard.

No FastAPI, no uvicorn -- deliberately. The reference dashboard's heavy async
stack was where a stream generator failed to notice disconnects and leaked a
thread per browser reconnect until the box overheated. `ThreadingHTTPServer` is
enough here and keeps the failure modes in plain sight, and it drops onto the
Jetson with nothing to install.

The MJPEG endpoint is the part that has to be right:

* Frames are produced once by the pipeline thread; this endpoint only forwards
  the latest bytes. It never encodes.
* It writes each new frame as it appears. When the browser closes the tab, the
  next write raises a socket error, the handler breaks out and the thread ends.
  That is the disconnect detection the reference build was missing.
* The page loads the stream once via an <img> tag; there is no timer
  reopening it, so connections do not accumulate.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ahfd.dashboard.html import DASHBOARD_HTML
from ahfd.dashboard.state import DashboardState

_BOUNDARY = "ahfdframe"


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        # Quieten the default per-request stderr logging.
        def log_message(self, *args) -> None:  # noqa: A003
            pass

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", DASHBOARD_HTML.encode())
            elif path == "/api/state":
                body = json.dumps(state.snapshot()).encode()
                self._send(200, "application/json", body)
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/ack/"):
                event_id = path[len("/api/ack/"):]
                state.acknowledge(event_id)
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self._send(404, "text/plain", b"not found")

        def _stream(self) -> None:
            """MJPEG multipart stream of the latest annotated frame."""
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=" + _BOUNDARY,
            )
            self.end_headers()

            last_seq = -1
            try:
                while True:
                    jpeg, seq = state.latest_frame()
                    if jpeg is None or seq == last_seq:
                        # Nothing new. Brief pause; do not busy-spin. A stale
                        # connection is still detected on the next real write.
                        time.sleep(0.02)
                        continue
                    last_seq = seq
                    # A write to a closed socket raises here -> we stop. That is
                    # the disconnect detection the reference build lacked.
                    self.wfile.write(("--" + _BOUNDARY + "\r\n").encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        ("Content-Length: " + str(len(jpeg)) + "\r\n\r\n").encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser went away; let the handler thread finish

    return Handler


class DashboardServer(ThreadingHTTPServer):
    # ThreadingHTTPServer already gives a thread per request. daemon_threads
    # means a viewer still mid-stream cannot block process shutdown.
    daemon_threads = True

    def __init__(self, state: DashboardState, host: str = "127.0.0.1", port: int = 8000):
        super().__init__((host, port), make_handler(state))
        self.state = state
