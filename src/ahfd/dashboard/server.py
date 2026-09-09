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
  reopening it, so connections do not accumulate. A source switch does not
  disturb it: the sequence number simply stops advancing and the browser holds
  the last part it received.

POSTs change what the camera is pointing at, so they carry a JSON body and are
checked for a same-origin header. Both matter: a path-only POST with no body is
a CORS *simple request*, which any page the nurse happens to have open in
another tab can fire at 127.0.0.1 without a preflight. That was already true of
/api/ack, where a drive-by page could silently clear a standing fall alert.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ahfd.dashboard.html import DASHBOARD_HTML
from ahfd.dashboard.state import DashboardState

_BOUNDARY = "ahfdframe"


def _clean(value):
    """A non-empty string, or None. Blank fields mean 'leave this alone'."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def make_handler(state: DashboardState, controller=None):
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

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, "application/json", json.dumps(payload).encode())

        def _same_origin(self) -> bool:
            """Reject a POST driven from another site's page.

            The dashboard is unauthenticated by design (localhost, one ward
            box), which is fine for reading. Requests with no Origin (curl, the
            tests) are allowed; a browser always sends one cross-origin.
            """
            origin = self.headers.get("Origin")
            return not origin or origin.split("//", 1)[-1] == self.headers.get("Host")

        def _body(self) -> dict | None:
            """Parse a small JSON object. None means malformed or too large.

            The announced bytes are always consumed first, even when the
            request is going to be rejected: replying to a body we never read
            resets the connection and the client never sees the reason.
            """
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if n <= 0 or n > 1 << 20:
                return None
            raw = self.rfile.read(n)
            if n > 8192:  # a control message is tiny by definition
                return None
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", DASHBOARD_HTML.encode())
            elif path == "/api/state":
                body = json.dumps(state.snapshot()).encode()
                self._send(200, "application/json", body)
            elif path == "/api/options":
                if controller is None:
                    self._json(503, {"ok": False, "error": "controls unavailable"})
                else:
                    self._json(200, controller.options())
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _drain(self) -> None:
            """Discard an unread request body.

            Answering before reading it makes the client see a reset socket
            instead of the response -- on Windows, a ConnectionAbortedError
            rather than the 403 that explains what happened.
            """
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return
            if 0 < n <= 1 << 20:
                self.rfile.read(n)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            same_origin = self._same_origin()
            # Only the switch handler parses a body; every other route has to
            # discard one before replying. See _drain.
            if path != "/api/switch" or not same_origin:
                self._drain()

            if not same_origin:
                self._json(403, {"ok": False, "error": "cross-origin POST refused"})
            elif path.startswith("/api/ack/"):
                state.acknowledge(path[len("/api/ack/"):])
                self._send(200, "application/json", b'{"ok":true}')
            elif path.startswith("/api/unack/"):
                state.unacknowledge(path[len("/api/unack/"):])
                self._send(200, "application/json", b'{"ok":true}')
            elif path == "/api/switch":
                self._switch()
            else:
                self._send(404, "text/plain", b"not found")

        def _switch(self) -> None:
            if controller is None:
                self._json(503, {"ok": False, "error": "controls unavailable"})
                return
            body = self._body()
            if body is None:
                self._json(400, {"ok": False, "error": "expected a small JSON object"})
                return
            rgb = body.get("show_rgb")
            code, result = controller.switch(
                source=_clean(body.get("source")),
                backend=_clean(body.get("backend")),
                show_rgb=rgb if isinstance(rgb, bool) else None,
            )
            self._json(code, result)

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

    def __init__(
        self,
        state: DashboardState,
        host: str = "127.0.0.1",
        port: int = 8000,
        controller=None,
    ):
        super().__init__((host, port), make_handler(state, controller))
        self.state = state
        self.controller = controller
