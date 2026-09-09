"""Owns the running pipeline so the browser can change it.

The dashboard used to bake the source and the model in before the server
started; a nurse who needed the other camera had to find whoever owned the
terminal. This object holds the current `PipelineRunner` instead, and a switch
is stop-old / build-new -- never restart-in-place, because a camera handle is
not something to reuse on faith.

Two rules keep a switch safe:

* The state's generation is bumped BEFORE the old runner is asked to stop, so
  anything it publishes afterwards is discarded. A thread wedged in a blocking
  read cannot corrupt the display even if it never dies.
* One switch at a time, enforced by a non-blocking lock. The second request is
  rejected rather than queued: a nurse double-clicking must not queue up two
  camera teardowns.

The MJPEG connection is untouched throughout. The stream handler only forwards
whatever `state.latest_frame()` holds, so the <img> opened at page load survives
any number of switches -- it simply shows the last frame of the old source until
the new one produces its first.
"""

from __future__ import annotations

import threading

from ahfd.capture import SOURCE_SCHEMES, probe_realsense
from ahfd.dashboard.runner import PipelineRunner
from ahfd.dashboard.state import DashboardState
from ahfd.pose import AVAILABLE_BACKENDS

# A control message is short by definition; anything longer is a mistake or an
# attack, and rejecting it early keeps the error message useful.
SOURCE_URI_MAX = 512


def build_source_options(cfg, probe, current: str | None = None) -> list[dict]:
    """The camera picker: the configured list, plus a RealSense if one is here."""
    seen: set[str] = set()
    out: list[dict] = []
    for s in cfg.dashboard.sources:
        if s.uri not in seen:
            seen.add(s.uri)
            out.append({"label": s.label, "uri": s.uri, "detected": None})

    if probe.devices and "rs://" not in seen:
        # One entry even for two cameras: open_source("rs://") takes no serial,
        # so a second would be a button that opens the first one anyway.
        d = probe.devices[0]
        label = d.name
        if len(probe.devices) > 1:
            label += " (first of " + str(len(probe.devices)) + ")"
        if d.usb2:
            label += " -- USB 2 link, depth+colour will not fit"
        seen.add("rs://")
        out.append({"label": label, "uri": "rs://", "detected": True})

    if current and current not in seen:
        # Whatever --source was given has to appear, or the picker lies about
        # what is running.
        out.insert(0, {"label": current, "uri": current, "detected": None})
    return out


def allowed_backends(cfg) -> list[str]:
    """The models this deployment offers, in the code's canonical order."""
    wanted = [b for b in AVAILABLE_BACKENDS if b in cfg.dashboard.backends]
    return wanted or list(AVAILABLE_BACKENDS)


class DashboardController:
    def __init__(
        self,
        cfg,
        calib_path,
        state: DashboardState,
        *,
        source: str | None = None,
        show_rgb: bool = False,
        rgb_authorised: bool = False,
        runner_factory=PipelineRunner,  # test seam: no camera, no model
        probe=probe_realsense,  # test seam: no hardware
        log=None,  # typer.echo in the CLI, silent in tests
    ):
        self.cfg = cfg
        self.calib_path = calib_path
        self.state = state
        self.source = source or cfg.source
        self.show_rgb = show_rgb
        # RGB can always be turned OFF from the page; turning it ON needs the
        # process to have been started with the authorisation. See set_rgb.
        self._rgb_authorised = rgb_authorised or show_rgb
        self._runner_factory = runner_factory
        self._runner: PipelineRunner | None = None
        self._abandoned: list[PipelineRunner] = []
        self._switching = threading.Lock()
        self._log = log or (lambda msg: None)
        # Probed once, at construction. Enumerating the USB bus per HTTP
        # request would be a request-triggered hardware poke on a shared box.
        self._sources = build_source_options(cfg, probe(), current=self.source)
        self._backends = allowed_backends(cfg)

    # ---- lifecycle (main thread) ---------------------------------------

    def start(self) -> None:
        self._spawn(
            self.state.begin_generation(
                source=self.source,
                source_label=self._label(self.source),
                backend=self.cfg.pose.backend,
                show_rgb=self.show_rgb,
            )
        )

    def stop(self) -> None:
        self._retire(self._runner, self.state.generation)
        self._runner = None

    # ---- control plane (HTTP handler threads) --------------------------

    def options(self) -> dict:
        """Everything the picker needs, fetched once at page load."""
        return {
            "ok": True,
            "sources": self._sources,
            "backends": self._backends,
            "allow_custom_source": self.cfg.dashboard.allow_custom_source,
            "allow_rgb": self._rgb_authorised,
        }

    def switch(self, *, source=None, backend=None, show_rgb=None) -> tuple[int, dict]:
        """Point the pipeline at a different camera and/or model.

        Returns (http_status, payload) so the request handler does no
        interpretation of its own.
        """
        if source is None and backend is None:
            if show_rgb is None:
                return 400, {"ok": False, "error": "nothing to change"}
            return self.set_rgb(show_rgb)  # no restart needed; see set_rgb

        problem = self._validate(source, backend)
        if problem:
            return 400, {"ok": False, "error": problem}

        if not self._switching.acquire(blocking=False):
            return 409, {"ok": False, "error": "a switch is already in progress"}

        # The teardown runs on its own thread so the POST returns immediately
        # and the page's next poll shows "switching", rather than the browser
        # hanging for the length of the join timeout. A plain Lock has no
        # owning thread, so acquiring here and releasing there is legal.
        threading.Thread(
            target=self._do_switch,
            args=(source, backend, show_rgb),
            daemon=True,
            name="ahfd-switch",
        ).start()
        return 202, {"ok": True, "switching": True}

    def set_rgb(self, on: bool) -> tuple[int, dict]:
        """Turn the RGB view on or off without restarting the pipeline.

        Asymmetric on purpose. Turning RGB OFF is always allowed -- tightening
        the privacy stance needs nobody's permission. Turning it ON requires
        the process to have been started with --rgb or dashboard.show_rgb, i.e.
        by someone who saw the AH/DPO warning. A browser button must not be
        able to reverse the ward's default on its own.
        """
        if on and not self._rgb_authorised:
            return 403, {
                "ok": False,
                "error": "RGB view is not authorised for this session -- restart "
                "with --rgb (needs AH/DPO sign-off)",
            }
        self.show_rgb = on
        if self._runner is not None:
            self._runner.show_rgb = on  # read per frame; takes effect on the next one
        self._log(
            "RGB view ON -- live video is shown" if on else "RGB view off -- skeleton only"
        )
        self.state.update_runtime(self.state.generation, show_rgb=on)
        return 200, {"ok": True, "show_rgb": on}

    # ---- internals ------------------------------------------------------

    def _do_switch(self, source, backend, show_rgb) -> None:
        try:
            uri = source or self.source
            # A copy, not a mutation: the old runner is still reading cfg.pose.*
            # on its own thread until it notices the stop event.
            cfg = self.cfg.model_copy(deep=True)
            if backend:
                cfg.pose.backend = backend
            if show_rgb is not None and (self._rgb_authorised or not show_rgb):
                self.show_rgb = show_rgb

            gen = self.state.begin_generation(
                source=uri,
                source_label=self._label(uri),
                backend=cfg.pose.backend,
                model=None,
                resolution=None,
                show_rgb=self.show_rgb,
            )
            self._retire(self._runner, gen)
            self.cfg, self.source = cfg, uri
            self._spawn(gen)
        except Exception as exc:  # noqa: BLE001 -- a control-plane bug must not
            self.state.publish_status(  # leave the page stuck on "switching"
                self.state.generation, "error", error="switch failed: " + str(exc)
            )
        finally:
            self._switching.release()

    def _retire(self, runner, gen: int) -> None:
        if runner is None:
            return
        runner.stop(timeout=self.cfg.dashboard.switch_timeout_s)
        if runner.alive:
            # Wedged in a blocking read. It cannot paint anything -- its
            # generation is fenced -- but it still holds the device, so
            # reopening the SAME camera will fail until it lets go. Say so,
            # rather than letting "could not open video source" look random.
            self._abandoned = [r for r in self._abandoned if r.alive] + [runner]
            self._log("previous capture has not released " + str(runner.source_uri))
            self.state.publish_status(
                gen,
                "switching",
                warning="the previous camera has not released yet; if this "
                "fails, wait a few seconds and try again",
            )

    def _spawn(self, gen: int) -> None:
        self._runner = self._runner_factory(
            self.source,
            self.cfg,
            self.calib_path,
            self.state,
            show_rgb=self.show_rgb,
            gen=gen,
        )
        self._runner.start()

    def _label(self, uri: str) -> str:
        for s in self._sources:
            if s["uri"] == uri:
                return s["label"]
        return uri

    def _validate(self, source, backend) -> str | None:
        if backend is not None and backend not in self._backends:
            return (
                "unknown backend "
                + repr(backend)
                + " (offered: "
                + ", ".join(self._backends)
                + ")"
            )
        if source is None:
            return None
        if not source or len(source) > SOURCE_URI_MAX:
            return "source URI is empty or too long"
        if any(source == s["uri"] for s in self._sources):
            return None
        if not self.cfg.dashboard.allow_custom_source:
            return "this dashboard only offers the configured cameras"
        # open_source also accepts a bare path, but the browser box deliberately
        # does not: a bare path from a text box is indistinguishable from a typo,
        # and file:// is one prefix more to type.
        if not source.startswith(SOURCE_SCHEMES):
            return "unknown source scheme -- use one of " + " ".join(SOURCE_SCHEMES)
        return None
