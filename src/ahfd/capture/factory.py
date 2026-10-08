"""URI-based source construction.

Schemes:
    webcam://0          built-in or USB camera by index
    file://path.mp4     video file (a bare path works too)
    rs://               live RealSense D435i (colour)
    rs://ir             live RealSense in infrared (low-light night vision)
    bag://path.bag      recorded RealSense playback (deterministic)
    seq://path/         directory of dataset images (not implemented yet)

The RealSense URI takes options as a query string, e.g.
``rs://?depth=1&max_laser=1`` or
``rs://ir?emitter=0&ir_index=1&w=1280&h=720``:

* ``ir`` (or the ``rs://ir`` shorthand) streams the left IR imager instead of
  colour -- a grayscale image for low light.
* ``emitter`` toggles the dot projector (``0`` = off). It defaults OFF in IR
  mode (the projected dots otherwise cover the scene; pair with an external IR
  floodlight so the room is lit) and is left at the device default otherwise.
* ``w`` / ``h`` set the stream resolution.
* ``depth`` enables aligned measurement depth.  It is off by default because
  filtering and alignment are relatively expensive.
* ``max_laser`` raises projector power for denser long-range depth.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from ahfd.capture.base import FrameSource

# The schemes `open_source` understands. Exported so that anything validating a
# URI before handing it over (the dashboard's picker) cannot drift from the
# dispatch below.
SOURCE_SCHEMES = ("webcam://", "file://", "rs://", "bag://", "seq://")


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("RealSense boolean options must use 1/0, true/false, yes/no, or on/off")


def parse_realsense_uri(uri: str) -> dict:
    """Parse a ``rs://`` URI into keyword arguments for ``RealSenseSource``.

    Pure and hardware-free so it can be unit-tested without a camera or
    pyrealsense2. The RealSense uses its OWN stream profile (1920x1080 colour,
    1280x720 IR by default) to match its calibration -- it deliberately ignores
    any generic capture size a caller passes (that size is for webcams). Set the
    resolution explicitly only with ``w``/``h`` in the query string, e.g.
    ``rs://ir?w=848&h=480``.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "rs":
        raise ValueError("RealSense URI must use the rs:// scheme")
    if parsed.netloc not in ("", "ir") or parsed.path not in ("", "/"):
        raise ValueError("RealSense URI authority/path must be empty or the 'ir' shorthand")
    if parsed.fragment:
        raise ValueError("RealSense URI fragments are not supported")
    allowed = {
        "ir",
        "emitter",
        "ir_index",
        "w",
        "h",
        "depth",
        "max_laser",
        "max_range",
        "spatial",
    }
    parsed_query = parse_qs(parsed.query, keep_blank_values=True)
    unknown = sorted(set(parsed_query) - allowed)
    if unknown:
        raise ValueError("unsupported RealSense URI option(s): " + ", ".join(unknown))
    repeated = sorted(key for key, values in parsed_query.items() if len(values) != 1)
    if repeated:
        raise ValueError("repeated RealSense URI option(s): " + ", ".join(repeated))
    if ("w" in parsed_query) != ("h" in parsed_query):
        raise ValueError("RealSense URI resolution requires both w and h")
    q = {key: values[0] for key, values in parsed_query.items()}

    infrared = parsed.netloc == "ir" or (_as_bool(q["ir"]) if "ir" in q else False)

    # Emitter: explicit query wins; otherwise off for IR (clean image), and
    # left at the device default (None) for colour.
    if "emitter" in q:
        emitter = _as_bool(q["emitter"])
    else:
        emitter = False if infrared else None

    kwargs: dict = {
        "infrared": infrared,
        "ir_index": int(q.get("ir_index", "1")),
        "emitter": emitter,
        "with_depth": _as_bool(q.get("depth", "0")),
        "max_laser": _as_bool(q.get("max_laser", "0")),
    }

    if "max_range" in q:
        kwargs["max_range_m"] = float(q["max_range"])
    if "spatial" in q:
        kwargs["spatial_magnitude"] = int(q["spatial"])

    # Only an explicit w/h in the URI overrides the device's default profile.
    if "w" in q and "h" in q:
        kwargs["ir_size" if infrared else "color_size"] = (int(q["w"]), int(q["h"]))
    return kwargs


def open_source(
    uri: str,
    width: int | None = None,
    height: int | None = None,
    *,
    device_serial: str | None = None,
    strict_depth_controls: bool = False,
) -> FrameSource:
    """Open a frame source from a URI.

    `width`/`height` request a capture resolution and apply only to live
    webcam sources; recordings and datasets have a fixed native size.
    """
    from ahfd.capture.video import VideoSource

    if (device_serial is not None or strict_depth_controls) and not uri.startswith(
        "rs://"
    ):
        raise ValueError(
            "device identity and strict depth controls are valid only for a live rs:// source"
        )

    if uri.startswith("webcam://"):
        return VideoSource(int(uri[len("webcam://") :]), uri=uri, width=width, height=height)

    if uri.startswith("file://"):
        return VideoSource(uri[len("file://") :], uri=uri)

    if uri.startswith("rs://"):
        from ahfd.capture.realsense import RealSenseSource

        # width/height are webcam capture hints; the RealSense uses its own
        # calibrated profile, so they are intentionally not forwarded here.
        return RealSenseSource(
            **parse_realsense_uri(uri),
            device_serial=device_serial,
            strict_depth_controls=strict_depth_controls,
        )

    if uri.startswith("bag://"):
        from ahfd.capture.realsense import BagSource

        return BagSource(uri[len("bag://") :])

    if uri.startswith("seq://"):
        from ahfd.capture.imageseq import ImageSequenceSource

        return ImageSequenceSource(uri[len("seq://") :], uri=uri)

    # Bare path -- treat as a video file.
    return VideoSource(uri, uri=uri)
