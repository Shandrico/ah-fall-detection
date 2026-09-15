"""URI-based source construction.

Schemes:
    webcam://0          built-in or USB camera by index
    file://path.mp4     video file (a bare path works too)
    rs://               live RealSense D435i (colour)
    rs://ir             live RealSense in infrared (low-light night vision)
    bag://path.bag      recorded RealSense playback (deterministic)
    seq://path/         directory of dataset images (not implemented yet)

The RealSense URI takes options as a query string, e.g.
``rs://ir?emitter=0&ir_index=1&w=1280&h=720``:

* ``ir`` (or the ``rs://ir`` shorthand) streams the left IR imager instead of
  colour -- a grayscale image for low light.
* ``emitter`` toggles the dot projector (``0`` = off). It defaults OFF in IR
  mode (the projected dots otherwise cover the scene; pair with an external IR
  floodlight so the room is lit) and is left at the device default otherwise.
* ``w`` / ``h`` set the stream resolution.
"""

from __future__ import annotations

from urllib.parse import urlparse

from ahfd.capture.base import FrameSource

# The schemes `open_source` understands. Exported so that anything validating a
# URI before handing it over (the dashboard's picker) cannot drift from the
# dispatch below.
SOURCE_SCHEMES = ("webcam://", "file://", "rs://", "bag://", "seq://")


def _as_bool(value: str) -> bool:
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def parse_realsense_uri(
    uri: str, width: int | None = None, height: int | None = None
) -> dict:
    """Parse a ``rs://`` URI into keyword arguments for ``RealSenseSource``.

    Pure and hardware-free so it can be unit-tested without a camera or
    pyrealsense2. ``width``/``height`` are a fallback for the resolution when
    the query string does not carry ``w``/``h``.
    """
    from urllib.parse import parse_qs

    parsed = urlparse(uri)
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}

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
    }

    w = int(q["w"]) if "w" in q else width
    h = int(q["h"]) if "h" in q else height
    if w and h:
        kwargs["ir_size" if infrared else "color_size"] = (w, h)
    return kwargs


def open_source(
    uri: str, width: int | None = None, height: int | None = None
) -> FrameSource:
    """Open a frame source from a URI.

    `width`/`height` request a capture resolution and apply only to live
    webcam sources; recordings and datasets have a fixed native size.
    """
    from ahfd.capture.video import VideoSource

    if uri.startswith("webcam://"):
        return VideoSource(int(uri[len("webcam://") :]), uri=uri, width=width, height=height)

    if uri.startswith("file://"):
        return VideoSource(uri[len("file://") :], uri=uri)

    if uri.startswith("rs://"):
        from ahfd.capture.realsense import RealSenseSource

        return RealSenseSource(**parse_realsense_uri(uri, width, height))

    if uri.startswith("bag://"):
        from ahfd.capture.realsense import BagSource

        return BagSource(uri[len("bag://") :])

    if uri.startswith("seq://"):
        from ahfd.capture.imageseq import ImageSequenceSource

        return ImageSequenceSource(uri[len("seq://") :], uri=uri)

    # Bare path -- treat as a video file.
    return VideoSource(uri, uri=uri)
