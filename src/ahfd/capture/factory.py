"""URI-based source construction.

Schemes:
    webcam://0          built-in or USB camera by index
    file://path.mp4     video file (a bare path works too)
    rs://               live RealSense D435i
    bag://path.bag      recorded RealSense playback (deterministic)
    seq://path/         directory of dataset images (not implemented yet)
"""

from __future__ import annotations

from ahfd.capture.base import FrameSource

# The schemes `open_source` understands. Exported so that anything validating a
# URI before handing it over (the dashboard's picker) cannot drift from the
# dispatch below.
SOURCE_SCHEMES = ("webcam://", "file://", "rs://", "bag://", "seq://")


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

        return RealSenseSource()

    if uri.startswith("bag://"):
        from ahfd.capture.realsense import BagSource

        return BagSource(uri[len("bag://") :])

    if uri.startswith("seq://"):
        from ahfd.capture.imageseq import ImageSequenceSource

        return ImageSequenceSource(uri[len("seq://") :], uri=uri)

    # Bare path -- treat as a video file.
    return VideoSource(uri, uri=uri)
