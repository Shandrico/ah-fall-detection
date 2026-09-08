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


def open_source(uri: str) -> FrameSource:
    """Open a frame source from a URI."""
    from ahfd.capture.video import VideoSource

    if uri.startswith("webcam://"):
        return VideoSource(int(uri[len("webcam://") :]), uri=uri)

    if uri.startswith("file://"):
        return VideoSource(uri[len("file://") :], uri=uri)

    if uri.startswith("rs://"):
        from ahfd.capture.realsense import RealSenseSource

        return RealSenseSource()

    if uri.startswith("bag://"):
        from ahfd.capture.realsense import BagSource

        return BagSource(uri[len("bag://") :])

    if uri.startswith("seq://"):
        raise NotImplementedError(
            "dataset image sequence source is not implemented yet (seq://)"
        )

    # Bare path -- treat as a video file.
    return VideoSource(uri, uri=uri)
