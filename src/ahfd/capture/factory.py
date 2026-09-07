"""URI-based source construction.

Schemes:
    webcam://0          built-in or USB camera by index
    file://path.mp4     video file (a bare path works too)
    rs://               live RealSense              (not implemented yet)
    bag://path.bag      recorded RealSense          (not implemented yet)
    seq://path/         directory of dataset images (not implemented yet)
"""

from __future__ import annotations

from ahfd.capture.base import FrameSource

_UNIMPLEMENTED = (
    ("rs://", "live RealSense"),
    ("bag://", "recorded RealSense .bag"),
    ("seq://", "dataset image sequence"),
)


def open_source(uri: str) -> FrameSource:
    """Open a frame source from a URI."""
    from ahfd.capture.video import VideoSource

    if uri.startswith("webcam://"):
        return VideoSource(int(uri[len("webcam://") :]), uri=uri)

    if uri.startswith("file://"):
        return VideoSource(uri[len("file://") :], uri=uri)

    for scheme, what in _UNIMPLEMENTED:
        if uri.startswith(scheme):
            raise NotImplementedError(
                what + " source is not implemented yet (" + scheme + ")"
            )

    # Bare path -- treat as a video file.
    return VideoSource(uri, uri=uri)
