"""Frame sources. The only place in the codebase where imagery lives."""

from ahfd.capture.base import Frame, FrameSource, SourceMeta
from ahfd.capture.factory import open_source

__all__ = ["Frame", "FrameSource", "SourceMeta", "open_source"]
