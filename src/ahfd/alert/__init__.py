"""Where events go.

Kept deliberately thin. This is the seam a later team would extend to reach a
nurse call system, a pager, or the teleconsultation side of the Alexandra
Hospital brief -- none of which is in scope here. Anything added must accept
an `Event` and nothing else, so imagery cannot leak through an alert path.
"""

from ahfd.alert.sinks import AlertSink, ConsoleSink, JsonlSink, MultiSink

__all__ = ["AlertSink", "ConsoleSink", "JsonlSink", "MultiSink"]
