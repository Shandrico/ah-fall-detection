"""Privacy-safe, derived-only onsite data collection.

The collection package deliberately has no dependency on ``capture.Frame``.
It persists JSON-safe measurements and operational telemetry, never pixels or
dense depth.  Raw capture remains a separate, explicitly gated debug workflow.
"""

from ahfd.collect.health import (
    HealthMonitor,
    HealthReason,
    HealthStatus,
    HealthTransition,
)
from ahfd.collect.session import SessionRecorder, new_pseudonymous_id, sha256_file

__all__ = [
    "HealthMonitor",
    "HealthReason",
    "HealthStatus",
    "HealthTransition",
    "SessionRecorder",
    "new_pseudonymous_id",
    "sha256_file",
]
