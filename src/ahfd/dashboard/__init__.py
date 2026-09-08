"""Nurse-facing web dashboard.

Standard-library server, one pipeline thread, frames encoded once and shared --
see server.py and state.py for why it is built this way rather than on the
async stack that overheated the reference implementation.

RGB display is opt-in (`dashboard.show_rgb`); the default is skeleton-only,
which keeps the ward privacy stance. Showing RGB is a decision for AH/DPO, not
a default.
"""

from ahfd.dashboard.runner import PipelineRunner
from ahfd.dashboard.server import DashboardServer
from ahfd.dashboard.state import DashboardState

__all__ = ["DashboardState", "DashboardServer", "PipelineRunner"]
