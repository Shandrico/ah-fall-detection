"""Debugging tools that are allowed to touch imagery.

This is the ONLY package permitted to write pixels to disk. The AST privacy
test (tests/test_privacy.py) scans every other module and fails the build if an
image write appears outside here, and the import-isolation this creates is
deliberate: the exemption is narrow, visible, and in one place.

Everything in here is gated at runtime by ahfd.privacy's triple switch, so
importing it is not the same as being allowed to use it.
"""

from ahfd.debug.raw_writer import RawRecorder

__all__ = ["RawRecorder"]
