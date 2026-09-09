"""What cameras are actually attached.

Enumeration is a report, not a side effect: this returns data so that the
`ahfd info` printout and the dashboard's camera picker are driven by the same
answer. pyrealsense2 is an optional dependency, so it is imported inside the
function and its absence is a result, not an exception.

Deliberately does NOT probe webcam indices. Opening `cv2.VideoCapture(1)` to
see whether it exists is slow on Windows and can grab a device another program
is using -- including the pipeline that is about to open it. Webcam entries
come from the config instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealSenseDevice:
    name: str
    serial: str = ""
    usb: str = ""  # "3.2" / "2.1", or "" if the device would not say

    @property
    def usb2(self) -> bool:
        """A D435i on a USB 2 link silently loses stream profiles."""
        return self.usb.startswith("2")


@dataclass(frozen=True)
class RealSenseProbe:
    installed: bool
    version: str | None = None
    devices: tuple[RealSenseDevice, ...] = ()
    error: str | None = None  # driver/permission failure during enumeration


def _device_info(rs, dev) -> RealSenseDevice:
    """Read one device's fields. Not all of them are always present."""

    def get(field_name: str) -> str:
        try:
            return str(dev.get_info(getattr(rs.camera_info, field_name)))
        except Exception:  # noqa: BLE001 -- a missing field is not a failure
            return ""

    return RealSenseDevice(
        name=get("name") or "RealSense", serial=get("serial_number"), usb=get("usb_type_descriptor")
    )


def probe_realsense() -> RealSenseProbe:
    """Enumerate connected RealSense cameras. Never raises.

    A half-installed librealsense throws on `query_devices()`; the dashboard
    must not 500 because of it, so the failure comes back as `error`.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return RealSenseProbe(installed=False)

    version = str(getattr(rs, "__version__", "(unknown)"))
    try:
        devices = tuple(_device_info(rs, d) for d in rs.context().query_devices())
    except Exception as exc:  # noqa: BLE001 -- report it, do not propagate
        return RealSenseProbe(installed=True, version=version, error=str(exc))

    return RealSenseProbe(installed=True, version=version, devices=devices)
