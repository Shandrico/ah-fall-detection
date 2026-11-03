"""Recent alert clips for the nurse page's "Review alerts" section.

Each alert saves a short clip (``ahfd.debug.alert_clips``): ``<stem>.mp4`` plus
``<stem>.json``. Only clips listed here are ever served: a name from the URL is
looked up in this list, never joined onto a path. Clips holding the camera image
are listed only when RGB is authorised.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


def list_alert_clips(root, *, allow_rgb: bool, hours: float = 24.0, now: float | None = None) -> list[dict]:
    """Clips from the last ``hours``, newest first."""
    if not root:
        return []
    root = Path(root)
    now = time.time() if now is None else now
    out = []
    try:
        metas = sorted(root.glob("*.json"))
    except OSError:
        return []
    for meta in metas:
        try:
            info = json.loads(meta.read_text())
            video = root / str(info["video"])
            started = datetime.fromisoformat(str(info["started"])).timestamp()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if video.parent != root or video.suffix != ".mp4" or not video.is_file():
            continue
        rgb = info.get("picture") != "stick figures only"
        if (rgb and not allow_rgb) or now - started > hours * 3600:
            continue
        alerts = info.get("alerts") or []
        first = alerts[0] if alerts else {}
        out.append({
            "name": video.name,
            "started": started,
            "clock": first.get("clock") or datetime.fromtimestamp(started).strftime("%H:%M:%S"),
            "types": list(dict.fromkeys(str(a.get("type")) for a in alerts)),
            "bed": first.get("zone") or first.get("near_bed"),
            "duration_s": info.get("duration_s"),
            "seek_s": max(0.0, float(info.get("seconds_before_first_alert") or 0.0) - 3.0),
            "picture": "camera image" if rgb else "stick figures",
        })
    out.sort(key=lambda c: c["started"], reverse=True)
    return out


def find_alert_clip(root, name: str, *, allow_rgb: bool) -> Path | None:
    for clip in list_alert_clips(root, allow_rgb=allow_rgb):
        if clip["name"] == name:
            return Path(root) / clip["name"]
    return None
