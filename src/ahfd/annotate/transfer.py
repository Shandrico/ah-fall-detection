"""Transfer offline labels from one camera's clip to the other camera's clip.

Both RealSense recordings of the same event carry colour-frame timestamps in the
*global-time* domain -- the same host clock -- so a label at clip-relative time
``t`` in camera A maps to camera B by a single constant offset: the difference
between the two clips' first-frame global timestamps. No per-frame alignment or
motion cross-correlation is needed, because global time removes any relative
clock drift between the two cameras. The pure shift lives here (unit-testable,
camera-free); the CLI command reads the two bags' first-frame stamps.
"""

from __future__ import annotations


def shift_segments(segments, delta_s, duration_s, *, key):
    """Shift ``segments`` by ``delta_s`` seconds, clipped to ``[0, duration_s]``.

    ``delta_s`` is ``t0_src - t0_dst`` (the source clip's first-frame time minus
    the target clip's), so ``t_dst = t_src + delta_s``. A segment that lands
    wholly outside the target clip is dropped; one straddling an edge is clipped
    to it. Returns ``(shifted, dropped, clipped)`` for reporting.
    """
    shifted, dropped, clipped = [], [], []
    for s in segments:
        a = round(s["start_s"] + delta_s, 2)
        b = round(s["end_s"] + delta_s, 2)
        lab = s.get(key)
        if b <= 0.0 or a >= duration_s:
            dropped.append((lab, s["start_s"], s["end_s"]))
            continue
        na, nb = max(0.0, a), min(duration_s, b)
        if (na, nb) != (a, b):
            clipped.append((lab, a, b, na, nb))
        shifted.append({"start_s": round(na, 2), "end_s": round(nb, 2), key: lab})
    return shifted, dropped, clipped
