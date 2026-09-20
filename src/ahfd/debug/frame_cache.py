"""Persist a classify-view preprocessing pass so re-opening a clip is instant.

Lives in ``ahfd.debug`` because it writes rendered frames (imagery) to disk,
which the privacy guard permits only in this package. This is a convenience for
staged dev recordings, NOT for a live ward: the cache holds colour frames, so it
is gitignored and meant to be deleted alongside the ``.db3`` it came from.
"""

from __future__ import annotations

import pickle
from pathlib import Path

_ROOT = "data/classify_cache"


def cache_path(stem: str, root: str = _ROOT) -> Path:
    return Path(root) / (stem + ".pkl")


def save_cache(stem: str, payload, root: str = _ROOT) -> Path:
    """Pickle the preprocessed run for ``stem`` (creates the dir)."""
    p = cache_path(stem, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_cache(stem: str, root: str = _ROOT):
    """Return the pickled run for ``stem``, or None if absent/unreadable."""
    p = cache_path(stem, root)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:  # a truncated/old cache -> just rebuild
        return None
