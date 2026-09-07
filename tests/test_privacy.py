"""Privacy enforcement tests.

This is the file to show a hospital's data-protection officer. It converts
"we do not persist video" from a promise into something the build checks.

The AST walk is deliberate rather than a regex: a regex over source text is
fooled by the word appearing in a comment or a docstring, and would also miss
nothing useful in exchange. Walking the syntax tree matches actual calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ahfd"

# Calls that put pixels somewhere permanent.
FORBIDDEN_CALLS = {
    "imwrite",
    "VideoWriter",
    "imsave",
    "recorder",
}

# The one package allowed to do it, gated by ahfd.privacy at runtime.
ALLOWED_PACKAGE = "debug"


def _module_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if ALLOWED_PACKAGE not in p.parts)


def _called_names(tree: ast.AST):
    """Yield (name, lineno) for every call in the tree, by its final attribute."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            yield func.attr, node.lineno
        elif isinstance(func, ast.Name):
            yield func.id, node.lineno


def test_source_tree_is_not_empty():
    """Guard the guard: an empty glob would make every test below vacuous."""
    files = _module_files()
    assert len(files) >= 5, "expected to find modules to scan, found " + str(files)


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_image_writing_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        (name, line) for name, line in _called_names(tree) if name in FORBIDDEN_CALLS
    ]
    assert not offenders, (
        str(path)
        + " writes imagery outside ahfd."
        + ALLOWED_PACKAGE
        + ": "
        + repr(offenders)
    )


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_binary_file_writes(path: Path):
    """Catch open(..., 'wb') too, which would sidestep the call blacklist."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        if not is_open:
            continue
        modes = [
            a.value
            for a in node.args[1:2]
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        modes += [
            kw.value.value
            for kw in node.keywords
            if kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ]
        if any("b" in m and ("w" in m or "a" in m or "x" in m) for m in modes):
            offenders.append(node.lineno)

    assert not offenders, str(path) + " opens a binary file for writing: " + repr(offenders)


def test_downstream_types_hold_no_image():
    """PoseFrame / PersonPose must have nowhere to put pixels."""
    from ahfd.types import PersonPose, PoseFrame

    banned = {"bgr", "rgb", "image", "img", "frame", "depth", "pixels"}
    for cls in (PersonPose, PoseFrame):
        fields = set(cls.__dataclass_fields__)
        leaked = fields & banned
        assert not leaked, cls.__name__ + " exposes image-ish fields: " + repr(leaked)
