"""Rendering.

`render_skeleton` consumes coordinates only (never imagery) and is the
privacy-safe default. `render_overlay` draws on the RGB frame for a development
view and the dashboard's opt-in RGB mode -- it displays pixels but never
persists them.
"""

from ahfd.viz.overlay import render_overlay
from ahfd.viz.skeleton_render import draw_people, render_skeleton

__all__ = ["render_skeleton", "render_overlay", "draw_people"]
