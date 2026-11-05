"""Geometry at the snap's conditioning edge draws no further from Cairo snapped
than from its points as made.

Each stress scene (``stress_scenes``) derives a direction or a decision from
points a grid step apart or closer: handles on or near their anchors under
thick joins, near-degenerate cubics, tiny circles and dots, a sphere's poles,
a cone's apex. Snapped, its pixels more than 24 levels off Cairo's frame are
no more than unsnapped, up to the shared edges whose antialiasing order the
snap settles (2%, and a few pixels). A quantity the snap broke (a normal, a
plane, a tangent) moves thousands.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.core_painter import CorePainter

from ..timeline.conftest import render
from .stress_scenes import FPS, SCENES

pytest.importorskip("arachne_core")

FULL = {"pixel_width": 854, "pixel_height": 480}


def _far_from(frame: np.ndarray, cairo: np.ndarray) -> int:
    d = np.abs(frame[..., :3].astype(int) - cairo[..., :3].astype(int)).max(-1)
    return int((d > 24).sum())


@pytest.mark.parametrize("scene", SCENES, ids=[s.__name__ for s in SCENES])
def test_snapped_is_no_further_from_cairo_than_unsnapped(scene, monkeypatch):
    _, cairo = render(scene, FPS, draw_path="ir_cairo", **FULL)
    _, snapped = render(scene, FPS, draw_path="core", **FULL)
    # The core handed the deltas as made: the reference, past the guard.
    monkeypatch.setattr(Canonical, "__call__", lambda self, delta: delta)
    monkeypatch.setattr(CorePainter, "_refuse_undecided", lambda self, sent: None)
    _, unsnapped = render(scene, FPS, draw_path="core", **FULL)
    far = _far_from(snapped[-1], cairo[-1]), _far_from(unsnapped[-1], cairo[-1])
    assert far[0] <= far[1] * 1.02 + 4, far
