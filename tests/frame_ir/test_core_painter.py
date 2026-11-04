"""3D frames drawn on the core, with Cairo's overlay drawn over them.

The core draws a 3D scene's frame without its overlay (mobjects fixed in
frame, images, background-image fills), and Cairo draws the overlay over the
core's pixels with CE's own camera code, in CE's order. Drawn onto the core's
frame as CE draws onto its own, the overlay rounds where CE's rounds: wherever
the core and CE agree beneath it, the overlaid frame is CE's, byte for byte,
translucent members included.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.painter import IrCairoPainter

from ..timeline.conftest import render
from . import core_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


def test_the_overlay_is_ce_s_wherever_the_core_and_ce_agree_beneath_it():
    _, ce = render(core_scenes.FixedLabelOverSquare, FPS, **MEDIUM)
    _, ce_bare = render(core_scenes.FixedLabelOverSquareBare, FPS, **MEDIUM)
    before = CorePainter.drawn_total
    _, core = render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    _, core_bare = render(
        core_scenes.FixedLabelOverSquareBare, FPS, draw_path="core", **MEDIUM
    )
    assert CorePainter.drawn_total > before, "drawn on the core"
    assert len(ce) == len(core) == len(ce_bare) == len(core_bare) > 0
    compared = 0
    for index, frames in enumerate(zip(ce, core, ce_bare, core_bare, strict=True)):
        a, b, a_bare, b_bare = frames
        agree = np.all(a_bare == b_bare, axis=-1)
        overlaid = np.any(a != a_bare, axis=-1)
        np.testing.assert_array_equal(b[agree], a[agree], err_msg=f"frame {index}")
        compared += int((agree & overlaid).sum())
    # The overlay's own pixels over agreeing content, translucent ones among
    # them, not a handful at its edge.
    assert compared > 1000, compared


def test_the_overlay_is_drawn_over_every_frame_that_has_one(monkeypatch):
    painted: list[CorePainter] = []
    held = CorePainter.__init__

    def kept(self, **kwargs):
        held(self, **kwargs)
        painted.append(self)

    monkeypatch.setattr(CorePainter, "__init__", kept)
    render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    assert sum(p.frames for p in painted) > 0
    assert sum(p.overlaid for p in painted) == sum(p.frames for p in painted)


def test_auto_draws_a_3d_scene_on_the_core():
    before = CorePainter.drawn_total
    render(core_scenes.FixedLabelOverSquare, FPS, draw_path="auto", **MEDIUM)
    assert CorePainter.drawn_total > before


def test_auto_draws_a_2d_scene_as_cairo_does_without_xf():
    _, direct = render(core_scenes.FlatSquares, FPS, **MEDIUM)
    before = (CorePainter.drawn_total, IrCairoPainter.drawn_total)
    _, auto = render(core_scenes.FlatSquares, FPS, draw_path="auto", **MEDIUM)
    assert (CorePainter.drawn_total, IrCairoPainter.drawn_total) == before
    assert len(direct) == len(auto) > 0
    for index, (a, b) in enumerate(zip(direct, auto, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
