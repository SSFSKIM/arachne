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


def test_a_painter_holds_none_of_the_frames_it_drew_its_overlay_over():
    """Cairo draws each frame's overlay through a context on that frame, and a
    context holds its frame: kept for the frames after, every frame of a render
    would stay in memory to its end (about 8 MB each at 1080p).
    """
    import gc
    import weakref
    from fractions import Fraction

    from arachne import Rectangle, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        square = Square(fill_opacity=1.0)
        bar = Rectangle(width=3, height=0.4, fill_opacity=0.5)
        camera.add_fixed_in_frame_mobjects(bar)
        painter = CorePainter()
        drawn = []
        for index in range(6):
            square.shift([0.1, 0.0, 0.0])
            frame = painter.paint(
                camera, [("moving", [square, bar])], t=Fraction(index, FPS)
            )
            drawn.append(weakref.ref(frame))
            del frame
        assert painter.overlaid == len(drawn)
        gc.collect()
        held = sum(ref() is not None for ref in drawn)
    # The last frame stays the drawing camera's picture until the next.
    assert held <= 1, f"{held} of {len(drawn)} frames held"


def test_a_live_frame_tests_depth_across_its_static_and_moving_captures():
    """Drawn as the scene runs, a frame is two captures, its static list and
    then its moving one, as CE draws it. The core describes the two as one
    frame and tests depth across both, as it does a frame drawn from the
    record: the square hides the stroke under it on either route, and the
    frames are the same, byte for byte.
    """
    from arachne.frame_ir.live import live_painter

    _, cairo = render(core_scenes.StrokeUnderSquare, FPS, **MEDIUM)
    _, recorded = render(core_scenes.StrokeUnderSquare, FPS, draw_path="core", **MEDIUM)
    scene, live = render(
        core_scenes.LiveStrokeUnderSquare, FPS, draw_path="core", **MEDIUM
    )
    found = live_painter(scene.renderer.camera)  # type: ignore[attr-defined]
    assert found is not None
    assert found.captures > 0, "drawn as the scene runs"
    assert len(live) == len(recorded) == len(cairo) > 0
    # Depth is at work: CE draws the stroke over the square.
    assert np.any(recorded[0] != cairo[0])
    for index, (a, b) in enumerate(zip(live, recorded, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


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


def _inside(frame: np.ndarray, margin: int = 2) -> np.ndarray:
    """Pixels at least ``margin`` px from any edge in ``frame``: where no
    neighbour differs from a pixel by more than a gradient's step, eroded
    by ``margin``.
    """
    rgb = frame[..., :3].astype(int)
    smooth = np.ones(rgb.shape[:2], bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        step = np.abs(rgb - np.roll(rgb, (dy, dx), axis=(0, 1))).max(axis=-1) <= 3
        smooth &= step & np.roll(step, (-dy, -dx), axis=(0, 1))
    inside = smooth.copy()
    for _ in range(margin):
        inside &= np.roll(inside, 1, 0) & np.roll(inside, -1, 0)
        inside &= np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    return inside & np.any(rgb > 0, axis=-1)


def test_a_lit_face_is_ce_s_colours_inside_it():
    """Each face of a shaded prism and surface is lit as CE lights it (two
    stops, a linear gradient), coloured as Cairo colours it: inside every
    face, away from its edges, the core's pixels are CE's.
    """
    size = {"pixel_width": 480, "pixel_height": 270}
    _, ce = render(core_scenes.ShadedSolids, FPS, **size)
    _, core = render(core_scenes.ShadedSolids, FPS, draw_path="core", **size)
    assert len(ce) == len(core) > 0
    compared = 0
    for index, (a, b) in enumerate(zip(ce, core, strict=True)):
        inside = _inside(a)
        np.testing.assert_array_equal(b[inside], a[inside], err_msg=f"frame {index}")
        compared += int(inside.sum())
    # Hundreds of pixels inside the faces of every frame, not a handful.
    assert compared > 500 * len(ce), compared
