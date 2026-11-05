"""The overlay's runs made again from the frame before's Cairo calls.

On the core, Cairo draws the overlay over the core's frame with CE's camera
code. A run the camera draws through its context that is the same as in the
frame before is not drawn again: the calls the camera made on its context
then are made on the new frame's. Every frame must be the frame drawn
without that reuse, byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.replay import Recording, replay

from ..timeline.conftest import render
from . import core_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


@pytest.fixture
def painted(monkeypatch):
    """The core painters the render made."""
    made: list[CorePainter] = []
    held = CorePainter.__init__

    def kept(self, **kwargs):
        held(self, **kwargs)
        made.append(self)

    monkeypatch.setattr(CorePainter, "__init__", kept)
    return made


def _surface(array):
    import cairo

    surface = cairo.ImageSurface.create_for_data(
        array, cairo.FORMAT_ARGB32, array.shape[1], array.shape[0]
    )
    return cairo.Context(surface)


def test_calls_made_again_draw_what_they_drew():
    """Over a translucent fill, a gradient and a stroke on pixels beneath
    them, the calls a recording kept draw the same bytes on another context.
    """
    import cairo

    rng = np.random.default_rng(3)
    beneath = rng.integers(0, 256, size=(40, 60, 4), dtype=np.uint8)
    drawn, again = np.array(beneath), np.array(beneath)
    recording = Recording(_surface(drawn))
    recording.new_path()
    recording.move_to(5.5, 5.25)
    recording.curve_to(20.0, 0.0, 40.0, 35.0, 55.0, 30.5)
    recording.close_path()
    recording.set_source_rgba(0.2, 0.6, 0.9, 0.45)
    recording.fill_preserve()
    gradient = cairo.LinearGradient(0.0, 0.0, 60.0, 40.0)
    gradient.add_color_stop_rgba(0.0, 1.0, 0.0, 0.0, 0.7)
    gradient.add_color_stop_rgba(1.0, 0.0, 0.0, 1.0, 0.3)
    recording.set_source(gradient)
    recording.set_line_width(3.5)
    recording.set_line_join(cairo.LINE_JOIN_ROUND)
    recording.set_line_cap(cairo.LINE_CAP_ROUND)
    recording.stroke_preserve()
    assert recording.begins_path()
    replay(_surface(again), recording.calls)
    np.testing.assert_array_equal(again, drawn)
    assert not np.array_equal(drawn, beneath)
    later = Recording(_surface(drawn))
    later.set_source_rgba(1.0, 1.0, 1.0, 1.0)
    later.fill_preserve()
    assert not later.begins_path()


def _both(scene_class, monkeypatch, **size):
    _, reused = render(scene_class, FPS, draw_path="core", **size)
    monkeypatch.setattr(CorePainter, "replay_overlay", False)
    _, drawn = render(scene_class, FPS, draw_path="core", **size)
    return reused, drawn


def _replayed(painted: list[CorePainter]) -> int:
    return sum(p.replays.replayed for p in painted if p.replays is not None)


@pytest.mark.parametrize(
    ("scene_class", "made_again"),
    [
        (core_scenes.FixedLabelOverSquare, True),
        # The label after the image in space, which changes every frame.
        (core_scenes.ImageInSpaceOrbit, True),
        (core_scenes.InterleavedOverlay, True),
        (core_scenes.BackgroundFilledOverlay, True),
        # A 2D scene: its overlay is the display alone, drawn every frame.
        (core_scenes.ZoomedFrameMoves, False),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_each_frame_with_its_overlay_made_again_is_the_frame_drawn_anew(
    scene_class, made_again, painted, monkeypatch
):
    reused, drawn = _both(scene_class, monkeypatch, **MEDIUM)
    assert len(reused) == len(drawn) > 0
    for index, (a, b) in enumerate(zip(reused, drawn, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    assert (_replayed(painted) > 0) is made_again


def test_glyphs_strokes_and_gradients_are_made_again_byte_for_byte_at_full_size(
    painted, monkeypatch
):
    """At 1080p, where antialiasing and fill rules show in more pixels."""
    reused, drawn = _both(
        core_scenes.InterleavedOverlay,
        monkeypatch,
        pixel_width=1920,
        pixel_height=1080,
    )
    for index, (a, b) in enumerate(zip(reused, drawn, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    assert _replayed(painted) > 0


def test_an_overlay_fixed_in_frame_is_drawn_once_over_an_orbit(painted):
    """Only the rotation changes from frame to frame, and nothing fixed in
    frame reads it: each painter draws the overlay's members on its first
    frame and makes every one again on every later frame.
    """
    render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    overlaid = [p for p in painted if p.overlaid]
    assert sum(p.overlaid for p in overlaid) > 2
    for painter in overlaid:
        replays = painter.replays
        assert replays is not None
        # The label's glyphs and the bar.
        assert replays.recorded > 1
        assert replays.replayed == replays.recorded * (painter.overlaid - 1)
