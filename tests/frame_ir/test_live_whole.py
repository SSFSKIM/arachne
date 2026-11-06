"""A live painter on the core reports the leaves of a capture it records as
whole: begun on its camera's own background as the camera's reset left it,
nothing of an earlier picture composited under it. A capture drawn over an
earlier picture's pixels still refuses (``LiveFrame``): its leaves are
flattened into its background.

Frames here are 192 x 108 pixels over CE's 8-unit frame height: 13.5 pixels a
unit, the origin at pixel (96, 54).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne.frame_ir.contributors import LiveFrame
from arachne.frame_ir.live import live, live_painter

from ..timeline.conftest import render
from . import camera_scenes

pytest.importorskip("arachne_core")

MEDIUM = {"pixel_width": 192, "pixel_height": 108}
CENTRE = (96, 54)


def _squares():
    from arachne import BLUE, RED, Square

    under = Square(side_length=2, fill_color=RED, fill_opacity=1.0, stroke_width=0)
    top = Square(side_length=1, fill_color=BLUE, fill_opacity=0.5, stroke_width=0)
    return under, top


def _captured(*captures, reset: bool = True):
    """The core's live painter of a 3D camera after one frame of
    ``captures`` (each a list of mobjects the camera captures in turn), the
    camera reset first unless ``reset`` is false.
    """
    from arachne import ThreeDCamera, tempconfig

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        with live(camera, Fraction(0), draw_path="core"):
            if reset:
                camera.reset()
            for mobjects in captures:
                camera.capture_mobjects(mobjects)
        found = live_painter(camera)
        assert found is not None
        return found.painter(camera)


def test_a_capture_begun_on_the_camera_s_reset_reports_its_leaves():
    under, top = _squares()
    painter = _captured([under, top])
    assert painter.live
    assert painter.whole
    (pixel,) = painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1)).values()
    assert len(pixel.leaves) == 2
    assert all(leaf.shown for leaf in pixel.leaves)
    assert len(painter.outlines()) == 2


def test_a_capture_continuing_one_whole_picture_is_whole():
    """A capture drawing other leaves over the frame's capture before it
    continues that capture's description, from its background: the frame
    is one picture, as a play's frame over its static picture is.
    """
    under, top = _squares()
    painter = _captured([under], [top])
    assert painter.whole
    (pixel,) = painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1)).values()
    assert len(pixel.leaves) == 2


def test_a_capture_drawn_over_an_earlier_picture_refuses():
    """The second capture draws a leaf of the first again: it is drawn over
    the first's pixels, which hold that leaf flattened.
    """
    under, top = _squares()
    painter = _captured([under], [under, top])
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))
    with pytest.raises(LiveFrame):
        painter.outlines()


def test_a_capture_without_a_reset_in_its_frame_refuses():
    """Whatever the camera's array holds, a capture not begun on its reset
    is not known whole.
    """
    under, top = _squares()
    painter = _captured([under, top], reset=False)
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))


def test_a_camera_of_the_scene_s_own_reports_its_last_whole_frame():
    """A camera class of the scene's own is described only as it runs, its
    records pre-projected: the last frame, a play's static picture
    continued or a capture of its own over a reset, is whole.
    """
    scene, _ = render(camera_scenes.ShiftedSolids, 15, draw_path="core", **MEDIUM)
    camera = scene.renderer.camera  # type: ignore[attr-defined]
    found = live_painter(camera)
    assert found is not None
    painter = found.painter(camera)
    assert painter.live
    assert painter.whole
    records = painter.state.records
    assert records
    assert all("pre_projected" in r.fields for r in records.values())
    report = painter.contributors((0, 0, 192, 108))
    assert report
    assert painter.outlines()


# What writes the camera's array outside a capture.


def _frame(*steps):
    """The core's live painter of a 3D camera after one frame of ``steps``,
    each a callable handed the camera, and the frame's pixels.
    """
    from arachne import ThreeDCamera, tempconfig

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        with live(camera, Fraction(0), draw_path="core"):
            for step in steps:
                step(camera)
        found = live_painter(camera)
        assert found is not None
        return found.painter(camera), camera.pixel_array.copy()


def _reset(camera):
    camera.reset()


def _capture(*mobjects):
    return lambda camera: camera.capture_mobjects(list(mobjects))


def _earlier_picture(camera):
    """An opaque patch about the frame's centre, as an earlier frame left it."""
    picture = np.zeros_like(camera.pixel_array)
    picture[44:64, 86:106] = (255, 0, 0, 255)
    return picture


def _composite(name):
    """A step that composites into the camera's array through its method
    ``name``, outside any capture.
    """
    from PIL import Image

    from arachne import RED, ImageMobject, PMobject, Square

    def step(camera):
        array = camera.pixel_array
        if name == "overlay_rgba_array":
            camera.overlay_rgba_array(array, _earlier_picture(camera))
        elif name == "overlay_PIL_image":
            camera.overlay_PIL_image(array, Image.fromarray(_earlier_picture(camera)))
        elif name == "display_multiple_vectorized_mobjects":
            camera.display_multiple_vectorized_mobjects([Square(fill_opacity=1)], array)
        elif name == "display_multiple_point_cloud_mobjects":
            cloud = PMobject(stroke_width=8).add_points([[0.0, 0.0, 0.0]], color=RED)
            camera.display_multiple_point_cloud_mobjects([cloud], array)
        elif name == "display_multiple_image_mobjects":
            image = ImageMobject(_earlier_picture(camera)).scale(0.5)
            camera.display_multiple_image_mobjects([image], array)
        else:
            raise AssertionError(name)

    return step


@pytest.mark.parametrize(
    "name",
    [
        "overlay_rgba_array",
        "overlay_PIL_image",
        "display_multiple_vectorized_mobjects",
        "display_multiple_point_cloud_mobjects",
        "display_multiple_image_mobjects",
    ],
)
def test_a_picture_composited_after_the_reset_refuses(name):
    """Something composited into the array the reset left, outside any
    capture: the capture is drawn over it, so it is not whole, whatever
    the array then holds (a path drawn through the core's context writes
    nothing: the record says what wrote, not the pixels).
    """
    _, top = _squares()
    painter, _ = _frame(_reset, _composite(name), _capture(top))
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))
    with pytest.raises(LiveFrame):
        painter.outlines()


def test_a_picture_composited_between_captures_refuses():
    """The capture after it continues the frame's capture before it, but
    what was composited between them is no capture's: not whole.
    """
    under, top = _squares()
    painter, _ = _frame(
        _reset, _capture(under), _composite("overlay_rgba_array"), _capture(top)
    )
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))


def test_a_reset_after_compositing_begins_a_whole_capture():
    _, top = _squares()
    painter, _ = _frame(_reset, _composite("overlay_rgba_array"), _reset, _capture(top))
    assert painter.whole


def _written(value):
    """A step that writes ``value(camera)`` into a patch of the camera's
    array directly, as a scene's code can, past every method.
    """

    def step(camera):
        camera.pixel_array[10:20, 10:20] = value(camera)

    return step


def test_a_write_into_the_array_after_the_reset_refuses():
    """No method hears it, but the capture does not begin on the background
    as the reset installed it.
    """
    _, top = _squares()
    painter, _ = _frame(_reset, _written(lambda _: (255, 0, 0, 255)), _capture(top))
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))


def test_a_write_of_the_background_s_own_values_keeps_whole():
    """The array still holds the background as the reset installed it:
    nothing is composited, and the record says reset.
    """
    _, top = _squares()
    patch = _written(lambda camera: np.asarray(camera.background)[10:20, 10:20])
    painter, _ = _frame(_reset, patch, _capture(top))
    assert painter.whole


def test_a_write_between_captures_never_reaches_the_frame():
    """A capture continuing the frame's capture before it is described
    from that capture's record, over the reset's background: a write into
    the array between them is not drawn under it, so the frame stays whole
    as drawn.
    """
    under, top = _squares()
    painter, written = _frame(
        _reset,
        _capture(under),
        _written(lambda _: (0, 255, 0, 255)),
        _capture(top),
    )
    _, plain = _frame(_reset, _capture(under), _capture(top))
    assert painter.whole
    assert np.array_equal(written, plain)


def test_a_camera_whose_display_code_paints_has_no_core_painter():
    """A camera class of the scene's own whose display method writes a
    pixel of its own: its draw reads beyond what XF carries, so the core
    never draws it (``painter.camera_path``), and no live core painter
    records a frame of it.
    """
    from arachne import ThreeDCamera, tempconfig

    class Painting(ThreeDCamera):
        def display_multiple_vectorized_mobjects(self, vmobjects, pixel_array):
            super().display_multiple_vectorized_mobjects(vmobjects, pixel_array)
            pixel_array[0, 0] = (255, 255, 255, 255)

    _, top = _squares()
    with tempconfig(MEDIUM):
        camera = Painting()
        with live(camera, Fraction(0), draw_path="core") as painter:
            camera.reset()
            camera.capture_mobjects([top])
        assert painter is None
        assert live_painter(camera) is None
        assert tuple(camera.pixel_array[0, 0]) == (255, 255, 255, 255)


# A held play's static picture.


def _begin_held_play(camera, static):
    """``LiveFrames.begin`` of a held play whose static picture draws
    ``static``, with the static layer on.
    """
    from types import SimpleNamespace

    from arachne.timeline.draw import LiveFrames

    scene = SimpleNamespace(renderer=SimpleNamespace(camera=camera))
    frames = LiveFrames(scene, Fraction(15), lambda *_: None, static_layer=True)
    play = SimpleNamespace(index=0, held=True, static=list(static), start=Fraction(0))
    frames.begin(play)
    return frames


def test_a_held_play_s_static_picture_is_whole():
    """Its reset comes before the frame's live scope, which is told of it:
    the static picture begins on it.
    """
    from arachne import ThreeDCamera, tempconfig

    under, top = _squares()
    with tempconfig({**MEDIUM, "draw_path": "core"}):
        camera = ThreeDCamera()
        _begin_held_play(camera, [under, top])
        found = live_painter(camera)
        assert found is not None
        painter = found.painter(camera)
        assert painter.whole
        (pixel,) = painter.contributors(
            (*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1)
        ).values()
        assert len(pixel.leaves) == 2


def test_a_held_play_s_static_picture_calls_the_camera_in_its_order():
    """A camera that counts its calls sees its reset before the route is
    read off the context it keeps, as it always has: the reset, the route's
    reading (the context kept, a fresh one made to compare with), then the
    capture's own context.
    """
    from arachne import ThreeDCamera, tempconfig

    calls: list[str] = []

    class Counting(ThreeDCamera):
        def reset(self):
            calls.append("reset")
            return super().reset()

        def get_cached_cairo_context(self, pixel_array):
            calls.append("get_cached_cairo_context")
            return super().get_cached_cairo_context(pixel_array)

        def make_cairo_context(self, *args, **kwargs):
            calls.append("make_cairo_context")
            return super().make_cairo_context(*args, **kwargs)

    under, top = _squares()
    with tempconfig({**MEDIUM, "draw_path": "core"}):
        camera = Counting()
        # A context kept for the array, as any earlier frame leaves one.
        camera.reset()
        camera.capture_mobjects([under])
        calls.clear()
        _begin_held_play(camera, [under, top])
    assert calls == [
        "reset",
        "get_cached_cairo_context",
        "make_cairo_context",
        "get_cached_cairo_context",
    ]
