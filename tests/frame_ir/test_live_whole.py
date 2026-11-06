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
