"""On the core, a live frame's camera array has one writer: the core.

The core draws each capture from its description, and CE's own draw of it
draws nowhere, so whatever else writes the camera's array within a frame
(the camera's code as it captures, a scene's code between two captures)
would be dropped by the core's picture. Each capture's boundaries are
checked: at its start the array holds what the core last drew into it in
the frame (or the background its reset installed), and at its end, before
the core draws over it, what it held at the start. A write either check sees
leaves the frame layered for attribution. Where the camera drew the frame
itself so far (the scene's first frame of a camera the scene's code
changed, or its first capture after a reset), the whole scene falls back to
its camera's own draw, which is Cairo's; elsewhere the frame is
``render.unable``. The write is never dropped.

Frames here are 192 x 108 pixels over CE's 8-unit frame height: 13.5 pixels a
unit, the origin at pixel (96, 54).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import ThreeDCamera
from arachne.frame_ir.contributors import LiveFrame
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import Unable
from arachne.frame_ir.live import live, live_painter

from ..timeline.conftest import render
from . import camera_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}
CENTRE = (96, 54)
WHITE = (255, 255, 255, 255)


@pytest.fixture
def findings(monkeypatch):
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    return lint.raised


def _fallbacks(findings: list) -> list:
    return [f for f in findings if f.code == "render.fallback"]


class WritingCamera(ThreeDCamera):
    """A camera whose one change is its projection, a change the gate proved,
    which also writes a patch of the camera's own array: CE shows the patch
    wherever nothing is drawn over it.
    """

    #: Whether the projection writes (a switch outside the camera's state).
    switch = {"on": True}

    def project_points(self, points):
        if self.switch["on"]:
            self.pixel_array[0:3, 0:3] = WHITE
        return super().project_points(points)


class RewritingCamera(ThreeDCamera):
    """A projection that writes the camera's array with the values it holds."""

    def project_points(self, points):
        self.pixel_array[0:3, 0:3] = np.array(self.pixel_array[0:3, 0:3])
        return super().project_points(points)


class WritingSolids(camera_scenes.Solids):
    """:class:`camera_scenes.Solids` through :class:`WritingCamera`."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=WritingCamera, **kwargs)


def _squares():
    from arachne import BLUE, RED, Square

    under = Square(side_length=2, fill_color=RED, fill_opacity=1.0, stroke_width=0)
    top = Square(side_length=1, fill_color=BLUE, fill_opacity=0.5, stroke_width=0)
    return under, top


def _frames(camera_class, *frames, draw_path="core"):
    """A fresh ``camera_class`` camera after the frames ``frames`` (each a
    list of steps, each a callable handed the camera), each frame live on
    ``draw_path``; the camera's array is copied after each frame.
    """
    from arachne import tempconfig

    pictures = []
    with tempconfig(MEDIUM):
        camera = camera_class()
        for index, steps in enumerate(frames):
            with live(camera, Fraction(index, FPS), draw_path=draw_path):
                for step in steps:
                    step(camera)
            pictures.append(camera.pixel_array.copy())
    return camera, pictures


def _core_painter(camera):
    found = live_painter(camera)
    assert found is not None
    painter = found.painter(camera)
    assert isinstance(painter, CorePainter)
    return painter


def _reset(camera):
    camera.reset()


def _capture(*mobjects):
    return lambda camera: camera.capture_mobjects(list(mobjects))


def _written(value):
    """A step that writes ``value(camera)`` into a patch of the camera's array
    directly, as a scene's code can, past every method.
    """

    def step(camera):
        camera.pixel_array[10:20, 10:20] = value(camera)

    return step


def _green(camera):
    return (0, 255, 0, 255)


def _held(camera):
    return np.array(camera.pixel_array[10:20, 10:20])


def _layered(painter):
    assert not painter.whole
    with pytest.raises(LiveFrame):
        painter.contributors((*CENTRE, CENTRE[0] + 1, CENTRE[1] + 1))


# A camera writing its array as it captures.


def test_a_camera_that_writes_its_array_as_it_projects_draws_as_cairo_does(
    findings,
):
    """The write is seen in the scene's first frame, which the camera draws
    itself: the whole scene is its own draw, as on Cairo, byte for byte.
    """
    _, cairo = render(WritingSolids, FPS, **MEDIUM)
    _, core = render(WritingSolids, FPS, draw_path="core", **MEDIUM)
    assert np.all(cairo[0][1, 1, :3] == 255), "Cairo shows the write"
    assert len(cairo) == len(core) > 0
    for index, (a, b) in enumerate(zip(cairo, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = _fallbacks(findings)
    assert "whole scene" in finding.message
    assert "wrote the camera's array" in finding.message


def test_a_capture_whose_camera_wrote_its_array_is_layered(findings):
    _, top = _squares()
    camera, (core,) = _frames(WritingCamera, [_reset, _capture(top)])
    _, (cairo,) = _frames(WritingCamera, [_reset, _capture(top)], draw_path="cairo")
    _layered(_core_painter(camera))
    np.testing.assert_array_equal(core, cairo)
    assert _fallbacks(findings)


def test_a_camera_first_writing_its_array_after_the_core_drew_frames_is_unable():
    """The core drew the scene's first frame: nothing of the camera's own draw
    is kept to fall back on, so the frame is not drawn at all.
    """
    _, top = _squares()

    class Later(WritingCamera):
        switch = {"on": False}

    def switch_on(camera):
        camera.switch["on"] = True

    with pytest.raises(Unable, match="render.unable: .*wrote the camera's array"):
        _frames(
            Later,
            [_reset, _capture(top)],
            [switch_on, _reset, _capture(top)],
        )


# A scene's code writing the array between two captures of one frame.


def test_a_write_between_captures_the_camera_drew_falls_back_to_its_own_draw(
    findings,
):
    """In the scene's first frame of a camera the scene's code changed, the
    camera draws each capture itself: the write is drawn under the next
    capture, as on Cairo, and the whole scene stays the camera's own draw.
    """
    under, top = _squares()
    steps = [_reset, _capture(under), _written(_green), _capture(top)]
    camera, (core,) = _frames(camera_scenes.ShiftedCamera, steps)
    _, (cairo,) = _frames(camera_scenes.ShiftedCamera, steps, draw_path="cairo")
    assert tuple(cairo[15, 15]) == (0, 255, 0, 255)
    np.testing.assert_array_equal(core, cairo)
    _layered(_core_painter(camera))
    [finding] = _fallbacks(findings)
    assert "whole scene" in finding.message


def test_a_write_between_captures_the_core_drew_is_unable():
    """The core drew the first capture: the frame Cairo draws has the
    camera's own draw of it under the write, which the core never made.
    """
    under, top = _squares()
    with pytest.raises(Unable, match="render.unable: .*between"):
        _frames(
            ThreeDCamera,
            [_reset, _capture(under), _written(_green), _capture(top)],
        )


def test_a_write_after_the_core_drew_frames_is_unable():
    """After the reset, on a frame after the scene's first: the core drew the
    frames before, so this one cannot be the camera's own draw.
    """
    _, top = _squares()
    with pytest.raises(Unable, match="render.unable"):
        _frames(
            ThreeDCamera,
            [_reset, _capture(top)],
            [_reset, _written(_green), _capture(top)],
        )


# What changes nothing.


@pytest.mark.parametrize("where", ["between-captures", "after-the-reset"])
def test_a_write_leaving_the_array_byte_equal_changes_nothing(where, findings):
    """In the scene's first frame and after it: no fallback, the frames are
    those drawn without the write, and the last is whole.
    """
    under, top = _squares()

    def frame(write):
        extra = [_written(_held)] if write else []
        if where == "between-captures":
            return [_reset, _capture(under), *extra, _capture(top)]
        return [_reset, *extra, _capture(under, top)]

    camera, written = _frames(ThreeDCamera, frame(True), frame(True))
    _, plain = _frames(ThreeDCamera, frame(False), frame(False))
    assert _core_painter(camera).whole
    assert not _fallbacks(findings)
    for a, b in zip(written, plain, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_projection_rewriting_its_array_s_own_values_changes_nothing(findings):
    """In the scene's first frame, which the camera draws itself, and after
    it, where the core draws: no fallback, and the frame is whole.
    """
    under, top = _squares()
    before = CorePainter.drawn_total
    camera, _ = _frames(
        RewritingCamera,
        [_reset, _capture(under, top)],
        [_reset, _capture(under, top)],
    )
    assert CorePainter.drawn_total > before
    assert _core_painter(camera).whole
    assert not _fallbacks(findings)


def test_points_and_images_drawn_on_the_core_are_the_core_s_alone(findings):
    """CE draws a point cloud and an image by writing the array, not through
    the context: on the core that draw is not wanted either, so it is no
    other writer.
    """
    before = CorePainter.drawn_total
    _, frames = render(camera_scenes.ShiftedPictures, FPS, draw_path="core", **MEDIUM)
    assert frames
    assert CorePainter.drawn_total > before
    assert not _fallbacks(findings), findings
