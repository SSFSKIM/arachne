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
its camera's own draw, which is Cairo's. A capture begun on a reset in a
later frame, or on a picture the scene set, draws over what was written as
its pixels, as Cairo draws over it: each shown equal to Cairo's frame, and
the later frame's, on content off the pixel grid, equal to the core's own
draw of it from a clean start. Where the core's own pixels are under the
write, the frame is ``render.unable`` (and the render draws the scene again
on Cairo). In a changed camera's first frame, which it draws itself, its
draw is compared once with Cairo's draw of the frame's description, which
catches a write no call of the camera's own made. The write is never
dropped.

Frames here are 192 x 108 pixels over CE's 8-unit frame height: 13.5 pixels a
unit, the origin at pixel (96, 54).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import Square, ThreeDCamera
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


def test_a_write_after_a_continued_capture_of_a_fixed_orientation_mobject_is_seen():
    """A capture continuing the frame's picture with a mobject of fixed
    orientation (whose centres the capture keeps): what the next capture
    must begin on is still known, so a write before it is seen.
    """
    from arachne import Dot

    under, top = _squares()
    tag = Dot([1.0, 1.0, 0.0])

    def fixed(camera):
        camera.add_fixed_orientation_mobjects(tag)

    with pytest.raises(Unable, match="render.unable: .*between"):
        _frames(
            ThreeDCamera,
            [
                fixed,
                _reset,
                _capture(under),
                _capture(tag),
                _written(_green),
                _capture(top),
            ],
        )


# What the core draws over as pixels: a capture begun on a reset or on a
# picture the scene set holds nothing of the core's, so what the array holds
# is drawn under the capture, as Cairo draws it.


class AlignedShift(ThreeDCamera):
    """A projection of the scene's own, a change the gate proved, sliding what
    it projects left by 27 pixels: on the pixel grid, where the core draws
    :func:`_aligned`'s squares as Cairo does.
    """

    def project_points(self, points):
        return super().project_points(points) + np.array([-2.0, 0.0, 0.0])


def _aligned():
    """An opaque square under a translucent one, every edge on the pixel grid
    (27 and 9 pixels from the centre): the core draws them byte for byte as
    Cairo does, so a frame of them is compared whole.
    """
    from arachne import BLUE, RED, Square

    under = Square(side_length=4, fill_color=RED, fill_opacity=1.0, stroke_width=0)
    top = Square(side_length=4 / 3, fill_color=BLUE, fill_opacity=0.5, stroke_width=0)
    return under, top


def _band(camera):
    """A band across both squares' edges and the background around them: drawn
    under the opaque square, blended under the translucent one, and shown
    where nothing covers it.
    """
    camera.pixel_array[50:58, 30:150] = (0, 255, 0, 255)


def _picture(camera):
    """A picture of the scene's own, set as the camera's array."""
    picture = np.array(camera.background)
    picture[:, :, 0] = 40
    picture[30:80, 60:130] = (200, 10, 10, 255)
    camera.set_pixel_array(picture)


#: A camera of the engine's, and one the scene's code changed, whose first
#: frame the camera draws itself before the core's description replaces it.
CAMERAS = pytest.mark.parametrize(
    "camera_class", [ThreeDCamera, AlignedShift], ids=["plain", "hedged"]
)


def _as_on_cairo(camera_class, *frames):
    """The frames ``frames`` on the core, each equal to Cairo's, and the
    camera.
    """
    camera, core = _frames(camera_class, *frames)
    _, cairo = _frames(camera_class, *frames, draw_path="cairo")
    for index, (a, b) in enumerate(zip(core, cairo, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    return camera, cairo


@CAMERAS
def test_a_write_after_a_reset_on_a_later_frame_is_drawn_under_its_capture(
    camera_class, findings
):
    """The core drew the frames before, and the reset left nothing of the
    core's in the array: the capture draws over what was written, as Cairo
    draws over it, and the frame is layered.
    """
    under, top = _aligned()
    camera, cairo = _as_on_cairo(
        camera_class,
        [_reset, _capture(under, top)],
        [_reset, _band, _capture(under, top)],
    )
    assert np.any(cairo[0] != cairo[1]), "the write shows"
    _layered(_core_painter(camera))
    assert not _fallbacks(findings)


@CAMERAS
@pytest.mark.parametrize("written", [False, True], ids=["set", "set-then-written"])
@pytest.mark.parametrize("frame", [0, 1], ids=["first-frame", "later-frame"])
def test_a_picture_set_as_the_array_is_drawn_under_its_capture(
    camera_class, written, frame, findings
):
    """A picture the scene set with ``set_pixel_array``, written into or not,
    in the scene's first frame or a later one: the capture draws over it as
    Cairo draws over it, and the frame is layered.
    """
    under, top = _aligned()
    setting = [_picture, *([_band] if written else []), _capture(under, top)]
    plain = [_reset, _capture(under, top)]
    frames = [setting, plain] if frame == 0 else [plain, setting]
    camera, _ = _as_on_cairo(camera_class, *frames)
    if frame == 1:
        _layered(_core_painter(camera))
    assert not _fallbacks(findings)


# Drawn over as pixels against the core's own draw, on content off the pixel
# grid, where the core's antialiasing is its own and Cairo's frame is no
# measure of it.


def _off_grid():
    """A fill under a stroked curve and a label: edges at fractions of a pixel
    everywhere, antialiased as the core antialiases them.
    """
    from arachne import (
        BLUE,
        LEFT,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        Circle,
        ParametricFunction,
        Text,
    )

    fill = Circle(radius=2.2, fill_color=BLUE, fill_opacity=0.8, stroke_width=0)
    curve = ParametricFunction(
        lambda t: np.array([t, 0.8 * np.sin(2 * t), 0.0]),
        t_range=[-3.3, 3.3],
        color=YELLOW,
        stroke_width=5,
    )
    label = Text("ax", font_size=40, color=WHITE)
    fill.shift(0.37 * RIGHT)
    label.move_to(0.21 * LEFT + 0.63 * UP)
    return fill, curve, label


def _ramp(camera):
    """A write of many values across the fill's, the curve's and the label's
    edges and the background around them.
    """
    ramp = np.linspace(0, 255, 152).astype(np.uint8)
    patch = camera.pixel_array[40:70, 20:172]
    patch[:, :, 0] = ramp
    patch[:, :, 1] = ramp[::-1]
    patch[:, :, 2] = 90
    patch[:, :, 3] = 255


def _moved(*mobjects):
    """A step moving ``mobjects`` by a distance off the pixel grid."""

    def step(camera):
        from arachne import DOWN, RIGHT

        for mobject in mobjects:
            mobject.shift(0.29 * RIGHT + 0.11 * DOWN)

    return step


@CAMERAS
@pytest.mark.parametrize("moved", [False, True], ids=["still", "moved"])
def test_a_write_after_a_reset_on_a_later_frame_is_the_core_s_own_fresh_draw(
    camera_class, moved, findings
):
    """The capture begun on a later frame's reset draws over what was written
    byte for byte as the core draws the same scene state from a clean start:
    a fresh camera whose background is what the array held as the capture
    began, which its reset installs and nothing writes over (a whole frame).
    """
    content = _off_grid()
    held = []

    def keep(camera):
        held.append(np.array(camera.pixel_array))

    moving = [_moved(*content)] if moved else []
    camera, (first, drawn) = _frames(
        camera_class,
        [_reset, _capture(*content)],
        [_reset, *moving, _ramp, keep, _capture(*content)],
    )
    _layered(_core_painter(camera))

    fresh = _off_grid()
    [picture] = held

    def install(camera):
        camera.background = picture

    moving = [_moved(*fresh)] if moved else []
    again, (expected,) = _frames(
        camera_class, [install, _reset, *moving, _capture(*fresh)]
    )
    assert _core_painter(again).whole
    assert not _fallbacks(findings)
    assert np.any(first != drawn), "the write shows"
    assert np.any(drawn[40:70, 20:172] != picture[40:70, 20:172]), "drawn over it"
    differs = np.argwhere(np.any(drawn != expected, axis=-1))
    assert not len(differs), (
        f"{len(differs)} pixels differ, first at {differs[:5].tolist()}:"
        f" {drawn[tuple(differs[:5].T)].tolist()} drawn,"
        f" {expected[tuple(differs[:5].T)].tolist()} fresh"
    )


# A write in the hedged camera's first frame that no producer call makes.

#: What :class:`InkingSquare` reads: the scene drawing it.
INK: dict = {}


class InkingSquare(Square):
    """A square of the scene's own whose stroke width, read as the camera
    draws it, writes a patch of the camera's array in the scene's first frame:
    code of the scene's that runs within a capture, outside every call of the
    camera's own code.
    """

    def get_stroke_width(self, background=False):
        scene = INK.get("scene")
        if scene is not None and scene.renderer.time == 0:
            scene.renderer.camera.pixel_array[0:3, 0:3] = WHITE
        return super().get_stroke_width(background)


class InkedSolids(camera_scenes.ShiftedSolids):
    """:class:`camera_scenes.ShiftedSolids` with an :class:`InkingSquare`."""

    def construct(self):
        INK["scene"] = self
        self.add(InkingSquare(side_length=1, fill_opacity=0.6, stroke_width=0))
        super().construct()


def test_a_write_in_the_first_frame_outside_the_camera_s_code_falls_back(findings):
    """The hedged camera's first frame is its own draw, compared once with
    Cairo's draw of the frame's description before the description replaces
    it: a write the description lacks leaves the camera's own draw standing,
    and the whole scene is drawn as on Cairo, byte for byte.
    """
    _, cairo = render(InkedSolids, FPS, **MEDIUM)
    _, core = render(InkedSolids, FPS, draw_path="core", **MEDIUM)
    assert np.all(cairo[0][1, 1, :3] == 255), "Cairo shows the write"
    assert not np.all(cairo[1][1, 1, :3] == 255), "in the first frame only"
    assert len(cairo) == len(core) > 0
    for index, (a, b) in enumerate(zip(cairo, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = _fallbacks(findings)
    assert "whole scene" in finding.message
    assert "first frame" in finding.message


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
