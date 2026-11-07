"""A shaded path's gradient runs between the ends its mobject gives, as CE's Cairo draws it.

``ThreeDCamera.modified_rgbas`` lights a shaded path's paint into two stops, and
``Camera.set_cairo_context_color`` lays them as a linear gradient between
``vmobject.get_gradient_start_and_end_points()``. For a shaded VMobject those are
its start and end corners, unless the mobject gives others: ``ThreeDAxes`` gives
every piece of an axis (and its tip) the axis's own edge centres along its light
source (``_set_axis_shading``), and a sheen of 0.2, so each axis is one ramp from
end to end. Each piece is one cubic, so its two lit corners are both its first
point: drawn between its corners, its gradient's ends meet and it is painted the
two stops' average, a step from piece to piece instead of CE's ramp (up to 20
levels at an axis's ends, more on a tip). The record carries the ends
(``extract`` reads ``get_gradient_start_and_end_points`` for every shaded path);
the core must draw between them. The ramp shows only where the sheen gives two
colours: on white it clamps to white, both stops are equal, and the average is
the ramp.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import DEGREES, GREY_C, Line, ThreeDAxes, ThreeDScene
from arachne.frame_ir.core_painter import CorePainter

from ..timeline.conftest import render

pytest.importorskip("arachne_core")

FPS = 15
FULL = {"pixel_width": 854, "pixel_height": 480}
#: Levels per channel the two paths may differ by inside a stroke.
LEVELS = 2
#: Levels the two lit stops of a ramp under test differ by at least, so the ramp is there to see.
RAMP = 10


class ThickAxes(ThreeDScene):
    """CE's ThreeDAxes, no tips or ticks, stroked thick enough that each piece's middle is covered whole, in
    CE's default colour (white, whose sheen clamps: each piece's two lit stops are equal).
    """

    colour = None

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES)
        config = {"stroke_width": 16, "include_tip": False, "include_ticks": False}
        if self.colour is not None:
            config["stroke_color"] = self.colour
        self.axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            axis_config=config,
        )
        self.add(self.axes)
        self.wait(1 / FPS)


class ThickGreyAxes(ThickAxes):
    """The same axes in grey, whose sheen gives each piece two colours: its lit stops differ, a ramp."""

    colour = GREY_C


class ThickShadedLine(ThreeDScene):
    """A shaded straight line of twenty cubics, with no ends of its own: its gradient runs between its lit
    corners (its first point and point ``((n - 1) // 6) * 3``) on both paths, its normals CE's fallback for
    collinear chords. The control for :class:`ThickGreyAxes`.
    """

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES)
        self.line = (
            Line([0, 0, -2], [0, 0, 2], stroke_width=16)
            .insert_n_curves(19)
            .set_shade_in_3d(True)
        )
        self.line.set_sheen(0.2)
        self.add(self.line)
        self.wait(1 / FPS)


def _pixel(camera, mob, point) -> tuple[int, int]:
    x, y = camera.transform_points_pre_display(mob, np.array([point], dtype=float))[0][
        :2
    ]
    return (
        int((x / camera.frame_width + 0.5) * camera.pixel_width),
        int((0.5 - y / camera.frame_height) * camera.pixel_height),
    )


def _stops(camera, mob) -> np.ndarray:
    """CE's two lit stops of ``mob``'s stroke, clamped as Cairo clamps them, in levels."""
    return (
        255 * np.clip(camera.modified_rgbas(mob, mob.get_stroke_rgbas()), 0, 1)[:, :3]
    )


def _ce_colour(camera, mob, point) -> np.ndarray:
    """CE's colour at ``point`` of ``mob``: its lit stops between its gradient's ends as given."""
    stops = _stops(camera, mob)
    ends = camera.transform_points_pre_display(
        mob, np.array(mob.get_gradient_start_and_end_points())
    )[:, :2]
    at = camera.transform_points_pre_display(mob, np.array([point], dtype=float))[0][:2]
    g = ends[1] - ends[0]
    t = float(np.clip(np.dot(at - ends[0], g) / np.dot(g, g), 0, 1))
    return stops[0] + t * (stops[1] - stops[0])


def _compare(scene_class, places_of):
    drawn = CorePainter.drawn_total
    scene, core = render(scene_class, FPS, draw_path="core", **FULL)
    assert CorePainter.drawn_total > drawn, "the frames were not the core's"
    _, cairo = render(scene_class, FPS, draw_path="cairo", **FULL)
    camera = scene.renderer.camera
    off = []
    for name, mob, point in places_of(scene):
        x, y = _pixel(camera, mob, point)
        mine, ce = core[-1][y, x, :3].astype(int), cairo[-1][y, x, :3].astype(int)
        if np.abs(mine - ce).max() > LEVELS:
            off.append(
                f"{name} at ({x}, {y}): core {mine.tolist()}, Cairo {ce.tolist()}, "
                f"CE's formula {np.round(_ce_colour(camera, mob, point), 1).tolist()}"
            )
    return scene, off


def _axis_places(scene):
    """Away from the origin and the pieces' seams: each axis's pieces 1, 4, 15 and 18, at their middles."""
    for label, axis in zip("xyz", scene.axes.axes, strict=True):
        for i in (1, 4, 15, 18):
            piece = axis.pieces[i]
            yield f"{label} piece {i}", piece, piece.points[[0, -1]].mean(0)


def test_an_axis_piece_is_drawn_along_its_axis_gradient_as_ce_draws_it():
    """Every grey axis piece's middle is the colour Cairo draws there: the axis's ramp between the ends
    ThreeDAxes gives its pieces, not its own corners' average.
    """
    scene, off = _compare(ThickGreyAxes, _axis_places)
    camera = scene.renderer.camera
    flat = [
        name
        for name, piece, _ in _axis_places(scene)
        if np.abs(np.diff(_stops(camera, piece), axis=0)).max() < RAMP
    ]
    assert not flat, (
        f"no ramp to draw: lit stops within {RAMP} levels of each other at {flat}"
    )
    assert not off, "\n".join(off)


def test_a_white_axis_piece_whose_stops_are_equal_is_drawn_as_ce_draws_it():
    """The control for the ramp: white axes, whose sheen clamps, so each piece's lit stops are equal and its
    corners' average is CE's colour, are drawn as Cairo draws them.
    """
    _, off = _compare(ThickAxes, _axis_places)
    assert not off, "\n".join(off)


def test_a_shaded_line_with_no_ends_of_its_own_is_drawn_as_ce_draws_it():
    """The control: a shaded collinear line of many cubics, its gradient between its lit corners and its
    normals CE's fallback for chords that do not cross, is drawn as Cairo draws it.
    """

    def places(scene):
        line = scene.line
        for i in (2, 7, 12, 17):
            yield f"cubic {i}", line, line.points[[4 * i, 4 * i + 3]].mean(0)

    _, off = _compare(ThickShadedLine, places)
    assert not off, "\n".join(off)
