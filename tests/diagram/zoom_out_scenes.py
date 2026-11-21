"""Scenes of zooms that come back: into a block and out again, under
declared timing, in and out and in again, two levels deep in order and out
of order, and returns the engine refuses.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DL,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    Block,
    Diagram,
    ImageMobject,
    MovingCameraScene,
    Node,
    Readout,
    Scene,
    ScreenRectangle,
    Square,
    ZoomInto,
    ZoomOut,
    config,
    set_timing,
)
from arachne.mobject.marks._style import opacities_of
from tests.diagram.diagram_scenes import zoomed_diagram

#: The declared diagram's drawings a second.
DRAWINGS = 12
#: The share of full opacity the neighbour shows before any zoom.
PARTIAL = 0.37


def shows(mobject) -> list:
    """How each member of ``mobject``'s family shows now."""
    return [opacities_of(m) for m in mobject.get_family()]


class ZoomAndBack(MovingCameraScene):
    """Still for 0.5 s; into the middle block over 0.5-2.5 s; held to 3 s;
    back over 3-5 s; held to 5.5 s. The neighbour the move-in cuts shows
    partly before it.
    """

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        #: The block left of the one zoomed into, which the zoom cuts.
        self.neighbour = diagram[0]
        self.neighbour.set_opacity(PARTIAL)
        self.diagram, self.middle = diagram, middle
        self.add(diagram)
        self.declare(diagram)
        self.wait(0.5)
        frame = self.camera.frame
        self.before = (frame.points, shows(diagram))
        self.zoom = ZoomInto(middle)
        self.play(self.zoom, run_time=2)
        self.wait(0.5)
        self.play(ZoomOut(self.zoom))
        self.after = (frame.points, shows(diagram))
        self.wait(0.5)

    def declare(self, diagram) -> None:
        """Nothing is declared here."""


class DeclaredZoomAndBack(ZoomAndBack):
    """The same, with a picture among the diagram's pieces, above the
    neighbour, drawn on twelves: the move-in cuts it and fades it, and the
    return brings it back. (A diagram is exact content, which stepped
    timing refuses; a picture in it is not.)
    """

    def declare(self, diagram) -> None:
        #: The declared picture, partly opaque.
        self.picture = Square(0.8).set_fill(BLUE, opacity=0.6).set_stroke(opacity=0.8)
        self.picture.move_to(diagram[0].get_top() + UP * 0.8)
        diagram.add(self.picture)
        self.picture_before = shows(self.picture)
        set_timing(self.picture, DRAWINGS)


class ZoomInOutIn(MovingCameraScene):
    """Into the middle block over 0-2 s, back over 2-4 s, and into it again,
    with a new zoom, over 4-6 s.
    """

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.neighbour = diagram[0]
        self.middle = middle
        self.add(diagram)
        first = ZoomInto(middle)
        self.play(first, run_time=2)
        self.play(ZoomOut(first))
        self.play(ZoomInto(middle), run_time=2)


def nested_diagram() -> tuple[Diagram, Block, Block]:
    """A row of two blocks; the second holds a node and a block, which holds
    a node of its own.
    """
    deep = Block("deep", width=1.4, height=1.0, inside=Diagram(Node("x", font_size=48)))
    inside = Diagram(Node("a", font_size=24), deep.shift(RIGHT * 1.6))
    outer = Block("outer", width=3.2, height=1.6, inside=inside)
    side = Block("side", width=2.2, height=1.4).shift(RIGHT * -4)
    return Diagram(side, outer.shift(RIGHT * 1.5)), outer, deep


class NestedInOrder(MovingCameraScene):
    """Into the outer block, into the deep one, out of the deep one, out of
    the outer one, a second each.
    """

    def construct(self):
        diagram, outer, deep = nested_diagram()
        self.add(diagram)
        frame = self.camera.frame
        self.before = (frame.points, shows(diagram))
        into_outer, into_deep = ZoomInto(outer), ZoomInto(deep)
        self.play(into_outer, run_time=1)
        self.inner_before = (frame.points, shows(diagram))
        self.play(into_deep, run_time=1)
        self.play(ZoomOut(into_deep))
        self.inner_after = (frame.points, shows(diagram))
        self.play(ZoomOut(into_outer))
        self.after = (frame.points, shows(diagram))


class NestedOutOfOrder(MovingCameraScene):
    """Into the outer block, into the deep one, and out of the outer one
    first.
    """

    def construct(self):
        diagram, outer, deep = nested_diagram()
        self.add(diagram)
        into_outer = ZoomInto(outer)
        self.play(into_outer, run_time=1)
        self.play(ZoomInto(deep), run_time=1)
        self.play(ZoomOut(into_outer))


class ReturnAfterTheCameraMoved(MovingCameraScene):
    """Into the middle block, the camera moved, and back."""

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.add(diagram)
        zoom = ZoomInto(middle)
        self.play(zoom)
        self.play(self.camera.frame.animate.shift(RIGHT * 0.5))
        self.play(ZoomOut(zoom))


class ReturnAfterAPieceLeft(MovingCameraScene):
    """Into the middle block, the neighbour it faded taken out, and back."""

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.add(diagram)
        zoom = ZoomInto(middle)
        self.play(zoom)
        diagram.remove(diagram[0])
        self.play(ZoomOut(zoom))


class ReturnWithoutMovingCamera(Scene):
    """A return asked of a scene whose camera's frame does not move, from a
    zoom played by hand on a frame of its own.
    """

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.add(diagram)
        zoom = ZoomInto(middle, frame=ScreenRectangle(height=config.frame_height))
        zoom.begin()
        zoom.finish()
        self.play(ZoomOut(zoom))


def hub() -> tuple[Diagram, Block, list]:
    """A row of three blocks, the middle one holding a diagram of two
    nodes, and a fourth block below the middle one; each connector of the
    row with the blocks it joins, the first labelled.
    """
    inner = Diagram(cell=(2.2, 1.4))
    first = inner.place(Node("u", radius=0.5, font_size=36), 0, 0)
    second = inner.place(Node("v", radius=0.5, font_size=36), 1, 0)
    inner.connect(first, second, route="straight")
    row = Diagram(cell=(3.6, 2.5))
    left = row.place(Block("a", width=2.0, height=1.3), 0, 0)
    target = row.place(Block("hub", width=2.6, height=1.5, inside=inner), 1, 0)
    right = row.place(Block("b", width=2.0, height=1.3), 2, 0)
    below = row.place(Block("c", width=2.0, height=1.3), 1, 1)
    joins = [
        (row.connect(left, target, label="in"), left, target),
        (row.connect(target, right), target, right),
        (row.connect((target, "bottom"), below), target, below),
    ]
    row.move_to(ORIGIN)
    return row, target, joins


class IntoAHubAndBack(MovingCameraScene):
    """Still to 0.6 s; into the hub over 0.6-3.1 s, held to 4.1 s; back
    over 4.1-6.6 s; and in again, with a new zoom, over 6.6-9.1 s.
    """

    def construct(self):
        row, target, self.joins = hub()
        self.add(row)
        self.wait(0.6)
        zoom = ZoomInto(target)
        self.play(zoom, run_time=2.5)
        self.wait(1)
        self.play(ZoomOut(zoom))
        self.play(ZoomInto(target), run_time=2.5)


class ZoomPastReadouts(MovingCameraScene):
    """Two readouts in the corners the move-in cuts, each changed while the
    frame is on the block: one from 4 to 9 (its glyph replaced), one from 4
    to 10 (one glyph become two); then the return.
    """

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.nine = Readout(4, "{:.0f}").to_corner(UL)
        self.ten = Readout(4, "{:.0f}").to_corner(DL)
        self.add(diagram, self.nine, self.ten)
        self.before = (shows(self.nine), shows(self.ten))
        (glyph,) = self.nine.label.family_members_with_points()
        #: How the glyph replaced showed before the move-in.
        self.glyph = opacities_of(glyph)
        zoom = ZoomInto(middle)
        self.play(zoom, run_time=2)
        self.nine.set_value(9)
        self.ten.set_value(10)
        self.while_in = (shows(self.nine), shows(self.ten))
        self.play(ZoomOut(zoom))


def picture() -> ImageMobject:
    """A small picture, its alpha a ramp, at 0.6 of its own opacity."""
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 0] = 200
    pixels[..., 2] = 90
    pixels[..., 3] = np.linspace(80, 255, 16, dtype=np.uint8)[None, :]
    return ImageMobject(pixels).set_opacity(0.6)


class ZoomPastAPicture(MovingCameraScene):
    """Still for 0.5 s; into the middle block over 0.5-2.5 s, which cuts a
    picture in the upper left corner; held to 3 s; back over 3-5 s; held to
    5.5 s.
    """

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.picture = picture().scale(2).to_corner(UL)
        self.add(diagram, self.picture)
        self.wait(0.5)
        self.before = np.array(self.picture.pixel_array)
        zoom = ZoomInto(middle)
        self.play(zoom, run_time=2)
        self.while_in = np.array(self.picture.pixel_array)
        self.wait(0.5)
        self.play(ZoomOut(zoom))
        self.after = np.array(self.picture.pixel_array)
        self.wait(0.5)
