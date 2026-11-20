"""Scenes of zooms that come back: into a block and out again, under
declared timing, in and out and in again, two levels deep in order and out
of order, and returns the engine refuses.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    ORIGIN,
    RIGHT,
    UP,
    Block,
    Diagram,
    MovingCameraScene,
    Node,
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


def computer() -> tuple[Diagram, Block, list]:
    """Case 7's two levels: a computer's blocks, the processor holding its
    core; and each connector of the computer with the blocks it joins.
    """
    core = Diagram(cell=(2.4, 1.5))
    control = core.place(Node("ctrl", radius=0.55, font_size=36), 0.5, 0)
    alu = core.place(Node("ALU", radius=0.55, font_size=36), 0, 1)
    registers = core.place(Node("regs", radius=0.55, font_size=36), 1, 1)
    core.connect(control, alu)
    core.connect(control, registers)
    core.connect(alu, registers, route="straight")
    machine = Diagram(cell=(3.4, 2.4))
    source = machine.place(Block("input", width=2.2, height=1.4), 0, 0)
    processor = machine.place(
        Block("processor", width=2.4, height=1.6, inside=core), 1, 0
    )
    sink = machine.place(Block("output", width=2.2, height=1.4), 2, 0)
    memory = machine.place(Block("memory", width=2.2, height=1.4), 1, 1)
    joins = [
        (machine.connect(source, processor, label="data"), source, processor),
        (machine.connect(processor, sink), processor, sink),
        (machine.connect((processor, "bottom"), memory), processor, memory),
    ]
    machine.move_to(ORIGIN)
    return machine, processor, joins


class IntoTheProcessorAndBack(MovingCameraScene):
    """Case 7's zoom over 0.6-3.1 s, held to 4.1 s; back over 4.1-6.6 s;
    and in again, with a new zoom, over 6.6-9.1 s.
    """

    def construct(self):
        machine, processor, self.joins = computer()
        self.add(machine)
        self.wait(0.6)
        zoom = ZoomInto(processor)
        self.play(zoom, run_time=2.5)
        self.wait(1)
        self.play(ZoomOut(zoom))
        self.play(ZoomInto(processor), run_time=2.5)
