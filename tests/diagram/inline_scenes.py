"""Diagrams whose connector's label, on its plate at the line's middle,
leaves a set stretch of the line showing on each side, whatever size the
platform's fonts give the label: its plate is stretched along the line to
the length that leaves it.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Block,
    Diagram,
    Flow,
    MovingCameraScene,
    Node,
    Scene,
    ZoomInto,
    config,
)
from arachne.mobject.diagram import ArcLength
from arachne.mobject.marks.blocks import INSIDE_SHARE

#: What the line of the scenes that hide it shows each side of its plate.
HIDDEN = 0.03
#: What the line of the scenes that keep it shows each side of its plate.
SHOWN = 0.15


def leave_showing(edge, stub: float, axis: int = 0) -> None:
    """Stretch ``edge``'s label along its line, which runs along ``axis``
    (0 across, 1 upright), until ``stub`` of the line shows on each side of
    its plate.
    """
    shaft = ArcLength(edge.points).length
    if axis == 0:
        edge.label.stretch_to_fit_width(shaft - 2 * stub)
    else:
        edge.label.stretch_to_fit_height(shaft - 2 * stub)


def put_back_on_its_line(edge) -> None:
    """Centre ``edge``'s label on its line's middle again."""
    path = ArcLength(edge.points)
    edge.label.move_to(path.point(path.length / 2))


def two_nodes(stub: float, **connect) -> Diagram:
    """Two nodes joined straight by a labelled connector whose line shows
    ``stub`` units either side of the label's plate.
    """
    diagram = Diagram()
    a = Node("alpha", radius=0.6).move_to(LEFT * 2)
    b = Node("beta", radius=0.6).move_to(RIGHT * 2)
    diagram.add(a, b)
    edge = diagram.connect(a, b, label="link", route="straight", **connect)
    leave_showing(edge, stub)
    return diagram


class LineHiddenByItsLabel(Scene):
    def construct(self):
        self.add(two_nodes(HIDDEN))
        self.wait(0.5)


class LineShowingPastItsLabel(Scene):
    def construct(self):
        self.add(two_nodes(SHOWN))
        self.wait(0.5)


class StubAtTheLeast(Scene):
    """The line showing ``stub`` each side; the twins a hair either way."""

    stub = 0.08

    def construct(self):
        self.add(two_nodes(self.stub))
        self.wait(0.5)


class StubJustUnderTheLeast(StubAtTheLeast):
    stub = 0.0799


class StubJustOverTheLeast(StubAtTheLeast):
    stub = 0.0801


class LineDrawnOverItsLabel(Scene):
    """The hiding label drawn before its line, which runs over the plate."""

    def construct(self):
        diagram = two_nodes(HIDDEN)
        label = diagram.edges[0].label
        diagram.remove(label)
        diagram.add_to_back(label)
        self.add(diagram)
        self.wait(0.5)


class LabelOnAClearPlate(Scene):
    """The hiding label's plate painting nothing."""

    def construct(self):
        diagram = two_nodes(HIDDEN)
        diagram.edges[0].label.background_rectangle.set_fill(opacity=0)
        self.add(diagram)
        self.wait(0.5)


def flowing_hidden_line() -> tuple[Diagram, Flow]:
    """The hiding label with a flow on its connector, put back on its line's
    middle after the flow moved it beside the line.
    """
    diagram = two_nodes(HIDDEN, tip=False)
    edge = diagram.edges[0]
    flow = Flow(edge, speed=1.0, tokens=2)
    put_back_on_its_line(edge)
    return diagram, flow


class HiddenLineWithAFlow(Scene):
    def construct(self):
        self.add(*flowing_hidden_line())
        self.wait(0.5)


class OneOfTwoLabelsHidesItsLine(Scene):
    """Three nodes in a row: the first connector's label hides its line,
    the second's leaves it showing.
    """

    def construct(self):
        diagram = Diagram()
        a = Node("alpha", radius=0.6).move_to(LEFT * 4)
        b = Node("beta", radius=0.6)
        c = Node("gamma", radius=0.6).move_to(RIGHT * 4)
        diagram.add(a, b, c)
        hiding = diagram.connect(a, b, label="hides", route="straight")
        leave_showing(hiding, HIDDEN)
        keeping = diagram.connect(b, c, label="keeps", route="straight")
        leave_showing(keeping, SHOWN)
        self.add(diagram)
        self.wait(0.5)


def stacked(stub: float) -> Diagram:
    """Two nodes, one over the other, joined straight by a labelled
    connector whose upright line shows ``stub`` units either side of the
    label's plate.
    """
    diagram = Diagram()
    a = Node("alpha", radius=0.6).move_to(UP)
    b = Node("beta", radius=0.6).move_to(DOWN)
    diagram.add(a, b)
    edge = diagram.connect(a, b, label="link", route="straight")
    leave_showing(edge, stub, axis=1)
    return diagram


class UprightLineInAFrameHalfAsWide(MovingCameraScene):
    """The camera's frame squeezed to half its width, its height kept,
    over an upright line showing ``stub`` scene units each side of its
    plate.
    """

    stub = 0.05

    def construct(self):
        self.camera.frame.stretch(0.5, 0)
        self.add(stacked(self.stub))
        self.wait(0.5)


class UprightHiddenLineInAFrameHalfAsWide(UprightLineInAFrameHalfAsWide):
    stub = HIDDEN


#: The block the zoom opens: its width and height.
BLOCK = (2.2, 1.4)
#: The height every inside glyph is set to, in scene units once in the
#: block: it reads at the zoom's legible height (20 pixels at 1080, with
#: its 1.1 margin) once the frame is 6 units tall, well before the frame
#: is narrow enough (3 tall) for the hidden line to show on screen.
INSIDE_GLYPHS = 6.0 * 20 * 1.1 / 1080


def set_glyph_height(text, height: float) -> None:
    """Stretch ``text`` upright until its glyphs, its plate aside, are
    ``height`` tall, as the zoom measures them.
    """
    plate = getattr(text, "background_rectangle", None)
    skip = {id(m) for m in plate.get_family()} if plate is not None else set()
    ys = np.concatenate(
        [
            m.points[:, 1]
            for m in text.get_family()
            if id(m) not in skip and len(m.points)
        ]
    )
    text.stretch(height / float(ys.max() - ys.min()), 1)


def zoomed_hidden_line() -> tuple[Diagram, Block]:
    """A block between two others holding two nodes joined by a labelled
    connector whose line, once the block is made, shows :data:`HIDDEN`
    units either side of the label's plate: too little on the whole frame,
    enough on the frame the zoom ends on. Every glyph inside is
    :data:`INSIDE_GLYPHS` tall, so the fade starts at one frame whatever
    the fonts.
    """
    inside = Diagram(cell=(2.6, 1.6))
    first = inside.place(Node("in", font_size=36), 0, 0)
    second = inside.place(Node("out", font_size=36), 1, 0)
    edge = inside.connect(first, second, label="signal", font_size=36)
    # The block scales its inside to its body's width: the nodes set it.
    scale = BLOCK[0] * INSIDE_SHARE / inside.width
    for text in (first.label, second.label, edge.label):
        set_glyph_height(text, INSIDE_GLYPHS / scale)
    leave_showing(edge, HIDDEN / scale)
    outer = Diagram(cell=(4.4, 2.0))
    outer.place(Block("intake", width=2.2, height=1.4), 0, 0)
    middle = outer.place(
        Block("process", width=BLOCK[0], height=BLOCK[1], inside=inside), 1, 0
    )
    outer.place(Block("output", width=2.2, height=1.4), 2, 0)
    outer.move_to(ORIGIN)
    return outer, middle


class ZoomOntoAHiddenLine(MovingCameraScene):
    """A zoom into the middle block over 0-2 s, held to 2.5 s."""

    def construct(self):
        diagram, middle = zoomed_hidden_line()
        self.add(diagram)
        self.play(ZoomInto(middle), run_time=2)
        self.wait(0.5)


class PillJustInsideTheFrame(Scene):
    """A connector running on past the frame's right edge to a node off
    it, its label's plate ``inside`` units inside that edge: past the plate
    the line shows that far, and runs on off the screen.
    """

    inside = 0.02

    def construct(self):
        diagram = Diagram()
        a = Node("alpha", radius=0.6).move_to(LEFT * 2)
        b = Node("beta", radius=0.6).move_to(RIGHT * 12)
        diagram.add(a, b)
        edge = diagram.connect(a, b, label="link", route="straight")
        plate = edge.label.background_rectangle
        edge.label.shift(
            RIGHT * (config.frame_width / 2 - self.inside - plate.get_right()[0])
        )
        self.add(diagram)
        self.wait(0.5)


class PillWellInsideTheFrame(PillJustInsideTheFrame):
    inside = 0.15


class PlateOnItsLineOnlyOffTheFrame(Scene):
    """A connector rising across the frame's right edge, its label's plate
    standing partly inside that edge, above the line there: the line runs
    under the plate only past the edge, off the screen.
    """

    def construct(self):
        diagram = Diagram()
        a = Node("alpha", radius=0.6).move_to(RIGHT * 3 + DOWN)
        b = Node("beta", radius=0.6).move_to(RIGHT * 12 + UP * 3)
        diagram.add(a, b)
        edge = diagram.connect(a, b, label="link", route="straight")
        start, end = np.asarray(edge.points[0]), np.asarray(edge.points[-1])
        edge_x = config.frame_width / 2
        rise = (end[1] - start[1]) / (end[0] - start[0])
        at_edge = start[1] + rise * (edge_x - start[0])
        plate = edge.label.background_rectangle
        corner = plate.get_corner(DOWN + LEFT)
        edge.label.shift(
            RIGHT * (edge_x - 0.1 - corner[0]) + UP * (at_edge + 0.12 - corner[1])
        )
        self.add(diagram)
        self.wait(0.5)
