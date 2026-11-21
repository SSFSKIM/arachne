"""Diagrams whose connector's label, on its plate at the line's middle,
leaves a set stretch of the line showing on each side, whatever width the
platform's fonts give the label: its plate is stretched to the width that
leaves it.
"""

from __future__ import annotations

from arachne import (
    LEFT,
    ORIGIN,
    RIGHT,
    Block,
    Diagram,
    Flow,
    MovingCameraScene,
    Node,
    Scene,
    ZoomInto,
)
from arachne.mobject.diagram import ArcLength
from arachne.mobject.marks.blocks import INSIDE_SHARE

#: What the line of the scenes that hide it shows each side of its plate.
HIDDEN = 0.03
#: What the line of the scenes that keep it shows each side of its plate.
SHOWN = 0.15


def leave_showing(edge, stub: float) -> None:
    """Stretch ``edge``'s label across its line until ``stub`` of the line
    shows on each side of its plate.
    """
    shaft = ArcLength(edge.points).length
    edge.label.stretch_to_fit_width(shaft - 2 * stub)


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


class HiddenLineWithAFlow(Scene):
    """The hiding label moved beside its line by a flow."""

    def construct(self):
        diagram = two_nodes(HIDDEN, tip=False)
        flow = Flow(diagram.edges[0], speed=1.0, tokens=2)
        self.add(diagram, flow)
        self.wait(0.5)


#: The block the zoom opens: its width and height.
BLOCK = (2.2, 1.4)


def zoomed_hidden_line() -> tuple[Diagram, Block]:
    """A block between two others holding two nodes joined by a labelled
    connector whose line, once the block is made, shows :data:`HIDDEN`
    units either side of the label's plate: too little on the whole frame,
    enough on the frame the zoom ends on.
    """
    inside = Diagram(cell=(2.6, 1.6))
    first = inside.place(Node("in", font_size=36), 0, 0)
    second = inside.place(Node("out", font_size=36), 1, 0)
    edge = inside.connect(first, second, label="signal", font_size=36)
    # The block scales its inside to its body's width: the nodes set it.
    scale = BLOCK[0] * INSIDE_SHARE / inside.width
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
