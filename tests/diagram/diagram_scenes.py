"""Scenes of diagrams: tokens flowing along connectors, and zooms into a
block whose inside is legible and whose inside is not.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Block,
    Circle,
    Diagram,
    Flow,
    MovingCameraScene,
    Node,
    Rectangle,
    Scene,
    ZoomInto,
)

#: The flows' speeds, scene units per second.
STRAIGHT_SPEED = 1.5
ELBOW_SPEED = 0.8


def flowing_diagram() -> tuple[Diagram, Flow, Flow]:
    """Two nodes joined straight, and two blocks joined round an elbow,
    each with a flow.
    """
    diagram = Diagram()
    a = Node("A").move_to(LEFT * 3 + DOWN)
    b = Node("B").move_to(RIGHT * 3 + DOWN)
    diagram.add(a, b)
    straight = diagram.connect((a, "right"), (b, "left"), route="straight", tip=False)
    source = Block("source", width=1.6, height=1.0).move_to(LEFT * 3 + UP * 2)
    sink = Block("sink", width=1.6, height=1.0).move_to(RIGHT * 2 + UP * 0.4)
    diagram.add(source, sink)
    elbow = diagram.connect((source, "right"), (sink, "top"))
    straight_flow = Flow(straight, speed=STRAIGHT_SPEED, tokens=1)
    elbow_flow = Flow(elbow, speed=ELBOW_SPEED, tokens=2, offset=0.1)
    return diagram, straight_flow, elbow_flow


class FlowThroughPlaysAndWaits(Scene):
    """Flows run through a wait, a play of something else and a second wait:
    3 s in all.
    """

    def construct(self):
        diagram, straight_flow, elbow_flow = flowing_diagram()
        marker = Circle(radius=0.3).move_to(DOWN * 3 + LEFT * 5)
        self.add(diagram, straight_flow, elbow_flow, marker)
        self.wait(1)
        self.play(marker.animate.shift(RIGHT * 2), run_time=1)
        self.wait(1)


def zoomed_diagram(inside_size: float, colliding: bool) -> tuple[Diagram, Block]:
    """Three blocks in a row joined by labelled connectors; the middle one
    holds a diagram of two nodes joined by a labelled connector, its labels
    ``inside_size`` points, and with ``colliding`` a third label set on the
    second's.
    """
    inside = Diagram(cell=(2.6, 1.6))
    first = inside.place(Node("in", font_size=inside_size), 0, 0)
    second = inside.place(Node("out", font_size=inside_size), 1, 0)
    inside.connect(first, second, label="signal", font_size=inside_size)
    if colliding:
        stray = Node("over", font_size=inside_size).move_to(second.get_center())
        stray.shift(RIGHT * 0.05)
        inside.add(stray)
    outer = Diagram(cell=(4.4, 2.0))
    left = outer.place(Block("intake", width=2.2, height=1.4), 0, 0)
    middle = outer.place(Block("process", width=2.2, height=1.4, inside=inside), 1, 0)
    right = outer.place(Block("output", width=2.2, height=1.4), 2, 0)
    outer.connect(left, middle, label="feed")
    outer.connect(middle, right, label="result")
    outer.move_to(DOWN * 0.2)
    return outer, middle


class LegibleZoom(MovingCameraScene):
    """A zoom into the middle block, over 0-2 s, held to 2.5 s."""

    inside_size = 36
    colliding = False

    def construct(self):
        diagram, middle = zoomed_diagram(self.inside_size, self.colliding)
        #: The block left of the one zoomed into, which the zoom cuts.
        self.neighbour = diagram[0]
        self.add(diagram)
        self.play(ZoomInto(middle), run_time=2)
        self.wait(0.5)


class IllegibleZoom(LegibleZoom):
    """The twin whose inside labels end too small to read, two of them on
    each other.
    """

    inside_size = 4
    colliding = True


class ZoomWithoutMovingCamera(Scene):
    """A zoom asked of a camera whose frame does not move."""

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.add(diagram)
        self.play(ZoomInto(middle))


#: The moving flow's speed and token count.
MOVING_SPEED = 1.2
MOVING_TOKENS = 4


class FlowOnAMovingEdge(Scene):
    """A flow inside its diagram: the diagram moves over 0.5-1.5 s, then
    the connector alone shrinks to half its length over 1.5-2.5 s; 3 s in
    all.
    """

    def construct(self):
        a = Node("A").move_to(LEFT * 3 + DOWN)
        b = Node("B").move_to(RIGHT * 3 + DOWN)
        diagram = Diagram(a, b)
        edge = diagram.connect((a, "right"), (b, "left"), route="straight", tip=False)
        flow = Flow(edge, speed=MOVING_SPEED, tokens=MOVING_TOKENS, offset=0.2)
        diagram.add(flow)
        self.add(diagram)
        self.wait(0.5)
        self.play(diagram.animate.shift(UP), run_time=1)
        self.play(edge.animate.scale(0.5), run_time=1)
        self.wait(0.5)


class ZoomWithAFrameOfItsOwn(Scene):
    """A zoom handed a frame no camera of the scene views through."""

    def construct(self):
        diagram, middle = zoomed_diagram(36, False)
        self.add(diagram)
        self.play(ZoomInto(middle, frame=Rectangle(width=16, height=9)))
