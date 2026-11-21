"""A flowing connector whose label, set beside its line, lands on a block
above it, and the same with the label set on the other side.
"""

from __future__ import annotations

from arachne import DOWN, LEFT, RIGHT, UP, Block, Diagram, Flow, Scene, Square


def crowded(side: str) -> tuple[Diagram, Flow]:
    """Two squares joined straight by a labelled connector with a flow, and
    a block close above the connector's middle; the label on ``side``.
    """
    a, b = Square(0.6).move_to(LEFT * 2.5), Square(0.6).move_to(RIGHT * 2.5)
    above = Square(0.8).move_to(UP * 0.75)
    diagram = Diagram(a, b, above)
    edge = diagram.connect(
        a, b, route="straight", tip=False, label="feed", label_side=side
    )
    return diagram, Flow(edge, speed=1.0, tokens=2)


class BesideUnderABlock(Scene):
    def construct(self):
        diagram, flow = crowded("above")
        self.add(diagram, flow)
        self.wait(0.5)


class BesideOnTheOtherSide(Scene):
    def construct(self):
        diagram, flow = crowded("below")
        self.add(diagram, flow)
        self.wait(0.5)


def labelled() -> tuple[Diagram, Flow]:
    """Two squares joined straight by a labelled connector with a flow, the
    label above its line.
    """
    a, b = Square(0.6).move_to(LEFT * 2.5), Square(0.6).move_to(RIGHT * 2.5)
    diagram = Diagram(a, b)
    edge = diagram.connect(a, b, route="straight", tip=False, label="feed")
    return diagram, Flow(edge, speed=1.0, tokens=2)


class CoveredThenRemoved(Scene):
    """The label on a square for 1 s: its flow taken out at 0.5 s, the
    square at 1 s; then 0.5 s more.
    """

    def construct(self):
        diagram, flow = labelled()
        over = Square(0.4).move_to(UP * 0.6)
        diagram.add(over)
        self.add(diagram, flow)
        self.wait(0.5)
        self.remove(flow)
        self.wait(0.5)
        diagram.remove(over)
        self.wait(0.5)


class PassingNear(Scene):
    """A diagonal connector and an elbow passing the label, well clear of
    it, though their boxes hold it.
    """

    def construct(self):
        diagram, flow = labelled()
        c = Square(0.4).move_to(LEFT * 2.7 + UP * 1.5)
        d = Square(0.4).move_to(RIGHT * 1.5 + DOWN * 2.7)
        e = Block("e", width=0.8, height=0.5).move_to(LEFT * 1.0 + UP * 1.8)
        f = Block("f", width=0.8, height=0.5).move_to(RIGHT * 1.8 + DOWN * 0.9)
        diagram.add(c, d, e, f)
        diagram.connect(c, d, route="straight", tip=False)
        diagram.connect((e, "bottom"), (f, "left"))
        self.add(diagram, flow)
        self.wait(0.5)


class Crossing(Scene):
    """A diagonal connector through the label."""

    def construct(self):
        diagram, flow = labelled()
        g = Square(0.4).move_to(LEFT * 1.6 + UP * 1.4)
        h = Square(0.4).move_to(RIGHT * 1.6 + DOWN * 0.8)
        diagram.add(g, h)
        diagram.connect(g, h, route="straight", tip=False)
        self.add(diagram, flow)
        self.wait(0.5)
