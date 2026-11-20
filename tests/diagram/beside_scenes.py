"""A flowing connector whose label, set beside its line, lands on a block
above it, and the same with the label set on the other side.
"""

from __future__ import annotations

from arachne import LEFT, RIGHT, UP, Diagram, Flow, Scene, Square


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
