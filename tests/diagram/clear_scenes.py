"""Diagrams whose labels crowd each other, run past the frame's edge or are
too small to read, beside one whose labels are clear.
"""

from __future__ import annotations

from arachne import DOWN, LEFT, RIGHT, UP, Diagram, Node, Scene, config


def two_nodes(apart: float) -> Diagram:
    """Two labelled nodes ``apart`` between centres, joined by a labelled
    connector.
    """
    diagram = Diagram()
    a = Node("alpha", radius=0.6).move_to(LEFT * apart / 2)
    b = Node("beta", radius=0.6).move_to(RIGHT * apart / 2)
    diagram.add(a, b)
    diagram.connect(a, b, label="link", route="straight")
    return diagram


class LabelsApart(Scene):
    def construct(self):
        self.add(two_nodes(4.0))
        self.wait(0.5)


class LabelsOnLabels(Scene):
    """A third node's label set on the second's."""

    def construct(self):
        diagram = two_nodes(4.0)
        stray = Node("gamma", radius=0.6).move_to(diagram[1].get_center() + UP * 0.1)
        diagram.add(stray)
        self.add(diagram)
        self.wait(0.5)


class PillPastTheEdge(Scene):
    """A connector's pill, below the nodes, 0.03 units past the bottom
    edge, its glyphs still inside.
    """

    def construct(self):
        diagram = two_nodes(4.0)
        a, b = diagram[0], diagram[1]
        edge = diagram.connect((a, "bottom"), (b, "bottom"), label="return")
        bottom = -config.frame_height / 2
        diagram.shift(DOWN * (edge.label.get_bottom()[1] - bottom + 0.03))
        self.add(diagram)
        self.wait(0.5)


class TinyLabel(Scene):
    """One node labelled far too small to read."""

    def construct(self):
        diagram = Diagram(Node("speck", radius=0.6, font_size=6).shift(LEFT * 2))
        diagram.add(Node("plain", radius=0.6).shift(RIGHT * 2))
        self.add(diagram)
        self.wait(0.5)
