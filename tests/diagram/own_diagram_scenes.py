"""A scene that makes a group class of its own named ``Diagram``, whose two
texts sit on each other: it is not the engine's diagram.
"""

from __future__ import annotations

from arachne import RIGHT, Scene, Text, VGroup


class Diagram(VGroup):
    """A scene's own group, named as the engine's diagram is."""


class OwnClassNamedDiagram(Scene):
    def construct(self):
        first = Text("alpha")
        second = Text("beta").move_to(first.get_center() + RIGHT * 0.1)
        self.add(Diagram(first, second))
        self.wait(0.5)
