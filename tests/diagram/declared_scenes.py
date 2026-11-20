"""A scene whose flows declare their speed: two flows, one stopped half
way through.
"""

from __future__ import annotations

from arachne import LEFT, RIGHT, UP, Diagram, Flow, Node, Scene

#: The flows' speeds, scene units per second.
FAST = 1.5
SLOW = 0.8


class FlowStopped(Scene):
    """Two flows on two straight connectors for half a second; then the
    slow one is stopped.
    """

    def construct(self) -> None:
        a, b = Node("A").move_to(LEFT * 3 + UP), Node("B").move_to(RIGHT * 3 + UP)
        c, d = Node("C").move_to(LEFT * 3), Node("D").move_to(RIGHT * 3)
        diagram = Diagram(a, b, c, d)
        top = diagram.connect((a, "right"), (b, "left"), route="straight", tip=False)
        low = diagram.connect((c, "right"), (d, "left"), route="straight", tip=False)
        fast = Flow(top, speed=FAST, tokens=2)
        slow = Flow(low, speed=SLOW, tokens=1)
        self.add(diagram, fast, slow)
        self.wait(0.5)
        slow.stop()
        self.wait(0.5)
