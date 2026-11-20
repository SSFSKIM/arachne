"""Scenes whose objects hold GPU-only capabilities at chosen points of their
scripts, for the compose tests.
"""

from __future__ import annotations

from arachne import BLUE, RED, Circle, Create, Dot, FadeIn, Scene, Square
from arachne.routing import declare


class LastPlay(Scene):
    """Three plays; with ``switch`` set, the second is coloured otherwise and
    the circle shown in the last holds a blur. The first play is the same
    either way.
    """

    switch = False

    def construct(self):
        self.play(Create(Square()))
        dot = Dot(color=RED if self.switch else BLUE)
        self.play(FadeIn(dot))
        circle = Circle()
        if self.switch:
            declare(circle, "effect.blur")
        self.play(FadeIn(circle))


class AfterLastPlay(Scene):
    """A capability set on a shown object once its last play has ended."""

    def construct(self):
        square = Square()
        self.play(Create(square))
        self.play(square.animate.shift([1, 0, 0]))
        declare(square, "effect.soft_mask")


class NoPlay(Scene):
    """A declared object added, with no play at all."""

    def construct(self):
        square = Square()
        declare(square, "layer.depth")
        self.add(square)


class HashDot(Dot):
    """A dot whose class defines ``__hash__``: drawing it runs the scene's
    code, so the record draws nothing and its frames are drawn as it runs.
    """

    def __hash__(self):
        return object.__hash__(self)


class Live(Scene):
    """A scene drawn as it runs, whose dot holds a capability."""

    def construct(self):
        dot = HashDot()
        declare(dot, "effect.blur")
        self.play(FadeIn(dot))


class Plain(Scene):
    """Nothing declared."""

    def construct(self):
        self.play(Create(Square()))
        self.play(FadeIn(Circle()))
