"""Scenes whose objects hold GPU-only capabilities at chosen points of their
scripts, for the routing tests. A line a test names carries a ``# @tag``.
"""

from __future__ import annotations

from arachne import Circle, Create, Dot, FadeIn, Scene, Square, VGroup
from arachne.routing import declare, withdraw


class Plain(Scene):
    """Nothing declared."""

    def construct(self):
        self.play(Create(Square()))
        self.wait(0.2)


class LastPlay(Scene):
    """A blurred circle, shown only in the last play."""

    def construct(self):
        self.play(Create(Square()))
        circle = Circle()  # @last-play-object
        declare(circle, "effect.blur")
        self.play(FadeIn(circle))  # @last-play


class AfterLastPlay(Scene):
    """A capability set on a shown object once its last play has ended."""

    def construct(self):
        square = Square()  # @after-object
        self.play(Create(square))
        declare(square, "effect.soft_mask")


class NoPlay(Scene):
    """A declared object added, with no play at all."""

    def construct(self):
        square = Square()  # @no-play-object
        declare(square, "layer.depth")
        self.add(square)


class NeverShown(Scene):
    """A declared object built and never shown."""

    def construct(self):
        circle = Circle()
        declare(circle, "effect.blur")
        self.play(Create(Square()))


class Withdrawn(Scene):
    """A capability withdrawn before the object is ever shown."""

    def construct(self):
        circle = Circle()
        declare(circle, "effect.blur")
        withdraw(circle, "effect.blur")
        self.play(FadeIn(circle))


class ComesAndGoes(Scene):
    """A capability held during the first play only, then withdrawn."""

    def construct(self):
        circle = Circle()
        declare(circle, "effect.window")
        self.play(FadeIn(circle))  # @comes
        withdraw(circle, "effect.window")
        self.play(circle.animate.shift([1, 0, 0]))


class Nested(Scene):
    """Two capabilities on two members of a group, the depth one first."""

    def construct(self):
        inner, outer = Dot(), Circle()
        declare(inner, "layer.depth")
        self.play(FadeIn(VGroup(inner, Square())))  # @nested-first
        declare(outer, "effect.blur", "effect.depth_of_field")
        self.play(FadeIn(outer))  # @nested-second


class HashDot(Dot):
    """A user's dot whose class defines ``__hash__``: drawing it runs the
    scene's code, so the record draws nothing and its frames are drawn as the
    scene runs.
    """

    def __hash__(self):
        return object.__hash__(self)


class Refused(Scene):
    """A record that refuses drawing, holding a capability."""

    def construct(self):
        dot = HashDot()
        declare(dot, "effect.blur")
        self.play(FadeIn(dot))  # @refused


class RemovedThenDeclared(Scene):
    """An object animated, removed once the last play has ended, then given
    a capability: the scene's last frame does not show it.
    """

    def construct(self):
        square = Square()
        self.play(square.animate.shift([1, 0, 0]))
        self.remove(square)
        declare(square, "effect.blur")
