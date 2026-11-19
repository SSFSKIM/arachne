"""Scenes the footprint tests execute: nested groups, a dot an updater moves
through a state-function play, an opaque square over another, and a crowded
frame from which one small object can be left out.

A scene and the one that leaves an object out share ``construct``, so every
mobject they both make has the same stable id in both.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Dot,
    Line,
    Scene,
    Square,
    Text,
    ValueTracker,
    VGroup,
    linear,
)


class Nested(Scene):
    """A group holding a group holding leaves, one leaf with a submobject of
    its own; one leaf moves.
    """

    def construct(self):
        holder = Square(side_length=1.5, color=BLUE)
        holder.add(Dot(color=RED))
        inner = VGroup(holder, Circle(radius=0.4, color=GREEN).shift(RIGHT * 2))
        outer = VGroup(inner, Square(side_length=0.5, color=YELLOW).shift(LEFT * 3))
        self.add(outer)
        self.play(inner[1].animate.shift(UP), run_time=0.5, rate_func=linear)
        self.wait(0.2)


class UpdatedDot(Scene):
    """A dot an updater moves after a tracker: a state-function play."""

    def construct(self):
        tracker = ValueTracker(-2)
        dot = Dot(radius=0.2, color=WHITE)
        dot.add_updater(lambda m: m.move_to(RIGHT * tracker.get_value()))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)


class Covered(Scene):
    """A small square wholly behind an opaque one, another half behind it."""

    def construct(self):
        hidden = Square(side_length=0.5, color=RED, fill_opacity=1)
        half = Square(side_length=0.5, color=GREEN, fill_opacity=1).shift(RIGHT)
        cover = Square(side_length=1.5, color=BLUE, fill_opacity=1, stroke_width=0)
        self.add(hidden, half, cover)
        self.wait(0.2)


class Crowded(Scene):
    """Forty small shapes, a thin line, a glyph and a dot; a square moves.

    ``left_out`` names what is made but never added.
    """

    left_out: str | None = None

    def construct(self):
        crowd = [
            Square(side_length=0.25, color=BLUE, fill_opacity=0.5).move_to(
                [-6 + 0.6 * (k % 10), 3 - 0.6 * (k // 10), 0]
            )
            for k in range(40)
        ]
        small = {
            "line": Line([2, 2, 0], [2.3, 2, 0], stroke_width=2),
            "glyph": Text("T", font_size=10).move_to([3, 2, 0]),
            "dot": Dot([4, 2, 0], radius=0.04),
        }
        mover = Square(side_length=0.5, color=RED).move_to([0, -2, 0])
        self.add(*crowd, *(m for name, m in small.items() if name != self.left_out))
        self.add(mover)
        self.play(mover.animate.shift(RIGHT * 2), run_time=0.4, rate_func=linear)


class CrowdedWithoutLine(Crowded):
    left_out = "line"


class CrowdedWithoutGlyph(Crowded):
    left_out = "glyph"


class CrowdedWithoutDot(Crowded):
    left_out = "dot"


class Panel(Scene):
    """A filled panel, ``offset`` to the right of the centre."""

    offset = 0.0

    def construct(self):
        panel = Square(side_length=2, color=BLUE, fill_opacity=1)
        self.add(panel.shift(RIGHT * self.offset))
        self.wait(0.2)


class PanelShifted(Panel):
    offset = 0.2
