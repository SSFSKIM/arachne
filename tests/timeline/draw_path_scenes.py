"""Scenes the draw-path differential renders and opens: one per route, held
plays, a Succession and plays ending by the end-window rule.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    PI,
    RED,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    Line,
    Rotate,
    Scene,
    Square,
    Succession,
    Transform,
    ValueTracker,
    always_redraw,
    smooth,
)


class TrustedOffGrid(Scene):
    """Trusted plays that begin between frames of the grid."""

    def construct(self):
        dot, square = Dot(LEFT * 2), Square()
        self.add(dot, square)
        self.wait(0.3)
        self.play(dot.animate.shift(RIGHT * 3), run_time=1.1)
        self.play(Rotate(square, PI / 3), rate_func=smooth, run_time=0.7)


class StateReadout(Scene):
    """A line redrawn from a tracker and a dot set from it: state plays."""

    def construct(self):
        t = ValueTracker(0)
        dot = Dot(color=BLUE)
        dot.add_updater(lambda m: m.move_to(RIGHT * t.get_value() + UP))
        line = always_redraw(lambda: Line(LEFT * 3, dot.get_center()))
        self.add(dot, line)
        self.play(t.animate.set_value(2), run_time=1.1)
        self.play(t.animate.set_value(-1), rate_func=smooth, run_time=0.6)


class Ticked(Scene):
    """A dt updater in the scene: every play ticks."""

    def construct(self):
        dot = Dot(LEFT * 3, color=RED)
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(dot)
        self.play(FadeIn(Square().shift(UP)), run_time=0.9)
        self.wait(1 / 3)


class Held(Scene):
    """A held wait, a frozen frame, then a move."""

    def construct(self):
        square = Square().shift(LEFT)
        self.add(square, Circle(radius=0.5).shift(RIGHT * 2))
        self.wait(0.5)
        self.wait(0.4, frozen_frame=True)
        self.play(square.animate.shift(DOWN), run_time=0.6)
        self.wait(0.2)


class Succeeding(Scene):
    """Succession parts ending off the grid: linear (exact instants) and
    smooth (bisected).
    """

    def construct(self):
        a, b, c = Square().shift(LEFT * 3), Circle(), Dot(RIGHT * 3)
        self.add(a, b, c)
        self.play(
            Succession(
                a.animate.shift(UP),
                Transform(b, a.copy().shift(DOWN * 2)),
                c.animate.shift(LEFT),
            ),
            run_time=1.1,
        )
        self.play(
            Succession(
                Transform(a, Circle().shift(DOWN)),
                c.animate.move_to(a),
                b.animate.scale(0.5),
                rate_func=smooth,
            ),
            run_time=0.9,
        )


class EndWindow(Scene):
    """Plays whose ends the end-window rule snaps: 1/10 then 0.2 ends at 3/10,
    a wait sized from the float clock lands on its cue, and a third of a
    second.
    """

    def construct(self):
        dot = Dot(LEFT * 2)
        self.add(dot)
        self.wait(0.1)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=2 / 3)
        self.wait(2.5 - self.renderer.time)
        self.play(dot.animate.rotate(np.pi / 4).shift(DOWN), run_time=1 / 3)
