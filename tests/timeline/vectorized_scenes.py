"""Scenes the vectorized-play tests execute: trusted plays whose leaves are
blends and reveals, composed by CE's groups under linear and nonlinear rate
functions, and trusted plays that stay scalar.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Add,
    AnimationGroup,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    LaggedStartMap,
    Rotate,
    Scene,
    ShowPassingFlash,
    Square,
    Succession,
    Transform,
    TransformFromCopy,
    Triangle,
    Uncreate,
    VGroup,
    Wait,
    linear,
)
from arachne.utils.rate_functions import (
    ease_in_out_cubic,
    ease_out_back,
    not_quite_there,
    running_start,
    rush_into,
    smooth,
    smootherstep,
    squish_rate_func,
    there_and_back,
)


def _row(count: int, mob=Square, **kwargs) -> VGroup:
    return VGroup(*(mob(**kwargs).scale(0.3) for _ in range(count))).arrange(RIGHT)


class Grouped(Scene):
    """An AnimationGroup with a lag and a nonlinear rate, its parts a blend
    and a reveal of several leaves each.
    """

    def construct(self):
        row = _row(3)
        self.add(row)
        self.play(
            AnimationGroup(
                Transform(row, _row(3, Circle).shift(UP)),
                Create(_row(2, Triangle).shift(DOWN)),
                lag_ratio=0.4,
                rate_func=smooth,
            ),
            run_time=1,
        )


class Lagged(Scene):
    """LaggedStart and LaggedStartMap, and a leaf with its own lag."""

    def construct(self):
        dots = [Dot(RIGHT * k) for k in range(4)]
        self.play(LaggedStart(*(FadeIn(d, shift=UP) for d in dots), lag_ratio=0.3))
        self.play(LaggedStartMap(FadeOut, VGroup(*dots), lag_ratio=0.5), run_time=1)
        row = _row(4)
        self.play(FadeIn(row, lag_ratio=0.4, shift=DOWN), run_time=0.8)


class Nested(Scene):
    """Groups nested in groups, a zero-length part and a waiting part."""

    def construct(self):
        a, b = Square().shift(LEFT * 2), Circle().shift(RIGHT * 2)
        self.add(a)
        self.play(
            AnimationGroup(
                LaggedStart(
                    Create(b), Transform(a, Triangle().shift(LEFT)), lag_ratio=0.5
                ),
                AnimationGroup(Wait(0.3), FadeIn(Dot(UP * 2)), lag_ratio=1),
                Add(Dot(DOWN * 2)),
                rate_func=rush_into,
            ),
            run_time=1.2,
        )


class Rated(Scene):
    """Leaf rate functions of every kind the twins handle: CE's factories'
    closures, functions whose twins call Python's ``pow``, functions that
    leave the unit interval, and a reversed rate.
    """

    def construct(self):
        sq = Square()
        self.add(sq)
        self.play(sq.animate(rate_func=squish_rate_func(smooth, 0.2, 0.8)).shift(UP))
        self.play(sq.animate(rate_func=not_quite_there(smooth, 0.7)).scale(0.5))
        self.play(
            Transform(sq, Circle(color=RED), rate_func=smootherstep), run_time=0.6
        )
        self.play(sq.animate(rate_func=ease_in_out_cubic).shift(LEFT), run_time=0.6)
        self.play(sq.animate(rate_func=running_start).shift(RIGHT * 2), run_time=0.6)
        self.play(Create(Circle().shift(UP), rate_func=ease_out_back), run_time=0.6)
        self.play(sq.animate(rate_func=there_and_back).set_fill(RED, 1), run_time=0.6)
        self.play(Uncreate(sq), run_time=0.6)


class Revealed(Scene):
    """Reveals of paths of many curves: Create over a group with a lag, a
    passing flash of a width that crosses both ends, and a linear reveal
    whose window lands on curve boundaries.
    """

    def construct(self):
        self.play(Create(VGroup(Circle(), Square().scale(0.5))), run_time=1)
        self.play(ShowPassingFlash(Circle(color=RED), time_width=0.6), run_time=1)
        self.play(Create(Square(), rate_func=linear), run_time=1)


class Copied(Scene):
    """TransformFromCopy, which interpolates its transform at ``1 - alpha``."""

    def construct(self):
        sq = Square().shift(LEFT)
        self.add(sq)
        self.play(TransformFromCopy(sq, Circle().shift(RIGHT)), run_time=0.8)


class Successions(Scene):
    """Successions, with part boundaries at exact instants and found by
    bisection: they keep the scalar path.
    """

    def construct(self):
        a, b = Square().shift(LEFT), Circle().shift(RIGHT)
        self.add(a, b)
        self.play(Succession(a.animate.shift(UP), Transform(b, Triangle())), run_time=1)
        self.play(
            Succession(Create(Square()), a.animate.shift(DOWN), rate_func=smooth),
            run_time=1,
        )


class Mixed(Scene):
    """A trusted play with a leaf outside the vectorized kinds (Rotate)
    beside a blend: the whole play stays scalar.
    """

    def construct(self):
        a, b = Square().shift(LEFT), Square().shift(RIGHT)
        self.add(a, b)
        self.play(Rotate(a, angle=np.pi / 3), b.animate.shift(UP), run_time=0.8)


class Brief(Scene):
    """A play shorter than a frame: one kept instant, nothing to batch."""

    def construct(self):
        sq = Square()
        self.add(sq)
        self.play(sq.animate.shift(UP), run_time=0.01)
        self.play(sq.animate.shift(RIGHT), run_time=0.5)
