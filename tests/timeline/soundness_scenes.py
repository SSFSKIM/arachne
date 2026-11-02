"""Scenes whose callbacks or mobjects look admissible but are not: each
changes what a frame shows depending on which instants are evaluated, or
runs scene code during an evaluation, so its play must tick.
"""

from __future__ import annotations

import random

import numpy as np

from arachne import (
    BLUE,
    RED,
    RIGHT,
    UP,
    AnimationGroup,
    Axes,
    Dot,
    Mobject,
    Rotate,
    Scene,
    Square,
    Transform,
    UpdateFromAlphaFunc,
    ValueTracker,
    ZoomedScene,
    linear,
)


class _Ramp(Scene):
    """A dot whose updater runs while a tracker ramps over one second."""

    def attach(self, dot: Dot, t: ValueTracker) -> None:
        raise NotImplementedError

    def construct(self):
        t = ValueTracker(0)
        dot = Dot()
        self.add(dot)
        self.attach(dot, t)
        self.play(t.animate.set_value(1))


class ArrayAugmented(_Ramp):
    """An alias of a closure array changed in place: it drifts by one step
    per evaluation.
    """

    def attach(self, dot, t):
        p = np.zeros(3)

        def step(m):
            q = p
            q += 0.02 * RIGHT
            m.move_to(q)

        dot.add_updater(step)


class ArrayOutKeyword(_Ramp):
    """A ufunc writing into a closure array through ``out=``."""

    def attach(self, dot, t):
        p = np.zeros(3)
        dot.add_updater(lambda m: m.move_to(np.add(p, 0.02 * RIGHT, out=p)))


class ArrayOutPositional(_Ramp):
    """The same write, with the output array passed positionally."""

    def attach(self, dot, t):
        p = np.zeros(3)
        dot.add_updater(lambda m: m.move_to(np.add(p, 0.02 * RIGHT, p)))


class PartialFill(Scene):
    """A fill set only between alpha 0.3 and 0.4, left as it was elsewhere."""

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d, lambda m, a: m.set_fill(RED if 0.3 < a < 0.4 else None)
            )
        )


class NoneFill(Scene):
    """A fill colour that is None: the setter leaves the colour as it was."""

    def construct(self):
        d = Square(fill_opacity=1)
        colour = None
        self.add(d)
        self.play(UpdateFromAlphaFunc(d, lambda m, a: m.set_fill(colour, a)))


class NarrowedColour(Scene):
    """``family=False``: outside the domain the setter's test covers."""

    def construct(self):
        d = Square()
        self.add(d)
        self.play(UpdateFromAlphaFunc(d, lambda m, a: m.set_color(RED, family=False)))


class RandomRate(Scene):
    """A callback animation timed by a rate function that draws at random."""

    def construct(self):
        d = Dot()
        self.add(d)
        self.play(
            AnimationGroup(
                UpdateFromAlphaFunc(
                    d, lambda m, a: m.set_x(a), rate_func=lambda t: random.random()
                )
            )
        )


class CallbackOverlap(Scene):
    """A Transform of ``d`` over 2 s while a callback part also sets ``d``
    for its first 0.8 s: CE stops the shorter part at the first frame after
    0.8 s, and the Transform overwrites from then on.
    """

    def construct(self):
        d = Dot()
        self.add(d)
        self.play(
            AnimationGroup(
                Transform(d, Dot(RIGHT), run_time=2),
                UpdateFromAlphaFunc(
                    Mobject(), lambda m, a: d.set_x(2 * a), run_time=0.8
                ),
            )
        )


class CountingGraph(Scene):
    """A dot on a graph whose function counts its calls."""

    def construct(self):
        calls = []

        def f(x):
            calls.append(1)
            return x * x / 4 + len(calls) / 100

        t = ValueTracker(0)
        axes = Axes(x_range=[0, 3], y_range=[0, 3], x_length=5, y_length=3)
        graph = axes.plot(lambda x: x * x / 4)
        graph.underlying_function = f
        dot = Dot()
        dot.add_updater(lambda m: m.move_to(axes.i2gp(t.get_value(), graph)))
        self.add(axes, graph, dot)
        self.play(t.animate.set_value(2))


class CountingDot(Dot):
    """A user's dot whose ``interpolate`` counts its calls."""

    calls: list[int] = []

    def interpolate(self, *args, **kwargs):
        CountingDot.calls.append(1)
        return super().interpolate(*args, **kwargs)


class CountedTransform(Scene):
    """A linear Transform of that dot: every type and function in it is the
    algebra's but the dot's own ``interpolate``.
    """

    def construct(self):
        d = CountingDot()
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class ViewCentre(Scene):
    """``Rotate`` about a read-only view of an array the script changes
    after the play.
    """

    def construct(self):
        p = np.array([1.0, 0.0, 0.0])
        c = p.view()
        c.flags.writeable = False
        square = Square()
        self.add(square)
        self.play(Rotate(square, angle=np.pi / 2, about_point=c))
        p[0] = 5


class ZoomBackground(ZoomedScene):
    """The zoomed camera's background turns from red to blue at 1/2 s."""

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=3, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(UP * 0.1)
        self.add(dot)
        self.zoomed_camera.frame.move_to(dot)
        self.activate_zooming()
        self.zoomed_camera.background_color = RED
        self.play(dot.animate.shift(RIGHT * 0.1), run_time=0.5)
        self.zoomed_camera.background_color = BLUE
        self.play(dot.animate.shift(RIGHT * 0.1), run_time=0.5)
