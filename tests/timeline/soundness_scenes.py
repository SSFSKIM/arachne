"""Scenes whose callbacks or mobjects look admissible but are not: each
changes what a frame shows depending on which instants are evaluated, or
runs scene code during an evaluation, so its play must tick.
"""

from __future__ import annotations

import functools
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
    NumberLine,
    Rotate,
    Scene,
    Square,
    Transform,
    UpdateFromAlphaFunc,
    ValueTracker,
    VGroup,
    ZoomedScene,
    always_redraw,
    color_to_rgb,
    linear,
    random_color,
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


# What the prover once admitted by default.


class PutAlongAxis(_Ramp):
    """A numpy function that writes into the closure array it is given."""

    def attach(self, dot, t):
        p = np.zeros(3)

        def drift():
            np.put_along_axis(p, np.array([0]), p[:1] + 0.02, axis=0)
            return p

        dot.add_updater(lambda m: m.move_to(drift()))


class RandomColour(_Ramp):
    """An engine helper that draws from the random generator's hidden state."""

    def attach(self, dot, t):
        dot.add_updater(lambda m: m.set_color(random_color()))


class PickedNone(Scene):
    """A fill picked from ``[None, RED]``: None outside 0.3 < alpha < 0.4,
    which leaves the fill as it was.
    """

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d, lambda m, a: m.set_fill([None, RED][int(0.3 < a < 0.4)])
            )
        )


class BoundReceiver(Scene):
    """A graph whose function is a number line's ``n2p``: the dot reading the
    graph reads the line, whose own updater runs after the dot's.
    """

    def construct(self):
        t = ValueTracker(0)
        ax = Axes(x_range=[0, 3], y_range=[0, 3], x_length=5, y_length=3)
        graph = ax.plot(lambda x: x / 2)
        line = NumberLine(x_range=[0, 3], length=3)
        graph.function = line.n2p
        dot = Dot()
        dot.add_updater(lambda m: m.move_to(ax.i2gp(0, graph)))
        line.add_updater(lambda m: m.set_y(t.get_value()))
        self.add(ax, graph, dot, line)
        self.play(t.animate.set_value(2))


class InstanceInterpolate(Scene):
    """A linear Transform of a dot whose ``interpolate`` is replaced on the
    instance by one that counts its calls.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        original = d.interpolate

        def counted(*args, **kwargs):
            InstanceInterpolate.calls.append(1)
            return original(*args, **kwargs)

        d.interpolate = counted
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class InstanceSetFill(_Ramp):
    """A group ``set_fill`` updater whose child's ``set_fill`` is replaced on
    the instance by one that also moves it a step per call.
    """

    def attach(self, dot, t):
        d = Square(fill_opacity=1)
        original = d.set_fill

        def stepping(*args, **kwargs):
            d.shift(0.02 * RIGHT)
            return original(*args, **kwargs)

        d.set_fill = stepping
        group = VGroup(d)
        group.add_updater(lambda m: m.set_fill(RED, t.get_value()))
        self.add(group)


# Found reviewing what else the prover admitted by default.


class InstanceAnimation(Scene):
    """A linear Transform whose own ``interpolate_mobject`` is replaced on
    the animation, counting its calls.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        animation = Transform(d, Dot(RIGHT * 2), rate_func=linear)
        original = animation.interpolate_mobject

        def counted(alpha):
            InstanceAnimation.calls.append(1)
            return original(alpha)

        animation.interpolate_mobject = counted
        self.add(d)
        self.play(animation)


class ShuffledBuilder(_Ramp):
    """A redraw that shuffles the group it builds: a draw from the random
    generator, at every evaluation.
    """

    def attach(self, dot, t):
        shown = always_redraw(
            lambda: (
                VGroup(*[Dot(radius=0.04 * k) for k in range(1, 6)]).shuffle().arrange()
            )
        )
        self.add(shown)


class InstanceUpdate(_Ramp):
    """A dot whose ``update``, which the scene calls at every tick, is
    replaced on the instance by one that also moves it a step.
    """

    def attach(self, dot, t):
        original = dot.update

        def stepping(dt=0, recursive=True):
            dot.shift(0.02 * RIGHT)
            return original(dt, recursive)

        dot.update = stepping


class PatchedEngine(Scene):
    """A linear Transform while the engine's ``Dot.interpolate`` is replaced
    by one that counts its calls (put back once the scene ends).
    """

    calls: list[int] = []

    def construct(self):
        original = Dot.interpolate

        def counted(mob, *args, **kwargs):
            PatchedEngine.calls.append(1)
            return original(mob, *args, **kwargs)

        Dot.interpolate = counted
        try:
            d = Dot()
            self.add(d)
            self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)
        finally:
            del Dot.interpolate


class CountingFill(Dot):
    """A user's dot whose fill the camera reads from a class value the
    script rebinds, counting the reads.
    """

    colour = RED
    calls: list[int] = []

    def get_fill_rgbas(self):
        CountingFill.calls.append(1)
        return np.array([[*color_to_rgb(CountingFill.colour), 1.0]])


class OverriddenFill(Scene):
    """That dot drawn through two waits, red then blue: drawn after the
    scene ran, the first wait would show blue.
    """

    def construct(self):
        CountingFill.colour = RED
        d = CountingFill(radius=0.3)
        self.add(d)
        self.wait(0.5)
        CountingFill.colour = BLUE
        self.wait(0.5)


class FillJoinsMidPlay(Scene):
    """That dot, red, added by an updater partway through a play."""

    def construct(self):
        CountingFill.colour = RED
        t = ValueTracker(0)
        holder = VGroup()

        def join(m, dt):
            if not m.submobjects and t.get_value() > 0.4:
                m.add(CountingFill(radius=0.3))

        holder.add_updater(join)
        self.add(holder)
        self.play(t.animate.set_value(1), rate_func=linear)


# Identity, not metadata.


class WrappedInterpolate(Scene):
    """``InstanceInterpolate`` with its counter dressed as the engine's
    method by ``functools.wraps``.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        original = d.interpolate

        @functools.wraps(original)
        def counted(*args, **kwargs):
            WrappedInterpolate.calls.append(1)
            return original(*args, **kwargs)

        d.interpolate = counted
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class WrappedPatch(Scene):
    """``PatchedEngine`` with its counter dressed as the engine's
    ``interpolate`` by ``functools.wraps``.
    """

    calls: list[int] = []

    def construct(self):
        original = Dot.interpolate

        @functools.wraps(original)
        def counted(mob, *args, **kwargs):
            WrappedPatch.calls.append(1)
            return original(mob, *args, **kwargs)

        Dot.interpolate = counted
        try:
            d = Dot()
            self.add(d)
            self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)
        finally:
            del Dot.interpolate


# What an array holds.


class ObjectArrayFill(Scene):
    """``PickedNone`` with the choice made by an array of objects."""

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d,
                lambda m, a: m.set_fill(
                    np.array([None, RED], dtype="O")[int(0.3 < a < 0.4)]
                ),
            )
        )


# What drawing runs: protocols, and mobjects that come and go mid-play.


class HashDot(Dot):
    """A user's dot whose class defines ``__hash__``, counting its calls."""

    calls: list[int] = []

    def __hash__(self):
        HashDot.calls.append(1)
        return object.__hash__(self)


class HashedDot(Scene):
    """That dot moved, then held."""

    dot_class: type = HashDot

    def construct(self):
        d = self.dot_class(radius=0.3)
        self.add(d)
        self.play(d.animate.shift(RIGHT), rate_func=linear)
        self.wait(0.5)


def play_fill_between(scene, low, high):
    """A second's play during which an updater holds a ``CountingFill``
    while its tracker is in ``[low, high)``: the dot joins and leaves within
    the play.
    """
    t = ValueTracker(0)
    holder = VGroup()

    def show(m):
        inside = low <= t.get_value() < high
        if inside and not m.submobjects:
            m.add(CountingFill(radius=0.3))
        elif not inside and m.submobjects:
            m.remove(*m.submobjects)

    holder.add_updater(show)
    scene.add(holder)
    scene.play(t.animate.set_value(1), rate_func=linear)
    scene.remove(holder)


class TransientFill(Scene):
    """A red ``CountingFill`` that joins and leaves within play 0, then a
    wait.
    """

    def construct(self):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)
        self.wait(0.5)


class TwoTransientFills(Scene):
    """``TransientFill``'s play, a wait, and the same play in blue."""

    def construct(self):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)
        self.wait(0.5)
        CountingFill.colour = BLUE
        play_fill_between(self, 0.3, 0.6)


class CaughtTransientFill(Scene):
    """``TransientFill``'s play inside a ``try`` that catches every error."""

    def construct(self):
        CountingFill.colour = RED
        try:  # noqa: SIM105 (written as scenes write it)
            play_fill_between(self, 0.3, 0.6)
        except Exception:  # noqa: BLE001, S110 (a scene that swallows errors)
            pass
        self.wait(0.5)


class DriftingTransientFill(Scene):
    """``TransientFill`` beside a square placed by how many times the scene
    has run: each execution begins its play from another state.
    """

    runs: list[int] = []

    def construct(self):
        DriftingTransientFill.runs.append(1)
        CountingFill.colour = RED
        self.add(Square().shift(RIGHT * len(DriftingTransientFill.runs)))
        play_fill_between(self, 0.3, 0.6)
