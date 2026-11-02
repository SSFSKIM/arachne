"""Scenes the stepping-tick tests execute: ticked plays whose dt-free work
runs at the frames, or stays on the tick because stepping work reads it.

A ticked play here ticks because a ``ValueTracker`` (drawn as nothing)
carries a dt updater, unless the scene says otherwise.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Animation,
    Dot,
    FadeIn,
    Line,
    MoveAlongPath,
    Scene,
    Square,
    Succession,
    TracedPath,
    UpdateFromFunc,
    ValueTracker,
    VGroup,
    VMobject,
    Wait,
    always_redraw,
    config,
    linear,
)
from arachne.utils.bezier import interpolate

COLOURS = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE] * 3


def clock(scene: Scene) -> ValueTracker:
    """A tracker that integrates time: it makes its plays tick, and nothing
    reads it.
    """
    clock = ValueTracker(0)
    clock.add_updater(lambda m, dt: m.increment_value(dt))
    scene.add(clock)
    return clock


class Unchanged(Scene):
    """dt-free work that keeps no history: a readout redrawn from a tracker,
    a dot following another. Skipping it between frames changes no frame.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        dot = Dot(LEFT)
        follower = Dot(color=RED, radius=0.05)
        follower.add_updater(lambda m: m.next_to(dot, UP))
        bar = always_redraw(lambda: Square(side_length=0.2 + t.get_value()))
        spinner = Square(side_length=0.5).shift(2 * RIGHT)
        spinner.add_updater(lambda m, dt: m.rotate(dt))
        self.add(dot, follower, bar, spinner)
        self.play(
            t.animate.set_value(1), dot.animate.shift(2 * RIGHT), rate_func=linear
        )


class TracedReader(Scene):
    """Two traces of dots that dt-free updaters move: each trace reads its
    dot at every tick, one through a bound method, one through a lambda.
    """

    def construct(self):
        t = ValueTracker(0)
        a = Dot()
        b = Dot(color=RED)
        a.add_updater(lambda m: m.move_to(np.array([np.cos(4 * t.get_value()), 0, 0])))
        b.add_updater(lambda m: m.move_to(np.array([0, np.sin(4 * t.get_value()), 0])))
        trace_a = TracedPath(a.get_center)
        trace_b = TracedPath(lambda: b.get_center() + UP)
        self.add(a, b, trace_a, trace_b)
        self.play(t.animate.set_value(1), rate_func=linear)


class TracedCollapse(Scene):
    """``route_scenes.Collapse`` with a trace of the path's end: a stepping
    reader, so the path's updater runs at every tick and every rate sees
    the collapse at 0.3 s.
    """

    def construct(self):
        t = ValueTracker(-0.3)
        path = VMobject().set_points_as_corners([LEFT, 0.5 * UP, RIGHT])
        path.add_updater(
            lambda m: m.put_start_and_end_on(LEFT, LEFT + RIGHT * abs(t.get_value()))
        )
        self.add(path, TracedPath(path.get_end))
        self.play(t.animate.set_value(0.7), rate_func=linear)


class StaleGuard(Scene):
    """A guard that checks a redraw against what it is drawn from, with a dt
    parameter it never reads: dt-free, it runs where the redraw does.
    """

    calls: list[float] = []

    def construct(self):
        clock(self)
        dot = Dot(LEFT)
        shown = always_redraw(lambda: Dot(color=RED).move_to(dot))
        calls = type(self).calls

        def check(m, dt):
            calls.append(0)
            if not np.allclose(shown.get_center(), dot.get_center()):
                raise AssertionError("the redraw is stale")

        guard = Dot(radius=0.01)
        guard.add_updater(check)
        self.add(dot, shown, guard)
        self.play(dot.animate.shift(3 * RIGHT), rate_func=linear)


def _staircase(t: ValueTracker, start: int, steps: int, shift: np.ndarray) -> VGroup:
    """``start`` overlapping squares and one more every ``1/steps`` of the
    tracker's range, each raised above the last: built anew, its members'
    order on screen is theirs; given to a redraw's target by ``become``, it
    is the target's, which keeps the members it was padded with.
    """
    count = start + int(steps * t.get_value())
    return VGroup(
        *[
            Square(side_length=1, fill_opacity=1, color=COLOURS[k])
            .shift(0.25 * k * RIGHT + shift)
            .set_z_index(k)
            for k in range(count)
        ]
    )


LOW = (3, 9, 2 * LEFT + UP)
HIGH = (2, 12, 2 * LEFT + DOWN)


class Staircases(Scene):
    """Two redraws whose targets keep history through ``become``'s padding:
    what they show depends on the instants they were built at.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        low = always_redraw(lambda: _staircase(t, *LOW))
        high = always_redraw(lambda: _staircase(t, *HIGH))
        self.add(low, high)
        self.play(t.animate.set_value(0.99), rate_func=linear)


class StaircasesPerFrame(Scene):
    """``Staircases`` as CE draws it: each redraw built at the play's frames
    only (the configured frame rate's), by hand, a wait of one frame each.
    """

    def construct(self):
        fps = round(config.frame_rate)
        t = ValueTracker(0)
        low = _staircase(t, *LOW)
        high = _staircase(t, *HIGH)
        self.add(low, high)
        for k in range(fps):
            t.set_value(interpolate(0, 0.99, k / fps))
            low.become(_staircase(t, *LOW))
            high.become(_staircase(t, *HIGH))
            self.wait(1 / fps)


class DynamicRead(Scene):
    """A dt updater that reads what it names at run time: its reads can't be
    bounded, so the play ticks everything.
    """

    def construct(self):
        dot = Dot()
        name = "dot"
        shown = always_redraw(lambda: Square(side_length=0.3).next_to(dot, UP))
        mover = Dot(color=RED)
        mover.add_updater(
            lambda m, dt: m.shift(dt * RIGHT).set_y(getattr(self, name).get_y())
        )
        self.dot = dot
        self.add(dot, shown, mover)
        self.play(dot.animate.shift(UP))


class NamedRead(Scene):
    """A dt updater reading an attribute ``getattr`` names by a constant: a
    bounded read, so the play steps.
    """

    def construct(self):
        dot = Dot()
        shown = always_redraw(lambda: Square(side_length=0.3).next_to(dot, UP))
        mover = Dot(color=RED)
        mover.add_updater(
            lambda m, dt: m.shift(dt * RIGHT).set_y(getattr(dot, "radius", 0))
        )
        self.add(dot, shown, mover)
        self.play(dot.animate.shift(UP))


class OwnAnimation(Animation):
    """An animation class the scene defines: what it reads is its own code."""

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.set_x(alpha)


class OwnAnimationScene(Scene):
    def construct(self):
        clock(self)
        dot = Dot()
        shown = always_redraw(lambda: Square(side_length=0.3).next_to(dot, UP))
        self.add(dot, shown)
        self.play(OwnAnimation(dot))


class EveryTick(Scene):
    """Interpolation runs at every tick, a redraw at the frames: an
    ``UpdateFromFunc`` counts its calls.
    """

    calls: list[int] = []

    def construct(self):
        dot = Dot(LEFT)
        trace = TracedPath(dot.get_center)
        counter = Dot(radius=0.01)
        shown = always_redraw(lambda: Square(side_length=0.2).next_to(dot, UP))
        calls = type(self).calls
        self.add(dot, trace, counter, shown)
        self.play(
            dot.animate.shift(2 * RIGHT),
            UpdateFromFunc(counter, lambda m: calls.append(0)),
            rate_func=linear,
        )


class LateJoiner(Scene):
    """A part that begins mid-play fades in a dot carrying a dt-free updater
    that a trace reads: its copy's updater joins the play at the event, and
    the trace must see it stepping.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        joiner = Dot(color=RED)
        joiner.add_updater(lambda m: m.move_to(np.array([t.get_value(), 0, 0])))
        trace = TracedPath(joiner.get_center)
        label = always_redraw(lambda: Square(side_length=0.2).next_to(trace, UP))
        self.add(trace, label)
        self.play(
            t.animate.set_value(2),
            Succession(Wait(0.5), FadeIn(joiner)),
            rate_func=linear,
        )


def _wander(t: ValueTracker):
    """A dt-free updater that moves its mobject with the tracker."""
    return lambda m: m.set_x(np.sin(3 * t.get_value()))


def _rising_reader(read):
    """A dot whose dt updater rises with time and takes its x from ``read``."""
    reader = Dot(color=RED, radius=0.05)
    reader.add_updater(lambda m, dt: m.shift(dt * UP).set_x(read()))
    return reader


class AttributeChain(Scene):
    """A dt updater that reads, through an attribute of the mobject it is
    handed, a mobject a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(_wander(t))
        reader = Dot(color=RED, radius=0.05)
        reader.source = source
        reader.add_updater(lambda m, dt: m.shift(dt * UP).set_x(m.source.get_x()))
        shown = always_redraw(lambda: Square(side_length=0.2).next_to(source, DOWN))
        self.add(source, reader, shown)
        self.play(t.animate.set_value(2), rate_func=linear)


class PathReader(Scene):
    """A dt updater reading a dot ``MoveAlongPath`` moves along a path that a
    dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        path = Line(LEFT, RIGHT)
        path.add_updater(lambda m: m.move_to(UP * np.sin(3 * t.get_value())))
        dot = Dot()
        reader = _rising_reader(dot.get_y)
        shown = always_redraw(lambda: Square(side_length=0.2).next_to(dot, DOWN))
        self.add(path, dot, reader, shown)
        self.play(MoveAlongPath(dot, path), t.animate.set_value(2), rate_func=linear)


class FunctionReader(Scene):
    """A dt updater reading a dot an ``UpdateFromFunc`` places on a mobject
    that a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        helper = Dot(color=BLUE)
        helper.add_updater(_wander(t))
        dot = Dot()
        reader = _rising_reader(dot.get_x)
        shown = always_redraw(lambda: Square(side_length=0.2).next_to(dot, DOWN))
        self.add(helper, dot, reader, shown)
        self.play(
            UpdateFromFunc(dot, lambda m: m.move_to(helper.get_center() + UP)),
            t.animate.set_value(2),
            rate_func=linear,
        )


class FadeBeside(Scene):
    """A FadeIn whose mobject carries a dt-free updater, beside a dt updater
    that reads neither: the fade's reads stay out of the closure.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        square = Square()
        square.add_updater(_wander(t))
        self.play(FadeIn(square), t.animate.set_value(1), rate_func=linear)
