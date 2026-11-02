"""Scenes the stepping-tick tests execute: ticked plays whose dt-free work
runs at the frames, or stays on the tick because stepping work reads it.

A ticked play here ticks because a ``ValueTracker`` (drawn as nothing)
carries a dt updater, unless the scene says otherwise.
"""

from __future__ import annotations

from builtins import getattr as lookup

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    ORIGIN,
    PURPLE,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Animation,
    Dot,
    FadeIn,
    Line,
    Mobject,
    MoveAlongPath,
    Scene,
    Square,
    Succession,
    TracedPath,
    Transform,
    UpdateFromAlphaFunc,
    UpdateFromFunc,
    ValueTracker,
    VGroup,
    VMobject,
    Wait,
    Wiggle,
    always_redraw,
    config,
    linear,
)
from arachne.utils.bezier import interpolate
from tests.timeline import stepping_classes, stepping_producer
from tests.timeline import stepping_shared as shared

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


# Paths to what a stepping reader reads, each resolved or counted as
# reaching anything. Each play also holds a dot whose dt-free updater nothing
# reads, so that its other ticks step.


def _idle() -> Dot:
    """A dot whose dt-free updater nothing reads: the plan skips it."""
    idle = Dot(color=BLUE, radius=0.05).shift(2 * DOWN)
    idle.add_updater(lambda m: m.set_x(2))
    return idle


class OwnerWrite(Scene):
    """A dt-free updater writes, through an attribute of the mobject it
    updates, the dot a dt updater reads.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        holder = Dot(color=BLUE)
        holder.source = source
        holder.add_updater(lambda m: m.source.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        self.add(t, holder, source, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ContainerWrite(Scene):
    """A dt-free updater writes the dot a dt updater reads through a list it
    makes: the list is its own, what it holds is not.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        holder = Dot(color=BLUE)

        def update(m):
            values = [source]
            values[0].set_x(t.get_value())

        holder.add_updater(update)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        self.add(t, holder, source, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ReusedFactory(Scene):
    """Two dt-free updaters made by one function, each writing the dot it
    closes over: the second writes the dot a dt updater reads.
    """

    def construct(self):
        t = ValueTracker(0)
        first, second = Dot(color=BLUE), Dot(color=GREEN)

        def writer(target):
            return lambda m: target.set_x(t.get_value())

        one, two = Dot(radius=0.05), Dot(radius=0.05)
        one.add_updater(writer(first))
        two.add_updater(writer(second))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * second.get_x() * RIGHT))
        self.add(t, first, second, one, two, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ClassRead(Scene):
    """A dt updater reads a dot through an attribute its holder's class
    holds, not the holder.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))

        class Holder:
            pass

        Holder.source = source
        holder = Holder()
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * holder.source.get_x() * RIGHT))
        self.add(t, source, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ArrayAlias(Scene):
    """A dt-free updater writes an array; a dt updater reads a view of it."""

    def construct(self):
        t = ValueTracker(0)
        original = np.array([0.0])
        view = original[:]
        holder = Dot(color=BLUE)

        def update(m):
            original[0] = t.get_value()

        holder.add_updater(update)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * view[0] * RIGHT))
        self.add(t, holder, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ModuleAlias(Scene):
    """A dt-free updater writes a module's attribute under one alias; a dt
    updater reads it under another.
    """

    def construct(self):
        shared.value = 0.0
        t = ValueTracker(0)
        holder = Dot(color=BLUE)
        holder.add_updater(stepping_producer.publish(t))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * shared.value * RIGHT))
        self.add(t, holder, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class PropertyRead(Scene):
    """A dt updater reads a property of a dot a dt-free updater resizes."""

    def construct(self):
        t = ValueTracker(0)
        other = Dot()
        other.add_updater(lambda m: m.scale_to_fit_width(0.25 + t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * other.width * RIGHT))
        self.add(t, other, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class MethodRead(Scene):
    """A dt updater reads a dot a dt-free updater moves, through a method."""

    def construct(self):
        t = ValueTracker(0)
        other = Dot()
        other.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * other.get_center()[0] * RIGHT))
        self.add(t, other, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class GroupRead(Scene):
    """A dt updater reads a group whose member a dt-free updater moves."""

    def construct(self):
        t = ValueTracker(0)
        other = Dot()
        other.add_updater(lambda m: m.set_x(t.get_value()))
        group = VGroup(other)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * group.get_center()[0] * RIGHT))
        self.add(t, group, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


# Callbacks that change which updaters there are


class NewReader(Scene):
    """An ``UpdateFromAlphaFunc`` gives a dot, a quarter of the way in, a dt
    updater reading a dot a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)

        def install(m, alpha):
            if alpha >= 0.25 and not m.updaters:
                m.add_updater(lambda mob, dt: mob.shift(dt * source.get_x() * RIGHT))

        self.add(t, source, output, _idle())
        self.play(
            t.animate.set_value(1),
            UpdateFromAlphaFunc(output, install),
            rate_func=linear,
        )


class HiddenInstaller(Scene):
    """An ``UpdateFromAlphaFunc`` calls, through a dict, a dot's
    ``add_updater`` it holds as a value: a call its source can't resolve,
    into what changes the structure.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        actions = {"install": output.add_updater}

        def reader(mob, dt):
            mob.shift(dt * source.get_x() * RIGHT)

        def install(m, alpha):
            if alpha >= 0.25 and not output.updaters:
                actions["install"](reader)

        self.add(t, source, output, _idle())
        self.play(
            t.animate.set_value(1),
            UpdateFromAlphaFunc(output, install),
            rate_func=linear,
        )


class GetterGuard(Scene):
    """A dt updater checks two views of one quantity through getters it
    unpacks from pairs, as the plugin's guards do: calls its source can't
    resolve, into code that changes no structure.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        sides = [("dot", source.get_x), ("tracker", t.get_value)]

        def check(m, dt):
            if dt > 0:
                values = [get() for _, get in sides]
                assert abs(values[0] - values[1]) < 1e-9

        guard = Mobject()
        guard.add_updater(check)
        self.add(t, source, guard, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class SkippedInstaller(Scene):
    """A dt-free updater nothing reads gives a dot, a fifth of the way in (a
    frame at 15 fps), a dt updater reading a dot a dt-free updater moves:
    the play plans again after that frame's full tick.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)

        def install(m):
            if t.get_value() >= 0.2 - 1e-9 and not output.updaters:
                output.add_updater(
                    lambda mob, dt: mob.shift(dt * source.get_x() * RIGHT)
                )

        installer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        installer.add_updater(install)
        self.add(t, source, output, installer, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class SkippedAdder(Scene):
    """A dt-free updater nothing reads adds to the scene, a fifth of the way
    in, a dot carrying a dt updater that reads a dot a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        reader = Dot(color=RED, radius=0.05)
        reader.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        scene = self

        def add(m):
            if t.get_value() >= 0.2 - 1e-9 and reader not in scene.mobjects:
                scene.add(reader)

        adder = Dot(color=GREEN, radius=0.05).shift(DOWN)
        adder.add_updater(add)
        self.add(t, source, adder, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class MemberJoins(Scene):
    """A dt updater reads a group's centre; a fifth of the way in, a
    dt-free updater nothing reads puts into the group a dot another dt-free
    updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        dot = Dot()
        mover = Dot(color=BLUE, radius=0.05).shift(UP)
        mover.add_updater(lambda m: dot.set_x(t.get_value()))
        group = VGroup(Dot(LEFT))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * group.get_center()[0] * UP))

        def join(m):
            if t.get_value() >= 0.2 - 1e-9 and dot not in group.submobjects:
                group.add(dot)

        joiner = Dot(color=GREEN, radius=0.05).shift(DOWN)
        joiner.add_updater(join)
        self.add(t, mover, group, output, joiner, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class SubmobjectAdded(Scene):
    """A dt-free updater nothing reads adds to a group, a fifth of the way
    in, a dot carrying a dt updater that reads a dot a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * source.get_x() * UP))
        group = VGroup(Dot(LEFT))

        def add(m):
            if t.get_value() >= 0.2 - 1e-9 and child not in group.submobjects:
                group.add(child)

        adder = Dot(color=GREEN, radius=0.05).shift(DOWN)
        adder.add_updater(add)
        self.add(t, source, group, adder, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class SubmobjectDropped(Scene):
    """A dt-free updater drops from a group, a fifth of the way in, a dot
    whose dt updater reads a dot a dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * source.get_x() * UP))
        group = VGroup(Dot(LEFT), child)

        def drop(m):
            if t.get_value() >= 0.2 - 1e-9 and child in group.submobjects:
                group.remove(child)

        dropper = Dot(color=GREEN, radius=0.05).shift(DOWN)
        dropper.add_updater(drop)
        self.add(t, source, group, dropper, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class ReaderRemoved(Scene):
    """An ``UpdateFromAlphaFunc`` takes, half way in, the dt updater from a
    dot that reads a dot whose dt-free updater keeps history.
    """

    def construct(self):
        source = Dot()
        source.add_updater(lambda m: m.shift(0.02 * RIGHT))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * source.get_x() * UP))

        def remove(m, alpha):
            if alpha >= 0.5:
                m.clear_updaters()

        self.add(source, output, _idle())
        self.play(UpdateFromAlphaFunc(output, remove), rate_func=linear)


class AliasedLookup(Scene):
    """A dt updater reads an attribute it names at run time through
    ``getattr`` under another name: its reads can't be bounded.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        holder = Dot(color=BLUE)
        holder.source = source
        name = "source"
        output = Dot(color=RED, radius=0.05)
        output.add_updater(
            lambda m, dt: m.shift(dt * lookup(holder, name).get_x() * RIGHT)
        )
        self.add(t, source, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


# Interpolation a stepping reader reads


def _middle_reader(middle) -> Dot:
    """A dot whose dt updater moves right at the speed of ``middle``'s y."""
    output = Dot(color=RED, radius=0.05)
    output.add_updater(lambda m, dt: m.shift(dt * middle.get_y() * RIGHT))
    return output


class AlphaReader(Scene):
    """A dt updater reads a dot an ``UpdateFromAlphaFunc`` places at a dot a
    dt-free updater moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_y(t.get_value()))
        middle = Dot(color=BLUE)
        self.add(t, source, middle, _middle_reader(middle), _idle())
        self.play(
            t.animate.set_value(1),
            UpdateFromAlphaFunc(middle, lambda m, a: m.set_y(source.get_y())),
            rate_func=linear,
        )


class TargetReader(Scene):
    """A dt updater reads a dot transforming into a dot a dt-free updater
    moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_y(t.get_value()))
        middle = Dot(color=BLUE)
        self.add(t, source, middle, _middle_reader(middle), _idle())
        self.play(t.animate.set_value(1), Transform(middle, source), rate_func=linear)


class WiggleReader(Scene):
    """A dt updater reads a square that wiggles while a dt-free updater
    moves it.
    """

    def construct(self):
        t = ValueTracker(0)
        middle = Square(side_length=0.5)
        middle.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * middle.get_x() * UP))
        self.add(t, middle, output, _idle())
        self.play(
            t.animate.set_value(1),
            Wiggle(middle, suspend_mobject_updating=False),
            rate_func=linear,
        )


class ConstantWrite(Scene):
    """A dt-free updater writes in place into its dot's centre, which is the
    engine's ``ORIGIN`` every dot made with a default holds (writable, as in
    CE); a dt updater reads ``ORIGIN``.
    """

    def construct(self):
        t = ValueTracker(0)
        holder = Dot(color=BLUE)

        def update(m):
            m.arc_center[0] = t.get_value()

        holder.add_updater(update)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * ORIGIN[0] * RIGHT))
        self.add(t, holder, output, _idle())
        try:
            self.play(t.animate.set_value(1), rate_func=linear)
        finally:
            ORIGIN[0] = 0.0


class StartInstall(Scene):
    """A dt-free updater nothing reads gives a dot, when the play first
    evaluates (its start, a full instant), a dt updater that reads a dot a
    dt-free updater moves.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        output = Dot(color=RED, radius=0.05)

        def install(m):
            if not output.updaters:
                output.add_updater(lambda o, dt: o.shift(dt * source.get_x() * RIGHT))

        installer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        installer.add_updater(install)
        self.add(t, source, installer, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class AliasAppend(Scene):
    """A dt-free updater nothing reads appends, a fifth of the way in, a dot
    carrying a dt reader to a group's members, through a local name for the
    list.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        group = VGroup()

        def install(m):
            if t.get_value() >= 0.2 - 1e-9 and child not in group.submobjects:
                parts = group.submobjects
                parts.append(child)

        installer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        installer.add_updater(install)
        self.add(t, source, installer, group, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class AliasAugment(Scene):
    """As :class:`AliasAppend`, with ``+=`` on the local name."""

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        group = VGroup()

        def install(m):
            if t.get_value() >= 0.2 - 1e-9 and child not in group.submobjects:
                parts = group.submobjects
                parts += [child]

        installer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        installer.add_updater(install)
        self.add(t, source, installer, group, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class BoolAlias(Scene):
    """A dt-free updater changes in place an array a dt updater reads,
    through a name an ``or`` binds to it.
    """

    def construct(self):
        original = np.array([1.0])
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * original[0] * RIGHT))

        def update(m):
            value = original or original
            value += 0.01

        producer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        producer.add_updater(update)
        self.add(producer, output, _idle())
        self.wait(1)


class IfAlias(Scene):
    """As :class:`BoolAlias`, through a conditional expression."""

    def construct(self):
        original = np.array([1.0])
        spare = np.array([1.0])
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * original[0] * RIGHT))

        def update(m):
            value = original if m.get_x() < 1 else spare
            value += 0.01

        producer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        producer.add_updater(update)
        self.add(producer, output, _idle())
        self.wait(1)


class BecomeRoot(Scene):
    """A dot whose own dt-free updater has it become, a quarter of the way
    in, a group of two: the copies of itself ``become`` pads it with carry
    that updater, which from then on moves them.
    """

    def construct(self):
        clock(self)
        root = Dot(color=RED, radius=0.05)

        def update(m):
            if m is root:
                if self.time >= 0.25 - 1e-9 and m.has_points():
                    m.become(VGroup(Dot(LEFT), Dot(RIGHT)))
            else:
                m.shift(0.01 * RIGHT)

        root.add_updater(update)
        self.add(root, _idle())
        self.wait(1)


class ClassShadow(Scene):
    """A dt updater integrates ``holder.source``, which ``holder`` inherits
    from its class's base; a dt-free updater binds the attribute on the
    class itself, a quarter of the way in, to a tracker another dt-free
    updater moves.
    """

    def construct(self):
        a, b = ValueTracker(0), ValueTracker(0)
        a.add_updater(lambda m: m.set_value(float(self.time)))
        b.add_updater(lambda m: m.set_value(10 * float(self.time)))

        class Parent:
            pass

        class Child(Parent):
            pass

        Parent.source = a
        holder = Child()
        changer = Mobject()
        changer.add_updater(stepping_producer.rebind_source(Child, b, self))
        output = ValueTracker(0)
        output.add_updater(stepping_classes.read_source(holder))
        shown = Dot(color=RED, radius=0.05)
        shown.add_updater(lambda m: m.set_x(output.get_value()))
        self.add(a, b, changer, output, shown, _idle())
        self.wait(1)


class SetterWrite(Scene):
    """A dt-free updater assigns a property whose setter moves a tracker a
    dt updater integrates.
    """

    def construct(self):
        source = ValueTracker(0)

        class Control:
            def __init__(self, source):
                self._source = source

            @property
            def value(self):
                return 0

            @value.setter
            def value(self, v):
                self._source.set_value(v)

        control = Control(source)

        def update(m):
            control.value = float(self.time)

        producer = Mobject()
        producer.add_updater(update)
        output = ValueTracker(0)
        output.add_updater(lambda m, dt: m.increment_value(dt * source.get_value()))
        shown = Dot(color=RED, radius=0.05)
        shown.add_updater(lambda m: m.set_x(output.get_value()))
        self.add(producer, source, output, shown, _idle())
        self.wait(1)


class OwnSetter(Scene):
    """A dt-free updater assigns a property of its own mobject whose setter
    moves a tracker a dt updater integrates.
    """

    def construct(self):
        source = ValueTracker(0)

        class Knob(Dot):
            @property
            def level(self):
                return 0

            @level.setter
            def level(self, v):
                source.set_value(v)

        def update(m):
            m.level = float(self.time)

        knob = Knob(color=GREEN, radius=0.05).shift(DOWN)
        knob.add_updater(update)
        output = ValueTracker(0)
        output.add_updater(lambda m, dt: m.increment_value(dt * source.get_value()))
        shown = Dot(color=RED, radius=0.05)
        shown.add_updater(lambda m: m.set_x(output.get_value()))
        self.add(knob, source, output, shown, _idle())
        self.wait(1)


class SceneInstaller(Scene):
    """A play with no stepping work: a dt-free updater gives the scene, a
    quarter of the way in, an updater integrating a tracker a dt-free updater
    moves.
    """

    def construct(self):
        t = ValueTracker(0)
        source = ValueTracker(0)
        source.add_updater(lambda m: m.set_value(t.get_value()))
        output = ValueTracker(0)

        def install(m):
            if self.time >= 0.25 and not self.updaters:
                self.add_updater(
                    lambda dt: output.increment_value(dt * source.get_value())
                )

        installer = Mobject()
        installer.add_updater(install)
        shown = Dot(color=RED, radius=0.05)
        shown.add_updater(lambda m: m.set_x(output.get_value()))
        self.add(t, source, output, installer, shown, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class EventAdd(Scene):
    """A part of a ``Succession`` fades in, a quarter of the way in, a dot
    whose dt updater reads a dot a dt-free updater moves.
    """

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * source.get_x() * RIGHT))
        self.add(t, source, _idle())
        self.play(
            t.animate.set_value(1),
            Succession(Wait(0.25), FadeIn(child, run_time=0.75)),
            rate_func=linear,
        )


class AddRemove(Scene):
    """A dt-free updater adds a dot carrying a dt updater to a group and
    takes it out again, within each run: the structure is the same after.
    """

    def construct(self):
        clock(self)
        child = Dot(color=RED, radius=0.05)
        child.add_updater(lambda m, dt: m.shift(dt * RIGHT))
        group = VGroup()

        def both(m):
            group.add(child)
            group.remove(child)

        installer = Dot(color=GREEN, radius=0.05).shift(DOWN)
        installer.add_updater(both)
        self.add(installer, group, _idle())
        self.wait(1)


class GroupBecome(Scene):
    """A group's dt-free updater has it become, a quarter of the way in, a
    group of two: ``become`` pads it with copies of its member, whose own
    dt-free updater they take along.
    """

    def construct(self):
        clock(self)
        member = Dot(color=RED, radius=0.05)
        member.add_updater(lambda m: m.shift(0.01 * RIGHT))
        group = VGroup(member)

        def grow(m):
            if self.time >= 0.25:
                m.become(VGroup(Dot(LEFT), Dot(RIGHT)))

        group.add_updater(grow)
        self.add(group, _idle())
        self.wait(1)


class InstanceShadow(Scene):
    """As :class:`ClassShadow`, with the attribute bound on the instance,
    which shadows the class's for it alone.
    """

    def construct(self):
        a, b = ValueTracker(0), ValueTracker(0)
        a.add_updater(lambda m: m.set_value(float(self.time)))
        b.add_updater(lambda m: m.set_value(10 * float(self.time)))

        class Parent:
            pass

        class Child(Parent):
            pass

        Parent.source = a
        holder = Child()
        changer = Mobject()
        changer.add_updater(stepping_producer.rebind_source(holder, b, self))
        output = ValueTracker(0)
        output.add_updater(stepping_classes.read_source(holder))
        shown = Dot(color=RED, radius=0.05)
        shown.add_updater(lambda m: m.set_x(output.get_value()))
        self.add(a, b, changer, output, shown, _idle())
        self.wait(1)


class NativeAlias(Scene):
    """A dt-free updater appends to a list through ``push = values.append``;
    a dt updater integrates the list's last item.
    """

    def construct(self):
        t = ValueTracker(0)
        values = [0.0]
        push = values.append
        producer = Mobject()
        producer.add_updater(lambda m: push(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * values[-1] * RIGHT))
        self.add(t, producer, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class EngineMethodAlias(Scene):
    """A dt-free updater moves a dot through ``move = source.set_x``; a dt
    updater integrates the dot's place.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        move = source.set_x
        producer = Mobject()
        producer.add_updater(lambda m: move(t.get_value()))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * source.get_x() * UP))
        self.add(t, source, producer, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)


class LabelSwap(Scene):
    """A dt-free updater nothing stepping reads replaces, at each frame, the
    member of a group with a new dot (a redrawn label); a dt updater reads
    another dot.
    """

    def construct(self):
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        label = VGroup(Dot(DOWN, color=GREEN))

        def redraw(m):
            m.remove(*m.submobjects)
            m.add(Dot(DOWN + t.get_value() * RIGHT, color=GREEN, radius=0.05))

        label.add_updater(redraw)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * source.get_x() * UP))
        self.add(t, source, label, output, _idle())
        self.play(t.animate.set_value(1), rate_func=linear)
