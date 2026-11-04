"""Scenes the timeline tests execute.

A comment ``# @name`` tags a line the tests look up by name with
:func:`tests.timeline.conftest.line_of`.
"""

from __future__ import annotations

import random

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Animation,
    Circle,
    Create,
    Dot,
    FadeIn,
    LaggedStart,
    Mobject,
    Scene,
    Square,
    Succession,
    ThreeDScene,
    Transform,
    ValueTracker,
    VGroup,
    linear,
)


class Timeline(Scene):
    """Plays, waits and markers of every kind."""

    def construct(self):
        self.next_section("intro")
        square = Square()  # @square
        self.play(Create(square), run_time=0.5)  # @create
        self.mark("beat", n=1)  # @mark
        self.wait(0.25)  # @held
        self.add_subcaption("hello", duration=1)
        dot = Dot()
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(dot)
        self.wait(0.25)  # @moving-wait
        self.wait_until(lambda: self.time >= 1)  # @wait-until
        group = VGroup(Circle(), Square().shift(DOWN))  # @group
        self.play(FadeIn(group), subcaption="fade", run_time=1 / 7)


class Sections(Scene):
    """A skipped section between two drawn ones."""

    def construct(self):
        self.play(FadeIn(Square()), run_time=0.5)
        self.next_section("skipped", skip_animations=True)
        dot = Dot()
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(dot)
        self.play(FadeIn(Circle()), run_time=0.5)
        self.next_section("shown")
        self.wait(0.5)


class Grid(Scene):
    """Three plays a tenth of a second long."""

    def construct(self):
        for _ in range(3):
            self.play(FadeIn(Square()), run_time=0.1)


class Composed(Scene):
    """Compositions with implicit run times, and one with a run time given."""

    def construct(self):
        def part(run_time):
            return Animation(Mobject(), run_time=run_time)

        self.play(Succession(part(0.1), part(0.2)))
        self.play(LaggedStart(part(1 / 3), part(1 / 3), lag_ratio=0.5))
        self.play(Succession(part(0.1), part(0.2)), run_time=0.5)


class Background(Scene):
    """The camera's background colour changes between two waits."""

    def construct(self):
        self.add(Square())
        self.camera.background_color = RED
        self.wait(0.5)
        self.camera.background_color = BLUE
        self.wait(0.5)


class SuppliedBackground(Scene):
    """The script gives the camera a red, then a blue background array
    between two waits (``Camera.set_background``).
    """

    def construct(self):
        self.add(Square())
        shape = (self.camera.pixel_height, self.camera.pixel_width, 4)
        self.camera.set_background(np.full(shape, (255, 0, 0, 255), dtype=np.uint8))
        self.wait(0.5)
        self.camera.set_background(np.full(shape, (0, 0, 255, 255), dtype=np.uint8))
        self.wait(0.5)


class WrittenBackground(Scene):
    """The script writes red into the camera's background in place."""

    def construct(self):
        self.add(Square())
        self.camera.background[:] = (255, 0, 0, 255)
        self.wait(0.5)


class Cue(Scene):
    """Waits sized from the float clock, as a cue sizes them."""

    def construct(self):
        self.wait(1 / 3)
        self.wait(1 / 3)
        self.wait(2.5 - self.renderer.time)


class DtSteps(Scene):
    """A dt updater, recording what it receives."""

    def construct(self):
        self.steps = []
        dot = Dot()
        dot.add_updater(lambda m, dt: self.steps.append(dt))
        self.add(dot)
        self.play(FadeIn(Square()), run_time=0.5)
        self.wait(1 / 7)


class Seeded(Scene):
    """Draws from the global generators."""

    def construct(self):
        self.drawn = (random.random(), np.random.random())  # noqa: NPY002


class PlayInUpdater(Scene):
    """An updater that calls play()."""

    def construct(self):
        dot = Dot()
        dot.add_updater(lambda m, dt: self.play(FadeIn(Square())))  # @inner
        self.add(dot)
        self.play(dot.animate.shift(UP))  # @outer


def _lazy_target(
    scene: Scene, part_time: float, *, ticked: bool = False
) -> tuple[Mobject, Mobject]:
    """``a`` transforms into ``b``, the second part of a Succession, while a
    parallel animation (after it in the play) moves ``b`` to the right at 6
    units a second. ``ticked`` adds a dt updater elsewhere, so the play ticks.
    """
    a = Dot(LEFT * 3)
    b = Dot()
    scene.add(a, b)
    if ticked:
        scene.add(Mobject().add_updater(lambda m, dt: None))
    scene.play(
        Succession(
            Animation(Mobject(), run_time=part_time),
            Transform(a, b, run_time=1 - part_time, rate_func=linear),
        ),
        b.animate(rate_func=linear, run_time=1).shift(RIGHT * 6),
    )
    return a, b


class LazyTargetOnTick(Scene):
    """The Succession's second part begins at 1/2, on the tick."""

    def construct(self):
        self.a, self.b = _lazy_target(self, 0.5)


class LazyTargetOffTick(Scene):
    """The Succession's second part begins at 1/7, between ticks: a trusted
    play, so at 1/7 exactly.
    """

    def construct(self):
        self.a, self.b = _lazy_target(self, 1 / 7)


class LazyTargetOffTickTicked(Scene):
    """As LazyTargetOffTick, with a dt updater in the scene: the play ticks,
    and the part begins at the first tick after 1/7.
    """

    def construct(self):
        self.a, self.b = _lazy_target(self, 1 / 7, ticked=True)


class Padding(Scene):
    """A group of one transformed into a group of three."""

    def construct(self):
        tracker = ValueTracker(0)
        self.add(tracker)
        one = VGroup(Square())  # @one
        three = VGroup(*(Circle().shift(RIGHT * i) for i in range(3)))
        self.play(Transform(one, three), run_time=1)


class Copies(Scene):
    """A mobject and its copy."""

    def construct(self):
        square = Square()
        self.add(square)
        twin = square.copy().shift(RIGHT)  # @copy
        self.add(twin)
        self.wait(0.1)


class Drawn(Scene):
    """Static and moving mobjects, a frozen wait, an updater and a
    Succession, for frames drawn from the record.
    """

    def construct(self):
        self.add(Square().shift(LEFT * 2))
        circle = Circle()
        self.play(Create(circle), run_time=0.5)
        self.wait(0.5)
        dot = Dot()
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(dot)
        self.play(
            Succession(FadeIn(Square().shift(UP)), circle.animate.shift(DOWN)),
            run_time=1.1,
        )
        self.wait(1 / 3)


class FixedOrientation(ThreeDScene):
    """A fixed-orientation dot whose centre function reads a variable the
    script rebinds between two waits.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        pivot = LEFT * 3
        dot = Dot(radius=0.3)
        self.add_fixed_orientation_mobjects(dot, center_func=lambda: pivot)
        self.wait(0.5)
        pivot = RIGHT * 3
        self.wait(0.5)


class Captured(Scene):
    """A group whose members are never added on their own, a dot added and
    then removed, a square made and never added, and a circle first shown
    in the last play.
    """

    def construct(self):
        self.group = VGroup(Circle(), Square())
        self.add(self.group)
        self.dot = Dot()
        self.add(self.dot)
        self.wait(0.1)  # @captured-wait
        self.remove(self.dot)
        self.unused = Square()
        self.play(self.group.animate.shift(RIGHT), run_time=0.5)
        self.late = Circle(radius=0.5)
        self.play(FadeIn(self.late), run_time=0.5)


class Pointed(Scene):
    """``Group(Group(a, b), Group(a))`` through one wait."""

    def construct(self):
        a, b = Square(), Dot(RIGHT)
        second = VGroup(a)
        self.add(VGroup(VGroup(a, b), second))
        self.repoint(second, a, b)
        self.wait(0.2)

    def repoint(self, group, a, b):
        group.remove(a)
        group.add(a)


class Repointed(Pointed):
    """As :class:`Pointed`, its second group pointed at ``b`` instead: the
    nodes, their kinds, child counts and values are the same, and only which
    node the second group holds differs.
    """

    def repoint(self, group, a, b):
        group.remove(a)
        group.add(b)
