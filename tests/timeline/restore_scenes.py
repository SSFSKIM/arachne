"""Scenes the restore tests execute: state-function plays whose callbacks
read closure cells, globals, trackers outside the scene and the scene's
attributes, with a frame hook, a ``Succession`` and a ``TracedPath`` beside
them.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DOWN,
    Circle,
    LEFT,
    RIGHT,
    UP,
    DecimalNumber,
    Dot,
    FadeIn,
    Line,
    Scene,
    Square,
    Succession,
    TracedPath,
    ValueTracker,
    always_redraw,
    linear,
)

#: Read by a callback, and rebound by the scene between its plays.
SHIFT = 0.0


class ClosureCell(Scene):
    """An updater reading a local of ``construct``, rebound between plays:
    the cell holds the second value once the scene has run.
    """

    def construct(self):
        tracker = ValueTracker(0)
        offset = 0.0
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + offset))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)
        offset = 1.5
        self.play(tracker.animate.set_value(0), run_time=1)


class ReboundGlobal(Scene):
    """An updater reading a module global the scene rebinds between plays."""

    def construct(self):
        global SHIFT
        SHIFT = 0.0
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)


class DetachedTracker(Scene):
    """A tracker read by ``always_redraw`` and a ``DecimalNumber``'s updater,
    never added to the scene nor animated, and set directly between plays:
    no capture reaches it. The dot below never moves, so a play draws it in
    its static picture.
    """

    def construct(self):
        tracker = ValueTracker(0.25)
        dot = Dot(LEFT)
        line = always_redraw(
            lambda: Line(dot.get_center(), RIGHT * tracker.get_value() + UP)
        )
        number = DecimalNumber(0).to_corner(UP + LEFT)
        number.add_updater(lambda m: m.set_value(tracker.get_value()))
        self.add(Dot(DOWN * 2), dot, line, number)
        self.play(dot.animate.shift(RIGHT), run_time=0.8)
        tracker.set_value(0.5)
        self.play(dot.animate.shift(UP), FadeIn(Circle(radius=0.3)), run_time=0.9)
        tracker.set_value(3)
        self.play(FadeIn(Square(color=BLUE)), run_time=0.8)


class SuspendedTrace(Scene):
    """A ``TracedPath`` that fades as it is drawn, suspended while a state
    play runs and resumed after: its time stands still in the middle play
    and moves on in the last.
    """

    def construct(self):
        dot = Dot(LEFT * 2)
        self.path = TracedPath(dot.get_center, dissipating_time=0.5)
        self.add(dot, self.path)
        self.play(dot.animate.shift(RIGHT * 2), run_time=0.6)
        self.path.suspend_updating()
        tracker = ValueTracker(0)
        mark = always_redraw(lambda: Dot([tracker.get_value(), -1, 0]))
        self.add(mark)
        self.play(tracker.animate.set_value(2), run_time=0.9)
        self.path.resume_updating()
        self.play(dot.animate.shift(UP), run_time=0.6)


class FrameCounter(Scene):
    """A frame hook counting the frames into a ``DecimalNumber`` while a
    state play runs: the count stands still between frames.
    """

    def construct(self):
        self.count = DecimalNumber(0, num_decimal_places=0).to_corner(UP + RIGHT)
        tracker = ValueTracker(0)
        dot = always_redraw(lambda: Dot([tracker.get_value(), 0, 0]))
        self.add(self.count, dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "count"):
            self.count.set_value(self.count.number + 1)


class HookRebinds(Scene):
    """A frame hook rebinding the scene attribute an updater reads."""

    def construct(self):
        self.offset = 0.0
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + self.offset))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)

    def update_to_time(self, t):
        super().update_to_time(t)
        self.offset = float(t) / 10


class SuccessionReadout(Scene):
    """A ``Succession`` whose event falls between frames, and an updater
    reading what its parts move.
    """

    def construct(self):
        tracker = ValueTracker(0)
        square = Square(side_length=0.5)
        dot = Dot(color=BLUE).add_updater(
            lambda m: m.move_to(square.get_center() + DOWN * tracker.get_value())
        )
        self.add(square, dot)
        self.play(
            Succession(
                tracker.animate(run_time=0.65, rate_func=linear).set_value(1),
                square.animate(run_time=0.65).shift(RIGHT * 2),
            )
        )
