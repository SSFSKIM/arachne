"""Scenes the restore tests execute: state-function plays whose callbacks
read closure cells, globals, trackers outside the scene and the scene's
attributes, with a frame hook, a ``Succession`` and a ``TracedPath`` beside
them.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    DecimalNumber,
    Dot,
    FadeIn,
    Line,
    Mobject,
    Scene,
    Square,
    Succession,
    TracedPath,
    UpdateFromAlphaFunc,
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


class HookAddsUpdater(Scene):
    """A frame hook attaching an updater, one admission never saw, that
    counts its calls.
    """

    def construct(self):
        self.calls = 0
        self.attached = False
        self.dot = Dot()
        tracker = ValueTracker(0)
        mark = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(self.dot, mark)
        self.play(tracker.animate.set_value(2), run_time=1)

    def count(self, mob, dt):
        self.calls += 1

    def update_to_time(self, t):
        super().update_to_time(t)
        if not self.attached:
            self.dot.add_updater(self.count)
            self.attached = True


class HookMutates(Scene):
    """A frame hook changing in place the list an updater reads, and the
    scene putting it back after the play.
    """

    def construct(self):
        self.offsets = [0.0]
        tracker = ValueTracker(0)
        dot = Dot().add_updater(
            lambda m: m.set_x(tracker.get_value() + self.offsets[0])
        )
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)
        self.offsets[0] = 0.0

    def update_to_time(self, t):
        super().update_to_time(t)
        self.offsets[0] = float(t)


class FailingUpdater(Scene):
    """An updater that sets its dot's x, then divides by zero at 1/7 s."""

    def construct(self):
        offset = 0.0
        tracker = ValueTracker(0.5)
        self.dot = Dot().add_updater(
            lambda m: m.set_x(tracker.get_value() + offset).set_y(
                1 / (float(self.time) - 1 / 7)
            )
        )
        self.add(self.dot)
        self.play(ValueTracker(0).animate.set_value(2), run_time=1)
        tracker.set_value(3)
        offset = 1.5


class FramePeriodic(Scene):
    """A dot that stands at the same place at every 15 fps frame and moves
    between them, beside a square no frame redraws.
    """

    def construct(self):
        self.add(Square(side_length=0.5).shift(LEFT * 3))
        dot = Dot(color=BLUE).add_updater(lambda m: m.set_y(2 * (self.time * 15 % 1)))
        self.add(dot)
        self.play(FadeIn(Circle(radius=0.3).shift(RIGHT * 3)), run_time=1)


class DrivenSquare(Scene):
    """A square added before the point-less driver whose updater places it
    from a tracker the play animates: CE's moving list begins at the driver,
    so the square is static while the driver moves it.
    """

    def construct(self):
        tracker = ValueTracker(0)
        square = Square(side_length=1, color=BLUE, fill_opacity=1).shift(LEFT * 3)
        driver = Mobject().add_updater(lambda m: square.set_x(tracker.get_value() - 3))
        self.add(square, driver)
        self.play(tracker.animate.set_value(6), run_time=1, rate_func=linear)


class TickedOffGrid(Scene):
    """A held wait ending between two 15 fps frames, then a ticked play."""

    def construct(self):
        self.add(Square(side_length=0.5))
        self.wait(0.97)
        drift = Dot(DOWN * 2).add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(drift)
        self.wait(0.5)


class HookRetargets(Scene):
    """A frame hook pointing the play's ``UpdateFromAlphaFunc`` at a dot
    outside the scene, which the scene moves after the play.
    """

    def construct(self):
        self.other = Dot(UP)
        dot = Dot()
        self.add(dot)
        self.play(
            UpdateFromAlphaFunc(
                dot, lambda m, alpha: m.set_x(alpha), run_time=1, rate_func=linear
            )
        )
        self.other.set_x(9)

    def update_to_time(self, t):
        super().update_to_time(t)
        self.animations[0].mobject = self.other


class FramelessBeforeTicked(Scene):
    """A held wait ending between two 15 fps frames, a trusted play too short
    for any frame, then a ticked play.
    """

    def construct(self):
        self.add(Square(side_length=0.5))
        self.wait(0.97)
        dot = Dot(RIGHT * 2)
        self.add(dot)
        self.play(dot.animate.shift(UP), run_time=0.001)
        self.add(Dot(DOWN * 2).add_updater(lambda m, dt: m.shift(RIGHT * dt)))
        self.wait(0.2)


class TrustedBeforeTicked(Scene):
    """A linear trusted move ending between two 60 Hz ticks, then a ticked
    play.
    """

    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT * 2), run_time=0.97, rate_func=linear)
        self.add(Dot(DOWN * 2).add_updater(lambda m, dt: m.shift(RIGHT * dt)))
        self.wait(0.2)


class TickedSuccession(Scene):
    """A ticked play (a dt updater) whose Succession event, at 0.65 s, falls
    on a tick between two 15 fps frames.
    """

    def construct(self):
        square = Square(side_length=0.5)
        drift = Dot(DOWN * 2).add_updater(lambda m, dt: m.shift(RIGHT * dt * 0.1))
        self.add(square, drift)
        self.play(
            Succession(
                square.animate(run_time=0.65, rate_func=linear).shift(RIGHT),
                square.animate(run_time=0.65, rate_func=linear).shift(UP),
            )
        )
