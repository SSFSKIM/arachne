"""Scenes the assertion bridge's tests execute.

A comment ``# @name`` tags a line the tests look up by name with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Camera,
    Dot,
    Scene,
    Text,
    ValueTracker,
    there_and_back,
)
from arachne.lint.bridge import declare, record, watch

#: Set by a test once its scene has recorded: a probe called after that
#: fails the test.
RECORDED = [False]


def _plot():
    """Axes over [0, 1] by [0, 1], 4 units a side, centred: the plotted
    region is x and y in [-2, 2].
    """
    return Axes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], x_length=4, y_length=4)


def planted(function):
    """Mark ``function`` as one the tests' patched prover admits unproven."""
    function.planted = True
    return function


class PointOff(Scene):
    """A point computed past the axes' x range, asserted on the plot at 1 s."""

    def construct(self):
        ax = _plot()
        self.add(ax)
        self.wait(1)
        off = np.array([ax.c2p(1.5, 0.5)])
        self.found = record("axes.outside", off, reference=ax)  # @point-off
        self.wait(0.5)


class PointOn(Scene):
    """:class:`PointOff`'s twin: the point is inside the range."""

    def construct(self):
        ax = _plot()
        self.add(ax)
        self.wait(1)
        self.found = record("axes.outside", np.array([ax.c2p(0.5, 0.5)]), reference=ax)
        self.wait(0.5)


class Unadded(Scene):
    """A dot and a raw point asserted on axes, none of the three ever added
    to the scene.
    """

    def construct(self):
        ax = _plot()
        self.wait(0.5)
        record("axes.outside", Dot(ax.c2p(0.5, 1.4)), reference=ax)  # @unadded-dot
        record("axes.outside", [ax.c2p(-0.2, 0.5)], reference=ax)  # @unadded-point
        self.wait(0.5)


class Margined(Scene):
    """A point just inside the plot, but within the margin of its edge."""

    def construct(self):
        ax = _plot()
        self.add(ax)
        self.wait(0.25)
        near = [ax.c2p(0.98, 0.5)]
        record("axes.outside", near, reference=ax, margin=0.2)  # @margined
        record("axes.outside", near, reference=ax)
        self.wait(0.25)


class PenOff(Scene):
    """A watched pen driven past the axes' x range during the play at 1-2 s,
    and left there.
    """

    def construct(self):
        ax = _plot()
        pen = Dot(ax.c2p(0.5, 0.5))
        self.add(ax, pen)
        watch("axes.outside", pen, reference=ax)  # @pen-watch
        self.wait(1)
        self.play(pen.animate.move_to(ax.c2p(2.0, 0.5)))  # @pen-play
        self.wait(0.5)


class PenOn(Scene):
    """:class:`PenOff`'s twin: the pen stays inside."""

    def construct(self):
        ax = _plot()
        pen = Dot(ax.c2p(0.5, 0.5))
        self.add(ax, pen)
        watch("axes.outside", pen, reference=ax)
        self.wait(1)
        self.play(pen.animate.move_to(ax.c2p(0.9, 0.5)))
        self.wait(0.5)


class PenBack(Scene):
    """A watched pen that leaves the plot and comes back: two excursions."""

    def construct(self):
        ax = _plot()
        pen = Dot(ax.c2p(0.5, 0.5))
        self.add(ax, pen)
        watch("axes.outside", pen, reference=ax)
        self.play(pen.animate.move_to(ax.c2p(1.5, 0.5)), run_time=0.5)  # @back-out
        self.play(pen.animate.move_to(ax.c2p(0.5, 0.5)), run_time=0.5)
        self.wait(0.5)
        self.play(pen.animate.move_to(ax.c2p(0.5, -0.5)), run_time=0.5)  # @back-down
        self.wait(0.25)


def _blip(alpha):
    """All the way for alpha in (0.3, 0.45), nothing elsewhere: at 15 fps
    over one second, the frames at 5/15 and 6/15 only.
    """
    return 1.0 if 0.3 < alpha < 0.45 else 0.0


class Blip(Scene):
    """A watched probe off the plot for two frames at 15 fps, both between
    two lint instants at 4 fps.
    """

    def construct(self):
        ax = _plot()
        tracker = ValueTracker(0.5)
        self.add(ax)
        watch("axes.outside", lambda: ax.c2p(tracker.get_value(), 0.5), reference=ax)
        self.play(tracker.animate(rate_func=_blip).set_value(2.0), run_time=1)  # @blip
        self.wait(0.25)


class ProbeOff(Scene):
    """A watched probe, a function of a tracker, driven off the plot; the
    probe fails if it is called once the scene has recorded.
    """

    def construct(self):
        ax = _plot()
        tracker = ValueTracker(0.5)
        self.add(ax)

        def tip():
            if RECORDED[0]:
                raise AssertionError("the probe was called after the scene recorded")
            return ax.c2p(tracker.get_value(), 0.5)

        watch("axes.outside", tip, reference=ax)  # @probe-watch
        self.play(tracker.animate.set_value(2.0))  # @probe-play
        self.wait(0.5)


class DriftChecked(Scene):
    """The probe demotes its first play (a planted history-dependent
    updater), so the session records it again: a recorded point off the
    plot, and a watched dot pushed off the top in the second play.
    """

    def construct(self):
        ax = _plot()
        self.add(ax)
        record("axes.outside", [ax.c2p(1.5, 0.5)], reference=ax)  # @drift-record
        tracker = ValueTracker(0)
        drifter = Dot(LEFT * 2).add_updater(
            planted(lambda m: m.set_x(max(m.get_x(), tracker.get_value())))
        )
        self.add(drifter)
        watch("axes.outside", drifter, reference=ax)  # @drift-watch
        self.play(tracker.animate.set_value(1.5), rate_func=there_and_back)
        drifter.clear_updaters()
        self.play(drifter.animate.shift(UP * 3), run_time=0.5)  # @drift-up


class Declared(Scene):
    """Terms declared before the first play and changed after it."""

    def construct(self):
        self.a = a = Text("a")
        self.b = b = Text("b").next_to(a, RIGHT)
        self.add(a, b)
        self.found = declare(text_gap=0.3, exempt=[a, (a, b)])  # @declare
        self.wait(0.5)
        declare(pad=0.05)  # @declare-pad
        self.wait(0.5)


class BadTerm(Scene):
    """A declaration of a term the bridge does not know."""

    def construct(self):
        declare(text_spacing=0.3)
        self.wait(0.25)


class UnknownCode(Scene):
    """A record of a code the bridge does not check."""

    def construct(self):
        record("overlap.text", [(0, 0, 0)], reference=_plot())
        self.wait(0.25)


class NoReference(Scene):
    """An ``axes.outside`` record with no axes to be outside of."""

    def construct(self):
        record("axes.outside", [(0, 0, 0)])
        self.wait(0.25)


def _through(method, name):
    """A function that calls ``method`` and returns what it returns, named
    ``name``: a camera method replaced by one that changes nothing drawn.
    """

    def replaced(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    replaced.__name__ = name
    return replaced


def _plays(scene):
    dot = Dot(LEFT)
    scene.add(dot)
    scene.play(dot.animate.shift(RIGHT * 2))
    scene.wait(0.25)


class NothingReplaced(Scene):
    """The plays the replaced-method scenes play, on the engine's camera."""

    def construct(self):
        _plays(self)


class InstanceReplaced(Scene):
    """The camera instance holds a function of the scene's in place of the
    method that lists what it draws.
    """

    def construct(self):
        camera = self.renderer.camera
        camera.get_mobjects_to_display = _through(
            Camera.get_mobjects_to_display, "get_mobjects_to_display"
        ).__get__(camera)
        _plays(self)


class ClassReplaced(Scene):
    """The engine's camera class has its method that lists what it draws
    replaced for the scene's run, and put back after.
    """

    def construct(self):
        original = vars(Camera)["get_mobjects_to_display"]
        Camera.get_mobjects_to_display = _through(original, "get_mobjects_to_display")
        try:
            _plays(self)
        finally:
            Camera.get_mobjects_to_display = original


class MemberReplaced(Scene):
    """A camera member no frame is drawn through, replaced on the instance."""

    def construct(self):
        camera = self.renderer.camera
        camera.get_coords_of_all_pixels = lambda: None
        _plays(self)


class HookedIn(Scene):
    """A watched function and a recorded point, both read from a count the
    scene's own frame hook keeps: the render's frames keep the point on the
    plot, while a record on a finer grid than the render's runs the hook
    more often and would take it off.
    """

    #: How far right the point moves for each call of the frame hook.
    STEP = 0.1

    def construct(self):
        ax = _plot()
        dot = Dot(DOWN * 3)
        self.calls = 0
        self.add(ax, dot)
        watch("axes.outside", self._tip, reference=ax)
        self.play(dot.animate.shift(RIGHT), run_time=1)  # @hooked-play
        record("axes.outside", [self._tip()], reference=ax)  # @hooked-record

    def _tip(self):
        return (-2 + self.STEP * self.calls, 0)

    def update_to_time(self, t):
        super().update_to_time(t)
        self.calls += 1


class HookedOut(HookedIn):
    """:class:`HookedIn` with a step that takes the point off the plot on the
    render's frames too.
    """

    STEP = 0.3


class LateUnchanged(Scene):
    """A dot's play whose frame hook counts its calls: :class:`LateCamera`
    without the replacement.
    """

    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)

    def update_to_time(self, t):
        super().update_to_time(t)
        self.late(t)

    def late(self, t):
        pass


class LateCamera(LateUnchanged):
    """The scene's frame hook replaces a camera method on the instance part
    way through the last play; the replacement fails once the scene has
    recorded.
    """

    def late(self, t):
        if t > 0.2 and not getattr(self, "_replaced", False):
            self._replaced = True
            camera = self.renderer.camera
            original = camera.transform_points_pre_display

            def transform_points_pre_display(*args, **kwargs):
                if RECORDED[0]:
                    raise AssertionError(
                        "the scene's camera code ran after it recorded"
                    )
                return original(*args, **kwargs)

            camera.transform_points_pre_display = transform_points_pre_display


class ClockWatch(Scene):
    """A watched function of the scene clock, off the plot for 0.3 < t <
    0.45 only, all of it inside a frozen wait.
    """

    def construct(self):
        ax = _plot()
        self.add(ax)
        watch(
            "axes.outside",
            lambda: ax.c2p(1.5 if 0.3 < self.time < 0.45 else 0.5, 0.5),
            reference=ax,
        )
        self.wait(1)  # @clock-wait


class TickWatch(Scene):
    """A pen a dt updater puts off the plot for 0.03 < t < 0.05 only: a tick
    of the 60 Hz clock that the frame at 1/24 shows at 24 fps.
    """

    def construct(self):
        ax = _plot()
        pen = Dot()
        pen.add_updater(
            lambda m, dt: m.set_x(3 if 0.03 < float(self.time) < 0.05 else 0)
        )
        self.add(ax, pen)
        watch("axes.outside", pen, reference=ax)
        self.wait(0.25)  # @tick-wait


class ConditionalWatches(Scene):
    """Bridge calls that depend on how often the scene's frame hook ran: an
    execution on a finer grid than the render's makes other calls.
    """

    def construct(self):
        ax = _plot()
        dot = Dot()
        self.count = 0
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        if self.count < 20:
            watch("axes.outside", lambda: ax.c2p(1.5, 0.5), reference=ax)
        watch("axes.outside", lambda: ax.c2p(0.5, 0.5), reference=ax)
        self.play(dot.animate.shift(RIGHT), run_time=1)  # @conditional-play
        off = 1.5 if self.count < 40 else 0.5
        record("axes.outside", [ax.c2p(off, 0.5)], reference=ax)  # @conditional-record

    def update_to_time(self, t):
        super().update_to_time(t)
        self.count += 1


class ReferenceRanges(Scene):
    """Points checked against two axes of other ranges, with margins."""

    def construct(self):
        narrow = Axes(
            x_range=[-2, 4, 1], y_range=[10, 12, 1], x_length=6, y_length=2
        ).shift(LEFT)
        broad = Axes(
            x_range=[-100, 100, 100], y_range=[-100, 100, 100], x_length=20, y_length=10
        )
        point = narrow.c2p(4.2, 11)
        self.out = record("axes.outside", point, reference=narrow)
        self.inside_broad = record("axes.outside", point, reference=broad)
        pen = Dot(narrow.c2p(3.9, 11))
        self.near = record(
            "axes.outside", pen.get_center(), reference=narrow, margin=0.05
        )
        self.margin = record(
            "axes.outside", pen.get_center(), reference=narrow, margin=0.2
        )
        self.add(pen)
        watch("axes.outside", pen, reference=narrow, margin=0.2)
        watch("axes.outside", pen, reference=broad, margin=0.2)
        self.wait(0.2)


class _Counted(Scene):
    """A scene whose frame hook counts its calls: a finer grid's execution
    counts more of them by the end of a play than the render's does.
    """

    def update_to_time(self, t):
        super().update_to_time(t)
        self.count += 1


class DivergentDuration(_Counted):
    """A second play whose length depends on the count, so the render's
    execution and a finer grid's end at different times; at 96 pixels wide
    the render's is the longer one, at any other width the shorter.
    """

    def construct(self):
        ax = _plot()
        self.count = 0
        dot = Dot()
        self.add(dot)
        watch("axes.outside", lambda: ax.c2p(1.5, 0.5), reference=ax)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        longer_on_render = self.renderer.camera.pixel_width == 96
        run_time = 0.5 if (self.count < 20) == longer_on_render else 0.25
        self.play(dot.animate.shift(RIGHT), run_time=run_time)
        record("axes.outside", [ax.c2p(1.5, 0.5)], reference=ax)


class ForwardOnlyBridge(_Counted):
    """A watch and a record made only where the count stays low: by the
    render's execution, never by a finer grid's.
    """

    def construct(self):
        ax = _plot()
        self.count = 0
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        if self.count < 20:
            watch("axes.outside", lambda: ax.c2p(1.5, 0.5), reference=ax)
            record("axes.outside", [ax.c2p(1.5, 0.5)], reference=ax)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)


class StoppedWatch(Scene):
    """A watched pen pushed off the plot at the tick a wait's stop condition
    ends the scene on, between two frames at 24 fps.
    """

    def construct(self):
        ax = _plot()
        pen = Dot()
        pen.add_updater(lambda m, dt: m.set_x(3 if float(self.time) >= 0.115 else 0))
        self.add(ax, pen)
        watch("axes.outside", pen, reference=ax)
        self.wait(1, stop_condition=lambda: float(self.time) >= 0.115)
