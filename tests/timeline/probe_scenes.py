"""Scenes the probe tests execute: state-function plays whose callbacks
keep history the proofs never saw, admitted only because the tests patch
the prover to take a :func:`planted` function as proven, and plays too short
to probe.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    Scene,
    Square,
    Succession,
    ValueTracker,
    always_redraw,
    linear,
    there_and_back,
)


def planted(function):
    """Mark ``function`` as one the tests' patched prover admits unproven."""
    function.planted = True
    return function


class PlantedDrift(Scene):
    """An updater keeping its dot at the furthest the tracker has reached, a
    setter whose result depends on every value it saw, during the first play
    only. Evaluated again at an instant, from the state there, it changes
    nothing: only a restore from an earlier state shows it.
    """

    def construct(self):
        tracker = ValueTracker(0)
        drifter = Dot(LEFT * 2).add_updater(
            planted(lambda m: m.set_x(max(m.get_x(), tracker.get_value())))
        )
        self.add(drifter)
        self.play(tracker.animate.set_value(2), rate_func=there_and_back)  # @drift
        drifter.clear_updaters()
        self.play(drifter.animate.shift(UP), run_time=0.5)


class PlantedGeometryDrift(Scene):
    """An updater that moves its dot on from wherever the last evaluation
    left it, by far less than the probe's tolerance.
    """

    def construct(self):
        drifter = Dot(LEFT * 2).add_updater(planted(lambda m: m.shift(RIGHT * 1e-12)))
        mover = Dot(DOWN)
        self.add(drifter, mover)
        self.play(mover.animate.shift(RIGHT * 2), run_time=1, rate_func=linear)


class PlantedPaintDrift(Scene):
    """An updater that fades its dot on from its last opacity, by as little
    as :class:`PlantedGeometryDrift` moves its dot.
    """

    def construct(self):
        drifter = Dot(LEFT * 2).add_updater(
            planted(lambda m: m.set_fill(opacity=m.get_fill_opacity() - 1e-12))
        )
        mover = Dot(DOWN)
        self.add(drifter, mover)
        self.play(mover.animate.shift(RIGHT * 2), run_time=1, rate_func=linear)


class ShortPlays(Scene):
    """Three state-function plays of one ``always_redraw`` readout, at 15
    fps: the first two hold one frame each besides their start, the third
    fifteen; a frozen wait before the third, and a trusted play after it.
    """

    def construct(self):
        tracker = ValueTracker(0)
        mark = always_redraw(lambda: Dot([tracker.get_value(), 0, 0]))
        self.add(mark)
        self.play(tracker.animate.set_value(0.5), run_time=0.1)
        self.play(tracker.animate.set_value(1), run_time=0.05)
        self.wait(0.2)
        self.play(tracker.animate.set_value(-1), run_time=1, rate_func=linear)
        self.remove(mark)
        self.play(Dot(UP).animate.shift(RIGHT), run_time=0.5)


class EventOnLastFrame(Scene):
    """A ``Succession`` whose second part begins on the play's last frame, at
    15 fps: that frame is an event, never a probed instant.
    """

    def construct(self):
        tracker = ValueTracker(0)
        square = Square(side_length=0.5)
        dot = Dot().add_updater(
            lambda m: m.move_to(square.get_center() + DOWN * tracker.get_value())
        )
        self.add(square, dot)
        self.play(
            Succession(
                tracker.animate(run_time=0.6, rate_func=linear).set_value(1),
                square.animate(run_time=0.05).shift(RIGHT),
            )
        )
