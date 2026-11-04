"""Scenes the probe tests execute: state-function plays whose callbacks
keep history the proofs never saw, admitted only because the tests patch
the prover to take a :func:`planted` function as proven, and plays too short
to probe.
"""

from __future__ import annotations

import random

from arachne import (
    BLACK,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Dot,
    Group,
    PMobject,
    Scene,
    Square,
    Succession,
    ValueTracker,
    ZoomedScene,
    always_redraw,
    linear,
    there_and_back,
)


def planted(function):
    """Mark ``function`` as one the tests' patched prover admits unproven."""
    function.planted = True
    return function


def planted_later(function):
    """Mark ``function`` as one the tests' patched prover refuses the first
    time it sees it, and admits unproven in every execution after.
    """
    function.planted = "later"
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


class TwoPassDrift(Scene):
    """Two plays of running-max setters, the second admitted only from the
    second execution on, so the probe demotes one play a pass; and a dot
    placed by an unseeded random draw, so every execution begins its first
    play elsewhere than the one before.
    """

    def construct(self):
        jitter = random.Random().random()
        tracker = ValueTracker(0)
        first = Dot(LEFT * 2 + UP * jitter).add_updater(
            planted(lambda m: m.set_x(max(m.get_x(), tracker.get_value())))
        )
        self.add(first)
        self.play(tracker.animate.set_value(2), rate_func=there_and_back)
        first.clear_updaters()
        second = Dot(LEFT * 2 + DOWN).add_updater(
            planted_later(lambda m: m.set_x(max(m.get_x(), tracker.get_value())))
        )
        self.add(second)
        self.play(tracker.animate.set_value(2), rate_func=there_and_back)  # @second


class PointPaintHook(Scene):
    """A frame hook fading a point cloud to black by 1e-12 at every frame, in a
    state play: its colours are paint, kept in its own rgbas array.
    """

    def construct(self):
        self.cloud = PMobject().add_points([[0, 0, 0]])
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(self.cloud, mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "cloud"):
            self.cloud.fade_to(BLACK, 1e-12)


class SharedChildHook(Scene):
    """Two groups sharing a dot, and a frame hook pointing the second group
    at the other dot after the first frame: the tree's shape and every
    node's values stay, its edges change.
    """

    def construct(self):
        self.left, self.right = Dot(LEFT), Dot(RIGHT)
        self.second = Group(self.left)
        self.add(Group(Group(self.left, self.right), self.second))
        tracker = ValueTracker(0)
        mover = Dot(DOWN).add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "second"):
            self.second.remove(self.left, self.right)
            self.second.add(self.left if t == 0 else self.right)


class CameraPaintHook(Scene):
    """A frame hook turning the background white after the first frame."""

    def construct(self):
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        self.camera.background_color = BLACK if t == 0 else WHITE


class ZoomedCameraPaintHook(ZoomedScene):
    """A frame hook fading the zoomed camera's background after the first
    frame: the paint of a camera that draws into a display.
    """

    def construct(self):
        self.activate_zooming(animate=False)
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "zoomed_camera"):
            self.zoomed_camera.background_opacity = 1.0 if t == 0 else 0.5
