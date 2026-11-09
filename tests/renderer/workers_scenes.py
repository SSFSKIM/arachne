"""Scenes the render-worker tests render: each play boundary is a place a
worker's range can start.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    LEFT,
    PI,
    RED,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    MovingCameraScene,
    Rotate,
    Scene,
    Square,
)
from tests.timeline.soundness_scenes import CountingFill, play_fill_between


class Steps(Scene):
    """Plays of several lengths, a frozen frame among them and a hold to
    the end.
    """

    def construct(self):
        dot = Dot(LEFT * 2)
        square = Square(side_length=1).shift(RIGHT)
        self.add(square)
        self.play(dot.animate.shift(RIGHT), run_time=0.4)
        self.play(Rotate(square, PI / 4), run_time=0.6)
        self.wait(0.4)
        self.play(FadeIn(Circle(radius=0.5, color=BLUE)), run_time=0.8)
        self.play(dot.animate.shift(UP), run_time=0.4)
        self.wait(0.6)


class FrozenThenPlays(Scene):
    """A frozen frame first, then two plays of a second each: at two
    workers the cut falls after the first play, at three after the frozen
    frame too.
    """

    def construct(self):
        dot = Dot(LEFT * 2, color=RED)
        self.add(Square(side_length=1.5), dot)
        self.wait(0.6)
        self.play(dot.animate.shift(RIGHT * 2), run_time=1)
        self.play(dot.animate.shift(UP), run_time=1)


class Zoomed(MovingCameraScene):
    """A MovingCamera zooming in, holding, then panning: each frame maps
    around the frame as it is then.
    """

    def construct(self):
        self.add(Circle(radius=0.6), Square(side_length=2).shift(RIGHT))
        self.play(self.camera.frame.animate.scale(0.5).shift(RIGHT * 0.5), run_time=0.6)
        self.wait(0.4)
        self.play(self.camera.frame.animate.shift(LEFT), run_time=0.6)


class CentreMoved(Scene):
    """A flat camera whose frame centre the script sets between two waits
    and a play: its paths keep mapping around the centre of its first
    frame.
    """

    def construct(self):
        circle = Circle(radius=0.6).shift(RIGHT)
        self.add(circle)
        self.wait(0.4)
        self.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.wait(0.4)
        self.play(circle.animate.shift(LEFT), run_time=0.4)


class LiveLater(Scene):
    """Two plays drawn from the record, then a play a mobject the record
    can't draw joins midway: from it on, the scene is drawn as it runs.
    """

    def construct(self):
        CountingFill.colour = RED
        dot = Dot(LEFT * 2)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.4)
        self.wait(0.4)
        play_fill_between(self, 0.3, 0.6)
        self.wait(0.2)


class OnePlay(Scene):
    """A single play: one range, nothing to share."""

    def construct(self):
        dot = Dot()
        self.play(dot.animate.shift(RIGHT), run_time=0.6)


class Pairs(Scene):
    """Two short plays."""

    def construct(self):
        dot = Dot(LEFT)
        self.play(dot.animate.shift(RIGHT), run_time=0.4)
        self.play(dot.animate.shift(UP), run_time=0.4)
