"""Scenes whose plays change some bodies and leave the others as they were."""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    ImageMobject,
    ImageMobjectFromCamera,
    MovingCamera,
    MovingCameraScene,
    Scene,
    Square,
    Text,
    VGroup,
    linear,
)


class Changes(MovingCameraScene):
    """Over five plays a dot slides, a square fades out and its outline comes
    back (its points never change), a pair gains a member and the camera
    pans; a label holds still all along.
    """

    def construct(self):
        self.timeline.arm_motion()
        label = Text("still").to_edge(UP)
        dot = Dot(LEFT * 2)
        square = Square(side_length=1).shift(RIGHT * 2)
        pair = VGroup(Dot(DOWN), Dot(DOWN + RIGHT))
        self.add(label, dot, square, pair)
        self.play(dot.animate.shift(RIGHT), rate_func=linear)
        self.play(square.animate.set_opacity(0), rate_func=linear)
        pair.add(Dot(DOWN + LEFT))
        self.wait()
        self.play(self.camera.frame.animate.shift(LEFT), rate_func=linear)
        square.set_stroke(opacity=1)
        self.wait()


class ArmedLate(Scene):
    """Four plays of a dot, the motion rule armed after the second."""

    armed = True

    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), rate_func=linear)
        self.wait()
        if self.armed:
            self.timeline.arm_motion()
        self.play(dot.animate.shift(LEFT), rate_func=linear)
        self.wait()


class Unarmed(ArmedLate):
    """The same plays, never armed."""

    armed = False


class LiveImage(Scene):
    """A display of another camera's picture, which turns opaque between two
    waits while the display holds the same view of it; beside it, an image of
    its own that holds still.
    """

    def construct(self):
        self.timeline.arm_motion()
        source = MovingCamera(pixel_width=16, pixel_height=16, background_opacity=0)
        still = ImageMobject(np.full((4, 4, 4), 255, dtype=np.uint8)).shift(LEFT * 3)
        self.add(ImageMobjectFromCamera(source), still)
        self.wait(0.25)
        source.pixel_array[:, :, 3] = 255
        self.wait(0.25)
