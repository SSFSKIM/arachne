"""Scenes the render-worker tests render: each play boundary is a place a
worker's range can start.
"""

from __future__ import annotations

import functools

import numpy as np

from arachne import (
    BLUE,
    LEFT,
    PI,
    RED,
    RIGHT,
    UP,
    Camera,
    Circle,
    Dot,
    FadeIn,
    ImageMobject,
    MovingCameraScene,
    Rotate,
    Scene,
    Square,
    ZoomedScene,
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


class SkippedCentre(Scene):
    """A frozen frame, the frame centre moved, a frozen frame in a skipped
    section, then a play. The skipped frame's static picture is drawn as
    the scene runs, before the frames waiting for the workers; every path
    still maps around the centre of the first frame.
    """

    def construct(self):
        circle = Circle(radius=0.6)
        self.add(circle)
        self.wait(0.4)
        self.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.next_section(skip_animations=True)
        self.wait(0.4)
        self.next_section(skip_animations=False)
        self.then(circle)

    def then(self, circle):
        self.play(circle.animate.shift(LEFT), run_time=0.4)


class SkippedCentreOnePlay(SkippedCentre):
    """:class:`SkippedCentre` with nothing after the skipped section: one
    play to draw.
    """

    def then(self, circle):
        pass


class SkippedCentreLiveLater(SkippedCentre):
    """:class:`SkippedCentre` whose last play a mobject the record can't
    draw joins midway: from it on, the scene is drawn as it runs.
    """

    def then(self, circle):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)


class SkippedCentreCameraCode(SkippedCentre):
    """:class:`SkippedCentre` whose camera's depth order is the scene's
    code, set on the camera, which does what the engine's does.
    """

    def construct(self):
        self.camera.get_mobjects_to_display = functools.partial(
            Camera.get_mobjects_to_display, self.camera
        )
        super().construct()


class SkippedCentreCameraCodeUndone(SkippedCentreCameraCode):
    """:class:`SkippedCentreCameraCode` whose code leaves the camera as the
    engine made it once the scene has run.
    """

    def construct(self):
        super().construct()
        del self.camera.get_mobjects_to_display


class SkippedCentreZoomed(ZoomedScene, SkippedCentre):
    """:class:`SkippedCentre` through a zoomed scene's camera, whose
    display is never shown.
    """


def drawn_now(scene):
    """The scene's frame, drawn now over the renderer's static picture."""
    scene.renderer.update_frame(scene)
    return np.array(scene.camera.get_image())


def show(scene, pixels):
    """``pixels`` shown full size in place of what the scene shows, held."""
    image = ImageMobject(pixels)
    image.height = scene.camera.frame_height
    image.move_to(scene.camera.frame_center)
    scene.clear()
    scene.add(image)
    scene.wait(0.4)


class SkippedCentreSnapshot(SkippedCentre):
    """:class:`SkippedCentre` whose script draws a frame after the skipped
    section, over its static picture, and shows it.
    """

    def then(self, circle):
        show(self, drawn_now(self))


class FrameRead(Scene):
    """A frozen frame, the frame centre moved, then the script reads a
    frame, drawn now, and shows it before a last play: read while the
    frozen frame waits to be drawn.
    """

    def construct(self):
        self.first()
        circle = Circle(radius=0.6)
        self.add(circle)
        self.wait(0.4)
        self.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.middle()
        self.add(circle)
        self.play(circle.animate.shift(LEFT), run_time=0.4)

    def first(self):
        pass

    def middle(self):
        show(self, self.read())

    def read(self):
        return drawn_now(self)


class FrameGot(FrameRead):
    """:class:`FrameRead` reading the renderer's frame as it is."""

    def read(self):
        return self.renderer.get_frame()


class CameraImageGot(FrameRead):
    """:class:`FrameRead` reading the camera's image as it is."""

    def read(self):
        return np.array(self.camera.get_image())


class FrameNotRead(FrameRead):
    """:class:`FrameRead` with a wait where it reads a frame."""

    def middle(self):
        self.wait(0.4)


class FrameReadFirst(FrameNotRead):
    """:class:`FrameNotRead` whose script reads a frame before its first
    play.
    """

    def first(self):
        drawn_now(self)
        self.renderer.get_frame()
