"""Scenes the JSON form tests execute.

A comment ``# @name`` tags a line the tests look up by name.
"""

from __future__ import annotations

from arachne import DEGREES, MovingCameraScene, Scene, Square, ThreeDScene
from tests.lint.shapes import framed_square


class Zoomed(MovingCameraScene):
    """A still square, then the view narrowed to half its width about the
    centre during 0.5-1.0 s: the square stays where it is and fills twice
    the picture.
    """

    def construct(self):
        square = Square(side_length=1).move_to((1, 0.5, 0))
        self.add(square)
        self.wait(0.5)
        frame = self.camera.frame
        self.play(frame.animate.set(width=frame.width / 2), run_time=0.5)
        self.wait(0.5)


class Tilted(ThreeDScene):
    """A square seen by a tilted, turned ThreeDCamera, stroked thin so its
    drawn pixels hug its projected corners.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        square = Square(side_length=2, stroke_width=1).move_to((1, 0.5, 0))
        self.add(square)
        self.wait(0.5)


class MadeInAHelper(Scene):
    """A square made in a helper module."""

    def construct(self):
        square = framed_square()  # @helper-call
        self.add(square)
        self.wait(0.5)
