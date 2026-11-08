"""Scenes the draw reuse tests render and open: frames whose recorded state
repeats (a wait whose updater changes nothing, a frozen frame), one whose
mobject changes only its line joint, and a camera that moves over mobjects
that stay still.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    RIGHT,
    Circle,
    LineJointType,
    MovingCameraScene,
    Scene,
    Square,
    Triangle,
)


class StillWait(Scene):
    """A square, and a wait whose updater changes nothing: drawn from the
    record (not a frozen frame), every frame showing the same state.
    """

    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.5)
        square.add_updater(lambda m, dt: None)
        self.add(square)
        self.wait(1)


class JointTurns(Scene):
    """A thick triangle whose corners turn round halfway through a wait:
    nothing changes but its ``joint_type``, an enum.
    """

    def construct(self):
        triangle = Triangle(stroke_width=60).scale(1.5)
        elapsed = [0.0]

        def turn(m, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.5:
                m.joint_type = LineJointType.ROUND

        triangle.joint_type = LineJointType.MITER
        triangle.add_updater(turn)
        self.add(triangle)
        self.wait(1)


class PanOverStill(MovingCameraScene):
    """A camera that pans over a circle and a square nothing moves."""

    def construct(self):
        self.add(Circle(radius=0.6), Square().shift(RIGHT * 2))
        self.play(self.camera.frame.animate.shift(RIGHT), run_time=0.5)


class Holds(Scene):
    """A frozen frame after a play: a session's frames inside it show one
    state, and a moving play's do not.
    """

    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.5)
        self.add(square)
        self.play(square.animate.shift(RIGHT), run_time=0.5)
        self.wait(1)
