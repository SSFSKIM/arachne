"""Scenes the draw reuse tests render and open: frames whose recorded state
repeats (a wait whose updater changes nothing, a frozen frame), one whose
mobject changes only its line joint, a camera that moves over mobjects
that stay still, a point drawn at a configured thickness, a camera whose
drawing the scene's code replaced, a camera whose display functions it
replaced, and a display showing another camera's live picture.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    RED,
    RIGHT,
    Circle,
    ImageMobjectFromCamera,
    LineJointType,
    MovingCameraScene,
    PMobject,
    Scene,
    Square,
    Triangle,
    VMobject,
    Wait,
)
from arachne.camera.moving_camera import MovingCamera


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


class StillPoint(Scene):
    """A thick point, and a wait whose updater changes nothing: the camera
    draws its thickness from the configured pixel size.
    """

    def construct(self):
        point = PMobject(stroke_width=9)
        point.add_points(np.array([[0.0, 0.0, 0.0]]))
        point.add_updater(lambda m, dt: None)
        self.add(point)
        self.wait(0.4)


class PatchedCamera(Scene):
    """A wait whose updater changes nothing, drawn by an engine camera whose
    capture the scene's code replaced on the instance: each capture marks
    the frame with how many captures it has made.
    """

    def construct(self):
        camera = self.camera
        original = camera.capture_mobjects
        calls = [0]

        def capture(*args, **kwargs):
            original(*args, **kwargs)
            calls[0] += 1
            camera.pixel_array[0, 0, :3] = calls[0]

        camera.capture_mobjects = capture
        square = Square(fill_opacity=0.5)
        square.add_updater(lambda m, dt: None)
        self.add(square)
        self.wait(0.4)


class ReplacedDisplay(Scene):
    """A wait whose updater changes nothing, drawn by a camera whose display
    function for vectorized mobjects the scene's code replaced: each call
    marks the frame with how many calls it has made.
    """

    def construct(self):
        original = self.camera.display_funcs[VMobject]
        calls = [0]

        def display(mobjects, pixel_array):
            original(mobjects, pixel_array)
            calls[0] += 1
            pixel_array[0, 0, :3] = calls[0]

        self.camera.display_funcs[VMobject] = display
        square = Square(fill_opacity=0.5)
        square.add_updater(lambda m, dt: None)
        self.add(square)
        self.wait(0.4)


class OtherCameraShown(Scene):
    """A display of a camera outside the scene, whose picture turns from red to
    blue between two plays of one wait: nothing the record holds changes.
    """

    def construct(self):
        other = MovingCamera()
        other.background_color = RED
        other.reset()
        display = ImageMobjectFromCamera(other).scale(0.4)
        display.add_updater(lambda m, dt: None)
        self.add(display)
        wait = Wait(run_time=0.4)
        self.play(wait)
        other.background_color = BLUE
        other.reset()
        self.play(wait)
