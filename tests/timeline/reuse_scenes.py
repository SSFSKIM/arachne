"""Scenes the draw reuse tests render and open: frames whose recorded state
repeats (a wait whose updater changes nothing, a frozen frame), one whose
mobject changes only its line joint, a camera that moves over mobjects
that stay still, a point drawn at a configured thickness, a camera whose
drawing the scene's code replaced, cameras whose display functions it
replaced (with its own, another camera's, a wrapper of the engine's, or an
object that forwards attribute reads to the engine's), a display showing another camera's live picture, and a depth sort that
reads a point outside the scene.
"""

from __future__ import annotations

import functools

import numpy as np

from arachne import (
    BLUE,
    OUT,
    RED,
    RIGHT,
    Circle,
    ImageMobjectFromCamera,
    LineJointType,
    Mobject,
    MovingCameraScene,
    PMobject,
    Point,
    Scene,
    Square,
    ThreeDScene,
    Triangle,
    VectorizedPoint,
    VMobject,
    Wait,
)
from arachne.camera.camera import Camera
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


class BorrowedDisplay(Scene):
    """A camera whose display function for vectorized mobjects is another
    camera's, whose line width changes between two plays of one wait.
    """

    def construct(self):
        other = Camera()
        other.cairo_line_width_multiple = 0.01
        self.camera.display_funcs[VMobject] = other.display_multiple_vectorized_mobjects
        square = Square(stroke_width=8)
        square.add_updater(lambda m, dt: None)
        self.add(square)
        wait = Wait(run_time=0.4)
        self.play(wait)
        other.cairo_line_width_multiple = 0.1
        self.play(wait)


class WrappedNothing(Scene):
    """A camera whose do-nothing display function the scene wrapped, keeping
    its name: each call marks the frame with how many calls it has made.
    """

    def construct(self):
        original = self.camera.display_funcs[Mobject]
        calls = [0]

        @functools.wraps(original)
        def display(mobjects, pixel_array):
            original(mobjects, pixel_array)
            calls[0] += 1
            pixel_array[0, 0, :3] = calls[0]

        self.camera.display_funcs[Mobject] = display
        mob = Mobject()
        mob.points = np.array([[0.0, 0.0, 0.0]])
        mob.add_updater(lambda m, dt: None)
        self.add(mob)
        self.wait(0.4)


class OutsideDepth(ThreeDScene):
    """Two overlapping squares sorted by depth, one by a point outside the
    scene (its ``z_index_group``), moved between two plays of one wait.
    """

    def construct(self):
        self.camera.should_apply_shading = False
        reference = VectorizedPoint(np.array([0.0, 0.0, -1.0]))
        red = Square(color=RED, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        blue = Square(color=BLUE, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        red.z_index_group = reference
        red.add_updater(lambda m, dt: None)
        self.add(red, blue)
        wait = Wait(run_time=0.4)
        self.play(wait)
        reference.shift(OUT * 2)
        self.play(wait)


class _ForwardedDisplay(Scene):
    """A camera whose display function for ``kind`` the scene replaced with an
    object that forwards every attribute read to the engine's function (its
    ``__code__``, ``__self__``, ``__func__``) but runs its own code when
    called: each call marks the frame with how many calls it has made.
    """

    kind: type = Mobject

    def construct(self):
        original = self.camera.display_funcs[self.kind]

        class Forwarding:
            calls = 0

            def __getattr__(self, name):
                return getattr(original, name)

            def __call__(self, mobjects, pixel_array):
                original(mobjects, pixel_array)
                self.calls += 1
                pixel_array[0, 0, :3] = self.calls

        self.camera.display_funcs[self.kind] = Forwarding()
        if self.kind is Mobject:
            mob = Mobject()
            mob.points = np.array([[0.0, 0.0, 0.0]])
        else:
            mob = Square(fill_opacity=0.5)
        mob.add_updater(lambda m, dt: None)
        self.add(mob)
        self.wait(0.4)


class ForwardedDisplay(_ForwardedDisplay):
    """:class:`_ForwardedDisplay` for vectorized mobjects."""

    kind = VMobject


class ForwardedNothing(_ForwardedDisplay):
    """:class:`_ForwardedDisplay` for the do-nothing function."""

    kind = Mobject


class LightMovedThenReplaced(ThreeDScene):
    """A shaded square lit by the camera's own light, which an updater moves
    through a wait; then the scene gives the camera another light, so that
    the camera the record is drawn with holds not the light that lit the
    wait.
    """

    def construct(self):
        square = Square(side_length=3, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        square.set_color(BLUE)
        light = self.camera.light_source
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            light.move_to(np.array([-20.0 * elapsed[0], 0.0, 10.0]))

        square.add_updater(drive)
        self.add(square)
        self.wait(0.4)
        self.camera.light_source = Point(np.array([0.0, 20.0, 10.0]))
