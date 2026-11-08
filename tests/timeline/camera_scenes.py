"""Scenes the camera tests execute: camera callbacks, zoomed displays, a
mapping camera, a depth sort and a display of another camera.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Dot,
    FadeIn,
    ImageMobjectFromCamera,
    MappingCamera,
    MovingCamera,
    Scene,
    Square,
    ThreeDScene,
    VectorizedPoint,
    ZoomedScene,
)


class PureCentre(ThreeDScene):
    """A fixed-orientation label whose centre function reads a moving dot:
    the camera callback is a readout, so its plays are state functions.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        dot = Dot(LEFT * 2, color=RED)
        label = Square(side_length=0.4)
        self.add(dot)
        self.add_fixed_orientation_mobjects(
            label, center_func=lambda: dot.get_center() + UP
        )
        self.play(dot.animate.shift(RIGHT * 3), run_time=1.1)
        self.play(dot.animate.shift(DOWN * 2))


class StatefulCentre(ThreeDScene):
    """The same label with a centre function that counts its calls."""

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        dot = Dot(LEFT * 2, color=RED)
        label = Square(side_length=0.4)
        calls = []

        def centre():
            calls.append(1)
            return dot.get_center() + UP * min(len(calls), 90) / 60

        self.add(dot)
        self.add_fixed_orientation_mobjects(label, center_func=centre)
        self.play(dot.animate.shift(RIGHT * 3), run_time=1.1)


class ZoomStill(ZoomedScene):
    """Zooming switched on at once, then the zoomed frame moved over the
    scene and a shape grown under it.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(color=YELLOW)
        self.add(dot, Circle().shift(LEFT * 2))
        self.zoomed_camera.frame.move_to(dot)
        self.activate_zooming(animate=False)
        self.play(self.zoomed_camera.frame.animate.shift(LEFT * 2))
        self.play(FadeIn(Square().shift(LEFT * 2)), run_time=0.75)
        self.wait(0.5)


class ZoomAnimated(ZoomedScene):
    """Zooming switched on with its own animations (the frame zooming in,
    the display popping out), then the frame scaled.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.4, zoomed_display_height=2, zoomed_display_width=2, **kwargs
        )

    def construct(self):
        self.add(Square(color=RED), Dot(RIGHT))
        self.wait(0.25)
        self.activate_zooming(animate=True)
        self.play(self.zoomed_camera.frame.animate.scale(0.5).shift(RIGHT))


class DisplayLater(ZoomedScene):
    """The zoomed display is on screen from the start, but its camera joins
    the scene's camera only at 1/2 s: until then the display shows its
    camera's empty picture, after that the zoomed dot.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=3, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(radius=0.2, color=YELLOW)
        self.add(dot)
        self.zoomed_camera.frame.move_to(dot)
        self.add_foreground_mobjects(self.zoomed_display)
        self.play(dot.animate.shift(UP * 0.1), run_time=0.5)
        self.renderer.camera.add_image_mobject_from_camera(self.zoomed_display)
        self.play(dot.animate.shift(DOWN * 0.1), run_time=0.5)


class DisplayLeaves(ZoomedScene):
    """The zoomed display stays on screen after its camera leaves the
    scene's camera at 1/2 s: it shows the last picture drawn into it.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=3, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(radius=0.2, color=YELLOW)
        self.add(dot)
        self.zoomed_camera.frame.move_to(dot)
        self.activate_zooming()
        self.play(dot.animate.shift(UP * 0.1), run_time=0.5)
        self.renderer.camera.image_mobjects_from_cameras.remove(self.zoomed_display)
        self.play(dot.animate.shift(DOWN * 0.5), run_time=0.5)


class Mapped(Scene):
    """A mapping camera that draws every point two units to the right."""

    def __init__(self, **kwargs):
        super().__init__(
            camera_class=MappingCamera,
            **kwargs,
        )
        self.renderer.camera.mapping_func = lambda p: p + 2 * RIGHT

    def construct(self):
        dot = Dot(LEFT * 3, radius=0.3)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT * 2))
        self.wait(0.25)


class DepthReferenceMoves(ThreeDScene):
    """A red square the depth sort places by a point the scene does not hold
    (its ``z_index_group``), over a blue one: behind it until the square's
    updater moves the point, partway through the wait, then in front.
    """

    def construct(self):
        self.camera.should_apply_shading = False
        reference = VectorizedPoint([0.0, 0.0, -1.0])
        red = Square(color=RED, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        blue = Square(color=BLUE, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        red.z_index_group = reference
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and reference.get_center()[2] < 0:
                reference.shift(OUT * 2)

        red.add_updater(drive)
        self.add(red, blue)
        self.wait(0.4)


class StaticDepthReferenceMoves(ThreeDScene):
    """The same two squares, still: a dot's updater moves the red square's
    reference point partway through a play that moves only the dot, so the
    squares are the play's static mobjects and their order changes.
    """

    def construct(self):
        self.camera.should_apply_shading = False
        reference = VectorizedPoint([0.0, 0.0, -1.0])
        red = Square(color=RED, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        blue = Square(color=BLUE, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        red.z_index_group = reference
        dot = Dot(LEFT * 3, radius=0.1)
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and reference.get_center()[2] < 0:
                reference.shift(OUT * 2)

        dot.add_updater(drive)
        self.add(red, blue)
        self.play(dot.animate.shift(DOWN), run_time=0.4)


def _other_camera_display() -> tuple[MovingCamera, ImageMobjectFromCamera]:
    """A red camera the scene's camera does not draw into, and a display of
    its picture.
    """
    other = MovingCamera(pixel_width=32, pixel_height=18, background_color=RED)
    other.reset()
    display = ImageMobjectFromCamera(other)
    display.scale_to_fit_height(3)
    return other, display


class OtherCameraStill(Scene):
    """A display of another camera whose picture never changes, over three
    plays.
    """

    def construct(self):
        _, display = _other_camera_display()
        self.add(display)
        self.wait(0.2)
        self.play(display.animate.shift(LEFT), run_time=0.2)
        self.wait(0.2)


class OtherCameraBetweenPlays(Scene):
    """A display of another camera that the scene repaints between plays:
    red, then blue, then red again.
    """

    def construct(self):
        other, display = _other_camera_display()
        self.add(display)
        self.wait(0.2)
        other.background_color = BLUE
        other.reset()
        self.play(display.animate.shift(LEFT), run_time=0.2)
        other.background_color = RED
        other.reset()
        self.wait(0.2)


class OtherCameraShown(Scene):
    """A display of a camera the scene's camera does not draw into: its
    updater gives that camera a blue background partway through the wait,
    over the red one it had.
    """

    def construct(self):
        other, display = _other_camera_display()
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and other.background_color != BLUE:
                other.background_color = BLUE
                other.reset()

        display.add_updater(drive)
        self.add(display)
        self.wait(0.4)
