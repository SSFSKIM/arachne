"""Scenes the camera tests execute: camera callbacks, zoomed displays, a
mapping camera, a depth sort, a display of another camera and a fill from an
image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

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
    Group,
    ImageMobjectFromCamera,
    MappingCamera,
    MovingCamera,
    PMobject,
    Point,
    Scene,
    Square,
    ThreeDScene,
    ValueTracker,
    VectorizedPoint,
    ZoomedScene,
    config,
    linear,
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


class DisplayRepaintedBeforeItJoins(ZoomedScene):
    """The zoomed display is on screen before its camera joins the scene's
    camera: that camera's picture is red for the first wait and blue for the
    second (repainted between them), then the camera joins and the display
    shows the zoomed dot.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=3, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(radius=0.2, color=YELLOW)
        self.add(dot)
        camera = self.zoomed_camera
        camera.frame.move_to(dot)
        camera.background_color = RED
        camera.reset()
        self.add_foreground_mobjects(self.zoomed_display)
        self.wait(0.2)
        camera.background_color = BLUE
        camera.reset()
        self.wait(0.2)
        self.renderer.camera.add_image_mobject_from_camera(self.zoomed_display)
        self.play(dot.animate.shift(UP * 0.1), run_time=0.2)


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


class RestoredDepthReference(ThreeDScene):
    """The red square of :class:`DepthReferenceMoves` behind the blue one,
    following a tracker by an updater (a state play, restored between its
    frames); after the play its reference point moves in front.
    """

    def construct(self):
        self.camera.should_apply_shading = False
        reference = VectorizedPoint([0.0, 0.0, -1.0])
        red = Square(color=RED, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        blue = Square(color=BLUE, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        red.z_index_group = reference
        slide = ValueTracker(0)
        red.add_updater(lambda mob: mob.set_x(slide.get_value()))
        self.add(red, blue)
        self.play(slide.animate.set_value(0.3), run_time=0.4, rate_func=linear)
        red.clear_updaters()
        reference.shift(OUT * 2)
        self.wait(0.2)


class PixelShapeChanged(Scene):
    """A point cloud on screen while an updater doubles the configured pixel
    width partway through the wait: its points are drawn thicker from then.
    """

    def construct(self):
        cloud = PMobject(stroke_width=4)
        cloud.add_points(np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]), color=YELLOW)
        start = config.pixel_width
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and config.pixel_width == start:
                config.pixel_width = start * 2

        cloud.add_updater(drive)
        self.add(cloud)
        self.wait(0.4)


def _lit_square() -> tuple[Square, Point]:
    """A shaded square, and a lamp off screen behind it."""
    square = Square(side_length=3, fill_opacity=1, stroke_width=0, shade_in_3d=True)
    square.set_color(BLUE)
    return square, Point(np.array([20.0, 0.0, -10.0]))


class LightReboundInPlay(ThreeDScene):
    """A shaded square lit by the camera's light until an updater rebinds
    the light, partway through the wait, to a lamp the scene holds.
    """

    def construct(self):
        square, lamp = _lit_square()
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and self.camera.light_source is not lamp:
                self.camera.light_source = lamp

        square.add_updater(drive)
        self.add(square, lamp)
        self.wait(0.4)


class LightReboundBetweenPlays(ThreeDScene):
    """As :class:`LightReboundInPlay`, the light rebound between two waits."""

    def construct(self):
        square, lamp = _lit_square()
        self.add(square, lamp)
        self.wait(0.2)
        self.camera.light_source = lamp
        self.wait(0.2)


class RestoredMovingDepthGroup(ThreeDScene):
    """Red and blue squares in one place, red placed by a depth group the
    scene holds, which an updater carries from behind the blue square to in
    front of it as a tracker slides (a state play, restored between its
    frames).
    """

    def construct(self):
        self.camera.should_apply_shading = False
        red = Square(color=RED, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        blue = Square(color=BLUE, fill_opacity=1, stroke_width=0, shade_in_3d=True)
        slide = ValueTracker(-0.9)
        group = VectorizedPoint([0.0, 0.0, -0.9])
        group.add_updater(lambda point: point.set_z(slide.get_value()))
        red.z_index_group = group
        self.add(red, blue, group)
        self.play(slide.animate.set_value(1.1), run_time=0.4, rate_func=linear)


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


class OtherCameraHidden(Scene):
    """A display of another camera, in a group: an updater takes it out of the
    group partway through the wait, repaints its camera blue while it is out,
    and puts it back.
    """

    def construct(self):
        other, display = _other_camera_display()
        group = Group(display)
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if 0.1 <= elapsed[0] < 0.3 and display in group.submobjects:
                group.remove(display)
            elif elapsed[0] >= 0.2 and other.background_color != BLUE:
                other.background_color = BLUE
                other.reset()
            elif elapsed[0] >= 0.3 and display not in group.submobjects:
                group.add(display)

        clock = Dot(LEFT * 6, radius=0.05)
        clock.add_updater(drive)
        self.add(group, clock)
        self.wait(0.5)


class ImageFillPainted(Scene):
    """A circle filled from a red image, which an updater paints blue in
    place partway through the wait: the camera keeps the image as it first
    read it, so the circle stays red.
    """

    def construct(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        circle = Circle(radius=1.5, fill_opacity=1, stroke_width=0)
        circle.color_using_background_image(image)
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and image.getpixel((0, 0)) != (0, 0, 255, 255):
                image.paste((0, 0, 255, 255), (0, 0, 8, 8))

        circle.add_updater(drive)
        self.add(circle)
        self.wait(0.4)


class ImageFillResized(Scene):
    """A circle filled from a red image, which an updater paints blue and
    shrinks in place partway through the wait: the image's name (its size
    among it) changes, under which the camera reads it anew, so the circle
    turns blue.
    """

    def construct(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        circle = Circle(radius=1.5, fill_opacity=1, stroke_width=0)
        circle.color_using_background_image(image)
        elapsed = [0.0]

        def drive(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and image.size != (4, 4):
                image.paste((0, 0, 255, 255), (0, 0, 8, 8))
                image.thumbnail((4, 4))

        circle.add_updater(drive)
        self.add(circle)
        self.wait(0.4)


class ImageFillZoomedAfterPainting(ZoomedScene):
    """A circle filled from a red image, painted blue in place after a wait,
    then zoomed: the scene's camera keeps the image red, as it first read it,
    and the zoomed camera, which first reads it after the painting, shows it
    blue.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        circle = Circle(radius=1.5, fill_opacity=1, stroke_width=0)
        circle.color_using_background_image(image)
        self.add(circle)
        self.wait(0.2)
        image.paste((0, 0, 255, 255), (0, 0, 8, 8))
        self.activate_zooming(animate=False)
        self.wait(0.2)


class _OwnFill(Dot):
    """A user's dot that computes its own fill: a play drawing it is drawn
    as the scene runs.
    """

    def get_fill_rgbas(self):
        return np.array([[0.0, 1.0, 0.0, 1.0]])


class ImageFillPaintedBeforeALivePlay(Scene):
    """A circle filled from a red image for a wait, painted blue in place,
    then a wait the record can't draw (a dot computing its own fill joins):
    the camera drawing it as the scene runs still holds the red it first
    read.
    """

    def construct(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        circle = Circle(radius=1.5, fill_opacity=1, stroke_width=0)
        circle.color_using_background_image(image)
        self.add(circle)
        self.wait(0.2)
        image.paste((0, 0, 255, 255), (0, 0, 8, 8))
        self.add(_OwnFill().shift(LEFT * 5))
        self.wait(0.2)


def _cloud() -> PMobject:
    cloud = PMobject(stroke_width=4)
    cloud.add_points(np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]), color=YELLOW)
    return cloud


class PixelShapeChangedAtTheEnd(Scene):
    """A point cloud for a wait; after it, the scene doubles the configured
    pixel width and ends: its final frame's points are drawn thicker.
    """

    def construct(self):
        self.add(_cloud())
        self.wait(0.2)
        config.pixel_width *= 2


class PixelShapeChangedMidPlay(Scene):
    """A point cloud whose updater doubles the configured pixel width halfway
    through a tracker's play: the frames after it draw the points thicker.
    """

    def construct(self):
        cloud = _cloud()
        slide = ValueTracker(0)
        widened = []

        def widen(mob):
            if slide.get_value() >= 0.5 and not widened:
                widened.append(True)
                config.pixel_width *= 2

        cloud.add_updater(widen)
        self.add(cloud, slide)
        self.play(slide.animate.set_value(1), run_time=0.4, rate_func=linear)


class PixelShapeChangedInSetup(Scene):
    """The configured pixel width doubled in ``setup``, before any play:
    every point is drawn thicker than at the pixel shape the scene was given.
    """

    def setup(self):
        config.pixel_width *= 2

    def construct(self):
        self.add(_cloud())
        self.wait(0.2)
