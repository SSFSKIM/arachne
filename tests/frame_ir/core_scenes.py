"""3D scenes for the core path: the overlay Cairo draws over the core's frame.

Each scene has a twin without its overlay (``overlay = False``), so a test can
tell the pixels the overlay changes from those the core and Cairo already
agree on beneath it.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLACK,
    BLUE,
    DEGREES,
    GREEN,
    ORIGIN,
    RED,
    WHITE,
    YELLOW,
    Circle,
    ImageMobject,
    ImageMobjectFromCamera,
    Line,
    MovingCamera,
    MultiCamera,
    PMobject,
    Prism,
    Rectangle,
    Scene,
    Square,
    Surface,
    Text,
    ThreeDScene,
    ZoomedScene,
)

from .camera_scenes import ShiftedCamera


class FixedLabelOverSquare(ThreeDScene):
    """A translucent label and a translucent bar fixed in the frame, over an
    opaque square lying in space, under an orbit: the overlay drawn over core
    content that changes every frame.
    """

    overlay = True

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        square = Square(side_length=4, color=BLUE, fill_opacity=1.0)
        self.add(square)
        if self.overlay:
            label = Text("Label", font_size=72, color=YELLOW).move_to(ORIGIN)
            label.set_fill(opacity=0.6)
            bar = Rectangle(width=3, height=0.4, color=RED, fill_opacity=0.5)
            bar.set_stroke(WHITE, width=2, opacity=0.7).shift([0.0, -1.0, 0.0])
            self.add_fixed_in_frame_mobjects(label, bar)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)


class FixedLabelOverSquareBare(FixedLabelOverSquare):
    """:class:`FixedLabelOverSquare` without its overlay."""

    overlay = False


class CornerLabel(ThreeDScene):
    """A white label fixed in the frame's top right corner, over the
    background alone, the camera tilted: the core draws nothing but the
    background, Cairo the whole label.

    At 854 x 480 it is a tie in Cairo's own rounding: two points of a
    glyph's flat edge lie at y = 3.375 exactly, which CE's matrix puts at
    9559.5 / 256 of a pixel exactly, and Cairo's 24.8 conversion rounds a
    half to even. A last bit either way of 3.375 rounds it otherwise, on
    CE's Cairo as on the core path's overlay.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        label = Text("Overlay in a corner, W = 2", font_size=30, color=WHITE)
        self.add_fixed_in_frame_mobjects(label.to_corner([1.0, 1.0, 0.0]))
        self.wait(0.1)


class CornerLabelOwnCamera(CornerLabel):
    """:class:`CornerLabel` through a camera class of the scene's own (drawn
    as the scene runs).
    """

    def __init__(self, **kwargs):
        super().__init__(camera_class=ShiftedCamera, **kwargs)


class FlatSquares(Scene):
    """A 2D scene: on ``auto`` it never goes through XF."""

    def construct(self):
        moved = Square(color=RED)
        self.add(Square(color=BLUE, fill_opacity=0.5), moved)
        self.play(moved.animate.shift([1.0, 0.0, 0.0]), run_time=0.2)


class ShadedSolids(ThreeDScene):
    """A prism and a gently curved surface lit by CE's light, each face a
    two-stop gradient, under a short orbit: nothing occludes itself, so
    CE's order and the core's depth agree inside every face.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES)
        prism = Prism(dimensions=[1.6, 1.2, 1.0], fill_color=RED).shift([-2.2, 0, 0])
        surface = Surface(
            lambda u, v: np.array([u, v, 0.15 * np.sin(u) * np.cos(v)]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(6, 6),
            checkerboard_colors=[BLUE, GREEN],
        ).shift([1.6, 0, 0])
        self.add(prism, surface)
        self.move_camera(theta=-30 * DEGREES, run_time=0.4)


class LiveSolids(ThreeDScene):
    """A surface and a square whose stroke width is the scene's own code
    (``scenes.WideStroke``), under a short orbit: drawn as the scene runs
    (``LiveFrames``), through the engine's camera.
    """

    def construct(self):
        from .scenes import WideStroke

        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.15 * np.sin(u) * np.cos(v)]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(6, 6),
            checkerboard_colors=[BLUE, GREEN],
        )
        self.add(surface, WideStroke(color=YELLOW).shift([0, 0, 0.5]))
        self.move_camera(theta=-30 * DEGREES, run_time=0.4)


class StrokeUnderSquare(ThreeDScene):
    """An opaque square at z = 0 over a stroke at z = -1, which moves: seen
    from above, the square hides the stroke where they cross, though CE
    draws the stroke after it. ``line`` is the stroke's class.
    """

    line = Line

    def occluder(self):
        """What lies over the stroke."""
        return Square(side_length=2, color=BLUE, fill_opacity=1.0)

    def construct(self):
        self.set_camera_orientation(phi=20 * DEGREES, theta=-60 * DEGREES)
        square = self.occluder()
        stroke = self.line(
            [-3.0, 0.0, -1.0], [3.0, 0.0, -1.0], color=YELLOW, stroke_width=8
        )
        self.add(square, stroke)
        self.play(stroke.animate.shift([0.0, 0.4, 0.0]), run_time=0.2)


class BlackSquareOverStroke(StrokeUnderSquare):
    """The occluder in the background's colour, its stroke too: its capture
    leaves the frame as it found it.
    """

    def occluder(self):
        return Square(side_length=2, color=BLACK, fill_opacity=1.0)


class TranslucentSquareOverStroke(StrokeUnderSquare):
    """A half-transparent occluder."""

    def occluder(self):
        return Square(side_length=2, color=BLUE, fill_opacity=0.5)


class OutlineOverStroke(StrokeUnderSquare):
    """An occluder of stroke alone, which the stroke crosses."""

    def occluder(self):
        return Square(side_length=2, color=BLUE, stroke_width=12)


class SquareOffFrameOverStroke(StrokeUnderSquare):
    """The occluder outside the frame: its capture draws nothing on it."""

    def occluder(self):
        return Square(side_length=2, color=BLUE, fill_opacity=1.0).shift(
            [30.0, 0.0, 0.0]
        )


class OwnLine(Line):
    """A line whose stroke width is read through the scene's own code (which
    only hands it on): a scene with one is drawn as it runs (``LiveFrames``).
    """

    def get_stroke_width(self, background=False):
        return super().get_stroke_width(background)


class LiveStrokeUnderSquare(StrokeUnderSquare):
    """:class:`StrokeUnderSquare` drawn as the scene runs: each frame is a
    capture of the square, then one of the stroke.
    """

    line = OwnLine


class ResizedDisplayPoints(Scene):
    """A MultiCamera showing a display half the frame's size, of a point
    cloud and a line that moves: CE resizes the display's camera to the
    display as it captures, and widens the cloud's blocks by the ratio of
    the configured size to the camera's (``adjusted_thickness``). ``line``
    is the line's class.
    """

    line = Line

    def __init__(self, **kwargs):
        super().__init__(camera_class=MultiCamera, **kwargs)

    def construct(self):
        frame = self.renderer.camera.frame
        display = ImageMobjectFromCamera(MovingCamera())
        display.stretch_to_fit_width(frame.width / 2)
        display.stretch_to_fit_height(frame.height / 2)
        display.move_to([frame.width / 4, frame.height / 4, 0.0])
        self.renderer.camera.add_image_mobject_from_camera(display)
        cloud = PMobject(stroke_width=4)
        rng = np.random.default_rng(0)
        cloud.add_points(rng.uniform(-3, 3, size=(40, 3)) * [1, 1, 0], color=YELLOW)
        line = self.line([-3.0, -2.0, 0.0], [3.0, -2.0, 0.0], color=RED)
        self.add(cloud, line, display)
        self.play(line.animate.shift([0.0, 1.0, 0.0]), run_time=0.2)


class LiveResizedDisplayPoints(ResizedDisplayPoints):
    """:class:`ResizedDisplayPoints` drawn as the scene runs."""

    line = OwnLine


def _pixels(opacity: int = 255) -> np.ndarray:
    pixels = np.zeros((8, 12, 4), dtype=np.uint8)
    pixels[..., 0] = np.arange(12) * 20
    pixels[..., 1] = np.arange(8)[:, None] * 30
    pixels[..., 3] = opacity
    return pixels


class ImageInSpaceOrbit(ThreeDScene):
    """An image lying in space and a label fixed in frame, under an orbit:
    the image's overlay run is drawn again every frame, the label's is
    made again from the frame before's calls.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(Square(side_length=4, color=BLUE, fill_opacity=1.0))
        self.add(ImageMobject(_pixels()).scale(1.5).shift([0.0, 0.0, 0.5]))
        label = Text("Orbit", font_size=48, color=YELLOW).to_corner([-1, 1, 0])
        self.add_fixed_in_frame_mobjects(label)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)


class InterleavedOverlay(ThreeDScene):
    """Members fixed in frame interleaved: a translucent label, a translucent
    image, a second label, a gradient-filled bar and a stroked circle, over
    a surface under an orbit, with the second label changing colour
    partway: glyphs, strokes, translucent fills and gradients made again
    from the frame before's calls, a run before an image and after it.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(
            Surface(
                lambda u, v: np.array([u, v, 0.3 * np.sin(u) * np.cos(v)]),
                u_range=[-2, 2],
                v_range=[-2, 2],
                resolution=(6, 6),
                checkerboard_colors=[BLUE, GREEN],
            )
        )
        first = Text("First", font_size=48, color=YELLOW).shift([-3.0, 2.5, 0.0])
        first.set_fill(opacity=0.6)
        image = ImageMobject(_pixels(160)).scale(1.2).shift([0.0, 2.5, 0.0])
        second = Text("Second", font_size=48, color=WHITE).shift([3.0, 2.5, 0.0])
        bar = Rectangle(width=4, height=0.5, fill_opacity=0.8, stroke_width=0)
        bar.set_fill([RED, BLUE]).shift([0.0, -2.5, 0.0])
        ring = Circle(radius=0.6, color=RED).set_stroke(width=6, opacity=0.7)
        ring.shift([4.5, -2.5, 0.0])
        self.add_fixed_in_frame_mobjects(first, image, second, bar, ring)
        self.move_camera(theta=-30 * DEGREES, run_time=0.4)
        # Fill and stroke style changing mid-play, no point moving.
        self.play(second.animate.set_color(RED), run_time=0.2)
        self.play(ring.animate.set_stroke(width=12, opacity=0.4), run_time=0.2)
        self.move_camera(theta=-20 * DEGREES, run_time=0.2)


class BackgroundFilledOverlay(ThreeDScene):
    """A shape fixed in frame filled from a background image, between two
    labels fixed in frame, over a surface under an orbit: the filled shape
    is drawn anew every frame, the labels around it made again; the
    filled shape's fill shifts partway.
    """

    def construct(self):
        from PIL import Image

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(
            Surface(
                lambda u, v: np.array([u, v, 0.3 * np.sin(u) * np.cos(v)]),
                u_range=[-2, 2],
                v_range=[-2, 2],
                resolution=(6, 6),
                checkerboard_colors=[BLUE, GREEN],
            )
        )
        array = np.zeros((40, 60, 4), dtype=np.uint8)
        array[..., 0] = np.linspace(0, 255, 60, dtype=np.uint8)
        array[..., 2] = 200
        array[..., 3] = 255
        disc = Circle(radius=1.0, fill_opacity=1).shift([0.0, 2.0, 0.0])
        disc.color_using_background_image(Image.fromarray(array, mode="RGBA"))
        before = Text("Before", font_size=40, color=YELLOW).shift([-3.5, 2.5, 0.0])
        after = Text("After", font_size=40, color=WHITE).shift([3.5, 2.5, 0.0])
        self.add_fixed_in_frame_mobjects(before, disc, after)
        self.move_camera(theta=-30 * DEGREES, run_time=0.4)
        self.play(disc.animate.shift([0.5, 0.0, 0.0]), run_time=0.2)


class ZoomedFrameMoves(ZoomedScene):
    """A zoomed display (a MultiCamera display) whose camera's frame moves,
    a label in the main frame beside it: the display's picture changes
    every frame.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        self.add(Circle(color=RED).shift([-2.0, 0.0, 0.0]), Square(color=BLUE))
        self.add(Text("Zoom", font_size=36).to_corner([-1, -1, 0]))
        self.activate_zooming(animate=False)
        self.play(
            self.zoomed_camera.frame.animate.shift([-1.5, 0.0, 0.0]), run_time=0.4
        )
