"""Scenes the Frame IR tests draw on both draw paths: each exercises a part
of what XF must carry for Cairo to draw a frame from it alone.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    Dot3D,
    FadeIn,
    ImageMobject,
    LineJointType,
    MovingCameraScene,
    PMobject,
    Polygon,
    Rectangle,
    Rotate,
    Scene,
    Square,
    Star,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    ValueTracker,
    VGroup,
    ZoomedScene,
    always_redraw,
)

#: Armed by a test once a scene's forward run has ended: a callback the
#: scene's code holds raises if it is called after that.
ARMED = {"on": False}


def _watch() -> None:
    if ARMED["on"]:
        raise AssertionError("scene code ran while a frame was drawn")


class Shapes2D(Scene):
    """Fills, strokes, a background stroke, joints and caps, a gradient, a
    dashed line, an arrow, text glyphs and a z-index, through a few plays.
    """

    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.6).shift(LEFT * 3)
        star = Star(color=YELLOW, fill_opacity=1).set_stroke(width=8)
        star.joint_type = LineJointType.ROUND
        grad = Rectangle(width=3, height=1).set_fill(
            color=[RED, GREEN, BLUE], opacity=1
        )
        grad.set_sheen_direction(RIGHT).shift(DOWN * 2)
        dashed = DashedLine(LEFT * 4 + UP * 2, RIGHT * 4 + UP * 2, color=ORANGE)
        arrow = Arrow(LEFT, RIGHT * 2 + UP).set_stroke(width=6)
        label = Text("XF", font_size=48).shift(RIGHT * 3 + DOWN)
        label.set_stroke(BLUE, width=3, background=True)
        top = Circle(radius=0.6, color=WHITE, fill_opacity=1).set_z_index(2)
        self.add(square, star, grad, dashed, label, top)
        self.play(Create(arrow), square.animate.shift(RIGHT), run_time=0.4)
        self.play(Rotate(star, 1.0), top.animate.shift(DOWN), run_time=0.4)
        self.wait(0.2)


class Morph(Scene):
    """A straight-path transform: its frames blend two stored shapes."""

    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.5)
        circle = Circle(color=RED, fill_opacity=0.9).shift(RIGHT)
        self.add(square)
        self.wait(0.2)
        self.play(Transform(square, circle), run_time=0.6)
        self.wait(0.2)


class LaggedMorph(Scene):
    """A transform of a group with a lag: each leaf at its own alpha."""

    def construct(self):
        group = VGroup(*(Square(0.5).shift(RIGHT * i) for i in range(-2, 3)))
        target = VGroup(
            *(
                Circle(radius=0.3, color=GREEN).shift(UP + RIGHT * i)
                for i in range(-2, 3)
            )
        )
        self.add(group)
        self.play(Transform(group, target, lag_ratio=0.3), run_time=0.6)


class Clouds(Scene):
    """A point cloud and an image."""

    def construct(self):
        cloud = PMobject(stroke_width=4)
        rng = np.random.default_rng(0)
        points = rng.uniform(-3, 3, size=(200, 3)) * np.array([1, 0.6, 0])
        cloud.add_points(points, color=YELLOW)
        pixels = np.zeros((8, 12, 4), dtype=np.uint8)
        pixels[..., 0] = np.arange(12) * 20
        pixels[..., 1] = np.arange(8)[:, None] * 30
        pixels[..., 3] = 255
        image = ImageMobject(pixels).scale(2).shift(UP)
        self.add(cloud, image)
        self.play(image.animate.shift(RIGHT).set_opacity(0.5), run_time=0.4)
        self.play(cloud.animate.shift(LEFT), run_time=0.4)


class BackgroundImageFill(Scene):
    """A shape filled from a background image (CE's background-colored
    vectorized mobjects).
    """

    def construct(self):
        array = np.zeros((40, 60, 4), dtype=np.uint8)
        array[..., 0] = np.linspace(0, 255, 60, dtype=np.uint8)
        array[..., 2] = 200
        array[..., 3] = 255
        image = Image.fromarray(array, mode="RGBA")
        circle = Circle(radius=1.5, fill_opacity=1)
        circle.color_using_background_image(image)
        self.add(Square(color=GREEN, fill_opacity=1), circle)
        self.play(circle.animate.shift(RIGHT), run_time=0.4)


class MovingFrame(MovingCameraScene):
    """A moving camera: the frame zooms and pans."""

    def construct(self):
        self.add(Square(color=BLUE, fill_opacity=0.5), Dot(RIGHT * 2, color=RED))
        self.play(self.camera.frame.animate.scale(0.5).move_to(RIGHT), run_time=0.4)


class MovedFlatCentre(Scene):
    """A flat camera whose frame centre changes after its first frame: CE's
    cached context keeps the first centre for paths, while points use the
    live one.
    """

    def construct(self):
        cloud = PMobject(stroke_width=6)
        cloud.add_points(np.array([[1.0, 1.0, 0.0], [-1.0, 0.5, 0.0]]), color=RED)
        self.add(Square(color=BLUE), cloud)
        self.wait(0.2)
        self.renderer.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.play(FadeIn(Circle(radius=0.5)), run_time=0.3)


class Zoomed(ZoomedScene):
    """A zoomed display (a MultiCamera display) over a moving shape."""

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(color=YELLOW)
        self.add(dot, Circle().shift(LEFT * 2))
        self.activate_zooming(animate=False)
        self.play(dot.animate.shift(RIGHT * 0.3), run_time=0.4)


class Lit3D(ThreeDScene):
    """A surface (one face group), a sphere, axes, a label fixed in frame, a
    fixed-orientation label and an orbit.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * np.pi / 180, theta=-45 * np.pi / 180)
        axes = ThreeDAxes(x_range=(-3, 3), y_range=(-3, 3), z_range=(-2, 2))
        surface = Surface(
            lambda u, v: np.array([u, v, 0.4 * np.sin(u) * np.cos(v)]),
            u_range=(-2, 2),
            v_range=(-2, 2),
            resolution=(8, 8),
            checkerboard_colors=[BLUE, GREEN],
        )
        ball = Dot3D(point=UP * 2 + OUT, color=RED)
        title = Text("3D", font_size=36).to_corner(UP + LEFT)
        mark = Square(0.4, color=ORANGE).shift(RIGHT * 2 + OUT)
        self.add(axes, surface, ball, mark)
        self.add_fixed_in_frame_mobjects(title)
        self.add_fixed_orientation_mobjects(mark)
        self.wait(0.2)
        self.move_camera(theta=-20 * np.pi / 180, run_time=0.4)
        self.play(ball.animate.shift(DOWN * 2), run_time=0.4)


class Orbit(ThreeDScene):
    """A camera-only play: an orbit over unchanged mobjects."""

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-30 * np.pi / 180)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.3 * u * v]),
            u_range=(-1.5, 1.5),
            v_range=(-1.5, 1.5),
            resolution=(6, 6),
        )
        self.add(surface, Polygon(LEFT, RIGHT, UP, color=YELLOW))
        self.move_camera(theta=30 * np.pi / 180, run_time=0.4)


class MovedThreeDCentre(ThreeDScene):
    """A 3D camera whose frame centre moves: projection reads the live centre,
    the cached context the first one.
    """

    def construct(self):
        self.set_camera_orientation(phi=50 * np.pi / 180, theta=-60 * np.pi / 180)
        self.add(Square(color=BLUE, fill_opacity=0.4), Dot3D(OUT, color=RED))
        self.wait(0.1)
        self.move_camera(frame_center=RIGHT + UP * 0.5, run_time=0.4)


class Watched(ThreeDScene):
    """Scene code that raises once armed: an updater, an always-redrawn
    shape and a fixed-orientation centre function.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        tracker = ValueTracker(0)
        dot = Dot(color=RED)

        def follow(mob):
            _watch()
            mob.move_to(RIGHT * tracker.get_value())

        dot.add_updater(follow)

        def ring():
            _watch()
            return Circle(radius=0.5 + 0.2 * tracker.get_value(), color=GREEN)

        def centre():
            _watch()
            return dot.get_center() + UP

        label = Square(0.3)
        self.add(dot, always_redraw(ring))
        self.add_fixed_orientation_mobjects(label, center_func=centre)
        self.play(tracker.animate.set_value(1), run_time=0.4)
        self.wait(0.2)


SCENES = [
    Shapes2D,
    Morph,
    LaggedMorph,
    Clouds,
    BackgroundImageFill,
    MovingFrame,
    MovedFlatCentre,
    Zoomed,
    Lit3D,
    Orbit,
    MovedThreeDCentre,
    Watched,
]
