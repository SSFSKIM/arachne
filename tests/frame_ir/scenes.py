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
    MappingCamera,
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
    ThreeDCamera,
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


class Reveals(Scene):
    """Partial reveals: each leaf of a Create drawn as a window of its path."""

    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.4)
        group = VGroup(
            Circle(color=RED).shift(LEFT * 2), Star(color=YELLOW).shift(RIGHT * 2)
        )
        self.play(Create(square), run_time=0.6)
        self.play(Create(group, lag_ratio=0.5), run_time=0.6)


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


# Drawn as the scene runs: a user override of a method the camera reads
# refuses the record, and the frames are drawn live.


class WideStroke(Square):
    """A square whose stroke width is the user's."""

    def get_stroke_width(self, background=False):
        return 3.0 if background else 9.0


class CountedFill(Square):
    """A square whose fill changes with every read of it: drawn as often as
    CE draws it, it ends where CE's ends.
    """

    calls = 0

    def get_fill_rgbas(self):
        type(self).calls += 1
        rgbas = np.array(super().get_fill_rgbas())
        rgbas[:, 0] = (type(self).calls % 9) / 9
        return rgbas


class LiveOverride(Scene):
    """A user override of a draw method."""

    def construct(self):
        square = WideStroke(color=BLUE)
        self.add(square)
        self.play(square.animate.shift(RIGHT), run_time=0.4)


class LiveCounted(Scene):
    """An override whose colour depends on how often it is called."""

    def construct(self):
        CountedFill.calls = 0
        square = CountedFill(fill_opacity=0.8)
        self.add(square, Circle(color=RED).shift(LEFT * 2))
        self.play(square.animate.shift(UP), run_time=0.4)
        self.wait(0.2)


class LiveMapped(Scene):
    """A MappingCamera: point clouds drawn where its mapping puts them."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)
        self.renderer.camera.mapping_func = lambda p: (
            p + np.array([0.5 * np.sin(p[1]), 0.0, 0.0])
        )

    def construct(self):
        cloud = PMobject(stroke_width=3)
        rng = np.random.default_rng(1)
        cloud.add_points(rng.uniform(-2, 2, size=(120, 3)) * [1, 1, 0], color=YELLOW)
        self.add(cloud, Dot(LEFT * 3, radius=0.3))
        self.play(cloud.animate.shift(RIGHT), run_time=0.4)
        self.wait(0.2)


class SplitAlone(Square):
    """A square that says its path has no subpaths, though it has points."""

    def get_subpath_split_indices_from_points(self, points, n_dims=3):
        return np.zeros((0, 2), dtype=int)


class LiveUnstated(Scene):
    """A read the record would answer otherwise than the mobject does."""

    def construct(self):
        self.add(SplitAlone(color=GREEN))
        self.wait(0.2)


class ShiftedCamera(ThreeDCamera):
    """A camera class of the scene's own: its projection slides the picture
    left (as a chapter's camera makes room for a text column).
    """

    def project_points(self, points):
        return super().project_points(points) + np.array([-1.0, 0.0, 0.0])


class UserCamera(ThreeDScene):
    """A 3D scene drawn through :class:`ShiftedCamera`."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=ShiftedCamera, **kwargs)

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        square = Square(fill_opacity=0.6, color=BLUE)
        self.add(square, Dot(RIGHT * 2))
        self.play(Rotate(square, np.pi / 4), run_time=0.4)
        self.wait(0.2)


class PatchedCamera(ThreeDScene):
    """The engine's ThreeDCamera, its projection replaced on the instance by
    the scene (as a chapter slides its axes beside a text column).
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        camera = self.renderer.camera
        project = camera.project_points

        def shifted(points):
            return project(points) + np.array([-1.0, 0.0, 0.0])

        camera.project_points = shifted
        self.add(*self.drawn())
        square = Square(fill_opacity=0.6, color=BLUE)
        self.add(square)
        self.play(Rotate(square, np.pi / 4), run_time=0.4)
        self.wait(0.2)

    def drawn(self) -> list:
        return [Dot(RIGHT * 2)]


class LivePatchedCamera(PatchedCamera):
    """As :class:`PatchedCamera`, with a user's path: drawn as it runs."""

    def drawn(self) -> list:
        return [WideStroke(color=GREEN).shift(LEFT * 2)]


def _reversed(get_mobjects_to_display):
    """A camera's display order, back to front: what a scene's patch of
    ``get_mobjects_to_display`` might make it.
    """

    def reversed_order(*args, **kwargs):
        return list(reversed(get_mobjects_to_display(*args, **kwargs)))

    return reversed_order


class Overlapping(Scene):
    """Two filled squares overlapping, one moving: their order shows."""

    def construct(self):
        self.patch(self.renderer.camera)
        back = Square(fill_opacity=1, color=BLUE)
        front = Square(fill_opacity=1, color=RED).shift(RIGHT * 0.5)
        self.add(back, front)
        self.play(front.animate.shift(UP * 0.5), run_time=0.4)

    def patch(self, camera) -> None:
        """Nothing: a test patches the engine's camera class."""


class PatchedOrder(Overlapping):
    """As :class:`Overlapping`, the camera's display order replaced on the
    instance by the scene.
    """

    def patch(self, camera) -> None:
        camera.get_mobjects_to_display = _reversed(camera.get_mobjects_to_display)


def _cloud() -> PMobject:
    cloud = PMobject(stroke_width=6)
    cloud.add_points(np.array([[1.0, 1.0, 0.0], [-1.0, 0.5, 0.0]]), color=RED)
    return cloud


# The mapping centre: CE freezes it into an array's Cairo context at the
# first draw through that context (a path), never at a point cloud's or an
# image's; a frame centre moved before that first draw is the one frozen.


class RecordCloudFirst(Scene):
    """A point cloud alone, then the frame centre moved, then a path."""

    def construct(self):
        self.add(_cloud())
        self.wait(0.2)
        self.renderer.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.add(Square())
        self.wait(0.2)


class LiveCloudFirst(Scene):
    """As :class:`RecordCloudFirst`, its path a user's: drawn live from the
    play the path joins, after record frames of the cloud alone.
    """

    def construct(self):
        self.add(_cloud())
        self.wait(0.2)
        self.renderer.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.add(WideStroke())
        self.wait(0.2)


class LiveFromStartCloudFirst(Scene):
    """Drawn live from the start (a user's path with no points draws
    nothing), a cloud alone, then the frame centre moved, then a path.
    """

    def construct(self):
        hollow = WideStroke()
        hollow.clear_points()
        self.add(hollow, _cloud())
        self.wait(0.2)
        self.renderer.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.add(Square())
        self.wait(0.2)


class LiveMovedCentre(Scene):
    """Drawn live, a path from the first frame, then the frame centre moved:
    the centre stays the first one.
    """

    def construct(self):
        self.add(WideStroke(), _cloud())
        self.wait(0.2)
        self.renderer.camera.frame_center = np.array([1.0, 0.5, 0.0])
        self.play(FadeIn(Circle(radius=0.5)), run_time=0.3)


MOVED_CENTRE_LIVE = [LiveCloudFirst, LiveFromStartCloudFirst, LiveMovedCentre]
LIVE_SCENES = [LiveOverride, LiveCounted, LiveMapped, *MOVED_CENTRE_LIVE]

SCENES = [
    Shapes2D,
    Morph,
    LaggedMorph,
    Reveals,
    Clouds,
    BackgroundImageFill,
    MovingFrame,
    MovedFlatCentre,
    RecordCloudFirst,
    Zoomed,
    Lit3D,
    Orbit,
    MovedThreeDCentre,
    Watched,
]
