"""Scenes that show one object alone at a time under every camera, so a
frame's drawn pixels can be held against the object's projected box.

Each object is shown for one ``wait(SHOW)``; at 4 frames a second the wait's
second frame shows it alone. A ThreeDCamera's frame centre is set before the
first frame (the centre a path's mapping freezes) and moved after it.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DEGREES,
    UP,
    Circle,
    CubicBezier,
    ImageMobject,
    Line,
    MovingCameraScene,
    PMobject,
    Polygon,
    Scene,
    Square,
    Text,
    ThreeDScene,
    VMobject,
)
from tests.lint.shapes import framed_square

#: How long each object is shown alone.
SHOW = 0.5


def _cloud(centre) -> PMobject:
    """A point cloud: a 9 by 5 grid of points, 0.8 by 0.4 wide."""
    xs, ys = np.meshgrid(np.linspace(-0.4, 0.4, 9), np.linspace(-0.2, 0.2, 5))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    cloud = PMobject(stroke_width=1)
    cloud.add_points(points + np.asarray(centre, dtype=float))
    return cloud


def _image(centre) -> ImageMobject:
    """A white image 0.8 high."""
    pixels = np.full((20, 30, 4), 255, dtype=np.uint8)
    return ImageMobject(pixels).set(height=0.8).move_to(centre)


class Flat(Scene):
    """A circle, a square and a label under the flat camera."""

    def construct(self):
        circle = Circle(radius=1, stroke_width=8).move_to((2, 1, 0))
        self.add(circle)
        self.wait(SHOW)
        self.remove(circle)
        square = Square(side_length=2).move_to((-2, -1, 0))
        self.add(square)
        self.wait(SHOW)
        self.remove(square)
        label = Text("LIT").move_to((1, -1, 0))
        self.add(label)
        self.wait(SHOW)


class MovingBefore(MovingCameraScene):
    """A MovingCamera moved and zoomed before the first frame: a circle, a
    point cloud and an image, each alone.
    """

    def construct(self):
        self.camera.frame.move_to((1, 0.5, 0)).set(width=10)
        circle = Circle(radius=0.6, stroke_width=6).move_to((2, 1, 0))
        self.add(circle)
        self.wait(SHOW)
        self.remove(circle)
        cloud = _cloud((0, 1, 0))
        self.add(cloud)
        self.wait(SHOW)
        self.remove(cloud)
        image = _image((1.5, -0.5, 0))
        self.add(image)
        self.wait(SHOW)


class MovingDuring(MovingCameraScene):
    """A MovingCamera panning and zooming while a still circle is shown."""

    def construct(self):
        circle = Circle(radius=0.8, stroke_width=6).move_to((1, 0, 0))
        self.add(circle)
        self.play(
            self.camera.frame.animate.move_to((1.5, 0.5, 0)).set(width=9),
            run_time=1,
        )
        self.wait(SHOW)


class ThreeDCentred(ThreeDScene):
    """A ThreeDCamera whose frame centre is set before the first frame, then
    moved after it: each half shows a projected path, a point cloud, an
    image, a label fixed in the frame and a label of fixed orientation,
    each alone.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        for centre in ((1, 0.5, 0), (-1, -0.5, 0.5)):
            self.set_camera_orientation(frame_center=centre)
            path = Circle(radius=0.6, stroke_width=6).move_to(centre)
            self.add(path)
            self.wait(SHOW)
            self.remove(path)
            cloud = _cloud(centre)
            self.add(cloud)
            self.wait(SHOW)
            self.remove(cloud)
            image = _image(centre)
            self.add(image)
            self.wait(SHOW)
            self.remove(image)
            fixed = Text("FIX").scale(0.6).move_to((2, 1.5, 0))
            self.add_fixed_in_frame_mobjects(fixed)
            self.wait(SHOW)
            self.remove(fixed)
            oriented = Text("ORI").scale(0.6).move_to(np.add(centre, UP))
            self.add_fixed_orientation_mobjects(oriented)
            self.add(oriented)
            self.wait(SHOW)
            self.remove(oriented)


class ThreeDMovedLater(ThreeDScene):
    """A ThreeDCamera centred on the origin for its first frame, its centre
    moved after it: a projected path, then a path, a point cloud and an
    image at the new centre.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        path = Circle(radius=0.6, stroke_width=6)
        self.add(path)
        self.wait(SHOW)
        self.remove(path)
        centre = (1.5, 0.5, 0)
        self.set_camera_orientation(frame_center=centre)
        path = Circle(radius=0.6, stroke_width=6).move_to(centre)
        self.add(path)
        self.wait(SHOW)
        self.remove(path)
        cloud = _cloud(centre)
        self.add(cloud)
        self.wait(SHOW)
        self.remove(cloud)
        image = _image(centre)
        self.add(image)
        self.wait(SHOW)


class HelperMade(Scene):
    """A square made in a helper module."""

    def construct(self):
        square = framed_square()  # @helper-call
        self.add(square)
        self.wait(SHOW)


def _ink_cases() -> list[tuple[str, object]]:
    """Objects whose ink is not their points' box: stroke caps and joins,
    curves whose control points lie off them, paths that paint nothing, a
    thick point cloud, a cloud partly off the frame and an image too thin to
    draw.
    """
    bezier = ([-2, 0, 0], [-2, 3, 0], [2, 3, 0], [2, 0, 0])
    return [
        ("butt_line", Line([0, -1, 0], [0, 1, 0], stroke_width=30)),
        ("thick_diagonal", Line([-1, -1, 0], [1, 1, 0], stroke_width=80)),
        (
            "butt_point",
            VMobject(stroke_width=30).set_points_as_corners([[0, 0, 0], [0, 0, 0]]),
        ),
        (
            "flat_fill",
            Polygon([0, -1, 0], [0, 1, 0], [0, -1, 0], stroke_width=0, fill_opacity=1),
        ),
        ("acute", Polygon([-1, -1, 0], [0, 1, 0], [0.2, -1, 0], stroke_width=60)),
        ("stroked_curve", CubicBezier(*bezier, stroke_width=4)),
        ("filled_curve", CubicBezier(*bezier, stroke_width=0, fill_opacity=1)),
        ("filled_circle", Circle(radius=1, stroke_width=0, fill_opacity=1)),
        ("thick_cloud", PMobject(stroke_width=5).add_points(np.array([[0, 0, 0]]))),
        (
            "edge_cloud",
            PMobject(stroke_width=4).add_points(
                np.array([[-7.12, 0, 0], [-7.1, 0.1, 0]])
            ),
        ),
        (
            "thin_image",
            ImageMobject(np.full((2, 2, 4), 255, dtype=np.uint8)).stretch(0, 0),
        ),
    ]


#: The names of :class:`InkEdges`'s objects, in the order it shows them.
INK_CASES = [name for name, _ in _ink_cases()]


class InkEdges(Scene):
    """Each object of :func:`_ink_cases` alone, one wait each."""

    def construct(self):
        for name, mob in _ink_cases():
            setattr(self, name, mob)
            self.add(mob)
            self.wait(SHOW)
            self.remove(mob)
