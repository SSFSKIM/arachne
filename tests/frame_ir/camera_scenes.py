"""Scenes drawn through a camera the scene's code changed: a camera class of
its own, or an engine camera with a method replaced on the instance or the
class, the scene's camera or a display's. XF describes each such frame by
what the camera itself produced (the points its pre-display code placed, the
colours it shaded, the order it drew in), so Cairo draws it back byte for
byte; a change that reaches the frame some other way is drawn on Cairo, with
the reason.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DEGREES,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UL,
    UP,
    YELLOW,
    Dot,
    ImageMobject,
    MovingCamera,
    PMobject,
    Rotate,
    Square,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
    VMobject,
    ZoomedScene,
)


class ShiftedCamera(ThreeDCamera):
    """A camera class of the scene's own: its projection slides what it
    projects to the left, as a scene would to place its origin on screen.
    """

    def project_points(self, points):
        return super().project_points(points) + np.array([-1.0, 0.3, 0.0])


class NamedCamera(ThreeDCamera):
    """A camera class of the scene's own whose pre-display code reads what no
    record states (each mobject's ``name``) to place it.
    """

    def transform_points_pre_display(self, mobject, points):
        out = np.array(super().transform_points_pre_display(mobject, points))
        if mobject.name == "Dot":
            out[:, 1] += 0.5
        return out


class Solids(ThreeDScene):
    """A lit surface, axes whose tips are shaded with a gradient, a dot, a
    label fixed in the frame and one of fixed orientation, under an orbit:
    everything a 3D camera's projection and shading reach. ``patch`` changes
    the camera.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.patch(self.renderer.camera)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.4 * np.sin(u) * np.cos(v)]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(6, 6),
            fill_opacity=0.8,
        )
        axes = ThreeDAxes(
            x_range=[-2, 2], y_range=[-2, 2], z_range=[-1, 1], x_length=4, y_length=4
        )
        corner = Text("A", font_size=28, color=YELLOW).to_corner(UL)
        tag = Text("p", font_size=28, color=GREEN).move_to([1.5, 1.5, 0.6])
        self.add_fixed_in_frame_mobjects(corner)
        self.add_fixed_orientation_mobjects(tag)
        self.add(axes, surface, tag, Dot([-1.0, -1.0, 0.0], color=RED))
        self.play(Rotate(surface, 30 * DEGREES, axis=OUT), run_time=0.4)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)
        self.wait(0.2)

    def patch(self, camera) -> None:
        """Nothing: the engine's camera as it made it."""


class ShiftedSolids(Solids):
    """:class:`Solids` through :class:`ShiftedCamera` (drawn as the scene runs:
    the record refuses a camera class of the scene's own).
    """

    def __init__(self, **kwargs):
        super().__init__(camera_class=ShiftedCamera, **kwargs)


class NamedSolids(Solids):
    """:class:`Solids` through :class:`NamedCamera` (drawn as the scene runs)."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=NamedCamera, **kwargs)


class PatchedSolids(Solids):
    """:class:`Solids` through the engine's camera, its projection replaced on
    the instance (drawn from the record).
    """

    def patch(self, camera) -> None:
        project = camera.project_points

        def shifted(points):
            out = project(points)
            out[:, 0] += 1.0
            out[:, 1] -= 0.4
            return out

        camera.project_points = shifted


class Pictures(ThreeDScene):
    """A point cloud and an image placed in space, under an orbit: the kinds a
    camera places through its pre-display code other than paths. ``patch``
    changes the camera.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.patch(self.renderer.camera)
        cloud = PMobject(stroke_width=4)
        rng = np.random.default_rng(0)
        cloud.add_points(rng.uniform(-2, 2, size=(80, 3)) * [1, 1, 0.3], color=YELLOW)
        pixels = np.zeros((8, 12, 4), dtype=np.uint8)
        pixels[..., 0] = np.arange(12) * 20
        pixels[..., 1] = np.arange(8)[:, None] * 30
        pixels[..., 3] = 255
        image = ImageMobject(pixels).scale(1.5).shift([0.0, 1.0, 0.5])
        self.add(cloud, image)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)

    def patch(self, camera) -> None:
        """Nothing: the engine's camera as it made it."""


class ShiftedPictures(Pictures):
    """:class:`Pictures` through :class:`ShiftedCamera` (drawn as the scene runs)."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=ShiftedCamera, **kwargs)


class PatchedPictures(Pictures):
    """:class:`Pictures` with :class:`PatchedSolids`' projection on the
    instance (drawn from the record).
    """

    patch = PatchedSolids.patch


class TintedSolids(Solids):
    """:class:`Solids` through the engine's camera, its shading replaced on
    the instance: a method other than the projection that changes the
    picture.
    """

    def patch(self, camera) -> None:
        shade = camera.modified_rgbas

        def tinted(vmobject, rgbas):
            out = np.array(shade(vmobject, rgbas))
            out[:, 2] = 0.5 * out[:, 2] + 0.5
            return out

        camera.modified_rgbas = tinted


class Counted(ThreeDCamera):
    """A camera class of the scene's own counting its projections, its
    shadings and the view values its rotation reads: overrides that could
    change state each time they run.
    """

    calls = {"project_points": 0, "modified_rgbas": 0, "get_phi": 0, "get_zoom": 0}

    def project_points(self, points):
        Counted.calls["project_points"] += 1
        return super().project_points(points) + np.array([0.5, 0.0, 0.0])

    def modified_rgbas(self, vmobject, rgbas):
        Counted.calls["modified_rgbas"] += 1
        return super().modified_rgbas(vmobject, rgbas)

    def get_phi(self):
        Counted.calls["get_phi"] += 1
        return super().get_phi()

    def get_zoom(self):
        Counted.calls["get_zoom"] += 1
        return super().get_zoom()


class CountedLive(Solids):
    """:class:`Solids` through :class:`Counted` (drawn as the scene runs)."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=Counted, **kwargs)


class CountedRecord(Solids):
    """:class:`Solids` through the engine's camera with :class:`Counted`'s
    methods put on the instance (drawn from the record).
    """

    def patch(self, camera) -> None:
        project = camera.project_points
        shade = camera.modified_rgbas

        def counted_projection(points):
            Counted.calls["project_points"] += 1
            return project(points) + np.array([0.5, 0.0, 0.0])

        def counted_shading(vmobject, rgbas):
            Counted.calls["modified_rgbas"] += 1
            return shade(vmobject, rgbas)

        camera.project_points = counted_projection
        camera.modified_rgbas = counted_shading
        for name in ("get_phi", "get_zoom"):
            read = getattr(camera, name)

            def counted_read(read=read, name=name):
                Counted.calls[name] += 1
                return read()

            setattr(camera, name, counted_read)


class BorderedCamera(ThreeDCamera):
    """A camera class of the scene's own whose constructor routes paths to a
    drawing of its own: the engine's, then a red border stroked by hand.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display_funcs[VMobject] = self.bordered

    def bordered(self, vmobjects, pixel_array):
        self.display_multiple_vectorized_mobjects(vmobjects, pixel_array)
        _border(self, pixel_array)


def _border(camera, pixel_array) -> None:
    """A red border stroked by hand, through the camera's Cairo context."""
    ctx = camera.get_cairo_context(pixel_array)
    ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
    ctx.set_line_width(0.05)
    ctx.rectangle(-2.0, -1.0, 4.0, 2.0)
    ctx.stroke()


class Bordered(Solids):
    """:class:`Solids` through :class:`BorderedCamera`."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=BorderedCamera, **kwargs)


class Rewired(Solids):
    """:class:`Solids` through the engine's camera, its ``display_funcs``
    routed by the scene's code to :class:`BorderedCamera`'s drawing: no method
    of the camera replaced.
    """

    def patch(self, camera) -> None:
        paint = camera.display_funcs[VMobject]

        def bordered(vmobjects, pixel_array):
            paint(vmobjects, pixel_array)
            _border(camera, pixel_array)

        camera.display_funcs[VMobject] = bordered


class Mirrored(MovingCamera):
    """A display camera class of the scene's own: it shows its view mirrored
    left to right about its frame's centre.
    """

    def transform_points_pre_display(self, mobject, points):
        out = np.array(super().transform_points_pre_display(mobject, points))
        out[:, 0] = 2 * self.frame_center[0] - out[:, 0]
        return out


class ZoomMirrored(ZoomedScene):
    """A zoomed display whose camera is of :class:`Mirrored`: the display
    shows the scene's code's picture.
    """

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.4, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        square = Square(0.6, fill_opacity=1, color=RED)
        mark = Square(0.2, fill_opacity=1, color=BLUE).shift(RIGHT * 0.2 + UP * 0.2)
        self.add(square, mark)
        self.activate_zooming(animate=False)
        self.zoomed_camera.__class__ = Mirrored
        self.play(mark.animate.shift(LEFT * 0.3), run_time=0.4)


class StrokedTwice(ThreeDScene):
    """The engine's camera, the way it paints a path replaced on the
    instance: it strokes every path a second time, thinner. What it draws is
    Cairo calls of the scene's own, which XF does not carry.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        camera = self.renderer.camera
        paint = camera.display_vectorized

        def twice(vmobject, ctx):
            paint(vmobject, ctx)
            ctx.set_line_width(0.01)
            ctx.stroke_preserve()
            return camera

        camera.display_vectorized = twice
        self.add(Square(fill_opacity=0.5, color=BLUE))
        self.wait(0.2)


#: Every scene XF describes through what its changed camera produced.
DESCRIBED = [ShiftedSolids, PatchedSolids, TintedSolids, ZoomMirrored]
