"""Where a mobject lands on the picture: the camera's own projection, read
without drawing.

A frame places each mobject by the camera's code: a ThreeDCamera's
``transform_points_pre_display`` (its projection, a label fixed in the frame
left as it is, a label of fixed orientation moved with its projected
centre), then the map from scene units to pixels. That map is not one map:

- a path is drawn through the frame array's Cairo context, whose matrix
  subtracts the frame centre the context was made with, at the array's first
  draw through it: a Camera or ThreeDCamera keeps that context, so later
  draws keep that first centre, while a MovingCamera makes a new one for
  every draw, around its frame as it is then;
- a point cloud or an image is mapped by ``points_to_subpixel_coords``,
  which subtracts the frame centre as it is.

A ThreeDCamera's projection already subtracts its live centre, so a path
there lands at ``R(p - c_live)·f·zoom - c_frozen`` and a point cloud or an
image at ``R(p - c_live)·f·zoom - c_live``: the centre counts twice whenever
the subtracted one is not zero. The projection here reproduces that, by
calling the camera's pre-display code and mapping each kind with the centre
its draw maps with.

Positions come in pixels of the camera's pixel array, and in home-frame
units: scene units of the frame the configuration describes, centred and y
up, which is where the picture shows a thing whatever the camera does.
Under a camera that never moves they are the scene's units, exactly.

The caller installs the values to project first (``Drawing.show`` of a
capture), as a draw does.
"""

from __future__ import annotations

__all__ = [
    "Box",
    "ThreeDView",
    "View",
    "draw_kind",
    "home_box",
    "member_box",
    "pixel_box",
    "pixel_points",
    "record_view",
    "render_mapping_centre",
    "screen_box",
    "view_of",
]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .. import config
from ..timeline.mapping import render_mapping_centre

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Record

#: ``(x_min, y_min, x_max, y_max)``: in pixels, y down, or in home-frame
#: units, y up.
Box = tuple[float, float, float, float]
Centre = tuple[float, float, float]
DrawKind = Literal["path", "points", "image"]


@dataclass(frozen=True)
class ThreeDView:
    """A ThreeDCamera's projection settings."""

    phi: float
    theta: float
    gamma: float
    focal_distance: float
    zoom: float
    exponential_projection: bool


@dataclass(frozen=True)
class View:
    """What maps a camera's pre-display points to the picture.

    ``frame_width``, ``frame_height`` and ``frame_center`` are the live
    frame's (a MovingCamera's frame mobject's): what a ThreeDCamera's
    projection and the map of a point cloud or an image subtract.
    ``mapping_centre`` is what a path's Cairo matrix subtracts: frozen at the
    array's first draw through it, live for a camera that makes a new
    context for each draw (a MovingCamera). ``home_width`` and
    ``home_height`` are the frame the configuration describes, the unit of
    home-frame boxes. ``configured_pixels`` is the configuration's
    ``(pixel_width, pixel_height)``, which a point cloud's thickness is
    scaled from (``Camera.adjusted_thickness``).
    """

    camera: Literal["flat", "moving", "three_d"]
    pixel_width: int
    pixel_height: int
    frame_width: float
    frame_height: float
    frame_center: Centre
    mapping_centre: Centre
    home_width: float
    home_height: float
    cairo_line_width_multiple: float
    three_d: ThreeDView | None
    configured_pixels: tuple[int, int]


def view_of(
    camera: Camera,
    *,
    mapping_centre: Sequence[float] | None = None,
    home: tuple[float, float] | None = None,
    configured_pixels: tuple[int, int] | None = None,
) -> View:
    """The view of ``camera`` as it holds its values now.

    ``mapping_centre`` is the centre a path's map subtracts (the frozen one,
    :func:`render_mapping_centre`); when ``None``, the one the camera's own
    context for its array was made with, if it keeps one, else its frame's
    centre now. A camera that makes a context for each draw maps around its
    frame as it is, whatever is given. ``home`` is the configuration's frame
    as ``(width, height)``; when ``None``, the active configuration's, as
    the camera sizes its frame to its pixels. ``configured_pixels`` is the
    configuration's pixel size; when ``None``, the active configuration's.
    """
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    kind: Literal["flat", "moving", "three_d"]
    three_d = None
    if isinstance(camera, ThreeDCamera):
        kind = "three_d"
        three_d = ThreeDView(
            phi=float(camera.get_phi()),
            theta=float(camera.get_theta()),
            gamma=float(camera.get_gamma()),
            focal_distance=float(camera.get_focal_distance()),
            zoom=float(camera.get_zoom()),
            exponential_projection=bool(camera.exponential_projection),
        )
    elif isinstance(camera, MovingCamera):
        kind = "moving"
    else:
        kind = "flat"
    centre = _centre(camera.frame_center)
    if kind == "moving":
        mapping = centre
    elif mapping_centre is not None:
        mapping = _centre(mapping_centre)
    else:
        mapping = _cached_mapping_centre(camera) or centre
    if configured_pixels is None:
        configured_pixels = (int(config.pixel_width), int(config.pixel_height))
    if home is None:
        home = configured_home(*configured_pixels)
    return View(
        camera=kind,
        pixel_width=int(camera.pixel_width),
        pixel_height=int(camera.pixel_height),
        frame_width=float(camera.frame_width),
        frame_height=float(camera.frame_height),
        frame_center=centre,
        mapping_centre=mapping,
        home_width=float(home[0]),
        home_height=float(home[1]),
        cairo_line_width_multiple=float(camera.cairo_line_width_multiple),
        three_d=three_d,
        configured_pixels=(int(configured_pixels[0]), int(configured_pixels[1])),
    )


def configured_home(pixel_width: int, pixel_height: int) -> tuple[float, float]:
    """The active configuration's frame at ``pixel_width`` by
    ``pixel_height``: its width, and the height a camera sizes its frame to
    for those pixels (``Camera.resize_frame_shape``).
    """
    width = float(config.frame_width)
    return width, width / (pixel_width / pixel_height)


def record_view(record: Record) -> View:
    """The view of ``record``'s camera as it holds its values now (a
    capture shown), mapping paths around the centre the render froze, with
    the frame and pixel size the scene was executed under as configured.
    """
    camera = record._scene.renderer.camera
    frozen = render_mapping_centre(record)
    return view_of(
        camera,
        mapping_centre=frozen if frozen is not None else camera.frame_center,
        home=record._scene.timeline.home,
        configured_pixels=record._scene.timeline.configured_pixels,
    )


def draw_kind(mob: Mobject) -> DrawKind | None:
    """How a camera draws ``mob`` itself: a path, a point cloud, an image,
    or nothing (``Camera.display_funcs``, in its order).
    """
    from arachne.mobject.types.image_mobject import AbstractImageMobject
    from arachne.mobject.types.point_cloud_mobject import PMobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    if isinstance(mob, VMobject):
        return "path"
    if isinstance(mob, PMobject):
        return "points"
    if isinstance(mob, AbstractImageMobject):
        return "image"
    return None


def pixel_points(
    camera: Camera, view: View, mob: Mobject, points: np.ndarray
) -> np.ndarray:
    """``points`` of ``mob`` where the camera draws them, as ``(n, 2)``
    pixel coordinates (x right, y down; pixel ``i`` spans ``[i, i + 1)``).
    """
    shown = _pre_display(camera, mob, points)
    return _placed(view, shown, _mapped_around(view, mob), home=False)


def pixel_box(
    camera: Camera, view: View, mob: Mobject, *, ink: bool = False
) -> Box | None:
    """The pixel box of ``mob`` and its family's points, as the camera
    places them; ``None`` when the family has no points.

    With ``ink``, the box of what the family's draws paint instead, ``None``
    when they paint nothing: a path's fill and strokes as Cairo extends them
    (the curve, not its control points; caps, joins and miters with the
    camera's settings), not cut to the frame; a point cloud's pixels,
    thickened as the camera thickens them at the configured size and kept
    to the frame as its draw keeps them; an image's corners, unless its draw
    skips it as too thin.
    """
    return _box(camera, view, mob, ink=ink, home=False)


def home_box(
    camera: Camera, view: View, mob: Mobject, *, ink: bool = False
) -> Box | None:
    """:func:`pixel_box` in home-frame units, computed from the points
    without going through pixels: under a camera that never moves, the
    points' own scene units, exactly.
    """
    return _box(camera, view, mob, ink=ink, home=True)


def member_box(
    camera: Camera,
    view: View,
    member: Mobject,
    *,
    ink: bool = False,
    home: bool = False,
) -> Box | None:
    """The box of one member's own draw, none of its children's: of its
    points as the camera places them, or with ``ink`` of what its draw
    paints (as :func:`pixel_box` says it), in pixels or with ``home`` in
    home-frame units; ``None`` when it has no points or paints nothing.
    """
    if not len(member.points):
        return None
    if ink:
        return _ink_box(camera, view, member, home=home)
    return _points_box(camera, view, member, home=home)


def screen_box(view: View, box: Box) -> Box:
    """A pixel box in home-frame units: scene units of the configuration's
    frame, centred, y up.
    """
    x_scale = view.home_width / view.pixel_width
    y_scale = view.home_height / view.pixel_height
    half_width, half_height = view.pixel_width / 2, view.pixel_height / 2
    return (
        (box[0] - half_width) * x_scale,
        (half_height - box[3]) * y_scale,
        (box[2] - half_width) * x_scale,
        (half_height - box[1]) * y_scale,
    )


def _cached_mapping_centre(camera: Camera) -> Centre | None:
    """The centre the camera's kept context for its array maps around,
    read from the context's matrix; ``None`` when it keeps none.
    """
    ctx = camera.get_cached_cairo_context(camera.pixel_array)
    if ctx is None:
        return None
    matrix = ctx.get_matrix()
    x = (camera.pixel_width / 2 - matrix.x0) / (camera.pixel_width / camera.frame_width)
    y = (matrix.y0 - camera.pixel_height / 2) / (
        camera.pixel_height / camera.frame_height
    )
    return (float(x), float(y), 0.0)


def _centre(value: Sequence[float] | np.ndarray) -> Centre:
    x, y, z = (float(v) for v in np.asarray(value, dtype=float).reshape(3))
    return (x, y, z)


def _pre_display(camera: Camera, mob: Mobject, points: np.ndarray) -> np.ndarray:
    """``points`` through the camera's pre-display code, as ``(n, 2)``."""
    if hasattr(camera, "reset_rotation_matrix"):
        # A ThreeDCamera projects with the matrix of its trackers as they are.
        camera.reset_rotation_matrix()
    shown = np.asarray(camera.transform_points_pre_display(mob, points), dtype=float)
    return shown[:, :2]


def _mapped_around(view: View, mob: Mobject) -> Centre:
    """The centre ``mob``'s draw maps around."""
    return view.mapping_centre if draw_kind(mob) == "path" else view.frame_center


def _placed(view: View, shown: np.ndarray, centre: Centre, *, home: bool) -> np.ndarray:
    """Pre-display points less ``centre``, in home-frame units or pixels."""
    shifted = shown - np.asarray(centre[:2])
    result = np.empty_like(shifted)
    if home:
        result[:, 0] = shifted[:, 0] * (view.home_width / view.frame_width)
        result[:, 1] = shifted[:, 1] * (view.home_height / view.frame_height)
    else:
        result[:, 0] = (
            shifted[:, 0] * (view.pixel_width / view.frame_width) + view.pixel_width / 2
        )
        result[:, 1] = (
            shifted[:, 1] * -(view.pixel_height / view.frame_height)
            + view.pixel_height / 2
        )
    return result


def _box(
    camera: Camera, view: View, mob: Mobject, *, ink: bool, home: bool
) -> Box | None:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for member in mob.family_members_with_points():
        found = (
            _ink_box(camera, view, member, home=home)
            if ink
            else _points_box(camera, view, member, home=home)
        )
        if found is not None:
            lows.append(np.array(found[:2]))
            highs.append(np.array(found[2:]))
    if not lows:
        return None
    low = np.min(lows, axis=0)
    high = np.max(highs, axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _points_box(camera: Camera, view: View, member: Mobject, *, home: bool) -> Box:
    """The box of a member's points, as its draw places them."""
    shown = _pre_display(camera, member, member.points)
    return _span(_placed(view, shown, _mapped_around(view, member), home=home))


def _span(placed: np.ndarray) -> Box:
    low, high = placed.min(axis=0), placed.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _ink_box(camera: Camera, view: View, member: Mobject, *, home: bool) -> Box | None:
    """What one member's own draw paints, or ``None`` for nothing."""
    kind = draw_kind(member)
    if kind == "path":
        corners = _path_ink(camera, view, member)
        if corners is None:
            return None
        return _span(_placed(view, corners, view.mapping_centre, home=home))
    if kind == "points":
        found = _cloud_ink(camera, view, member)
        if found is None or not home:
            return found
        return screen_box(view, found)
    if kind == "image":
        if not _image_painted(camera, view, member):
            return None
        return _points_box(camera, view, member, home=home)
    return None


def _path_ink(camera: Camera, view: View, member: Mobject) -> np.ndarray | None:
    """The corners of the box a path's fill and strokes paint, in
    pre-display scene units: Cairo's own extents of the path the camera
    builds, stroked with the camera's width, join and cap, under the matrix
    the camera's context maps with (Cairo flattens curves to a tolerance in
    pixels). ``None`` when nothing is painted.
    """
    import cairo

    from arachne.camera.camera import CAP_STYLE_MAP, LINE_JOIN_MAP
    from arachne.mobject.types.vectorized_mobject import VMobject

    assert isinstance(member, VMobject)
    ctx = cairo.Context(_extents_surface())
    centre = view.mapping_centre
    x_scale = view.pixel_width / view.frame_width
    y_scale = view.pixel_height / view.frame_height
    ctx.set_matrix(
        cairo.Matrix(
            x_scale,
            0,
            0,
            -y_scale,
            view.pixel_width / 2 - centre[0] * x_scale,
            view.pixel_height / 2 + centre[1] * y_scale,
        )
    )
    if hasattr(camera, "reset_rotation_matrix"):
        camera.reset_rotation_matrix()
    camera.set_cairo_context_path(ctx, member)
    extents: list[tuple[float, float, float, float]] = []
    if _painted(member.get_fill_opacities()):
        extents.append(ctx.fill_extents())
    for background in (True, False):
        width = float(member.get_stroke_width(background))
        if width <= 0 or not _painted(member.get_stroke_opacities(background)):
            continue
        ctx.set_line_width(width * view.cairo_line_width_multiple)
        ctx.set_line_join(LINE_JOIN_MAP[member.joint_type])
        ctx.set_line_cap(CAP_STYLE_MAP[member.cap_style])
        extents.append(ctx.stroke_extents())
    found = [e for e in extents if e[2] > e[0] and e[3] > e[1]]
    if not found:
        return None
    box = np.array(found)
    return np.array(
        [
            [box[:, 0].min(), box[:, 1].min()],
            [box[:, 2].max(), box[:, 3].max()],
        ]
    )


def _painted(opacities: object) -> bool:
    values = np.asarray(opacities, dtype=float)
    return bool(values.size and np.any(values > 0))


_SURFACE: list[Any] = []


def _extents_surface() -> Any:
    """A surface for Cairo to measure paths on; extents ignore its bounds."""
    import cairo

    if not _SURFACE:
        _SURFACE.append(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
    return _SURFACE[0]


def _cloud_ink(camera: Camera, view: View, member: Mobject) -> Box | None:
    """The pixels a point cloud's draw paints: each point's pixel (the
    camera truncates), thickened by the camera's nudges at the thickness it
    scales to from the configured pixel size, those on the frame only.
    """
    shown = _pre_display(camera, member, member.points)
    coords = _placed(view, shown, view.frame_center, home=False).astype(np.int64)
    # Camera.adjusted_thickness, scaled from the configured pixel size rather
    # than from whatever configuration is active now.
    configured = view.configured_pixels[0] + view.configured_pixels[1]
    factor = configured / (view.pixel_width + view.pixel_height)
    thickness = 1 + (member.get_stroke_width() - 1) * factor
    thick = camera.thickened_coordinates(coords, thickness)
    thick = thick[camera.on_screen_pixels(thick)]
    if not len(thick):
        return None
    low, high = thick.min(axis=0), thick.max(axis=0) + 1
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _image_painted(camera: Camera, view: View, member: Mobject) -> bool:
    """Whether the camera's draw of an image paints it: it skips a quad with
    no side, or one less than half a pixel high from its longest side
    (``Camera.display_image_mobject``), and a raster whose every pixel is
    fully transparent paints nothing (its opacity is in its alpha).
    """
    raster = getattr(member, "pixel_array", None)
    if raster is not None:
        raster = np.asarray(raster)
        if raster.ndim == 3 and raster.shape[-1] == 4 and not raster[..., 3].any():
            return False
    shown = _pre_display(camera, member, member.points)
    target = _placed(view, shown, view.frame_center, home=False)
    target = target - target.astype(np.int64).min(axis=0)
    vertices = [target[i] for i in (0, 1, 3, 2)]
    sides = [vertices[(i + 1) % 4] - vertices[i] for i in range(4)]
    lengths = np.linalg.norm(sides, axis=1)
    longest = int(np.argmax(lengths))
    if lengths[longest] == 0:
        return False
    base, previous = sides[longest], sides[(longest - 1) % 4]
    height = abs(base[0] * previous[1] - base[1] * previous[0]) / lengths[longest]
    return bool(height >= 0.5)
