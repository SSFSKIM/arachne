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
    "engine_trace",
    "hold_rotation",
    "held_still",
    "home_box",
    "member_box",
    "member_place",
    "pixel_box",
    "pixel_points",
    "point_thickness",
    "record_view",
    "render_mapping_centre",
    "screen_box",
    "view_of",
]

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakKeyDictionary

import numpy as np

from .. import config
from ..timeline.mapping import render_mapping_centre

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject
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
        home=record._scene._arachne_timeline.home,
        configured_pixels=record._scene._arachne_timeline.configured_pixels,
    )


def draw_kind(mob: Mobject) -> DrawKind | None:
    """How a camera draws ``mob`` itself: a path, a point cloud, an image,
    or nothing (``Camera.display_funcs``, in its order).
    """
    for kind_type, kind in _draw_kinds():
        if isinstance(mob, kind_type):
            return kind
    return None


#: The types :func:`draw_kind` tells apart, in its order, once imported.
_DRAW_KINDS: list[tuple[type, DrawKind]] = []


def _draw_kinds() -> list[tuple[type, DrawKind]]:
    if not _DRAW_KINDS:
        from arachne.mobject.types.image_mobject import AbstractImageMobject
        from arachne.mobject.types.point_cloud_mobject import PMobject
        from arachne.mobject.types.vectorized_mobject import VMobject

        _DRAW_KINDS.extend(
            [(VMobject, "path"), (PMobject, "points"), (AbstractImageMobject, "image")]
        )
    return _DRAW_KINDS


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


def member_place(
    camera: Camera, view: View, member: Mobject
) -> tuple[Box, np.ndarray] | None:
    """What :func:`member_box` (with ``ink`` and ``home``) and
    :func:`pixel_points` of its points give for one member, together:
    ``None`` when its draw paints nothing. A path the camera traces with
    the engine's own code is projected once for both, as that trace
    projects its points once and traces those.
    """
    if not len(member.points):
        return None
    if draw_kind(member) == "path" and engine_trace(camera):
        _rotate(camera)
        shown = camera.transform_points_pre_display(member, member.points)
        ink = _ink_box(camera, view, member, home=True, shown=shown)
        if ink is None:
            return None
        flat = np.asarray(shown, dtype=float)[:, :2]
        return ink, _placed(view, flat, _mapped_around(view, member), home=False)
    ink = member_box(camera, view, member, ink=True, home=True)
    if ink is None:
        return None
    points = np.asarray(member.points, dtype=float)
    return ink, pixel_points(camera, view, member, points)


def engine_trace(camera: Camera) -> bool:
    """Whether ``camera`` traces a path with the engine's own code:
    ``Camera.set_cairo_context_path`` and the trace it ends in as the engine
    defined them, neither overridden by its class nor replaced, on the class
    or the instance. Asked once while the camera is :func:`held_still`.
    """
    hold = _hold_of(camera)
    if hold is None:
        return _engine_trace(camera)
    if hold.engine_trace is None:
        hold.engine_trace = _engine_trace(camera)
    return hold.engine_trace


def _engine_trace(camera: Camera) -> bool:
    from arachne.camera.camera import Camera
    from arachne.timeline.frozen import class_members

    taken = class_members(Camera)
    kind = type(camera)
    return all(
        name not in vars(camera) and getattr(kind, name, None) is taken.get(name)
        for name in ("set_cairo_context_path", "_trace_path")
    )


def point_thickness(view: View, member: Mobject) -> float:
    """The side of the square a point cloud's draw paints for each of its
    points, in home-frame units: as thick as the camera thickens it at the
    configured pixel size, wherever the point is and however much of the
    square the frame keeps.
    """
    side = len(_nudges(_cloud_thickness(view, member)))
    return side * view.home_width / view.pixel_width


def _cloud_thickness(view: View, member: Mobject) -> float:
    """The thickness a camera thickens a cloud's pixels by
    (``Camera.adjusted_thickness``), scaled from the configured pixel size
    rather than from whatever configuration is active now.
    """
    configured = view.configured_pixels[0] + view.configured_pixels[1]
    factor = configured / (view.pixel_width + view.pixel_height)
    return 1 + (float(member.get_stroke_width()) - 1) * factor


def _nudges(thickness: float) -> range:
    """The offsets, along each axis, a camera thickens a pixel by
    (``Camera.get_thickening_nudges``).
    """
    whole = int(thickness)
    return range(-whole // 2 + 1, whole // 2 + 1)


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


#: Each ThreeDCamera's rotation matrix as :func:`_rotate` last made it, with
#: the angles it was made from.
_ROTATIONS: WeakKeyDictionary[Any, tuple[tuple[float, float, float], np.ndarray]] = (
    WeakKeyDictionary()
)

#: What the engine's ThreeDCamera makes its rotation matrix with, and from.
_ROTATION_CODE = (
    "reset_rotation_matrix",
    "generate_rotation_matrix",
    "get_phi",
    "get_theta",
    "get_gamma",
)


class _Hold:
    """What is asked of a camera once while it is :func:`held_still`."""

    __slots__ = ("camera", "engine_trace", "rotated")

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        #: :func:`engine_trace` of the camera, once asked.
        self.engine_trace: bool | None = None
        #: Whether :func:`_rotate` has made the engine's matrix for the
        #: angles held.
        self.rotated = False


#: The cameras held still, innermost last.
_HOLDS: list[_Hold] = []


@contextmanager
def held_still(camera: Camera) -> Iterator[None]:
    """Hold ``camera`` still while what is inside places many members
    through it and moves none of what a projection reads: its trackers, its
    matrix, its code. :func:`engine_trace` is asked once, and the engine's
    rotation (:func:`_rotate`) is made at most once, at the first projection
    that makes it, as the first of many would; a rotation of the camera's
    own code is made at every projection still. Nothing is made on entry,
    so a read that projects nothing leaves the camera as it was.
    """
    _HOLDS.append(_Hold(camera))
    try:
        yield
    finally:
        _HOLDS.pop()


def _hold_of(camera: Camera) -> _Hold | None:
    if _HOLDS and _HOLDS[-1].camera is camera:
        return _HOLDS[-1]
    return None


def _rotate(camera: Camera) -> None:
    """Have a ThreeDCamera project with the matrix of its trackers as they
    are. The engine's matrix is made from its three angles alone, so it is
    made again only when they moved since it was last made here, or when
    something else replaced it (the camera's own capture); a camera whose
    class makes it in code of its own makes it again every time. While the
    camera is :func:`held_still`, neither moves after the first time.
    """
    hold = _hold_of(camera)
    if hold is not None and hold.rotated:
        return
    from arachne.camera.three_d_camera import ThreeDCamera

    if not hasattr(camera, "reset_rotation_matrix"):
        return
    if not isinstance(camera, ThreeDCamera) or not all(
        name not in vars(camera)
        and getattr(type(camera), name) is getattr(ThreeDCamera, name)
        for name in _ROTATION_CODE
    ):
        camera.reset_rotation_matrix()
        return
    if hold is not None:
        hold.rotated = True
    angles = (camera.get_phi(), camera.get_theta(), camera.get_gamma())
    made = _ROTATIONS.get(camera)
    if made is not None and made[0] == angles and camera.rotation_matrix is made[1]:
        return
    camera.reset_rotation_matrix()
    _ROTATIONS[camera] = (angles, camera.rotation_matrix)


def hold_rotation(camera: Camera) -> None:
    """Leave ``camera`` as a projection through it does: a ThreeDCamera's
    rotation made for its angles as they are, which its depth sort reads.
    """
    _rotate(camera)


def _pre_display(camera: Camera, mob: Mobject, points: np.ndarray) -> np.ndarray:
    """``points`` through the camera's pre-display code, as ``(n, 2)``."""
    _rotate(camera)
    shown = np.asarray(camera.transform_points_pre_display(mob, points), dtype=float)
    return shown[:, :2]


def _mapped_around(view: View, mob: Mobject) -> Centre:
    """The centre ``mob``'s draw maps around."""
    return view.mapping_centre if draw_kind(mob) == "path" else view.frame_center


def _placed(view: View, shown: np.ndarray, centre: Centre, *, home: bool) -> np.ndarray:
    """Pre-display points less ``centre``, in home-frame units or pixels:
    each coordinate scaled, and for pixels moved, by its own factor and
    offset, both columns at once.
    """
    shifted = shown - np.asarray(centre[:2])
    dtype = shifted.dtype
    if home:
        scale = (
            view.home_width / view.frame_width,
            view.home_height / view.frame_height,
        )
        return shifted * np.array(scale, dtype=dtype)
    scale = (
        view.pixel_width / view.frame_width,
        -(view.pixel_height / view.frame_height),
    )
    offset = (view.pixel_width / 2, view.pixel_height / 2)
    return shifted * np.array(scale, dtype=dtype) + np.array(offset, dtype=dtype)


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


def _ink_box(
    camera: Camera,
    view: View,
    member: Mobject,
    *,
    home: bool,
    shown: np.ndarray | None = None,
) -> Box | None:
    """What one member's own draw paints, or ``None`` for nothing. A
    path's points already through the camera's pre-display code are
    ``shown`` (:func:`member_place`).
    """
    kind = draw_kind(member)
    if kind == "path":
        corners = _path_ink(camera, view, member, shown)
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


def _path_ink(
    camera: Camera, view: View, member: Mobject, shown: np.ndarray | None = None
) -> np.ndarray | None:
    """The corners of the box a path's fill and strokes paint, in
    pre-display scene units: Cairo's own extents of the path the camera
    builds, stroked with the camera's width, join and cap, under the matrix
    the camera's context maps with (Cairo flattens curves to a tolerance in
    pixels). ``None`` when nothing is painted. With ``shown``, the path
    is traced through those points, as the engine's trace traces the points
    it projects.
    """
    assert isinstance(member, _path_type())
    cairo, caps, joins = _cairo()
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
    _rotate(camera)
    if shown is None:
        camera.set_cairo_context_path(ctx, member)
    else:
        camera._trace_path(ctx, member, shown)
    extents: list[tuple[float, float, float, float]] = []
    if _painted(member.get_fill_opacities()):
        extents.append(ctx.fill_extents())
    for background in (True, False):
        width = float(member.get_stroke_width(background))
        if width <= 0 or not _painted(member.get_stroke_opacities(background)):
            continue
        ctx.set_line_width(width * view.cairo_line_width_multiple)
        ctx.set_line_join(joins[member.joint_type])
        ctx.set_line_cap(caps[member.cap_style])
        extents.append(ctx.stroke_extents())
    found = [e for e in extents if e[2] > e[0] and e[3] > e[1]]
    if not found:
        return None
    return np.array(_extents_box(found))


def _extents_box(found: list[tuple[float, float, float, float]]) -> list[list[float]]:
    """The corners of the box around Cairo's extents ``found``: the least
    corner and the greatest. None is NaN (each passed a comparison), so
    Python's ``min`` and ``max`` take NumPy's values, but for the sign of a
    zero, which NumPy takes there.
    """
    if len(found) == 1:
        x0, y0, x1, y1 = found[0]
        return [[x0, y0], [x1, y1]]
    if any(value == 0 for extent in found for value in extent):
        box = np.array(found)
        return [
            [box[:, 0].min(), box[:, 1].min()],
            [box[:, 2].max(), box[:, 3].max()],
        ]
    columns = list(zip(*found, strict=True))
    return [
        [min(columns[0]), min(columns[1])],
        [max(columns[2]), max(columns[3])],
    ]


def _painted(opacities: object) -> bool:
    values = np.asarray(opacities, dtype=float)
    return bool(values.size) and bool((values > 0).any())


@cache
def _cairo() -> tuple[Any, dict[Any, Any], dict[Any, Any]]:
    """Cairo, and the camera's maps of a path's cap and join styles to it."""
    import cairo

    from arachne.camera.camera import CAP_STYLE_MAP, LINE_JOIN_MAP

    return cairo, CAP_STYLE_MAP, LINE_JOIN_MAP


@cache
def _path_type() -> type[VMobject]:
    from arachne.mobject.types.vectorized_mobject import VMobject

    return VMobject


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
    thick = camera.thickened_coordinates(coords, _cloud_thickness(view, member))
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
