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
    "pixel_box",
    "pixel_points",
    "record_view",
    "render_mapping_centre",
    "screen_box",
    "view_of",
]

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .. import config

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, Record

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
    home-frame boxes.
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


def view_of(
    camera: Camera,
    *,
    mapping_centre: Sequence[float] | None = None,
    home: tuple[float, float] | None = None,
) -> View:
    """The view of ``camera`` as it holds its values now.

    ``mapping_centre`` is the centre a path's map subtracts (the frozen one,
    :func:`render_mapping_centre`); when ``None``, the one the camera's own
    context for its array was made with, if it keeps one, else its frame's
    centre now. A camera that makes a context for each draw maps around its
    frame as it is, whatever is given. ``home`` is the configuration's frame
    as ``(width, height)``; when ``None``, the active configuration's, as
    the camera sizes its frame to its pixels.
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
    if home is None:
        home = configured_home(int(config.pixel_width), int(config.pixel_height))
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
    capture shown), mapping paths around the centre the render froze, and
    with the frame the scene was executed under as home.
    """
    camera = record._scene.renderer.camera
    frozen = render_mapping_centre(record)
    return view_of(
        camera,
        mapping_centre=frozen if frozen is not None else camera.frame_center,
        home=record._scene.timeline.home,
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
    shifted = _shifted(camera, view, mob, points)
    result = np.empty_like(shifted)
    result[:, 0] = (
        shifted[:, 0] * (view.pixel_width / view.frame_width) + view.pixel_width / 2
    )
    result[:, 1] = (
        shifted[:, 1] * -(view.pixel_height / view.frame_height) + view.pixel_height / 2
    )
    return result


def pixel_box(
    camera: Camera, view: View, mob: Mobject, *, ink: bool = False
) -> Box | None:
    """The pixel box of ``mob`` and its family's points, as the camera
    places them; ``None`` when the family has no points. With ``ink``, a
    path's box is widened by half the stroke Cairo draws, and a point
    cloud's by the pixels each point is thickened to.
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


def render_mapping_centre(record: Record) -> Centre | None:
    """The centre the render of ``record`` froze into its frame array's
    Cairo context: the frame's centre at the first draw that paints a path
    through it, in the order a render draws (each drawn play's static
    picture, then its frames; the final frame when the scene played
    nothing). ``None`` for a camera that maps every draw around its frame as
    it is then (a MovingCamera), and when no drawn frame paints a path.
    Found from the record's captures, without drawing.
    """
    from arachne.camera.moving_camera import MovingCamera

    if record in _FROZEN:
        return _FROZEN[record]
    camera = record._scene.renderer.camera
    found = None if isinstance(camera, MovingCamera) else _first_path_centre(record)
    if record.final is not None:
        # A finished record draws no more frames.
        _FROZEN[record] = found
    return found


_FROZEN: weakref.WeakKeyDictionary[Record, Centre | None] = weakref.WeakKeyDictionary()


def _first_path_centre(record: Record) -> Centre | None:
    from arachne.timeline.clock import exact, frames_in
    from arachne.timeline.draw import Drawing

    camera = record._scene.renderer.camera
    fps = exact(camera.frame_rate)
    with Drawing(record) as drawing:

        def centre_if_path(
            capture: Capture, targets: Sequence[Mobject]
        ) -> Centre | None:
            drawing.show(capture, targets)
            if _paints_path(camera, targets):
                return _centre(camera.frame_center)
            return None

        for play in record.plays:
            frames = frames_in(play.start, play.end, fps)
            drawn = not play.skipped and bool(frames)
            # A frozen frame's static picture is drawn whatever is skipped.
            if play.static and (drawn or play.held):
                found = centre_if_path(
                    record._captures[play.static_seq], list(play.static)
                )
                if found is not None:
                    return found
            if not drawn:
                continue
            captures = [play.samples[0]] if play.held else _frame_captures(play, frames)
            for capture in captures:
                found = centre_if_path(
                    capture, list(capture.moving) or capture.draw_list()
                )
                if found is not None:
                    return found
        if not record.plays and record.final is not None:
            return centre_if_path(record.final, record.final.draw_list())
    return None


def _frame_captures(play: Any, frames: Sequence[Any]) -> list[Capture]:
    """The captures the play's frames draw, each once, in order."""
    found: list[Capture] = []
    for t in frames:
        try:
            capture = play.capture_at(t)
        except LookupError:
            # A record off the render's grid: the last capture before.
            capture = play.base_at(t)
        if not found or found[-1] is not capture:
            found.append(capture)
    return found


def _paints_path(camera: Camera, targets: Sequence[Mobject]) -> bool:
    """Whether drawing ``targets`` paints a path through the frame array's
    context: a VMobject with points, unless a background image fills it (it
    draws into an array of its own).
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    for mob in camera.get_mobjects_to_display(targets):
        if camera.type_or_raise(mob) is VMobject and not mob.get_background_image():
            return True
    return False


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


def _shifted(
    camera: Camera, view: View, mob: Mobject, points: np.ndarray
) -> np.ndarray:
    """``points`` through the camera's pre-display code, less the centre
    their draw maps around, as ``(n, 2)`` scene units of the live frame.
    """
    if hasattr(camera, "reset_rotation_matrix"):
        # A ThreeDCamera projects with the matrix of its trackers as they are.
        camera.reset_rotation_matrix()
    shown = np.asarray(camera.transform_points_pre_display(mob, points), dtype=float)
    centre = view.mapping_centre if draw_kind(mob) == "path" else view.frame_center
    return shown[:, :2] - np.asarray(centre[:2])


def _box(
    camera: Camera, view: View, mob: Mobject, *, ink: bool, home: bool
) -> Box | None:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for member in mob.family_members_with_points():
        if home and ink and draw_kind(member) == "points":
            # A point's ink is whole pixels: placed there, then read in units.
            box = screen_box(view, _member_pixel_box(camera, view, member, ink=True))
            low, high = np.array(box[:2]), np.array(box[2:])
        elif home:
            shifted = _shifted(camera, view, member, member.points)
            scale = np.array(
                [
                    view.home_width / view.frame_width,
                    view.home_height / view.frame_height,
                ]
            )
            placed = shifted * scale
            low, high = placed.min(axis=0), placed.max(axis=0)
            if ink and draw_kind(member) == "path":
                pad = _half_stroke(member) * view.cairo_line_width_multiple * scale
                low, high = low - pad, high + pad
        else:
            box = _member_pixel_box(camera, view, member, ink=ink)
            low, high = np.array(box[:2]), np.array(box[2:])
        lows.append(low)
        highs.append(high)
    if not lows:
        return None
    low = np.min(lows, axis=0)
    high = np.max(highs, axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _member_pixel_box(camera: Camera, view: View, member: Mobject, *, ink: bool) -> Box:
    """One member's pixel box, widened with ``ink`` to what its draw paints
    around its points.
    """
    placed = pixel_points(camera, view, member, member.points)
    low, high = placed.min(axis=0), placed.max(axis=0)
    kind = draw_kind(member) if ink else None
    if kind == "path":
        half = _half_stroke(member) * view.cairo_line_width_multiple
        pad = np.array(
            [
                half * view.pixel_width / view.frame_width,
                half * view.pixel_height / view.frame_height,
            ]
        )
        low, high = low - pad, high + pad
    elif kind == "points":
        # Each point paints the pixel it truncates to, nudged both ways by the
        # camera's thickening.
        thickness = camera.adjusted_thickness(member.get_stroke_width())
        nudges = camera.get_thickening_nudges(thickness)
        low = np.floor(low) + float(nudges.min())
        high = np.floor(high) + float(nudges.max()) + 1
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _half_stroke(member: Mobject) -> float:
    """Half the widest stroke the member draws, in stroke-width units; zero
    when it draws none (no width, or no opacity).
    """
    widest = 0.0
    for background in (True, False):
        width = float(member.get_stroke_width(background))
        if width <= 0:
            continue
        opacities = np.asarray(
            member.get_stroke_opacities(background),
            dtype=float,
        )
        if opacities.size and np.any(opacities > 0):
            widest = max(widest, width)
    return widest / 2
