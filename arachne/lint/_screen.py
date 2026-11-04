"""The screen projection the lint reads, standing in for the shared one.

The lint reads where things are on the picture through this module alone,
whose names and signatures are the shared projection's
(``arachne.camera.projection``): :class:`View`, :func:`view_of`,
:func:`record_view`, :func:`configured_home`, :func:`draw_kind`,
:func:`pixel_points`, :func:`pixel_box`, :func:`home_box`,
:func:`screen_box` and :func:`render_mapping_centre`; and
:func:`scene_line`, an origin's line in the scene file. Until the shared
projection is in the tree, this is a plain version of it: every kind of
mobject is mapped around the frame's live centre (no frozen centre), the
home frame is the active configuration's, and an object's scene line is its
creating line. Once it is, this module re-exports it, ``scene_line`` reads
``Origin.scene_line``, and nothing else changes.
"""

from __future__ import annotations

__all__ = [
    "Box",
    "ThreeDView",
    "View",
    "configured_home",
    "draw_kind",
    "home_box",
    "pixel_box",
    "pixel_points",
    "record_view",
    "render_mapping_centre",
    "scene_line",
    "screen_box",
    "view_of",
]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from arachne._config import config

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.origins import Origin
    from arachne.timeline.record import Record

#: ``(x_min, y_min, x_max, y_max)``: in pixels, y down, or in home-frame
#: units, y up.
Box = tuple[float, float, float, float]
Centre = tuple[float, float, float]


@dataclass(frozen=True)
class ThreeDView:
    phi: float
    theta: float
    gamma: float
    focal_distance: float
    zoom: float
    exponential_projection: bool


@dataclass(frozen=True)
class View:
    """What a camera shows: its pixel size, its live frame, the centre a
    path's map subtracts, the home frame and the stroke width multiple.
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
    camera: Any,
    *,
    mapping_centre: Sequence[float] | None = None,
    home: tuple[float, float] | None = None,
) -> View:
    """The view of ``camera`` as it holds its values now."""
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    kind: Literal["flat", "moving", "three_d"] = "flat"
    if isinstance(camera, ThreeDCamera):
        kind = "three_d"
    elif isinstance(camera, MovingCamera):
        kind = "moving"
    centre = _centre(camera.frame_center)
    mapped = centre if mapping_centre is None else _centre(mapping_centre)
    if home is None:
        home = configured_home(int(config.pixel_width), int(config.pixel_height))
    return View(
        kind,
        int(camera.pixel_width),
        int(camera.pixel_height),
        float(camera.frame_width),
        float(camera.frame_height),
        centre,
        mapped,
        float(home[0]),
        float(home[1]),
        float(camera.cairo_line_width_multiple),
        None,
    )


def configured_home(pixel_width: int, pixel_height: int) -> tuple[float, float]:
    """The active configuration's frame at ``pixel_width`` by ``pixel_height``."""
    width = float(config.frame_width)
    return width, width / (pixel_width / pixel_height)


def record_view(record: Record) -> View:
    """The view of ``record``'s camera as it holds its values now."""
    return view_of(record._scene.renderer.camera)


def draw_kind(mob: Mobject) -> Literal["path", "points", "image"] | None:
    """How a camera draws ``mob`` itself: a path, a point cloud, an image,
    or nothing.
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
    camera: Any, view: View, mob: Mobject, points: np.ndarray
) -> np.ndarray:
    """``points`` of ``mob`` where the camera draws them, ``(n, 2)`` pixels."""
    if hasattr(camera, "reset_rotation_matrix"):
        camera.reset_rotation_matrix()
    shown = np.asarray(camera.transform_points_pre_display(mob, points), dtype=float)
    centre = view.mapping_centre if draw_kind(mob) == "path" else view.frame_center
    x = (shown[:, 0] - centre[0]) * view.pixel_width / view.frame_width
    y = (shown[:, 1] - centre[1]) * view.pixel_height / view.frame_height
    return np.column_stack([x + view.pixel_width / 2, view.pixel_height / 2 - y])


def pixel_box(
    camera: Any, view: View, mob: Mobject, *, ink: bool = False
) -> Box | None:
    """The pixel box of ``mob``'s family as drawn; with ``ink``, each path
    widened by half its stroke.
    """
    boxes = []
    for member in mob.family_members_with_points():
        found = pixel_points(camera, view, member, member.points)
        low, high = found.min(axis=0), found.max(axis=0)
        pad = 0.0
        if ink and draw_kind(member) == "path":
            pad = _half_stroke(member) * view.cairo_line_width_multiple
            pad *= view.pixel_width / view.frame_width
        boxes.append((low[0] - pad, low[1] - pad, high[0] + pad, high[1] + pad))
    if not boxes:
        return None
    return (
        float(min(b[0] for b in boxes)),
        float(min(b[1] for b in boxes)),
        float(max(b[2] for b in boxes)),
        float(max(b[3] for b in boxes)),
    )


def home_box(camera: Any, view: View, mob: Mobject, *, ink: bool = False) -> Box | None:
    """:func:`pixel_box` in home-frame units."""
    box = pixel_box(camera, view, mob, ink=ink)
    return None if box is None else screen_box(view, box)


def screen_box(view: View, box: Box) -> Box:
    """A pixel box in home-frame units: centred, y up."""
    unit_x = view.home_width / view.pixel_width
    unit_y = view.home_height / view.pixel_height
    x0, y0, x1, y1 = box
    return (
        (x0 - view.pixel_width / 2) * unit_x,
        (view.pixel_height / 2 - y1) * unit_y,
        (x1 - view.pixel_width / 2) * unit_x,
        (view.pixel_height / 2 - y0) * unit_y,
    )


def render_mapping_centre(record: Record) -> Centre | None:
    """The centre the render froze its path map at; ``None``: the live one."""
    return None


def scene_line(origin: Origin) -> int | None:
    """The line of the scene file that made the object."""
    source = origin.source_line
    return source.line if source is not None else None


def _half_stroke(member: Mobject) -> float:
    """Half the widest stroke the member draws, in stroke-width units."""
    widest = 0.0
    for background in (True, False):
        width = float(member.get_stroke_width(background))
        opacities = np.asarray(member.get_stroke_opacities(background), dtype=float)
        if width > 0 and opacities.size and np.any(opacities > 0):
            widest = max(widest, width)
    return widest / 2


def _centre(value: Sequence[float] | np.ndarray) -> Centre:
    x, y, z = (float(v) for v in np.asarray(value, dtype=float).reshape(3))
    return (x, y, z)
