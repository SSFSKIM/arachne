"""Extraction: what the camera would draw, as XF.

:class:`FrameStream` turns a frame, as the scene's mobjects and camera hold it
now, into an XF :class:`~.xf.Delta` against what it sent before. It reads what
the camera's own capture would read and in its order: each list of mobjects
the frame is drawn from (a play's static list, then its moving list) goes
through the camera's ``get_mobjects_to_display``, so the families, the
z-index order and a ThreeDCamera's depth sort are CE's own. Every mobject
attribute or method it reads here is one the drawing path reads, and joins
the names whose user override makes a mobject draw live
(:func:`arachne.timeline.dispatch.draw_names` seeds from this module), so on
the record path extraction runs no scene code.

A leaf's record is made once per state: while a mobject holds the node state
it was extracted at (by identity, :class:`~arachne.timeline.record.NodeState`),
its record is reused as it is, and a frame that only moves the camera sends
the view and the order alone.

Nothing here imports the core.
"""

from __future__ import annotations

__all__ = ["FrameStream", "Unable"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import revealed
from .xf import (
    FEATURE_BACKGROUND_IMAGE,
    Delta,
    PaintTable,
    Record,
    Shape,
)

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject


class Unable(Exception):
    """A frame XF 1 cannot describe exactly (the IR check counts it)."""


@dataclass(frozen=True)
class Blend:
    """A leaf drawn as a straight-path blend: CE's ``interpolate`` of the
    ``start`` and ``end`` mobjects' points and style at ``alpha``.
    """

    start: Any
    end: Any
    alpha: float


@dataclass(frozen=True)
class Reveal:
    """A leaf drawn as a reveal window of the ``start`` mobject's path: the
    part between ``lo`` and ``hi`` (``pointwise_become_partial``).
    """

    start: Any
    lo: float
    hi: float


#: What the record states of a leaf at an instant.
Term = Blend | Reveal


#: The mobject attributes extraction reads off a mobject, beside what the
#: camera's own capture reads: a user override of one draws live
#: (``dispatch.draw_methods``). Keep it in step with this module's reads.
MOBJECT_READS = (
    # paths
    "points",
    "get_fill_rgbas",
    "get_stroke_rgbas",
    "get_stroke_width",
    "get_gradient_start_and_end_points",
    "get_background_image",
    "tolerance_for_point_equality",
    "n_points_per_cubic_curve",
    "joint_type",
    "cap_style",
    "shade_in_3d",
    # blends: the style fields compared with the interpolated ends
    "fill_rgbas",
    "stroke_rgbas",
    "background_stroke_rgbas",
    "stroke_width",
    "background_stroke_width",
    # depth (a sort point only under a ThreeDCamera, whose own depth sort
    # reads get_z_index_reference_point already)
    "z_index",
    # point clouds and images
    "rgbas",
    "get_pixel_array",
    "resampling_algorithm",
    # surfaces and displays
    "submobjects",
    "resolution",
    "u_index",
    "v_index",
    "get_family",
    "camera",
)

#: CE's default tolerance for closing a subpath (XF assumes it).
_TOLERANCE = 1e-6


def _camera_kind(camera: Camera) -> str:
    from arachne.camera.camera import Camera
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.multi_camera import MultiCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    kind = type(camera)
    if kind is ThreeDCamera:
        return "three_d"
    if kind in (MovingCamera, MultiCamera):
        return "moving"
    if kind is Camera:
        return "flat"
    raise Unable(f"render.unable: a {kind.__name__} draws what XF 1 does not describe")


def _floats(value: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(value, dtype=float).reshape(-1))


def _paint(rows: Any, uploads: dict[bytes, Any]) -> Any:
    """A paint field from CE's RGBA rows: ``None`` for none, the rows inline
    for one or two stops, a table key for more.
    """
    rows = np.asarray(rows, dtype=float).reshape(-1, 4)
    if len(rows) == 0:
        return None
    if len(rows) <= 2:
        rows = rows.copy()
        rows.flags.writeable = False
        return rows
    table = PaintTable(rows)
    uploads[table.key] = table
    return table.key


def _path_shape(points: np.ndarray, uploads: dict[bytes, Any]) -> bytes:
    shape = Shape("path", points)
    uploads[shape.key] = shape
    return shape.key


def _joint(vmob: Any) -> str:
    return str(vmob.joint_type.name).lower()


def _cap(vmob: Any) -> str:
    return str(vmob.cap_style.name).lower()


@dataclass
class _Cached:
    """A leaf's record as last extracted: the node state it was made from,
    the shapes and tables it refers to, its depth fields and the record.
    """

    state: Any
    base: dict[str, Any]
    kind: str
    uploads: dict[bytes, Any]
    extras: tuple[Any, ...] | None = None
    record: Record | None = None


class FrameStream:
    """Turns frames into XF deltas for one drawer, keeping what it sent.

    Make one per drawing process (after any fork) and feed it the frames in
    the order they are drawn.
    """

    def __init__(self) -> None:
        self._slots: dict[int, tuple[Any, int]] = {}
        self._cache: dict[int, _Cached] = {}
        self._held: dict[int, Record] = {}
        self._held_keys: set[bytes] = set()
        self._view: dict[str, Any] | None = None
        self._order: list[tuple[str, int, int]] | None = None
        self._seq = 0
        #: The blend records of the frame being extracted, by leaf id.
        self._blends: dict[int, Record] = {}
        #: Each display of the last frame's MultiCamera, with the sub-frame of
        #: its last capture: what CE leaves in the display's camera.
        self.last_sub_frames: list[tuple[Any, Any]] = []
        #: Frames whose view maps paths around another centre than the
        #: projection's (CE's cached-context offset).
        self.mapping_offset_frames = 0
        self.frames = 0

    # Slots

    def _slot(self, mob: Any, objects: list[tuple[int, str, str]]) -> int:
        found = self._slots.get(id(mob))
        if found is not None and found[0] is mob:
            return found[1]
        slot = len(self._slots)
        self._slots[id(mob)] = (mob, slot)
        from arachne.timeline.origins import origin_of

        origin = origin_of(mob)
        line = origin.source_line
        objects.append((slot, origin.id, "" if line is None else str(line)))
        return slot

    # Frames

    def frame(
        self,
        camera: Camera,
        groups: Sequence[tuple[str, Sequence[Mobject]]],
        *,
        t: Fraction = Fraction(0),
        terms: Mapping[int, Term] | None = None,
    ) -> Delta:
        """The delta of the frame the camera draws from ``groups`` (each a
        layer and the list one capture of the camera is handed, in order), as
        the mobjects and the camera hold it now. ``terms`` gives, by mobject
        id, the leaves a straight-path blend states.
        """
        kind = _camera_kind(camera)
        for attr, default in (
            ("image_mode", "RGBA"),
            ("n_channels", 4),
            ("pixel_array_dtype", "uint8"),
        ):
            if getattr(camera, attr) != default:
                raise Unable(f"render.unable: the camera's {attr} is not {default}")
        self._blends = {}
        objects: list[tuple[int, str, str]] = []
        uploads: dict[bytes, Any] = {}
        records: dict[int, Record] = {}
        order: list[tuple[str, int, int]] = []
        view = self._view_of(camera, kind, uploads)
        displays: list[Any] = list(vars(camera).get("image_mobjects_from_cameras", ()))
        sub_frames: dict[int, Any] = {}
        for layer, mobjects in groups:
            if displays:
                camera.update_sub_cameras()  # type: ignore[attr-defined]
            leaves = self._leaves(camera, mobjects)
            sub_frames = {}
            if displays:
                sub_frames = self._sub_frames(
                    camera, displays, mobjects, layer, objects, uploads, records, terms
                )
            self._place(
                camera,
                mobjects,
                leaves,
                layer,
                objects,
                uploads,
                records,
                order,
                terms,
                sub_frames,
            )
        # Each capture resets every display's camera before drawing into it.
        self.last_sub_frames = [(d, sub_frames[id(d)]) for d in displays if sub_frames]
        return self._delta(view, records, order, uploads, objects, t)

    def _leaves(self, camera: Camera, mobjects: Sequence[Mobject]) -> list[Mobject]:
        reset = getattr(camera, "reset_rotation_matrix", None)
        if reset is not None:
            reset()
        return camera.get_mobjects_to_display(list(mobjects))

    def _sub_frames(
        self,
        camera: Any,
        displays: list[Any],
        mobjects: Sequence[Mobject],
        layer: str,
        objects: list[tuple[int, str, str]],
        uploads: dict[bytes, Any],
        records: dict[int, Record],
        terms: Mapping[int, Term] | None,
    ) -> dict[int, tuple[int, dict[str, Any], list[tuple[str, int, int]]]]:
        """Each display's picture in this capture, as CE's MultiCamera makes
        it: its camera handed the capture's list (less the display itself,
        unless cameras may capture their own), in the camera's display order.
        """
        from arachne.utils.iterables import list_difference_update

        found: dict[int, tuple[int, dict[str, Any], list[tuple[str, int, int]]]] = {}
        plans = []
        for rank, display in enumerate(displays):
            sub_camera = display.camera
            to_add = list(mobjects)
            if not camera.allow_cameras_to_capture_their_own_display:
                to_add = list_difference_update(to_add, display.get_family())
            sub_leaves = self._leaves(sub_camera, to_add)
            sub_order: list[tuple[str, int, int]] = []
            found[id(display)] = (
                rank,
                self._view_of(sub_camera, "moving", uploads),
                sub_order,
            )
            plans.append((sub_camera, to_add, sub_leaves, sub_order))
        # Every display's sub-frame is known before any is filled: a display
        # can show another (or, if allowed, itself).
        for sub_camera, to_add, sub_leaves, sub_order in plans:
            self._place(
                sub_camera,
                to_add,
                sub_leaves,
                layer,
                objects,
                uploads,
                records,
                sub_order,
                terms,
                found,
            )
        return found

    def _place(
        self,
        camera: Any,
        mobjects: Sequence[Mobject],
        leaves: list[Mobject],
        layer: str,
        objects: list[tuple[int, str, str]],
        uploads: dict[bytes, Any],
        records: dict[int, Record],
        order: list[tuple[str, int, int]],
        terms: Mapping[int, Term] | None,
        sub_frames: Mapping[int, Any],
    ) -> None:
        from arachne.mobject.types.image_mobject import (
            AbstractImageMobject,
            ImageMobjectFromCamera,
        )
        from arachne.mobject.types.point_cloud_mobject import PMobject
        from arachne.mobject.types.vectorized_mobject import VMobject

        faces = self._faces(mobjects, leaves)
        for leaf in leaves:
            kind = camera.type_or_raise(leaf)
            if kind is VMobject:
                face = faces.get(id(leaf))
                if face is not None:
                    surface, index = face
                    slot = self._slot(surface, objects)
                    if slot not in records:
                        records[slot] = self._face_group(camera, surface, uploads)
                    order.append((layer, slot, index))
                    continue
                slot = self._slot(leaf, objects)
                term = terms.get(id(leaf)) if terms else None
                record = None
                if term is not None:
                    # One term record a frame, wherever the leaf is drawn.
                    record = self._blends.get(id(leaf))
                    if record is None:
                        if isinstance(term, Reveal):
                            record = self._reveal_record(camera, leaf, term, uploads)
                        else:
                            record = self._blend_record(camera, leaf, term, uploads)
                        if record is not None:
                            self._blends[id(leaf)] = record
                if record is None:
                    record = self._leaf_record(camera, leaf, "path", uploads)
            elif kind is PMobject:
                slot = self._slot(leaf, objects)
                record = self._leaf_record(camera, leaf, "points", uploads)
            elif kind is AbstractImageMobject:
                slot = self._slot(leaf, objects)
                if isinstance(leaf, ImageMobjectFromCamera) and id(leaf) in sub_frames:
                    record = records.get(slot)
                    if (
                        record is None
                        or record.fields.get("sub_frame") is not sub_frames[id(leaf)]
                    ):
                        record = self._display_record(
                            camera, leaf, sub_frames[id(leaf)]
                        )
                else:
                    if isinstance(leaf, ImageMobjectFromCamera):
                        # Its pixels are its camera's, which its state does
                        # not hold: never reused.
                        self._cache.pop(id(leaf), None)
                    record = self._leaf_record(camera, leaf, "image", uploads)
            else:
                continue  # CE draws nothing for a plain Mobject
            held = records.get(slot)
            if held is not None and held is not record:
                raise Unable(
                    f"render.unable: a {type(leaf).__name__} drawn two ways in one"
                    " frame (a display in two captures of a frame)"
                )
            records[slot] = record
            order.append((layer, slot, 0))

    # Records

    def _leaf_record(
        self, camera: Any, leaf: Any, kind: str, uploads: dict[bytes, Any]
    ) -> Record:
        """``leaf``'s record, reused while it holds the state it was made at."""
        from arachne.timeline.record import NodeState

        cached = self._cache.get(id(leaf))
        if cached is None or cached.kind != kind or not _holds(cached.state, leaf):
            state = NodeState.of(leaf)
            own: dict[bytes, Any] = {}
            if kind == "path":
                base = self._path_fields(camera, leaf, own)
            elif kind == "points":
                base = self._points_fields(leaf, own)
            else:
                base = self._image_fields(leaf, own)
            cached = _Cached(state, base, kind, own)
            self._cache[id(leaf)] = cached
        extras = self._depth(camera, leaf)
        if cached.record is None or cached.extras != extras:
            fields = dict(cached.base)
            fields.update(_depth_fields(extras))
            cached.extras = extras
            cached.record = Record(kind, fields)
        uploads.update(cached.uploads)
        return cached.record

    def _path_fields(
        self, camera: Any, vmob: Any, uploads: dict[bytes, Any]
    ) -> dict[str, Any]:
        if vmob.tolerance_for_point_equality != _TOLERANCE:
            raise Unable("render.unable: a path closed by another point tolerance")
        if vmob.n_points_per_cubic_curve != 4:
            raise Unable("render.unable: a path of other than cubic curves")
        fields: dict[str, Any] = {"shape": _path_shape(vmob.points, uploads)}
        self._style(vmob, fields, uploads)
        image = vmob.get_background_image()
        if image:
            displayer = camera.get_background_colored_vmobject_displayer()
            array = np.asarray(displayer.get_background_array(image))
            shape = Shape("image", array)
            uploads[shape.key] = shape
            fields["fill_image"] = shape.key
            fields["features"] = FEATURE_BACKGROUND_IMAGE
        return fields

    def _style(
        self, vmob: Any, fields: dict[str, Any], uploads: dict[bytes, Any]
    ) -> None:
        fills = vmob.get_fill_rgbas()
        strokes = vmob.get_stroke_rgbas()
        backs = vmob.get_stroke_rgbas(True)
        fields["fill"] = _paint(fills, uploads)
        fields["stroke"] = _paint(strokes, uploads)
        fields["background_stroke"] = _paint(backs, uploads)
        fields["stroke_width"] = float(vmob.get_stroke_width())
        fields["background_stroke_width"] = float(vmob.get_stroke_width(True))
        fields["joint"] = _joint(vmob)
        fields["cap"] = _cap(vmob)
        # The camera reads the gradient's ends wherever a paint has two stops
        # or more, and a shading camera gives a 3D mobject's paints two.
        shaded = bool(getattr(vmob, "shade_in_3d", False))
        if shaded or max(len(fills), len(strokes), len(backs)) > 1:
            start, end = vmob.get_gradient_start_and_end_points()
            fields["gradient"] = _floats(start) + _floats(end)

    def _points_fields(self, pmob: Any, uploads: dict[bytes, Any]) -> dict[str, Any]:
        table = PaintTable(pmob.rgbas)
        uploads[table.key] = table
        return {
            "shape": _path_shape(pmob.points, uploads),
            "colors": table.key,
            "radius": float(pmob.stroke_width),
        }

    def _image_fields(self, image: Any, uploads: dict[bytes, Any]) -> dict[str, Any]:
        pixels = np.asarray(image.get_pixel_array())
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise Unable("render.unable: an image that is not u8 RGBA")
        shape = Shape("image", pixels)
        uploads[shape.key] = shape
        return {
            "shape": shape.key,
            "corners": _floats(image.points),
            "resampling": int(image.resampling_algorithm),
        }

    def _display_record(
        self, camera: Any, display: Any, sub_frame: tuple[int, dict[str, Any], Any]
    ) -> Record:
        fields: dict[str, Any] = {
            "corners": _floats(display.points),
            "resampling": int(display.resampling_algorithm),
            "sub_frame": sub_frame,
        }
        fields.update(_depth_fields(self._depth(camera, display)))
        return Record("image", fields)

    def _blend_record(
        self, camera: Any, vmob: Any, blend: Blend, uploads: dict[bytes, Any]
    ) -> Record | None:
        """``vmob`` as a straight-path blend of its start and end, or ``None``
        when the blend does not give its points and style bit for bit.
        """
        from arachne.mobject.values import Paint
        from arachne.utils.bezier import interpolate

        start, end, alpha = blend.start, blend.end, blend.alpha
        if not all(_plain_path(m) for m in (vmob, start, end)):
            return None
        for m in (start, end):
            for width in ("stroke_width", "background_stroke_width"):
                # CE clamps a negative width where it reads it, after the
                # interpolation: the blend would mix the unclamped values.
                if not _non_negative(getattr(m, width)):
                    return None
        a, b = np.asarray(start.points), np.asarray(end.points)
        if a.shape != b.shape or not np.array_equal(
            interpolate(a, b, alpha), np.asarray(vmob.points)
        ):
            return None
        for attr in Paint.FIELDS:
            if attr in ("sheen_direction", "sheen_factor"):
                continue
            mixed = (
                getattr(end, attr)
                if alpha == 1.0
                else interpolate(getattr(start, attr), getattr(end, attr), alpha)
            )
            if not np.array_equal(np.asarray(mixed), np.asarray(getattr(vmob, attr))):
                return None
        fields: dict[str, Any] = {
            "shape": _path_shape(a, uploads),
            "blend": (_path_shape(b, uploads), _IDENTITY, float(alpha)),
        }
        own: dict[bytes, Any] = {}
        self._style(start, fields, own)
        fields.pop("gradient", None)
        paint_b: dict[str, Any] = {}
        self._style(end, paint_b, own)
        paint_b.pop("gradient", None)
        if (
            paint_b.pop("joint") != fields["joint"]
            or paint_b.pop("cap") != fields["cap"]
            or fields["joint"] != _joint(vmob)
            or fields["cap"] != _cap(vmob)
        ):
            return None
        uploads.update(own)
        fields["paint_b"] = paint_b
        fields["mix"] = float(alpha)
        if max(_stops(fields), _stops(paint_b)) > 1:
            gs, ge = vmob.get_gradient_start_and_end_points()
            fields["gradient"] = _floats(gs) + _floats(ge)
        fields.update(_depth_fields(self._depth(camera, vmob)))
        return Record("path", fields)

    def _reveal_record(
        self, camera: Any, vmob: Any, reveal: Reveal, uploads: dict[bytes, Any]
    ) -> Record | None:
        """``vmob`` as a reveal window of its start's path, or ``None`` when
        the window does not give its points bit for bit.
        """
        if not _plain_path(vmob) or not _plain_path(reveal.start):
            return None
        a = np.asarray(reveal.start.points)
        if not np.array_equal(
            revealed(a, reveal.lo, reveal.hi), np.asarray(vmob.points)
        ):
            return None
        own: dict[bytes, Any] = {}
        fields: dict[str, Any] = {
            "shape": _path_shape(a, own),
            "reveal": (float(reveal.lo), float(reveal.hi)),
        }
        self._style(vmob, fields, own)
        fields.update(_depth_fields(self._depth(camera, vmob)))
        uploads.update(own)
        return Record("path", fields)

    def _faces(
        self, mobjects: Sequence[Mobject], leaves: list[Mobject]
    ) -> dict[int, tuple[Any, int]]:
        """The leaves that are faces of a surface drawn as one face group,
        each with its surface and its index there.
        """
        if not any(hasattr(leaf, "u_index") for leaf in leaves):
            return {}
        from arachne.mobject.three_d.three_dimensions import Surface
        from arachne.utils.family import extract_mobject_family_members

        found: dict[int, tuple[Any, int]] = {}
        for owner in extract_mobject_family_members(list(mobjects)):
            if not isinstance(owner, Surface) or id(owner) in found:
                continue
            faces = list(owner.submobjects)
            if _one_group(faces):
                for index, face in enumerate(faces):
                    found[id(face)] = (owner, index)
        return found

    def _face_group(
        self, camera: Any, surface: Any, uploads: dict[bytes, Any]
    ) -> Record:
        from arachne.timeline.record import NodeState

        faces = list(surface.submobjects)
        cached = self._cache.get(id(surface))
        if (
            cached is None
            or cached.kind != "face_group"
            or len(cached.state) != len(faces) + 1
            or not _holds(cached.state[0], surface)
            or not all(
                _holds(state, face)
                for state, face in zip(cached.state[1:], faces, strict=True)
            )
        ):
            states = (NodeState.of(surface), *(NodeState.of(face) for face in faces))
            own: dict[bytes, Any] = {}
            first = faces[0]
            fields: dict[str, Any] = {
                "shape": _path_shape(
                    np.concatenate([np.asarray(face.points) for face in faces]), own
                )
            }
            self._style(first, fields, own)
            fills = PaintTable(
                np.concatenate([face.get_fill_rgbas() for face in faces])
            )
            strokes = PaintTable(
                np.concatenate([face.get_stroke_rgbas() for face in faces])
            )
            own[fills.key] = fills
            own[strokes.key] = strokes
            fields["fill"] = None
            fields["stroke"] = None
            fields["face_fill"] = fills.key
            fields["face_stroke"] = strokes.key
            fields["uv"] = np.array(
                [[face.u_index, face.v_index] for face in faces], dtype=np.uint32
            )
            resolution = surface.resolution
            u_res, v_res = (
                (resolution, resolution) if isinstance(resolution, int) else resolution
            )
            fields["grid"] = (int(u_res), int(v_res))
            fields.update(_depth_fields(self._depth(camera, first)))
            cached = _Cached(states, fields, "face_group", own)
            cached.record = Record("face_group", fields)
            self._cache[id(surface)] = cached
        uploads.update(cached.uploads)
        assert cached.record is not None
        return cached.record

    # Depth and view

    def _depth(self, camera: Any, leaf: Any) -> tuple[Any, ...]:
        """The depth fields of ``leaf`` under ``camera``: space, z-index,
        sort point, ``shade_in_3d`` and the fixed-orientation centre.
        """
        shaded = bool(getattr(leaf, "shade_in_3d", False))
        z_index = float(leaf.z_index)
        fixed_in_frame = getattr(camera, "fixed_in_frame_mobjects", None)
        if fixed_in_frame is None:
            return ("screen", z_index, None, shaded, None)
        if leaf in fixed_in_frame:
            return ("overlay", z_index, None, shaded, None)
        centre = None
        orientation = camera.fixed_orientation_mobjects
        if leaf in orientation:
            centre = _floats(orientation[leaf]())
        sort_point = _floats(leaf.get_z_index_reference_point()) if shaded else None
        return ("world", z_index, sort_point, shaded, centre)

    def _view_of(
        self,
        camera: Any,
        kind: str,
        uploads: dict[bytes, Any],
        *,
        background: np.ndarray | None = None,
        freeze: bool = True,
    ) -> dict[str, Any]:
        """The view of ``camera``, over its background or ``background``.
        With ``freeze``, the centre CE's context maps around is frozen now if
        no draw has frozen it yet; without, an array no draw has frozen maps
        around the frame's centre (nothing drawn there reads it).
        """
        if background is None:
            background = np.asarray(camera.background)
        shape = Shape("image", background)
        uploads[shape.key] = shape
        array = camera.pixel_array
        centres = camera.pixel_array_to_mapping_centre
        if freeze and camera.get_cached_cairo_context(array) is None:
            # CE freezes the centre into the context at its first draw into
            # an array (a MovingCamera at every draw): so does this one.
            camera.get_cairo_context(array)
        mapping = _floats(centres.get(id(array), camera.frame_center))
        view: dict[str, Any] = {
            "pixel_width": int(camera.pixel_width),
            "pixel_height": int(camera.pixel_height),
            "frame_width": float(camera.frame_width),
            "frame_height": float(camera.frame_height),
            "frame_center": _floats(camera.frame_center),
            "mapping_centre": mapping,
            "background": shape.key,
            "camera": kind,
            "cairo_line_width_multiple": float(camera.cairo_line_width_multiple),
            "use_z_index": bool(camera.use_z_index),
        }
        if kind == "three_d":
            view.update(
                phi=float(camera.get_phi()),
                theta=float(camera.get_theta()),
                gamma=float(camera.get_gamma()),
                focal_distance=float(camera.get_focal_distance()),
                zoom=float(camera.get_zoom()),
                exponential_projection=bool(camera.exponential_projection),
                light_source=_floats(camera.light_source.points[0]),
                shading_factor=float(camera.shading_factor),
                should_apply_shading=bool(camera.should_apply_shading),
            )
        return view

    # The delta

    def _delta(
        self,
        view: dict[str, Any],
        records: dict[int, Record],
        order: list[tuple[str, int, int]],
        uploads: dict[bytes, Any],
        objects: list[tuple[int, str, str]],
        t: Fraction,
    ) -> Delta:
        self.frames += 1
        if view["mapping_centre"] != view["frame_center"]:
            self.mapping_offset_frames += 1
        delta = Delta(t=t, seq=self._seq, objects=objects)
        self._seq += 1
        if view != self._view:
            delta.view = view
            self._view = view
        used: set[bytes] = {view["background"]}
        for record in records.values():
            used.update(record.keys)
        for key in used:
            if key not in self._held_keys:
                item = uploads[key]
                if isinstance(item, Shape):
                    delta.upload.append(item)
                else:
                    delta.upload_paints.append(item)
        for slot, record in records.items():
            if self._held.get(slot) is not record:
                delta.set.append((slot, record))
        delta.remove = [slot for slot in self._held if slot not in records]
        delta.release = [key for key in self._held_keys if key not in used]
        self._held = records
        self._held_keys = used
        if order != self._order:
            delta.order = order
            self._order = order
        return delta


def _holds(state: Any, mob: Any) -> bool:
    """Whether ``mob`` holds the node state ``state``: its values the same
    by identity, but for the points its placement computes on first reading
    (extraction reads them, which keeps them in the mobject's dictionary).
    """
    own = vars(mob)
    if state.matches(own):
        return True
    if "_points" in own and "_placement" in own and "_points" not in state.keys:
        bare = dict(own)
        del bare["_points"]
        return bool(state.matches(bare))
    return False


_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def _stops(fields: Mapping[str, Any]) -> int:
    most = 0
    for name in ("fill", "stroke", "background_stroke"):
        value = fields.get(name)
        if isinstance(value, bytes):
            most = max(most, 3)
        elif value is not None:
            most = max(most, len(value))
    return most


def _plain_path(vmob: Any) -> bool:
    """Whether ``vmob`` is a path a term's geometry can state: cubic curves
    closed at CE's tolerance, no background-image fill, unshaded.
    """
    return (
        vmob.tolerance_for_point_equality == _TOLERANCE
        and vmob.n_points_per_cubic_curve == 4
        and not vmob.get_background_image()
        and not bool(getattr(vmob, "shade_in_3d", False))
    )


def _non_negative(width: Any) -> bool:
    """Whether a stroke width is a number of at least 0."""
    try:
        return float(width) >= 0
    except (TypeError, ValueError):
        return False


def _depth_fields(extras: tuple[Any, ...]) -> dict[str, Any]:
    space, z_index, sort_point, shaded, centre = extras
    fields: dict[str, Any] = {
        "space": space,
        "z_index": z_index,
        "shade_in_3d": shaded,
    }
    if sort_point is not None:
        fields["sort_point"] = sort_point
    if centre is not None:
        fields["fixed_orientation_centre"] = centre
    return fields


def _one_group(faces: list[Any]) -> bool:
    """Whether ``faces`` draw as one face group: uv faces of one kind and
    point count, one style but their fill and stroke colours, each one stop.
    """
    from arachne.mobject.three_d.three_dimensions import ThreeDVMobject

    if not faces:
        return False
    first = faces[0]
    count = len(first.points)
    reference = None
    for face in faces:
        if type(face) is not ThreeDVMobject or not hasattr(face, "u_index"):
            return False
        if len(face.points) != count or count == 0:
            return False
        if len(face.get_fill_rgbas()) != 1 or len(face.get_stroke_rgbas()) != 1:
            return False
        if face.get_background_image() or "get_gradient_start_and_end_points" in vars(
            face
        ):
            return False
        style = (
            face.get_stroke_width(),
            face.get_stroke_width(True),
            face.joint_type,
            face.cap_style,
            bool(getattr(face, "shade_in_3d", False)),
            face.z_index,
            face.tolerance_for_point_equality,
            _floats(face.get_stroke_rgbas(True)),
        )
        if reference is None:
            reference = style
        elif style != reference:
            return False
    return True
