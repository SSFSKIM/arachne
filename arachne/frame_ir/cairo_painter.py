"""Cairo drawing from XF alone.

:class:`CairoDrawer` draws a frame described by an XF stream
(:class:`~.xf.StreamState`) with CE's own camera code: it builds light
stand-ins for the records (proxies exposing what the camera reads off a
mobject: points, paints, widths, joint and cap, gradient ends, a background
image, ``shade_in_3d``; an image's pixels and corners; a point cloud's points,
colours and width), gives a camera of the view's kind the view's values, and
hands the proxies, in the order's sequence, to the camera's display functions
as its capture would. The bytes are then the direct path's by construction: no
Cairo call is written again here. Blends and reveal windows are materialized
with Arachne's own ``interpolate`` and ``pointwise_become_partial`` on the
stored shapes (:mod:`.geometry`), as the forward run computed them.
"""

from __future__ import annotations

__all__ = ["CairoDrawer", "Unsupported"]

import itertools as it
from collections.abc import Sequence
from typing import Any

import numpy as np

from arachne.camera.camera import Camera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.constants import CapStyleType, LineJointType
from arachne.mobject.types.image_mobject import AbstractImageMobject
from arachne.mobject.types.point_cloud_mobject import PMobject
from arachne.mobject.types.vectorized_mobject import VMobject

from .geometry import blended, revealed
from .xf import FEATURE_BACKGROUND_IMAGE, FEATURE_PRE_PROJECTED, Record, StreamState

#: The record kinds and feature bits Cairo draws.
_DRAWN_KINDS = frozenset({"path", "face_group", "points", "image"})
_DRAWN_FEATURES = FEATURE_BACKGROUND_IMAGE | FEATURE_PRE_PROJECTED


class Unsupported(Exception):
    """``render.unsupported``: a frame's description holds a record kind, a
    feature bit or a field value the drawer does not draw. The frame is
    refused whole.
    """

    code = "render.unsupported"

    def __init__(
        self, kind: str, slot: int, features: int = 0, field: str | None = None
    ) -> None:
        what = f" with feature bits {features:#x}" if features else ""
        if field is not None:
            what += f" with {field}"
        super().__init__(
            f"{self.code}: slot {slot} is a {kind} record{what}"
            " this drawer does not draw; the frame is not drawn"
        )
        self.kind = kind
        self.slot = slot
        self.features = features
        self.field = field


#: A placement that leaves a shape where it is.
_IDENTITY = np.eye(3, 4)


def _unsupported_field(fields: dict[str, Any]) -> str | None:
    """The first field of a record whose value this drawer does not draw:
    a placement other than the identity, a dash, a clip, an opacity other
    than 1 (CE sets none of them).
    """
    place = fields.get("place")
    if place is not None and not np.array_equal(
        np.reshape(np.asarray(place, dtype=float), (3, 4)), _IDENTITY
    ):
        return "a placement"
    blend = fields.get("blend")
    if blend is not None and not np.array_equal(
        np.reshape(np.asarray(blend[1], dtype=float), (3, 4)), _IDENTITY
    ):
        return "a blend placement"
    for name in ("dash", "clip"):
        if fields.get(name) is not None:
            return f"a {name}"
    for name in ("opacity", "image_opacity"):
        if float(fields.get(name, 1.0)) != 1.0:
            return f"an {name.replace('_', ' ')} other than 1"
    return None


_EMPTY = np.zeros((0, 4))
_EMPTY.flags.writeable = False


def _frozen(array: Any) -> np.ndarray:
    array = np.array(array, dtype=float)
    array.flags.writeable = False
    return array


def _rows(value: Any, state: StreamState) -> np.ndarray:
    rows = state.paint(value)
    return _EMPTY if rows is None else rows


class _PathProxy(VMobject):
    """What the camera reads off a vectorized mobject, from a record."""

    def get_fill_rgbas(self) -> Any:
        return self._xf_fill

    def get_stroke_rgbas(self, background: bool = False) -> Any:
        return self._xf_background if background else self._xf_stroke

    def get_stroke_width(self, background: bool = False) -> float:
        own = self.__dict__
        return float(own["_xf_background_width"] if background else own["_xf_width"])

    def get_background_image(self) -> Any:
        return self._xf_image

    def get_gradient_start_and_end_points(self) -> Any:
        ends = self._xf_gradient
        if ends is None:
            return super().get_gradient_start_and_end_points()
        return ends


class _PointsProxy(PMobject):
    """What the camera reads off a point cloud, from a record."""


class _ImageProxy(AbstractImageMobject):
    """What the camera reads off an image, from a record: its pixels are a
    shape's, or a display's picture as it stands when drawn.
    """

    def get_pixel_array(self) -> Any:
        source = self._xf_pixels
        return source() if callable(source) else source


def _bare(cls: type, **attributes: Any) -> Any:
    proxy: Any = object.__new__(cls)
    proxy.__dict__.update(attributes)
    return proxy


def _pre_of(mobject: Any) -> dict[str, Any] | None:
    """What a pre-projected proxy's record says its camera produced."""
    return getattr(mobject, "__dict__", {}).get("_xf_pre")


class _Produced:
    """A painter camera's part for pre-projected records: their points are
    placed where the record says the camera that drew them placed them, and
    painted in the colours it says that camera painted with; everything else
    as the engine's camera does.
    """

    def transform_points_pre_display(self, mobject: Any, points: Any) -> Any:
        pre = _pre_of(mobject)
        if pre is None:
            return super().transform_points_pre_display(mobject, points)  # type: ignore[misc]
        own = mobject.__dict__
        if points is own.get("_points"):
            return pre["points"]
        if pre["gradient"] is not None and points is own.get("_xf_gradient"):
            return pre["gradient"]
        raise AssertionError("a pre-projected record holds no such points")

    def get_fill_rgbas(self, vmobject: Any) -> Any:
        pre = _pre_of(vmobject)
        if pre is None:
            return super().get_fill_rgbas(vmobject)  # type: ignore[misc]
        return pre["fill"]

    def get_stroke_rgbas(self, vmobject: Any, background: bool = False) -> Any:
        pre = _pre_of(vmobject)
        if pre is None:
            return super().get_stroke_rgbas(vmobject, background=background)  # type: ignore[misc]
        return pre["background_stroke" if background else "stroke"]


class _PainterCamera(_Produced, Camera):
    """A flat camera mapping frame to pixels around the view's mapping centre."""

    _xf_centre: tuple[float, ...] = (0.0, 0.0, 0.0)

    def get_cairo_context(self, pixel_array: Any) -> Any:
        return _context(self, pixel_array)


class _Painter3DCamera(_Produced, ThreeDCamera):
    """A ThreeDCamera mapping frame to pixels around the view's mapping centre."""

    _xf_centre: tuple[float, ...] = (0.0, 0.0, 0.0)

    def get_cairo_context(self, pixel_array: Any) -> Any:
        return _context(self, pixel_array)


#: What a 3D painter camera calls to turn its view into a rotation matrix.
_ROTATION = (
    "reset_rotation_matrix",
    "generate_rotation_matrix",
    "get_rotation_matrix",
    "get_phi",
    "get_theta",
    "get_gamma",
    "get_focal_distance",
    "get_zoom",
)


def _pin_engine_methods(cls: type, base: type, names: Sequence[str]) -> None:
    """Give ``cls`` ``base``'s definitions of ``names`` as the engine made
    them, taken when ``arachne`` was imported, so that a patch the scene's
    code made to an engine camera class never runs inside the drawer: the
    scene's camera code runs in its own capture alone, as often as on Cairo.
    """
    from arachne.timeline.frozen import class_members, engine_class

    for name in names:
        for klass in base.__mro__:
            if engine_class(klass) and name in class_members(klass):
                setattr(cls, name, class_members(klass)[name])
                break
        else:  # pragma: no cover - the engine's camera defines each
            raise LookupError(f"{base.__name__} has no {name}")


_pin_engine_methods(_Painter3DCamera, ThreeDCamera, _ROTATION)


def _context(camera: Any, pixel_array: np.ndarray) -> Any:
    """CE's ``get_cairo_context``, its matrix built from the centre CE froze."""
    contexts: dict[int, Any] = camera.__dict__.setdefault("_xf_contexts", {})
    centre = camera._xf_centre
    key = id(pixel_array)
    found = contexts.get(key)
    if (
        found is not None
        and found[0] is pixel_array
        and found[1]
        == (
            centre,
            camera.pixel_width,
            camera.pixel_height,
            camera.frame_width,
            camera.frame_height,
        )
    ):
        return found[2]
    ctx = camera.make_cairo_context(pixel_array, np.array(centre))
    contexts[key] = (
        pixel_array,
        (
            centre,
            camera.pixel_width,
            camera.pixel_height,
            camera.frame_width,
            camera.frame_height,
        ),
        ctx,
    )
    return ctx


class CairoDrawer:
    """Draws XF frames with CE's camera code."""

    def __init__(self) -> None:
        self._cameras: dict[tuple[str, Any], Any] = {}
        self._arrays: dict[Any, np.ndarray] = {}
        self._proxies: dict[int, tuple[Record, Any]] = {}
        self._state: StreamState | None = None

    # Cameras and arrays

    def _camera(self, view: dict[str, Any], role: Any) -> Any:
        kind = "three_d" if view["camera"] == "three_d" else "flat"
        camera: Any = self._cameras.get((kind, role))
        if camera is None:
            cls = _Painter3DCamera if kind == "three_d" else _PainterCamera
            camera = cls(
                pixel_width=view["pixel_width"], pixel_height=view["pixel_height"]
            )
            self._cameras[(kind, role)] = camera
        camera.pixel_width = view["pixel_width"]
        camera.pixel_height = view["pixel_height"]
        camera.frame_width = view["frame_width"]
        camera.frame_height = view["frame_height"]
        camera.cairo_line_width_multiple = view["cairo_line_width_multiple"]
        camera.use_z_index = view["use_z_index"]
        camera._xf_centre = tuple(view["mapping_centre"])
        if kind == "three_d":
            camera._frame_center.points = np.array([view["frame_center"]])
            camera.phi_tracker.set_value(view["phi"])
            camera.theta_tracker.set_value(view["theta"])
            camera.gamma_tracker.set_value(view["gamma"])
            camera.focal_distance_tracker.set_value(view["focal_distance"])
            camera.zoom_tracker.set_value(view["zoom"])
            camera.exponential_projection = view["exponential_projection"]
            camera.light_source.points = np.array([view["light_source"]])
            camera.shading_factor = view["shading_factor"]
            camera.should_apply_shading = view["should_apply_shading"]
            camera.reset_rotation_matrix()
        else:
            camera.frame_center = np.array(view["frame_center"])
        return camera

    def _array(self, view: dict[str, Any], role: Any) -> np.ndarray:
        """The array ``role`` draws into, reset to the view's background."""
        state = self._state
        assert state is not None
        background = state.shapes[view["background"]].data
        array = self._arrays.get(role)
        if array is None or array.shape != background.shape:
            array = np.array(background)
            self._arrays[role] = array
        else:
            np.copyto(array, background)
        return array

    # Proxies

    def _proxy(self, slot: int, part: int, record: Record) -> Any:
        found = self._proxies.get(slot)
        if found is None or found[0] is not record:
            found = (record, self._make(slot, record))
            self._proxies[slot] = found
        proxy = found[1]
        return proxy[part] if isinstance(proxy, list) else proxy

    def _make(self, slot: int, record: Record) -> Any:
        state = self._state
        assert state is not None
        f = record.fields
        kind = record.kind
        if kind == "path":
            return self._path(state, f)
        if kind == "face_group":
            return self._faces(state, f)
        if kind == "points":
            return _bare(
                _PointsProxy,
                _points=state.shapes[f["shape"]].data,
                _rgbas=state.paints[f["colors"]].rows,
                stroke_width=f["radius"],
                submobjects=[],
                _xf_pre=_pre(f, state),
            )
        if kind == "image":
            if "sub_frame" in f:
                arrays = self._arrays

                def pixels(slot: int = slot) -> np.ndarray:
                    return arrays[("display", slot)]

                source: Any = pixels
            else:
                source = state.shapes[f["shape"]].data
            return _bare(
                _ImageProxy,
                _points=_frozen(np.reshape(f["corners"], (4, 3))),
                resampling_algorithm=f["resampling"],
                _xf_pixels=source,
                submobjects=[],
                _xf_pre=_pre(f, state),
            )
        raise NotImplementedError(f"the Cairo drawer has no {kind} records")

    def _path(self, state: StreamState, f: dict[str, Any]) -> Any:
        points = state.shapes[f["shape"]].data
        reveal = f.get("reveal")
        if reveal is not None:
            points = _frozen(revealed(points, *reveal))
        fill = _rows(f.get("fill"), state)
        stroke = _rows(f.get("stroke"), state)
        back = _rows(f.get("background_stroke"), state)
        width = f["stroke_width"]
        back_width = f["background_stroke_width"]
        blend = f.get("blend")
        if blend is not None:
            points = _frozen(blended(points, state.shapes[blend[0]].data, blend[2]))
            b = f["paint_b"]
            mix = f["mix"]
            fill, stroke, back, width, back_width = (
                _mixed(a, z, mix)
                for a, z in (
                    (fill, _rows(b.get("fill"), state)),
                    (stroke, _rows(b.get("stroke"), state)),
                    (back, _rows(b.get("background_stroke"), state)),
                    (width, b["stroke_width"]),
                    (back_width, b["background_stroke_width"]),
                )
            )
        gradient = f.get("gradient")
        image = f.get("fill_image")
        return _bare(
            _PathProxy,
            _points=points,
            n_points_per_cubic_curve=4,
            tolerance_for_point_equality=1e-6,
            shade_in_3d=f.get("shade_in_3d", False),
            joint_type=LineJointType[f["joint"].upper()],
            cap_style=CapStyleType[f["cap"].upper()],
            submobjects=[],
            _xf_fill=fill,
            _xf_stroke=stroke,
            _xf_background=back,
            _xf_width=width,
            _xf_background_width=back_width,
            _xf_image=None if image is None else "xf:" + image.hex(),
            _xf_gradient=None
            if gradient is None
            else (np.array(gradient[:3]), np.array(gradient[3:])),
            _xf_pre=_pre(f, state),
        )

    def _faces(self, state: StreamState, f: dict[str, Any]) -> list[Any]:
        points = state.shapes[f["shape"]].data
        fills = state.paints[f["face_fill"]].rows
        strokes = state.paints[f["face_stroke"]].rows
        count = len(f["uv"])
        size = len(points) // count
        back = _rows(f.get("background_stroke"), state)
        common = {
            "n_points_per_cubic_curve": 4,
            "tolerance_for_point_equality": 1e-6,
            "shade_in_3d": f.get("shade_in_3d", False),
            "joint_type": LineJointType[f["joint"].upper()],
            "cap_style": CapStyleType[f["cap"].upper()],
            "_xf_background": back,
            "_xf_width": f["stroke_width"],
            "_xf_background_width": f["background_stroke_width"],
            "_xf_image": None,
            "_xf_gradient": None,
        }
        return [
            _bare(
                _PathProxy,
                _points=points[i * size : (i + 1) * size],
                submobjects=[],
                _xf_fill=fills[i : i + 1],
                _xf_stroke=strokes[i : i + 1],
                **common,
            )
            for i in range(count)
        ]

    def proxy_of(
        self,
        record: Record,
        shapes: dict[bytes, Any],
        paints: dict[bytes, Any],
    ) -> Any:
        """The proxy the camera would read ``record`` through, its keys
        found among ``shapes`` and ``paints``.
        """
        state = StreamState()
        state.shapes = shapes
        state.paints = paints
        held = self._state
        self._state = state
        try:
            return self._make(-1, record)
        finally:
            self._state = held

    # Drawing

    def capture(
        self,
        state: StreamState,
        view: dict[str, Any],
        order: Sequence[tuple[str, int, int]],
        background: np.ndarray,
    ) -> np.ndarray:
        """A new array: ``background`` with ``order`` (entries of ``state``)
        drawn over it by a camera of ``view``.
        """
        self._state = state
        camera = self._camera(view, "capture")
        array = np.array(background)
        camera.pixel_array = array
        self._draw_entries(camera, order, array)
        return array

    def draw(self, state: StreamState) -> np.ndarray:
        """The frame ``state`` describes, as the camera's pixel array (the
        drawer's own: copy it to keep it).
        """
        self._refuse_unknown(state, state.order)
        self._state = state
        assert state.view is not None
        return self._draw_order(state, state.order, self._array(state.view, "main"))

    def draw_over(
        self,
        state: StreamState,
        frame: np.ndarray,
        entries: Sequence[tuple[str, int, int]],
    ) -> np.ndarray:
        """``frame`` (written in place, and returned) with ``entries`` of
        ``state`` drawn over it in their order, by a camera of the view, as
        CE's capture draws them: a 3D scene's overlay over the core's frame.
        """
        self._refuse_unknown(state, entries)
        self._state = state
        return self._draw_order(state, entries, frame)

    def _draw_order(
        self,
        state: StreamState,
        order: Sequence[tuple[str, int, int]],
        array: np.ndarray,
    ) -> np.ndarray:
        view = state.view
        assert view is not None
        live = set(state.records)
        for slot in [s for s in self._proxies if s not in live]:
            del self._proxies[slot]
        camera = self._camera(view, "main")
        camera.pixel_array = array
        self._seed_images(camera, state)
        self._orient(camera, state)
        for _, run in it.groupby(order, key=lambda entry: entry[0]):
            entries = list(run)
            self._draw_displays(entries)
            self._draw_entries(camera, entries, array)
        return array

    @staticmethod
    def _refuse_unknown(
        state: StreamState, order: Sequence[tuple[str, int, int]]
    ) -> None:
        """Refuse the frame if a record of ``order`` (or of a display's
        sub-frame it reaches) has a kind, a feature bit or a field value Cairo
        does not draw.
        """
        todo = [slot for _, slot, _ in order]
        seen: set[int] = set()
        while todo:
            slot = todo.pop()
            if slot in seen:
                continue
            seen.add(slot)
            record = state.records[slot]
            features = int(record.fields.get("features", 0))
            if record.kind not in _DRAWN_KINDS or features & ~_DRAWN_FEATURES:
                raise Unsupported(record.kind, slot, features & ~_DRAWN_FEATURES)
            field = _unsupported_field(record.fields)
            if field is not None:
                raise Unsupported(record.kind, slot, field=field)
            sub_frame = record.fields.get("sub_frame")
            if sub_frame is not None:
                todo.extend(entry[1] for entry in sub_frame[2])

    def _seed_images(self, camera: Any, state: StreamState) -> None:
        """Give the camera's background-image displayer the images the
        records' fills multiply by, under the names their proxies give.
        """
        names = {
            r.fields["fill_image"]
            for r in state.records.values()
            if r.fields.get("fill_image") is not None
        }
        if not names:
            return
        displayer = camera.get_background_colored_vmobject_displayer()
        if displayer.pixel_array.shape != camera.pixel_array.shape:
            camera.background_colored_vmobject_displayer = None
            displayer = camera.get_background_colored_vmobject_displayer()
        for key in names:
            displayer.file_name_to_pixel_array_map["xf:" + key.hex()] = state.shapes[
                key
            ].data

    def _orient(self, camera: Any, state: StreamState) -> None:
        """Tell a 3D camera which proxies are fixed in frame and which keep
        their orientation about a recorded centre.
        """
        if not isinstance(camera, ThreeDCamera):
            return
        fixed: set[Any] = set()
        oriented: dict[Any, Any] = {}
        for slot, part in {(s, p) for _, s, p in state.order}:
            record = state.records[slot]
            proxy = self._proxy(slot, part, record)
            space = record.fields.get("space")
            if space == "overlay":
                fixed.add(proxy)
            centre = record.fields.get("fixed_orientation_centre")
            if centre is not None:
                value = np.array(centre)
                oriented[proxy] = lambda value=value: value
        camera.fixed_in_frame_mobjects = fixed
        camera.fixed_orientation_mobjects = oriented

    def _draw_entries(
        self, camera: Any, entries: Sequence[tuple[str, int, int]], array: np.ndarray
    ) -> None:
        """Hand the entries' proxies to the camera's display functions, in
        runs of one type, as its capture does.
        """
        state = self._state
        assert state is not None
        if isinstance(camera, ThreeDCamera):
            camera.reset_rotation_matrix()
        proxies = [
            self._proxy(slot, part, state.records[slot]) for _, slot, part in entries
        ]
        for group_type, group in it.groupby(proxies, camera.type_or_raise):
            camera.display_funcs[group_type](list(group), array)

    def _draw_displays(self, entries: Sequence[tuple[str, int, int]]) -> None:
        """Draw the pictures of the displays among ``entries``, as a
        MultiCamera captures into its displays' cameras before its own
        capture: every one reset, then each in its rank.
        """
        state = self._state
        assert state is not None
        displays = []
        seen: set[int] = set()
        for _, slot, _ in entries:
            sub_frame = state.records[slot].fields.get("sub_frame")
            if sub_frame is not None and slot not in seen:
                seen.add(slot)
                displays.append((sub_frame[0], slot, sub_frame))
        if not displays:
            return
        displays.sort()
        arrays = [
            (slot, sub_frame, self._array(sub_frame[1], ("display", slot)))
            for _, slot, sub_frame in displays
        ]
        for slot, (_, sub_view, sub_order), sub_array in arrays:
            sub_camera = self._camera(sub_view, ("display", slot))
            sub_camera.pixel_array = sub_array
            self._draw_entries(sub_camera, sub_order, sub_array)


def _pre(f: dict[str, Any], state: StreamState) -> dict[str, Any] | None:
    """A pre-projected record's produced values, as the painter camera hands
    them to CE's display code: the placed points, the gradient's placed ends
    as CE's camera receives them back, and the paints' RGBA rows.
    """
    pre = f.get("pre_projected")
    if pre is None:
        return None
    gradient = pre.get("gradient")
    return {
        "points": state.shapes[pre["points"]].data,
        "gradient": None
        if gradient is None
        else np.array([gradient[:3], gradient[3:]], dtype=float),
        **{
            name: _rows(pre.get(name), state)
            for name in ("fill", "stroke", "background_stroke")
        },
    }


def _mixed(a: Any, b: Any, mix: float) -> Any:
    """``a`` and ``b`` mixed as CE's style interpolation mixes them."""
    from arachne.utils.bezier import interpolate

    if mix == 1.0:
        return b
    return interpolate(a, b, mix)
