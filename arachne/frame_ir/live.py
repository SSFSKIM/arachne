"""Live frames in XF: the values the camera's own capture consumed.

A record that refuses is drawn while the scene runs (``LiveFrames``), and a
scene drawn without its record draws through ``update_frame``. There the
scene's code runs as often as CE runs it, since an override can count its
calls or change state: so a live frame is never read a second time. The
camera's own capture runs once, each mobject it draws handed to its display
functions inside a recording proxy that passes every read through and keeps
what it returned, and the frame's records are built from those values alone.
Under a ``MappingCamera`` the recording keeps the points ``mapping_func``
mapped (CE maps the points of point clouds, never paths or images).

Each capture is one XF frame: the camera's picture before it, as the view's
background, and the leaves it drew over it, in the order it drew them. Its
records are checked against what the camera read: a read the record would
answer otherwise (a user override of a method the record computes, a value
read two ways) is ``render.unable``, never drawn wrong. Cairo then draws the
frame from XF and its picture replaces the capture's, so whatever reads the
camera next (the frame written, a static picture kept, a display's camera
read by its MultiCamera) holds the described frame.

With ``draw`` (the ``ir_cairo`` path) the capture is the real draw, which
the description is compared with (:attr:`LivePainter.differing`); without,
Cairo's calls go to a context that draws nothing (the core path's capture).
"""

from __future__ import annotations

__all__ = ["LivePainter", "live"]

import contextlib
import os
import weakref
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

import numpy as np

from .cairo_painter import CairoDrawer
from .extract import (
    FrameStream,
    Unable,
    _depth_fields,
    _floats,
    _paint,
    _path_shape,
)
from .xf import FEATURE_BACKGROUND_IMAGE, PaintTable, Record, Shape

#: Reads whose value a record states but whose replay answers in its own
#: terms: a background image is named by its key there.
_TRUTH_ONLY = frozenset({"get_background_image"})

#: Reads with a ``background`` switch, kept apart by it.
_BACKGROUND_SWITCH = frozenset({"get_stroke_rgbas", "get_stroke_width"})


class _Recording:
    """A drawn mobject as the camera's display code sees it: every read is
    the mobject's own, and is kept (:attr:`_xf_log`) with what it returned.
    """

    __slots__ = ("_xf_leaf", "_xf_log")

    def __init__(self, leaf: Any) -> None:
        self._xf_leaf = leaf
        self._xf_log: list[tuple[str, Any, Any, Any]] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._xf_leaf, name)
        log = self._xf_log
        if callable(value) and not isinstance(value, type):

            def call(*args: Any, **kwargs: Any) -> Any:
                result = value(*args, **kwargs)
                log.append((name, args, kwargs, result))
                return result

            return call
        log.append((name, None, None, value))
        return value

    # The camera keeps mobjects in sets and dicts (fixed in frame, fixed
    # orientation): the proxy is found where its mobject is.
    def __hash__(self) -> int:
        return hash(self._xf_leaf)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Recording):
            return other._xf_leaf is self._xf_leaf
        return other is self._xf_leaf


class _NoContext:
    """A Cairo context that draws nothing."""

    def __getattr__(self, name: str) -> Any:
        return _nothing


def _nothing(*args: Any, **kwargs: Any) -> None:
    return None


def _key(name: str, args: Any, kwargs: Any) -> tuple[str, Any]:
    if args is None:
        return (name, None)
    if name in _BACKGROUND_SWITCH:
        background = args[0] if args else kwargs.get("background", False)
        return (name, bool(background))
    if not args and not kwargs:
        return (name, ())
    return (name, "with arguments")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


class _Reads:
    """A recording's log, by read."""

    _MISSING = object()

    def __init__(self, recording: _Recording) -> None:
        self.leaf = recording._xf_leaf
        self.log = recording._xf_log
        self.values: dict[tuple[str, Any], Any] = {}
        for name, args, kwargs, value in self.log:
            key = _key(name, args, kwargs)
            if key[1] == "with arguments":
                continue
            held = self.values.get(key, self._MISSING)
            if held is self._MISSING:
                self.values[key] = value
            elif not _same(held, value):
                raise Unable(
                    f"render.unable: a {type(self.leaf).__name__}'s {name} read two"
                    " ways in one capture"
                )

    def get(self, name: str, switch: Any = None, default: Any = None) -> Any:
        found = self.values.get((name, switch), self._MISSING)
        return default if found is self._MISSING else found


class _LiveStream(FrameStream):
    """A frame stream whose leaves are a capture's: a slot lives while its
    leaf is drawn (a MappingCamera draws fresh copies every capture).
    """

    def keep_only(self, slots: set[int]) -> None:
        self._slots = {k: v for k, v in self._slots.items() if v[1] in slots}

    def _slot(self, mob: Any, objects: list[tuple[int, str, str]]) -> int:
        found = self._slots.get(id(mob))
        if found is not None and found[0] is mob:
            return found[1]
        slot = self._next_slot = getattr(self, "_next_slot", -1) + 1
        self._slots[id(mob)] = (mob, slot)
        from arachne.timeline.origins import origin_of

        origin = origin_of(mob)
        line = origin.source_line
        objects.append((slot, origin.id, "" if line is None else str(line)))
        return slot


def _view_kind(camera: Any) -> str:
    """The view a live capture of ``camera`` is drawn back with (a
    MappingCamera's mapping is in the points it recorded).
    """
    from arachne.camera.camera import Camera
    from arachne.camera.mapping_camera import MappingCamera
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.multi_camera import MultiCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    kind = type(camera)
    if kind is ThreeDCamera:
        return "three_d"
    if kind in (MovingCamera, MultiCamera):
        return "moving"
    if kind in (Camera, MappingCamera):
        return "flat"
    raise Unable(f"render.unable: a {kind.__name__} draws what XF 1 does not describe")


class LivePainter:
    """Live captures of one camera (and its displays' cameras) through XF.

    ``captures`` counts the captures described; with ``draw``, ``differing``
    counts those whose real draw differs from the picture drawn from XF.
    """

    def __init__(self, *, draw: bool = True) -> None:
        self.draw = draw
        self._painters: dict[int, tuple[Any, Any]] = {}
        self._checker = CairoDrawer()
        self.captures = 0
        self.differing = 0

    def painter(self, camera: Any) -> Any:
        """The painter of ``camera``'s captures."""
        found = self._painters.get(id(camera))
        if found is None or found[0] is not camera:
            from .painter import IrCairoPainter

            painter = IrCairoPainter(stream=_LiveStream())
            found = (camera, painter)
            self._painters[id(camera)] = found
        return found[1]

    @contextlib.contextmanager
    def around(self, camera: Any) -> Iterator[LivePainter]:
        """Within: ``camera``'s captures, and its displays' cameras', are
        described in XF and drawn back from it.
        """
        displays = vars(camera).get("image_mobjects_from_cameras", ())
        cameras = [camera, *(display.camera for display in displays)]
        undo: list[Any] = []
        try:
            # Installed one by one, so a failure undoes what was installed.
            undo.extend(
                self._install(each) for each in cameras if "_xf_live" not in vars(each)
            )
            yield self
        finally:
            for restore in reversed(undo):
                restore()

    def _install(self, camera: Any) -> Any:
        from arachne.camera.camera import Camera
        from arachne.camera.mapping_camera import MappingCamera
        from arachne.mobject.mobject import Mobject

        own = vars(camera)
        funcs = camera.display_funcs
        camera.display_funcs = {
            kind: func if kind is Mobject else self._display(camera, kind, func)
            for kind, func in funcs.items()
        }
        own["capture_mobjects"] = self._capture(camera, camera.capture_mobjects)
        own["_xf_live"] = []
        added = ["capture_mobjects", "_xf_live"]
        if (
            isinstance(camera, MappingCamera)
            and type(camera).points_to_pixel_coords
            is MappingCamera.points_to_pixel_coords
        ):

            def mapped(mobject: Any, points: Any) -> Any:
                points = np.apply_along_axis(camera.mapping_func, 1, points)
                if isinstance(mobject, _Recording):
                    mobject._xf_log.append(("__mapped__", None, None, points))
                return Camera.points_to_pixel_coords(camera, mobject, points)

            own["points_to_pixel_coords"] = mapped
            added.append("points_to_pixel_coords")
        if not self.draw:
            context = camera.get_cairo_context

            def no_context(pixel_array: Any) -> Any:
                context(pixel_array)  # CE's context still freezes its centre
                return _NoContext()

            own["get_cairo_context"] = no_context
            added.append("get_cairo_context")

        def restore() -> None:
            camera.display_funcs = funcs
            for name in added:
                own.pop(name, None)

        return restore

    def _display(self, camera: Any, kind: type, func: Any) -> Any:
        def display(mobjects: list[Any], pixel_array: Any) -> Any:
            drawn = [_Recording(mob) for mob in mobjects]
            frames = vars(camera)["_xf_live"]
            if frames:
                frames[-1].extend((kind, recording) for recording in drawn)
            return func(drawn, pixel_array)

        return display

    def _capture(self, camera: Any, capture: Any) -> Any:
        def described(mobjects: Any, **kwargs: Any) -> None:
            frames = vars(camera)["_xf_live"]
            background = np.array(camera.pixel_array)
            drawn: list[tuple[type, _Recording]] = []
            centres: dict[int, list[Any]] = {}
            frames.append(drawn)
            restore = self._record_centres(camera, centres)
            try:
                capture(mobjects, **kwargs)
            finally:
                restore()
                frames.pop()
            pixels = self._describe(camera, drawn, centres, background)
            self.captures += 1
            if self.draw and not np.array_equal(camera.pixel_array, pixels):
                self.differing += 1
            np.copyto(camera.pixel_array, pixels)

        return described

    @staticmethod
    def _record_centres(camera: Any, centres: dict[int, list[Any]]) -> Any:
        """Keep each fixed-orientation centre the capture asks for."""
        orientation = vars(camera).get("fixed_orientation_mobjects")
        if not orientation:
            return _nothing
        held = dict(orientation)

        def recorded(mob: Any, func: Any) -> Any:
            def centre() -> Any:
                value = func()
                centres.setdefault(id(mob), []).append(_floats(value))
                return value

            return centre

        for mob, func in held.items():
            orientation[mob] = recorded(mob, func)

        def restore() -> None:
            orientation.clear()
            orientation.update(held)

        return restore

    # The frame

    def _describe(
        self,
        camera: Any,
        drawn: list[tuple[type, _Recording]],
        centres: dict[int, list[Any]],
        background: np.ndarray,
    ) -> np.ndarray:
        kind = _view_kind(camera)
        painter = self.painter(camera)
        stream: _LiveStream = painter.stream
        uploads: dict[bytes, Any] = {}
        objects: list[tuple[int, str, str]] = []
        records: dict[int, Record] = {}
        order: list[tuple[str, int, int]] = []
        view = stream._view_of(
            camera, kind, uploads, background=background, freeze=False
        )
        held = stream._held
        for leaf_kind, recording in drawn:
            leaf = recording._xf_leaf
            record = self._record(camera, leaf_kind, recording, centres, uploads)
            slot = stream._slot(leaf, objects)
            known = held.get(slot)
            if known is not None and _same_record(known, record):
                record = known
            placed = records.get(slot)
            if placed is not None and not _same_record(placed, record):
                raise Unable(
                    f"render.unable: a {type(leaf).__name__} drawn two ways in one"
                    " capture"
                )
            records[slot] = record if placed is None else placed
            order.append(("moving", slot, 0))
        t = Fraction(stream.frames)
        delta = stream._delta(view, records, order, uploads, objects, t)
        stream.keep_only(set(records))
        return painter.draw(delta)

    def _record(
        self,
        camera: Any,
        leaf_kind: type,
        recording: _Recording,
        centres: dict[int, list[Any]],
        uploads: dict[bytes, Any],
    ) -> Record:
        from arachne.mobject.types.point_cloud_mobject import PMobject
        from arachne.mobject.types.vectorized_mobject import VMobject

        reads = _Reads(recording)
        own: dict[bytes, Any] = {}
        if leaf_kind is VMobject:
            kind, fields = "path", self._path_fields(camera, reads, own)
        elif leaf_kind is PMobject:
            kind, fields = "points", self._points_fields(reads, own)
        else:
            kind, fields = "image", self._image_fields(reads, own)
        fields.update(_depth_fields(self._depth(camera, reads, centres)))
        record = Record(kind, fields)
        self._check(record, recording, own)
        uploads.update(own)
        return record

    @staticmethod
    def _path_fields(
        camera: Any, reads: _Reads, uploads: dict[bytes, Any]
    ) -> dict[str, Any]:
        if reads.get("n_points_per_cubic_curve", default=4) != 4:
            raise Unable("render.unable: a path of other than cubic curves")
        points = np.asarray(reads.get("points", default=np.zeros((0, 3))), float)
        fields: dict[str, Any] = {"shape": _path_shape(points, uploads)}
        fields["fill"] = _paint(reads.get("get_fill_rgbas", (), _NONE), uploads)
        fields["stroke"] = _paint(reads.get("get_stroke_rgbas", False, _NONE), uploads)
        fields["background_stroke"] = _paint(
            reads.get("get_stroke_rgbas", True, _NONE), uploads
        )
        fields["stroke_width"] = float(reads.get("get_stroke_width", False, 0.0))
        fields["background_stroke_width"] = float(
            reads.get("get_stroke_width", True, 0.0)
        )
        joint = reads.get("joint_type")
        cap = reads.get("cap_style")
        fields["joint"] = "auto" if joint is None else str(joint.name).lower()
        fields["cap"] = "auto" if cap is None else str(cap.name).lower()
        gradient = reads.get("get_gradient_start_and_end_points", ())
        if gradient is not None:
            start, end = gradient
            fields["gradient"] = _floats(start) + _floats(end)
        image = reads.get("get_background_image", ())
        if image:
            displayer = camera.get_background_colored_vmobject_displayer()
            array = np.asarray(displayer.get_background_array(image))
            shape = Shape("image", array)
            uploads[shape.key] = shape
            fields["fill_image"] = shape.key
            fields["features"] = FEATURE_BACKGROUND_IMAGE
        return fields

    @staticmethod
    def _points_fields(reads: _Reads, uploads: dict[bytes, Any]) -> dict[str, Any]:
        points = reads.get("__mapped__")
        if points is None:
            points = reads.get("points", default=np.zeros((0, 3)))
        table = PaintTable(np.asarray(reads.get("rgbas", default=_NONE), float))
        uploads[table.key] = table
        return {
            "shape": _path_shape(np.asarray(points, float), uploads),
            "colors": table.key,
            "radius": float(reads.get("stroke_width", default=0.0)),
        }

    @staticmethod
    def _image_fields(reads: _Reads, uploads: dict[bytes, Any]) -> dict[str, Any]:
        pixels = np.asarray(reads.get("get_pixel_array", ()))
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise Unable("render.unable: an image that is not u8 RGBA")
        shape = Shape("image", pixels)
        uploads[shape.key] = shape
        return {
            "shape": shape.key,
            "corners": _floats(reads.get("points")),
            "resampling": int(reads.get("resampling_algorithm", default=0)),
        }

    @staticmethod
    def _depth(
        camera: Any, reads: _Reads, centres: dict[int, list[Any]]
    ) -> tuple[Any, ...]:
        """Space, z-index, ``shade_in_3d`` and the fixed-orientation centre:
        the order is the capture's, so no sort point.
        """
        leaf = reads.leaf
        own = vars(leaf)
        shaded = bool(reads.get("shade_in_3d", default=own.get("shade_in_3d", False)))
        z_index = float(own.get("z_index", 0.0))
        fixed_in_frame = vars(camera).get("fixed_in_frame_mobjects")
        if fixed_in_frame is None:
            return ("screen", z_index, None, shaded, None)
        if leaf in fixed_in_frame:
            return ("overlay", z_index, None, shaded, None)
        asked = centres.get(id(leaf), [])
        if any(value != asked[0] for value in asked):
            raise Unable(
                f"render.unable: a {type(leaf).__name__}'s fixed-orientation centre"
                " moved within one capture"
            )
        return ("world", z_index, None, shaded, asked[0] if asked else None)

    def _check(
        self, record: Record, recording: _Recording, uploads: dict[bytes, Any]
    ) -> None:
        """Every read the camera made answers the same off the record."""
        shapes = {k: v for k, v in uploads.items() if isinstance(v, Shape)}
        paints = {k: v for k, v in uploads.items() if isinstance(v, PaintTable)}
        replay = self._checker.proxy_of(record, shapes, paints)
        mapped = any(entry[0] == "__mapped__" for entry in recording._xf_log)
        for name, args, kwargs, value in recording._xf_log:
            if name == "__mapped__" or (mapped and name == "points"):
                continue  # the record holds the mapped points
            try:
                answer = getattr(replay, name)
                if args is not None:
                    answer = answer(*args, **kwargs)
            except Exception:  # noqa: BLE001 - any failure is a mismatch
                answer = _NONE
            agrees = (
                bool(answer) == bool(value)
                if name in _TRUTH_ONLY
                else answer is not _NONE and _same(answer, value)
            )
            if not agrees:
                leaf = recording._xf_leaf
                raise Unable(
                    f"render.unable: a {type(leaf).__name__}'s {name} is not what"
                    " its XF record states"
                )


#: No value (a read that was not made, or that failed on replay).
_NONE: Any = np.zeros((0, 4))


def _same_record(a: Record, b: Record) -> bool:
    if a.kind != b.kind or a.fields.keys() != b.fields.keys():
        return False
    return all(_same(a.fields[k], b.fields[k]) for k in a.fields)


_LIVE: weakref.WeakKeyDictionary[Any, tuple[int, LivePainter]] = (
    weakref.WeakKeyDictionary()
)


@contextlib.contextmanager
def live(camera: Any, draw_path: str | None = None) -> Iterator[LivePainter | None]:
    """Within: ``camera`` captures through the draw path ``draw_path`` (the
    option of that name, unless given). On ``cairo`` nothing changes.
    """
    from .painter import DRAW_PATHS

    if draw_path is None:
        from arachne import config

        draw_path = config.draw_path
    if not DRAW_PATHS[draw_path]:
        yield None
        return
    found = _LIVE.get(camera)
    pid = os.getpid()
    if found is None or found[0] != pid:
        found = (pid, LivePainter())
        _LIVE[camera] = found
    with found[1].around(camera) as painter:
        yield painter


def live_painter(camera: Any) -> LivePainter | None:
    """The live painter of ``camera`` in this process, if it has drawn."""
    found = _LIVE.get(camera)
    return None if found is None or found[0] != os.getpid() else found[1]
