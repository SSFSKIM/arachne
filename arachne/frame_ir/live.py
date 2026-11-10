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
Cairo's calls go to a context that draws nothing (the core path's capture),
but for the captures the core keeps as the camera drew them
(:meth:`LivePainter._sample`). On the core the camera's array then has one
writer within a frame, the core, checked at each capture's start and end: a
write by anything else leaves the frame layered for attribution, and falls
back to the camera's own draw where nothing of the core's is under it, or is
``render.unable``; it is never dropped (:meth:`LivePainter._sole_writer`).
A capture begun on a reset after the scene's first frame, or on a picture the
scene set, holds nothing of the core's either: it draws over what the array
holds as its pixels, as Cairo draws over it. A hedged camera's own draw of
the scene's first frame is compared once with Cairo's draw of the frame's
description before the description replaces it, so a write in it that no
producer call made is not dropped either (:meth:`LivePainter._settle_opening`).
"""

from __future__ import annotations

__all__ = ["LivePainter", "live"]

import contextlib
import os
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from arachne.mobject.values import FrozenArray

from .cairo_painter import CairoDrawer
from .canonical import canonical_order, frame_grid
from .core_painter import CorePainter
from .extract import (
    FrameStream,
    Unable,
    _depth_fields,
    _face_lists,
    _floats,
    _paint,
    _path_shape,
    _uv_faces,
    view_values,
)
from .xf import (
    FEATURE_BACKGROUND_IMAGE,
    FEATURE_PRE_PROJECTED,
    PaintTable,
    Record,
    Shape,
)

#: Reads whose value a record states but whose replay answers in its own
#: terms: a background image is named by its key there.
_TRUTH_ONLY = frozenset({"get_background_image"})

#: Reads with a ``background`` switch, kept apart by it.
_BACKGROUND_SWITCH = frozenset({"get_stroke_rgbas", "get_stroke_width"})


class _Recording:
    """A drawn mobject as the camera's display code sees it: every read is
    the mobject's own, and is kept (:attr:`_xf_log`) with what it returned.
    """

    __slots__ = ("_xf_inside", "_xf_leaf", "_xf_log")

    def __init__(self, leaf: Any) -> None:
        self._xf_leaf = leaf
        self._xf_log: list[tuple[str, Any, Any, Any]] = []
        #: The log's entries read inside a method whose result the record
        #: carries (:func:`_producing`): never asked of the record.
        self._xf_inside: set[int] = set()

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


#: The camera's own methods whose results for a drawn mobject a
#: pre-projected record states: where its pre-display code placed the
#: points (the mobject's, and its gradient's ends), and the colours it
#: painted them with (its shading included).
_PRODUCED = ("transform_points_pre_display", "get_fill_rgbas", "get_stroke_rgbas")
#: How a produced value is kept in a recording's log, by method.
_PRE_DISPLAY = "__pre_display__"
_CAMERA_RGBAS = "__camera_rgbas__"


#: The camera's methods that write a picture into the array they are handed
#: (CE's ``Camera``; no engine subclass adds one): within a capture its
#: draw, outside one a picture composited into the array, no capture's
#: (``LivePainter._compositing``). ``set_pixel_array`` (``reset``,
#: ``set_frame_to_background``) sets the array instead; it is watched apart.
_COMPOSITING = (
    "overlay_rgba_array",
    "overlay_PIL_image",
    "display_multiple_vectorized_mobjects",
    "display_multiple_non_background_colored_vmobjects",
    "display_multiple_background_colored_vmobjects",
    "display_vectorized",
    "apply_fill",
    "apply_stroke",
    "display_multiple_point_cloud_mobjects",
    "display_point_cloud",
    "display_multiple_image_mobjects",
    "display_image_mobject",
)


#: Of :data:`_COMPOSITING`, the methods CE's capture writes the camera's
#: array through by numbers rather than through its Cairo context (a point
#: cloud's points; an image, or a path filled through a background image,
#: composited), with the position of the array among their arguments.
_BY_NUMBERS = {"overlay_PIL_image": 0, "display_point_cloud": 4}


#: Cameras whose captures are their own draw from now on
#: (``render.fallback``), with why: what their code changed while producing
#: a record's values, the capture their draw first differed at, or what
#: wrote their array beside the core (``LivePainter._sole_writer``).
_UNSETTLED: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

#: Why a camera's state, changed as it draws, sends it to its own draw.
_BEYOND = (
    "its draw reads state beyond the points, colours and order XF carries of"
    " the camera's capture"
)

#: What a camera's array holds after its reset, as an expected picture
#: (``LivePainter._expected``): its background, read when compared.
_BACKGROUND = object()


@dataclass
class _Described:
    """What a camera's pixels are on the core, described: the pixels the
    description drew, the background it drew over, and the captures' draws
    it was made of (each a leaf's kind and recording, in order) with the
    fixed-orientation centres they asked for. ``whole`` is whether that
    background is the camera's own as its reset left it, nothing of an
    earlier picture composited into it.
    """

    camera: Any
    pixels: np.ndarray
    background: np.ndarray
    drawn: list[tuple[type, _Recording]]
    centres: dict[int, list[Any]]
    whole: bool = False


#: An array the camera holds is compared element by element up to this size,
#: and by identity above it (its pixel arrays: the camera's real draw, compared
#: with the description, sees whatever writing into them shows).
_COPIED_ELEMENTS = 1 << 16
#: What a camera holds that is told by its value.
_VALUES = (bool, int, float, complex, str, bytes, np.generic)


def _held(value: Any) -> tuple[Any, ...]:
    """What of ``value`` (an entry of a camera's instance state) a change is
    told by: an array's contents, a number's or a string's value, a
    container's members by identity, any other object's identity. A
    read-only ``FrozenArray`` is told by identity alone, and so is a large
    array.
    """
    if value is None or isinstance(value, _VALUES):
        return ("value", value)
    if isinstance(value, np.ndarray):
        if isinstance(value, FrozenArray) and not value.flags.writeable:
            return ("object", value)
        if value.size <= _COPIED_ELEMENTS:
            return ("array", value, value.copy())
        return ("object", value)
    if isinstance(value, dict):
        return ("dict", value, tuple(value.items()))
    if isinstance(value, list | tuple):
        return ("sequence", value, tuple(value))
    if isinstance(value, set | frozenset):
        return ("set", value, frozenset(value))
    return ("object", value)


def _same_held(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "value":
        return _same(a[1], b[1])
    if a[1] is not b[1]:
        return False
    if a[0] == "array":
        return _same(a[2], b[2])
    if a[0] in ("dict", "sequence"):
        return len(a[2]) == len(b[2]) and all(
            _identical(x, y) for x, y in zip(a[2], b[2], strict=True)
        )
    if a[0] == "set":
        return bool(a[2] == b[2])
    return True


def _identical(x: Any, y: Any) -> bool:
    if isinstance(x, tuple) and isinstance(y, tuple):  # a dict's items
        return len(x) == len(y) and all(p is q for p, q in zip(x, y, strict=True))
    return x is y


#: A camera's state: its instance entries (:func:`_held`) and its view's values.
_State = tuple[dict[str, Any], dict[str, Any]]


def _camera_state(camera: Any, kind: str) -> _State:
    """What ``camera`` holds now: every entry of its instance state
    (:func:`_held`), and the values its view states, read through the
    objects that hold them (its trackers and its frame change in place).
    """
    held = {k: _held(v) for k, v in vars(camera).items()}
    return held, view_values(camera, kind)


def _changed(before: _State, after: _State) -> str | None:
    """The first entry of a camera's state that differs between two
    :func:`_camera_state` snapshots, or ``None``.
    """
    held, view = before
    now, now_view = after
    added_or_removed = held.keys() ^ now.keys()
    if added_or_removed:
        return str(min(added_or_removed))
    for name, value in held.items():
        if not _same_held(value, now[name]):
            return name
    for name, value in view.items():
        if not _same(value, now_view[name]):
            return name
    return None


def _producing(
    name: str,
    method: Any,
    camera: Any,
    watched: Callable[[], bool],
    guarded: Callable[[], bool],
) -> Any:
    """``method`` (``camera``'s ``name``, whatever the scene made it), keeping
    what it returns for each recorded mobject in that mobject's log. While
    ``watched()``, a call that changes what the camera holds
    (:func:`_camera_state`) is kept in :data:`_UNSETTLED`: the frame reads
    that state beyond the values XF
    carries of the capture, so the camera's own draw stands. While
    ``guarded()`` (the camera drawing a capture itself that the core will
    draw over), so is a call that writes the camera's array: the camera's
    code is then a writer of it beside the core.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    kind = "three_d" if isinstance(camera, ThreeDCamera) else "flat"

    def produced(mobject: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(mobject, _Recording):
            return method(mobject, *args, **kwargs)
        log = mobject._xf_log
        start = len(log)
        before = _camera_state(camera, kind) if watched() else None
        pixels = np.array(camera.pixel_array) if guarded() else None
        result = method(mobject, *args, **kwargs)
        changed = (
            None if before is None else _changed(before, _camera_state(camera, kind))
        )
        if changed is not None and camera not in _UNSETTLED:
            _UNSETTLED[camera] = (
                f"{name} changed the camera's {changed} while it drew; {_BEYOND}"
            )
        if (
            pixels is not None
            and camera not in _UNSETTLED
            and not np.array_equal(pixels, camera.pixel_array)
        ):
            _UNSETTLED[camera] = f"{name} wrote the camera's array while it drew"
        # What the camera's code read off the mobject while it produced is
        # its own: the drawer is handed what it produced and never asks.
        mobject._xf_inside.update(range(start, len(log)))
        # Kept as it is now: the camera may reuse the array it returned.
        kept = np.array(result, dtype=float)
        if name == "transform_points_pre_display":
            points = args[0] if args else kwargs["points"]
            log.append((_PRE_DISPLAY, points, None, kept))
        else:
            background = bool(args[0] if args else kwargs.get("background", False))
            which = "fill" if name == "get_fill_rgbas" else background
            log.append((_CAMERA_RGBAS, which, None, kept))
        return result

    return produced


def _own_code(camera: Any) -> Any:
    from .painter import _own_code

    return _own_code(camera)


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
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    try:
        if bool(a == b):
            return True
    except (TypeError, ValueError):
        return False
    return _nan(a) and _nan(b)


def _nan(value: Any) -> bool:
    return isinstance(value, float | np.floating) and bool(value != value)


class _Reads:
    """A recording's log, by read."""

    _MISSING = object()

    def __init__(self, recording: _Recording) -> None:
        self.leaf = recording._xf_leaf
        self.log = recording._xf_log
        self.values: dict[tuple[str, Any], Any] = {}
        for name, args, kwargs, value in self.log:
            if name in (_PRE_DISPLAY, _CAMERA_RGBAS):
                continue  # what the camera produced, not a read
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
        self._sent = {k: v for k, v in self._sent.items() if k in slots}

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
    from arachne.timeline.frozen import engine_class

    kind = type(camera)
    if isinstance(camera, Camera):
        # A camera class of the scene's own draws as the engine's it
        # subclasses, but for what its records state it produced.
        kind = next(k for k in kind.__mro__ if engine_class(k))
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
    ``path`` is the draw path the pictures are drawn on (``ir_cairo`` or
    ``core``).
    """

    def __init__(self, *, draw: bool = True, path: str = "ir_cairo") -> None:
        self.draw = draw
        self.path = path
        #: The play the frame within :meth:`around` belongs to, if known.
        self.play: Any = None
        #: The frames :meth:`around` has drawn, the one within it included.
        self._frames = 0
        #: Whether the frame within :meth:`around` belongs to the scene's
        #: first: no frame of the scene, live or recorded, has been delivered
        #: before it (``painter.delivered``).
        self._opening = False
        #: Per hedged camera (by id), within the scene's first frame: the
        #: camera, its own draw so far and the description of it.
        self._chain: dict[int, tuple[Any, Any, Any]] = {}
        #: Per camera in :attr:`_chain`: its own draw of the picture the
        #: description of its last capture draws over, which Cairo draws that
        #: description over to compare (:meth:`_settle_opening`).
        self._unders: dict[int, Any] = {}
        #: Per hedged camera (by id): its own draw of the static picture under
        #: the scene's first frame, and the description that replaced it.
        self._statics: dict[int, tuple[Any, Any]] = {}
        #: Per camera (by id): the play last watched, and the frame watched
        #: in it (its first). A camera absent here has not captured yet.
        self._watched: dict[int, tuple[Any, int]] = {}
        #: Whether the capture being drawn is watched call by call, and
        #: whether the camera's own draw of it is made.
        self._watching = draw
        self._real = draw
        self._painters: dict[int, tuple[Any, Any]] = {}
        self._checker = CairoDrawer()
        self.captures = 0
        self.differing = 0
        #: Cameras (by id) whose captures stand as they drew them.
        self._fallen: set[int] = set()
        #: What of each camera (by id) is the scene's code, as
        #: ``painter.user_camera`` names it, read before live drawing wrapped it.
        self._own_codes: dict[int, tuple[str, str]] = {}
        #: The scene's instant the captures within :meth:`around` draw.
        self.instant = Fraction(0)
        #: The scene's camera and the draw path its frames were asked for on,
        #: within :meth:`around`.
        self._scene: tuple[Any, str] | None = None
        #: On the core, per camera (by id): what its last capture in the frame
        #: within :meth:`around` drew, described; and what its last play's
        #: static picture is.
        self._last: dict[int, _Described] = {}
        self._static_pictures: dict[int, _Described] = {}
        #: On the core, per camera (by id): the array the renderer keeps of
        #: its last play's static picture (:meth:`keep`), and the description
        #: of it.
        self._kept: dict[int, tuple[Any, _Described]] = {}
        #: On the core, per camera (by id), within :meth:`around`: the
        #: described picture its array holds, by what put it there (its
        #: capture, or the renderer setting a static picture it kept as its
        #: array), or ``None``. A capture that starts on one describes the
        #: frame from it (:meth:`_under`).
        self._holding: dict[int, _Described | None] = {}
        #: On the core, per camera (by id), within :meth:`around`: whether
        #: its array was last set by a reset (its own background), no
        #: capture drawn into it since.
        self._reset: dict[int, bool] = {}
        #: On the core, within :meth:`around`: the cameras (by id) whose
        #: array one of their methods wrote outside a capture
        #: (:data:`_COMPOSITING`) since it was last set or captured.
        self._composited: set[int] = set()
        #: On the core, per camera (by id), within :meth:`around`: what its
        #: array must hold when its next capture begins (the core is its one
        #: writer within a frame), and whether nothing of the core's is in
        #: it: what the core or the camera itself last drew into it, or what
        #: was set as it (its background, a static picture the renderer
        #: kept); absent where a capture draws over whatever it holds.
        self._expected: dict[int, tuple[Any, bool]] = {}
        #: An array of each shape (and type) that CE's own draws by numbers
        #: on the core go into, nothing reading it (:meth:`_compositing`).
        self._scratches: dict[tuple[int, ...], np.ndarray] = {}
        #: Per camera (by id), within :meth:`around`: the painter whose pixels
        #: its array holds (``painter.painter_kind``).
        self._drawn_by: dict[int, str] = {}
        #: On the core, within :meth:`around`: the frame's input grid, over
        #: the scene camera's view and every display camera's
        #: (``canonical.input_grid``, once per frame), which the scene
        #: camera's painter snaps the core's geometry to.
        self.grid: float | None = None
        #: The scene's mobjects, within :meth:`around`: a surface among
        #: their families draws its faces as one face group.
        self._roots: list[Any] = []

    def painter(self, camera: Any) -> Any:
        """The painter of ``camera``'s captures: the path's for the scene's
        camera, and Cairo from XF (``ir_cairo``, unsnapped) for a display's,
        as a frame drawn from the record has a display's picture drawn by
        Cairo from the scene's values (``CorePainter``'s overlay): the core
        does not widen a point cloud's blocks by a display camera's size, as
        CE does.
        """
        found = self._painters.get(id(camera))
        if found is None or found[0] is not camera:
            from .painter import painter_class

            if self._is_scene_camera(camera):
                painter = painter_class(self.path)(stream=_LiveStream())
            else:
                painter = painter_class("ir_cairo")(stream=_LiveStream())
            painter.live = True
            found = (camera, painter)
            self._painters[id(camera)] = found
            self._give_grid(painter)
        return found[1]

    def _give_grid(self, painter: Any) -> None:
        """``painter`` snaps at the frame's grid (:attr:`grid`)."""
        canonical = getattr(painter, "canonical", None)
        if canonical is not None:
            canonical.frame_grid = self.grid

    def _is_scene_camera(self, camera: Any) -> bool:
        return self._scene is not None and camera is self._scene[0]

    def _kind(self, camera: Any) -> str:
        """The painter ``camera``'s descriptions are drawn by."""
        from .painter import painter_kind

        return painter_kind(self.path if self._is_scene_camera(camera) else "ir_cairo")

    @contextlib.contextmanager
    def around(
        self,
        camera: Any,
        t: Fraction,
        play: Any = None,
        asked: str = "core",
        *,
        reset: bool = False,
        roots: Sequence[Any] = (),
    ) -> Iterator[LivePainter]:
        """Within: ``camera``'s captures, and its displays' cameras', are
        described in XF and drawn back from it, as one frame at instant ``t``
        of the play ``play`` (any key that tells plays apart, if known), of
        a scene whose frames were asked for on the draw path ``asked``. With
        ``reset``, ``camera``'s array is as its reset just left it, set
        before the frame's route was read: its first capture begins on it.
        ``roots`` are the scene's mobjects.
        """
        from .painter import deliver, delivered

        self.instant = t
        self._roots = list(roots)
        self.play = play
        self._scene = (camera, asked)
        self._frames += 1
        self._last.clear()
        self._holding.clear()
        self._reset.clear()
        self._composited.clear()
        self._expected.clear()
        if reset:
            self._reset[id(camera)] = True
            self._expected[id(camera)] = (_BACKGROUND, True)
        self._drawn_by.clear()
        self._opening = not delivered(camera, asked)
        displays = vars(camera).get("image_mobjects_from_cameras", ())
        cameras = [camera, *(display.camera for display in displays)]
        # On the core, the frame's grid, as a frame drawn from the record
        # decides it: its geometry is snapped to it, and its depth sort keys.
        self.grid = frame_grid(camera) if self.path == "core" else None
        for _, painter in self._painters.values():
            self._give_grid(painter)
        undo: list[Any] = []
        static = isinstance(play, tuple) and play[:1] == ("static",)
        try:
            # Installed one by one, so a failure undoes what was installed.
            undo.extend(
                self._install(each) for each in cameras if "_xf_live" not in vars(each)
            )
            with (
                canonical_order(camera, self.grid)
                if self.grid is not None
                else contextlib.nullcontext()
            ):
                yield self
            self._settle_opening(static)
            if static:
                # This play's static pictures, and no earlier play's.
                self._static_pictures = dict(self._last)
        finally:
            for restore in reversed(undo):
                restore()
        # A play's static picture is drawn under its first frame: the scene's
        # first frame is both.
        if not static:
            deliver(camera, asked, self._drawn_by.get(id(camera), self._kind(camera)))

    def _settle_opening(self, static: bool) -> None:
        """The scene's first frame has been drawn by each hedged camera
        (:meth:`_hedged`) as it draws, and described alongside. Where the
        camera changed nothing, the description replaces its picture, so the
        frame is the core's as every later one is; a static picture keeps
        its own draw beside it, for a fallback to draw over (:meth:`_behind`).

        Before it does, the camera's array, as its draw of the frame left it,
        is compared once with Cairo's draw of the frame's description: code
        the producer calls do not run (a mobject's of the scene's own, as the
        camera draws it) can have written the array as it drew, and the
        description would drop that. On any difference the camera's own draw
        stands, and the whole scene falls back to it (:meth:`_fall_back`):
        no frame of the scene has been delivered yet.
        """
        unders, self._unders = self._unders, {}
        for key, (camera, real, described) in self._chain.items():
            if camera in _UNSETTLED:
                continue
            drawn = self.painter(camera).drawn_by_cairo(unders[key])
            if not np.array_equal(camera.pixel_array, drawn):
                _UNSETTLED[camera] = (
                    "something other than the camera's own draw wrote its array"
                    " while it drew the scene's first frame, which the frame's"
                    " description lacks"
                )
                self._layered(camera)
                self._drawn_by[key] = "cairo"
                held, self._real = self._real, True
                try:
                    self._fall_back(camera)
                finally:
                    self._real = held
                continue
            np.copyto(camera.pixel_array, described)
            self._drawn_by[key] = self._kind(camera)
            if static:
                self._statics[key] = (real, described)
        self._chain.clear()

    def _hedged(self, camera: Any) -> bool:
        """Whether ``camera``'s captures in the scene's first frame are drawn
        as the camera draws them while the frame is drawn: on the core, a
        camera whose code the watch watches (one the scene's code changed),
        so a change seen in that frame falls back with nothing of the core's
        drawn before it.
        """
        return not self.draw and id(camera) in self._own_codes

    def _behind(self, camera: Any, now: Any) -> Any:
        """The picture the description of ``camera``'s capture about to run
        draws over, where the camera draws it itself: the description of the
        picture now under it, where that picture is the camera's own draw of
        the scene's first frame, or the description of it. In the latter
        case the camera's own draw is put back under its capture, so the
        camera draws over its own picture throughout.
        """
        key = id(camera)
        chain = self._chain.get(key)
        for pair in (chain[1:] if chain else None, self._statics.get(key)):
            if pair is None:
                continue
            real, described = pair
            if np.array_equal(now, real):
                return described
            if np.array_equal(now, described):
                np.copyto(camera.pixel_array, real)
                return described
        return now

    def _drawn_under(self, key: int, background: Any) -> Any:
        """The camera's own draw of the picture a description of its capture
        draws over (``background``): the draw a description of the scene's
        first frame stands for (:meth:`_behind`), or the picture itself.
        """
        chain = self._chain.get(key)
        for pair in (chain[1:] if chain else None, self._statics.get(key)):
            if pair is not None and background is pair[1]:
                return pair[0]
        return background

    def _under(
        self, camera: Any, drawn: list[tuple[type, _Recording]]
    ) -> _Described | None:
        """The description a capture of ``camera`` that draws ``drawn``
        continues as one frame: the picture the camera's array holds by what
        put it there (:attr:`_holding`), the frame's capture before it or the
        play's static picture installed as its array. The frame is then
        described from that picture's background with the draws of every
        capture in it, depth tested across all of them, as a frame drawn from
        the record is. Decided by what the captures drew, whatever their
        pixels show; not where a leaf of the picture is drawn again (CE draws
        such a leaf twice: in the picture's pixels, and again).
        """
        if self.draw:
            return None
        picture = self._holding.get(id(camera))
        if picture is None or picture.camera is not camera:
            return None
        drawing = {id(recording._xf_leaf) for _, recording in drawn}
        if any(id(r._xf_leaf) in drawing for _, r in picture.drawn):
            return None
        return picture

    def keep(self, camera: Any, pixels: Any) -> None:
        """The renderer keeps ``pixels``, a copy of ``camera``'s array, as the
        static picture of the play just drawn, to set back as its array under
        the play's frames: set back so (the very array), a live frame
        continues its description (:meth:`_installed`).
        """
        described = self._static_pictures.get(id(camera))
        if described is None or camera in _UNSETTLED:
            self._kept.pop(id(camera), None)
        else:
            self._kept[id(camera)] = (pixels, described)

    def _installed(self, camera: Any, pixel_array: Any) -> _Described | None:
        """The described picture ``camera``'s array holds once ``pixel_array``
        is set as it: the static picture the renderer kept, where it is the
        array the renderer said it keeps (:meth:`keep`); else none. A reset,
        or a picture the scene's code sets, is drawn over as its pixels,
        whatever they are like.
        """
        kept = self._kept.get(id(camera))
        if kept is not None and pixel_array is kept[0]:
            return kept[1]
        return None

    def _install(self, camera: Any) -> Any:
        from arachne.camera.camera import Camera
        from arachne.camera.mapping_camera import MappingCamera
        from arachne.mobject.mobject import Mobject
        from arachne.timeline.camera_code import engine_own

        own = vars(camera)
        funcs = camera.display_funcs
        camera.display_funcs = {
            kind: func if kind is Mobject else self._display(camera, kind, func)
            for kind, func in funcs.items()
        }
        own["capture_mobjects"] = engine_own(
            self._capture(camera, camera.capture_mobjects)
        )
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

            own["points_to_pixel_coords"] = engine_own(mapped)
            added.append("points_to_pixel_coords")
        held: dict[str, Any] = {}
        if not self.draw:
            context = camera.get_cairo_context

            def no_context(pixel_array: Any) -> Any:
                found = context(pixel_array)  # CE's context still freezes its centre
                return found if self._real else _NoContext()

            own["get_cairo_context"] = engine_own(no_context)
            added.append("get_cairo_context")
            set_array = camera.set_pixel_array

            def installing(pixel_array: Any, *args: Any, **kwargs: Any) -> Any:
                # A reset, a static picture the renderer kept, or the scene's own.
                found = set_array(pixel_array, *args, **kwargs)
                key = id(camera)
                installed = self._installed(camera, pixel_array)
                self._holding[key] = installed
                self._reset[key] = pixel_array is camera.background
                self._composited.discard(key)
                # The next capture begins on what was set: the background
                # (nothing of the core's in it), a picture the core drew and
                # the renderer kept, or the scene's own, which the capture
                # draws over as its pixels, so nothing is expected of it.
                if self._reset[key]:
                    self._expected[key] = (_BACKGROUND, True)
                elif installed is not None:
                    self._expected[key] = (pixel_array, False)
                else:
                    self._expected.pop(key, None)
                return found

            if "set_pixel_array" in own:
                held["set_pixel_array"] = own["set_pixel_array"]
            own["set_pixel_array"] = engine_own(installing)
            added.append("set_pixel_array")
            for name in _COMPOSITING:
                if name in own:
                    held[name] = own[name]
                own[name] = engine_own(
                    self._compositing(camera, name, getattr(camera, name))
                )
                added.append(name)
        own_code = _own_code(camera)
        if own_code is not None:
            # Named now: once wrapped, the scene's callables on the instance
            # read as the engine's.
            self._own_codes[id(camera)] = own_code
            # A camera the scene's code changed: what it produces for each
            # mobject it draws is kept, and its records are pre-projected.
            # Where the capture is the real draw, what it holds is watched
            # call by call, and the draw stands where it changes. The core
            # draws only cameras whose changes the gate proved
            # (``painter.PROVEN_ON_THE_GATE``) and watches a sample of their
            # captures (:meth:`_sample`).
            for name in _PRODUCED:
                if name in own:
                    held[name] = own[name]
                method = getattr(camera, name)
                own[name] = engine_own(
                    _producing(
                        name,
                        method,
                        camera,
                        lambda: self._watching,
                        lambda: self._guarding(camera),
                    )
                )
                added.append(name)
            own["_xf_pre_projected"] = True
            added.append("_xf_pre_projected")

        def restore() -> None:
            camera.display_funcs = funcs
            for name in added:
                own.pop(name, None)
            own.update(held)

        return restore

    def _compositing(self, camera: Any, name: str, method: Any) -> Any:
        """``method``, ``camera``'s ``name``, a method that writes the array
        it is handed (:data:`_COMPOSITING`): called outside a capture, the
        picture the camera's array holds is no capture's, nor its reset's.
        Called within a capture the core draws, one that writes the array by
        numbers (:data:`_BY_NUMBERS`) is CE's own draw of a leaf the core
        draws from its description: like a path drawn through the core's
        context, it draws nowhere (into an array nothing reads).
        """
        position = _BY_NUMBERS.get(name)

        def compositing(*args: Any, **kwargs: Any) -> Any:
            if not vars(camera)["_xf_live"]:
                self._composited.add(id(camera))
            elif position is not None and not self._real:
                args, kwargs = self._nowhere(camera, position, args, kwargs)
            return method(*args, **kwargs)

        return compositing

    def _nowhere(
        self, camera: Any, position: int, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[tuple, dict[str, Any]]:
        """``args`` and ``kwargs`` with the camera's array, where they hand it
        (at ``position``, or as ``pixel_array``), replaced by an array of the
        same shape that nothing reads.
        """
        array = camera.pixel_array
        if len(args) > position and args[position] is array:
            args = (*args[:position], self._scratch(array), *args[position + 1 :])
        elif kwargs.get("pixel_array") is array:
            kwargs = {**kwargs, "pixel_array": self._scratch(array)}
        return args, kwargs

    def _scratch(self, array: np.ndarray) -> np.ndarray:
        found = self._scratches.get(array.shape)
        if found is None or found.dtype != array.dtype:
            found = self._scratches[array.shape] = np.zeros_like(array)
        return found

    def _display(self, camera: Any, kind: type, func: Any) -> Any:
        from arachne.timeline.camera_code import engine_own

        def display(mobjects: list[Any], pixel_array: Any) -> Any:
            drawn = [_Recording(mob) for mob in mobjects]
            frames = vars(camera)["_xf_live"]
            if frames:
                frames[-1].extend((kind, recording) for recording in drawn)
            return func(drawn, pixel_array)

        # The engine's while it draws, standing for what it wraps
        # (``display_funcs_rewired``).
        display.__wrapped__ = func  # type: ignore[attr-defined]
        return engine_own(display)

    def _capture(self, camera: Any, capture: Any) -> Any:
        def described(mobjects: Any, **kwargs: Any) -> None:
            # A capture can run inside another (a MultiCamera's display's
            # camera, captured within the camera's own): it is watched by its
            # own camera's rule, and the capture around it goes on by its own.
            held = (self._watching, self._real)
            self._sample(camera)
            try:
                self._described(camera, capture, mobjects, kwargs)
            finally:
                self._watching, self._real = held

        return described

    def _described(
        self, camera: Any, capture: Any, mobjects: Any, kwargs: dict[str, Any]
    ) -> None:
        """``camera``'s ``capture`` of ``mobjects``, described in XF and drawn
        back from it, as :meth:`_sample` decided for it.
        """
        frames = vars(camera)["_xf_live"]
        key = id(camera)
        background = start = np.array(camera.pixel_array)
        # On the core, the camera's array has one writer within a frame:
        # the core. The capture begins on what was last drawn into it in
        # the frame, by the core or by the camera itself where it draws the
        # scene's first frame, or on what was set as it (:attr:`_expected`).
        # Anything else wrote it; a write that leaves it byte-equal wrote
        # nothing. That compare only refuses (:meth:`_sole_writer`).
        expected = self._expected.pop(key, None)
        written = None
        if (
            not self.draw
            and expected is not None
            and camera not in _UNSETTLED
            and not np.array_equal(start, self._picture(camera, expected[0]))
        ):
            written = (
                "something other than the core wrote the camera's array before"
                " its capture (after its reset, or between two captures of one"
                " frame)"
            )
            if expected[0] is _BACKGROUND and not self._opening:
                # After a reset, on a frame after the scene's first: the array
                # holds nothing of the core's, so the capture draws over what
                # was written as its pixels (``background``), as Cairo draws
                # over it. In the scene's first frame the camera's own draw
                # stands instead (``_sole_writer``).
                self._layered(camera)
            else:
                self._sole_writer(camera, written, mine=expected[1])
        # Whole: begun on the camera's reset with nothing written into its
        # array since, or continuing a picture that was, nothing written
        # since that picture's capture. What wrote is the record's: set
        # (``set_pixel_array``), drawn by a capture, or composited by one of
        # the camera's methods outside a capture (``_compositing``); a write
        # no method hears (a scene's code assigning into the array) is
        # refused by the compare above, which never makes a capture whole.
        composited = key in self._composited
        self._composited.discard(key)
        whole = self._reset.pop(key, False) and not composited and written is None
        if self._real and not self.draw:
            background = self._behind(camera, background)
        drawn: list[tuple[type, _Recording]] = []
        centres: dict[int, list[Any]] = {}
        # Where the faces drawn find their surface: the scene's families, and
        # the list captured (a play's static list holds faces alone).
        owners = [*self._roots, *(mobjects if isinstance(mobjects, list) else ())]
        frames.append(drawn)
        restore = self._record_centres(camera, centres)
        try:
            capture(mobjects, **kwargs)
        finally:
            restore()
            frames.pop()
        if camera in _UNSETTLED:
            # The camera's code changed what it holds while it produced
            # what a record states (its draw reads that state beyond what
            # XF carries), or something else wrote its array: its own draw
            # stands, from then on.
            self._layered(camera)
            self._drawn_by[key] = "cairo"
            self._fall_back(camera)
            return
        if (
            not self.draw
            and not self._real
            and not np.array_equal(camera.pixel_array, start)
        ):
            # Before the core draws over it: what wrote the array as the
            # camera captured is no draw of the capture's (which draws
            # nowhere on the core), and the core's picture would drop it.
            self._sole_writer(
                camera,
                "something other than the core wrote the camera's array while"
                " it captured",
                mine=False,
            )
        under = self._under(camera, drawn)
        if under is not None:
            background, whole = under.background, under.whole and not composited
            merged = dict(under.centres)
            for leaf, values in centres.items():
                merged[leaf] = [*merged.get(leaf, []), *values]
            drawn, centres = [*under.drawn, *drawn], merged
        if self._real and not self.draw:
            self._unders[key] = self._drawn_under(key, background)
        pixels = self._describe(camera, drawn, centres, background, owners)
        painter = self.painter(camera)
        if isinstance(painter, CorePainter):
            # The core painter's frame is this capture's: whole or not, as
            # recorded here, never as its pixels look.
            painter.whole = whole
        if not self.draw:
            described = _Described(camera, pixels, background, drawn, centres, whole)
            self._last[id(camera)] = described
            self._holding[id(camera)] = described
        if self.draw and not np.array_equal(camera.pixel_array, pixels):
            if vars(camera).get("_xf_pre_projected"):
                # A camera described from its own capture drew other than
                # its description (state XF cannot see: its Cairo
                # context's, held in place, outside the camera): its own
                # draw stands, from then on.
                _UNSETTLED[camera] = (
                    f"its capture at t={self.instant} drew other than XF describes"
                    f" it; {_BEYOND}"
                )
                self._drawn_by[id(camera)] = "cairo"
                self._fall_back(camera)
                return
            self.differing += 1
        self.captures += 1
        if self._real and not self.draw:
            # The scene's first frame stays the camera's own draw while it
            # is drawn, every capture in it: a change seen in any of them
            # then leaves nothing of the core's in it. Its description
            # replaces it once the frame is drawn (:meth:`_settle_opening`).
            real = np.array(camera.pixel_array)
            self._chain[key] = (camera, real, pixels)
            self._drawn_by[key] = "cairo"
            if not self.draw:
                self._expected[key] = (real, True)
        else:
            np.copyto(camera.pixel_array, pixels)
            self._drawn_by[key] = self._kind(camera)
            if not self.draw:
                self._expected[key] = (pixels, False)

    @staticmethod
    def _picture(camera: Any, expected: Any) -> Any:
        return camera.background if expected is _BACKGROUND else expected

    def _guarding(self, camera: Any) -> bool:
        """Whether ``camera``'s capture is drawn as the camera draws it, for
        the core to draw over once the frame is drawn (the scene's first
        frame of a hedged camera, :meth:`_hedged`): the camera's code writing
        its array as it produces would be dropped then, so it is watched for
        (:func:`_producing`).
        """
        return not self.draw and self._real and camera not in _UNSETTLED

    def _layered(self, camera: Any) -> None:
        """``camera``'s core painter, if it has one, holds no capture known
        whole (:attr:`.CorePainter.whole`).
        """
        painter = self.painter(camera)
        if isinstance(painter, CorePainter):
            painter.whole = False

    def _sole_writer(self, camera: Any, why: str, *, mine: bool) -> None:
        """Something other than the core wrote ``camera``'s array within the
        frame, ``why``, and the core's picture of the capture would drop it.
        The frame is layered. Where the array holds nothing the core drew
        (``mine``) and no frame of the scene has been delivered, the capture
        is drawn as the camera draws it and the whole scene falls back to
        the camera's own draw, which is Cairo's (:meth:`_fall_back`).
        Otherwise the frame is ``render.unable``: the core keeps no draw of
        the camera's own under its picture, and a scene's frames are drawn
        by one painter (``painter.deliver``).
        """
        self._layered(camera)
        if mine and self._opening:
            _UNSETTLED[camera] = why
            self._real = True
            return
        raise Unable(
            f"render.unable: a {type(camera).__name__}: {why}, which a frame the"
            " core draws would drop; the core keeps no draw of the camera's own to"
            " fall back on in this frame, and a scene's frames are drawn by one"
            " painter"
        )

    def _sample(self, camera: Any) -> None:
        """Whether ``camera``'s capture about to run is watched call by call,
        and whether the camera's own draw of it is made. With a real draw
        (``ir_cairo``) both, always. On the core, which draws no capture of
        the camera's own, the watch samples: every capture of the first frame
        of each play (a play's static picture is a frame of its own). A
        hedged camera (:meth:`_hedged`) also draws the scene's first frame
        (with the static picture drawn under it) as it draws, every capture
        in it, so a change seen there sends the whole scene to Cairo from its
        first frame; a camera fallen so draws every later capture itself.
        """
        if self.draw:
            self._watching = self._real = True
            return
        key = id(camera)
        watched = self._watched.get(key)
        if watched is None or (self.play is not None and watched[0] != self.play):
            watched = self._watched[key] = (self.play, self._frames)
        fallen = camera in _UNSETTLED
        self._watching = watched[1] == self._frames and not fallen
        self._real = (self._opening and self._hedged(camera)) or fallen

    def _fall_back(self, camera: Any) -> None:
        """The capture just drawn stands as the camera drew it, with one
        ``render.fallback`` per camera saying why (:data:`_UNSETTLED`). On the
        core the scene's first frame is drawn so; a change first seen in a
        later play, after the core has drawn frames, is ``render.unable``.
        """
        reason = _UNSETTLED[camera]
        if not self._real:
            raise Unable(
                f"render.unable: a {type(camera).__name__}: {reason}, after the"
                " core had drawn its frames"
            )
        if id(camera) in self._fallen:
            return
        self._fallen.add(id(camera))
        from .lint import raise_finding
        from .painter import fall_back

        if not self.draw and self._scene is not None:
            # The whole scene is its camera's own draw from its first frame.
            fall_back(*self._scene)

        own = self._own_codes.get(id(camera))
        name = type(camera).__name__
        subject, why = own if own is not None else (name, f"a {name}")
        where = "" if self.draw else " for the whole scene"
        raise_finding(
            "render.fallback", subject, f"{why} is drawn on Cairo{where}: {reason}"
        )

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
        owners: list[Any],
    ) -> np.ndarray:
        """The capture's frame: each leaf drawn, its record made from what the
        capture read of it, in the order drawn; a surface's uv faces, all
        drawn, as one face group (``owners``: where surfaces are found), each
        face a part of it at its own place in that order.
        """
        kind = _view_kind(camera)
        painter = self.painter(camera)
        stream: _LiveStream = painter.stream
        uploads: dict[bytes, Any] = {}
        objects: list[tuple[int, str, str]] = []
        records: dict[int, Record] = {}
        order: list[tuple[str, int, int]] = []
        # The capture itself froze the centre if it drew through the context.
        view = stream._view_of(camera, kind, uploads, background=background)
        held = stream._held
        stream._contain(owners)
        orientation = vars(camera).get("fixed_orientation_mobjects")
        made = [
            (
                recording._xf_leaf,
                self._record(camera, leaf_kind, recording, centres, uploads),
            )
            for leaf_kind, recording in drawn
        ]
        faces = _face_groups(made, owners, uploads)
        for leaf, record in made:
            mob, part = leaf, 0
            face = faces.get(id(leaf))
            if face is not None:
                mob, part, record = face
            slot = stream._slot(mob, objects)
            if face is not None and slot in records:
                # A later face of a group already placed: one record.
                order.append(("moving", slot, part))
                continue
            # A face group's depth is its first face's.
            facing = leaf if face is None else _uv_faces(mob)[0]
            record = stream._sourced(slot, record, mob, facing, orientation)
            known = held.get(slot)
            if known is not None and _same_record(known, record):
                record = known
            placed = records.get(slot)
            if placed is not None and not _same_record(placed, record):
                raise Unable(
                    f"render.unable: a {type(mob).__name__} drawn two ways in one"
                    " capture"
                )
            records[slot] = record if placed is None else placed
            order.append(("moving", slot, part))
        delta = stream._delta(view, records, order, uploads, objects, self.instant)
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
        if vars(camera).get("_xf_pre_projected"):
            fields["pre_projected"] = _produced(recording, kind, own)
            fields["features"] = int(fields.get("features", 0)) | FEATURE_PRE_PROJECTED
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
        inside = recording._xf_inside
        for index, (name, args, kwargs, value) in enumerate(recording._xf_log):
            if name == "__mapped__" or (mapped and name == "points"):
                continue  # the record holds the mapped points
            if name in (_PRE_DISPLAY, _CAMERA_RGBAS) or index in inside:
                continue  # what the camera produced, or read to produce it
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


def _produced(
    recording: _Recording, kind: str, uploads: dict[bytes, Any]
) -> dict[str, Any]:
    """What the camera produced for the mobject ``recording`` drew, as a
    pre-projected record states it: where its pre-display code placed the
    mobject's points (``points``) and its gradient's ends (``gradient``),
    and, for a path, the colours it painted with. A value produced two ways
    in one capture, or a placement of points the record does not hold, is
    ``render.unable``.
    """
    log = recording._xf_log
    # The arrays the camera was handed come from the mobject's reads: the
    # points (corners of an image), and each reading of its gradient's ends.
    points_read = [v for n, a, _, v in log if n == "points" and a is None]
    gradient_read = [
        v for n, a, _, v in log if n == "get_gradient_start_and_end_points"
    ]
    found: dict[str, Any] = {}

    def keep(name: str, value: Any) -> None:
        if name in found and not _same(found[name], value):
            raise Unable(
                f"render.unable: a {type(recording._xf_leaf).__name__}'s"
                f" {name} produced two ways in one capture"
            )
        found.setdefault(name, value)

    for name, source, _, value in log:
        if name == _PRE_DISPLAY:
            if any(source is p for p in points_read):
                target = "points"
            elif any(source is g for g in gradient_read):
                target = "gradient"
            else:
                raise Unable(
                    "render.unable: a camera placed points its record does not"
                    f" hold, for a {type(recording._xf_leaf).__name__}"
                )
            value = np.array(value, dtype=float)
            if value.ndim != 2 or value.shape[1] != 3:
                raise Unable(
                    "render.unable: a camera's pre-display gave points of shape"
                    f" {value.shape}"
                )
            keep(target, value)
        elif name == _CAMERA_RGBAS:
            target = {"fill": "fill", False: "stroke", True: "background_stroke"}[
                source
            ]
            keep(target, np.asarray(value, dtype=float).reshape(-1, 4))
    points = found.get("points", np.zeros((0, 3)))
    pre: dict[str, Any] = {"points": _path_shape(points, uploads)}
    gradient = found.get("gradient")
    pre["gradient"] = None if gradient is None else _floats(gradient)
    if kind == "path":
        for name in ("fill", "stroke", "background_stroke"):
            rows = found.get(name)
            pre[name] = None if rows is None else _paint(rows, uploads)
    return pre


def _face_groups(
    made: list[tuple[Any, Record]], owners: list[Any], uploads: dict[bytes, Any]
) -> dict[int, tuple[Any, int, Record]]:
    """The leaves of ``made`` (each with the record its capture's reads made)
    that are faces of a surface drawn as one face group, as the record's
    frames group them (``extract._faces``), by leaf id: the surface, the
    face's index among its uv faces, and the group's record. A surface
    groups only where each of its uv faces is in ``made``, every draw of a
    face one way (a face drawn two ways is left to be refused as a path
    is). Its surfaces are found by their submobjects (no method of theirs
    runs).
    """
    if not any(hasattr(leaf, "u_index") for leaf, _ in made):
        return {}
    from arachne.mobject.three_d.three_dimensions import Surface

    drawn: dict[int, Record | None] = {}
    for leaf, record in made:
        held = drawn.setdefault(id(leaf), record)
        if held is not None and held is not record and not _same_record(held, record):
            drawn[id(leaf)] = None
    found: dict[int, tuple[Any, int, Record]] = {}
    seen: set[int] = set()
    stack = list(owners)
    while stack:
        owner = stack.pop()
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        stack.extend(owner.submobjects)
        if not isinstance(owner, Surface):
            continue
        faces = _uv_faces(owner)
        records = [drawn.get(id(face)) for face in faces]
        if not faces or any(record is None for record in records):
            # Not every face drawn, or one drawn two ways.
            continue
        group = _folded(owner, faces, records, uploads)  # type: ignore[arg-type]
        if group is not None:
            for index, face in enumerate(faces):
                found[id(face)] = (owner, index, group)
    return found


#: Path fields a face of a group has of its own: the group takes its
#: points, colours and stroke width face by face, and the first face's
#: gradient and width as the record's.
_FACE_OWN = frozenset({"shape", "fill", "stroke", "gradient", "stroke_width"})
#: Path fields no face of a group carries.
_NOT_FACE = frozenset({"fill_image", "features", "pre_projected"})


def _folded(
    surface: Any, faces: list[Any], records: list[Record], uploads: dict[bytes, Any]
) -> Record | None:
    """The face group ``surface``'s uv faces ``faces`` draw as, from the
    records their capture made (``records``), as the record's frames make it
    (``FrameStream._face_group``): the faces' points in turn and where each
    ends, their fill and stroke rows and stroke widths, their uv indices and
    the surface's grid, and the first face's style and depth. ``None`` where
    they do not draw as one group: one style but what each face carries of
    its own (its points, however many, its colours, each one stop, and its
    stroke width), and no face stating its own gradient ends, which a face
    group does not draw (``extract._one_group``).
    """
    if any("get_gradient_start_and_end_points" in vars(face) for face in faces):
        return None
    if any(record.kind != "path" for record in records):
        return None
    completed = [
        _inactive_strokes(face, record.fields, uploads)
        for face, record in zip(faces, records, strict=True)
    ]
    if any(fields is None for fields in completed):
        return None
    first = completed[0]
    assert first is not None
    points: list[np.ndarray] = []
    fills: list[np.ndarray] = []
    strokes: list[np.ndarray] = []
    widths: list[float] = []
    for fields in completed:
        assert fields is not None
        if fields.keys() != first.keys():
            return None
        if _NOT_FACE & fields.keys():
            return None
        if any(
            not _same(fields[name], value)
            for name, value in first.items()
            if name not in _FACE_OWN
        ):
            return None
        fill, stroke = fields["fill"], fields["stroke"]
        if not all(
            isinstance(rows, np.ndarray) and len(rows) == 1 for rows in (fill, stroke)
        ):
            return None
        data = np.asarray(uploads[fields["shape"]].data)
        if len(data) == 0:
            return None
        points.append(data)
        fills.append(fill)
        strokes.append(stroke)
        widths.append(float(fields["stroke_width"]))
    fields = {"shape": _path_shape(np.concatenate(points), uploads)}
    fields.update((name, value) for name, value in first.items() if name != "shape")
    face_fill = PaintTable(np.concatenate(fills))
    face_stroke = PaintTable(np.concatenate(strokes))
    uploads[face_fill.key] = face_fill
    uploads[face_stroke.key] = face_stroke
    fields["fill"] = None
    fields["stroke"] = None
    fields["face_fill"] = face_fill.key
    fields["face_stroke"] = face_stroke.key
    fields["uv"] = np.array(
        [[face.u_index, face.v_index] for face in faces], dtype=np.uint32
    )
    fields.update(_face_lists([len(data) for data in points], widths))
    resolution = surface.resolution
    u_res, v_res = (
        (resolution, resolution) if isinstance(resolution, int) else resolution
    )
    fields["grid"] = (int(u_res), int(v_res))
    return Record("face_group", fields)


def _inactive_strokes(
    face: Any, fields: dict[str, Any], uploads: dict[bytes, Any]
) -> dict[str, Any] | None:
    """``fields`` with each stroke drawn 0 wide given the paint the record's
    frames give it. CE reads no stroke colour at that width
    (``Camera.apply_stroke``); the record reads the face's own
    (``VMobject.get_stroke_rgbas``). Here it is read off the face's paint,
    its state, and nothing of the face or its class runs: only where the
    accessor, the colours' property and the paint's storage are each the
    engine's own (:func:`_engine_paint`). ``None`` otherwise.
    """
    found = dict(fields)
    for name, width, attribute in (
        ("stroke", "stroke_width", "stroke_rgbas"),
        ("background_stroke", "background_stroke_width", "background_stroke_rgbas"),
    ):
        if found.get(name) is not None or found.get(width) != 0:
            continue
        paint = _engine_paint(face, attribute)
        if paint is None:
            return None
        # As the property reads the field, and the accessor a missing one.
        rows = vars(paint).get(attribute)
        found[name] = _paint(np.zeros((1, 4)) if rows is None else rows, uploads)
    return found


def _engine_paint(face: Any, attribute: str) -> Any:
    """``face``'s paint (:class:`~.values.Paint`), off its own state, where
    ``get_stroke_rgbas`` reads ``attribute`` of it by the engine's code
    alone: neither the face nor its class puts another accessor, property
    or paint over the engine's (each compared by identity, no descriptor
    run, with what the engine's ``VMobject`` held when the engine was
    imported). ``None`` otherwise.
    """
    import inspect

    from arachne.mobject.types.vectorized_mobject import VMobject
    from arachne.mobject.values import Paint
    from arachne.timeline.frozen import class_members

    engine = class_members(VMobject)
    own = vars(face)
    missing = object()
    for member in ("get_stroke_rgbas", attribute, "_paint"):
        if member != "_paint" and member in own:
            return None
        held = inspect.getattr_static(type(face), member, missing)
        if held is missing or held is not engine.get(member, missing):
            return None
    paint = own.get("_paint")
    return paint if type(paint) is Paint else None


def _same_record(a: Record, b: Record) -> bool:
    if a.kind != b.kind or a.fields.keys() != b.fields.keys():
        return False
    return all(_same(a.fields[k], b.fields[k]) for k in a.fields)


_LIVE: weakref.WeakKeyDictionary[Any, tuple[int, LivePainter]] = (
    weakref.WeakKeyDictionary()
)


@contextlib.contextmanager
def live(
    camera: Any,
    t: Fraction,
    draw_path: str | None = None,
    play: Any = None,
    *,
    reset: bool = False,
    roots: Sequence[Any] = (),
) -> Iterator[LivePainter | None]:
    """Within: ``camera`` captures through the draw path ``draw_path`` (the
    option of that name, unless given), each capture a frame at the scene's
    instant ``t`` of the play ``play`` (a key that tells plays apart, if
    known). On ``cairo`` nothing changes. ``reset`` says the camera was
    reset just before, nothing written into its array since: its first
    capture begins on that reset (:attr:`.CorePainter.whole`), as one reset
    within does. ``roots`` are the scene's mobjects: a surface among their
    families draws its faces as one face group though a capture's list
    holds its faces alone.
    """
    from .painter import DRAW_PATHS, camera_path

    if draw_path is None:
        from arachne import config

        draw_path = config.draw_path
    path = camera_path(camera, draw_path)
    if not DRAW_PATHS[path]:
        yield None
        return
    found = _LIVE.get(camera)
    pid = os.getpid()
    if found is None or found[0] != pid or found[1].path != path:
        # On the core, CE's own draw of the capture is not wanted: the core
        # draws the picture (its real draw would only be thrown away).
        found = (pid, LivePainter(draw=path != "core", path=path))
        _LIVE[camera] = found
    with found[1].around(
        camera, t, play, draw_path, reset=reset, roots=roots
    ) as painter:
        yield painter


def keep_static_picture(camera: Any, pixels: Any) -> None:
    """The renderer keeps ``pixels``, a copy of ``camera``'s array, as the
    static picture just drawn (:meth:`LivePainter.keep`).
    """
    painter = live_painter(camera)
    if painter is not None:
        painter.keep(camera, pixels)


def live_painter(camera: Any) -> LivePainter | None:
    """The live painter of ``camera`` in this process, if it has drawn."""
    found = _LIVE.get(camera)
    return None if found is None or found[0] != os.getpid() else found[1]
