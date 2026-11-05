"""The painter seam: how a shown frame becomes pixels, by draw path.

``Drawing`` hands a frame to the painter its draw path names (the option
``draw_path``). On ``cairo`` there is none: the camera draws the shown
mobjects itself, as it always has. On ``ir_cairo`` the frame is turned into an
XF delta (:class:`~.extract.FrameStream`), the core checks the delta whole
(``arachne_core``, the ``core`` extra), and Cairo draws the frame from the XF
stream alone (:class:`~.cairo_painter.CairoDrawer`). On ``core`` the core
draws the frame, and Cairo draws its overlay over it
(:class:`~.core_painter.CorePainter`); ``auto`` is ``core`` for 3D scenes and
``cairo`` for 2D ones (:func:`scene_path`). Frames drawn as the scene
runs (``LiveFrames``, ``update_frame``) take the same path through
:func:`~.live.live`, described from what the camera's own capture consumed.

A painter and its stream are made in the process that draws, after any fork:
:func:`painter_for` keeps one per record and process.
"""

from __future__ import annotations

__all__ = [
    "ASSUMED_STATE",
    "CARRIED_MEMBERS",
    "PROVEN_ON_THE_GATE",
    "IrCairoPainter",
    "camera_path",
    "captured",
    "camera_changes",
    "changed_members",
    "deliver",
    "deliveries",
    "delivered",
    "fall_back",
    "fell_back",
    "painter_kind",
    "painter_for",
    "scene_path",
    "uncarried",
    "unmodelled_state",
]

import contextlib
import inspect
import os
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from .cairo_painter import CairoDrawer
from .canonical import Canonical
from .extract import FrameStream, Term, Unable
from .lint import raise_finding
from .xf import Delta, StreamState

#: The draw paths, and whether each describes frames in XF (``auto`` names
#: one of them per scene: :func:`scene_path`).
DRAW_PATHS = {"cairo": False, "ir_cairo": True, "core": True}


def user_camera(camera: Any) -> tuple[str, str] | None:
    """What of ``camera``, or of a display camera it draws through
    (:func:`~arachne.timeline.camera_code.drawn_cameras`), is the scene's own
    code, as ``(subject, why)``: a method of an engine camera class replaced
    on the class (:func:`~arachne.timeline.camera_code.replaced_methods`); a
    camera's class, a subclass of one of the engine's cameras defined
    outside the engine; or a method of its class replaced on the instance.
    ``None`` for engine cameras as the engine made them, or for no camera at
    all.
    """
    from arachne.timeline.camera_code import drawn_cameras

    for each in drawn_cameras(camera):
        own = _own_code(each)
        if own is not None:
            if each is camera:
                return own
            subject, why = own
            return subject, (
                f"{why.rstrip(',')}, a display camera of the {type(camera).__name__},"
            )
    return None


def _own_code(camera: Any) -> tuple[str, str] | None:
    """:func:`user_camera` for ``camera`` alone."""
    from arachne.camera.camera import Camera
    from arachne.timeline.camera_code import display_funcs_rewired, replaced_methods
    from arachne.timeline.frozen import engine_class

    if not isinstance(camera, Camera):
        return None
    kind = type(camera)
    replaced = replaced_methods(camera)
    patched = ", ".join(r.qualname for r in replaced if r.form == "class")
    if patched:
        return (
            patched,
            f"{patched}, replaced on the engine's camera class by the scene's code,",
        )
    if not engine_class(kind):
        base = next(k for k in kind.__mro__ if engine_class(k))
        try:
            where = f"{inspect.getsourcefile(kind)}:{inspect.getsourcelines(kind)[1]}"
        except (OSError, TypeError):
            where = kind.__module__
        return kind.__qualname__, (
            f"{kind.__qualname__} ({where}), a camera class of the scene's own"
            f" subclassing {base.__name__}"
        )
    shadowed = ", ".join(r.qualname for r in replaced if r.form == "instance")
    if shadowed:
        return (
            shadowed,
            f"{shadowed}, replaced on the camera instance by the scene's code,",
        )
    if display_funcs_rewired(camera):
        subject = f"{kind.__name__}.display_funcs"
        return (
            subject,
            f"{subject}, rewired on the camera instance by the scene's code,",
        )
    return None


#: The camera members a scene's code may change and XF still describe: each
#: reaches a frame only through what the camera's own capture produces and
#: the frame records (the points its pre-display code places, the colours it
#: shades with, which mobjects it draws in which order, the values its
#: projection reads), or only builds the camera.
CARRIED_MEMBERS = frozenset(
    {
        "__init__",
        "transform_points_pre_display",
        "project_points",
        "project_point",
        "generate_rotation_matrix",
        "get_rotation_matrix",
        "reset_rotation_matrix",
        "get_phi",
        "get_theta",
        "get_gamma",
        "get_focal_distance",
        "get_zoom",
        "modified_rgbas",
        "get_fill_rgbas",
        "get_stroke_rgbas",
        "get_mobjects_to_display",
    }
)

#: Names a class body defines that are no member it overrides.
_CLASS_BOOKKEEPING = frozenset(
    {
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__annotate__",
        "__annotate_func__",
        "__annotations_cache__",
        "__firstlineno__",
        "__static_attributes__",
        "__classcell__",
        "__orig_bases__",
        "__parameters__",
    }
)


def camera_changes(camera: Any) -> tuple[tuple[str, str], ...]:
    """How the scene's code changed ``camera``, as ``(form, member)`` pairs,
    sorted: ``"subclass"`` for a name its own classes define over an engine
    camera's; ``"class"`` and ``"instance"`` for a method replaced on an
    engine class or on the instance
    (:func:`~arachne.timeline.camera_code.replaced_methods`); and
    ``("instance", "display_funcs")`` when ``display_funcs`` holds anything
    other than the engine's own display methods
    (:func:`~arachne.timeline.camera_code.display_funcs_rewired`).
    """
    from arachne.camera.camera import Camera
    from arachne.timeline.camera_code import display_funcs_rewired, replaced_methods
    from arachne.timeline.frozen import engine_class

    if not isinstance(camera, Camera):
        return ()
    kind = type(camera)
    base = next(k for k in kind.__mro__ if engine_class(k))
    changes = {
        ("subclass", name)
        for klass in kind.__mro__
        if klass is not object and not engine_class(klass)
        for name in vars(klass)
        if name not in _CLASS_BOOKKEEPING and hasattr(base, name)
    }
    changes.update((r.form, r.name) for r in replaced_methods(camera))
    if display_funcs_rewired(camera):
        changes.add(("instance", "display_funcs"))
    return tuple(sorted(changes))


def changed_members(camera: Any) -> tuple[str, ...]:
    """The engine camera members the scene's code changed on ``camera``
    (:func:`camera_changes`), each once, sorted.
    """
    return tuple(sorted({member for _, member in camera_changes(camera)}))


def uncarried(camera: Any) -> tuple[Any, str] | None:
    """The first camera ``camera`` draws with (itself, or a display camera,
    :func:`~arachne.timeline.camera_code.drawn_cameras`) and the first member
    the scene's code changed there whose effect can reach the frame other
    than through what XF carries of a capture (:data:`CARRIED_MEMBERS`):
    Cairo calls of the scene's own, pixels written by hand, a pixel mapping.
    An engine camera class's ``__init__`` patched counts too: the drawer
    builds engine cameras of its own, and would run it there.
    """
    from arachne.timeline.camera_code import drawn_cameras, replaced_methods

    for each in drawn_cameras(camera):
        if any(
            r.form == "class" and r.name == "__init__" for r in replaced_methods(each)
        ):
            return each, "__init__"
        for name in changed_members(each):
            if name not in CARRIED_MEMBERS:
                return each, name
    return None


#: The camera changes (:func:`camera_changes`) the corpus gate has drawn on
#: ``ir_cairo`` with its pixel backstop on, every frame equal to the camera's
#: own draw: what a camera the scene's code changed may hold to be drawn on
#: ``core``, which draws only the description (it keeps no draw of the
#: camera's own to compare with or fall back on mid-scene).
#: At this revision the gate's corpus proves a camera class of the scene's own
#: overriding its constructor and ``project_points``, and ``project_points``
#: replaced on the instance (21 scenes, every capture and sample equal).
PROVEN_ON_THE_GATE = frozenset(
    {
        ("subclass", "__init__"),
        ("subclass", "project_points"),
        ("instance", "project_points"),
    }
)


def unproven_on_core(camera: Any) -> tuple[str, str] | None:
    """The first change the scene's code made to a camera ``camera`` draws
    with that the gate has not proven (:data:`PROVEN_ON_THE_GATE`), or
    ``None``.
    """
    from arachne.timeline.camera_code import drawn_cameras

    for each in drawn_cameras(camera):
        for change in camera_changes(each):
            if change not in PROVEN_ON_THE_GATE:
                return change
    return None


def _is_uint8(value: Any) -> bool:
    try:
        return bool(np.dtype(value) == np.uint8)
    except TypeError:
        return False


#: A camera's drawing state that XF does not carry and its drawers take as
#: given, as ``(attribute, the value taken, whether a value is it)``: what
#: CE's display code reads of the camera beyond the view XF states
#: (``extract.view_values``), the background and mapping centre, and what a
#: changed camera's capture produces. The core writes 8-bit RGBA, a full
#: channel as 255 (CE writes a point cloud's colours by ``rgb_max_val``), and
#: Cairo's drawers are engine cameras made with these values. The rest of
#: what the display code reads is carried, or never reaches a frame
#: (``max_allowable_norm``: ``adjust_out_of_range_points`` is not called).
ASSUMED_STATE: tuple[tuple[str, str, Callable[[Any], bool]], ...] = (
    ("rgb_max_val", "255", lambda value: value == 255),
    ("pixel_array_dtype", "uint8", _is_uint8),
    ("image_mode", "RGBA", lambda value: value == "RGBA"),
    ("n_channels", "4", lambda value: value == 4),
)


#: What of a Cairo context the core's drawers take as CE's
#: ``make_cairo_context`` leaves it: CE's draws set the rest (source, line
#: width, cap, join, path) for each mobject they draw.
_CONTEXT_STATE: tuple[tuple[str, Callable[[Any], Any]], ...] = (
    ("matrix", lambda ctx: tuple(ctx.get_matrix())),
    ("clip", lambda ctx: ctx.clip_extents()),
    ("operator", lambda ctx: ctx.get_operator()),
    ("antialias", lambda ctx: ctx.get_antialias()),
    ("tolerance", lambda ctx: ctx.get_tolerance()),
    ("fill rule", lambda ctx: ctx.get_fill_rule()),
)
_MADE_CONTEXTS: dict[tuple[Any, ...], tuple[Any, ...]] = {}


def _context_changed(camera: Any) -> tuple[str, Any] | None:
    """The first state of the Cairo context ``camera`` keeps for its array
    (CE draws every path through it) that is not what CE's
    ``make_cairo_context`` makes for the centre it froze, as ``(name,
    value)``; ``None`` when it keeps none, or one as made.
    """
    get_cached = getattr(camera, "get_cached_cairo_context", None)
    array = getattr(camera, "pixel_array", None)
    if get_cached is None or array is None:
        return None
    context = get_cached(array)
    if context is None:
        return None
    centre = camera.pixel_array_to_mapping_centre.get(id(array), camera.frame_center)
    key = (
        type(camera),
        array.shape,
        camera.pixel_width,
        camera.pixel_height,
        camera.frame_width,
        camera.frame_height,
        tuple(float(v) for v in centre),
    )
    made = _MADE_CONTEXTS.get(key)
    if made is None:
        fresh = camera.make_cairo_context(np.zeros_like(array), centre)
        made = tuple(read(fresh) for _, read in _CONTEXT_STATE)
        _MADE_CONTEXTS[key] = made
    for (name, read), expected in zip(_CONTEXT_STATE, made, strict=True):
        value = read(context)
        if value != expected:
            return name, value
    return None


def unmodelled_state(camera: Any) -> tuple[Any, str, Any, str] | None:
    """The first state of a camera ``camera`` draws with (itself, or a
    display camera) that the core's drawers (the core, and Cairo's engine
    cameras for its overlay) do not draw as the camera would, as ``(camera,
    attribute, value, the value taken)``, or ``None``: a value of
    :data:`ASSUMED_STATE` other than the one taken, however it was set (a
    constructor of the scene's own, the scene's code); the Cairo context it
    keeps for its array, where its state is not as CE makes it (a clip
    installed, :func:`_context_changed`); and the camera's
    pixel size against the configured one, by whose ratio CE widens a point
    cloud's blocks (``adjusted_thickness``), which the core draws at the
    configured size's.
    """
    from arachne import config
    from arachne.timeline.camera_code import drawn_cameras

    missing = object()
    for each in drawn_cameras(camera):
        for name, taken, holds in ASSUMED_STATE:
            value = getattr(each, name, missing)
            if value is not missing and not holds(value):
                return each, name, value, taken
        changed = _context_changed(each)
        if changed is not None:
            name, value = changed
            return (
                each,
                f"Cairo context's {name}",
                value,
                "as get_cairo_context makes it",
            )
    size = camera.pixel_width + camera.pixel_height
    configured = config.pixel_width + config.pixel_height
    if size != configured:
        return camera, "pixel_width + pixel_height", size, str(configured)
    return None


def captured(camera: Any, draw_path: str) -> bool:
    """Whether frames of ``camera`` on ``draw_path`` are described from the
    camera's own capture: on a path that describes frames, a camera the
    scene's code changed (:func:`user_camera`) in ways XF carries. Its code
    then runs once per capture, as CE runs it, and the records hold what it
    produced (pre-projected records). Every frame of a scene that fell back
    to its camera's own draw (:func:`fall_back`) is too, whatever the camera
    has become since: the live painter that fell back draws it.
    """
    if not DRAW_PATHS.get(draw_path):
        return False
    return user_camera(camera) is not None or fell_back(camera, draw_path)


def camera_path(camera: Any, draw_path: str) -> str:
    """The draw path ``camera`` draws on when ``draw_path`` is asked for.

    A camera the scene's code changed (:func:`user_camera`: its own class, or
    a method of the engine's replaced on the class or the instance; the
    camera itself or a display camera it draws through) draws what its code
    makes. Where every change reaches the frame through what the camera's
    capture produces (:data:`CARRIED_MEMBERS`), frames are described from
    that capture (:func:`captured`) and stay on the path; on ``core``, which
    keeps no draw of the camera's own to fall back on mid-scene, only where
    the gate has proven each change (:data:`PROVEN_ON_THE_GATE`), the rest
    drawn on Cairo for the whole scene. Otherwise, on a path that describes
    frames, it draws on Cairo, with a ``render.fallback`` finding naming
    what changed and the member XF cannot carry. A camera
    that is no engine camera's subclass stays on the path and is
    ``render.unable`` there. On ``core``, a camera whose drawing state is not
    what the core's drawers take as given (:func:`unmodelled_state`), changed
    or not, draws on Cairo for the whole scene, with a ``render.fallback``
    finding.

    A scene's frames are drawn by one painter, one movie one painter
    (:func:`deliver`): its first drawing decides the path, and a scene on
    Cairo stays there. On the core a fallback to Cairo is possible only while
    no frame has been delivered; once the core has delivered one, a drawing
    that would take another path is ``render.unable``. (On ``ir_cairo`` Cairo
    draws every frame either way, so a fallback can come at any frame.) A
    scene that fell back to its camera's own draw (:func:`fall_back`) keeps
    its path, whatever the camera has become: every frame of it is drawn
    from the camera's captures (:func:`captured`) by the live painter that
    fell back, which draws them as the camera does.
    """
    draw_path = scene_path(camera, draw_path)
    route = _route(camera, draw_path)
    if route.fallen:
        assert route.path is not None, "a scene falls back from a path"
        return route.path
    wanted, fallback = _wanted_path(camera, draw_path)
    path = route.path
    if path is None or (wanted == "cairo" and route.painter != "core"):
        path = route.path = wanted
    elif wanted != path and path != "cairo":
        why = (
            fallback[1].replace(" is drawn on Cairo", " would be drawn on Cairo", 1)
            if fallback is not None
            else f"it would be drawn on {wanted}"
        )
        raise Unable(
            f"render.unable: {why}; but the scene's frames before were delivered"
            f" on {path}, and a scene's frames are drawn on one path"
        )
    if fallback is not None and path == wanted:
        raise_finding("render.fallback", *fallback)
    return path


def painter_kind(draw_path: str) -> str:
    """The painter whose pixels a frame drawn on ``draw_path`` delivers:
    ``core``, or ``cairo`` for ``cairo`` and ``ir_cairo`` alike (Cairo draws
    every pixel of both).
    """
    return "core" if draw_path == "core" else "cairo"


class _Route:
    """The path a scene's frames take, the painter of the frames delivered
    (:func:`painter_kind`; ``None`` before the first), and whether the scene
    fell back to its camera's own draw from its first frame
    (:func:`fall_back`).
    """

    __slots__ = ("fallen", "painter", "path")

    def __init__(self) -> None:
        self.path: str | None = None
        self.painter: str | None = None
        self.fallen = False


#: Per scene camera in this process: its routes, by the draw path asked for
#: (a session drawing a record on another path draws another picture).
_ROUTES: weakref.WeakKeyDictionary[Any, tuple[int, dict[str, _Route]]] = (
    weakref.WeakKeyDictionary()
)


def _route(camera: Any, draw_path: str) -> _Route:
    """The route of the frames the scene of ``camera`` draws, asked for on
    ``draw_path``.
    """
    pid = os.getpid()
    found = _ROUTES.get(camera)
    if found is None or found[0] != pid:
        found = (pid, {})
        _ROUTES[camera] = found
    return found[1].setdefault(scene_path(camera, draw_path), _Route())


#: Within :func:`deliveries`: the painter of each frame delivered in this
#: process, in order.
_DELIVERIES: list[str] | None = None


def deliver(camera: Any, draw_path: str, painter: str) -> None:
    """A frame of the scene ``camera`` draws, asked for on ``draw_path``, has
    been delivered, its pixels drawn by ``painter`` (:func:`painter_kind`; a
    play's static picture is not a frame: it is drawn under the play's
    frames). One movie, one painter: a frame drawn by another painter than
    the scene's frames before is ``render.unable``. From now on the scene's
    frames stay on the path they took (:func:`camera_path`).
    """
    route = _route(camera, draw_path)
    if route.painter is None:
        route.painter = painter
    elif painter != route.painter:
        raise Unable(
            f"render.unable: a frame drawn by {painter} after the scene's frames"
            f" before were drawn by {route.painter}, and a scene's frames are"
            " drawn by one painter"
        )
    if _DELIVERIES is not None:
        _DELIVERIES.append(painter)


@contextlib.contextmanager
def deliveries() -> Iterator[list[str]]:
    """Within: the painter of each frame delivered in this process
    (:func:`deliver`), in order, in the list yielded.
    """
    global _DELIVERIES
    held = _DELIVERIES
    found: list[str] = []
    _DELIVERIES = found
    try:
        yield found
    finally:
        _DELIVERIES = held


def fall_back(camera: Any, draw_path: str) -> None:
    """The scene ``camera`` draws, asked for on ``draw_path``, fell back to
    its camera's own draw from its first frame, which stands as the camera
    drew it (``live.LivePainter``): every later frame is the camera's own
    draw too, whatever the camera becomes: the live painter that fell back
    draws every later frame (:func:`camera_path`, :func:`captured`).
    """
    _route(camera, draw_path).fallen = True


def fell_back(camera: Any, draw_path: str) -> bool:
    """Whether the scene ``camera`` draws, asked for on ``draw_path``, fell
    back to its camera's own draw (:func:`fall_back`).
    """
    return _route(camera, draw_path).fallen


def delivered(camera: Any, draw_path: str) -> bool:
    """Whether a frame of the scene ``camera`` draws, asked for on
    ``draw_path``, has been delivered (:func:`deliver`).
    """
    return _route(camera, draw_path).painter is not None


def _wanted_path(camera: Any, draw_path: str) -> tuple[str, tuple[str, str] | None]:
    """The path :func:`camera_path` takes for ``camera`` on ``draw_path`` (a
    path, not ``auto``) as the camera is now, before the scene's route: the
    path and, for a fallback, the ``render.fallback`` finding's subject and
    message.
    """
    if draw_path == "core":
        unmodelled = unmodelled_state(camera)
        if unmodelled is not None:
            each, name, value, taken = unmodelled
            own = user_camera(camera)
            subject, why = own if own is not None else (type(each).__name__, "")
            why = why.rstrip(",") or f"a {type(camera).__name__}"
            whose = (
                "its"
                if each is camera
                else f"its display camera's ({type(each).__name__})"
            )
            return "cairo", (
                subject,
                f"{why} is drawn on Cairo for the whole scene: {whose} {name} is"
                f" {value!r}, which XF does not carry and the core's drawers"
                f" take to be {taken}",
            )
    own = user_camera(camera) if DRAW_PATHS[draw_path] else None
    if own is None:
        return draw_path, None
    found = uncarried(camera)
    subject, why = own
    if found is None and draw_path == "core":
        unproven = unproven_on_core(camera)
        if unproven is None:
            return draw_path, None
        form, member = unproven
        return "cairo", (
            subject,
            f"{why} is drawn on Cairo for the whole scene: no gate has drawn a"
            f" camera with {member} changed ({form}) on ir_cairo against its own"
            " draw, and the core keeps no draw of the camera's own to fall back on",
        )
    if found is None:
        return draw_path, None
    _, member = found
    return "cairo", (
        subject,
        f"{why} is drawn on Cairo: {member} reaches the frame other than through"
        " the points, colours and order XF carries of the camera's capture",
    )


def scene_path(camera: Any, draw_path: str) -> str:
    """The draw path ``draw_path`` names for a scene drawn by ``camera``:
    ``auto`` is ``core`` for a 3D scene (its camera a ``ThreeDCamera``) and
    ``cairo`` for every other; any other path is itself.
    """
    if draw_path == "auto":
        from arachne.camera.three_d_camera import ThreeDCamera

        return "core" if isinstance(camera, ThreeDCamera) else "cairo"
    if draw_path not in DRAW_PATHS:
        raise ValueError(
            f"draw_path {draw_path!r} is not one of {sorted([*DRAW_PATHS, 'auto'])}"
        )
    return draw_path


class IrCairoPainter:
    """Frames through XF, drawn by Cairo from the stream alone.

    Every delta goes through XF's bytes: the core parses them and applies
    them to its ``Stream`` (never drawing there), so a delta the core would
    refuse is an error here, and Cairo draws from the records the core read
    back. ``keep`` keeps every delta sent (``deltas``), for tests; by default
    the class's ``keep_all``.
    """

    keep_all = False
    #: Frames drawn from XF by every painter in this process.
    drawn_total = 0

    def __init__(
        self,
        *,
        keep: bool | None = None,
        stream: FrameStream | None = None,
        canonical: bool = False,
    ) -> None:
        try:
            import arachne_core
        except ImportError as error:  # pragma: no cover - environment
            raise ImportError(
                "draw_path ir_cairo needs arachne_core: install the 'core' extra"
            ) from error
        self.core: Any = arachne_core.Stream()
        self.stream = FrameStream() if stream is None else stream
        self.state = StreamState()
        self.drawer = CairoDrawer()
        self.keep = self.keep_all if keep is None else keep
        #: With ``canonical``, the deltas' geometry snapped as the core's is
        #: (:mod:`.canonical`): a picture drawn on the core path beside the
        #: core's frames (a display's, drawn as the scene runs).
        self.canonical = Canonical(planes=False) if canonical else None
        self.deltas: list[Delta] = []
        self.last: Delta | None = None
        #: Frames this painter drew.
        self.frames = 0

    def paint(
        self,
        camera: Any,
        groups: Sequence[tuple[str, Sequence[Any]]],
        *,
        t: Fraction = Fraction(0),
        terms: Mapping[int, Term] | None = None,
    ) -> np.ndarray:
        """The frame the camera draws from ``groups`` (each a layer and the
        list one capture is handed), from its XF description alone.
        """
        return self.draw(self.stream.frame(camera, groups, t=t, terms=terms))

    def draw(self, delta: Delta) -> np.ndarray:
        """The frame ``delta`` brings the stream to, drawn from XF alone."""
        import arachne_core

        # Through the wire: the bytes, parsed back by the core, are what the
        # core applies and what Cairo draws from.
        if self.canonical is not None:
            delta = self.canonical(delta)
        read = arachne_core.FrameDelta.from_bytes(delta.to_core().to_bytes())
        self.core.apply(read)
        self.state.apply(Delta.from_contents(read.contents()))
        self.last = delta
        self.frames += 1
        IrCairoPainter.drawn_total += 1
        if self.keep:
            self.deltas.append(delta)
        return self.drawer.draw(self.state).copy()

    def settle(self) -> None:
        """Leave each display's camera of the last frame holding the picture
        CE's last capture leaves in it (what a display shows once its camera
        has left the MultiCamera, drawn as the scene runs).
        """
        for display, (_, view, order) in self.stream.last_sub_frames:
            camera = display.camera
            camera.pixel_array = self.drawer.capture(
                self.state, view, order, camera.background
            )


_PAINTERS: weakref.WeakKeyDictionary[Any, tuple[int, str, Any]] = (
    weakref.WeakKeyDictionary()
)


def painter_for(owner: Any, draw_path: str, role: str = "frames") -> Any:
    """The painter of ``draw_path`` for ``owner`` (a record) in this process,
    made on first use; ``None`` on ``cairo``. ``role`` keeps apart streams
    that draw different things (a play's static pictures and its frames).
    """
    if draw_path not in DRAW_PATHS:
        raise ValueError(f"draw_path {draw_path!r} is not one of {sorted(DRAW_PATHS)}")
    if not DRAW_PATHS[draw_path]:
        return None
    painters = _PAINTERS.get(owner)
    pid = os.getpid()
    if painters is None or painters[0] != pid or painters[1] != draw_path:
        painters = (pid, draw_path, {})
        _PAINTERS[owner] = painters
    found = painters[2].get(role)
    if found is None:
        found = painters[2][role] = painter_class(draw_path)()
    return found


def painter_class(draw_path: str) -> type:
    """The painter class of a draw path that describes frames in XF."""
    if draw_path == "core":
        from .core_painter import CorePainter

        return CorePainter
    return IrCairoPainter
