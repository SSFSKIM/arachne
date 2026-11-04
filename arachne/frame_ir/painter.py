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
    "CARRIED_MEMBERS",
    "IrCairoPainter",
    "camera_path",
    "captured",
    "changed_members",
    "painter_for",
    "scene_path",
    "uncarried",
]

import inspect
import os
import weakref
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from .cairo_painter import CairoDrawer
from .extract import FrameStream, Term
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
    from arachne.timeline.camera_code import replaced_methods
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
        "__firstlineno__",
        "__static_attributes__",
        "__classcell__",
        "__orig_bases__",
        "__parameters__",
    }
)


def changed_members(camera: Any) -> tuple[str, ...]:
    """The engine camera members the scene's code changed on ``camera``: the
    names its own classes define over an engine camera's, and the methods
    replaced on the instance or on an engine class
    (:func:`~arachne.timeline.camera_code.replaced_methods`).
    """
    from arachne.camera.camera import Camera
    from arachne.timeline.camera_code import replaced_methods
    from arachne.timeline.frozen import engine_class

    if not isinstance(camera, Camera):
        return ()
    kind = type(camera)
    base = next(k for k in kind.__mro__ if engine_class(k))
    names = {
        name
        for klass in kind.__mro__
        if klass is not object and not engine_class(klass)
        for name in vars(klass)
        if name not in _CLASS_BOOKKEEPING and hasattr(base, name)
    }
    names.update(r.name for r in replaced_methods(camera))
    return tuple(sorted(names))


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


def captured(camera: Any, draw_path: str) -> bool:
    """Whether frames of ``camera`` on ``draw_path`` are described from the
    camera's own capture: on a path that describes frames, a camera the
    scene's code changed (:func:`user_camera`) in ways XF carries. Its code
    then runs once per capture, as CE runs it, and the records hold what it
    produced (pre-projected records).
    """
    return bool(DRAW_PATHS.get(draw_path)) and user_camera(camera) is not None


def camera_path(camera: Any, draw_path: str) -> str:
    """The draw path ``camera`` draws on when ``draw_path`` is asked for.

    A camera the scene's code changed (:func:`user_camera`: its own class, or
    a method of the engine's replaced on the class or the instance; the
    camera itself or a display camera it draws through) draws what its code
    makes. Where every change reaches the frame through what the camera's
    capture produces (:data:`CARRIED_MEMBERS`), frames are described from
    that capture (:func:`captured`) and stay on the path. Otherwise, on a
    path that describes frames, it draws on Cairo, with a ``render.fallback``
    finding naming what changed and the member XF cannot carry. A camera
    that is no engine camera's subclass stays on the path and is
    ``render.unable`` there.
    """
    draw_path = scene_path(camera, draw_path)
    own = user_camera(camera) if DRAW_PATHS[draw_path] else None
    if own is None:
        return draw_path
    found = uncarried(camera)
    subject, why = own
    if found is None and draw_path == "core":
        raise_finding(
            "render.fallback",
            subject,
            f"{why} is drawn on Cairo: the core does not draw what a camera's"
            " own capture placed yet",
        )
        return "cairo"
    if found is None:
        return draw_path
    _, member = found
    raise_finding(
        "render.fallback",
        subject,
        f"{why} is drawn on Cairo: {member} reaches the frame other than through"
        " the points, colours and order XF carries of the camera's capture",
    )
    return "cairo"


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
        self, *, keep: bool | None = None, stream: FrameStream | None = None
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
