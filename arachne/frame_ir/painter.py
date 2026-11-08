"""The painter seam: how a shown frame becomes pixels, by draw path.

``Drawing`` hands a frame to the painter its draw path names (the option
``draw_path``). On ``cairo`` there is none: the camera draws the shown
mobjects itself, as it always has. On ``ir_cairo`` the frame is turned into an
XF delta (:class:`~.extract.FrameStream`), the core checks the delta whole
(``arachne_core``, the ``core`` extra), and Cairo draws the frame from the XF
stream alone (:class:`~.cairo_painter.CairoDrawer`). Frames drawn as the scene
runs (``LiveFrames``, ``update_frame``) take the same path through
:func:`~.live.live`, described from what the camera's own capture consumed.

A painter and its stream are made in the process that draws, after any fork:
:func:`painter_for` keeps one per record and process.
"""

from __future__ import annotations

__all__ = ["IrCairoPainter", "camera_path", "painter_for"]

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

#: The draw paths, and whether each describes frames in XF.
DRAW_PATHS = {"cairo": False, "ir_cairo": True}


def user_camera(camera: Any) -> tuple[str, str] | None:
    """What of ``camera``, or of a display camera it draws through
    (:func:`~arachne.timeline.camera_code.drawn_cameras`), is the scene's own
    code, as ``(subject, why)``: a method of an engine camera class replaced
    on the class (:func:`~arachne.timeline.camera_code.replaced_methods`); a
    camera's class, a subclass of one of the engine's cameras defined
    outside the engine; a method of its class replaced on the instance; or
    an entry of its ``display_funcs``, through which it draws each kind of
    mobject, replaced. ``None`` for engine cameras as the engine made them,
    or for no camera at all.
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
    displays = _replaced_displays(camera)
    if displays:
        return (
            displays,
            f"{displays}, replaced in the camera's display_funcs by the scene's code,",
        )
    return None


#: The camera's display function for each kind of mobject, as
#: ``Camera.__init__`` makes them: the method of that name, bound to the
#: camera; the do-nothing function for any other mobject (``None``).
_DISPLAY_METHODS = {
    "VMobject": "display_multiple_vectorized_mobjects",
    "PMobject": "display_multiple_point_cloud_mobjects",
    "AbstractImageMobject": "display_multiple_image_mobjects",
    "Mobject": None,
}


def _replaced_displays(camera: Any) -> str:
    """The entries of ``camera.display_funcs`` that are not the engine's own,
    named by kind (``display_funcs`` itself when its kinds are not the
    engine's); empty when every one is. A live painter's own wrappers
    (:mod:`.live`), installed while it describes the camera's captures,
    stand over entries checked before it installed them.
    """
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.image_mobject import AbstractImageMobject
    from arachne.mobject.types.point_cloud_mobject import PMobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    own = vars(camera)
    if "_xf_live" in own:
        return ""
    funcs = own.get("display_funcs")
    kinds = {
        kind.__name__: kind
        for kind in (VMobject, PMobject, AbstractImageMobject, Mobject)
    }
    if not isinstance(funcs, dict) or set(funcs) != set(kinds.values()):
        return "display_funcs"
    kind = type(camera)
    changed = []
    for name, method in _DISPLAY_METHODS.items():
        func = funcs[kinds[name]]
        if method is None:
            # The do-nothing lambda Camera.__init__ makes.
            engine = (
                getattr(func, "__qualname__", None)
                == "Camera.__init__.<locals>.<lambda>"
                and getattr(func, "__module__", None) == Camera.__module__
            )
        else:
            # Bound to this camera, as its class has it: one bound to another
            # camera (a shallow copy shares the original's dictionary) draws
            # with that camera's settings.
            engine = getattr(func, "__self__", None) is camera and getattr(
                func, "__func__", None
            ) is getattr(kind, method, None)
        if not engine:
            changed.append(f"display_funcs[{name}]")
    return ", ".join(changed)


def camera_path(camera: Any, draw_path: str) -> str:
    """The draw path ``camera`` draws on when ``draw_path`` is asked for. A
    camera the scene's code changed (:func:`user_camera`: its own class, or
    a method of the engine's replaced on the class or the instance; the
    camera itself or a display camera it draws through) can
    change how anything is drawn, which XF 1 does not describe: on a path
    that describes frames it draws on Cairo, with a ``render.fallback``
    finding naming what changed. (A record refuses a camera class of the
    scene's own, ``camera.unsupported``, so that is met as the scene runs; a
    replaced method can be met on the record too.) A camera that is no
    engine camera's subclass stays on the path and is ``render.unable``
    there.
    """
    if draw_path not in DRAW_PATHS:
        raise ValueError(f"draw_path {draw_path!r} is not one of {sorted(DRAW_PATHS)}")
    own = user_camera(camera) if DRAW_PATHS[draw_path] else None
    if own is None:
        return draw_path
    subject, why = own
    raise_finding(
        "render.fallback",
        subject,
        f"{why} is drawn on Cairo: XF 1 describes the engine's cameras only",
    )
    return "cairo"


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


def painter_for(
    owner: Any, draw_path: str, role: str = "frames"
) -> IrCairoPainter | None:
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
        found = painters[2][role] = IrCairoPainter()
    return found
