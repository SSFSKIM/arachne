"""Draw reads: every read made while a frame is drawn from the record,
checked against what the record holds for that frame.

A frame drawn from the record (``Drawing.sample``, ``static``, ``final``,
``restored``, and a restored frame on Cairo) is drawn after its play ran, so
a value the drawing reads live that the record does not hold is drawn as it
is now, where CE draws it as it was at the frame. Each read the drawing code
makes of a mobject, a camera, the configuration, the camera's table of
display functions or a file must be one of:

- a field of a mobject's state as the record gave it for the frame's
  capture (a key of its ``__dict__``, present or absent), the mobject still
  holding it, and a value nothing can change in place since: a plain value,
  a read-only array, a mobject (checked as any), or a type of
  :data:`IMMUTABLE_TYPES`;
- a keyed camera field: a camera input the capture recorded, holding the
  recorded value, or one the record installs from them;
- an entry of a reviewed list below, with why the record need not hold it.

Anything else is an unlisted read, reported with the type of the object,
the attribute and the line that read it (and the nearest line of the
camera's, a painter's or the record's code above it).

How reads are seen. While a frame is drawn, and only then, ``Mobject``,
``Camera`` and ``ManimConfig`` get a ``__getattribute__`` that does what
``object.__getattribute__`` does and notes the read; ``ManimConfig`` also gets
a noting ``__getitem__``, ``Mobject`` a ``__setattr__`` and ``__delattr__``
that note what the drawing itself writes, and each drawn camera's
``display_funcs`` is replaced by a dict that notes each lookup. A plain read,
``getattr``, ``hasattr``, a property (its name, then every read its getter
makes), ``vars()`` and ``__dict__`` all go through ``__getattribute__``; a
read of ``__dict__`` itself is a read of the whole state, so it holds only
where the whole state is the recorded one. Each hook returns or raises what
the original did, and all are removed when the frame is done, so nothing the
engine does changes. ``sys.setprofile`` and ``sys.settrace`` see calls and
lines, not attribute reads, so neither can do this; an audit hook sees the
files the drawing opens, and only that is what one is used for.

What the record's own code reads (giving mobjects and the camera their
recorded states, the route's evaluation of an instant between captures) is
not a drawing read of the scene: those functions are the :data:`SITES`. A
frame described in XF is drawn by the Cairo drawer from its description
alone, through proxies and a camera of its own; what it reads of those is
checked against what the description gives them (:data:`DRAWER_PROXIES`,
:data:`DRAWER_CAMERA`).
"""

from __future__ import annotations

import enum
import functools
import importlib
import os
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

#: Set to trace every frame a test draws from the record and fail the test on
#: an unlisted read.
ENV = "ARACHNE_DRAW_READS"

_ENGINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENGINE_ROOT = os.path.join(_ENGINE, "arachne") + os.sep
_CONFIG_CODE = os.path.join(_ENGINE_ROOT, "_config") + os.sep
_HERE = os.path.abspath(__file__)

# ── the reviewed lists ───────────────────────────────────────────────────────

#: Camera attributes a draw reads that the record neither holds nor installs,
#: and why it need not.
CAMERA_CONSTANTS: dict[str, str] = {
    "pixel_array": (
        "What the camera draws into: every frame starts the scene's camera afresh, "
        "from its background or the play's static picture, before drawing; a "
        "MultiCamera draws into its displays' cameras before it draws the displays."
    ),
    "pixel_width": (
        "The pixel shape the frame is drawn at, the drawing's own: a session draws "
        "at a quality of its own."
    ),
    "pixel_height": "As ``pixel_width``.",
    "pixel_array_to_cairo_context": (
        "A cache of the Cairo context made for a pixel array, made and kept at the "
        "same first draw into that array whether the frame is drawn live or from "
        "the record."
    ),
    "pixel_array_to_mapping_centre": (
        "The frame centre each cached context was made with, a recorded input when "
        "it was made."
    ),
    "background_colored_vmobject_displayer": (
        "The camera's displayer of image-filled mobjects, made at its first use; it "
        "reads each image file once by name and keeps it, the same whether the frame "
        "is drawn live or from the record. A scene that rewrites an image file it "
        "names during a play is not a supported use."
    ),
    "display_funcs": (
        "The table of display functions, made with the camera; each entry a draw "
        "looks up is checked to be the camera's own engine method."
    ),
    "__dict__": (
        "Read to look up one installed field (``image_mobjects_from_cameras``, by "
        "the frame's extraction) or the camera's own mobjects (whose states are "
        "checked as any mobject's)."
    ),
}

#: Camera attributes a draw reads that are caches of recorded inputs: each must
#: hold what it is computed from them now.
CAMERA_DERIVED: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "rotation_matrix": (
        "A ThreeDCamera's rotation, computed from its trackers (recorded mobjects).",
        lambda camera: camera.generate_rotation_matrix(),
    ),
}

#: An attribute of a camera that holds one of its own mobjects.
CAMERA_OWN_MOBJECTS = (
    "A camera's own mobject (a tracker, the frame centre, the light, a moving "
    "camera's frame): keyed, the one the capture recorded bound there "
    "(``CameraExtras.bindings``, installed with the camera's inputs), whose "
    "state is recorded (``camera_nodes``) and checked as any mobject's. A play "
    "during which one is rebound is refused (``camera.rebound``)."
)

#: An attribute a camera does not have.
CAMERA_ABSENT = (
    "Neither the camera nor its class holds the name (a ``hasattr`` or a "
    "``getattr`` with a default): a fact of the camera's class."
)

#: A camera method replaced on the camera instance.
CAMERA_REPLACED = (
    "A method of the camera replaced on its instance or on its engine class (by the "
    "scene, or by a test counting what is drawn): the scene's own code, which the record finds and "
    "reports (``Record.camera_replaced``) and does not vouch for. What it calls of "
    "the engine's is traced as any read is."
)

#: An entry of a camera's table of display functions replaced by the scene.
DISPLAY_REPLACED = (
    "An entry of the camera's ``display_funcs`` the scene replaced: the scene's own code, "
    "which the record finds and reports (``Record.camera_replaced``, form "
    "``display_funcs``) and does not vouch for, as for a replaced method "
    "(:data:`CAMERA_REPLACED`). Exempt are the lookup of the reported entry and the reads "
    "of the camera it is bound to when that is another camera (the scene drew through that "
    "camera's settings); what else the scene's function reads is traced as any read is."
)

#: Configuration options a draw reads that are keyed: each read holds only at
#: a capture before the first that found the option changed.
CONFIG_KEYED: dict[str, str] = {
    "pixel_width": (
        "The configured pixel shape a point's thickness is scaled by "
        "(``Camera.adjusted_thickness``): the drawing's own quality, as the scene's "
        "was while the record found it as its first capture did "
        "(``Record.pixel_shape_changed``); from the first capture that finds "
        "another, the record refuses (``config.pixel_shape``)."
    ),
    "pixel_height": "As ``pixel_width``.",
}

#: Class attributes of the engine's classes that are not code (functions,
#: properties, descriptors), by ``(class, name)``, and why the record need not
#: hold them. Code is the engine's own, as it was when the engine was imported
#: (``frozen.class_members``); a scene that replaces it patches the engine,
#: which the record does not vouch for.
CLASS_CONSTANTS: dict[tuple[str, str], str] = {}

#: Mobject attributes outside ``__dict__`` that a draw reads.
MOBJECT_CONSTANTS: dict[str, str] = {
    "__class__": (
        "The type is not part of a recorded state, and no engine code assigns "
        "``__class__``: a mobject's class is the one it was made with."
    ),
}

#: Types of a value a drawing reads off a recorded state, beyond the plain
#: ones, that nothing can change in place, and why.
IMMUTABLE_TYPES: dict[str, str] = {
    "arachne.mobject.values.Paint": (
        "A vectorized mobject's style as one immutable value, its arrays frozen."
    ),
    "arachne.timeline.record.DepthReference": (
        "A depth reference point the record made at a capture, its centre a "
        "read-only array."
    ),
}

#: A value of a recorded state whose drawing inputs the record keeps apart.
KEPT_APART = (
    "A display's camera, whose picture the record keeps per play "
    "(``Capture.pictures``) or whose inputs it records per capture "
    "(``CameraExtras.sub_cameras``); or a fill's image, of which the drawing "
    "gives each camera its read under the name the image had at the capture "
    "(``Record.image_read``)."
)

#: Mobjects the drawing makes for itself, by type, and why reading them reads
#: no state of the scene's.
SCRATCH_TYPES: dict[str, str] = {
    "arachne.frame_ir.geometry._Path": (
        "A bare path the frame's description makes from points it was given, to "
        "compute a reveal window with the engine's own function "
        "(``geometry.revealed``); the points were read, and checked, as a term's."
    ),
}

#: The Cairo drawer's proxies of described mobjects, by type: each read
#: of one must be of its own fields or of engine code.
DRAWER_PROXIES: dict[str, str] = {
    f"arachne.frame_ir.cairo_painter.{name}": (
        "What the camera reads off a described mobject, built by the drawer from "
        "one record's fields of the frame's description (``CairoDrawer._make``) "
        "and holding nothing else."
    )
    for name in ("_PathProxy", "_PointsProxy", "_ImageProxy")
}

#: The Cairo drawer's cameras, by type.
DRAWER_CAMERA_TYPES = frozenset(
    f"arachne.frame_ir.cairo_painter.{name}"
    for name in ("_PainterCamera", "_Painter3DCamera")
)

_FROM_VIEW = (
    "Set from the frame's view (XF's camera state) by ``CairoDrawer._camera`` "
    "before each frame is drawn."
)
_ENGINE_DEFAULT = (
    "The engine's default, which the drawer's camera keeps: a frame of a camera "
    "holding another is not described (``render.unable``, ``FrameStream._view_of``)."
)

#: Attributes of the Cairo drawer's camera a draw reads, and where each one's
#: value comes from. Its construction reads configuration defaults
#: (``CairoDrawer._camera`` is a site); an attribute a draw reads that is not
#: here is one the description does not give.
DRAWER_CAMERA: dict[str, str] = {
    **dict.fromkeys(
        (
            "pixel_width",
            "pixel_height",
            "frame_width",
            "frame_height",
            "frame_center",
            "_frame_center",
            "_xf_centre",
            "cairo_line_width_multiple",
            "exponential_projection",
            "should_apply_shading",
            "light_source",
            "phi_tracker",
            "theta_tracker",
            "gamma_tracker",
            "focal_distance_tracker",
            "zoom_tracker",
        ),
        _FROM_VIEW,
    ),
    "fixed_in_frame_mobjects": (
        "The described mobjects in frame space, from each record's ``space`` "
        "(``CairoDrawer._orient``)."
    ),
    "fixed_orientation_mobjects": (
        "Each described mobject's fixed-orientation centre, from its record "
        "(``CairoDrawer._orient``)."
    ),
    "pixel_array": "The drawer's array, reset to the view's background each frame.",
    "__dict__": "Read for the drawer's own cache of Cairo contexts (``_context``).",
    "background_colored_vmobject_displayer": (
        "The drawer camera's displayer of image-filled paths, given the "
        "description's images by name (``CairoDrawer._seed_images``)."
    ),
    "display_funcs": "The drawer camera's own table of its engine methods.",
    **dict.fromkeys(
        ("pixel_array_dtype", "image_mode", "rgb_max_val"), _ENGINE_DEFAULT
    ),
}

#: A mobject a trusted play's term is made of.
TERM_ENDS = (
    "The start or end mobject of a trusted play's term (``PlayRecord.leaf_terms``, "
    "from the animation's recorded state): a leaf is described as their blend or "
    "reveal only when it gives the leaf's recorded points and paint bit for bit "
    "(``FrameStream._blend_record``, ``_reveal_record``), else as its own state."
)

#: Files a draw opens, by extension, and why the record need not hold them.
FILES: dict[str, str] = {}

#: Functions whose reads are not drawing reads of the scene, by
#: ``module:qualname``, and why.
SITES: dict[str, str] = {
    "arachne.timeline.draw:Drawing.show": (
        "Walks the record from the frame's roots, saves each reached mobject's "
        "live state and installs its recorded one; what it leaves is the frame's "
        "state."
    ),
    "arachne.timeline.draw:Drawing.show_camera": "Installs the camera's recorded inputs.",
    "arachne.timeline.draw:Drawing._install_camera": (
        "Saves the camera's live inputs and installs the recorded ones."
    ),
    "arachne.timeline.draw:camera_nodes": (
        "Finds the camera's mobjects, the roots the record walks with the scene's."
    ),
    "arachne.timeline.record:PlayRecord.leaf_terms": (
        "Evaluates a trusted play's terms from its animations' recorded states "
        "(see TERM_ENDS for what is read of them after)."
    ),
    "arachne.frame_ir.cairo_painter:CairoDrawer._camera": (
        "Builds the drawer's camera for a view, or takes the one it built, and "
        "gives it the view's inputs: what a draw then reads of it is checked "
        "against DRAWER_CAMERA."
    ),
}

#: A frame's ``prepare`` (the route's evaluation of an instant between
#: captures, or a restore) is a site too: what it leaves the frame's mobjects
#: holding, those it made included, is the frame's state.

# ── the tracer ───────────────────────────────────────────────────────────────


class Read(NamedTuple):
    """An unlisted read: the type of the object read, the attribute, the
    line that read it, and why it is not one the record holds.
    """

    owner: str
    attribute: str
    site: str
    why: str

    def __str__(self) -> str:
        return f"{self.owner}.{self.attribute} at {self.site}: {self.why}"


@dataclass
class _Given:
    """A state the record gave a mobject for a frame: the recorded version
    (``None`` for what ``prepare`` left), and what the record's code left it
    holding (``None`` until that code returns).
    """

    mob: Any
    recorded: Any
    prepared: bool
    held: Any = None


@dataclass
class _Frame:
    """One frame being drawn from the record; one within another (what a
    play's static picture painted, read at its own capture) shares the
    states the outer one gave.
    """

    drawing: Any
    capture: Any
    given: dict[int, _Given] = field(default_factory=dict)
    #: Per mobject (by id): the attributes the drawing set on it since the
    #: record gave its state; what it reads of them it computed from reads
    #: checked as it made them.
    written: dict[int, set[str]] = field(default_factory=dict)
    #: Per mobject (by id): whether it holds the state given, as last found.
    holds: dict[int, bool] = field(default_factory=dict)
    #: Each camera's recorded inputs at the capture, by id.
    cameras: dict[int, Any] = field(default_factory=dict)
    #: The mobjects (by id) a trusted play's terms are made of.
    terms: set[int] = field(default_factory=set)


class Tracer:
    """Traces the frames drawn from the record while it is installed."""

    def __init__(self) -> None:
        self.found: dict[Read, None] = {}
        self.frames: list[_Frame] = []
        #: How many frames were traced (for the tests of this tracer).
        self.traced = 0
        self._site_depth = 0
        self._preparing = 0
        self._busy = False
        self._patches: list[tuple[Any, str, Any]] = []
        self._hooks: list[tuple[type, str, Any]] = []
        #: The hooks as the engine's own overrides while installed, so that
        #: the camera-code detector does not read them as the scene's.
        self._installed = ExitStack()
        self._tables: list[tuple[Any, Any]] = []
        #: The mobjects of the drawer's cameras (by id) read in the frame
        #: being drawn: each set from the view (DRAWER_CAMERA).
        self._drawer_owned: dict[int, Any] = {}

    def take(self) -> list[Read]:
        """The unlisted reads found since last taken."""
        found = list(self.found)
        self.found.clear()
        return found

    # Installing

    def install(self) -> None:
        from arachne.timeline import draw, routes
        from arachne.timeline.record import NodeState, PlayRecord

        def capture_arg(
            drawing: Any, play: Any, capture: Any, *a: Any, **k: Any
        ) -> Any:
            return capture

        self._wrap_frame(draw.Drawing, "sample", capture_arg, prepare_at=None)
        self._wrap_frame(draw.Drawing, "restored", capture_arg, prepare_at=3)
        self._wrap_frame(
            draw.Drawing,
            "static",
            lambda drawing, play, *a, **k: drawing.record._captures[play.static_seq],
        )
        self._wrap_frame(
            draw.Drawing, "final", lambda drawing, *a, **k: drawing.record.final
        )
        self._wrap_frame(draw.Drawing, "_painted", capture_arg)
        self._wrap_frame(
            routes,
            "_restored_frame",
            lambda drawing, resolved, *a, **k: resolved.capture,
            prepare_at="resolved",
        )
        for site in SITES:
            module_name, qualname = site.split(":")
            owner: Any = importlib.import_module(module_name)
            *path, name = qualname.split(".")
            for part in path:
                owner = getattr(owner, part)
            self._wrap_site(owner, name)
        self._wrap_install(NodeState)
        self._wrap_terms(PlayRecord)
        sys.addaudithook(self._audit)

    def uninstall(self) -> None:
        """Put back everything :meth:`install` wrapped. (Its audit hook stays,
        as every audit hook does, and sees nothing once no frame is traced.)
        """
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def _patch(self, owner: Any, name: str, value: Any) -> None:
        original = (
            owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        )
        self._patches.append((owner, name, original))
        setattr(owner, name, value)

    def _wrap_frame(
        self,
        owner: Any,
        name: str,
        capture_of: Callable[..., Any],
        *,
        prepare_at: int | str | None = None,
    ) -> None:
        original = getattr(owner, name)
        tracer = self

        def traced(*args: Any, **kwargs: Any) -> Any:
            drawing = args[0]
            capture = capture_of(*args, **kwargs)
            if kwargs.get("prepare") is not None:
                kwargs["prepare"] = tracer._prepared(kwargs["prepare"])
            elif isinstance(prepare_at, int) and len(args) > prepare_at:
                args = (
                    *args[:prepare_at],
                    tracer._prepared(args[prepare_at]),
                    *args[prepare_at + 1 :],
                )
            elif prepare_at == "resolved":
                resolved = args[1]
                args = (
                    args[0],
                    resolved._replace(prepare=tracer._prepared(resolved.prepare)),
                )
            with tracer._frame(drawing, capture):
                return original(*args, **kwargs)

        traced.__wrapped__ = original  # type: ignore[attr-defined]
        self._patch(owner, name, traced)

    def _wrap_site(self, owner: Any, name: str) -> None:
        original = getattr(owner, name)
        tracer = self

        def site(*args: Any, **kwargs: Any) -> Any:
            tracer._site_depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                if tracer._site_depth == 1 and tracer.frames:
                    tracer._settle(tracer.frames[-1])
                tracer._site_depth -= 1

        site.__wrapped__ = original  # type: ignore[attr-defined]
        self._patch(owner, name, site)

    def _wrap_install(self, node_state: type) -> None:
        original = node_state.install
        tracer = self

        def install(state: Any, mob: Any) -> None:
            original(state, mob)
            if tracer.frames:
                frame = tracer.frames[-1]
                frame.given[id(mob)] = _Given(mob, state, tracer._preparing > 0)
                frame.written.pop(id(mob), None)
                for each in tracer.frames:
                    each.holds.pop(id(mob), None)

        self._patch(node_state, "install", install)

    def _wrap_terms(self, play_record: type) -> None:
        original = play_record.leaf_terms  # the site's wrapper, installed above
        tracer = self

        def leaf_terms(play: Any, t: Any) -> Any:
            found = original(play, t)
            if tracer.frames:
                terms = tracer.frames[-1].terms
                tracer._busy = True
                try:
                    for term in found.values():
                        for end in (
                            getattr(term, "start", None),
                            getattr(term, "end", None),
                        ):
                            if end is not None:
                                terms.update(id(mob) for mob in _live_family([end]))
                finally:
                    tracer._busy = False
            return found

        self._patch(play_record, "leaf_terms", leaf_terms)

    def _settle(self, frame: _Frame) -> None:
        """Take what each mobject the record just gave a state holds now."""
        from arachne.timeline.record import NodeState

        for given in frame.given.values():
            if given.held is None:
                given.held = NodeState.of(given.mob)

    def _prepared(self, prepare: Callable[[Any], None]) -> Callable[[Any], None]:
        """``prepare`` as a site: what it leaves each of the frame's mobjects
        holding, those it made included, is the frame's state of it.
        """
        from arachne.timeline.draw import camera_nodes
        from arachne.timeline.record import NodeState

        tracer = self

        def prepared(drawing: Any) -> None:
            tracer._site_depth += 1
            tracer._preparing += 1
            try:
                prepare(drawing)
                frame = tracer.frames[-1]
                capture = frame.capture
                roots = [
                    *capture.roots,
                    *capture.foreground,
                    *capture.moving,
                    *camera_nodes(drawing.camera),
                ]
                for mob in _live_family(roots):
                    frame.given[id(mob)] = _Given(mob, None, True, NodeState.of(mob))
                    frame.written.pop(id(mob), None)
                    frame.holds.pop(id(mob), None)
            finally:
                tracer._site_depth -= 1
                tracer._preparing -= 1

        return prepared

    @contextmanager
    def _frame(self, drawing: Any, capture: Any) -> Iterator[None]:
        if drawing.captured:
            # Frames described from the camera's own captures are drawn by its
            # code as it runs now (``arachne.frame_ir.live``): its reads of the
            # live scene are what that route is, and it reuses no frame.
            yield
            return
        frame = _Frame(drawing, capture)
        if capture.camera is not None:
            self._cameras_of(frame, drawing.camera, capture.camera)
        outer = not self.frames
        if not outer:
            frame.given = self.frames[-1].given
            frame.written = self.frames[-1].written
            frame.terms = self.frames[-1].terms
        self.frames.append(frame)
        if outer:
            self._busy = True
            try:
                self._hook(drawing.camera)
            finally:
                self._busy = False
        try:
            yield
        finally:
            self.frames.pop()
            if outer:
                self._unhook()
                self._drawer_owned.clear()
            self.traced += 1

    def _cameras_of(self, frame: _Frame, camera: Any, extras: Any) -> None:
        frame.cameras[id(camera)] = extras
        for sub_camera, sub_extras in extras.sub_cameras:
            self._cameras_of(frame, sub_camera, sub_extras)

    # The hooks, while a frame is drawn

    def _quiet(self) -> bool:
        return self._busy or self._site_depth > 0

    def _hook(self, camera: Any) -> None:
        from arachne._config.utils import ManimConfig
        from arachne.camera.camera import Camera
        from arachne.mobject.mobject import Mobject

        tracer = self
        get = object.__getattribute__

        def noting(kind: str) -> Callable[[Any, str], Any]:
            def __getattribute__(obj: Any, name: str) -> Any:  # noqa: N807
                try:
                    value = get(obj, name)
                except AttributeError:
                    if not tracer._quiet():
                        tracer._note(kind, obj, name, sys._getframe(1))
                    raise
                if not tracer._quiet():
                    tracer._note(kind, obj, name, sys._getframe(1))
                return value

            return __getattribute__

        def writing(original: Callable[..., None]) -> Callable[..., None]:
            def write(obj: Any, name: str, *value: Any) -> None:
                original(obj, name, *value)
                if not tracer._quiet() and tracer.frames:
                    tracer.frames[-1].written.setdefault(id(obj), set()).add(name)
                    for each in tracer.frames:
                        each.holds.pop(id(obj), None)

            return write

        for name, default in (
            ("__setattr__", object.__setattr__),
            ("__delattr__", object.__delattr__),
        ):
            self._override(
                Mobject, name, writing(Mobject.__dict__.get(name) or default)
            )
        for cls, kind in (
            (Mobject, "mobject"),
            (Camera, "camera"),
            (ManimConfig, "config"),
        ):
            self._override(cls, "__getattribute__", noting(kind))
        item = ManimConfig.__dict__["__getitem__"]

        def __getitem__(config: Any, key: str) -> Any:  # noqa: N807
            if not tracer._quiet():
                tracer._note("config", config, key, sys._getframe(1))
            return item(config, key)

        self._override(ManimConfig, "__getitem__", __getitem__)
        for each in _cameras_drawn(camera):
            own = vars(each)
            table = own.get("display_funcs")
            if type(table) is dict:
                own["display_funcs"] = _NotingTable(table, each, self)
                self._tables.append((each, table))

    def _override(self, cls: type, name: str, new: Callable[..., Any]) -> None:
        """Put ``new`` on ``cls`` as ``name`` until :meth:`_unhook`, as the
        engine's own override of what the class held (``camera_code``).
        """
        from arachne.timeline.camera_code import engine_installed

        original = cls.__dict__.get(name)
        self._hooks.append((cls, name, original))
        self._installed.enter_context(engine_installed(new, original))
        setattr(cls, name, new)

    def _unhook(self) -> None:
        self._installed.close()
        while self._hooks:
            cls, name, original = self._hooks.pop()
            if original is None:
                delattr(cls, name)
            else:
                setattr(cls, name, original)
        while self._tables:
            camera, table = self._tables.pop()
            vars(camera)["display_funcs"] = table

    def _audit(self, event: str, args: tuple[Any, ...]) -> None:
        """A file the drawing opens: what it reads is no recorded state."""
        if event != "open" or not self.frames or self._quiet():
            return
        self._busy = True
        try:
            caller = sys._getframe(1)
            if caller.f_code.co_filename.startswith("<frozen importlib"):
                return  # a module imported while drawing
            path = str(args[0])
            if os.path.splitext(path)[1].lower() not in FILES:
                self.found.setdefault(
                    Read(
                        "file",
                        path,
                        _site(caller),
                        "a file opened while a frame is drawn",
                    ),
                    None,
                )
        except Exception:  # noqa: BLE001, S110 - an audit hook must never raise
            pass
        finally:
            self._busy = False

    # Judging a read

    def _note(self, kind: str, obj: Any, name: str, caller: types.FrameType) -> None:
        self._busy = True
        try:
            if kind == "config":
                why = self._config_why(self.frames[-1], name, caller)
            elif kind == "camera":
                why = self._camera_why(obj, name)
            else:
                why = self._mobject_why(obj, name)
            if why is not None:
                self.found.setdefault(
                    Read(_type_name(obj), name, _site(caller), why), None
                )
        finally:
            self._busy = False

    @staticmethod
    def _config_why(frame: _Frame, name: str, caller: types.FrameType) -> str | None:
        if caller.f_code.co_filename.startswith(_CONFIG_CODE):
            return None  # the configuration's own code reading itself
        if name in CONFIG_KEYED:
            changed = frame.drawing.record.pixel_shape_changed
            if changed is None or frame.capture.seq < changed:
                return None
            return "a configuration option the scene changed by this capture"
        return "a configuration option no list entry covers"

    def _mobject_why(self, mob: Any, name: str) -> str | None:
        own = object.__getattribute__(mob, "__dict__")
        if _type_name(mob) in DRAWER_PROXIES or (
            self._drawer_owned.get(id(mob)) is mob
        ):
            return _drawer_own_why(mob, own, name)
        member = _class_member(type(mob), name)
        if member is not None and name != "__dict__":
            owner, value = member
            if _is_data_descriptor(value) or name not in own:
                if name in MOBJECT_CONSTANTS:
                    return None
                return _member_why(owner, name, value)
        frame = self.frames[-1]
        if id(mob) in frame.terms or _type_name(mob) in SCRATCH_TYPES:
            return None  # TERM_ENDS, SCRATCH_TYPES
        why = self._given_why(frame, mob)
        if why is not None or name == "__dict__":
            return why
        if name in own and name not in frame.written.get(id(mob), ()):
            return _held_by_reference(name, own[name], frame.drawing.record)
        return None

    def _given_why(self, frame: _Frame, mob: Any) -> str | None:
        """Why ``mob`` does not hold the state the record gave it for the
        frame; ``None`` when it does.
        """
        key = id(mob)
        holds = frame.holds.get(key)
        if holds is None:
            holds = frame.holds[key] = self._holds(frame, mob)
        if holds:
            return None
        given = frame.given.get(key)
        if given is None:
            if frame.drawing.record.node_state(mob, frame.capture) is None:
                return "a mobject the record does not hold"
            return (
                "a mobject the record holds but did not give its state for this frame"
            )
        if not given.prepared and given.recorded is not frame.drawing.record.node_state(
            mob, frame.capture
        ):
            return "a mobject given its state at another capture than the frame's"
        held = given.held or given.recorded
        own = object.__getattribute__(mob, "__dict__")
        changed = sorted(
            {k for k in own if k not in held.keys or held.get(k) is not own[k]}
            | {k for k in held.keys if k not in own}
        )
        return (
            "a mobject whose state changed since the record gave it "
            f"({', '.join(changed) or 'its lists'})"
        )

    @staticmethod
    def _holds(frame: _Frame, mob: Any) -> bool:
        given = frame.given.get(id(mob))
        if given is None:
            return False
        if not given.prepared and given.recorded is not frame.drawing.record.node_state(
            mob, frame.capture
        ):
            return False
        return _same_state(
            given.held or given.recorded,
            object.__getattribute__(mob, "__dict__"),
            frame.written.get(id(mob), set()),
        )

    def _camera_why(self, camera: Any, name: str) -> str | None:
        from arachne.mobject.mobject import Mobject
        from arachne.timeline.draw import _same_value

        frame = self.frames[-1]
        if _type_name(camera) in DRAWER_CAMERA_TYPES:
            return self._drawer_camera_why(camera, name)
        extras = frame.cameras.get(id(camera))
        if extras is None:
            if name == "pixel_array" and id(camera) in frame.drawing._pictures_saved:
                return None  # a display's camera, showing the picture the record kept
            if self._borrowed(camera):
                return None  # DISPLAY_REPLACED
            return "a camera the capture recorded no inputs of"
        own = object.__getattribute__(camera, "__dict__")
        member = _class_member(type(camera), name)
        if member is not None and name != "__dict__":
            owner, value = member
            if _is_data_descriptor(value) or name not in own:
                why = _member_why(owner, name, value)
                if why is not None and _replaced_camera_member(owner, name, value):
                    return None  # CAMERA_REPLACED
                return why
            if callable(own[name]) and _is_code(value):
                return None  # CAMERA_REPLACED
        if name in CAMERA_CONSTANTS:
            return None
        for key, value in extras.values:
            if key == name:
                if name in own and _same_value(own[name], value):
                    return None
                return "a camera input that no longer holds its recorded value"
        if name in _INSTALLED_CAMERA_FIELDS:
            return None
        if name in CAMERA_DERIVED and name in own:
            if np.array_equal(own[name], CAMERA_DERIVED[name][1](camera)):
                return None
            return "a cache that no longer holds what its recorded inputs give"
        value = own.get(name)
        if isinstance(value, Mobject):
            if any(key == name and mob is value for key, mob in extras.bindings):
                return None  # CAMERA_OWN_MOBJECTS
            return "a camera's own mobject other than the one the capture recorded"
        if name not in own and member is None:
            return None  # CAMERA_ABSENT
        return "a camera attribute the record neither holds nor installs"

    def _drawer_camera_why(self, camera: Any, name: str) -> str | None:
        from arachne.mobject.mobject import Mobject

        own = object.__getattribute__(camera, "__dict__")
        member = _class_member(type(camera), name)
        if member is not None and name != "__dict__":
            owner, value = member
            if _is_data_descriptor(value) or name not in own:
                return _drawer_member_why(owner, name, value)
        if name in DRAWER_CAMERA:
            value = own.get(name)
            if isinstance(value, Mobject):
                self._drawer_owned[id(value)] = value
            return None
        if name in CAMERA_DERIVED and name in own:
            if np.array_equal(own[name], CAMERA_DERIVED[name][1](camera)):
                return None
            return "a cache of the drawer's camera that its view's inputs do not give"
        if name not in own and member is None:
            return None  # CAMERA_ABSENT
        return "an attribute of the drawer's camera that the description does not give"

    def _note_entry(
        self, camera: Any, kind: Any, entry: Any, caller: types.FrameType
    ) -> None:
        if self._quiet():
            return
        why = _entry_why(camera, entry)
        if why is not None and not self._reported(camera, kind, entry):
            name = f"display_funcs[{getattr(kind, '__name__', kind)}]"
            self.found.setdefault(
                Read(_type_name(camera), name, _site(caller), why), None
            )

    def _borrowed(self, camera: Any) -> bool:
        """Whether a display function the record reports as the scene's is a
        method bound to ``camera``, a camera the capture recorded no inputs
        of: the scene drew through that camera's settings
        (``DISPLAY_REPLACED``).
        """
        return any(
            getattr(each.method, "__self__", None) is camera
            for each in self._replaced_displays()
        )

    def _reported(self, camera: Any, kind: Any, entry: Any) -> bool:
        """Whether the record reports ``entry`` as the scene's replacement of
        ``camera``'s display function for ``kind`` (``Record.camera_replaced``,
        ``DISPLAY_REPLACED``). Only the lookup of that entry is exempt.
        """
        name = f"display_funcs[{getattr(kind, '__name__', kind)}]"
        return any(
            each.name == name and each.method is entry
            for each in self._replaced_displays()
        )

    def _replaced_displays(self) -> list[Any]:
        """The display functions the scene replaced on the cameras the frame is
        drawn with, as the camera-code detector finds them now.
        """
        from arachne.timeline.camera_code import drawn_cameras, replaced_methods

        held, self._busy = self._busy, True
        try:
            return [
                each
                for camera in drawn_cameras(self.frames[-1].drawing.camera)
                for each in replaced_methods(camera)
                if each.form == "display_funcs"
            ]
        finally:
            self._busy = held


#: Camera attributes the record installs from the capture's inputs
#: (``CameraExtras.install``) rather than holding as values.
_INSTALLED_CAMERA_FIELDS = frozenset(
    {
        "_background",
        "_built_background",
        "fixed_in_frame_mobjects",
        "fixed_orientation_mobjects",
        "image_mobjects_from_cameras",
    }
)


class _NotingTable(dict):
    """A camera's ``display_funcs``, noting each lookup."""

    def __init__(self, table: dict, camera: Any, tracer: Tracer) -> None:
        super().__init__(table)
        self._camera = camera
        self._tracer = tracer

    def __getitem__(self, kind: Any) -> Any:
        entry = super().__getitem__(kind)
        self._tracer._note_entry(self._camera, kind, entry, sys._getframe(1))
        return entry


def _entry_why(camera: Any, entry: Any) -> str | None:
    """Why a display function is not the engine's own for ``camera``."""
    from arachne.timeline.frozen import engine_code

    if isinstance(entry, types.MethodType):
        if entry.__self__ is camera and engine_code(entry.__func__):
            return None
        return "a display function that is not the camera's own engine method"
    if isinstance(entry, types.FunctionType) and engine_code(entry):
        return None
    return "a display function the engine did not define"


def _member_why(owner: type, name: str, value: Any) -> str | None:
    """Why a class attribute read is not one the record need not hold."""
    from arachne.timeline.frozen import class_members, engine_class

    if owner.__module__ == "builtins":
        return None if _is_code(value) else f"{owner.__qualname__}.{name}"
    if not engine_class(owner):
        return f"a member of {owner.__qualname__}, a class the scene defined"
    if class_members(owner).get(name, _MISSING) is not value:
        return f"{owner.__qualname__}.{name} is not the engine's own"
    if _is_code(value) or (owner.__qualname__, name) in CLASS_CONSTANTS:
        return None
    return f"a class attribute of {owner.__qualname__} no list entry covers"


#: The module of the Cairo drawer, whose classes are the drawer's own.
_DRAWER_MODULE = "arachne.frame_ir.cairo_painter"


def _drawer_member_why(owner: type, name: str, value: Any) -> str | None:
    """Why a class attribute read of one of the drawer's objects is not
    engine code.
    """
    if owner.__module__ == _DRAWER_MODULE:
        return None if _is_code(value) else f"a class attribute of {owner.__qualname__}"
    return _member_why(owner, name, value)


def _drawer_own_why(obj: Any, own: dict[str, Any], name: str) -> str | None:
    """Why a read of a drawer's proxy, or of its camera's own mobject, is not
    of a field the drawer gave it or of engine code.
    """
    member = _class_member(type(obj), name)
    if member is not None and name != "__dict__":
        owner, value = member
        if _is_data_descriptor(value) or name not in own:
            if name in MOBJECT_CONSTANTS:
                return None
            return _drawer_member_why(owner, name, value)
    if name == "__dict__" or name in own:
        return None
    return "an attribute of a drawer's mobject that the description does not give"


def _replaced_camera_member(owner: type, name: str, value: Any) -> bool:
    """Whether ``value`` is the scene's replacement of a member an engine
    camera class defined (``Record.camera_replaced`` reports it).
    """
    from arachne.camera.camera import Camera
    from arachne.timeline.frozen import class_members, engine_class

    return (
        engine_class(owner)
        and issubclass(owner, Camera)
        and name in class_members(owner)
        and class_members(owner)[name] is not value
    )


def _cameras_drawn(camera: Any) -> list[Any]:
    found = [camera]
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        sub = vars(display).get("camera")
        if sub is not None and sub is not camera:
            found.append(sub)
    return found


_MISSING = object()
#: A mobject's lists the record keeps by their contents.
_LISTS = frozenset({"submobjects", "updaters"})


def _same_state(state: Any, own: dict[str, Any], written: set[str]) -> bool:
    """Whether a mobject's ``own`` dictionary holds ``state``, but for what
    the drawing wrote into it since (``written``) and the points its
    placement computes on first reading them (a cache of the placement,
    which the state holds).
    """
    if not written and state.matches(own):
        return True
    placement = own.get("_placement")
    computed = placement is not None and own.get("_points") is placement.materialized
    held = dict(zip(state.keys, state.values, strict=True))
    for key in held.keys() | own.keys():
        if key in written or (key == "_points" and computed):
            continue
        if key not in held or key not in own or held[key] is not own[key]:
            return False
    for name, contents in (
        ("submobjects", state.submobjects),
        ("updaters", state.updaters),
    ):
        live = own.get(name)
        if (
            name not in written
            and live is not None
            and (
                len(live) != len(contents)
                or not all(a is b for a, b in zip(live, contents, strict=True))
            )
        ):
            return False
    return True


def _held_by_reference(name: str, value: Any, record: Any) -> str | None:
    """Why ``value``, a field of a recorded state, is not held by the record
    as it was at the capture: an object the record keeps by reference that
    can be changed in place. ``None`` when it can't.
    """
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.frozen import engine_code

    if isinstance(value, np.ndarray):
        if _immutable(value):
            return None
        return "an array the record keeps by reference, writable in place"
    if isinstance(value, (list, dict, set, bytearray)):
        if name in _LISTS:
            return None  # kept by their contents
        return f"a {type(value).__name__} the record keeps by reference"
    if isinstance(value, Mobject):
        return None  # its state is checked as any mobject's
    if isinstance(value, Camera) or record.models_image(value):
        return None  # KEPT_APART
    if isinstance(value, types.FunctionType):
        return None if engine_code(value) else "a function the engine did not define"
    if _immutable(value):
        return None
    kind = type(value)
    if f"{kind.__module__}.{kind.__qualname__}" == "PIL.Image.Image":
        return "an image the record keeps by reference, which can be painted in place"
    return (
        f"a {kind.__module__}.{kind.__qualname__} the record keeps by reference, "
        "which no list entry says is immutable"
    )


#: Plain values: nothing changes one in place.
_PLAIN = (type(None), bool, int, float, complex, str, bytes, np.generic, enum.Enum)


def _immutable(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        array: Any = value
        while isinstance(array, np.ndarray):
            if array.flags.writeable:
                return False
            array = array.base
        return True
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable(item) for item in value)
    return _type_name(value) in IMMUTABLE_TYPES


def _live_family(roots: list[Any]) -> Iterator[Any]:
    """Every mobject reachable from ``roots`` through the submobjects they
    hold now, each once.
    """
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        mob = stack.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        yield mob
        stack.extend(reversed(vars(mob).get("submobjects", ())))


def _class_member(cls: type, name: str) -> tuple[type, Any] | None:
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass, klass.__dict__[name]
    return None


def _is_data_descriptor(value: Any) -> bool:
    kind = type(value)
    return hasattr(kind, "__set__") or hasattr(kind, "__delete__")


_CODE = (
    types.FunctionType,
    property,
    staticmethod,
    classmethod,
    types.MemberDescriptorType,
    types.GetSetDescriptorType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.BuiltinFunctionType,
    functools.cached_property,
    functools._lru_cache_wrapper,
)


def _is_code(value: Any) -> bool:
    return isinstance(value, _CODE)


def _type_name(obj: Any) -> str:
    kind = type(obj)
    return f"{kind.__module__}.{kind.__qualname__}"


#: Where the code that decides what a frame draws lives: a read made below it
#: is placed by the nearest such line too.
_DRAWING_CODE = tuple(
    os.path.join(_ENGINE_ROOT, part) for part in ("camera", "frame_ir", "timeline")
)


def _site(caller: types.FrameType | None) -> str:
    """The innermost line outside this file that made the read, and the
    nearest line of the camera's, a painter's or the record's code above it
    when that is another.
    """
    frame = caller
    while frame is not None and frame.f_code.co_filename == _HERE:
        frame = frame.f_back
    if frame is None:
        return "?"
    site = _line(frame)
    if not frame.f_code.co_filename.startswith(_DRAWING_CODE):
        above = frame.f_back
        while above is not None and not above.f_code.co_filename.startswith(
            _DRAWING_CODE
        ):
            above = above.f_back
        if above is not None:
            site += f", via {_line(above)}"
    return site


def _line(frame: types.FrameType) -> str:
    filename = frame.f_code.co_filename
    if filename.startswith(_ENGINE + os.sep):
        filename = os.path.relpath(filename, _ENGINE)
    return f"{filename}:{frame.f_lineno} ({frame.f_code.co_qualname})"


_TRACER: Tracer | None = None


@contextmanager
def untraced() -> Iterator[None]:
    """Within: reads are not judged. For a test's own code that watches a
    drawing from inside it (a spy on a painter), whose reads of the engine
    are the test's, not the drawing's. Nothing changes when no frame is traced.
    """
    if _TRACER is None:
        yield
        return
    _TRACER._site_depth += 1
    try:
        yield
    finally:
        _TRACER._site_depth -= 1


def tracer() -> Tracer:
    """The process's tracer, installed on first use."""
    global _TRACER
    if _TRACER is None:
        from arachne.timeline.frozen import freeze

        # The engine's identities first: none of them is a wrapper of this file.
        freeze()
        made = Tracer()
        made.install()
        _TRACER = made
    return _TRACER
