"""Drawing from the record.

A frame is drawn from a capture, never from the live scene: the mobjects it
reaches are given their recorded states for the draw and their own states
back afterwards, and the camera's executable inputs (ThreeDCamera's
fixed-orientation centre functions) were evaluated in the forward run and
are replaced by their recorded values. The draw is the two layers of CE
0.22.1's split (#4936): the play's static mobjects, then its moving
mobjects over them: the list flattened when it began
(``Scene.begin_animations``), with what the scene added to it since.

Every frame shows each mobject at its state there. CE decides what moves
when the play begins, and that decision is kept; but CE paints the static
mobjects once, then, and a driver added after them that writes them (moves
a body, replaces a readout's glyphs, turns the camera, moves its light)
leaves them frozen. Here each frame's static list is rebuilt from its
capture: the scene's roots there flattened as CE flattens them, minus the
moving families, in the camera's draw order. The play's static picture
covers the longest prefix of that list equal leaf for leaf, at unchanged
versions, to what it painted, while the camera's inputs are those it was
painted with; the rest is drawn at the frame's state. ``static_layer=False``
draws the whole list in every frame. Frames drawn live (:class:`LiveFrames`)
reuse no static picture, since drawing them runs the scene's code, which can
read what no recorded state holds; only a frozen frame, during which nothing
runs, keeps CE's picture under it.
"""

from __future__ import annotations

__all__ = ["CameraExtras", "Drawing", "LiveFrames", "camera_nodes", "static_list"]

import operator
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.values import FrozenBackground
    from arachne.typing import PixelArray

    from .record import Capture, NodeState, PlayRecord, Record


#: Camera attributes that are not plain drawing values: the pixel shape and
#: what is built from it (a session draws at its own quality), caches, and
#: the background array, recorded apart (``CameraExtras.background``).
_NOT_CAMERA_VALUES = frozenset(
    {
        "pixel_width",
        "pixel_height",
        "pixel_array",
        "_background",
        "_built_background",
        "pixel_array_to_cairo_context",
        "pixel_array_to_mapping_centre",
        "display_funcs",
        "canvas",
        "background_colored_vmobject_displayer",
        "rotation_matrix",
        "frame_rate",
        "fixed_in_frame_mobjects",
        "fixed_orientation_mobjects",
        "image_mobjects_from_cameras",
    }
)
#: The values the background is built from (``Camera.init_background``).
_BACKGROUND_INPUTS = frozenset(
    {"_background_color", "_background_opacity", "background_image"}
)


@dataclass(frozen=True)
class CameraExtras:
    """A camera's drawing inputs that are not mobjects.

    ``values`` are the camera's own plain values a draw reads (background
    colour, opacity and image, frame centre and size, line-width multiple,
    z-index use, a ThreeDCamera's shading and projection settings): every
    attribute holding a number, string, colour or small array, but the pixel
    shape and what is built from it. ``background`` is a background array the
    scene supplied (``Camera.set_background``), read-only like every camera
    background; ``None`` when it is the one the camera builds from its values. A
    ThreeDCamera also has the mobjects fixed in the frame, and each
    fixed-orientation mobject's centre as its function returned it at the
    captured instant. A MultiCamera has the displays it draws its
    sub-cameras into (``image_mobjects_from_cameras``), which a scene adds
    and removes as it goes (a ``ZoomedScene`` activating its zoom), and the
    inputs of each display's own camera, which draws into it
    (``sub_cameras``, recorded the same way).
    """

    values: tuple[tuple[str, Any], ...]
    fixed_in_frame: frozenset[Mobject] = frozenset()
    centers: tuple[tuple[Mobject, np.ndarray], ...] = ()
    background: FrozenBackground | None = None
    displays: tuple[Mobject, ...] | None = None
    sub_cameras: tuple[tuple[Camera, CameraExtras], ...] = ()

    @classmethod
    def observe(
        cls, camera: Camera, previous: CameraExtras | None
    ) -> CameraExtras | None:
        """Evaluate ``camera``'s inputs now; ``previous`` itself when they are
        what it already holds. ``None`` for an object that is not a camera.
        """
        from arachne.camera.camera import Camera
        from arachne.camera.three_d_camera import ThreeDCamera

        if not isinstance(camera, Camera):
            return None
        values = _camera_values(camera, previous.values if previous else None)
        background = _supplied_background(camera)
        fixed: frozenset[Mobject] = frozenset()
        centers: tuple[tuple[Mobject, np.ndarray], ...] = ()
        if isinstance(camera, ThreeDCamera):
            fixed = frozenset(camera.fixed_in_frame_mobjects)
            centers = _centers(camera.fixed_orientation_mobjects)
        displays = _displays(camera)
        sub_cameras = _sub_cameras(camera, displays, previous)
        if previous is None:
            return cls(values, fixed, centers, background, displays, sub_cameras)
        if (
            displays is not None
            and previous.displays is not None
            and len(displays) == len(previous.displays)
            and all(a is b for a, b in zip(displays, previous.displays, strict=True))
        ):
            displays = previous.displays
        if previous.fixed_in_frame == fixed:
            fixed = previous.fixed_in_frame
        if len(previous.centers) == len(centers) and all(
            a is c and np.array_equal(b, d)
            for (a, b), (c, d) in zip(previous.centers, centers, strict=True)
        ):
            centers = previous.centers
        if (
            values is previous.values
            and fixed is previous.fixed_in_frame
            and centers is previous.centers
            and background is previous.background
            and displays is previous.displays
            and sub_cameras is previous.sub_cameras
        ):
            return previous
        return cls(values, fixed, centers, background, displays, sub_cameras)

    def install(self, camera: Camera) -> None:
        """Give ``camera`` these inputs."""
        from arachne.camera.three_d_camera import ThreeDCamera

        _set_camera_values(camera, self.values)
        if self.background is None:
            camera.background = camera._built_background
        else:
            camera.background = _fit(
                self.background, camera.pixel_height, camera.pixel_width
            )
        if isinstance(camera, ThreeDCamera):
            camera.fixed_in_frame_mobjects = set(self.fixed_in_frame)
            camera.fixed_orientation_mobjects = {
                mob: _constant(value) for mob, value in self.centers
            }
        if self.displays is not None:
            camera.image_mobjects_from_cameras = list(self.displays)  # type: ignore[attr-defined]


def _saved_inputs(camera: Any) -> tuple[Any, Any, Any, Any]:
    """A camera's live inputs, as they are (calling no centre function)."""
    orientation = (
        (camera.fixed_in_frame_mobjects, camera.fixed_orientation_mobjects)
        if hasattr(camera, "fixed_orientation_mobjects")
        else None
    )
    return (
        _camera_values(camera, None),
        (camera.background, camera._built_background),
        orientation,
        vars(camera).get("image_mobjects_from_cameras"),
    )


def _restore_inputs(camera: Any, saved: tuple[Any, Any, Any, Any]) -> None:
    values, backgrounds, orientation, displays = saved
    _set_camera_values(camera, values)
    camera.background, camera._built_background = backgrounds
    if orientation is not None:
        camera.fixed_in_frame_mobjects, camera.fixed_orientation_mobjects = orientation
    if displays is not None:
        camera.image_mobjects_from_cameras = displays


def _sub_cameras(
    camera: Camera,
    displays: tuple[Mobject, ...] | None,
    previous: CameraExtras | None,
) -> tuple[tuple[Camera, CameraExtras], ...]:
    """The inputs of each display's camera now; ``previous.sub_cameras``
    itself when they are the same.
    """
    from arachne.camera.camera import Camera

    before = previous.sub_cameras if previous is not None else ()
    earlier = {id(sub): extras for sub, extras in before}
    found: list[tuple[Camera, CameraExtras]] = []
    for display in displays or ():
        sub = getattr(display, "camera", None)
        if isinstance(sub, Camera) and sub is not camera:
            extras = CameraExtras.observe(sub, earlier.get(id(sub)))
            assert extras is not None
            found.append((sub, extras))
    if len(found) == len(before) and all(
        a is c and b is d for (a, b), (c, d) in zip(found, before, strict=True)
    ):
        return before
    return tuple(found)


def _displays(camera: Camera) -> tuple[Mobject, ...] | None:
    """A MultiCamera's displays, in the order it draws them; ``None`` for
    another camera.
    """
    displays = vars(camera).get("image_mobjects_from_cameras")
    return None if displays is None else tuple(displays)


def _centers(
    orientations: dict[Mobject, Callable[[], Any]],
) -> tuple[tuple[Mobject, np.ndarray], ...]:
    values: dict[int, np.ndarray] = {}
    centers = []
    for mob, center_func in orientations.items():
        value = values.get(id(center_func))
        if value is None:
            value = np.array(center_func(), dtype=float)
            value.flags.writeable = False
            values[id(center_func)] = value
        centers.append((mob, value))
    return tuple(centers)


def _camera_values(
    camera: Camera, previous: tuple[tuple[str, Any], ...] | None
) -> tuple[tuple[str, Any], ...]:
    """The camera's plain drawing values now, by attribute name; ``previous``
    itself when they are the same.
    """
    from arachne.utils.color import ManimColor

    found: list[tuple[str, Any]] = []
    for key, value in vars(camera).items():
        if key in _NOT_CAMERA_VALUES:
            continue
        if isinstance(value, np.ndarray):
            if value.size > 16:
                continue
            value = value.copy()
            value.flags.writeable = False
        elif not (
            value is None
            or isinstance(value, (bool, int, float, str, np.generic, ManimColor))
        ):
            continue
        found.append((key, value))
    values = tuple(found)
    if (
        previous is not None
        and len(previous) == len(values)
        and all(
            a == c and _same_value(b, d)
            for (a, b), (c, d) in zip(previous, values, strict=True)
        )
    ):
        return previous
    return values


def _same_value(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b))
    return bool(a == b)


def _set_camera_values(camera: Camera, values: tuple[tuple[str, Any], ...]) -> None:
    """Give ``camera`` the plain ``values``, rebuilding its background when
    what it is built from changed.
    """
    own = vars(camera)
    rebuild = False
    for key, value in values:
        current = own.get(key, _MISSING)
        if current is not _MISSING and _same_value(current, value):
            continue
        own[key] = value.copy() if isinstance(value, np.ndarray) else value
        rebuild = rebuild or key in _BACKGROUND_INPUTS
    if rebuild:
        camera.init_background()


_MISSING = object()


def _supplied_background(camera: Camera) -> FrozenBackground | None:
    """The background array the scene supplied, itself: it is read-only, so
    it holds the pixels it held when supplied. ``None`` when the background
    is the one the camera built from its values.
    """
    live = camera.background
    if live is None or live is camera._built_background:
        return None
    return live


def _fit(background: FrozenBackground, height: int, width: int) -> FrozenBackground:
    """``background`` at a pixel shape of ``height`` by ``width``: itself at
    its own, else resampled to the nearest pixel (a session drawing at
    another quality than the script ran at).
    """
    from arachne.mobject.values import FrozenBackground, adopt

    if background.shape[:2] == (height, width):
        return background
    rows = np.arange(height) * background.shape[0] // height
    columns = np.arange(width) * background.shape[1] // width
    return adopt(np.asarray(background)[rows][:, columns], FrozenBackground)


def _constant(value: np.ndarray) -> Callable[[], np.ndarray]:
    return lambda: value


def camera_nodes(camera: Camera) -> list[Mobject]:
    """The mobjects a camera draws with that the scene may not hold: a
    ThreeDCamera's trackers, frame centre and light, a MovingCamera's frame.
    """
    from arachne.mobject.mobject import Mobject

    found: list[Mobject] = []
    for value in vars(camera).values():
        if isinstance(value, Mobject):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            found.extend(item for item in value if isinstance(item, Mobject))
    # A MultiCamera's displays draw their own cameras, whose frames move.
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        sub_camera = getattr(display, "camera", None)
        if sub_camera is not None and sub_camera is not camera:
            found.extend(
                value
                for value in vars(sub_camera).values()
                if isinstance(value, Mobject)
            )
    return found


def static_list(
    camera: Camera, roots: Sequence[Mobject], moving: Sequence[Mobject]
) -> list[Mobject]:
    """The static mobjects of a frame whose scene holds ``roots`` (in CE's
    draw order, the foreground last) and moves ``moving``: as
    ``Scene.get_moving_and_static_mobjects`` flattens them, from the
    mobjects' states now.
    """
    from arachne.utils.family import extract_mobject_family_members
    from arachne.utils.iterables import list_difference_update

    z = camera.use_z_index
    return list_difference_update(
        extract_mobject_family_members(
            roots, use_z_index=z, only_those_with_points=True
        ),
        extract_mobject_family_members(moving, use_z_index=z),
    )


@dataclass
class _Picture:
    """What a play's static picture painted: each leaf in the camera's draw
    order, with its version at the play's static capture; and the last
    picture of a prefix of them drawn since, with its length.
    """

    painted: tuple[tuple[Mobject, NodeState | None], ...]
    prefix: tuple[int, PixelArray] | None = None


@dataclass(frozen=True)
class _Validity:
    """How long a play's static picture holds, read from its record.

    ``nodes`` are the ids of the static mobjects the picture depends on (not
    the play's moving ones); ``until`` the seq of the first capture at which
    one of them, or one of the camera's own mobjects, changed, and
    ``camera_until`` that of the camera's alone (``None``: never).
    """

    nodes: frozenset[int]
    until: int | None
    camera_until: int | None


def _same_items(a: Sequence[Any], b: Sequence[Any]) -> bool:
    return a is b or (len(a) == len(b) and all(map(operator.is_, a, b)))


def _paints_its_list(camera: Camera) -> bool:
    """Whether the camera's capture paints the leaves it is handed one after
    another and nothing else, so that a list painted in two captures paints
    what it paints in one: CE's flat, moving and 3D cameras (a MultiCamera
    also captures into its displays' cameras, afresh at each capture).
    """
    from arachne.camera.camera import Camera
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    return type(camera) in (Camera, MovingCamera, ThreeDCamera)


def _reset_rotation(camera: Camera) -> None:
    """Make a ThreeDCamera's depth order read its rotation now: its
    ``get_mobjects_to_display`` reads the rotation its last capture set.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    if isinstance(camera, ThreeDCamera):
        camera.reset_rotation_matrix()


def _depths_ordered(camera: Camera, painted: Sequence[Mobject]) -> bool:
    """Whether every depth a ThreeDCamera sorts ``painted`` by is a number
    (``ThreeDCamera.get_mobjects_to_display``'s key); always for another
    camera.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    if not isinstance(camera, ThreeDCamera):
        return True
    rotation = camera.get_rotation_matrix()
    return not any(
        getattr(mob, "shade_in_3d", False)
        and np.isnan(np.dot(mob.get_z_index_reference_point(), rotation.T)[2])
        for mob in painted
    )


def _one_batch(camera: Camera, a: Mobject, b: Mobject) -> bool:
    """Whether ``a`` and ``b``, next to each other in a list the camera
    paints, are painted as one batch: vectorized mobjects filled from one
    background image, which the camera composites together.
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    if (
        camera.type_or_raise(a) is not VMobject
        or camera.type_or_raise(b) is not VMobject
    ):
        return False
    image = a.get_background_image()
    return bool(image) and bool(image == b.get_background_image())


class Drawing:
    """Draws captures of a record with the scene's camera.

    Inside its ``with`` block, :meth:`show` gives the mobjects a capture
    reaches their recorded states; on leaving it, every mobject shown and the
    camera get back what they held before.

    ``static_layer`` (the option of that name, unless given) says whether a
    play's static picture is reused under its frames while it holds, or
    every frame draws its static mobjects: the same frames either way.

    ``draw_path`` (the option of that name, unless given) says how a shown
    frame becomes pixels: on ``cairo`` the camera draws the shown mobjects;
    on ``ir_cairo`` a painter (:mod:`arachne.frame_ir.painter`) describes
    the frame in XF and draws it from that description alone. Its frames are
    the frame's static list then its moving list, as with the static layer
    off, from a stream kept per record and process across drawings.
    """

    def __init__(
        self,
        record: Record,
        *,
        static_layer: bool | None = None,
        draw_path: str | None = None,
    ) -> None:
        from arachne import config

        if static_layer is None:
            static_layer = config.static_layer
        if draw_path is None:
            draw_path = config.draw_path
        from arachne.frame_ir.painter import painter_for

        self.record = record
        self.static_layer = static_layer
        self.draw_path = draw_path
        #: The painters of the draw path (``None`` on ``cairo``): one for
        #: frames, one for the plays' static pictures.
        self.painter = painter_for(record, draw_path)
        self.static_painter = painter_for(record, draw_path, "static")
        #: The painter that drew last, which settles the cameras on exit.
        self._painted_by: Any = None
        self.camera: Camera = record._scene.renderer.camera
        #: Per play (by index): what its static picture painted, and how
        #: long that picture holds.
        self._pictures: dict[int, _Picture] = {}
        self._validity: dict[int, _Validity] = {}
        self._saved: dict[int, tuple[Mobject, NodeState]] = {}
        #: Every camera given recorded inputs (the scene's, and the cameras
        #: of its displays), with what it held before.
        self._cameras_saved: dict[int, tuple[Any, tuple[Any, Any, Any, Any]]] = {}
        #: Display cameras whose picture was replaced, with what they held.
        self._pictures_saved: dict[int, tuple[Any, PixelArray]] = {}

    def __enter__(self) -> Drawing:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._painted_by is not None:
                self._painted_by.settle()
        finally:
            self._painted_by = None
            for mob, state in self._saved.values():
                state.install(mob)
            self._saved.clear()
            for camera, saved in self._cameras_saved.values():
                _restore_inputs(camera, saved)
            self._cameras_saved.clear()
            for sub_camera, picture in self._pictures_saved.values():
                sub_camera.pixel_array = picture
            self._pictures_saved.clear()

    def show(self, capture: Capture, roots: Sequence[Mobject]) -> None:
        """Give every mobject reachable from ``roots`` (and the camera's own)
        its state at ``capture``, and the camera its recorded inputs.
        """
        from .record import NodeState

        if capture.camera is not None:
            # First, so the camera's own mobjects reached below are the capture's.
            self._install_camera(self.camera, capture.camera)
        saved = self._saved
        reached = list(self.record._walk(capture, [*roots, *camera_nodes(self.camera)]))
        for mob, _ in reached:
            if id(mob) not in saved:
                saved[id(mob)] = (mob, NodeState.of(mob))
        for mob, state in reached:
            state.install(mob)
        if capture.camera is not None and capture.camera.displays is not None:
            self._show_display_pictures(capture, [mob for mob, _ in reached])

    def _install_camera(self, camera: Camera, extras: CameraExtras) -> None:
        """Give ``camera`` and its displays' cameras their recorded inputs,
        saving what they held first.
        """
        if id(camera) not in self._cameras_saved:
            self._cameras_saved[id(camera)] = (camera, _saved_inputs(camera))
        extras.install(camera)
        for sub_camera, sub_extras in extras.sub_cameras:
            self._install_camera(sub_camera, sub_extras)

    def _show_display_pictures(self, capture: Capture, reached: list[Mobject]) -> None:
        """A display on screen whose camera the MultiCamera does not draw into
        at ``capture`` shows its camera's picture as CE left it: the first one,
        when the display joins later.
        """
        from arachne.mobject.types.image_mobject import ImageMobjectFromCamera

        assert capture.camera is not None
        assert capture.camera.displays is not None
        joined = {id(display) for display in capture.camera.displays}
        for mob in reached:
            if not isinstance(mob, ImageMobjectFromCamera) or id(mob) in joined:
                continue
            picture = self.record.display_picture(mob, capture)
            if picture is None:
                continue
            sub_camera: Any = mob.camera
            if id(sub_camera) not in self._pictures_saved:
                self._pictures_saved[id(sub_camera)] = (
                    sub_camera,
                    sub_camera.pixel_array,
                )
            sub_camera.pixel_array = picture

    def static(self, play: PlayRecord) -> PixelArray | None:
        """The play's static picture: its static mobjects as they were when
        its animations began, or ``None`` when it has none.
        """
        if not play.static:
            return None
        capture = self.record._captures[play.static_seq]
        self.show(capture, play.static)
        camera = self.camera
        if self.static_painter is not None:
            pixels = self.static_painter.paint(
                camera, [("static", list(play.static))], t=capture.instant
            )
            self._painted_by = self.static_painter
            camera.set_pixel_array(pixels)
            return pixels
        camera.reset()
        camera.capture_mobjects(list(play.static))
        self._pictures[play.index] = self._painted(play, capture)
        return camera.pixel_array.copy()

    def _painted(self, play: PlayRecord, capture: Capture) -> _Picture:
        """What the play's static picture paints, its static mobjects shown at
        its static ``capture``: the leaves the camera's capture of them paints,
        in that order, with their versions there.
        """
        camera = self.camera
        _reset_rotation(camera)
        painted = camera.get_mobjects_to_display(list(play.static))
        return _Picture(
            tuple((mob, self.record.node_state(mob, capture)) for mob in painted)
        )

    def sample(
        self,
        play: PlayRecord,
        capture: Capture,
        static: PixelArray | None,
        *,
        prepare: Callable[[Drawing], None] | None = None,
    ) -> PixelArray:
        """The frame at ``capture``: its static mobjects, then the play's
        moving mobjects there over them, or the whole scene when nothing
        moves. ``static`` is the play's static picture (:meth:`static`),
        which stands for the static mobjects while they and the camera's
        inputs are unchanged since the play began. ``prepare`` runs once the
        capture is shown (a trusted play's evaluation at an instant between
        captures).
        """
        targets = list(capture.moving) or capture.draw_list()
        if self.painter is not None:
            return self._paint(play, capture, targets, prepare)
        if self.static_layer and self._picture_holds(play, capture):
            self.show(capture, targets)
            if prepare is not None:
                prepare(self)
            if static is not None:
                self.camera.set_frame_to_background(static)
            else:
                self.camera.reset()
        else:
            self._draw_static(play, capture, static, targets, prepare)
        self.camera.capture_mobjects(targets)
        return self.camera.pixel_array.copy()

    def _paint(
        self,
        play: PlayRecord,
        capture: Capture,
        targets: list[Mobject],
        prepare: Callable[[Drawing], None] | None,
    ) -> PixelArray:
        """The frame at ``capture`` from the painter: its static list rebuilt
        at the frame's state, then the moving list, each as one capture of
        the camera would draw it.
        """
        assert self.painter is not None
        self.show(capture, [*capture.draw_list(), *targets])
        if prepare is not None:
            prepare(self)
        camera = self.camera
        listed = static_list(camera, capture.draw_list(), capture.moving)
        terms = None
        if prepare is None and play.route == "trusted":
            terms = {
                id(mob): term for mob, term in play.leaf_terms(capture.instant).items()
            }
        pixels = self.painter.paint(
            camera,
            [("static", listed), ("moving", targets)],
            t=capture.instant,
            terms=terms,
        )
        self._painted_by = self.painter
        camera.set_pixel_array(pixels)
        return pixels

    def _picture_holds(self, play: PlayRecord, capture: Capture) -> bool:
        """Whether the frame at ``capture`` has the static list the play's
        static picture painted, every member at the version it painted, and
        the camera's inputs it was painted with.
        """
        start = self.record._captures[play.static_seq]
        if not (
            capture.camera is start.camera
            and _same_items(capture.roots, start.roots)
            and _same_items(capture.foreground, start.foreground)
            and _same_items(capture.moving, start.moving)
        ):
            return False
        validity = self._validity_of(play)
        if validity.until is not None and capture.seq >= validity.until:
            return False
        # A moving family that took in a static mobject moves it now.
        nodes = validity.nodes
        return not any(
            id(mob) in nodes for mob, _ in self.record._walk(capture, capture.moving)
        )

    def _validity_of(self, play: PlayRecord) -> _Validity:
        validity = self._validity.get(play.index)
        if validity is None:
            record = self.record
            start = record._captures[play.static_seq]
            if start.camera is not None:
                # The camera's own mobjects as the play began (a MultiCamera's
                # displays then), not as the drawing found the camera.
                self._install_camera(self.camera, start.camera)
            moving = {id(mob) for mob in start.moving}
            depends = list(self._depends(start, moving))
            camera = [mob for mob, _ in record._walk(start, camera_nodes(self.camera))]
            validity = self._validity[play.index] = _Validity(
                frozenset(id(mob) for mob in depends if id(mob) not in moving),
                record.first_change([*depends, *camera], start),
                record.first_change(camera, start),
            )
        return validity

    def _depends(self, capture: Capture, moving: set[int]) -> Iterator[Mobject]:
        """The mobjects a static picture painted at ``capture`` depends on:
        every one reachable from the scene's roots that does not move, with
        the whole family under it (a moving member included: the camera
        paints a static mobject's family with it).
        """
        record = self.record
        seen: set[int] = set()
        stack = [(mob, False) for mob in reversed(capture.draw_list())]
        while stack:
            mob, under_static = stack.pop()
            if id(mob) in seen or (id(mob) in moving and not under_static):
                continue
            state = record.node_state(mob, capture)
            if state is None:
                continue
            seen.add(id(mob))
            yield mob
            stack.extend((sub, True) for sub in reversed(state.submobjects))

    def _draw_static(
        self,
        play: PlayRecord,
        capture: Capture,
        static: PixelArray | None,
        targets: list[Mobject],
        prepare: Callable[[Drawing], None] | None,
    ) -> None:
        """Draw the frame's static list at ``capture`` into the camera, over
        as much of the play's static picture as still holds.
        """
        start = self.record._captures[play.static_seq]
        validity = self._validity_of(play)
        same_camera = capture.camera is start.camera and (
            validity.camera_until is None or capture.seq < validity.camera_until
        )
        picture = None
        if self.static_layer and static is not None and same_camera:
            picture = self._picture_of(play)
        # Every root: the frame's static list is rebuilt from their states.
        self.show(capture, [*capture.draw_list(), *targets])
        if prepare is not None:
            prepare(self)
        camera = self.camera
        listed = static_list(camera, capture.draw_list(), capture.moving)
        if static is not None and picture is not None:
            _reset_rotation(camera)
            # The leaves its capture of the list paints, in that order.
            painted = camera.get_mobjects_to_display(listed)
            kept = self._kept(picture, painted, capture)
            if kept:
                if kept < len(picture.painted):
                    static = self._prefix_picture(picture, painted, kept)
                camera.set_frame_to_background(static)
                if kept < len(painted):
                    camera.capture_mobjects(painted[kept:], include_submobjects=False)
                return
        camera.reset()
        camera.capture_mobjects(listed)

    def _picture_of(self, play: PlayRecord) -> _Picture:
        """What the play's static picture painted, found without painting it
        when another drawing painted it (a session keeps its pictures).
        """
        picture = self._pictures.get(play.index)
        if picture is None:
            capture = self.record._captures[play.static_seq]
            self.show(capture, play.static)
            picture = self._pictures[play.index] = self._painted(play, capture)
        return picture

    def _kept(self, picture: _Picture, painted: list[Mobject], capture: Capture) -> int:
        """How many of the leaves the frame paints for its static list
        (``painted``, in the camera's order) a picture of ``picture``'s first
        leaves stands for: the longest prefix equal to them, at the versions
        they were painted at, cut where painting it apart from the rest
        paints what painting them at once does.
        """
        record = self.record
        kept = 0
        for (mob, state), now in zip(picture.painted, painted, strict=False):
            if now is not mob or record.node_state(mob, capture) is not state:
                break
            kept += 1
        whole = len(picture.painted)
        if not _paints_its_list(self.camera):
            # The picture stands for all of the list or for none of it.
            return whole if kept == whole == len(painted) else 0
        if (
            kept
            and not kept == whole == len(painted)
            and not _depths_ordered(self.camera, painted)
        ):
            # A depth that is no number sorts no list stably: the parts of a
            # list sorted apart may come out in another order than the whole.
            return 0
        while 0 < kept < len(painted) and _one_batch(
            self.camera, painted[kept - 1], painted[kept]
        ):
            kept -= 1
        return kept

    def _prefix_picture(
        self, picture: _Picture, painted: list[Mobject], kept: int
    ) -> PixelArray:
        """A picture of the frame's first ``kept`` static leaves, painted
        again only when the cut moves: they are as the static picture painted
        them.
        """
        if picture.prefix is not None and picture.prefix[0] == kept:
            return picture.prefix[1]
        camera = self.camera
        camera.reset()
        camera.capture_mobjects(painted[:kept], include_submobjects=False)
        found = camera.pixel_array.copy()
        picture.prefix = (kept, found)
        return found

    def final(self) -> PixelArray:
        """The scene as its script left it, as CE draws its last frame."""
        capture = self.record.final
        assert capture is not None, "the scene has not finished"
        targets = capture.draw_list()
        self.show(capture, targets)
        if self.painter is not None:
            pixels = self.painter.paint(
                self.camera, [("static", targets)], t=capture.instant
            )
            self._painted_by = self.painter
            self.camera.set_pixel_array(pixels)
            return pixels
        self.camera.reset()
        self.camera.capture_mobjects(targets)
        return self.camera.pixel_array.copy()


class LiveFrames:
    """Draws a play's frames from the live scene while it runs, as CE draws
    them but for its static picture: for a record that can't draw them
    afterwards (``Record.refusal``). Each frame draws its static list, rebuilt
    from the live scene, then the moving list; only a frozen frame keeps
    CE's static picture (``static_layer``).

    The route hands it every evaluated instant (:meth:`pending`); the frames
    an instant answers are drawn just before the next instant is evaluated
    (:meth:`flush`), while the scene still holds it, and only those inside
    the play's actual span.
    """

    def __init__(
        self,
        scene: Any,
        fps: Fraction,
        write: Callable[[PixelArray, int], None],
        *,
        static_layer: bool | None = None,
    ) -> None:
        if static_layer is None:
            from arachne import config

            static_layer = config.static_layer
        self.scene = scene
        self.camera: Camera = scene.renderer.camera
        self.fps = fps
        self.write = write
        self.static_layer = static_layer
        self._static: PixelArray | None = None
        self._pending: tuple[Fraction, Fraction] | None = None
        self._open = False

    def begin(self, play: PlayRecord) -> None:
        """The play's animations have begun. A frozen frame's static picture
        is drawn now, as CE draws it: nothing runs while the frame is held, so
        it is drawn under the frame, and the renderer keeps it as its
        background after the play. With ``static_layer`` off, none is.
        """
        self._open = True
        self._pending = None
        self._static = None
        if play.held and play.static and self.static_layer:
            from arachne.frame_ir.live import live

            self.camera.reset()
            with live(self.camera):
                self.camera.capture_mobjects(list(play.static))
            self._static = self.camera.pixel_array.copy()

    def pending(self, instant: Fraction, until: Fraction) -> None:
        """The scene holds ``instant``, which answers up to ``until``."""
        self._pending = (instant, until)

    def flush(self, upto: Fraction) -> None:
        """Draw the frames the pending instant answers before ``upto``."""
        from .clock import frames_in

        if self._pending is None or not self._open:
            return
        instant, until = self._pending
        self._pending = None
        frames = frames_in(instant, min(until, upto), self.fps)
        if not frames:
            return
        scene = self.scene
        foreground = set(map(id, scene.foreground_mobjects))
        roots = [
            *(m for m in scene.mobjects if id(m) not in foreground),
            *scene.foreground_mobjects,
        ]
        targets = list(scene.moving_mobjects) or roots
        camera = self.camera
        from arachne.frame_ir.live import live

        with live(camera):
            if self._static is not None:
                camera.set_frame_to_background(self._static)
            else:
                # No static picture: drawing runs the scene's code, which can
                # read what no state holds (a class attribute, a global, a
                # call count).
                camera.reset()
                camera.capture_mobjects(
                    static_list(camera, roots, scene.moving_mobjects)
                )
            camera.capture_mobjects(targets)
        self.write(camera.pixel_array.copy(), len(frames))

    def end(self, end: Fraction) -> PixelArray | None:
        """The play ended at ``end``: draw what the last instant still owes,
        and return its static picture.
        """
        self.flush(end)
        self._open = False
        return self._static
