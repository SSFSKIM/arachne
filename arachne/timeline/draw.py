"""Drawing from the record.

A frame is drawn from a capture, never from the live scene: the mobjects it
reaches are given their recorded states for the draw and their own states
back afterwards, and the camera's executable inputs (ThreeDCamera's
fixed-orientation centre functions) were evaluated in the forward run and
are replaced by their recorded values. The draw itself is CE's: the play's
static picture, then its moving mobjects over it: the list flattened when
it began (``Scene.begin_animations``), with what the scene added to it since.
"""

from __future__ import annotations

__all__ = ["CameraExtras", "Drawing", "LiveFrames", "camera_nodes"]

from collections.abc import Callable, Sequence
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


class Drawing:
    """Draws captures of a record with the scene's camera.

    Inside its ``with`` block, :meth:`show` gives the mobjects a capture
    reaches their recorded states; on leaving it, every mobject shown and the
    camera get back what they held before.
    """

    def __init__(self, record: Record) -> None:
        self.record = record
        self.camera: Camera = record._scene.renderer.camera
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
        self.show(self.record._captures[play.static_seq], play.static)
        self.camera.reset()
        self.camera.capture_mobjects(list(play.static))
        return self.camera.pixel_array.copy()

    def sample(
        self,
        play: PlayRecord,
        capture: Capture,
        static: PixelArray | None,
        *,
        prepare: Callable[[Drawing], None] | None = None,
    ) -> PixelArray:
        """The frame at ``capture``: the play's moving mobjects there over its
        static picture, or the whole scene when nothing moves. ``prepare``
        runs once the capture is shown (a trusted play's evaluation at an
        instant between its captures).
        """
        targets = list(capture.moving) or capture.draw_list()
        self.show(capture, targets)
        if prepare is not None:
            prepare(self)
        if static is not None:
            self.camera.set_frame_to_background(static)
        else:
            self.camera.reset()
        self.camera.capture_mobjects(targets)
        return self.camera.pixel_array.copy()

    def final(self) -> PixelArray:
        """The scene as its script left it, as CE draws its last frame."""
        capture = self.record.final
        assert capture is not None, "the scene has not finished"
        targets = capture.draw_list()
        self.show(capture, targets)
        self.camera.reset()
        self.camera.capture_mobjects(targets)
        return self.camera.pixel_array.copy()


class LiveFrames:
    """Draws a play's frames from the live scene while it runs, as CE draws
    them: for a record that can't draw them afterwards (``Record.refusal``).

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
    ) -> None:
        self.scene = scene
        self.camera: Camera = scene.renderer.camera
        self.fps = fps
        self.write = write
        self._static: PixelArray | None = None
        self._pending: tuple[Fraction, Fraction] | None = None
        self._open = False

    def begin(self, play: PlayRecord) -> None:
        """The play's animations have begun: draw its static picture."""
        self._open = True
        self._pending = None
        self._static = None
        if play.static:
            self.camera.reset()
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
        targets = list(scene.moving_mobjects) or [
            *(m for m in scene.mobjects if id(m) not in foreground),
            *scene.foreground_mobjects,
        ]
        if self._static is not None:
            self.camera.set_frame_to_background(self._static)
        else:
            self.camera.reset()
        self.camera.capture_mobjects(targets)
        self.write(self.camera.pixel_array.copy(), len(frames))

    def end(self, end: Fraction) -> PixelArray | None:
        """The play ended at ``end``: draw what the last instant still owes,
        and return its static picture.
        """
        self.flush(end)
        self._open = False
        return self._static
