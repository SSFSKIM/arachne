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

__all__ = ["CameraExtras", "Drawing", "camera_nodes"]

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    captured instant.
    """

    values: tuple[tuple[str, Any], ...]
    fixed_in_frame: frozenset[Mobject] = frozenset()
    centers: tuple[tuple[Mobject, np.ndarray], ...] = ()
    background: FrozenBackground | None = None

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
        if previous is None:
            return cls(values, fixed, centers, background)
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
        ):
            return previous
        return cls(values, fixed, centers, background)

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
        self._camera_saved: tuple[Any, Any, Any] | None = None

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
        if self._camera_saved is not None:
            values, backgrounds, orientation = self._camera_saved
            _set_camera_values(self.camera, values)
            self.camera.background, self.camera._built_background = backgrounds
            if orientation is not None:
                camera: Any = self.camera
                camera.fixed_in_frame_mobjects, camera.fixed_orientation_mobjects = (
                    orientation
                )
            self._camera_saved = None

    def show(self, capture: Capture, roots: Sequence[Mobject]) -> None:
        """Give every mobject reachable from ``roots`` (and the camera's own)
        its state at ``capture``, and the camera its recorded inputs.
        """
        from .record import NodeState

        saved = self._saved
        reached = list(self.record._walk(capture, [*roots, *camera_nodes(self.camera)]))
        for mob, _ in reached:
            if id(mob) not in saved:
                saved[id(mob)] = (mob, NodeState.of(mob))
        for mob, state in reached:
            state.install(mob)
        if capture.camera is not None:
            if self._camera_saved is None:
                # The live inputs, saved as they are (calling no centre function).
                camera: Any = self.camera
                orientation = (
                    (camera.fixed_in_frame_mobjects, camera.fixed_orientation_mobjects)
                    if hasattr(camera, "fixed_orientation_mobjects")
                    else None
                )
                self._camera_saved = (
                    _camera_values(camera, None),
                    (camera.background, camera._built_background),
                    orientation,
                )
            capture.camera.install(self.camera)

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
        self, play: PlayRecord, capture: Capture, static: PixelArray | None
    ) -> PixelArray:
        """The frame at ``capture``: the play's moving mobjects there over its
        static picture, or the whole scene when nothing moves.
        """
        targets = list(capture.moving) or capture.draw_list()
        self.show(capture, targets)
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
