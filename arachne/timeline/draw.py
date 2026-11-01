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
    from arachne.typing import PixelArray

    from .record import Capture, NodeState, PlayRecord, Record


@dataclass(frozen=True)
class CameraExtras:
    """A ThreeDCamera's drawing inputs that are not mobjects: the mobjects
    fixed in the frame, and each fixed-orientation mobject's centre as its
    function returned it at the captured instant.
    """

    fixed_in_frame: frozenset[Mobject]
    centers: tuple[tuple[Mobject, np.ndarray], ...]

    @classmethod
    def observe(
        cls, camera: Camera, previous: CameraExtras | None
    ) -> CameraExtras | None:
        """Evaluate ``camera``'s inputs now; ``previous`` itself when they are
        what it already holds. ``None`` for a camera that has none.
        """
        from arachne.camera.three_d_camera import ThreeDCamera

        if not isinstance(camera, ThreeDCamera):
            return None
        orientations = camera.fixed_orientation_mobjects
        fixed = camera.fixed_in_frame_mobjects
        values: dict[int, np.ndarray] = {}
        centers = []
        for mob, center_func in orientations.items():
            value = values.get(id(center_func))
            if value is None:
                value = np.array(center_func(), dtype=float)
                value.flags.writeable = False
                values[id(center_func)] = value
            centers.append((mob, value))
        frozen = frozenset(fixed)
        if previous is None:
            return cls(frozen, tuple(centers))
        if previous.fixed_in_frame == frozen:
            frozen = previous.fixed_in_frame
        same_centers = len(previous.centers) == len(centers) and all(
            a is c and np.array_equal(b, d)
            for (a, b), (c, d) in zip(previous.centers, centers, strict=True)
        )
        if same_centers and frozen is previous.fixed_in_frame:
            return previous
        return cls(frozen, previous.centers if same_centers else tuple(centers))

    def install(self, camera: Camera) -> None:
        """Give ``camera`` these inputs."""
        camera.fixed_in_frame_mobjects = set(self.fixed_in_frame)  # type: ignore[attr-defined]
        camera.fixed_orientation_mobjects = {  # type: ignore[attr-defined]
            mob: _constant(value) for mob, value in self.centers
        }


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
        self._camera_saved: tuple[Any, Any] | None = None

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
            camera: Any = self.camera
            camera.fixed_in_frame_mobjects, camera.fixed_orientation_mobjects = (
                self._camera_saved
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
                camera: Any = self.camera
                self._camera_saved = (
                    camera.fixed_in_frame_mobjects,
                    camera.fixed_orientation_mobjects,
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
