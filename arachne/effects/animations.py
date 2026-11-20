"""Effects that change over a play: :class:`ChangeBlur` moves an object's
blur, :class:`RackFocus` moves the scene's focus.

Each writes, at every instant of its play the timeline evaluates, the value
its rate function has reached: the object's spec through
:func:`~.layers.set_blur`, or the camera's focus through the seam. Neither is
in the trusted table, so their plays are ticked, and every tick's value is
recorded.
"""

from __future__ import annotations

__all__ = ["ChangeBlur", "RackFocus"]

import math
import weakref
from typing import TYPE_CHECKING, Any

from arachne.animation.animation import Animation
from arachne.compose import layer as seam
from arachne.compose.layer import layer_of

from .layers import _camera_of, set_blur

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene


def _finite(value: float, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is a finite number, not {value!r}.")
    return number


#: How close to an end of its run a play's progress is taken to be there, so
#: that its last value is its target itself, not a rounding away from it.
_END = 1e-12


def _progress(animation: Animation, alpha: float) -> float:
    """How far ``animation`` has come at ``alpha``, through its rate function,
    each end snapped to. ``alpha`` is held to its run first: a play longer
    than the animation hands it more than 1 after it ends, and it then holds
    its target (a rate function may still overshoot within the run).
    """
    progress = animation.get_sub_alpha(min(max(alpha, 0.0), 1.0), 0, 1)
    if abs(progress - 1) < _END:
        return 1.0
    if abs(progress) < _END:
        return 0.0
    return progress


def _between(start: float, end: float, progress: float) -> float:
    """The value ``progress`` of the way from ``start`` to ``end``: each end
    exactly at 0 and 1.
    """
    return start * (1 - progress) + end * progress


class ChangeBlur(Animation):
    """Move ``mobject``'s blur from what it is when the play begins (none
    is 0) to ``to`` scene units over the play. A blur that ends at 0 leaves
    no blur; the object's other effects are kept. The object's updaters
    keep running through the play: a blur changes nothing they write.
    """

    def __init__(self, mobject: Mobject, to: float, **kwargs: Any) -> None:
        target = _finite(to, "A blur")
        if target < 0:
            raise ValueError(f"A blur is a length, 0 or more, not {to!r}.")
        kwargs.setdefault("suspend_mobject_updating", False)
        super().__init__(mobject, **kwargs)
        self.to = target
        self._from = 0.0

    def create_starting_mobject(self) -> Mobject:
        # The blur it starts from is all it reads of the object's start.
        return self.mobject

    def begin(self) -> None:
        spec = layer_of(self.mobject)
        self._from = 0.0 if spec is None or spec.blur is None else spec.blur
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        sigma = _between(self._from, self.to, _progress(self, alpha))
        set_blur(self.mobject, sigma if sigma > 0 else None)


class RackFocus(Animation):
    """Move ``scene``'s focus from where it is when the play begins to the
    depth ``to``, and its aperture to ``aperture`` (kept when ``None``), over
    the play. The scene needs a focus first (:func:`~.layers.set_focus`).
    """

    def __init__(
        self,
        scene: Scene,
        to: float,
        *,
        aperture: float | None = None,
        **kwargs: Any,
    ) -> None:
        _camera_of(scene, "RackFocus")
        self.to = _finite(to, "A depth")
        self.aperture = None if aperture is None else _finite(aperture, "An aperture")
        if self.aperture is not None and self.aperture < 0:
            raise ValueError(f"An aperture is 0 or more, not {aperture!r}.")
        super().__init__(None, **kwargs)
        self._scene = weakref.ref(scene)
        self._from = (0.0, 0.0)

    def _camera(self) -> Camera:
        scene = self._scene()
        assert scene is not None, "a play's scene outlives it"
        return _camera_of(scene, "RackFocus")

    def create_starting_mobject(self) -> Mobject:
        return self.mobject

    def begin(self) -> None:
        found = seam.focus_at(vars(self._camera()))
        if found is None:
            raise ValueError(
                "RackFocus moves the focus the scene has, and it has none: give"
                " it one first with set_focus(scene, depth, aperture)"
            )
        self._from = found
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        progress = _progress(self, alpha)
        depth, aperture = self._from
        to_aperture = aperture if self.aperture is None else self.aperture
        seam.set_focus(
            self._camera(),
            _between(depth, self.to, progress),
            max(0.0, _between(aperture, to_aperture, progress)),
        )
