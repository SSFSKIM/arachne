"""Effects a scene sets on an object, one at a time: a blur, a mask, a depth
under the scene's focus.

Each setter replaces one field of the object's :class:`~.compose.LayerSpec`
through :func:`~.compose.set_layer`, and keeps every other field it holds, so
two effects set one after the other are both held. That is the only way
these names write a spec, so the seam's rule holds for them: an object
carries an effect only with its scene routed to the GPU and with what the
effect reads (a mask) recorded with it. An object left with no effect is no
longer a layer of its own.

Numbers are in scene units, as the seam's are: a blur is a Gaussian's
standard deviation, whose edge spreads 2.563 sigma between its 10 % and
90 % levels; a mask's feather is the width of its soft edge; a depth is
blurred by the scene's focus, ``aperture * |depth - focus|``.

An object an effect is set on declares its effects' numbers as quantities
(:mod:`arachne.quantities`), read from the spec each capture records:
``effect.blur``, ``effect.mask_feather`` and ``effect.depth``, in ``u``,
each present while the spec holds its field. The focus is the camera's, and
is no object's quantity. These are quantity names; the routing table's
capability ids (:mod:`arachne.routing`), one of which is also spelled
``effect.blur``, are a namespace of their own, and nothing here reads a name
of one as a name of the other.
"""

from __future__ import annotations

__all__ = ["clear_effects", "set_blur", "set_depth", "set_focus", "set_mask"]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from arachne import quantities
from arachne.compose import layer as seam
from arachne.compose.layer import SPEC, LayerSpec, layer_of
from arachne.quantities import At, Quantity

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject

#: The prefix of the effects' quantities.
PREFIX = "effect."
quantities.register(PREFIX, owner="arachne.effects", units=("u",))


def set_blur(mobject: Mobject, sigma: float | None) -> None:
    """Blur ``mobject`` by a Gaussian of standard deviation ``sigma`` scene
    units; ``None`` or 0 removes its blur.
    """
    _write(mobject, blur=None if sigma is None or sigma == 0 else float(sigma))


def set_mask(
    mobject: Mobject,
    mask: Mobject | None,
    *,
    feather: float = 0.0,
    invert: bool = False,
) -> None:
    """Cut ``mobject`` out by ``mask``'s picture, its edge feathered over
    ``feather`` scene units; ``invert`` keeps what the mask would remove.
    ``None`` removes its mask. The mask is drawn only through ``mobject``,
    and is recorded with it, never added to the scene.
    """
    from arachne.mobject.mobject import Mobject

    if mask is not None and not isinstance(mask, Mobject):
        raise TypeError(f"A mask is a mobject, not {type(mask).__name__}.")
    _write(mobject, mask=mask, mask_feather=float(feather), mask_invert=bool(invert))


def set_depth(mobject: Mobject, depth: float | None) -> None:
    """Place ``mobject`` at the scene depth ``depth``, where the scene's
    focus blurs it (:func:`set_focus`); ``None`` removes its depth.
    """
    _write(mobject, depth=None if depth is None else float(depth))


def clear_effects(mobject: Mobject) -> None:
    """Remove every effect of ``mobject``: it is no longer a layer of its
    own, and the GPU-only capabilities its effects declared are withdrawn.
    The object's rows are one set, owned by no one: a row of another id that
    the object declared itself stays, but one of the same id as an effect's
    is withdrawn with it.
    """
    seam.set_layer(mobject, None)


def set_focus(scene: Any, depth: float, aperture: float) -> None:
    """Focus ``scene``'s camera at ``depth``: an object at depth ``d``
    (:func:`set_depth`) is blurred by ``aperture * |d - depth|`` scene units,
    combined with its own blur. Objects with no depth are not blurred by it.
    """
    seam.set_focus(_camera_of(scene, "set_focus"), depth, aperture)


def _camera_of(scene: Any, asker: str) -> Camera:
    from arachne.camera.camera import Camera

    camera = getattr(scene, "camera", None)
    if not isinstance(camera, Camera):
        raise TypeError(
            f"{asker} takes the scene whose camera it focuses, not"
            f" {type(scene).__name__}."
        )
    return camera


def _write(mobject: Mobject, **fields: Any) -> None:
    """Replace ``fields`` of ``mobject``'s spec, keeping the rest; with no
    effect left, ``mobject`` is no layer of its own.
    """
    from arachne.mobject.mobject import Mobject

    if not isinstance(mobject, Mobject):
        raise TypeError(f"An effect is set on a mobject, not {type(mobject).__name__}.")
    spec = replace(layer_of(mobject) or LayerSpec(), **fields)
    if not spec.capabilities:
        seam.set_layer(mobject, None)
        return
    seam.set_layer(mobject, spec)
    held = mobject.__dict__.get(quantities.SOURCES, ())
    if not any(type(each) is EffectNumbers for each in held):
        quantities.attach(mobject, EffectNumbers())


@dataclass(frozen=True, slots=True)
class EffectNumbers:
    """The source of an object's effect quantities. It holds nothing: what
    it answers is read from the spec the capture records, so a blur set
    twice is one quantity and a cleared effect leaves none.
    """


@quantities.source(EffectNumbers)
def _numbers(
    found: EffectNumbers, held: Mapping[str, Any], at: At
) -> Iterable[Quantity]:
    spec = held.get(SPEC)
    if not isinstance(spec, LayerSpec):
        return ()
    numbers = []
    if spec.blur is not None:
        numbers.append(Quantity(f"{PREFIX}blur", spec.blur, "u"))
    if spec.mask is not None:
        numbers.append(Quantity(f"{PREFIX}mask_feather", spec.mask_feather, "u"))
    if spec.depth is not None:
        numbers.append(Quantity(f"{PREFIX}depth", spec.depth, "u"))
    return numbers
