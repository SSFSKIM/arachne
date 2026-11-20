"""The layer seam: how a recorded object says "I am a layer of my own, with
these effects".

:func:`set_layer` gives an object a :class:`LayerSpec`, the only way one
gets one, so that an object cannot carry an effect without its scene being
routed to the GPU (it declares the spec's capabilities, :mod:`arachne.routing`)
and without what the effect reads being recorded: a mask, or the content of
a window, may never be added to the scene, and is made a capture dependency
of the object (:data:`arachne.timeline.record.DEPENDENCIES`), recorded with
it at every capture that reaches it and painted only through it.
:func:`set_focus` gives a camera the depth in focus, which blurs every layer
with a declared depth by how far it lies from it.

The spec and what it names are recorded state, and its numbers are in scene
units and scene depth, so a 480p draft and a 1080p final blur the same share
of the picture (:func:`pixels`).
"""

from __future__ import annotations

__all__ = [
    "FOCUS_APERTURE",
    "FOCUS_DEPTH",
    "SPEC",
    "LayerSpec",
    "focus_at",
    "layer_of",
    "pixels",
    "set_focus",
    "set_layer",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arachne import routing
from arachne.timeline.record import DEPENDENCIES

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject

#: The attribute of an object holding its :class:`LayerSpec`.
SPEC = "_layer_spec"
#: The camera's attributes holding the depth in focus and the aperture
#: (:func:`set_focus`): plain numbers, so the record keeps them with the
#: camera's other values at every capture.
FOCUS_DEPTH = "_focus_depth"
FOCUS_APERTURE = "_focus_aperture"


@dataclass(frozen=True)
class LayerSpec:
    """What a layer of its own does to the object it is set on.

    ``blur`` is a Gaussian's standard deviation in scene units. ``mask`` is
    an object whose picture's alpha, its edge feathered by ``mask_feather``
    scene units, multiplies the layer (``mask_invert`` keeps what it would
    remove). ``depth`` is the layer's scene depth, blurred under the
    camera's focus (:func:`set_focus`). ``window`` holds objects drawn as a
    stack of their own and shown in this object's four corners in place of
    the object's own picture: at the size those corners are shown at in
    pixels, as the objects alone would be drawn at that size, where the
    corners are an axis-aligned rectangle neither mirrored nor turned; at
    the frame's size anywhere else.
    """

    blur: float | None = None
    mask: Mobject | None = None
    mask_feather: float = 0.0
    mask_invert: bool = False
    depth: float | None = None
    window: tuple[Mobject, ...] | None = None

    def __post_init__(self) -> None:
        if self.blur is not None and not (math.isfinite(self.blur) and self.blur > 0):
            raise ValueError(f"A blur is a positive length, not {self.blur!r}.")
        if not (math.isfinite(self.mask_feather) and self.mask_feather >= 0):
            raise ValueError(
                f"A mask's feather is a length, 0 or more, not {self.mask_feather!r}."
            )
        if self.mask is None and (self.mask_feather or self.mask_invert):
            raise ValueError("A mask's feather and inversion need a mask.")
        if self.depth is not None and not math.isfinite(self.depth):
            raise ValueError(f"A depth is a number, not {self.depth!r}.")
        if self.window is not None:
            if not self.window:
                raise ValueError("A window shows at least one object.")
            object.__setattr__(self, "window", tuple(self.window))

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The GPU-only capabilities this spec holds."""
        found = []
        if self.blur is not None:
            found.append("effect.blur")
        if self.mask is not None:
            found.append("effect.soft_mask")
        if self.window is not None:
            found.append("effect.window")
        if self.depth is not None:
            found.append("layer.depth")
        return tuple(found)

    @property
    def dependencies(self) -> tuple[Mobject, ...]:
        """The objects the spec reads that are drawn only through it."""
        found = [] if self.mask is None else [self.mask]
        found.extend(self.window or ())
        return tuple(found)


def set_layer(mobject: Mobject, spec: LayerSpec | None) -> None:
    """Make ``mobject`` a layer of its own drawn as ``spec`` says, or, with
    ``None``, no longer one: the spec recorded on it, its capabilities
    declared (and those the spec it replaces held, and it does not,
    withdrawn), and the objects it reads made its capture dependencies.
    """
    own = vars(mobject)
    before = own.get(SPEC)
    if spec is None:
        own.pop(SPEC, None)
        own.pop(DEPENDENCIES, None)
    else:
        if any(member is mobject for member in spec.dependencies):
            raise ValueError("A layer's mask or window does not hold the layer itself.")
        own[SPEC] = spec
        if spec.dependencies:
            own[DEPENDENCIES] = spec.dependencies
        else:
            own.pop(DEPENDENCIES, None)
    held = set(spec.capabilities) if spec is not None else set()
    if before is not None:
        # Only what the spec it replaces declared: a row the object declared
        # on its own stays.
        gone = [row for row in before.capabilities if row not in held]
        if gone:
            routing.withdraw(mobject, *gone)
    if held:
        routing.declare(mobject, *sorted(held))


def layer_of(mobject: Mobject) -> LayerSpec | None:
    """The spec ``mobject`` holds now, if it is a layer of its own."""
    found: LayerSpec | None = vars(mobject).get(SPEC)
    return found


def set_focus(camera: Camera, depth: float, aperture: float) -> None:
    """Give ``camera`` the scene depth in focus: a layer at depth ``d`` is
    blurred by ``aperture * |d - depth|`` scene units, its own blur combined
    with it. The focus blurs only layers with a declared depth, which hold
    their own capability; it holds none of its own.
    """
    if not (math.isfinite(depth) and math.isfinite(aperture) and aperture >= 0):
        raise ValueError(
            f"A focus is a depth and an aperture of 0 or more: {depth!r}, {aperture!r}."
        )
    vars(camera)[FOCUS_DEPTH] = float(depth)
    vars(camera)[FOCUS_APERTURE] = float(aperture)


def focus_at(values: Mapping[str, Any]) -> tuple[float, float] | None:
    """The depth in focus and the aperture among a camera's recorded
    values, if it has a focus.
    """
    depth, aperture = values.get(FOCUS_DEPTH), values.get(FOCUS_APERTURE)
    if depth is None or aperture is None:
        return None
    return float(depth), float(aperture)


def pixels(camera: Camera, length: float) -> float:
    """``length`` scene units in the camera's pixels: the one conversion of
    the seam's numbers, by the frame's width.
    """
    return length * camera.pixel_width / camera.frame_width
