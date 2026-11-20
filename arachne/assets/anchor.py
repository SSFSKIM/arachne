"""Anchors: points of the material of a drawn object, followed as it moves.

An :class:`Anchor` holds a mobject and how a point is attached to it; its
:meth:`~Anchor.point` is the scene point the mobject's state puts there now,
read from that state alone: no scene code runs, so a capture of the record
answers it as the live scene does.

A rig's anchors are its manifest's, each on a named part:

- ``outline``: the anchor lies on the part's rest outline (within
  :data:`ON_OUTLINE` canvas units) and is resolved, when the asset is
  loaded, to one cubic curve of one shape of it and a place along that
  curve. Every stored outline has the rest outline's structure, so that curve
  and place name the same landmark in each: the anchor is that point of the
  part's outline as posed, through ``S``, ``W``, ``G`` and ``F``
  (:mod:`.pose`). A swelling part's tip takes its anchor with it.
- ``interior``: a free point inside the part, carried through ``W``, ``G``
  and ``F``; on a reshaped part, at its keyed positions, interpolated as the
  outline is.

A slot's drawings each have an outline of their own, so in manifest 1.0 an
anchor on a slot is ``interior``, carried through the slot's ``G`` and
``F``; an ``outline`` anchor on a slot is ``asset.manifest_invalid``.
"""

from __future__ import annotations

__all__ = ["ON_OUTLINE", "Anchor", "Inside", "OnOutline", "attachments"]

import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .manifest import AnchorSpec, PartSpec, refuse
from .pose import Model, shape_weight, shaped, transforms, warped

if TYPE_CHECKING:
    from ..mobject.mobject import Mobject
    from .rig import Rig

#: How far, in canvas units, an ``outline`` anchor may lie from its part's
#: rest outline.
ON_OUTLINE = 0.5


class Attachment(Protocol):
    """How a point is attached to a mobject."""

    def point(self, mobject: Any) -> np.ndarray | None:
        """The scene point the mobject's state puts the anchor at now."""


@dataclass(frozen=True, eq=False)
class Anchor:
    """A point of ``mobject``'s material, attached as ``attachment`` says;
    ``name`` is the asset's name for it. An anchor whose attachment is
    ``None`` names nothing that exists: it has no point, and what is
    attached to it draws nothing.

    It is a value: it never changes, and a copy of it is itself, still
    attached to the same mobject.
    """

    mobject: Mobject
    attachment: Attachment | None
    name: str | None = None

    @property
    def resolved(self) -> bool:
        """Whether the anchor names a point that exists."""
        return self.attachment is not None

    def point(self) -> np.ndarray | None:
        """The scene point the anchor is at, from the mobject's state as it
        is; ``None`` for an anchor that is not resolved.
        """
        if self.attachment is None:
            return None
        return self.attachment.point(self.mobject)

    def __copy__(self) -> Anchor:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Anchor:
        return self


def _transform(model: Model, name: str, values: np.ndarray) -> np.ndarray:
    return transforms(model, values)[name]


def _scene(rig: Rig, canvas: np.ndarray) -> np.ndarray:
    placement = rig.placement()
    if placement is None:
        # A rig scaled to nothing stands at one point.
        return np.array(rig.carrier.submobjects[0].points[0], dtype=float)
    return placement(canvas[None])[0]


@dataclass(frozen=True)
class OnOutline:
    """An ``outline`` anchor: the place ``at`` (0 to 1) along the cubic
    ``curve`` of the shape ``shape`` of the part ``part``'s outline.
    """

    part: str
    shape: int
    curve: int
    at: float

    def point(self, mobject: Any) -> np.ndarray | None:
        rig: Rig = mobject
        model, values = rig.model, np.asarray(rig._values)
        outline = shaped(model, self.part, values)[self.shape]
        controls = outline[4 * self.curve : 4 * self.curve + 4]
        controls = warped(model, self.part, values, controls)
        affine = _transform(model, self.part, values)
        controls = controls @ affine[:, :2].T + affine[:, 2]
        return _scene(rig, _bezier(controls, self.at))


@dataclass(frozen=True)
class Inside:
    """An ``interior`` anchor of the part or slot ``part``: at ``at``, or
    on a reshaped part at ``keys`` (one point per key of its ``shape``
    driver).
    """

    part: str
    at: tuple[float, float]
    keys: tuple[tuple[float, float], ...] = ()

    def point(self, mobject: Any) -> np.ndarray | None:
        rig: Rig = mobject
        model, values = rig.model, np.asarray(rig._values)
        place = np.array([self.at], dtype=float)
        weight = shape_weight(model, self.part, values) if self.keys else None
        if weight is not None:
            k, w = weight
            low, high = np.array(self.keys[k]), np.array(self.keys[k + 1])
            place = (low if w == 0 else high if w == 1 else (1 - w) * low + w * high)[
                None
            ]
        if isinstance(model.placed[self.part], PartSpec):
            place = warped(model, self.part, values, place)
        affine = _transform(model, self.part, values)
        return _scene(rig, place[0] @ affine[:, :2].T + affine[:, 2])


def _bezier(controls: np.ndarray, t: float) -> np.ndarray:
    p0, p1, p2, p3 = controls
    s = 1 - t
    return s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3


#: Each model's anchors by name, resolved once.
_ATTACHMENTS: weakref.WeakKeyDictionary[Model, Mapping[str, OnOutline | Inside]] = (
    weakref.WeakKeyDictionary()
)


def attachments(model: Model) -> Mapping[str, OnOutline | Inside]:
    """The attachment of every anchor of ``model``'s asset, by name;
    ``asset.manifest_invalid`` for an ``outline`` anchor on a slot or off
    its part's outline.
    """
    found = _ATTACHMENTS.get(model)
    if found is None:
        found = _ATTACHMENTS[model] = MappingProxyType(
            {
                spec.name: _attach(model, spec, i)
                for i, spec in enumerate(model.manifest.anchors)
            }
        )
    return found


def _attach(model: Model, spec: AnchorSpec, index: int) -> OnOutline | Inside:
    if spec.attach == "interior":
        return Inside(spec.part, spec.at, tuple(k.point for k in spec.keys))
    name = model.name
    if not isinstance(model.placed[spec.part], PartSpec):
        reason = (
            f"the outline anchor {spec.name!r} is on the slot {spec.part!r}, whose"
            " drawings each have an outline of their own: an anchor on a slot is"
            " interior"
        )
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {name} is invalid at anchors[{index}].attach:"
            f" {reason}",
            asset=name,
            member=f"anchors[{index}].attach",
            reason=reason,
            name=spec.name,
        )
    found = _nearest(model.rest[spec.part], np.asarray(spec.at, dtype=float))
    if found is None or found[0] > ON_OUTLINE:
        distance = "far" if found is None else f"{found[0]:.3g} canvas units"
        reason = (
            f"the outline anchor {spec.name!r} at {list(spec.at)} is {distance} from"
            f" the outline of its part {spec.part!r}, beyond {ON_OUTLINE}"
        )
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {name} is invalid at anchors[{index}].at:"
            f" {reason}",
            asset=name,
            member=f"anchors[{index}].at",
            reason=reason,
            name=spec.name,
        )
    _, shape, curve, at = found
    return OnOutline(spec.part, shape, curve, at)


#: How many places along a curve the search for the nearest first reads.
_SAMPLES = 129


def _nearest(
    shapes: tuple[np.ndarray, ...], target: np.ndarray
) -> tuple[float, int, int, float] | None:
    """The point of ``shapes``' cubic curves nearest ``target``: its
    distance, shape, curve and place along the curve; ``None`` for no curve.
    Each curve is read at :data:`_SAMPLES` places, and the nearest refined
    by Newton's method on the distance's derivative, so a landmark on the
    outline is found to the precision of its numbers.
    """
    best: tuple[float, int, int, float] | None = None
    places = np.linspace(0.0, 1.0, _SAMPLES)
    for shape, points in enumerate(shapes):
        for curve, controls in enumerate(points.reshape(-1, 4, 2)):
            samples = np.array([_bezier(controls, t) for t in places])
            at = float(places[int(np.argmin(np.linalg.norm(samples - target, axis=1)))])
            for _ in range(50):
                off = _bezier(controls, at) - target
                d1, d2 = _derivatives(controls, at)
                slope = float(off @ d1)
                curvature = float(d1 @ d1 + off @ d2)
                if curvature <= 0:
                    break
                step = min(max(at - slope / curvature, 0.0), 1.0)
                if step == at:
                    break
                at = step
            distance = float(np.linalg.norm(_bezier(controls, at) - target))
            if best is None or distance < best[0]:
                best = (distance, shape, curve, at)
    return best


def _derivatives(controls: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The first and second derivatives of the cubic ``controls`` at ``t``."""
    p0, p1, p2, p3 = controls
    s = 1 - t
    first = 3 * s**2 * (p1 - p0) + 6 * s * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
    second = 6 * s * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
    return first, second
