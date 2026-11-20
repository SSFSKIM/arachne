"""A rig's geometry: its art, computed whole from rest by its parameter
values, its slots' choices and where it is placed.

Everything here is in canvas coordinates (the part files' user units, y
down) until the last step, the placement ``F``: the affine map from the
canvas to the scene that the rig's carrier holds (:mod:`.rig`). For the part
``i``, :func:`posed_shapes` gives

    points_i = F( G_i( W_i( S_i( rest_i ) ) ) )

- ``S_i``: the part's stored outline at its ``shape`` driver's value,
  interpolated point to point between the two keys about it, in place of
  the rest outline (a part has at most one);
- ``W_i``: the bilinear displacement of the part's lattice, the offsets of
  its ``warp`` drivers added, each interpolated between its keys;
- ``G_i = G_parent(i) ∘ L_i``, ``L_i`` the part's own transform about its
  pivot: scale, then rotation, then translation, the outputs of its drivers
  of one kind added (angles, translations) or multiplied (scales). An angle
  turns the part counterclockwise as drawn.

Every computation starts from rest, so a pose depends on the values alone,
never on the poses before it. A slot is placed as a part is (it takes no
``warp`` or ``shape``), and its shown drawing is drawn under its ``G``.

A :class:`Model` holds what an asset's rigs share and never write: the rest
outlines, the stored outlines and lattices as read-only arrays, the parts in
an order that puts every parent before its children, and the canvas points
the carrier holds the images of.
"""

from __future__ import annotations

__all__ = [
    "CARRIER_WIDTH",
    "CanvasMap",
    "Model",
    "RigValues",
    "driver_output",
    "model_of",
    "own_transform",
    "posed_shapes",
    "posed_with",
    "shape_weight",
    "shaped",
    "transforms",
    "warped",
]

import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from ..mobject.values import FrozenArray
from .manifest import DriverSpec, Manifest, ParameterSpec, PartSpec, SlotSpec
from .reader import Asset

#: How wide, in scene units, the triangle the carrier holds is at the
#: asset's own size: small enough that a rig's box is its art's.
CARRIER_WIDTH = 1e-4

#: An affine map of the canvas: ``(2, 3)``, applied as ``m[:, :2] @ p +
#: m[:, 2]``.
Affine: TypeAlias = np.ndarray

_IDENTITY: Affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_IDENTITY.flags.writeable = False


class RigValues(FrozenArray):
    """A rig's parameter values, in its manifest's order."""

    value_name = "A rig's parameter values"
    replacement = "pose the rig with rig.pose(name=value) or the animation Pose"


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array, dtype=float)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class Model:
    """What every rig of one asset shares, read once from it.

    ``order`` names every part and slot, parents first. ``rest`` holds each
    part's shapes, and ``drawings`` each slot's drawings' shapes (by slot,
    then drawing), as ``(n, 2)`` canvas points; ``outlines`` each reshaped
    part's stored outlines, by the keys of its ``shape`` driver.
    ``drivers`` holds, for each part or slot, the drivers that act on it with
    the index of their parameter. ``carrier`` is the canvas point ``c`` and
    the length ``e`` whose images ``F(c)``, ``F(c + (e, 0))`` and
    ``F(c + (0, e))`` the carrier holds, fixed for the asset: ``c`` is the
    mean of the art's points at the default pose, so its image lies inside
    the art's hull at that pose whatever ``F`` is, and ``e`` makes the
    triangle :data:`CARRIER_WIDTH` scene units wide at the asset's size.
    """

    asset: Asset
    order: tuple[str, ...]
    placed: Mapping[str, PartSpec | SlotSpec]
    parameters: tuple[ParameterSpec, ...]
    index: Mapping[str, int]
    defaults: np.ndarray
    rest: Mapping[str, tuple[np.ndarray, ...]]
    drawings: Mapping[str, Mapping[str, tuple[np.ndarray, ...]]]
    outlines: Mapping[str, tuple[tuple[float, tuple[np.ndarray, ...]], ...]]
    drivers: Mapping[str, tuple[tuple[int, DriverSpec], ...]]
    carrier: tuple[np.ndarray, float]

    def __copy__(self) -> Model:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Model:
        return self

    @property
    def natural(self) -> CanvasMap:
        """``F`` as the manifest's ``size`` places the canvas: the canvas
        point ``origin`` at the scene's origin, ``scene_units_per_unit``
        scene units to a canvas unit, y up.
        """
        size = self.asset.manifest.size
        scale = size.scene_units_per_unit
        origin_x, origin_y = size.origin
        return CanvasMap(
            np.array([[scale, 0.0], [0.0, -scale], [0.0, 0.0]]),
            np.array([-origin_x * scale, origin_y * scale, 0.0]),
        )


#: Each asset's model, made with its first rig.
_MODELS: weakref.WeakKeyDictionary[Asset, Model] = weakref.WeakKeyDictionary()


def model_of(asset: Asset) -> Model:
    """The model of ``asset``, shared by all its rigs."""
    model = _MODELS.get(asset)
    if model is None:
        model = _MODELS[asset] = _model(asset)
    return model


def _canvas_points(shapes: Sequence[Any]) -> tuple[np.ndarray, ...]:
    return tuple(_frozen(np.asarray(shape.points)[:, :2]) for shape in shapes)


def _model(asset: Asset) -> Model:
    manifest = asset.manifest
    placed: dict[str, PartSpec | SlotSpec] = {p.name: p for p in manifest.parts}
    placed |= {s.name: s for s in manifest.slots}
    order: list[str] = []

    def visit(name: str) -> None:
        if name in order:
            return
        parent = placed[name].parent
        if parent is not None:
            visit(parent)
        order.append(name)

    for name in placed:
        visit(name)
    drivers: dict[str, list[tuple[int, DriverSpec]]] = {name: [] for name in placed}
    outlines: dict[str, tuple[tuple[float, tuple[np.ndarray, ...]], ...]] = {}
    for index, parameter in enumerate(manifest.parameters):
        for driver in parameter.drives:
            drivers[driver.part].append((index, driver))
            if driver.kind == "shape":
                outlines[driver.part] = tuple(
                    (key.at, _canvas_points(asset.outline(str(key.value))))
                    for key in driver.keys
                )
    rest = {p.name: _canvas_points(asset.rest(p.name)) for p in manifest.parts}
    drawings = {
        slot.name: MappingProxyType(
            {
                d.name: _canvas_points(asset.drawing(slot.name, d.name))
                for d in slot.drawings
            }
        )
        for slot in manifest.slots
    }
    scale = manifest.size.scene_units_per_unit
    model = Model(
        asset=asset,
        order=tuple(order),
        placed=MappingProxyType(placed),
        parameters=manifest.parameters,
        index=MappingProxyType({p.name: i for i, p in enumerate(manifest.parameters)}),
        defaults=_frozen(np.array([p.default for p in manifest.parameters])),
        rest=MappingProxyType(rest),
        drawings=MappingProxyType(drawings),
        outlines=MappingProxyType(outlines),
        drivers=MappingProxyType({k: tuple(v) for k, v in drivers.items()}),
        carrier=(np.zeros(2), CARRIER_WIDTH / scale),
    )
    # The carrier stands at the mean of the art's points at the default
    # pose: inside the art's hull wherever an affine map puts it.
    affines = transforms(model, model.defaults)
    shown = [
        points
        for name in model.order
        for points in posed_with(
            model,
            name,
            model.defaults,
            affines[name],
            None if name in rest else _default_drawing(manifest, name),
        )
    ]
    centre = _frozen(np.concatenate(shown).mean(axis=0))
    return replace(model, carrier=(centre, CARRIER_WIDTH / scale))


def _default_drawing(manifest: Manifest, slot: str) -> str:
    """The default drawing of the slot ``slot``."""
    return next(s.default for s in manifest.slots if s.name == slot)


# Drivers


def _between(ats: Sequence[float], value: float) -> tuple[int, float]:
    """The key ``k`` and weight ``w`` that put ``value`` at ``1 - w`` of the
    key ``k`` and ``w`` of the next, among keys at ``ats`` (rising);
    exactly a key at its own ``at``.
    """
    k = 0
    while k < len(ats) - 2 and value > ats[k + 1]:
        k += 1
    low, high = ats[k], ats[k + 1]
    if value <= low:
        return k, 0.0
    if value >= high:
        return k, 1.0
    return k, (value - low) / (high - low)


def _blend(a: Any, b: Any, w: float) -> Any:
    if w == 0.0:
        return a
    if w == 1.0:
        return b
    return (1 - w) * np.asarray(a, dtype=float) + w * np.asarray(b, dtype=float)


def driver_output(driver: DriverSpec, value: float) -> Any:
    """``driver``'s output at the parameter value ``value``: interpolated
    linearly between the keys about it, a key's own at its ``at``. A
    rotation is in radians, a translation and warp offsets in canvas units.
    """
    k, w = _between([key.at for key in driver.keys], value)
    out = _blend(driver.keys[k].value, driver.keys[k + 1].value, w)
    if driver.kind == "rotate" and driver.unit == "deg":
        return math.radians(float(out))
    return out


def own_transform(model: Model, name: str, values: np.ndarray) -> Affine:
    """``L``: the own transform of the part or slot ``name`` at ``values``,
    about its pivot: scale, then rotation, then translation.
    """
    acting = model.drivers[name]
    if not acting:
        return _IDENTITY
    angle, scale = 0.0, 1.0
    shift = np.zeros(2)
    for index, driver in acting:
        if driver.kind in ("warp", "shape"):
            continue
        out = driver_output(driver, float(values[index]))
        if driver.kind == "rotate":
            angle += float(out)
        elif driver.kind == "translate":
            shift = shift + np.asarray(out, dtype=float)
        else:
            scale *= float(out)
    cos, sin = math.cos(angle), math.sin(angle)
    # Counterclockwise as drawn: the canvas's y points down.
    linear = scale * np.array([[cos, sin], [-sin, cos]])
    pivot = np.asarray(model.placed[name].pivot, dtype=float)
    offset = pivot - linear @ pivot + shift
    return np.column_stack([linear, offset])


def _compose(outer: Affine, inner: Affine) -> Affine:
    """``outer ∘ inner``."""
    linear = outer[:, :2] @ inner[:, :2]
    return np.column_stack([linear, outer[:, :2] @ inner[:, 2] + outer[:, 2]])


def transforms(model: Model, values: np.ndarray) -> dict[str, Affine]:
    """``G`` of every part and slot at ``values``: its own transform after
    its parent's.
    """
    found: dict[str, Affine] = {}
    for name in model.order:
        own = own_transform(model, name, values)
        parent = model.placed[name].parent
        found[name] = own if parent is None else _compose(found[parent], own)
    return found


def _apply(affine: Affine, points: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def _offsets(model: Model, part: str, values: np.ndarray) -> np.ndarray | None:
    """The sum of the ``warp`` drivers' lattice offsets on ``part`` at
    ``values``, or ``None`` when it has none.
    """
    total = None
    for index, driver in model.drivers[part]:
        if driver.kind == "warp":
            out = np.asarray(driver_output(driver, float(values[index])), dtype=float)
            total = out if total is None else total + out
    return total


def warped(
    model: Model, part: str, values: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """``W``: ``points`` (canvas, ``(n, 2)``) of ``part`` displaced by its
    lattice at ``values``: the offsets of the cell a point lies in,
    blended bilinearly; a point beyond the lattice takes the displacement of
    its nearest edge.
    """
    offsets = _offsets(model, part, values)
    if offsets is None:
        return points
    spec = model.placed[part]
    assert isinstance(spec, PartSpec)
    assert spec.lattice is not None
    lattice = spec.lattice
    x0, y0, x1, y1 = lattice.bounds
    cols, rows = lattice.cols, lattice.rows
    u = np.clip((points[:, 0] - x0) / (x1 - x0) * cols, 0, cols)
    v = np.clip((points[:, 1] - y0) / (y1 - y0) * rows, 0, rows)
    i = np.minimum(np.floor(u).astype(int), cols - 1)
    j = np.minimum(np.floor(v).astype(int), rows - 1)
    fu, fv = (u - i)[:, None], (v - j)[:, None]
    grid = offsets.reshape(rows + 1, cols + 1, 2)
    moved = (
        (1 - fu) * (1 - fv) * grid[j, i]
        + fu * (1 - fv) * grid[j, i + 1]
        + (1 - fu) * fv * grid[j + 1, i]
        + fu * fv * grid[j + 1, i + 1]
    )
    return points + moved


def shaped(model: Model, part: str, values: np.ndarray) -> tuple[np.ndarray, ...]:
    """``S``: the outline of ``part`` at ``values``: its stored outline at
    its ``shape`` driver's value, else its rest outline.
    """
    stored = model.outlines.get(part)
    if stored is None:
        return model.rest[part]
    index, driver = next((i, d) for i, d in model.drivers[part] if d.kind == "shape")
    k, w = _between([at for at, _ in stored], float(values[index]))
    low, high = stored[k][1], stored[k + 1][1]
    return tuple(_blend(a, b, w) for a, b in zip(low, high, strict=True))


def shape_weight(
    model: Model, part: str, values: np.ndarray
) -> tuple[int, float] | None:
    """Where ``part``'s ``shape`` driver stands among its keys at
    ``values``: the key and the weight of the next, as :func:`shaped` blends
    them; ``None`` for a part with no ``shape`` driver.
    """
    for index, driver in model.drivers[part]:
        if driver.kind == "shape":
            return _between([key.at for key in driver.keys], float(values[index]))
    return None


def posed_shapes(
    model: Model, name: str, values: np.ndarray, drawing: str | None = None
) -> tuple[np.ndarray, ...]:
    """The canvas points of each shape of the part ``name``, or of the
    slot ``name``'s ``drawing``, posed at ``values``: ``G(W(S(rest)))``.
    """
    affine = transforms(model, values)[name]
    return posed_with(model, name, values, affine, drawing)


def posed_with(
    model: Model,
    name: str,
    values: np.ndarray,
    affine: Affine,
    drawing: str | None = None,
) -> tuple[np.ndarray, ...]:
    """:func:`posed_shapes` with ``G`` given."""
    if drawing is not None:
        shapes = model.drawings[name][drawing]
        return tuple(_apply(affine, points) for points in shapes)
    return tuple(
        _apply(affine, warped(model, name, values, points))
        for points in shaped(model, name, values)
    )


# The placement


@dataclass(frozen=True)
class CanvasMap:
    """``F``, an affine map of the canvas into the scene: a canvas point
    ``p`` goes to ``linear @ p + offset``, ``linear`` ``(3, 2)``.
    """

    linear: np.ndarray
    offset: np.ndarray

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The scene points of the canvas points ``points``, ``(n, 2)``."""
        return points @ self.linear.T + self.offset

    def placed_like(self, model: Model, other: Model) -> CanvasMap:
        """``F`` for a rig of ``other`` standing where this map, ``F`` of a
        rig of ``model``, stands: each moved alike from where its own
        manifest's ``size`` places it.
        """
        here, there = model.asset.manifest.size, other.asset.manifest.size
        ratio = there.scene_units_per_unit / here.scene_units_per_unit
        shift = np.array(
            [
                here.origin[0] - there.origin[0] * ratio,
                here.origin[1] - there.origin[1] * ratio,
            ]
        )
        return CanvasMap(self.linear * ratio, self.linear @ shift + self.offset)

    @classmethod
    def from_carrier(
        cls, model: Model, points: Sequence[np.ndarray]
    ) -> CanvasMap | None:
        """``F`` from the carrier's three points (the images of ``c``,
        ``c + (e, 0)`` and ``c + (0, e)``), or ``None`` when they span no
        plane: a rig scaled to nothing.
        """
        centre, length = model.carrier
        p0, p1, p2 = (np.asarray(p, dtype=float) for p in points)
        across, down = (p1 - p0) / length, (p2 - p0) / length
        widest = max(float(np.linalg.norm(across)), float(np.linalg.norm(down)))
        if not math.isfinite(widest) or widest == 0:
            return None
        if float(np.linalg.norm(np.cross(across, down))) <= 1e-12 * widest**2:
            return None
        linear = np.column_stack([across, down])
        return cls(linear, p0 - linear @ centre)

    def carrier(self, model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The carrier's three points under this map."""
        centre, length = model.carrier
        return (
            self(centre[None])[0],
            self((centre + [length, 0.0])[None])[0],
            self((centre + [0.0, length])[None])[0],
        )
