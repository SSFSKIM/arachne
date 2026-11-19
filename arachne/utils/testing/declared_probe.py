"""Engine-owned objects that declare quantities, for the engine's own tests.

None of them is a simulation and no scene of a lesson uses them; they are
the engine's, so the hook of :mod:`arachne.quantities` serves them:

- :class:`DeclaredProbe`, a small square whose provider declares a number
  (``probe.amount``) and a name value (``probe.setting``) from its own
  attributes;
- :class:`BakedNumber`, an attached source reading one number of a bake's
  channel at the instant's tick;
- :class:`ProbeGrid`, a square standing for a grid of cells whose values a
  bake holds by tick: a point source, read nearest or linear;
- :func:`bake_rows`, which puts a bake of given rows into the record of the
  scene running now.

Their names are under the prefix ``probe.``, which this module owns.
"""

from __future__ import annotations

__all__ = [
    "PREFIX",
    "BakedNumber",
    "DeclaredProbe",
    "ProbeGrid",
    "bake_rows",
]

import hashlib
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arachne import quantities
from arachne.mobject.geometry.polygram import Square
from arachne.quantities import At, PointRead, Quantity
from arachne.timeline.bake import BakeContent, BakeIdentity, BakeRef, Channel, bakes_of
from arachne.timeline.recorder import running

#: The prefix of the probes' names.
PREFIX = "probe."
quantities.register(PREFIX, owner="arachne.utils.testing", units=("K",))

#: What made a probe's bake.
IDENTITY = BakeIdentity(
    producer="arachne.utils.testing",
    producer_version="1",
    solver="rows given",
    solver_version="1",
    integrator="none",
    build="python",
)


class DeclaredProbe(Square):
    """A small square declaring ``probe.amount`` (a number, to 0.01) and
    ``probe.setting`` (a name) from its attributes ``amount`` and
    ``setting``.
    """

    def __init__(self, amount: float = 0.0, setting: str = "off", **kwargs: Any):
        super().__init__(side_length=0.5, **kwargs)
        self.amount = amount
        self.setting = setting

    def set_amount(self, amount: float) -> DeclaredProbe:
        self.amount = amount
        return self

    def set_setting(self, setting: str) -> DeclaredProbe:
        self.setting = setting
        return self


@quantities.provider(DeclaredProbe)
def _declared(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    return (
        Quantity("probe.amount", held["amount"], "1", tolerance=0.01),
        Quantity("probe.setting", held["setting"], "name"),
    )


@dataclass(frozen=True, slots=True)
class BakedNumber:
    """A source: the number at ``cell`` of the bake's ``channel`` at the
    instant's tick, as ``name`` in ``unit``; none outside the bake's stay.
    """

    bake: BakeRef
    channel: str
    name: str
    unit: str
    cell: tuple[int, ...] = ()


@quantities.source(BakedNumber)
def _baked(found: BakedNumber, held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    row = None if at.bakes is None else at.bakes.row(found.bake.ordinal, at.tick)
    if row is None:
        reason = f"no bake holds tick {at.tick}"
        return (Quantity(found.name, None, found.unit, unreadable=reason),)
    return (Quantity(found.name, float(row[found.channel][found.cell]), found.unit),)


class ProbeGrid(Square):
    """A square standing for a grid of cells, the values of its channel
    ``channel`` (in ``unit``) a bake holds by tick, ``[nx, ny, nz]`` a row
    for a scalar channel and ``[nx, ny, nz, n]`` for a vector one.
    Cell ``(i, j, k)`` is centred at ``origin + spacing * (i, j, k)`` in
    scene units. Read ``"nearest"``, its domain is the cells' extents; read
    ``"linear"``, the span between the outermost cells' centres.
    """

    def __init__(
        self,
        bake: BakeRef,
        *,
        channel: str,
        unit: str,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        spacing: float = 1.0,
        interpolation: str = "nearest",
        **kwargs: Any,
    ):
        super().__init__(side_length=0.5, **kwargs)
        self.bake = bake
        self.channel = channel
        self.unit = unit
        self.grid_origin = tuple(float(v) for v in origin)
        self.spacing = float(spacing)
        self.interpolation = interpolation


def _grid_channels(held: Mapping[str, Any]) -> Mapping[str, str]:
    return {held["channel"]: held["unit"]}


def _grid_value(
    held: Mapping[str, Any],
    at: At,
    channel: str,
    position: tuple[float, ...],
    space: str,
) -> PointRead:
    bake: BakeRef = held["bake"]
    row = None if at.bakes is None else at.bakes.row(bake.ordinal, at.tick)
    if row is None:
        return PointRead("outside", why="time")
    values = row[channel]
    point = (*position, *(0.0,) * (3 - len(position)))
    if space == "grid":
        grid = point
    else:
        grid = tuple(
            (p - o) / held["spacing"]
            for p, o in zip(point, held["grid_origin"], strict=True)
        )
    shape = values.shape[:3]
    interpolation = held["interpolation"]
    if interpolation == "nearest":
        cell = tuple(math.floor(g + 0.5) for g in grid)
        if not all(0 <= i < n for i, n in zip(cell, shape, strict=True)):
            return PointRead("outside", why="domain")
        return PointRead(
            "value", _plain(values[cell]), interpolation="nearest", cells=(cell,)
        )
    if not all(0 <= g <= n - 1 for g, n in zip(grid, shape, strict=True)):
        return PointRead("outside", why="domain")
    axes = []
    for g, n in zip(grid, shape, strict=True):
        low = min(math.floor(g), n - 2) if n > 1 else 0
        part = g - low
        axes.append(((low, 1.0 - part), (low + 1, part)) if n > 1 else ((0, 1.0),))
    total = np.zeros(values.shape[3:])
    cells = []
    for corner in itertools.product(*axes):
        weight = math.prod(w for _, w in corner)
        if weight > 0:
            cell = tuple(i for i, _ in corner)
            cells.append(cell)
            total = total + weight * values[cell]
    return PointRead(
        "value", _plain(total), interpolation="linear", cells=tuple(sorted(cells))
    )


def _plain(found: np.ndarray) -> float | tuple[float, ...]:
    """A cell's value: a number, or a vector's numbers."""
    if np.ndim(found) == 0:
        return float(found)
    return tuple(float(v) for v in np.ravel(found))


quantities.point_source(ProbeGrid, channels=_grid_channels, value_at=_grid_value)


def bake_rows(
    rows: Sequence[Any], *, channel: str, unit: str, first_tick: int = 0
) -> BakeRef:
    """Put a bake into the record of the scene running now: ``rows`` (one
    array a tick, from ``first_tick``) as the channel ``channel`` in
    ``unit``, sealed at once; its reference.
    """
    recorder = running()
    if recorder is None:
        raise RuntimeError("A probe's bake is made while a scene's script runs.")
    bakes = bakes_of(recorder.ensure_record())
    values = np.asarray(rows, dtype=float)
    content = BakeContent(
        IDENTITY, len(values), {channel: Channel("f64", unit, values)}
    )
    declaration = hashlib.blake2b(content.key, digest_size=16).digest()
    ordinal = bakes.begin()
    bakes.seal(
        ordinal, content, first_tick=first_tick, closed=True, declaration=declaration
    )
    return BakeRef(ordinal)
