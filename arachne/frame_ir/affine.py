"""A small path as a canonical shape and a placement: the numeric contract
instancing batches on (:mod:`.instancing`).

:func:`factor` writes a path's points ``P`` as ``o + M C``: ``o`` its first
point, ``M`` a frame taken from its own points, ``C`` its canonical shape,
so every affine image of one path has one canonical shape. ``C`` is the
shape emitted: rounded to :data:`PROTOTYPE_GRID`, a grid fine enough that
the shapes of one family's members are one shape (each its canonical
values, a float's noise apart, rounded alike) and coarse enough to stand
far above that noise. The contract is read on what is emitted: ``C``
placed by ``[M | o]`` as the core places a point (:func:`placed`) stands
within :data:`RECONSTRUCTION` scene units of ``P``, coordinate by
coordinate, or the path is not factored. So the placement's magnification
of ``C``'s rounding (about ``|M| PROTOTYPE_GRID / 2``) is inside the bound,
and a path placed past about a thousand scene units a side is refused.

The placement itself is snapped afterwards as any placement is
(:mod:`.canonical`: its linear part a ratio, its translation a position),
as the leaf's own points would have been.

Elementwise IEEE arithmetic alone, no reduction a vector unit or a BLAS
kernel orders, so the factoring is the same on every machine.
"""

from __future__ import annotations

__all__ = [
    "MOST_FACTORED_POINTS",
    "PROTOTYPE_GRID",
    "RECONSTRUCTION",
    "Factored",
    "factor",
    "placed",
]

from dataclasses import dataclass
from typing import Any

import numpy as np

#: The most points a path is factored with: a small path's (an arrow's shaft
#: has 4, its tip 12, a dot 32).
MOST_FACTORED_POINTS = 256
#: How far, in scene units, a factored path's emitted shape placed back may
#: stand from its own points, in any coordinate.
RECONSTRUCTION = 1e-9
#: The grid a canonical shape is emitted on.
PROTOTYPE_GRID = 2.0**-40
#: How far, relative to a path's extent, a point must stand from the line or
#: plane of the frame's earlier axes to give it another axis.
_RANK = 1e-6


@dataclass(frozen=True)
class Factored:
    """A path's points as its canonical ``shape`` (``(n, 3)``, on
    :data:`PROTOTYPE_GRID`) placed by the affine ``place`` (``(3, 4)``):
    :func:`placed`.
    """

    shape: np.ndarray
    place: np.ndarray


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _norm(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _first(rows: np.ndarray, size: np.ndarray, bound: float) -> int | None:
    """The first row whose ``size`` exceeds ``bound``."""
    found = np.flatnonzero(size > bound)
    return int(found[0]) if len(found) else None


def factor(points: Any) -> Factored | None:
    """``points`` (``(n, 3)``) factored into a canonical shape and a
    placement, or ``None``: too many points, a value not finite, or the
    shape emitted (rounded to :data:`PROTOTYPE_GRID`) placed back off the
    points by more than :data:`RECONSTRUCTION`.

    The frame is taken from the points in their order, so every affine
    image of one path gets one canonical shape: its origin the first point;
    its first axis the first point apart from it; its second the first point
    off that axis's line, its third the first off their plane, each by more
    than :data:`_RANK` of the path's extent. An axis a path does not span
    (a line spans one, a flat shape two) is made perpendicular to those it
    spans, as long as the first, and the canonical shape is 0 along it.
    Elementwise IEEE arithmetic alone, no reduction a vector unit or a BLAS
    kernel orders, so the factoring is the same on every machine.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not 0 < len(p) <= MOST_FACTORED_POINTS:
        return None
    if not np.isfinite(p).all():
        return None
    origin = p[0].copy()
    d = p - origin
    size = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    extent = float(size.max())
    if extent == 0.0:
        # A path at one point: placed there, its shape at the origin.
        place = np.zeros((3, 4))
        place[:, :3] = np.eye(3)
        place[:, 3] = origin
        return Factored(np.zeros_like(p), place)
    i = _first(d, size, _RANK * extent)
    assert i is not None
    u = d[i]
    across = np.array([_norm(_cross(u, row)) for row in d])
    j = _first(d, across, _RANK * _norm(u) * extent)
    if j is None:
        # A line: any perpendicular, as long as u.
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(u)))] = 1.0
        v = _cross(u, axis)
        v = v * (_norm(u) / _norm(v))
    else:
        v = d[j]
    n = _cross(u, v)
    beyond = np.abs(d[:, 0] * n[0] + d[:, 1] * n[1] + d[:, 2] * n[2])
    k = _first(d, beyond, _RANK * _norm(n) * extent)
    w = n * (_norm(u) / _norm(n)) if k is None else d[k]
    m = np.column_stack([u, v, w])
    # m's inverse by its adjugate, entry by entry.
    cof = np.array(
        [
            [
                m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
                m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
                m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
            ],
            [
                m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2],
                m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
                m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
            ],
            [
                m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],
                m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
                m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
            ],
        ]
    )
    det = m[0, 0] * cof[0, 0] + m[0, 1] * cof[1, 0] + m[0, 2] * cof[2, 0]
    if not (det != 0.0 and np.isfinite(det)):
        return None
    inverse = cof / det
    shape = _apply(inverse, d)
    if k is None:
        shape[:, 2] = 0.0
    if j is None:
        shape[:, 1] = 0.0
    # The shape emitted is rounded to its grid; the placement magnifies that
    # rounding, so the bound is read on what is emitted, placed as the core
    # places it.
    shape = _snap(shape, PROTOTYPE_GRID)
    place = np.zeros((3, 4))
    place[:, :3] = m
    place[:, 3] = origin
    if not float(np.abs(placed(place, shape) - p).max()) <= RECONSTRUCTION:
        return None
    return Factored(shape, place)


def placed(place: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``rows`` (``(n, 3)``) placed by the affine ``place`` (``(3, 4)``) as
    the core places a point (``xf::place``): each coordinate
    ``((a x + b y) + c z) + t``, in that order, no product fused.
    """
    return _apply(place[:, :3], rows) + place[:, 3]


def _apply(m: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``m @ row`` for each row, as the core places a point: each output the
    sum of its three products in order.
    """
    return np.column_stack(
        [
            m[r, 0] * rows[:, 0] + m[r, 1] * rows[:, 1] + m[r, 2] * rows[:, 2]
            for r in range(3)
        ]
    )


def _snap(values: np.ndarray, grid: float) -> np.ndarray:
    return np.round(values / grid) * grid + 0.0
