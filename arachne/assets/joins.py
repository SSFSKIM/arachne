"""Where a stroke's miter joins are drawn otherwise than its file means.

SVG cuts a miter join to a bevel where the miter's length, over the stroke's
width, exceeds ``stroke-miterlimit`` (4 when the file says nothing); the
engine strokes every miter join at cairo's limit of 10 and carries no limit
of a shape's own. At a corner whose segments meet at the angle ``θ`` that
ratio is ``1 / sin(θ / 2)``, so the two limits disagree exactly on the
corners whose ratio lies between them. :func:`disagreeing_corner` finds one.
"""

from __future__ import annotations

__all__ = ["ENGINE_LIMIT", "disagreeing_corner"]

import math
from collections.abc import Iterator
from itertools import pairwise

import numpy as np

from ..mobject.types.vectorized_mobject import VMobject

#: The miter limit the engine strokes every miter join with: cairo's own.
ENGINE_LIMIT = 10.0
#: How much longer than its own a miter must be to count as past a limit,
#: so a smooth joint (ratio 1) never sits in a band whose floor is 1.
_SLACK = 1e-9
#: How long a tangent must be to have a direction.
_EPSILON = 1e-9


def disagreeing_corner(
    shape: VMobject, limit: float
) -> tuple[float, np.ndarray] | None:
    """The first corner of ``shape``'s outline whose miter the file's
    ``limit`` and :data:`ENGINE_LIMIT` treat apart, as the angle its
    segments meet at, in degrees, and the point; ``None`` when there is none.

    A corner is any joint where the tangent turns: between two segments of a
    subpath, and at the closing point of a subpath the engine closes.
    """
    low, high = sorted((limit, ENGINE_LIMIT))
    if low == high:
        return None
    for incoming, outgoing, point in _joints(shape):
        cosine = float(np.dot(-incoming, outgoing))
        angle = math.acos(min(1.0, max(-1.0, cosine)))
        half = math.sin(angle / 2)
        ratio = math.inf if half == 0 else 1 / half
        if low * (1 + _SLACK) < ratio <= high:
            return math.degrees(angle), point
    return None


def _joints(shape: VMobject) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Each joint's incoming and outgoing unit tangents, and its point."""
    for subpath in shape.get_subpaths():
        curves = [
            (start, end, curve[3])
            for curve in np.reshape(subpath[:, :2], (-1, 4, 2))
            if (start := _direction(curve[1:] - curve[0])) is not None
            and (end := _direction(curve[3] - curve[2::-1])) is not None
        ]
        for (_, incoming, point), (outgoing, _, _) in pairwise(curves):
            yield incoming, outgoing, point
        if curves and shape.consider_points_equals_2d(subpath[0], subpath[-1]):
            yield curves[-1][1], curves[0][0], subpath[0, :2]


def _direction(vectors: np.ndarray) -> np.ndarray | None:
    """The first of ``vectors`` long enough to have a direction, made unit."""
    for vector in vectors:
        length = float(np.hypot(*vector))
        if length > _EPSILON:
            return vector / length
    return None
