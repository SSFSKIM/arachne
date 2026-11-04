"""A path record's geometry made into points: a blend of two shapes, or a
reveal window of one, computed by the engine's own functions so that they
reproduce the points the scene's run computed bit for bit. The drawer
draws with these, and extraction states a blend or a reveal only where
they give the leaf's points exactly.
"""

from __future__ import annotations

__all__ = ["blended", "revealed"]

from typing import Any

import numpy as np

from arachne.mobject.types.vectorized_mobject import VMobject


class _Path(VMobject):
    """Bare points for the engine's path functions to work on."""


def _bare(points: Any) -> Any:
    path: Any = object.__new__(_Path)
    path.__dict__.update(
        _points=np.asarray(points, dtype=float),
        dim=3,
        n_points_per_cubic_curve=4,
        submobjects=[],
    )
    return path


def blended(a: Any, b: Any, alpha: float) -> np.ndarray:
    """``a`` and ``b`` blended on a straight path at ``alpha`` (CE's
    ``interpolate``).
    """
    from arachne.utils.bezier import interpolate

    return np.asarray(interpolate(np.asarray(a), np.asarray(b), alpha))


def revealed(points: Any, lo: float, hi: float) -> np.ndarray:
    """The part of the path ``points`` between ``lo`` and ``hi``, as
    ``VMobject.pointwise_become_partial`` makes it of a path holding no
    points before.
    """
    target = _bare(np.zeros((0, 3)))
    VMobject.pointwise_become_partial(target, _bare(points), lo, hi)
    return np.asarray(target.points)
