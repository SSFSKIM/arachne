"""A path of few curves is split anchor by anchor, and the split is the one the
arrays give: the same rows, of the same type, for every kind of anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import VMobject
from arachne.mobject.types import vectorized_mobject as vm

ATOL = VMobject().tolerance_for_point_equality


def _anchors(rng: np.random.Generator, curves: int, dims: int) -> np.ndarray:
    """``curves`` cubic curves whose touching anchors are equal, near, at the
    tolerance's edge, apart or not finite, each kind drawn at random.
    """
    points = rng.normal(scale=rng.choice([1e-3, 1.0, 1e4]), size=(4 * curves, dims))
    for boundary in range(4, 4 * curves, 4):
        start = points[boundary]
        kind = rng.integers(8)
        if kind in (0, 3, 4, 5, 6):
            # Equal anchors, and for kinds 3 to 6 one component then not
            # finite, so that it alone decides the boundary.
            points[boundary - 1] = start
        elif kind == 1:
            points[boundary - 1] = start + rng.normal(scale=ATOL / 4, size=dims)
        elif kind == 2:
            # On either side of the threshold, one ulp apart.
            edge = ATOL + vm._SPLIT_RTOL * np.abs(start)
            points[boundary - 1] = start + np.nextafter(edge, rng.choice([0, np.inf]))
        if kind == 3:
            points[boundary - 1, rng.integers(dims)] = np.nan
        elif kind == 4:
            points[boundary - 1, 0] = points[boundary, 0] = np.inf
        elif kind == 5:
            points[boundary - 1, 0], points[boundary, 0] = np.inf, -np.inf
        elif kind == 6:
            points[boundary, rng.integers(dims)] = rng.choice([np.inf, -np.inf, np.nan])
        # kind 7: apart.
    return points


@pytest.mark.parametrize("dims", [2, 3])
def test_a_few_curves_split_anchor_by_anchor_as_the_arrays_split_them(dims):
    rng = np.random.default_rng(20261106)
    for _ in range(3000):
        curves = int(rng.integers(2, vm._FEW_BOUNDARIES + 2))
        points = _anchors(rng, curves, 3)
        boundaries = np.arange(4, len(points), 4)
        few = vm._split_few(points, boundaries, dims, ATOL)
        many = vm._split_many(points, boundaries, dims, ATOL)
        assert few.dtype == many.dtype
        assert few.shape == many.shape
        assert np.array_equal(few, many)


def test_the_split_takes_the_anchor_by_anchor_way_for_few_curves_only(monkeypatch):
    taken = []
    for name in ("_split_few", "_split_many"):
        original = getattr(vm, name)

        def split(*args, original=original, name=name):
            taken.append(name)
            return original(*args)

        monkeypatch.setattr(vm, name, split)
    path = VMobject()
    few = np.zeros((4 * (vm._FEW_BOUNDARIES + 1), 3))
    path.get_subpath_split_indices_from_points(few)
    path.get_subpath_split_indices_from_points(
        np.zeros((4 * (vm._FEW_BOUNDARIES + 2), 3))
    )
    path.get_subpath_split_indices_from_points(few.astype(np.float32))
    assert taken == ["_split_few", "_split_many", "_split_many"]
