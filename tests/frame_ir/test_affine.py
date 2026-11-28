"""A small path factored into a canonical shape and a placement
(:mod:`arachne.frame_ir.affine`): the numeric contract instancing batches on.

The shape factor returns is the one emitted, on :data:`PROTOTYPE_GRID`; placed
as the core places a point, it reproduces the path's points within
:data:`RECONSTRUCTION` scene units, the placement's magnification of the
shape's rounding included, or the path is not factored.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import Circle, Triangle
from arachne.frame_ir import affine
from arachne.frame_ir.affine import (
    MOST_FACTORED_POINTS,
    PROTOTYPE_GRID,
    RECONSTRUCTION,
    factor,
    placed,
)


def _turned(points: np.ndarray, angle: float, scale: float, shift) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s, 0.2], [s, c, -0.1], [0.3, 0.1, 0.9]]) * scale
    return points @ m.T + np.asarray(shift, dtype=float)


def _line() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [3.0, 1.5, 0.0]])


def _tip() -> np.ndarray:
    """An arrow's tip as CE lays it out: three straight cubics, closed."""
    corners = [(1.0, -0.25, 0.0), (1.45, 0.0, 0.0), (1.0, 0.25, 0.0), (1.0, -0.25, 0.0)]
    return np.concatenate(
        [np.linspace(a, b, 4) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    )


def _within(found: affine.Factored, points: np.ndarray) -> float:
    """How far the emitted shape, placed as the core places it, stands from
    ``points``.
    """
    return float(np.abs(placed(found.place, found.shape) - points).max())


PATHS = {
    "triangle": lambda: Triangle().points,
    "line": _line,
    "tip": _tip,
    "circle": lambda: Circle().points,
}
IMAGES = [(0.7, 0.4, (1, 2, 3)), (-2.1, 3.0, (-4, 0.5, 0)), (1.3, 0.013, (6.5, -3.6, 0))]


@pytest.mark.parametrize("name", PATHS)
@pytest.mark.parametrize("image", IMAGES, ids=["small", "large", "tiny"])
def test_the_emitted_shape_placed_reproduces_the_points_within_the_bound(name, image):
    points = _turned(PATHS[name](), *image)
    found = factor(points)
    assert found is not None
    # The shape is the one emitted: on the prototype's grid already.
    assert np.array_equal(found.shape, np.round(found.shape / PROTOTYPE_GRID) * PROTOTYPE_GRID)
    assert _within(found, points) <= RECONSTRUCTION


@pytest.mark.parametrize("name", ["triangle", "line", "tip"])
def test_affine_images_of_one_path_share_one_emitted_shape(name):
    shapes = []
    for image in IMAGES:
        found = factor(_turned(PATHS[name](), *image))
        assert found is not None
        shapes.append(found.shape)
    for other in shapes[1:]:
        assert np.array_equal(other, shapes[0])


def test_a_placement_that_magnifies_the_shape_s_rounding_past_the_bound_is_refused():
    # Placed at a hundred thousand scene units a side, the shape's rounding on
    # its grid (half of 2^-40) grows past 1e-9: refused, though the shape
    # before its rounding maps back within it.
    points = _turned(Triangle().points, 0.731, 1e5, (0.2, -1.3, 0.4))
    assert factor(points) is None
    assert factor(_turned(Triangle().points, 0.731, 10.0, (0.2, -1.3, 0.4))) is not None


def test_a_path_too_long_or_not_finite_is_not_factored(monkeypatch):
    assert factor(np.zeros((MOST_FACTORED_POINTS + 1, 3))) is None
    assert factor(np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])) is None
    circle = _turned(Circle().points, 0.731, 0.37, (0.2, -1.3, 0.4))
    found = factor(circle)
    assert found is not None
    assert 0 < _within(found, circle) <= RECONSTRUCTION
    monkeypatch.setattr(affine, "RECONSTRUCTION", 0.0)
    assert factor(circle) is None


def test_a_path_at_one_point_is_placed_there():
    dot = factor(np.full((4, 3), 2.5))
    assert dot is not None
    assert not dot.shape.any()
    assert _within(dot, np.full((4, 3), 2.5)) == 0.0


def test_placed_is_the_core_s_arithmetic():
    """Each coordinate ``((a x + b y) + c z) + t``, in that order."""
    place = np.array([[0.1, 0.2, 0.3, 0.4], [1e16, 1.0, -1e16, 0.0], [0.0, 0.0, 1.0, 1e-300]])
    rows = np.array([[1.0, 3.0, 1.0], [7.0, 1e-17, 2.0]])
    want = [
        [((a * r[0] + b * r[1]) + c * r[2]) + t for a, b, c, t in place] for r in rows
    ]
    assert placed(place, rows).tolist() == want
