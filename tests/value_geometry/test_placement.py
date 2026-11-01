"""shift, scale and rotate compose an affine placement over a large mobject's
shape; its points are computed once, when read, and its bounds come exactly
from the shape's wherever the placement keeps the axes' order.

Composed maps round differently from CE's step-by-step arithmetic, so placed
points are compared with it to within ``TOLERANCE``; a mobject too small to
be placed keeps CE's arithmetic exactly.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import (
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    UL,
    UP,
    UR,
    Circle,
    Dot,
    Group,
    Mobject,
    ParametricFunction,
    Square,
    ValueTracker,
    VectorizedPoint,
    VGroup,
    VMobject,
)
from arachne.mobject.placement import MIN_PLACED_POINTS, Placement
from arachne.utils.hashing import get_json
from arachne.utils.space_ops import rotation_matrix

#: Placed points against CE's arithmetic, in scene units.
TOLERANCE = 1e-12

DIRECTIONS = [ORIGIN, UP, DOWN, LEFT, RIGHT, UL, UR, DL, -UR, OUT, UP + OUT]


def curve(points: int = 4 * MIN_PLACED_POINTS) -> VMobject:
    rng = np.random.default_rng(points)
    return VMobject().set_points(rng.normal(size=(points, 3)))


def ce_shift(points, vector):
    return points + vector


def ce_scale(points, factor, about):
    return factor * (points - about) + about


def ce_rotate(points, angle, axis, about):
    return np.dot(points - about, rotation_matrix(angle, axis).T) + about


def ce_centre(points):
    """CE's get_center on a VMobject: the middle of its anchors' box."""
    anchors = np.concatenate((points[::4], points[3::4]))
    return (anchors.min(axis=0) + anchors.max(axis=0)) / 2


def test_a_large_mobject_is_placed_over_its_shape():
    mob = curve()
    shape = mob.points
    mob.shift(RIGHT).scale(2).rotate(0.3, axis=UP + OUT)
    placement = mob._current_placement()
    assert isinstance(placement, Placement)
    assert placement.shape is shape


def test_placed_points_match_ces_arithmetic():
    mob = curve()
    reference = np.array(mob.points)
    mob.shift(2 * RIGHT + UP)
    reference = ce_shift(reference, 2 * RIGHT + UP)
    mob.scale(1.5)
    reference = ce_scale(reference, 1.5, ce_centre(reference))
    mob.rotate(0.7, axis=OUT, about_point=LEFT)
    reference = ce_rotate(reference, 0.7, OUT, LEFT)
    mob.scale(-0.5, about_point=UP)
    reference = ce_scale(reference, -0.5, UP)
    mob.rotate(1.1, axis=RIGHT + UP)
    reference = ce_rotate(reference, 1.1, RIGHT + UP, ce_centre(reference))
    mob.shift(DOWN)
    reference = ce_shift(reference, DOWN)
    np.testing.assert_allclose(mob.points, reference, rtol=0, atol=TOLERANCE)


def test_a_small_mobject_keeps_ces_arithmetic_exactly():
    square = Square()
    assert square.get_num_points() < MIN_PLACED_POINTS
    reference = np.array(square.points)
    square.shift(RIGHT).scale(2).rotate(0.3)
    reference = ce_shift(reference, RIGHT)
    reference = ce_scale(reference, 2, ce_centre(reference))
    reference = ce_rotate(reference, 0.3, OUT, ce_centre(reference))
    assert square._current_placement() is None
    np.testing.assert_array_equal(square.points, reference)


def test_moves_without_reads_compute_no_points():
    from arachne.mobject.values import count_work

    mob = curve()
    with count_work() as work:
        mob.shift(RIGHT)
        mob.shift(UP)
        mob.scale(3)  # about the centre: read from the shape's bounds
        mob.scale(-2, about_edge=UL)
        mob.move_to(LEFT)
        mob.next_to(ORIGIN, RIGHT)
    assert work.arrays == 0
    assert mob.__dict__.get("_points") is None


def test_points_are_computed_once_per_placement_and_shared_by_copies():
    from arachne.mobject.values import count_work

    mob = curve().shift(RIGHT).rotate(0.4)
    twin = mob.copy()
    with count_work() as work:
        first = mob.points
        again = mob.points
        theirs = twin.points
    assert work.arrays == 1
    assert first is again is theirs
    assert twin._current_placement() is mob._current_placement()


def test_assigning_points_replaces_the_placement():
    mob = curve().shift(RIGHT)
    new = np.zeros((8, 3))
    mob.points = new
    assert mob._current_placement() is None
    np.testing.assert_array_equal(mob.points, new)
    # A move after the assignment starts from the assigned points.
    mob.set_points(curve().points).shift(UP)
    assert mob._current_placement().shape is not None


def test_an_internal_rewrite_of_placed_points_wins_over_the_placement():
    mob = curve().shift(RIGHT)
    mob.append_points(np.ones((4, 3)))
    assert mob._current_placement() is None
    assert mob.get_num_points() == 4 * MIN_PLACED_POINTS + 4
    np.testing.assert_array_equal(mob.points[-1], np.ones(3))


@pytest.mark.parametrize(
    "move",
    [
        pytest.param(lambda m: m.shift(RIGHT, UP), id="shift"),
        pytest.param(lambda m: m.shift(np.array([0.5, -2.0, 0.25])), id="shift array"),
        pytest.param(lambda m: m.scale(2.5), id="scale"),
        pytest.param(lambda m: m.scale(-0.75, about_edge=UR), id="negative scale"),
        pytest.param(lambda m: m.scale(np.array([2.0, -1.0, 0.5])), id="axis scale"),
        pytest.param(lambda m: m.scale(0), id="scale to a point"),
        pytest.param(lambda m: m.shift(UP).scale(3).shift(LEFT), id="chain"),
    ],
)
@pytest.mark.parametrize("make", [curve, lambda: VGroup(curve(), curve(1500), Dot())])
def test_bounds_of_axis_preserving_placements_are_exact(move, make):
    mob = make()
    move(mob)
    # Read with the kept bounds, then compute the points and read them as CE did.
    fast = {tuple(d): mob.get_critical_point(d) for d in DIRECTIONS}
    sizes = (mob.width, mob.height, mob.depth, mob.get_x(), mob.get_y(UP))
    boundary = np.asarray(mob.get_points_defining_boundary())
    for direction, point in fast.items():
        expected = [
            boundary[:, dim].min()
            if direction[dim] < 0
            else boundary[:, dim].max()
            if direction[dim] > 0
            else (boundary[:, dim].min() + boundary[:, dim].max()) / 2
            for dim in range(3)
        ]
        np.testing.assert_array_equal(point, expected)
    every = np.concatenate([m.points for m in mob.get_family() if m.has_points()])
    assert sizes[0] == max(every[:, 0]) - min(every[:, 0])
    assert sizes[1] == max(every[:, 1]) - min(every[:, 1])
    assert sizes[2] == max(every[:, 2]) - min(every[:, 2])
    assert sizes[3] == (boundary[:, 0].min() + boundary[:, 0].max()) / 2
    assert sizes[4] == boundary[:, 1].max()


FAMILIES = {
    "square": Square,
    "circle": Circle,
    "nested groups": lambda: VGroup(
        VGroup(Square(), Circle().shift(UP)), Dot(LEFT * 3)
    ),
    "a single point": lambda: VectorizedPoint(UR),
    "with an empty member": lambda: VGroup(Square(), VMobject()),
    "a group of mobjects": lambda: Group(Square(), Mobject().set_points([[1, 2, 3]])),
    "a tracker in a group": lambda: Group(Square(), ValueTracker(4)),
}


@pytest.mark.parametrize("make", FAMILIES.values(), ids=FAMILIES.keys())
def test_kept_bounds_read_as_ces_boundary_does(make):
    mob = make()
    boundary = np.asarray(mob.get_points_defining_boundary())
    for direction in DIRECTIONS:
        expected = np.zeros(3)
        for dim in range(3):
            column = boundary[:, dim]
            if direction[dim] < 0:
                expected[dim] = np.min(column)
            elif direction[dim] == 0:
                expected[dim] = (np.min(column) + np.max(column)) / 2
            else:
                expected[dim] = np.max(column)
        np.testing.assert_array_equal(mob.get_critical_point(direction), expected)


def test_an_empty_family_reads_as_in_ce():
    empty = VGroup(VMobject())
    np.testing.assert_array_equal(empty.get_center(), np.zeros(3))
    assert empty.width == 0
    with pytest.raises(IndexError):
        empty.get_x()


def test_points_that_dont_split_into_curves_still_raise():
    mob = VMobject()
    mob.points = np.zeros((5, 3))
    with pytest.raises(ValueError):
        mob.get_center()


def test_non_finite_moves_keep_ces_arithmetic():
    mob = curve()
    with np.errstate(invalid="ignore"):
        mob.shift(np.array([np.inf, 0, 0]))
    assert mob._current_placement() is None
    assert np.isinf(mob.points[:, 0]).all()


def test_a_per_point_shift_keeps_ces_arithmetic():
    mob = curve()
    offsets = np.arange(mob.get_num_points() * 3, dtype=float).reshape(-1, 3)
    before = np.array(mob.points)
    mob.shift(offsets)
    assert mob._current_placement() is None
    np.testing.assert_array_equal(mob.points, before + offsets)


def test_a_tracker_is_never_placed():
    tracker = ValueTracker(2)
    group = Group(curve(), tracker)
    group.shift(RIGHT)
    assert tracker._current_placement() is None
    assert tracker.get_value() == 3


def test_placed_mobjects_hash_by_their_points():
    left, right = curve().shift(LEFT), curve().shift(RIGHT)
    assert get_json(left) != get_json(right)
    assert get_json(left) == get_json(copy.deepcopy(left))


def test_a_parametric_plot_is_placed_and_drawn_from_its_points():
    plot = ParametricFunction(
        lambda t: np.array([t, np.sin(t), 0]), t_range=[-3, 3, 0.01]
    )
    assert plot.get_num_points() >= MIN_PLACED_POINTS
    reference = np.array(plot.points) + 2 * UP
    plot.shift(2 * UP)
    assert plot._current_placement() is not None
    np.testing.assert_allclose(plot.points, reference, rtol=0, atol=TOLERANCE)


class ShiftsAfterMapping(VMobject):
    """A subclass whose point map also shifts up, as a user's override might."""

    def apply_points_function_about_point(
        self, func, about_point=None, about_edge=None
    ):
        super().apply_points_function_about_point(func, about_point, about_edge)
        return self.shift(UP)


@pytest.mark.parametrize(
    "points", [8, 4 * MIN_PLACED_POINTS], ids=["small", "placed size"]
)
@pytest.mark.parametrize(
    "move",
    [
        pytest.param(lambda m: m.scale(2, about_point=ORIGIN), id="scale"),
        pytest.param(lambda m: m.rotate(0.5, about_point=ORIGIN), id="rotate"),
    ],
)
def test_scale_and_rotate_go_through_an_overridden_point_map(move, points):
    # CE's scale and rotate called apply_points_function_about_point.
    mob = ShiftsAfterMapping().set_points(np.zeros((points, 3)))
    move(mob)
    np.testing.assert_array_equal(mob.get_center(), [0, 1, 0])


def test_shift_never_went_through_the_point_map():
    # CE's shift added the vector to each member's points directly.
    mob = ShiftsAfterMapping().set_points(np.zeros((8, 3)))
    mob.shift(RIGHT)
    np.testing.assert_array_equal(mob.get_center(), [1, 0, 0])


def test_the_hash_follows_the_geometry_not_its_storage():
    base = curve()
    placed, stored = base.copy(), base.copy()
    placed.shift(RIGHT)
    stored.points = stored.points + RIGHT
    assert placed._current_placement() is not None
    before = get_json(placed)
    _ = placed.points  # compute the placed points
    assert get_json(placed) == before
    assert get_json(stored) == before
