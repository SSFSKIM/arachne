"""A ValueTracker's points are a read-only value, as every mobject's are, and its value
is read from them as in CE: the x coordinate of its one point (x and y for a
ComplexValueTracker). Geometry keeps every coordinate of that point, so a tracker in a
group moves, measures and lays out as CE's did.

The expected numbers are CE's: a Dot's box is +-0.08 about its centre, and a tracker
counts as its one point.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import (
    DOWN,
    OUT,
    PI,
    RIGHT,
    UP,
    ComplexValueTracker,
    Dot,
    Group,
    ValueTracker,
)
from arachne.utils.paths import path_along_arc


def test_the_value_reads_from_the_point():
    tracker = ValueTracker(2.5)
    assert tracker.get_value() == 2.5
    assert type(tracker.get_value()) is np.float64
    np.testing.assert_array_equal(tracker.points, [[2.5, 0, 0]])
    tracker.set_value(-1)
    np.testing.assert_array_equal(tracker.points, [[-1, 0, 0]])


def test_writing_into_the_points_raises_naming_set_value():
    tracker = ValueTracker(1)
    with pytest.raises(ValueError, match="set_value"):
        tracker.points[0, 0] = 3
    with pytest.raises(ValueError, match="set_value"):
        tracker.points += 1
    with pytest.raises(ValueError, match="set_value"):
        tracker.points.dtype = np.int64
    assert tracker.get_value() == 1


def test_assigning_points_keeps_them_whole():
    tracker = ValueTracker()
    tracker.points = [[4.0, 2.0, -1.0]]
    assert tracker.get_value() == 4.0
    np.testing.assert_array_equal(tracker.points, [[4, 2, -1]])
    with pytest.raises(ValueError, match="set_value"):
        tracker.points = np.zeros((0, 3))


def test_set_value_changes_only_the_value_coordinate():
    tracker = ValueTracker(1).shift(UP + 2 * OUT)
    tracker.set_value(5)
    np.testing.assert_array_equal(tracker.points, [[5, 1, 2]])
    tracker.increment_value(1)
    np.testing.assert_array_equal(tracker.points, [[6, 1, 2]])


def test_a_group_shift_moves_the_box_and_keeps_the_tracker_whole():
    tracker = ValueTracker(2)
    group = Group(Dot(), tracker).shift(UP)
    np.testing.assert_allclose(group.get_center(), [0.96, 1, 0])
    np.testing.assert_allclose(tracker.points, [[2, 1, 0]])
    assert tracker.get_value() == 2
    np.testing.assert_allclose(group.get_corner(UP + RIGHT), [2, 1.08, 0])


def test_layout_places_the_tracker_point_as_in_ce():
    tracker = ValueTracker(5).next_to(Dot(), UP, buff=1)
    np.testing.assert_allclose(tracker.points, [[0, 1.08, 0]])
    assert tracker.get_value() == 0

    dot, tracker = Dot(), ValueTracker(2)
    Group(dot, tracker).arrange(DOWN, buff=0.5)
    np.testing.assert_allclose(dot.get_center(), [0, 0.25, 0])
    np.testing.assert_allclose(tracker.points, [[0, -0.33, 0]], atol=1e-12)
    assert tracker.get_value() == 0


def test_a_group_scale_scales_the_tracker_about_the_box_centre():
    dot, tracker = Dot().shift(UP), ValueTracker(2)
    group = Group(dot, tracker)
    np.testing.assert_allclose(group.get_center(), [0.96, 0.54, 0])
    group.scale(2)
    np.testing.assert_allclose(tracker.points, [[3.04, -0.54, 0]])
    assert tracker.get_value() == pytest.approx(3.04)
    np.testing.assert_allclose(group.get_center(), [0.96, 0.54, 0])
    np.testing.assert_allclose(group.get_corner(DOWN + RIGHT), [3.04, -0.54, 0])


def test_a_rotation_keeps_the_point_so_a_second_one_completes_it():
    tracker = ValueTracker(1).rotate(PI / 2, about_point=np.zeros(3))
    np.testing.assert_allclose(tracker.points, [[0, 1, 0]], atol=1e-12)
    tracker.rotate(PI / 2, about_point=np.zeros(3))
    np.testing.assert_allclose(tracker.points, [[-1, 0, 0]], atol=1e-12)
    assert tracker.get_value() == pytest.approx(-1)


def test_interpolation_and_animate_go_by_points():
    start, end = ValueTracker(1.0), ValueTracker(2.5)
    assert ValueTracker().interpolate(start, end, 0.7).get_value() == 2.05
    tracker = ValueTracker(0)
    tracker.target = ValueTracker(10)
    tracker.interpolate(tracker.copy(), tracker.target, 0.5)
    assert tracker.get_value() == 5


def test_a_copy_has_its_own_value():
    tracker = ValueTracker(1)
    clone = copy.deepcopy(tracker)
    clone.set_value(5)
    assert tracker.get_value() == 1
    assert tracker.copy().get_value() == 1


def test_the_value_casts_as_ce_did():
    tracker = ValueTracker(3)
    assert type(tracker.get_value()) is np.float64
    tracker.set_value(np.float32(0.5))
    assert tracker.get_value() == 0.5
    assert type(tracker.get_value()) is np.float64


def test_points_assigned_as_integers_still_hold_a_fractional_value():
    tracker = ValueTracker()
    tracker.points = np.array([[1, 0, 0]])
    tracker.set_value(2.5)
    assert tracker.get_value() == 2.5


def test_complex_tracker_keeps_both_parts_and_the_third_coordinate():
    tracker = ComplexValueTracker(2 - 3j)
    assert tracker.get_value() == 2 - 3j
    np.testing.assert_array_equal(tracker.points, [[2, -3, 0]])
    tracker.shift(UP + OUT)
    assert tracker.get_value() == 2 - 2j
    np.testing.assert_array_equal(tracker.points, [[2, -2, 1]])
    tracker.set_value(1j)
    np.testing.assert_array_equal(tracker.points, [[0, 1, 1]])
    with pytest.raises(ValueError, match="set_value"):
        tracker.points[0, :2] = (0, 0)


def test_complex_tracker_on_an_arc_path():
    start, end = ComplexValueTracker(1), ComplexValueTracker(1j)
    mid = ComplexValueTracker().interpolate(
        start, end, 0.5, path_func=path_along_arc(np.pi / 2)
    )
    expected = path_along_arc(np.pi / 2)(start.points, end.points, 0.5)[0]
    assert mid.get_value() == complex(expected[0], expected[1])
