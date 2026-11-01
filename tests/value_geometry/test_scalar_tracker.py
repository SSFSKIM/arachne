"""A ValueTracker holds its value as a number, with a read-only points view kept for
compatibility: the view moves with the value, and geometry on the view moves the value
as it did in CE.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import RIGHT, UP, ComplexValueTracker, Group, ValueTracker
from arachne.utils.paths import path_along_arc


def test_the_value_is_a_number_and_the_points_follow_it():
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
    assert tracker.get_value() == 1


def test_assigning_points_sets_the_value():
    tracker = ValueTracker()
    tracker.points = [[4.0, 0, 0]]
    assert tracker.get_value() == 4.0


def test_geometry_on_a_tracker_moves_its_value_as_in_ce():
    tracker = ValueTracker(1)
    tracker.shift(2 * RIGHT + UP)
    assert tracker.get_value() == 3
    Group(tracker).scale(2, about_point=np.zeros(3))
    assert tracker.get_value() == 6


def test_interpolation_and_animate_go_by_value():
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


def test_complex_tracker_keeps_both_parts():
    tracker = ComplexValueTracker(2 - 3j)
    assert tracker.get_value() == 2 - 3j
    np.testing.assert_array_equal(tracker.points, [[2, -3, 0]])
    tracker.shift(UP)
    assert tracker.get_value() == 2 - 2j
    with pytest.raises(ValueError, match="set_value"):
        tracker.points[0, :2] = (0, 0)


def test_complex_tracker_on_an_arc_path():
    start, end = ComplexValueTracker(1), ComplexValueTracker(1j)
    mid = ComplexValueTracker().interpolate(
        start, end, 0.5, path_func=path_along_arc(np.pi / 2)
    )
    expected = path_along_arc(np.pi / 2)(start.points, end.points, 0.5)[0]
    assert mid.get_value() == complex(expected[0], expected[1])
