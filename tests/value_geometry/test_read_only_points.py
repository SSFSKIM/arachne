"""A mobject's points are a read-only value: writes into it raise, assignment builds
a new value, and a holder of the old value never sees it change.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pytest

from arachne import Circle, Dot, Mobject, Square, VMobject
from arachne.mobject.values import FrozenArray, FrozenPoints, freeze

POINTS_FIX = "set_points"


def _write_through_get_points(mob: Mobject) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        points = mob.get_points()
    points[0] = (1, 0, 0)


def square_points() -> np.ndarray:
    return np.array(Square().points)


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda m: m.points.__setitem__(0, (1, 0, 0)), id="item"),
        pytest.param(lambda m: m.points.__setitem__((slice(None), 0), 0), id="column"),
        pytest.param(lambda m: m.points[:, 1].__setitem__(0, 5), id="through-a-view"),
        pytest.param(lambda m: m.points.__iadd__(1), id="iadd"),
        pytest.param(lambda m: m.points.__imul__(2), id="imul"),
        pytest.param(lambda m: np.copyto(m.points, 0), id="copyto"),
        pytest.param(lambda m: np.copyto(dst=m.points, src=0), id="copyto-keyword"),
        pytest.param(lambda m: np.put(m.points, [0], 1), id="put"),
        pytest.param(lambda m: m.points.fill(0), id="fill"),
        pytest.param(lambda m: m.points.sort(axis=0), id="sort"),
        pytest.param(lambda m: m.points.setflags(write=True), id="setflags"),
        pytest.param(lambda m: _write_through_get_points(m), id="via-get_points"),
    ],
)
def test_an_in_place_write_raises_naming_the_setter(write):
    square = Square()
    before = np.array(square.points)
    with pytest.raises(ValueError, match=POINTS_FIX):
        write(square)
    np.testing.assert_array_equal(square.points, before)


def test_augmented_assignment_on_the_property_raises():
    square = Square()
    with pytest.raises(ValueError, match=POINTS_FIX):
        square.points += 1
    with pytest.raises(ValueError, match="apply_points_function_about_point"):
        square.points *= 2


def test_reading_and_computing_work_and_return_plain_arrays():
    square = Square()
    points = square.points
    assert isinstance(points, FrozenPoints)
    assert not points.flags.writeable
    shifted = points + 1
    assert type(shifted) is np.ndarray
    assert shifted.flags.writeable
    assert type(np.max(points, axis=0)) is np.ndarray
    assert type(points.sum()) is np.float64
    copied = points.copy()
    assert type(copied) is np.ndarray
    copied[0] = 7
    assert copied[0, 0] == 7
    # A copy of a value under a computation's own buffers keeps working.
    buffer = np.zeros_like(points)
    np.copyto(buffer, points)
    np.testing.assert_array_equal(buffer, points)


def test_assignment_copies_the_callers_array():
    mob = VMobject()
    mine = square_points()
    mob.points = mine
    mine[0] = (9, 9, 9)  # still the caller's own array, still writable
    np.testing.assert_array_equal(mob.points, square_points())


def test_set_points_copies_the_callers_array():
    mob = VMobject()
    mine = square_points()
    mob.set_points(mine)
    mine[0] = (9, 9, 9)
    np.testing.assert_array_equal(mob.points, square_points())


def test_every_mobject_has_set_points():
    mob = Mobject()
    mob.set_points([[1, 2, 3]])
    np.testing.assert_array_equal(mob.points, [[1, 2, 3]])
    assert not mob.points.flags.writeable


def test_assigning_an_existing_value_shares_it():
    a, b = Square(), VMobject()
    b.points = a.points
    assert b.points is a.points


def test_an_operation_leaves_the_old_value_unchanged():
    square = Square()
    old = square.points
    snapshot = np.array(old)
    square.shift(2 * np.array([1.0, 0, 0])).scale(3).rotate(1).stretch(2, 1)
    np.testing.assert_array_equal(old, snapshot)
    assert square.points is not old


def test_deepcopy_gives_an_equal_read_only_value():
    circle = Circle()
    clone = copy.deepcopy(circle)
    assert clone.points is not circle.points
    np.testing.assert_array_equal(clone.points, circle.points)
    assert isinstance(clone.points, FrozenPoints)
    with pytest.raises(ValueError, match=POINTS_FIX):
        clone.points[0] = 0


def test_pickle_keeps_the_value_read_only():
    value = freeze(square_points(), FrozenPoints)
    restored = pickle.loads(pickle.dumps(value))
    assert isinstance(restored, FrozenPoints)
    assert not restored.flags.writeable
    np.testing.assert_array_equal(restored, value)


def test_freeze_shares_read_only_values_and_copies_the_rest():
    value = freeze(square_points(), FrozenPoints)
    assert freeze(value, FrozenPoints) is value
    assert isinstance(freeze(value, FrozenArray), FrozenArray)
    writable_view = np.zeros((2, 3)).view(FrozenPoints)
    assert freeze(writable_view, FrozenPoints) is not writable_view


def test_a_dot_still_moves_and_measures():
    dot = Dot().move_to([1, 2, 0])
    np.testing.assert_allclose(dot.get_center(), [1, 2, 0])
    dot.set_x(-1)
    np.testing.assert_allclose(dot.get_center(), [-1, 2, 0])


def test_a_value_prints_as_the_array_it_holds():
    square = Square()
    assert repr(square.points) == repr(np.array(square.points))
    assert repr(square.points[0]).startswith("array(")
    assert str(square.points) == str(np.array(square.points))
