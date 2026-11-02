"""The rate and path functions the trusted algebra admits, and their array
twins.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import ORIGIN, OUT, RIGHT, UP
from arachne.timeline.rates import (
    MONOTONE_RATE_FUNCTIONS,
    RATE_FUNCTIONS,
    admitted_path,
    admitted_rate,
    array_twin,
    frozen_path,
    is_monotone,
)
from arachne.utils import paths
from arachne.utils import rate_functions as rf
from arachne.utils.bezier import interpolate

#: A dense grid over the unit interval and a little beyond it, where the
#: rate functions clip.
GRID = np.concatenate([np.linspace(-0.25, 1.25, 20001), np.linspace(0, 1, 60 * 7 + 1)])


def _names(functions) -> list[str]:
    return sorted(f.__name__ for f in functions)


def _by_name(name: str):
    return next(f for f in RATE_FUNCTIONS if f.__name__ == name)


@pytest.mark.parametrize("name", _names(RATE_FUNCTIONS))
def test_an_array_twin_equals_its_scalar_function_at_every_instant(name):
    function = _by_name(name)
    twin = array_twin(function)
    expected = np.array([function(float(t)) for t in GRID])
    assert np.array_equal(twin(GRID), expected)


def test_every_rate_function_ce_ships_is_admitted():
    shipped = {
        value
        for key, value in vars(rf).items()
        if callable(value)
        and not isinstance(value, type)
        and not key.startswith("_")
        and value.__module__ == rf.__name__
    }
    factories = {rf.squish_rate_func, rf.not_quite_there, rf.unit_interval, rf.zero}
    assert shipped - factories <= RATE_FUNCTIONS


@pytest.mark.parametrize("name", _names(RATE_FUNCTIONS))
def test_the_monotone_set_holds_exactly_the_functions_that_never_decrease(name):
    function = _by_name(name)
    values = np.array([function(float(t)) for t in np.linspace(0, 1, 20001)])
    never_decreases = bool(np.all(np.diff(values) >= 0))
    assert (function in MONOTONE_RATE_FUNCTIONS) == never_decreases


def test_a_factory_closure_is_admitted_over_an_admitted_function():
    squished = rf.squish_rate_func(rf.smooth, 0.2, 0.7)
    assert admitted_rate(squished)
    assert is_monotone(squished)
    twin = array_twin(squished)
    assert np.array_equal(twin(GRID), [squished(float(t)) for t in GRID])
    almost = rf.not_quite_there(rf.there_and_back, 0.8)
    assert admitted_rate(almost)
    assert not is_monotone(almost)
    assert np.array_equal(array_twin(almost)(GRID), [almost(float(t)) for t in GRID])


def test_a_user_rate_function_is_not_admitted_whatever_it_computes():
    def my_linear(t: float) -> float:
        return t

    assert not admitted_rate(my_linear)
    assert not admitted_rate(lambda t: rf.smooth(t))
    assert not admitted_rate(rf.squish_rate_func(my_linear, 0, 1))
    assert not is_monotone(my_linear)


def test_ce_paths_are_admitted_and_others_are_not():
    assert admitted_path(interpolate)
    assert admitted_path(paths.path_along_arc(np.pi / 2))
    assert admitted_path(paths.clockwise_path())
    assert admitted_path(paths.spiral_path(np.pi))
    assert not admitted_path(lambda a, b, t: interpolate(a, b, t))
    assert not admitted_path(paths.path_along_arc(np.array(1.0)))


def test_a_frozen_path_holds_detached_copies_of_every_array_it_binds():
    """A read-only view still changes with its base, so it is copied too."""
    base = np.array([ORIGIN, OUT])
    centres = base.view()
    centres.flags.writeable = False
    circles = paths.path_along_circles(1.0, centres)
    assert admitted_path(circles)
    frozen = frozen_path(circles)
    start, end = np.array([RIGHT]), np.array([UP])
    expected = circles(start, end, 0.3)
    base[0] = 5
    assert not np.array_equal(circles(start, end, 0.3), expected)
    assert np.array_equal(frozen(start, end, 0.3), expected)
    assert admitted_path(frozen)
    assert frozen_path(interpolate) is interpolate
