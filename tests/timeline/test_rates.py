"""The rate and path functions the trusted algebra admits, and their array
twins.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import ORIGIN, OUT, RIGHT, UP
from arachne.timeline.rates import (
    MONOTONE_RATE_FUNCTIONS,
    RATE_FUNCTIONS,
    admitted_path,
    admitted_rate,
    array_twin,
    engine_rate,
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


class _EqualTo:
    """A callable whose hash and equality are ``target``'s, counting its
    own calls: it runs its own code whatever it compares equal to.
    """

    def __init__(self, target):
        self.target = target
        self.calls = 0

    def __hash__(self):
        return hash(self.target)

    def __eq__(self, other):
        return other is self.target or other is self

    def __call__(self, t):
        self.calls += 1
        return self.target(t)


class _Carrying:
    """A callable that carries a factory closure's code and cells."""

    def __init__(self, closure):
        self.__code__ = closure.__code__
        self.__closure__ = closure.__closure__

    def __call__(self, *args):
        return args[-1]


class _Drifting(float):
    """A float whose arithmetic and comparisons are its own code."""

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)


@pytest.mark.parametrize("name", _names(RATE_FUNCTIONS))
def test_a_callable_only_equal_to_an_engine_rate_is_not_admitted(name):
    lookalike = _EqualTo(_by_name(name))
    assert not admitted_rate(lookalike)
    assert not is_monotone(lookalike)
    with pytest.raises(KeyError):
        array_twin(lookalike)
    assert lookalike.calls == 0


def test_a_factory_closure_over_a_lookalike_is_not_admitted():
    for closure in (
        rf.squish_rate_func(_EqualTo(rf.smooth), 0.1, 0.9),
        rf.not_quite_there(_EqualTo(rf.linear), 0.8),
        rf.squish_rate_func(rf.not_quite_there(_EqualTo(rf.smooth), 0.9), 0, 1),
    ):
        assert not admitted_rate(closure)
        assert not is_monotone(closure)
        with pytest.raises(KeyError):
            array_twin(closure)


def test_a_callable_carrying_a_factorys_code_is_not_admitted():
    for closure in (
        rf.squish_rate_func(rf.smooth, 0.1, 0.9),
        rf.not_quite_there(rf.smooth, 0.8),
    ):
        lookalike = _Carrying(closure)
        assert not admitted_rate(lookalike)
        assert not is_monotone(lookalike)
        with pytest.raises(KeyError):
            array_twin(lookalike)


def test_a_factory_closure_binding_a_number_of_its_own_type_is_not_admitted():
    for closure in (
        rf.squish_rate_func(rf.smooth, _Drifting(0.1), 0.9),
        rf.squish_rate_func(rf.smooth, 0.1, _Drifting(0.9)),
        rf.not_quite_there(rf.smooth, _Drifting(0.8)),
        rf.squish_rate_func(rf.smooth, np.timedelta64(0), np.timedelta64(1)),
    ):
        assert not admitted_rate(closure)
        assert not is_monotone(closure)


def test_a_factory_closure_whose_cell_holds_itself_is_not_admitted():
    closure = rf.squish_rate_func(rf.smooth, 0.1, 0.9)
    cells = dict(zip(closure.__code__.co_freevars, closure.__closure__, strict=True))
    cells["func"].cell_contents = closure
    assert not admitted_rate(closure)
    assert not is_monotone(closure)


@pytest.mark.parametrize(
    "closure",
    [
        rf.squish_rate_func(rf.smooth, 0.2, 0.7),
        rf.squish_rate_func(rf.linear, 0, 1),
        rf.squish_rate_func(rf.there_and_back, np.float64(0.25), np.float32(0.75)),
        rf.squish_rate_func(rf.smooth, Fraction(1, 4), np.int64(1)),
        rf.not_quite_there(rf.ease_in_sine, np.float64(0.9)),
        rf.squish_rate_func(rf.not_quite_there(rf.smooth, 0.9), 0.1, 0.8),
        rf.not_quite_there(rf.squish_rate_func(rf.rush_into, 0.3, 0.6), 1),
    ],
)
def test_engine_factory_closures_stay_admitted_at_any_depth(closure):
    assert admitted_rate(closure)
    twin = array_twin(closure)
    assert np.array_equal(twin(GRID), [closure(float(t)) for t in GRID])


def test_every_engine_rate_stays_admitted_by_itself():
    for function in RATE_FUNCTIONS:
        assert admitted_rate(function), function.__name__
        assert is_monotone(function) == (function in MONOTONE_RATE_FUNCTIONS)
    assert is_monotone(
        rf.squish_rate_func(rf.not_quite_there(rf.smooth, 0.9), 0.1, 0.8)
    )
    assert not is_monotone(rf.squish_rate_func(rf.smooth, 0.8, 0.1))


def test_an_engine_rate_is_found_with_its_factory_layers():
    inner = rf.not_quite_there(rf.smooth, 0.9)
    outer = rf.squish_rate_func(inner, 0.1, 0.8)
    found = engine_rate(outer)
    assert found is not None
    assert found.base is rf.smooth
    assert [kind for kind, _ in found.layers] == ["squish", "not_quite_there"]
    assert found.layers[0][1]["func"] is inner
    assert engine_rate(rf.linear) == (rf.linear, ())
    assert engine_rate(_EqualTo(rf.linear)) is None


def test_a_path_lookalike_is_not_admitted():
    for closure in (
        paths.path_along_arc(np.pi / 2),
        paths.spiral_path(np.pi),
        paths.path_along_circles(1.0, np.zeros(3)),
    ):
        assert admitted_path(closure)
        assert not admitted_path(_Carrying(closure))
    assert not admitted_path(_EqualTo(interpolate))
    assert not admitted_path(paths.path_along_arc(_Drifting(1.0)))
    assert not admitted_path(paths.path_along_circles(1.0, np.zeros(3).view(_Points)))
    assert admitted_path(paths.path_along_arc(np.float64(1.0)))


class _Points(np.ndarray):
    """An array type of a scene's own."""


class _Scaling(float):
    """A float whose product is its own code."""

    def __mul__(self, other):
        return float(self) * other

    __rmul__ = __mul__


@pytest.mark.parametrize(
    "factory",
    [
        lambda axis: paths.spiral_path(1.0, axis=axis),
        lambda axis: paths.path_along_arc(1.0, axis=axis),
        lambda axis: paths.path_along_circles(1.0, np.zeros(3), axis=axis),
    ],
)
def test_a_path_whose_axis_is_not_plain_numbers_is_not_admitted(factory):
    assert admitted_path(factory(np.array([0.0, 0.0, 1.0])))
    assert admitted_path(factory([0, 0, 2]))
    assert not admitted_path(factory(np.array([0.0, 0.0, _Scaling(1.0)], dtype=object)))


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
