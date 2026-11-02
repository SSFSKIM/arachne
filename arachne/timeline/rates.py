"""The rate and path functions the trusted algebra admits, and the array
twins of the rate functions.

A rate or path function is admitted by identity, never by what its code looks
like: CE's own rate functions (the module attributes of
:mod:`arachne.utils.rate_functions`), the closures CE's factories return
(``squish_rate_func`` and ``not_quite_there`` over an admitted function, with
plain numbers bound), and CE's straight, arc and circle paths. Anything else,
a user's lambda included, is unproven, whatever it computes.

Every admitted rate function has an *array twin* that evaluates it over an
array of instants at once, equal to calling the scalar original on each.
"""

from __future__ import annotations

__all__ = [
    "MONOTONE_RATE_FUNCTIONS",
    "RATE_FUNCTIONS",
    "admitted_path",
    "admitted_rate",
    "array_twin",
    "is_monotone",
    "path_arrays",
]

from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np

from arachne.utils import paths
from arachne.utils import rate_functions as rf
from arachne.utils.bezier import interpolate

ArrayFunction = Callable[[np.ndarray], np.ndarray]


def _unit_interval(inner: ArrayFunction) -> ArrayFunction:
    """The twin of a function under ``@unit_interval``: 0 below the interval,
    1 above it (and for NaN, as the scalar's ``else`` gives), ``inner``
    inside.
    """

    def twin(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        inside = (t >= 0) & (t <= 1)
        result = np.where(t < 0, 0.0, 1.0)
        result[inside] = inner(t[inside])
        return result

    return twin


def _zero(inner: ArrayFunction) -> ArrayFunction:
    """The twin of a function under ``@zero``: 0 outside the interval."""

    def twin(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        inside = (t >= 0) & (t <= 1)
        result = np.zeros_like(t)
        result[inside] = inner(t[inside])
        return result

    return twin


def _branch(
    t: np.ndarray, low: np.ndarray, below: ArrayFunction, above: ArrayFunction
) -> np.ndarray:
    """``below(t)`` where ``low`` holds, ``above(t)`` elsewhere, each evaluated
    only where it is used.
    """
    result = np.empty_like(t)
    result[low] = below(t[low])
    result[~low] = above(t[~low])
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


def _smooth(t: np.ndarray, inflection: float = 10.0) -> np.ndarray:
    error = 1.0 / (1 + np.exp(inflection / 2))
    value = (_sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error)
    return np.minimum(np.maximum(value, 0), 1)


def _smooth_unit(t: np.ndarray) -> np.ndarray:
    """``smooth`` as a nested call sees it: through its ``@unit_interval``."""
    return _unit_interval(_smooth)(t)


def _ease_out_bounce(t: np.ndarray) -> np.ndarray:
    n1, d1 = 7.5625, 2.75
    result = np.empty_like(t)
    first = t < 1 / d1
    second = ~first & (t < 2 / d1)
    third = ~first & ~second & (t < 2.5 / d1)
    rest = ~(first | second | third)
    result[first] = n1 * t[first] * t[first]
    u = t[second] - 1.5 / d1
    result[second] = n1 * u * u + 0.75
    u = t[third] - 2.25 / d1
    result[third] = n1 * u * u + 0.9375
    u = t[rest] - 2.625 / d1
    result[rest] = n1 * u * u + 0.984375
    return result


def _ease_out_bounce_unit(t: np.ndarray) -> np.ndarray:
    return _unit_interval(_ease_out_bounce)(t)


def _running_start(t: np.ndarray, pull_factor: float = -0.5) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    mt4 = mt3 * mt
    return (
        15 * t2 * mt4 * pull_factor
        + 20 * t3 * mt3 * pull_factor
        + 15 * t4 * mt2
        + 6 * t5 * mt
        + t6
    )


def _ease_in_elastic(t: np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    result = np.empty_like(t)
    ends = (t == 0) | (t == 1)
    result[t == 0] = 0
    result[t == 1] = 1
    u = t[~ends]
    result[~ends] = -np.power(2.0, 10 * u - 10) * np.sin((u * 10 - 10.75) * c4)
    return result


def _ease_out_elastic(t: np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    result = np.empty_like(t)
    ends = (t == 0) | (t == 1)
    result[t == 0] = 0
    result[t == 1] = 1
    u = t[~ends]
    result[~ends] = np.power(2.0, -10 * u) * np.sin((u * 10 - 0.75) * c4) + 1
    return result


def _ease_in_out_elastic(t: np.ndarray) -> np.ndarray:
    c5 = (2 * np.pi) / 4.5
    result = np.empty_like(t)
    result[t == 0] = 0
    result[t == 1] = 1
    low = (t != 0) & (t != 1) & (t < 0.5)
    high = (t != 0) & (t != 1) & ~(t < 0.5)
    u = t[low]
    result[low] = -(np.power(2.0, 20 * u - 10) * np.sin((20 * u - 11.125) * c5)) / 2
    u = t[high]
    result[high] = (
        np.power(2.0, -20 * u + 10) * np.sin((20 * u - 11.125) * c5)
    ) / 2 + 1
    return result


def _ease_in_out_expo(t: np.ndarray) -> np.ndarray:
    result = np.empty_like(t)
    result[t == 0] = 0
    result[t == 1] = 1
    low = (t != 0) & (t != 1) & (t < 0.5)
    high = (t != 0) & (t != 1) & ~(t < 0.5)
    result[low] = np.power(2.0, 20 * t[low] - 10) / 2
    result[high] = (2 - np.power(2.0, -20 * t[high] + 10)) / 2
    return result


def _there_and_back_with_pause(
    t: np.ndarray, pause_ratio: float = 1.0 / 3
) -> np.ndarray:
    a = 2.0 / (1.0 - pause_ratio)
    result = np.empty_like(t)
    first = t < 0.5 - pause_ratio / 2
    middle = ~first & (t < 0.5 + pause_ratio / 2)
    last = ~(first | middle)
    result[first] = _smooth_unit(a * t[first])
    result[middle] = 1
    result[last] = _smooth_unit(a - a * t[last])
    return result


def _wiggle(t: np.ndarray, wiggles: float = 2) -> np.ndarray:
    return _TWINS[rf.there_and_back](t) * np.sin(wiggles * np.pi * t)


#: Each admitted rate function and its array twin.
_TWINS: dict[Callable[..., float], ArrayFunction] = {}


def _twin(function: Callable[..., float], twin: ArrayFunction) -> None:
    _TWINS[function] = twin


def _pow(base: np.ndarray, exponent: float) -> np.ndarray:
    """Python's ``pow(base, exponent)`` (and ``base ** exponent``) on floats,
    elementwise: an array of exponents keeps numpy off its fast paths
    (``x**2`` as ``x*x``), which round differently from the C ``pow`` Python
    calls.
    """
    return np.power(base, np.full_like(base, float(exponent)))


_twin(rf.linear, _unit_interval(lambda t: t))
_twin(rf.smooth, _unit_interval(_smooth))
_twin(rf.smoothstep, _unit_interval(lambda t: 3 * _pow(t, 2) - 2 * _pow(t, 3)))
_twin(
    rf.smootherstep,
    _unit_interval(lambda t: 6 * _pow(t, 5) - 15 * _pow(t, 4) + 10 * _pow(t, 3)),
)
_twin(
    rf.smoothererstep,
    _unit_interval(
        lambda t: 35 * _pow(t, 4) - 84 * _pow(t, 5) + 70 * _pow(t, 6) - 20 * _pow(t, 7)
    ),
)
_twin(rf.rush_into, _unit_interval(lambda t: 2 * _smooth_unit(t / 2.0)))
_twin(rf.rush_from, _unit_interval(lambda t: 2 * _smooth_unit(t / 2.0 + 0.5) - 1))
_twin(rf.slow_into, _unit_interval(lambda t: np.sqrt(1 - (1 - t) * (1 - t))))
_twin(
    rf.double_smooth,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: 0.5 * _smooth_unit(2 * u),
            lambda u: 0.5 * (1 + _smooth_unit(2 * u - 1)),
        )
    ),
)
_twin(
    rf.there_and_back,
    _zero(
        lambda t: _smooth_unit(
            _branch(t, t < 0.5, lambda u: 2 * u, lambda u: 2 * (1 - u))
        )
    ),
)
_twin(rf.there_and_back_with_pause, _zero(_there_and_back_with_pause))
_twin(rf.running_start, _unit_interval(_running_start))
_twin(rf.wiggle, _zero(_wiggle))
_twin(rf.lingering, _unit_interval(lambda t: _squish(lambda u: u, 0, 0.8)(t)))
_twin(rf.exponential_decay, _unit_interval(lambda t: 1 - np.exp(-t / 0.1)))
_twin(rf.ease_in_sine, _unit_interval(lambda t: 1 - np.cos((t * np.pi) / 2)))
_twin(rf.ease_out_sine, _unit_interval(lambda t: np.sin((t * np.pi) / 2)))
_twin(rf.ease_in_out_sine, _unit_interval(lambda t: -(np.cos(np.pi * t) - 1) / 2))
_twin(rf.ease_in_quad, _unit_interval(lambda t: t * t))
_twin(rf.ease_out_quad, _unit_interval(lambda t: 1 - (1 - t) * (1 - t)))
_twin(
    rf.ease_in_out_quad,
    _unit_interval(
        lambda t: _branch(
            t, t < 0.5, lambda u: 2 * u * u, lambda u: 1 - _pow(-2 * u + 2, 2) / 2
        )
    ),
)
_twin(rf.ease_in_cubic, _unit_interval(lambda t: t * t * t))
_twin(rf.ease_out_cubic, _unit_interval(lambda t: 1 - _pow(1 - t, 3)))
_twin(
    rf.ease_in_out_cubic,
    _unit_interval(
        lambda t: _branch(
            t, t < 0.5, lambda u: 4 * u * u * u, lambda u: 1 - _pow(-2 * u + 2, 3) / 2
        )
    ),
)
_twin(rf.ease_in_quart, _unit_interval(lambda t: t * t * t * t))
_twin(rf.ease_out_quart, _unit_interval(lambda t: 1 - _pow(1 - t, 4)))
_twin(
    rf.ease_in_out_quart,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: 8 * u * u * u * u,
            lambda u: 1 - _pow(-2 * u + 2, 4) / 2,
        )
    ),
)
_twin(rf.ease_in_quint, _unit_interval(lambda t: t * t * t * t * t))
_twin(rf.ease_out_quint, _unit_interval(lambda t: 1 - _pow(1 - t, 5)))
_twin(
    rf.ease_in_out_quint,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: 16 * u * u * u * u * u,
            lambda u: 1 - _pow(-2 * u + 2, 5) / 2,
        )
    ),
)
_twin(
    rf.ease_in_expo,
    _unit_interval(
        lambda t: _branch(t, t == 0, np.zeros_like, lambda u: _pow2(10 * u - 10))
    ),
)
_twin(
    rf.ease_out_expo,
    _unit_interval(
        lambda t: _branch(t, t == 1, np.ones_like, lambda u: 1 - _pow2(-10 * u))
    ),
)
_twin(rf.ease_in_out_expo, _unit_interval(_ease_in_out_expo))
_twin(rf.ease_in_circ, _unit_interval(lambda t: 1 - np.sqrt(1 - _pow(t, 2))))
_twin(rf.ease_out_circ, _unit_interval(lambda t: np.sqrt(1 - _pow(t - 1, 2))))
_twin(
    rf.ease_in_out_circ,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: (1 - np.sqrt(1 - _pow(2 * u, 2))) / 2,
            lambda u: (np.sqrt(1 - _pow(-2 * u + 2, 2)) + 1) / 2,
        )
    ),
)
_C1 = 1.70158
_twin(rf.ease_in_back, _unit_interval(lambda t: (_C1 + 1) * t * t * t - _C1 * t * t))
_twin(
    rf.ease_out_back,
    _unit_interval(lambda t: 1 + (_C1 + 1) * _pow(t - 1, 3) + _C1 * _pow(t - 1, 2)),
)
_C2 = _C1 * 1.525
_twin(
    rf.ease_in_out_back,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: (_pow(2 * u, 2) * ((_C2 + 1) * 2 * u - _C2)) / 2,
            lambda u: (_pow(2 * u - 2, 2) * ((_C2 + 1) * (u * 2 - 2) + _C2) + 2) / 2,
        )
    ),
)
_twin(rf.ease_in_elastic, _unit_interval(_ease_in_elastic))
_twin(rf.ease_out_elastic, _unit_interval(_ease_out_elastic))
_twin(rf.ease_in_out_elastic, _unit_interval(_ease_in_out_elastic))
_twin(rf.ease_in_bounce, _unit_interval(lambda t: 1 - _ease_out_bounce_unit(1 - t)))
_twin(rf.ease_out_bounce, _unit_interval(_ease_out_bounce))
_twin(
    rf.ease_in_out_bounce,
    _unit_interval(
        lambda t: _branch(
            t,
            t < 0.5,
            lambda u: (1 - _ease_out_bounce_unit(1 - 2 * u)) / 2,
            lambda u: (1 + _ease_out_bounce_unit(2 * u - 1)) / 2,
        )
    ),
)


def _pow2(exponent: np.ndarray) -> np.ndarray:
    """Python's ``pow(2, x)`` for float ``x``, elementwise."""
    return np.power(2.0, exponent)


#: CE's rate functions, admitted by identity: every module attribute that is
#: a rate function (the factories ``squish_rate_func`` and ``not_quite_there``
#: excepted; their closures are admitted through :func:`admitted_rate`).
RATE_FUNCTIONS: frozenset[Callable[..., float]] = frozenset(_TWINS)

#: CE's rate functions that never decrease, by identity. A group whose time
#: runs through one of these crosses each part boundary once, at an instant
#: bisection can find.
MONOTONE_RATE_FUNCTIONS: frozenset[Callable[[float], float]] = frozenset(
    {
        rf.linear,
        rf.smooth,
        rf.smoothstep,
        rf.smootherstep,
        rf.smoothererstep,
        rf.rush_into,
        rf.rush_from,
        rf.slow_into,
        rf.double_smooth,
        rf.lingering,
        rf.exponential_decay,
        rf.ease_in_sine,
        rf.ease_out_sine,
        rf.ease_in_out_sine,
        rf.ease_in_quad,
        rf.ease_out_quad,
        rf.ease_in_out_quad,
        rf.ease_in_cubic,
        rf.ease_out_cubic,
        rf.ease_in_out_cubic,
        rf.ease_in_quart,
        rf.ease_out_quart,
        rf.ease_in_out_quart,
        rf.ease_in_quint,
        rf.ease_out_quint,
        rf.ease_in_out_quint,
        rf.ease_in_expo,
        rf.ease_out_expo,
        rf.ease_in_out_expo,
        rf.ease_in_circ,
        rf.ease_out_circ,
        rf.ease_in_out_circ,
    }
)

_SQUISH_CODE = getattr(rf.squish_rate_func(rf.linear), "__code__")  # noqa: B009
_NOT_QUITE_CODE = getattr(rf.not_quite_there(rf.linear), "__code__")  # noqa: B009


def _closure(function: Any) -> dict[str, Any] | None:
    """The values a closure binds, by name, or ``None`` when it binds none."""
    cells = getattr(function, "__closure__", None)
    if cells is None:
        return None
    try:
        return dict(
            zip(
                function.__code__.co_freevars,
                (c.cell_contents for c in cells),
                strict=True,
            )
        )
    except ValueError:  # an empty cell
        return None


def _number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _factory_parts(function: Any) -> tuple[str, dict[str, Any]] | None:
    """``("squish", cells)`` or ``("not_quite_there", cells)`` for a closure
    one of CE's rate factories returned over an admitted function and plain
    numbers; ``None`` for anything else.
    """
    code = getattr(function, "__code__", None)
    if code is _SQUISH_CODE:
        cells = _closure(function)
        if (
            cells is not None
            and admitted_rate(cells.get("func"))
            and _number(cells.get("a"))
            and _number(cells.get("b"))
        ):
            return "squish", cells
    elif code is _NOT_QUITE_CODE:
        cells = _closure(function)
        if (
            cells is not None
            and admitted_rate(cells.get("func"))
            and _number(cells.get("proportion"))
        ):
            return "not_quite_there", cells
    return None


def admitted_rate(function: Any) -> bool:
    """Whether ``function`` is a rate function the trusted algebra admits."""
    try:
        if function in RATE_FUNCTIONS:
            return True
    except TypeError:  # unhashable
        return False
    return _factory_parts(function) is not None


def is_monotone(function: Any) -> bool:
    """Whether ``function`` is an admitted rate function that never decreases."""
    try:
        if function in MONOTONE_RATE_FUNCTIONS:
            return True
    except TypeError:
        return False
    parts = _factory_parts(function)
    if parts is None:
        return False
    kind, cells = parts
    if kind == "squish":
        return cells["a"] <= cells["b"] and is_monotone(cells["func"])
    return cells["proportion"] >= 0 and is_monotone(cells["func"])


def _squish(inner: ArrayFunction, a: float, b: float) -> ArrayFunction:
    def twin(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if a == b:
            return np.full_like(t, a)
        new_t = np.where(t < a, 0.0, np.where(t > b, 1.0, (t - a) / (b - a)))
        return inner(new_t)

    return twin


def array_twin(function: Callable[..., float]) -> ArrayFunction:
    """The array twin of an admitted rate function: it evaluates ``function``
    over an array of alphas, equal to calling it on each.
    """
    twin = _TWINS.get(function)
    if twin is not None:
        return twin
    parts = _factory_parts(function)
    if parts is None:
        raise KeyError(f"{function!r} is not an admitted rate function.")
    kind, cells = parts
    inner = array_twin(cells["func"])
    if kind == "squish":
        return _squish(inner, cells["a"], cells["b"])
    proportion = cells["proportion"]
    return lambda t: proportion * inner(np.asarray(t, dtype=float))


_ARC_CODE = paths.path_along_arc(1.0).__code__
_CIRCLES_CODE = paths.path_along_circles(1.0, np.zeros(3)).__code__
_SPIRAL_CODE = paths.spiral_path(1.0).__code__


def _points(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype.kind in "fiu"


def admitted_path(function: Any) -> bool:
    """Whether ``function`` is a path function the trusted algebra admits:
    the straight path (``interpolate``) or a closure of CE's arc, spiral or
    circle paths whose angle is a number. The arrays a closure binds may be
    the script's own (``Rotate`` binds its ``about_point``): a trusted play
    runs no script code, and :func:`path_arrays` lets an evaluation after
    the play see them as the play did.
    """
    if function is interpolate:
        return True
    code = getattr(function, "__code__", None)
    if code is _ARC_CODE:
        cells = _closure(function)
        return cells is not None and _number(cells.get("arc_angle"))
    if code is _SPIRAL_CODE:
        cells = _closure(function)
        return cells is not None and _number(cells.get("angle"))
    if code is _CIRCLES_CODE:
        cells = _closure(function)
        return (
            cells is not None
            and _number(cells.get("arc_angle"))
            and _points(cells.get("circles_centers"))
        )
    return False


def path_arrays(function: Any) -> tuple[np.ndarray, ...]:
    """The arrays an admitted path's closure binds that can still change."""
    cells = _closure(function) or {}
    return tuple(
        value
        for value in cells.values()
        if isinstance(value, np.ndarray) and value.flags.writeable
    )
