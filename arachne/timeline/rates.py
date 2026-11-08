"""The rate and path functions the trusted algebra admits, and the array
twins of the rate functions.

A rate or path function is admitted by identity, never by what its code looks
like or what it compares equal to: CE's own rate functions (the module
attributes of :mod:`arachne.utils.rate_functions`, each the very object), the
closures CE's factories return (``squish_rate_func`` and ``not_quite_there``:
a plain function whose code is the factory's own, over an admitted function,
with plain numbers bound), and CE's straight, arc and circle paths. Anything
else, a user's lambda or a callable whose hash and equality are ``linear``'s
included, is unproven, whatever it computes. :func:`engine_rate` is the one
rule for rate functions; :func:`admitted_rate`, :func:`is_monotone` and
:func:`array_twin` read it.

Every admitted rate function has an *array twin* that evaluates it over an
array of instants at once, equal to calling the scalar original on each.
"""

from __future__ import annotations

__all__ = [
    "MONOTONE_RATE_FUNCTIONS",
    "RATE_FUNCTIONS",
    "EngineRate",
    "admitted_path",
    "admitted_rate",
    "array_twin",
    "engine_rate",
    "frozen_path",
    "is_monotone",
]

import types
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np

from arachne.utils import paths
from arachne.utils import rate_functions as rf
from arachne.utils.bezier import interpolate

from .identity import ByIdentity

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
    result[~ends] = -_pow2(10 * u - 10) * np.sin((u * 10 - 10.75) * c4)
    return result


def _ease_out_elastic(t: np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    result = np.empty_like(t)
    ends = (t == 0) | (t == 1)
    result[t == 0] = 0
    result[t == 1] = 1
    u = t[~ends]
    result[~ends] = _pow2(-10 * u) * np.sin((u * 10 - 0.75) * c4) + 1
    return result


def _ease_in_out_elastic(t: np.ndarray) -> np.ndarray:
    c5 = (2 * np.pi) / 4.5
    result = np.empty_like(t)
    result[t == 0] = 0
    result[t == 1] = 1
    low = (t != 0) & (t != 1) & (t < 0.5)
    high = (t != 0) & (t != 1) & ~(t < 0.5)
    u = t[low]
    result[low] = -(_pow2(20 * u - 10) * np.sin((20 * u - 11.125) * c5)) / 2
    u = t[high]
    result[high] = (_pow2(-20 * u + 10) * np.sin((20 * u - 11.125) * c5)) / 2 + 1
    return result


def _ease_in_out_expo(t: np.ndarray) -> np.ndarray:
    result = np.empty_like(t)
    result[t == 0] = 0
    result[t == 1] = 1
    low = (t != 0) & (t != 1) & (t < 0.5)
    high = (t != 0) & (t != 1) & ~(t < 0.5)
    result[low] = _pow2(20 * t[low] - 10) / 2
    result[high] = (2 - _pow2(-20 * t[high] + 10)) / 2
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


#: Python's own ``pow`` over arrays. numpy's ``power`` rounds differently
#: from the C library's ``pow`` that Python's ``**`` calls, by fast paths
#: (``x**2`` as ``x*x``) and by the SIMD kernels some CPUs select, so a twin
#: that must equal CE's Python arithmetic calls it element by element.
_PYTHON_POW = np.frompyfunc(pow, 2, 1)


def _pow(base: np.ndarray, exponent: float) -> np.ndarray:
    """Python's ``base ** exponent`` on floats, elementwise."""
    return np.asarray(
        _PYTHON_POW(np.asarray(base, dtype=float), float(exponent)), dtype=float
    )


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
    return np.asarray(_PYTHON_POW(2.0, np.asarray(exponent, dtype=float)), dtype=float)


#: CE's rate functions: every module attribute that is a rate function (the
#: factories ``squish_rate_func`` and ``not_quite_there`` excepted; their
#: closures are admitted through :func:`engine_rate`). A set also takes what
#: only compares equal to a member, so admission never asks it:
#: :func:`admitted_rate` asks by identity.
RATE_FUNCTIONS: frozenset[Callable[..., float]] = frozenset(_TWINS)

#: CE's rate functions that never decrease. A group whose time runs through
#: one of these crosses each part boundary once, at an instant bisection can
#: find. As with :data:`RATE_FUNCTIONS`, ask :func:`is_monotone`.
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

#: Each of CE's rate functions and its array twin, by identity.
_ENGINE_RATES: ByIdentity[Callable[..., float], ArrayFunction] = ByIdentity(_TWINS)
_MONOTONE = ByIdentity.of(MONOTONE_RATE_FUNCTIONS)

_SQUISH_CODE = getattr(rf.squish_rate_func(rf.linear), "__code__")  # noqa: B009
_NOT_QUITE_CODE = getattr(rf.not_quite_there(rf.linear), "__code__")  # noqa: B009

#: The numbers each factory's closure binds, by name.
_FACTORY_NUMBERS = {"squish": ("a", "b"), "not_quite_there": ("proportion",)}

#: The numbers an admitted closure may bind, by exact type: Python's
#: integers, floats and fractions, and numpy's integer and float scalars. A
#: subclass may define its own arithmetic and comparisons, which are a
#: script's code.
_NUMBERS = ByIdentity.of(
    (
        int,
        float,
        Fraction,
        *(np.byte, np.short, np.intc, np.int_, np.longlong),
        *(np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong),
        *(np.half, np.single, np.double, np.longdouble),
    )
)


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
    return type(value) in _NUMBERS


class EngineRate(NamedTuple):
    """A rate function as CE's: the rate function at its core, and the
    factory closures over it, outermost first, each as its kind
    (``"squish"`` or ``"not_quite_there"``) and the values it binds by name.
    """

    base: Callable[[float], float]
    layers: tuple[tuple[str, Mapping[str, Any]], ...]


def _layer(function: Any) -> tuple[str, dict[str, Any]] | None:
    """``function`` as one of CE's factory closures (a plain function whose
    code is the factory's own, binding plain numbers), or ``None``.
    """
    if type(function) is not types.FunctionType:
        return None
    code = function.__code__
    if code is _SQUISH_CODE:
        kind = "squish"
    elif code is _NOT_QUITE_CODE:
        kind = "not_quite_there"
    else:
        return None
    cells = _closure(function)
    if cells is None or not all(_number(cells.get(n)) for n in _FACTORY_NUMBERS[kind]):
        return None
    return kind, cells


def engine_rate(function: Any) -> EngineRate | None:
    """``function`` as CE's rate function, or ``None``: one of CE's rate
    functions itself, or a closure a rate factory returned over one, at any
    depth (:func:`_layer`). Each step is by identity, so a callable that only
    compares equal to one of CE's, or only carries a factory's code, is not
    one, and none of its own code runs to tell.
    """
    layers: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[int] = set()
    while function not in _ENGINE_RATES:
        if id(function) in seen:  # a closure whose cell holds itself
            return None
        seen.add(id(function))
        layer = _layer(function)
        if layer is None:
            return None
        layers.append(layer)
        function = layer[1].get("func")
    return EngineRate(function, tuple(layers))


def admitted_rate(function: Any) -> bool:
    """Whether ``function`` is a rate function the trusted algebra admits."""
    return engine_rate(function) is not None


def is_monotone(function: Any) -> bool:
    """Whether ``function`` is an admitted rate function that never decreases."""
    rate = engine_rate(function)
    if rate is None or rate.base not in _MONOTONE:
        return False
    for kind, cells in rate.layers:
        if kind == "squish" and not cells["a"] <= cells["b"]:
            return False
        if kind == "not_quite_there" and not cells["proportion"] >= 0:
            return False
    return True


def _squish(inner: ArrayFunction, a: float, b: float) -> ArrayFunction:
    def twin(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if a == b:
            return np.full_like(t, a)
        new_t = np.where(t < a, 0.0, np.where(t > b, 1.0, (t - a) / (b - a)))
        return inner(new_t)

    return twin


def _not_quite_there(inner: ArrayFunction, proportion: float) -> ArrayFunction:
    return lambda t: proportion * inner(np.asarray(t, dtype=float))


def array_twin(function: Callable[..., float]) -> ArrayFunction:
    """The array twin of an admitted rate function: it evaluates ``function``
    over an array of alphas, equal to calling it on each.
    """
    rate = engine_rate(function)
    if rate is None:
        raise KeyError(f"{function!r} is not an admitted rate function.")
    twin = _ENGINE_RATES[rate.base]
    for kind, cells in reversed(rate.layers):
        if kind == "squish":
            twin = _squish(twin, cells["a"], cells["b"])
        else:
            twin = _not_quite_there(twin, cells["proportion"])
    return twin


#: The code of each of CE's path closures, by identity, and the name of the
#: angle it binds.
_PATH_ANGLES: ByIdentity[types.CodeType, str] = ByIdentity(
    {
        paths.path_along_arc(1.0).__code__: "arc_angle",
        paths.path_along_circles(1.0, np.zeros(3)).__code__: "arc_angle",
        paths.spiral_path(1.0).__code__: "angle",
    }
)


def _points(value: Any) -> bool:
    """A plain numpy array of numbers: not a subclass, whose arithmetic is
    its own code, and not of objects, whose arithmetic is theirs.
    """
    return type(value) is np.ndarray and value.dtype.kind in ("f", "i", "u")


def admitted_path(function: Any) -> bool:
    """Whether ``function`` is a path function the trusted algebra admits:
    the straight path (``interpolate``) or a closure of CE's arc, spiral or
    circle paths (a plain function whose code is the path's own) whose
    angle is a number (:data:`_NUMBERS`) and whose every other value (the
    axis, the circles' centres) is a plain array of numbers. The arrays may
    be the script's own (``Rotate`` binds its ``about_point``): a trusted
    play runs no script code, and :func:`frozen_path` lets an evaluation
    after the play see them as the play did.
    """
    if function is interpolate:
        return True
    if type(function) is not types.FunctionType:
        return False
    angle = _PATH_ANGLES.get(function.__code__)
    if angle is None:
        return False
    cells = _closure(function)
    return (
        cells is not None
        and _number(cells.get(angle))
        and all(_points(v) for k, v in cells.items() if k != angle)
    )


def frozen_path(function: Any) -> Any:
    """``function`` with every array its closure binds replaced by a
    detached, read-only copy of what it holds now; ``function`` itself when it
    binds none. A read-only view still changes with its base, so every array
    is copied, whatever its flags.
    """
    cells = getattr(function, "__closure__", None)
    if not cells:
        return function
    try:
        contents = [cell.cell_contents for cell in cells]
    except ValueError:  # an empty cell
        return function
    if not any(isinstance(value, np.ndarray) for value in contents):
        return function
    frozen = tuple(
        types.CellType(_detached(value) if isinstance(value, np.ndarray) else value)
        for value in contents
    )
    return types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        frozen,
    )


def _detached(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.flags.writeable = False
    return copy
