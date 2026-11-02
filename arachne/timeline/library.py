"""The library calls a state-function callback (route 2) may make.

A callback may call a library function only when it is an entry of the
table (:func:`entry`), looked up by identity: a builtin, a ``math`` or numpy
function, one of numpy's own ufuncs, or an engine helper. Nothing is
admitted by the module it comes from or the namespace it is found in, so a
function the table does not list (``np.put_along_axis``, ``random_color``,
a ufunc made with ``np.frompyfunc``) makes the play tick, however pure it
looks.

Every entry keeps one effect contract: called in its admitted form, it
changes none of its arguments, returns the same value for the same
arguments, and reads no hidden state (a random generator, a clock, the
environment, a file). The admitted form is part of the entry:

- ``positional``: how many arguments it may be given by position. Beyond
  them sit the output array (``np.clip(a, lo, hi, out)``) or options the
  contract was not checked for. ``None`` admits any number, for functions
  that have no output parameter (``max``, ``zip``, the engine's
  ``average_color``).
- ``keywords``: the keywords it may be given; nothing else, so ``out=``,
  ``where=`` and ``casting=`` are never admitted.
- ``allocates``: it always returns a new value, never one of its arguments
  or a view of one. Where this is not certain the entry says ``False``, and
  the proof treats the result as the argument it may be (an array the
  callback did not make stays one it may not change).
- ``calls``: it calls a callable it is given (``sorted``'s ``key``,
  ``binary_search``'s ``function``); each such callable must be proven a
  readout in turn.
- ``takes_mobjects``: it may be given mobjects, of which it reads only
  their count or identity, or which it hands back in a container (``len``,
  ``isinstance``, ``list``, ``zip``).

Methods of plain values (an array's ``sum``, a string's ``format``) are
admitted the same way, by name, from :data:`VALUE_METHODS`: the receiver is
a plain value, whose type the engine's own values (numbers, arrays,
strings, colours) bound.
"""

from __future__ import annotations

__all__ = ["VALUE_METHODS", "Entry", "entry", "library", "value_method"]

import builtins
import inspect
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Entry:
    """A library function's admitted call form (see the module docstring)."""

    name: str
    positional: int | None
    keywords: frozenset[str] = frozenset()
    allocates: bool = False
    calls: bool = False
    takes_mobjects: bool = False


def _kw(*names: str) -> frozenset[str]:
    return frozenset(names)


# Builtins: name -> entry.
_BUILTINS: dict[str, Entry] = {
    entry.name: entry
    for entry in (
        Entry("abs", 1, allocates=True),
        Entry("round", 2, _kw("ndigits"), allocates=True),
        Entry("int", 1, allocates=True),
        Entry("float", 1, allocates=True),
        Entry("complex", 2, allocates=True),
        Entry("bool", 1, allocates=True),
        Entry("str", 1, allocates=True),
        Entry("len", 1, allocates=True, takes_mobjects=True),
        Entry("isinstance", 2, allocates=True, takes_mobjects=True),
        Entry("range", 3, allocates=True),
        Entry("sum", 2, _kw("start")),
        Entry("pow", 3),
        Entry("divmod", 2, allocates=True),
        Entry("any", 1, allocates=True),
        Entry("all", 1, allocates=True),
        Entry("min", None, _kw("key", "default"), calls=True),
        Entry("max", None, _kw("key", "default"), calls=True),
        Entry("sorted", 1, _kw("key", "reverse"), calls=True),
        Entry("map", None, calls=True),
        Entry("filter", 2, calls=True),
        Entry("list", 1, takes_mobjects=True),
        Entry("tuple", 1, takes_mobjects=True),
        Entry("reversed", 1, takes_mobjects=True),
        Entry("zip", None, _kw("strict"), takes_mobjects=True),
        Entry("enumerate", 2, _kw("start"), takes_mobjects=True),
        Entry("set", 1, allocates=True),
        Entry("frozenset", 1, allocates=True),
        Entry("dict", 1),
        Entry("slice", 3, allocates=True),
    )
}

#: ``math`` functions, all of which compute a number from numbers.
_MATH = (
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "comb",
    "copysign",
    "cos",
    "cosh",
    "degrees",
    "dist",
    "erf",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "fmod",
    "fsum",
    "gamma",
    "gcd",
    "hypot",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "perm",
    "pow",
    "prod",
    "radians",
    "remainder",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
)

#: numpy's own ufuncs, called with their inputs only.
_UFUNCS = (
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "cbrt",
    "ceil",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "hypot",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "matmul",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "rint",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
)

#: numpy functions: name -> (positional, keywords, allocates). ``positional``
#: stops before the output parameter of those that have one.
_NUMPY: dict[str, tuple[int | None, frozenset[str], bool]] = {
    "abs": (1, _kw(), True),
    "allclose": (2, _kw("rtol", "atol"), True),
    "amax": (1, _kw("axis"), True),
    "amin": (1, _kw("axis"), True),
    "append": (2, _kw("axis"), True),
    "arange": (3, _kw("dtype"), True),
    "argmax": (1, _kw("axis"), True),
    "argmin": (1, _kw("axis"), True),
    "around": (2, _kw("decimals"), True),
    "array": (2, _kw("dtype"), True),
    "array_equal": (2, _kw(), True),
    "asarray": (2, _kw("dtype"), False),
    "average": (2, _kw("axis", "weights"), True),
    "clip": (3, _kw(), True),
    "column_stack": (1, _kw(), True),
    "concatenate": (1, _kw("axis"), True),
    "copy": (1, _kw(), True),
    "cross": (2, _kw("axis"), True),
    "cumsum": (1, _kw("axis"), True),
    "diff": (1, _kw("n", "axis"), True),
    "dot": (2, _kw(), True),
    "eye": (1, _kw("dtype"), True),
    "full": (2, _kw("dtype"), True),
    "hstack": (1, _kw(), True),
    "identity": (1, _kw("dtype"), True),
    "interp": (3, _kw("left", "right", "period"), True),
    "isclose": (2, _kw("rtol", "atol"), True),
    "linspace": (3, _kw("num", "endpoint", "dtype"), True),
    "max": (1, _kw("axis"), True),
    "mean": (1, _kw("axis"), True),
    "min": (1, _kw("axis"), True),
    "ones": (1, _kw("dtype"), True),
    "outer": (2, _kw(), True),
    "prod": (1, _kw("axis"), True),
    "ravel": (1, _kw(), False),
    "reshape": (2, _kw(), False),
    "round": (2, _kw("decimals"), True),
    "stack": (1, _kw("axis"), True),
    "sum": (1, _kw("axis"), True),
    "tile": (2, _kw(), True),
    "transpose": (1, _kw(), False),
    "vstack": (1, _kw(), True),
    "where": (3, _kw(), True),
    "zeros": (1, _kw("dtype"), True),
}
_NUMPY_LINALG: dict[str, tuple[int | None, frozenset[str], bool]] = {
    "det": (1, _kw(), True),
    "inv": (1, _kw(), True),
    "norm": (2, _kw("ord", "axis"), True),
    "solve": (2, _kw(), True),
}

#: Engine helpers, by module: each computes a value from its arguments,
#: changes none of them and draws nothing at random (``random_color`` and
#: ``random_bright_color`` are not here). Their call form is their
#: signature: every parameter, by position or by name.
_ENGINE: dict[str, tuple[str, ...]] = {
    "arachne.utils.space_ops": (
        "R3_to_complex",
        "angle_between_vectors",
        "angle_of_vector",
        "cartesian_to_spherical",
        "center_of_mass",
        "compass_directions",
        "complex_to_R3",
        "cross",
        "cross2d",
        "get_unit_normal",
        "line_intersection",
        "midpoint",
        "norm_squared",
        "normalize",
        "regular_vertices",
        "rotate_vector",
        "rotation_about_z",
        "rotation_matrix",
        "spherical_to_cartesian",
        "z_to_vector",
    ),
    "arachne.utils.bezier": (
        "integer_interpolate",
        "interpolate",
        "inverse_interpolate",
        "match_interpolate",
        "mid",
    ),
    "arachne.utils.simple_functions": ("clip", "sigmoid"),
    "arachne.utils.color.core": (
        "average_color",
        "color_gradient",
        "color_to_int_rgb",
        "color_to_int_rgba",
        "color_to_rgb",
        "color_to_rgba",
        "hex_to_rgb",
        "interpolate_color",
        "invert_color",
        "rgb_to_color",
        "rgb_to_hex",
        "rgba_to_color",
    ),
}
#: Engine helpers that call a function they are given.
_ENGINE_CALLING: dict[str, tuple[str, ...]] = {
    "arachne.utils.simple_functions": ("binary_search",),
}


def _signature_entry(function: Callable[..., Any], name: str, *, calls: bool) -> Entry:
    """An entry whose call form is ``function``'s signature: each parameter
    by position or by name, a ``*args`` taking any number.
    """
    parameters = inspect.signature(function).parameters.values()
    positional: int | None = 0
    keywords: set[str] = set()
    for parameter in parameters:
        if parameter.kind is parameter.VAR_POSITIONAL:
            positional = None
        elif parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            if positional is not None:
                positional += 1
            if parameter.kind is parameter.POSITIONAL_OR_KEYWORD:
                keywords.add(parameter.name)
        elif parameter.kind is parameter.KEYWORD_ONLY:
            keywords.add(parameter.name)
    return Entry(name, positional, frozenset(keywords), calls=calls)


def _math_entry(name: str) -> Entry:
    keywords = _kw("rel_tol", "abs_tol") if name == "isclose" else _kw()
    return Entry(f"math.{name}", None, keywords, allocates=True)


def _inputs(ufunc: np.ufunc) -> int:
    """How many inputs a ufunc takes: its arguments before the outputs."""
    return int(ufunc.nargs - ufunc.nout)


def _entries() -> Iterable[tuple[Any, Entry]]:
    import importlib

    from arachne.utils.color.core import ManimColor

    for name, value in _BUILTINS.items():
        yield getattr(builtins, name), value
    for name in _MATH:
        yield getattr(math, name), _math_entry(name)
    for name in _UFUNCS:
        ufunc = getattr(np, name)
        assert isinstance(ufunc, np.ufunc), name
        yield ufunc, Entry(f"numpy.{name}", _inputs(ufunc), allocates=True)
    for space, prefix, table in (
        (np, "numpy", _NUMPY),
        (np.linalg, "numpy.linalg", _NUMPY_LINALG),
    ):
        for name, (positional, keywords, allocates) in table.items():
            value = getattr(space, name)
            if isinstance(value, np.ufunc):
                # np.abs and its kin are ufuncs under another name.
                yield value, Entry(f"{prefix}.{name}", _inputs(value), allocates=True)
            else:
                yield value, Entry(f"{prefix}.{name}", positional, keywords, allocates)
    for tables, calls in ((_ENGINE, False), (_ENGINE_CALLING, True)):
        for module_name, names in tables.items():
            module = importlib.import_module(module_name)
            for name in names:
                function = getattr(module, name)
                yield function, _signature_entry(function, name, calls=calls)
    yield ManimColor, Entry("ManimColor", 2, _kw("alpha"), allocates=True)


@cache
def _table() -> dict[int, tuple[Any, Entry]]:
    # Keyed by identity, holding the function so its id is never reused.
    return {id(function): (function, value) for function, value in _entries()}


def entry(function: Any) -> Entry | None:
    """The library entry ``function`` is, by identity, or ``None``."""
    found = _table().get(id(function))
    if found is None or found[0] is not function:
        return None
    return found[1]


def library() -> list[tuple[Any, Entry]]:
    """Every entry with its function, for tests and documentation."""
    return list(_table().values())


#: Methods of plain values: name -> (positional, keywords, returns a view of
#: the receiver). ``positional`` stops before an output parameter.
VALUE_METHODS: dict[str, tuple[int | None, frozenset[str] | None, bool]] = {
    "copy": (0, _kw(), False),
    "dot": (1, _kw(), False),
    "sum": (1, _kw("axis"), False),
    "mean": (1, _kw("axis"), False),
    "max": (1, _kw("axis"), False),
    "min": (1, _kw("axis"), False),
    "reshape": (None, _kw(), True),
    "flatten": (0, _kw(), False),
    "tolist": (0, _kw(), False),
    "astype": (1, _kw(), True),
    "round": (1, _kw(), False),
    "clip": (2, _kw(), False),
    "conjugate": (0, _kw(), False),
    "item": (None, _kw(), False),
    "format": (None, None, False),
    "upper": (0, _kw(), False),
    "lower": (0, _kw(), False),
    "strip": (1, _kw(), False),
    "split": (2, _kw(), False),
    "join": (1, _kw(), False),
    "replace": (3, _kw(), False),
}


def value_method(name: str) -> tuple[int | None, frozenset[str] | None, bool] | None:
    """The admitted call form of a plain value's method ``name``: how many
    positional arguments, which keywords (``None``: any, for ``format``),
    and whether it returns a view of the receiver; ``None`` when it is not
    admitted.
    """
    return VALUE_METHODS.get(name)
