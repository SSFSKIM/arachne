"""What a world refuses as a scene declares it: a callback force that reads
a displayed object (``sim.force_input``), and a solver feature that is
withheld (``sim.withheld``).

**A callback force reads the state it is handed.** It is called once for
each new tick with the world's state at the tick before, by body handle, and
reads another world through that world's :meth:`~.World.at_tick`. A
displayed mobject is a different clock: between two frames it holds the last
frame's values (:mod:`arachne.timeline.ticking`), so a force computed from
one changes with the frame rate. So a callback whose code can reach a
mobject or the scene directly is refused as it is declared: through its
closure, its defaults, the globals its code loads (its own and its nested
functions'), a bound method's object, a ``functools.partial``'s arguments,
a callable object's attributes, and the lists, tuples, sets and dicts any of
these hold. A world is not refused (its :meth:`~.World.at_tick` is
tick-indexed), nor is a body handle. What the check cannot see, a mobject
reached through an attribute of another object, is not supported: it reads
the last frame's value at a tick no frame reads.

**A withheld feature** is one whose arithmetic is not yet the same on every
machine: :data:`WITHHELD` names each, with the reason. Declaring one raises
``sim.withheld``. At the solver pin nothing is withheld.
"""

from __future__ import annotations

__all__ = ["WITHHELD", "check_feature", "check_force"]

import dis
import functools
import types
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from arachne.lint.raised import RaisedFinding

#: The solver features a scene may not declare, each with why.
WITHHELD: Mapping[str, str] = MappingProxyType({})

#: How deep into lists, tuples, sets and dicts the check looks.
_DEPTH = 4


def check_feature(feature: str) -> None:
    """Refuse ``feature`` (``joint.pin.limits``, ``shape.convex_hull``,
    ``force.spring``, ...) if it is withheld.
    """
    reason = WITHHELD.get(feature)
    if reason is not None:
        raise RaisedFinding(
            "sim.withheld",
            f"the solver feature {feature} is withheld: {reason}",
            detail={"feature": feature},
        )


def check_force(function: Any) -> None:
    """Refuse a callback force whose code can reach a mobject or the scene,
    naming each name it reaches one through.
    """
    if not callable(function):
        raise TypeError(f"A callback force is callable, not {function!r}.")
    names = sorted(
        {name for name, value in _reachable(function) if _displayed(value, _DEPTH)}
    )
    if names:
        force = getattr(function, "__qualname__", type(function).__qualname__)
        raise RaisedFinding(
            "sim.force_input",
            f"the callback force {force} reads {', '.join(names)}, a mobject or"
            " the scene, whose values between frames are the last frame's, not"
            " the tick's",
            detail={"force": force, "names": names},
        )


def _reachable(function: Any) -> Iterator[tuple[str, Any]]:
    """Each value ``function``'s code can reach without an attribute: by the
    name it is reached through.
    """
    if isinstance(function, types.MethodType):
        yield "self", function.__self__
        yield from _reachable(function.__func__)
        return
    if isinstance(function, functools.partial):
        for index, value in enumerate(function.args):
            yield f"argument {index}", value
        yield from function.keywords.items()
        yield from _reachable(function.func)
        return
    if isinstance(function, types.FunctionType):
        code = function.__code__
        for name, cell in zip(
            code.co_freevars, function.__closure__ or (), strict=True
        ):
            try:
                yield name, cell.cell_contents
            except ValueError:  # a cell not bound yet
                continue
        defaults = function.__defaults__ or ()
        first = code.co_argcount - len(defaults)
        yield from zip(
            code.co_varnames[first : code.co_argcount], defaults, strict=True
        )
        yield from (function.__kwdefaults__ or {}).items()
        missing = object()
        for name in _global_names(code):
            value = function.__globals__.get(name, missing)
            if value is not missing:
                yield name, value
        return
    if isinstance(function, (types.BuiltinFunctionType, type)):
        return
    # A callable object: what it holds, and what its __call__ reaches.
    yield from (getattr(function, "__dict__", None) or {}).items()
    call = vars(type(function)).get("__call__")
    if isinstance(call, types.FunctionType):
        yield from _reachable(types.MethodType(call, function))


def _global_names(code: types.CodeType) -> Iterator[str]:
    """The globals ``code`` and the code nested in it load."""
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            yield str(instruction.argval)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _global_names(const)


def _displayed(value: Any, depth: int) -> bool:
    """Whether ``value`` is, or directly holds, a mobject (a world aside) or a
    scene.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    from .world import World

    if isinstance(value, Scene):
        return True
    if isinstance(value, Mobject):
        return not isinstance(value, World)
    if depth <= 0:
        return False
    if isinstance(value, dict):
        return any(_displayed(item, depth - 1) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_displayed(item, depth - 1) for item in value)
    return False
