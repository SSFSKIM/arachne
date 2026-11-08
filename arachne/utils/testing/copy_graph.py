"""Whether two deep copies of one object are the same copy.

:func:`copy_differences` walks an original and two copies of it side by side
and reports where the copies differ: in which objects they share with the
original, in how their new objects alias each other, or in what those new
objects hold. Two copies with no difference are interchangeable: each new
object of one has its counterpart in the other, holding the same values, and
both share exactly the same objects with the original.
"""

from __future__ import annotations

import copyreg
import types
from collections import Counter
from typing import Any

import numpy as np

__all__ = ["copy_differences", "reachable"]

_SCALARS = (type(None), bool, int, float, complex, str, bytes, np.generic)
# Code, which a deep copy shares and the walk doesn't enter.
_CODE = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
    types.CodeType,
    type,
)


def _edges(x: Any, opaque: Counter[str] | None = None) -> list[tuple[str, Any]]:
    """What a deep copy reaches from ``x``, each with how it is reached."""
    if isinstance(x, _SCALARS) or isinstance(x, _CODE):
        return []
    if isinstance(x, types.MethodType):
        return [(".__func__", x.__func__), (".__self__", x.__self__)]
    if isinstance(x, (list, tuple)):
        return [(f"[{i}]", v) for i, v in enumerate(x)]
    if isinstance(x, dict):
        out: list[tuple[str, Any]] = []
        for i, (k, v) in enumerate(x.items()):
            out.append((f"<key {i}>", k))
            out.append((f"[{k!r:.40}]", v))
        return out
    if isinstance(x, (set, frozenset)):
        return [(f"<item {i}>", v) for i, v in enumerate(x)]
    out = []
    if isinstance(x, np.ndarray):
        if x.base is not None:
            out.append(("<base>", x.base))
        if x.dtype.hasobject:
            out += [(f"<item {i}>", v) for i, v in enumerate(x.ravel().tolist())]
    own = getattr(x, "__dict__", None)
    if isinstance(own, dict):
        out.append(("<__dict__>", own))
    slots = copyreg._slotnames(type(x))
    for name in slots:
        if hasattr(x, name):
            out.append((f".{name}", getattr(x, name)))
    if opaque is not None and not out and not isinstance(x, np.ndarray) and not slots:
        opaque[f"{type(x).__module__}.{type(x).__qualname__}"] += 1
    return out


def reachable(x: Any) -> dict[int, Any]:
    """Every object reached from ``x`` by the walk :func:`copy_differences`
    makes, by id; code and classes are reached but not walked into.
    """
    seen: dict[int, Any] = {}
    stack = [x]
    while stack:
        obj = stack.pop()
        if id(obj) not in seen:
            seen[id(obj)] = obj
            stack.extend(v for _, v in _edges(obj))
    return seen


def _same_leaf(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and a.flags.writeable == b.flags.writeable
            and (a.dtype.hasobject or a.tobytes() == b.tobytes())
        )
    if isinstance(a, _SCALARS):
        return repr(a) == repr(b)
    return True


def copy_differences(
    original: Any,
    reference: Any,
    candidate: Any,
    *,
    limit: int = 20,
    opaque: Counter[str] | None = None,
) -> list[str]:
    """Where ``candidate``, a copy of ``original``, differs from ``reference``,
    another copy of it: at most ``limit`` differences, each naming the path
    to it from the root. An empty list means the two are the same copy.

    An object reached from ``original`` is *shared*: a copy that holds it
    holds the original's object itself. At every place the walk reaches,
    either both copies hold the same shared object, or both hold new objects
    of one type that pair up consistently (an object of ``reference`` is
    always met with the same object of ``candidate``, and the other way
    round), holding the same values: arrays by dtype, shape, writability and
    bytes, numbers and strings by ``repr``, containers and instances item by
    item.
    """
    shared = reachable(original)
    paired: dict[int, Any] = {}
    paired_back: dict[int, Any] = {}
    found: list[str] = []
    stack: list[tuple[str, Any, Any]] = [("", reference, candidate)]
    while stack and len(found) < limit:
        path, a, b = stack.pop()
        a_shared, b_shared = id(a) in shared, id(b) in shared
        if a_shared or b_shared:
            if a is not b:
                if a_shared and b_shared:
                    found.append(f"{path or '<root>'}: shares another object")
                elif b_shared:
                    found.append(
                        f"{path or '<root>'}: shares {type(b).__name__} that the reference copies"
                    )
                else:
                    found.append(
                        f"{path or '<root>'}: copies {type(a).__name__} that the reference shares"
                    )
            continue
        if type(a) is not type(b):
            found.append(f"{path or '<root>'}: {type(a).__name__} against {type(b).__name__}")
            continue
        known = paired.get(id(a))
        if known is not None or id(b) in paired_back:
            if known is not b or paired_back.get(id(b)) is not a:
                found.append(f"{path or '<root>'}: aliases differently")
            continue
        paired[id(a)] = b
        paired_back[id(b)] = a
        if not _same_leaf(a, b):
            found.append(f"{path or '<root>'}: {a!r:.60} against {b!r:.60}")
            continue
        edges_a, edges_b = _edges(a, opaque), _edges(b)
        if [name for name, _ in edges_a] != [name for name, _ in edges_b]:
            found.append(f"{path or '<root>'}: holds different fields")
            continue
        for (name, va), (_, vb) in zip(reversed(edges_a), reversed(edges_b)):
            stack.append((f"{path}{name}", va, vb))
    # Keep the paired objects alive until the walk is over, so that no id is
    # reused while it is still being compared.
    del paired, paired_back
    return found
