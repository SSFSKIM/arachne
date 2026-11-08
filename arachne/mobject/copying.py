"""A mobject's deep copy, made without :func:`copy.deepcopy`'s general machinery
(``fast_copy``).

:meth:`.Mobject.copy` is :func:`copy.deepcopy`, and :mod:`copy` dispatches
every value it meets through a Python call: a mobject's dozens of numbers and
flags, and each point of the SVG path a glyph keeps. This module makes the
same copy by the same rules, with the common kinds of value handled inline:

- values :mod:`copy` returns as they are (numbers, strings, functions,
  classes) are returned as they are;
- lists, dicts and tuples are copied as :mod:`copy` copies them;
- a mobject is copied as :meth:`.Mobject.__deepcopy__` copies it;
- a value with its own ``__deepcopy__`` (a read-only array, a
  :class:`~.Paint`) is asked for its copy;
- an instance of a plain Python class, which :mod:`copy` rebuilds from its
  ``__reduce_ex__`` (a colour, an SVG path and its points), is rebuilt
  directly: a new instance whose ``__dict__`` is a copy of the original's;
- anything else is handed to :func:`copy.deepcopy`.

The copy is the one :func:`copy.deepcopy` makes: the same objects are new and
the same are shared, the memo records the same originals, and two references
to one object in the original are two references to one object in the copy.
"""

from __future__ import annotations

import copy
import copyreg
import enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .mobject import Mobject

_M = TypeVar("_M", bound="Mobject")

__all__ = ["copy_mobject"]

_nil: list[Any] = []

# What copy.deepcopy reads, from copy and copyreg: private, so typeshed
# doesn't declare them.
_DISPATCH: dict[type, Any] = copy._deepcopy_dispatch  # type: ignore[attr-defined]
_NEWOBJ: Any = copyreg.__newobj__  # type: ignore[attr-defined]
_slotnames: Any = copyreg._slotnames  # type: ignore[attr-defined]

# The types copy.deepcopy returns as they are, without a memo entry: its
# atomic set where it has one (3.14), else the types its dispatch table sends
# to ``_deepcopy_atomic``.
if hasattr(copy, "_atomic_types"):
    _ATOMIC: frozenset[type] = frozenset(copy._atomic_types)
else:
    _ATOMIC = frozenset(
        cls
        for cls, copier in _DISPATCH.items()
        if copier is copy._deepcopy_atomic  # type: ignore[attr-defined]
    )

# How each class's instances are copied, found once per class.
_ATOM, _LIST, _DICT, _TUPLE, _MOBJECT, _OWN, _PLAIN, _GENERAL = range(8)
_KINDS: dict[type, int] = dict.fromkeys(_ATOMIC, _ATOM)
_KINDS.update({list: _LIST, dict: _DICT, tuple: _TUPLE})

_HEAPTYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class defined in Python
_OBJECT_STATE = (object.__reduce_ex__, object.__reduce__, object.__getstate__)


def _classify(cls: type, x: Any) -> int:
    from .mobject import Mobject

    if cls in _DISPATCH:
        # The dispatch table's other entries (bound methods) are left to it.
        kind = _GENERAL
    elif issubclass(cls, type):
        kind = _ATOM
    elif getattr(cls, "__deepcopy__", None) is Mobject.__deepcopy__:
        kind = _MOBJECT
    elif getattr(cls, "__deepcopy__", None) is enum.Enum.__deepcopy__:
        # An enum member is its own copy, and copy.deepcopy keeps no memo
        # entry for it.
        kind = _ATOM
    elif getattr(cls, "__deepcopy__", None) is not None:
        kind = _GENERAL if _custom_lookup(cls) else _OWN
    elif _plain(cls, x):
        kind = _PLAIN
    else:
        kind = _GENERAL
    _KINDS[cls] = kind
    return kind


def _custom_lookup(cls: type) -> bool:
    """Whether reading an attribute of an instance may run the class's code."""
    return (
        cls.__getattribute__ is not object.__getattribute__
        or getattr(cls, "__getattr__", None) is not None
    )


def _plain(cls: type, x: Any) -> bool:
    """Whether :mod:`copy` rebuilds ``cls``'s instances as ``object`` reduces
    them: ``cls.__new__(cls)``, then the original's ``__dict__``, copied, as
    the new instance's.
    """
    if (
        cls in copyreg.dispatch_table
        or (cls.__reduce_ex__, cls.__reduce__, cls.__getstate__) != _OBJECT_STATE
        or _custom_lookup(cls)
        or any(
            hasattr(cls, name)
            for name in ("__setstate__", "__getnewargs_ex__", "__getnewargs__")
        )
        or not all(base is object or base.__flags__ & _HEAPTYPE for base in cls.__mro__)
        or not cls.__dictoffset__
        or _slotnames(cls)
    ):
        return False
    # And the instance at hand reduces that way.
    reduced = x.__reduce_ex__(4)
    state = x.__dict__ or None
    return (
        len(reduced) == 5
        and reduced[0] is _NEWOBJ
        and reduced[1] == (cls,)
        and reduced[2] is state
        and reduced[3] is None
        and reduced[4] is None
    )


def _copy(x: Any, memo: dict[int, Any], alive: list[Any]) -> Any:
    """``copy.deepcopy(x, memo)``, where ``alive`` is the memo's list of the
    originals it keeps alive (``memo[id(memo)]``).
    """
    cls: Any = type(x)
    kinds = _KINDS
    kind = kinds.get(cls)
    if kind is _ATOM:
        return x
    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y
    if kind is None:
        kind = _classify(cls, x)
        if kind is _ATOM:
            return x
    if kind is _PLAIN:
        state = x.__dict__
        if "__deepcopy__" in state or "__reduce_ex__" in state or "__reduce__" in state:
            return copy.deepcopy(x, memo)
        y = cls.__new__(cls)
        memo[d] = y
        if state:
            s = id(state)
            copied = memo.get(s, _nil)
            if copied is _nil:
                copied = {}
                memo[s] = copied
                for key, value in state.items():
                    if kinds.get(type(key)) is not _ATOM:
                        key = _copy(key, memo, alive)
                    if kinds.get(type(value)) is not _ATOM:
                        value = _copy(value, memo, alive)
                    copied[key] = value
                alive.append(state)
            y.__dict__.update(copied)
    elif kind is _LIST:
        y = []
        memo[d] = y
        append = y.append
        for a in x:
            append(a if kinds.get(type(a)) is _ATOM else _copy(a, memo, alive))
    elif kind is _OWN:
        y = x.__deepcopy__(memo)
    elif kind is _MOBJECT:
        # An instance's own ``__deepcopy__`` would be the one copy.deepcopy
        # calls.
        if "__deepcopy__" in x.__dict__:
            return copy.deepcopy(x, memo)
        y = _copy_mobject(x, memo, alive)
    elif kind is _DICT:
        y = {}
        memo[d] = y
        for key, value in x.items():
            if kinds.get(type(key)) is not _ATOM:
                key = _copy(key, memo, alive)
            if kinds.get(type(value)) is not _ATOM:
                value = _copy(value, memo, alive)
            y[key] = value
    elif kind is _TUPLE:
        items = [a if kinds.get(type(a)) is _ATOM else _copy(a, memo, alive) for a in x]
        try:
            return memo[d]
        except KeyError:
            pass
        for k, j in zip(x, items, strict=True):
            if k is not j:
                y = tuple(items)
                break
        else:
            y = x
    else:
        return copy.deepcopy(x, memo)
    if y is not x:
        memo[d] = y
        alive.append(x)
    return y


def _copy_mobject(mob: _M, memo: dict[int, Any], alive: list[Any]) -> _M:
    """What :meth:`.Mobject.__deepcopy__` does, each value copied by
    :func:`_copy`.
    """
    cls = mob.__class__
    result = cls.__new__(cls)
    memo[id(mob)] = result
    kinds = _KINDS
    for k, v in mob.__dict__.items():
        if kinds.get(type(v)) is not _ATOM:
            v = _copy(v, memo, alive)
        setattr(result, k, v)
    result.original_id = str(id(mob))
    return result


def copy_mobject(mob: _M, memo: dict[int, Any]) -> _M:
    """``mob``'s deep copy with ``memo``, as :meth:`.Mobject.__deepcopy__`
    makes it.
    """
    alive = memo.get(id(memo))
    if alive is None:
        alive = memo[id(memo)] = []
    return _copy_mobject(mob, memo, alive)
