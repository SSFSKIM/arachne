"""A mobject's deep copy, made without :func:`copy.deepcopy`'s general machinery
where that is safe (``fast_copy``).

:meth:`.Mobject.copy` is :func:`copy.deepcopy`, and :mod:`copy` dispatches
every value it meets through a Python call: a mobject's dozens of numbers and
flags, and each point of the SVG path a glyph keeps. This module copies the
common values inline and hands every other value to :func:`copy.deepcopy`
itself.

**The domain.** A value is copied here only if it is one of:

- an *atomic* value: one whose exact type :mod:`copy` returns as it is
  (numbers, strings, ``None``, functions, ``type``…), with no memo entry for it;
- an exact ``list``, ``dict`` or ``tuple``, while :mod:`copy`'s dispatch table
  still sends that type to its own copier;
- an instance of a *checked* class whose instance ``__dict__`` holds none of
  the copy hooks (``__deepcopy__``, ``__setstate__``, ``__reduce_ex__``,
  ``__reduce__``, ``__getstate__``). A checked class has ``object``'s
  ``__getattribute__``, no entry in :mod:`copy`'s dispatch table, and is one
  of:

  - a *mobject* class, whose ``__deepcopy__`` is :meth:`.Mobject.__deepcopy__`;
  - an *own-copy* class, with any other ``__deepcopy__`` (a read-only array, a
    :class:`~.Paint`, an enum);
  - a *plain* class, which :mod:`copy` rebuilds through ``object``'s reduce: no
    ``__deepcopy__``, no :data:`copyreg.dispatch_table` entry, ``object``'s
    ``__reduce_ex__``, ``__reduce__``, ``__getstate__`` and ``__new__``, no
    ``__setstate__``, ``__getnewargs_ex__``, ``__getnewargs__`` or
    ``__getattr__``, an instance ``__dict__``, no slots, no class among its
    bases but ``object`` that is not defined in Python or that sets
    ``__class__``, and :func:`copyreg.__newobj__` as :mod:`copyreg` defines
    it; and whose class the memo holds no entry for.

Every other value goes to :func:`copy.deepcopy` with the same memo, and so
does an atomic value the memo holds.

**Why the copy is copy.deepcopy's.** Inside the domain, each case runs the
steps :func:`copy.deepcopy` runs for that value, in its order: the memo is
read first and a hit returned (but for an atomic value's, which 3.14 skips:
:func:`copy.deepcopy` answers that one as the running Python does); an atomic
value is returned as it is; a list, dict or tuple is copied as ``copy``'s own
copier does, a dict's value before its key; a mobject's or own-copy class's
``__deepcopy__`` is the one ``getattr(x, "__deepcopy__")`` finds, because
neither the instance nor a custom attribute lookup can supply another; a
plain instance is rebuilt as ``copy._reconstruct`` rebuilds ``object``'s
reduction of it: ``cls.__new__(cls)`` (its one argument, the class, copies to
itself, the memo holding nothing for it), then a deep copy of its
``__dict__`` through the same dict copy, applied with ``update`` because
neither its class nor its ``__new__`` gives the new instance a
``__setstate__``. Every copy that is not its original is entered in the memo
and kept alive in ``memo[id(memo)]``, created when first needed, as
``copy._keep_alive`` does.

**Classes are checked again in every copy.** A class's kind is found once and
kept by the class's identity with the hooks it was found from; each copy
reads those hooks again the first time it meets the class and finds the kind
again if one changed, so a scene that gives its own class new hooks between
two copies has the second follow them. A hook changed by code running inside
a copy (another object's ``__deepcopy__``) is followed from the next copy on.
"""

from __future__ import annotations

import copy
import copyreg
import weakref
from collections import Counter
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .mobject import Mobject

_M = TypeVar("_M", bound="Mobject")

__all__ = ["FALLBACKS", "copy_mobject"]

_nil: list[Any] = []
_deepcopy = copy.deepcopy

# What copy.deepcopy reads from copy and copyreg: private, so typeshed doesn't
# declare them.
_DISPATCH: dict[type, Any] = copy._deepcopy_dispatch  # type: ignore[attr-defined]
_COPY_LIST = _DISPATCH[list]
_COPY_DICT = _DISPATCH[dict]
_COPY_TUPLE = _DISPATCH[tuple]
_slotnames: Any = copyreg._slotnames  # type: ignore[attr-defined]
_NEWOBJ = copyreg.__newobj__  # type: ignore[attr-defined]

# The types copy.deepcopy returns as they are, with no memo entry: its atomic
# set where it has one (3.14), else the types its dispatch table sends to
# ``_deepcopy_atomic``. Read once; a value of one is also read from the memo
# first, so a caller's memo entry for it is answered by copy.deepcopy itself.
if hasattr(copy, "_atomic_types"):
    _ATOMIC: frozenset[type] = frozenset(copy._atomic_types)
else:
    _ATOMIC = frozenset(
        cls
        for cls, copier in _DISPATCH.items()
        if copier is copy._deepcopy_atomic  # type: ignore[attr-defined]
    )

#: How many values each copy handed to copy.deepcopy because they were
#: outside the domain, by type name: a measure, never read by the copier.
FALLBACKS: Counter[str] = Counter()

_MOBJECT, _OWN, _PLAIN, _GENERAL = range(4)
_HEAPTYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class defined in Python
#: The copy hooks an instance's own ``__dict__`` can supply in place of its
#: class's.
_SHADOWING = frozenset(
    ("__deepcopy__", "__setstate__", "__reduce_ex__", "__reduce__", "__getstate__")
)
_OBJECT_GETATTRIBUTE = object.__getattribute__

#: Each class's kind, by the class's id: (a weak reference to the class, its
#: kind, the hooks the kind was found from).
_DECIDED: dict[int, tuple[Any, int, tuple[Any, ...]]] = {}
#: The kinds this copy has checked, by the class's id; emptied as each copy
#: begins.
_CHECKED: dict[int, int] = {}


def _hooks(cls: type, kind: int) -> tuple[Any, ...]:
    """What a ``kind`` decision about ``cls`` read: the hooks it resolved."""
    common = (
        _DISPATCH.get(cls),
        getattr(cls, "__deepcopy__", None),
        cls.__getattribute__,
    )
    if kind != _PLAIN:
        return common
    return (
        *common,
        cls.__reduce_ex__,
        cls.__reduce__,
        getattr(cls, "__getstate__", None),
        getattr(cls, "__setstate__", None),
        getattr(cls, "__getnewargs_ex__", None),
        getattr(cls, "__getnewargs__", None),
        getattr(cls, "__getattr__", None),
        cls.__new__,
        copyreg.dispatch_table.get(cls),
        copyreg.__newobj__,  # type: ignore[attr-defined]
        cls.__mro__,
    )


def _classify(cls: type) -> int:
    from .mobject import Mobject

    if (
        cls in _DISPATCH
        or issubclass(cls, type)
        or cls.__getattribute__ is not _OBJECT_GETATTRIBUTE
    ):
        return _GENERAL
    own = getattr(cls, "__deepcopy__", None)
    if own is Mobject.__deepcopy__:
        return _MOBJECT
    if own is not None:
        return _OWN
    if (
        cls not in copyreg.dispatch_table
        and cls.__reduce_ex__ is object.__reduce_ex__
        and cls.__reduce__ is object.__reduce__
        and getattr(cls, "__getstate__", None) is object.__getstate__
        and cls.__new__ is object.__new__
        and copyreg.__newobj__ is _NEWOBJ  # type: ignore[attr-defined]
        and not any(
            hasattr(cls, name)
            for name in (
                "__setstate__",
                "__getnewargs_ex__",
                "__getnewargs__",
                "__getattr__",
            )
        )
        and all(base is object or base.__flags__ & _HEAPTYPE for base in cls.__mro__)
        and cls.__dictoffset__
        and not _slotnames(cls)
        # object's reduce reads the instance's __class__: no class overrides it.
        and all("__class__" not in vars(base) for base in cls.__mro__[:-1])
    ):
        return _PLAIN
    return _GENERAL


def _kind_of(cls: type) -> int:
    """``cls``'s kind, its hooks read again in this copy."""
    key = id(cls)
    decided = _DECIDED.get(key)
    if decided is not None and decided[0]() is cls:
        kind = decided[1]
        if kind == _GENERAL or decided[2] == _hooks(cls, kind):
            _CHECKED[key] = kind
            return kind
    kind = _classify(cls)
    _DECIDED[key] = (
        weakref.ref(cls),
        kind,
        () if kind == _GENERAL else _hooks(cls, kind),
    )
    _CHECKED[key] = kind
    return kind


def _keep_alive(x: Any, memo: dict[int, Any]) -> None:
    try:
        memo[id(memo)].append(x)
    except KeyError:
        memo[id(memo)] = [x]


def _fallback(x: Any, memo: dict[int, Any]) -> Any:
    FALLBACKS[type(x).__name__] += 1
    return _deepcopy(x, memo)


def _copy(x: Any, memo: dict[int, Any]) -> Any:
    """``copy.deepcopy(x, memo)``."""
    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        # copy.deepcopy answers the memo for every value but an atomic one on
        # 3.14, which it returns as it is: it answers this one itself.
        return _deepcopy(x, memo) if type(x) in _ATOMIC else y
    cls: Any = type(x)
    if cls in _ATOMIC:
        return x
    if cls is list or cls is dict or cls is tuple:
        y = _copy_builtin(x, cls, memo)
        if y is x:
            return y
    else:
        kind = _CHECKED.get(id(cls))
        if kind is None:
            kind = _kind_of(cls)
        if kind == _GENERAL:
            return _fallback(x, memo)
        state = getattr(x, "__dict__", None)
        if state is not None and not _SHADOWING.isdisjoint(state):
            return _fallback(x, memo)
        if kind == _MOBJECT:
            y = _copy_mobject(x, memo)
        elif kind == _OWN:
            y = x.__deepcopy__(memo)
        elif id(cls) in memo:
            # copy._reconstruct copies the class, its one argument, and a
            # caller's memo entry for it is answered by copy.deepcopy.
            return _fallback(x, memo)
        else:
            y = cls.__new__(cls)
            memo[d] = y
            if state:
                y.__dict__.update(_copy(state, memo))
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo)
    return y


def _copy_builtin(x: Any, cls: type, memo: dict[int, Any]) -> Any:
    """What copy's own copier of an exact list, dict or tuple does, if copy's
    dispatch table still sends ``cls`` to it. A value outside the memo whose
    type is atomic is its own copy, inline.
    """
    copier = _DISPATCH.get(cls)
    atomic = _ATOMIC
    if cls is list and copier is _COPY_LIST:
        y: Any = []
        memo[id(x)] = y
        append = y.append
        for a in x:
            append(a if id(a) not in memo and type(a) in atomic else _copy(a, memo))
        return y
    if cls is dict and copier is _COPY_DICT:
        y = {}
        memo[id(x)] = y
        for key, value in x.items():
            # The value first, as copy's ``y[deepcopy(key)] = deepcopy(value)``.
            if id(value) in memo or type(value) not in atomic:
                value = _copy(value, memo)
            if id(key) in memo or type(key) not in atomic:
                key = _copy(key, memo)
            y[key] = value
        return y
    if cls is tuple and copier is _COPY_TUPLE:
        items = [
            a if id(a) not in memo and type(a) in atomic else _copy(a, memo) for a in x
        ]
        try:
            return memo[id(x)]
        except KeyError:
            pass
        for k, j in zip(x, items, strict=True):
            if k is not j:
                return tuple(items)
        return x
    return _fallback(x, memo)


def _copy_mobject(mob: _M, memo: dict[int, Any]) -> _M:
    """What :meth:`.Mobject.__deepcopy__` does, each value copied by
    :func:`_copy` (a value outside the memo whose type is atomic inline).
    """
    cls = mob.__class__
    result = cls.__new__(cls)
    memo[id(mob)] = result
    atomic = _ATOMIC
    for k, v in mob.__dict__.items():
        if id(v) in memo or type(v) not in atomic:
            v = _copy(v, memo)
        setattr(result, k, v)
    result.original_id = str(id(mob))
    return result


def copy_mobject(mob: _M, memo: dict[int, Any]) -> _M:
    """``mob``'s deep copy with ``memo``, as :meth:`.Mobject.__deepcopy__`
    makes it.
    """
    _CHECKED.clear()
    return _copy_mobject(mob, memo)
