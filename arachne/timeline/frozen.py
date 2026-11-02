"""The identities admission compares against, taken once, when ``arachne`` is
imported and before any scene module runs.

Admission vouches for the engine's and the libraries' code by identity. If
it read those identities when first needed, a scene that rebinds a name
first (``np.copy = f``) would have its own callable read into a table. So
:func:`freeze` builds every table at import time (the library table, the
constructors, setters and reads, the draw methods, the classes the engine
defines, what each of those classes holds) and the set of the engine's code
objects. From then on, a name a scene rebinds is only an unlisted callable,
and a class member a scene replaces is not the one taken.

An engine function is one whose code object the engine compiled: the code
of every function its modules and classes define from its own files, every
code nested in those (``plot``'s wrapper, the gradient points
``ThreeDAxes`` gives its pieces), and what ``dataclass`` generates for its
classes. Neither ``__module__``, ``__qualname__`` nor ``functools.wraps``
changes a function's code, so none of them makes a scene's function the
engine's. Code another library shares among every function it wraps
(``contextlib``'s, ``reprlib``'s) is not the engine's.

The set compares code objects by equality, which is structural, not by
pointer: a copy of the engine's code (``f.__code__.replace()``) put in a
scene's function passes. That is forging the engine's identity, which the
proof does not cover (it covers scene code that uses the engine's and the
libraries' APIs as documented, without patching them).
"""

from __future__ import annotations

__all__ = ["class_members", "engine_class", "engine_code", "freeze"]

import functools
import os
import sys
import types
from collections.abc import Mapping
from typing import Any

_CODES: set[types.CodeType] = set()
_CLASSES: dict[type, Mapping[str, Any]] = {}
_FROZEN = False
#: Where the engine's files are.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def engine_code(function: Any) -> bool:
    """Whether ``function`` is a Python function of the engine's own code
    (its code object equal to one the engine compiled).
    """
    if not _FROZEN:
        freeze()
    return isinstance(function, types.FunctionType) and function.__code__ in _CODES


def engine_class(cls: Any) -> bool:
    """Whether ``cls`` is a class the engine defined by the time it was
    imported.
    """
    if not _FROZEN:
        freeze()
    try:
        return cls in _CLASSES
    except TypeError:  # unhashable
        return False


def class_members(cls: type) -> Mapping[str, Any]:
    """What the engine class ``cls`` held when the engine was imported."""
    if not _FROZEN:
        freeze()
    return _CLASSES[cls]


def freeze() -> None:
    """Take the engine's identities and build every admission table."""
    global _FROZEN
    if _FROZEN:
        return
    _FROZEN = True
    import importlib

    for name in ("arachne.timeline.trusted", "arachne.utils.family_ops"):
        importlib.import_module(name)
    seen: set[int] = set()
    for name, module in list(sys.modules.items()):
        if name == "arachne" or name.startswith("arachne."):
            for value in list(vars(module).values()):
                _take(value, name, seen)
    # The engine's classes its modules do not name: bases and subclasses.
    stack = list(_CLASSES)
    while stack:
        cls = stack.pop()
        for klass in (*cls.__mro__, *cls.__subclasses__()):
            if klass not in _CLASSES and klass.__module__.startswith("arachne"):
                _take(klass, klass.__module__, seen)
                stack.append(klass)
    _build_tables()


def _take(value: Any, module: str, seen: set[int]) -> None:
    if id(value) in seen:
        return
    if isinstance(value, type):
        if value.__module__ != module:
            return
        seen.add(id(value))
        _CLASSES[value] = types.MappingProxyType(dict(vars(value)))
        for item in list(vars(value).values()):
            _take_member(item, seen)
    elif isinstance(value, types.FunctionType):
        if value.__module__ == module:
            seen.add(id(value))
            _take_function(value)
    elif isinstance(value, functools._lru_cache_wrapper):
        _take(value.__wrapped__, module, seen)


def _take_member(item: Any, seen: set[int]) -> None:
    if isinstance(item, property):
        for accessor in (item.fget, item.fset, item.fdel):
            if accessor is not None:
                _take_member(accessor, seen)
    elif isinstance(item, (staticmethod, classmethod)):
        _take_member(item.__func__, seen)
    elif isinstance(item, functools._lru_cache_wrapper):
        _take_member(item.__wrapped__, seen)
    elif isinstance(item, types.FunctionType):
        _take_function(item)
    elif isinstance(item, type) and item.__module__.startswith("arachne"):
        _take(item, item.__module__, seen)


def _take_function(function: types.FunctionType) -> None:
    _take_code(function.__code__)
    wrapped = getattr(function, "__wrapped__", None)
    if isinstance(wrapped, types.FunctionType):
        _take_function(wrapped)


def _take_code(code: types.CodeType) -> None:
    if code in _CODES:
        return
    filename = code.co_filename
    # The engine's files, or code generated for one of its classes
    # (``dataclass``'s ``<string>``); not another library's.
    if not (filename.startswith(_ROOT) or filename.startswith("<")):
        return
    _CODES.add(code)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _take_code(const)


def _build_tables() -> None:
    from arachne.camera.camera import Camera

    from . import dispatch, library, setters

    library.library()
    setters.freeze_tables()
    for cls in list(_CLASSES):
        if issubclass(cls, Camera):
            dispatch.draw_methods(cls)  # type: ignore[arg-type]
            dispatch.draw_protocols(cls)  # type: ignore[arg-type]
    dispatch.engine_patches()
