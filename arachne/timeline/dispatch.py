"""Whether the engine's own code runs when a mobject's or an animation's
methods are called.

Admission vouches for what the engine's methods do: the trusted algebra
(route 1) for ``interpolate``, ``pointwise_become_partial`` and the rest an
animation calls on its mobjects, the setter table and the reads (route 2)
for what a callback calls and what a group's setter calls on each child. An
object runs other code under those names in two ways, and either takes it
out of both routes:

- its class is defined outside the engine and overrides an engine name
  (:func:`user_overrides`);
- it holds a callable on the instance under a name its class defines
  (:func:`instance_shadows`): ``d.interpolate = f`` replaces the method for
  that object alone.

Dispatch is checked on the objects themselves, every member of a family,
never on a class standing for them.
"""

from __future__ import annotations

__all__ = [
    "dispatch_refusal",
    "family_refusal",
    "instance_shadows",
    "user_overrides",
]

import inspect
from collections.abc import Iterable
from functools import cache
from typing import Any

#: What a scene's subclass of an engine class may define without taking its
#: instances out of the algebra.
_OWN_DEFINITIONS = frozenset(
    {
        "__init__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__firstlineno__",
        "__static_attributes__",
        "__orig_bases__",
        "__parameters__",
    }
)
#: Hooks that change every attribute access, defined or not by the engine.
_ACCESS_HOOKS = frozenset(
    {"__getattr__", "__getattribute__", "__setattr__", "__delattr__"}
)
_MISSING = object()


def _engine_class(cls: type) -> bool:
    return cls.__module__ == "arachne" or cls.__module__.startswith("arachne.")


def user_overrides(cls: type) -> tuple[str, ...]:
    """The engine methods (and properties) a class defined outside the
    engine overrides: an animation calls them on the mobject (``interpolate``,
    ``pointwise_become_partial``, ``align_points`` and the rest), so an
    override runs scene code inside an evaluation. Empty for the engine's own
    classes, and for a subclass that only defines ``__init__``, class values
    or methods of its own. Read afresh at each call: a class may be changed
    between plays.
    """
    if all(klass is object or _engine_class(klass) for klass in cls.__mro__):
        return ()
    engine: set[str] = set()
    for klass in cls.__mro__:
        if klass is object or _engine_class(klass):
            engine.update(vars(klass))
    found: list[str] = []
    for klass in cls.__mro__:
        if klass is object or _engine_class(klass):
            continue
        for key, value in vars(klass).items():
            if key in _OWN_DEFINITIONS:
                continue
            code = callable(value) or isinstance(
                value, (property, staticmethod, classmethod)
            )
            if code and (key in engine or key in _ACCESS_HOOKS) and key not in found:
                found.append(key)
    return tuple(found)


def instance_shadows(obj: Any) -> tuple[str, ...]:
    """The names under which ``obj`` holds a callable of its own that hides
    an attribute its class defines (``d.interpolate = f``).
    """
    from arachne.mobject.mobject import Mobject

    cls = type(obj)
    return tuple(
        key
        for key, value in getattr(obj, "__dict__", {}).items()
        if callable(value)
        and not isinstance(value, (type, Mobject))
        and inspect.getattr_static(cls, key, _MISSING) is not _MISSING
    )


def dispatch_refusal(obj: Any) -> str | None:
    """Why calling the engine's methods on ``obj`` may run other code, or
    ``None``.
    """
    name = type(obj).__name__
    overrides = user_overrides(type(obj))
    if overrides:
        return f"a {name}, whose class overrides the engine's {', '.join(overrides)}"
    shadows = instance_shadows(obj)
    if shadows:
        return f"a {name} whose {', '.join(shadows)} is replaced on the instance"
    return None


def family_refusal(mobjects: Iterable[Any]) -> str | None:
    """:func:`dispatch_refusal` of the first member of the mobjects'
    families that has one.
    """
    classes: dict[type, tuple[str, ...]] = {}
    for mobject in mobjects:
        for member in mobject.get_family():
            cls = type(member)
            if cls not in classes:
                classes[cls] = user_overrides(cls)
            if classes[cls]:
                return (
                    f"a {cls.__name__}, whose class overrides the engine's"
                    f" {', '.join(classes[cls])}"
                )
            shadows = instance_shadows(member)
            if shadows:
                return (
                    f"a {cls.__name__} whose {', '.join(shadows)} is replaced on"
                    " the instance"
                )
    return None


#: Modules besides the engine's whose functions engine classes hold: what
#: ``dataclass`` and ``Enum`` generate.
_GENERATED_BY = frozenset({"dataclasses", "enum"})


def engine_patches() -> tuple[str, ...]:
    """The engine classes' attributes a scene has replaced (``Dot.interpolate
    = f``, ``VMobject.set_fill = g``): a function, or another callable, that
    is not the engine's own. Every mobject of that class then runs it where
    the engine's would run, so no play is proven while one stands.
    """
    found: list[str] = []
    for cls in _all_engine_classes():
        for key, value in vars(cls).items():
            if not _engine_made(value):
                found.append(f"{cls.__name__}.{key}")
    return tuple(found)


def _engine_made(value: Any, *, accessor: bool = False) -> bool:
    import functools
    import operator
    import types

    if isinstance(value, property):
        return all(
            f is None or _engine_made(f, accessor=True)
            for f in (value.fget, value.fset, value.fdel)
        )
    if isinstance(value, (staticmethod, classmethod)):
        return _engine_made(value.__func__)
    if isinstance(value, types.FunctionType):
        module = value.__module__ or ""
        return (
            module in _GENERATED_BY
            or module == "arachne"
            or module.startswith("arachne.")
        )
    if isinstance(value, functools._lru_cache_wrapper):
        return _engine_made(value.__wrapped__)
    if isinstance(value, operator.attrgetter):
        return accessor
    if isinstance(value, types.BuiltinFunctionType):
        # ``object.__new__`` and its kin, bound to a class.
        return isinstance(value.__self__, type)
    if isinstance(value, (types.WrapperDescriptorType, types.MethodDescriptorType)):
        return True
    return not callable(value) or isinstance(value, type)


@cache
def _all_engine_classes() -> tuple[type, ...]:
    """Every engine class a mobject, animation, camera or scene is made of,
    the bases the engine does not export included.
    """
    from .setters import engine_classes

    found: dict[type, None] = {}
    for cls in engine_classes():
        for klass in cls.__mro__:
            if _engine_class(klass):
                found[klass] = None
    return tuple(found)
    return None
