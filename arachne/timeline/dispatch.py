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
never on a class standing for them. Drawing is held to the same rule
(:func:`draw_refusal`): a mobject that runs the scene's code where the
camera calls the engine's makes its record refuse record-based frames.
"""

from __future__ import annotations

__all__ = [
    "dispatch_refusal",
    "draw_methods",
    "draw_refusal",
    "engine_patches",
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
    an attribute its class defines (``d.interpolate = f``). A function the
    engine itself put there is the engine's code (``ThreeDAxes`` gives its
    axes' pieces their own gradient points).
    """
    from arachne.mobject.mobject import Mobject

    cls = type(obj)
    return tuple(
        key
        for key, value in getattr(obj, "__dict__", {}).items()
        if callable(value)
        and not isinstance(value, (type, Mobject))
        and not _engine_function(value)
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


# Drawing


#: Names in the camera's draw code that hold no mobject: the camera itself,
#: the Cairo context and the modules it computes with.
_NOT_MOBJECTS = frozenset(
    {"self", "ctx", "np", "cairo", "it", "math", "config", "logger", "Image"}
)


def draw_refusal(camera: Any, mobjects: Iterable[Any]) -> str | None:
    """Why frames of ``mobjects`` drawn through ``camera`` (and the cameras
    of its displays) can't come from a record: a member of their families
    whose class overrides, or whose instance replaces, a method the draw
    calls (:func:`draw_methods`). Drawing it runs the scene's code, which
    may answer differently once the scene has run, so its frames are drawn
    while the scene runs, as CE draws them.
    """
    names: set[str] = set(draw_methods(type(camera)))  # type: ignore[arg-type]
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        sub_camera = getattr(display, "camera", None)
        if sub_camera is not None:
            names |= draw_methods(type(sub_camera))  # type: ignore[arg-type]
    classes: dict[type, list[str]] = {}
    for mobject in mobjects:
        for member in mobject.get_family():
            cls = type(member)
            if cls not in classes:
                classes[cls] = [n for n in user_overrides(cls) if n in names]
            replaced = classes[cls] or [
                n for n in instance_shadows(member) if n in names
            ]
            if replaced:
                return (
                    f"draw.user_override: {cls.__name__}.{replaced[0]}: the camera"
                    " calls the scene's code while it draws, so its frames are drawn"
                    " while the scene runs and its record answers no instant"
                )
    return None


@cache
def draw_methods(camera_class: type) -> frozenset[str]:
    """The mobject attributes a draw through ``camera_class`` may reach, read
    from the engine's code rather than listed: every attribute the camera's
    methods (and the engine functions they call by name) read off an object
    other than the camera, closed over what the engine's mobject methods of
    those names read off ``self``. A read off a local that is no mobject (a
    set's ``add``) is kept: it only refuses more.
    """
    defined = _mobject_definitions()
    seed: set[str] = set()
    for klass in camera_class.__mro__:
        if not _engine_class(klass):
            continue
        for value in vars(klass).values():
            function = _function_of(value)
            if function is None:
                continue
            others, _, called = _reads(function)
            seed |= others
            for name in called:
                helper = function.__globals__.get(name)
                if _engine_function(helper):
                    seed |= _reads(helper)[0]
    found: set[str] = set()
    todo = [name for name in seed if name in defined]
    while todo:
        name = todo.pop()
        if name in found:
            continue
        found.add(name)
        for function in defined[name]:
            todo.extend(n for n in _reads(function)[1] if n in defined)
    return frozenset(found)


def _function_of(value: Any) -> Any:
    import types

    if isinstance(value, property):
        value = value.fget
    elif isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    return value if isinstance(value, types.FunctionType) else None


def _engine_function(value: Any) -> bool:
    import types

    module = getattr(value, "__module__", None) or ""
    return isinstance(value, types.FunctionType) and (
        module == "arachne" or module.startswith("arachne.")
    )


@cache
def _mobject_definitions() -> dict[str, tuple[Any, ...]]:
    """Each name an engine mobject class defines, with the functions it
    names there (a property's getter included).
    """
    from arachne.mobject.mobject import Mobject

    found: dict[str, list[Any]] = {}
    for cls in _all_engine_classes():
        if issubclass(cls, Mobject):
            for key, value in vars(cls).items():
                function = _function_of(value)
                found.setdefault(key, [])
                if function is not None:
                    found[key].append(function)
    return {key: tuple(functions) for key, functions in found.items()}


@cache
def _reads(function: Any) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """What ``function``'s source reads: attributes off names other than the
    camera and its tools, attributes off ``self``, and the names it calls.
    """
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return frozenset(), frozenset(), frozenset()
    others: set[str] = set()
    own: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                own.add(node.attr)
            elif node.value.id not in _NOT_MOBJECTS:
                others.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    return frozenset(others), frozenset(own), frozenset(called)
