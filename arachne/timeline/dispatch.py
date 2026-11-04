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

from .frozen import class_members, engine_class, engine_code

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
    return cls is object or engine_class(cls)


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
    an attribute its class defines (``d.interpolate = f``). A function of
    the engine's own code (:func:`.engine_code`) is the engine's, as
    ``ThreeDAxes`` gives its axes' pieces their own gradient points, so long
    as it closes over nothing that is not; ``functools.wraps`` makes no
    function the engine's.
    """
    from arachne.mobject.mobject import Mobject

    cls = type(obj)
    return tuple(
        key
        for key, value in getattr(obj, "__dict__", {}).items()
        if callable(value)
        and not isinstance(value, (type, Mobject))
        and not _engine_closure(value)
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


def engine_patches() -> tuple[str, ...]:
    """The engine classes' attributes a scene has replaced (``Dot.interpolate
    = f``, ``VMobject.set_fill = g``): code, or another callable, other than
    the object the class held when the engine was imported
    (:func:`.class_members`), or one put on a class over a name it inherits.
    Every mobject of that class then runs it where the engine's would run,
    so no play is proven while one stands. Compared by identity: no
    ``functools.wraps`` makes a replacement the engine's.
    """
    found: list[str] = []
    for cls in _all_engine_classes():
        taken = class_members(cls)
        for key, value in vars(cls).items():
            if not _code(value):
                continue
            before = taken.get(key, _MISSING)
            if before is value:
                continue
            if before is _MISSING and not any(key in vars(k) for k in cls.__mro__[1:]):
                # A name of its own, which none of the engine's code calls.
                continue
            found.append(f"{cls.__name__}.{key}")
    return tuple(found)


def _code(value: Any) -> bool:
    """Whether a class member runs as code when read or called: a function,
    a descriptor of one, or another callable (a class is a value).
    """
    if isinstance(value, (property, staticmethod, classmethod)):
        return True
    return callable(value) and not isinstance(value, type)


@cache
def _all_engine_classes() -> tuple[type, ...]:
    """Every engine class a mobject, animation, camera or scene is made of,
    the bases the engine does not export included.
    """
    from .setters import engine_classes

    found: dict[type, None] = {}
    for cls in engine_classes():
        for klass in cls.__mro__:
            if engine_class(klass):
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
    of its displays) can't come from a record: the first member of their
    families that :func:`draw_member_refusal` refuses.
    """
    names = draw_names(camera)
    classes: dict[type, str | None] = {}
    for mobject in mobjects:
        for member in mobject.get_family():
            reason = draw_member_refusal(names, member, classes)
            if reason is not None:
                return reason
    return None


def draw_names(camera: Any) -> tuple[frozenset[str], frozenset[str]]:
    """The methods and the protocol methods a draw through ``camera`` and
    its displays' cameras calls on a mobject.
    """
    cameras = [camera]
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        sub_camera = getattr(display, "camera", None)
        if sub_camera is not None:
            cameras.append(sub_camera)
    methods: set[str] = set()
    protocols: set[str] = set()
    for each in cameras:
        methods |= draw_methods(type(each))  # type: ignore[arg-type]
        protocols |= draw_protocols(type(each))  # type: ignore[arg-type]
    return frozenset(methods), frozenset(protocols)


def draw_member_refusal(
    names: tuple[frozenset[str], frozenset[str]],
    member: Any,
    classes: dict[type, str | None] | None = None,
) -> str | None:
    """Why drawing ``member`` runs the scene's code, or ``None``: its class
    overrides a draw method or defines a protocol method the draw invokes
    (``__hash__``, ``__iter__``), or its instance replaces a draw method.
    Drawing it may answer differently once the scene has run, so its frames
    are drawn while the scene runs, as CE draws them.
    """
    methods, protocols = names
    cls = type(member)
    if classes is None or cls not in classes:
        found = [n for n in user_overrides(cls) if n in methods] + [
            n for n in user_definitions(cls) if n in protocols
        ]
        reason = found[0] if found else None
        if classes is not None:
            classes[cls] = reason
    else:
        reason = classes[cls]
    if reason is None:
        shadows = [n for n in instance_shadows(member) if n in methods]
        reason = shadows[0] if shadows else None
    if reason is None:
        return None
    return (
        f"draw.user_override: {cls.__name__}.{reason}: the camera calls the"
        " scene's code while it draws, so its frames are drawn while the scene"
        " runs and its record answers no instant"
    )


def user_definitions(cls: type) -> tuple[str, ...]:
    """Every name a class defined outside the engine defines in its body,
    in the classes of its MRO the engine did not define: what Python looks
    up on the type for a protocol (``__hash__``, ``__bool__``) included,
    whether or not the engine defines it.
    """
    found: list[str] = []
    for klass in cls.__mro__:
        if _engine_class(klass):
            continue
        for key in vars(klass):
            if key not in _OWN_DEFINITIONS and key not in found:
                found.append(key)
    return tuple(found)


def _extraction_functions() -> list[Any]:
    """The functions of the Frame IR's extraction, which reads what the
    camera's capture reads (and must call no scene code either): its
    module's functions and its classes' methods.
    """
    from arachne.frame_ir import extract

    found: list[Any] = []
    for value in vars(extract).values():
        if (
            _engine_function(value)
            and getattr(value, "__module__", None) == extract.__name__
        ):
            found.append(value)
        elif isinstance(value, type) and value.__module__ == extract.__name__:
            found.extend(
                f for f in map(_function_of, vars(value).values()) if f is not None
            )
    return found


@cache
def draw_methods(camera_class: type) -> frozenset[str]:
    """The mobject attributes a draw through ``camera_class`` may reach, read
    from the engine's code rather than listed: every attribute the camera's
    methods (and the engine functions they call by name) read off an object
    other than the camera, closed over what the engine's mobject methods of
    those names read off ``self``. A read off a local that is no mobject (a
    set's ``add``) is kept: it only refuses more. The Frame IR's extraction
    reads a frame as the camera would, and what it reads joins the set.
    """
    defined = _mobject_definitions()
    seed: set[str] = set()
    for function in _extraction_functions():
        seed |= _reads(function)[0]
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
    return engine_code(value)


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


#: Protocol methods a format or a message calls, which change no frame.
_NOT_DRAWN = frozenset({"__repr__", "__str__", "__format__", "__init_subclass__"})


@cache
def draw_protocols(camera_class: type) -> frozenset[str]:
    """The protocol methods a draw through ``camera_class`` may invoke on a
    mobject, read from the engine's code: what each construct of the draw
    path (the camera's methods, the engine functions they call, the
    mobject methods :func:`draw_methods` reaches, and the record's own
    drawing in :mod:`.draw` and :mod:`.record`) invokes implicitly: a
    comparison ``__eq__``, a loop ``__iter__``, a key ``__hash__``, a test
    ``__bool__``, an attribute read ``__getattribute__``. ``__repr__``,
    ``__str__``, ``__format__`` and ``__init_subclass__`` are left out.
    """
    from . import draw, record

    functions: list[Any] = []
    for klass in camera_class.__mro__:
        if _engine_class(klass) and klass is not object:
            for value in vars(klass).values():
                function = _function_of(value)
                if function is not None:
                    functions.append(function)
                    for name in _reads(function)[2]:
                        helper = function.__globals__.get(name)
                        if _engine_function(helper):
                            functions.append(helper)
    functions.extend(_extraction_functions())
    defined = _mobject_definitions()
    for name in draw_methods(camera_class):
        functions.extend(defined.get(name, ()))
    for module in (draw, record):
        for value in vars(module).values():
            if _engine_function(value):
                functions.append(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                functions.extend(
                    f for f in map(_function_of, vars(value).values()) if f is not None
                )
    found: set[str] = set()
    for function in functions:
        found |= _protocols(function)
    return frozenset(found - _NOT_DRAWN)


#: What a comparison invokes, by the name of its ``ast`` operator.
_MEMBERSHIP = ("__contains__", "__iter__", "__getitem__", "__eq__", "__hash__")
_COMPARISONS = {
    "Eq": ("__eq__",),
    "NotEq": ("__ne__", "__eq__"),
    "Lt": ("__lt__", "__gt__"),
    "LtE": ("__le__", "__ge__"),
    "Gt": ("__gt__", "__lt__"),
    "GtE": ("__ge__", "__le__"),
    "In": _MEMBERSHIP,
    "NotIn": _MEMBERSHIP,  # codespell:ignore
}
_OPERATORS = {
    "Add": "add",
    "Sub": "sub",
    "Mult": "mul",
    "MatMult": "matmul",
    "Div": "truediv",
    "FloorDiv": "floordiv",
    "Mod": "mod",
    "Pow": "pow",
    "LShift": "lshift",
    "RShift": "rshift",
    "BitOr": "or",
    "BitXor": "xor",
    "BitAnd": "and",
}
_UNARY = {"USub": "__neg__", "UAdd": "__pos__", "Invert": "__invert__"}
_ITERATION = ("__iter__", "__next__", "__getitem__", "__len__", "__length_hint__")
_HASHING = ("__hash__", "__eq__")
_TRUTH = ("__bool__", "__len__")
#: Builtins by what they invoke on what they are given.
_BUILTIN_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "len": ("__len__",),
    "bool": _TRUTH,
    "hash": _HASHING,
    "iter": _ITERATION,
    "next": ("__next__",),
    "reversed": ("__reversed__", "__len__", "__getitem__"),
    "abs": ("__abs__",),
    "int": ("__int__", "__index__", "__trunc__"),
    "float": ("__float__", "__index__"),
    "complex": ("__complex__", "__float__", "__index__"),
    "round": ("__round__",),
    "getattr": ("__getattribute__", "__getattr__"),
    "hasattr": ("__getattribute__", "__getattr__"),
    "setattr": ("__setattr__",),
    "delattr": ("__delattr__",),
    "divmod": ("__divmod__", "__rdivmod__"),
    "pow": ("__pow__", "__rpow__"),
    "copy": ("__copy__", "__reduce_ex__", "__reduce__", "__getstate__", "__setstate__"),
    "deepcopy": (
        "__deepcopy__",
        "__reduce_ex__",
        "__reduce__",
        "__getstate__",
        "__setstate__",
    ),
}
for _name in (
    "list",
    "tuple",
    "sum",
    "any",
    "all",
    "zip",
    "enumerate",
    "map",
    "filter",
):
    _BUILTIN_PROTOCOLS[_name] = _ITERATION
for _name in ("set", "frozenset", "dict"):
    _BUILTIN_PROTOCOLS[_name] = (*_ITERATION, *_HASHING)
for _name in ("sorted", "min", "max"):
    _BUILTIN_PROTOCOLS[_name] = (*_ITERATION, "__lt__", "__gt__")
#: Container methods that hash or compare what they are given.
_KEYED_METHODS = frozenset(
    {"add", "get", "setdefault", "pop", "discard", "remove", "index", "count", "update"}
)


@cache
def _protocols(function: Any) -> frozenset[str]:
    """The protocol methods ``function``'s constructs may invoke."""
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return frozenset()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                found.update(_COMPARISONS.get(type(op).__name__, ()))
        elif isinstance(node, ast.BinOp | ast.AugAssign):
            stem = _OPERATORS[type(node.op).__name__]
            found.update((f"__{stem}__", f"__r{stem}__", f"__i{stem}__"))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                found.update(_TRUTH)
            else:
                found.add(_UNARY[type(node.op).__name__])
        elif isinstance(node, ast.For | ast.comprehension | ast.Starred):
            found.update(_ITERATION)
        elif isinstance(node, ast.If | ast.While | ast.IfExp | ast.Assert | ast.BoolOp):
            found.update(_TRUTH)
        elif isinstance(node, ast.Set | ast.Dict | ast.SetComp | ast.DictComp):
            found.update(_HASHING)
        elif isinstance(node, ast.Subscript):
            found.update(("__getitem__", "__setitem__", "__delitem__", "__index__"))
            found.update(_HASHING)
        elif isinstance(node, ast.Attribute):
            found.update(("__getattribute__", "__getattr__"))
            if isinstance(node.ctx, ast.Store):
                found.add("__setattr__")
            elif isinstance(node.ctx, ast.Del):
                found.add("__delattr__")
        elif isinstance(node, ast.With):
            found.update(("__enter__", "__exit__"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                found.update(_BUILTIN_PROTOCOLS.get(func.id, ()))
            elif isinstance(func, ast.Attribute):
                found.update(_BUILTIN_PROTOCOLS.get(func.attr, ()))
                if func.attr in _KEYED_METHODS:
                    found.update(_HASHING)
            # What a call reaches may be an object with its own `__call__`.
            found.add("__call__")
    return frozenset(found)


def _engine_closure(value: Any) -> bool:
    """Whether ``value`` is the engine's code closing over nothing but values
    and the engine's code (and mobjects, which it reads as data).
    """
    import types

    if not engine_code(value):
        return False
    for cell in value.__closure__ or ():
        try:
            held = cell.cell_contents
        except ValueError:
            continue
        if isinstance(held, types.FunctionType | types.MethodType) and not (
            engine_code(getattr(held, "__func__", held))
        ):
            return False
    return True
