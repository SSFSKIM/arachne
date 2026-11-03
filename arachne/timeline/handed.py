"""Whether a call may call what it is handed, for an argument the write check
can't resolve (:meth:`.ticking._Site._handed_callables`).

An argument the check resolves to a callable is read as called wherever it
goes. One it can't resolve (read out of a container, handed back by a call,
a loop's variable) may be a function of the scene's that writes anything,
so the call is taken for an unknown one unless its callee is shown never to
call it:

- a function or method whose source the check reads (the scene's, the
  engine's, a library's) calls a parameter when its code calls the
  parameter, or anything it took from it, or hands it to a call that may
  call it (followed as far as :data:`.ticking.MAX_DEPTH`), or keeps it where
  a later call can reach it (an attribute, an item);
- a native callable calls what it is handed when it is listed as one that
  does (a key, a mapped function, a replacement); one listed as never
  calling it (``len``, ``isinstance``, a ufunc, NumPy's makers, a list's
  ``append``) does not; any other native, and a callee the check can't
  resolve, may.

When it can't be told, the argument is called: never the other way.

What may be called is every callable an argument may be or hold. A value
is free of callables only when its kind shows it (a constant, a number, a
string, an array of numbers); a container, a dict unpacked as keywords, what
an operator or a comprehension makes, and any form not read here carry
every callable their parts may hold. A resolved object holds what
:func:`held_callables` reads in it, or anything when that can't be read.
"""

from __future__ import annotations

__all__ = ["CALLED_NATIVE", "NATIVE_UNLISTED", "UNKNOWN_CALLEE", "Callers"]

import ast
import collections
import contextlib
import enum
import functools
import inspect
import re
import types
from collections.abc import Callable
from typing import Any

import numpy as np

from .coverage import FREE_KINDS, NATIVE_HOLDERS, Rule, rule_of

#: Why a callee may call an argument (each a reason
#: :meth:`Callers.calls_argument` gives).
CALLED_NATIVE = "a native that calls what it is handed"
NATIVE_UNLISTED = "a native not listed either way"
UNKNOWN_CALLEE = "a callee the check can't resolve"
CALLS_IT = "its code calls it"
FOLLOWING_STOPS = "its code can't be followed"

#: Builtins that never call what they are handed (an operand's own dunder
#: methods aside, which the check reads apart).
_NEVER_CALL_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "float",
        "format",
        "frozenset",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)
#: Builtins that call a ``key`` they are handed, and only that.
_KEY_TAKERS = frozenset({"sorted", "min", "max"})
#: Builtin classes whose objects call the function they are made with.
_CALLING_BUILTIN_CLASSES = frozenset({"map", "filter"})
#: NumPy's callables that call a function they are handed; NumPy's others
#: never call what they are handed.
_NUMPY_CALLERS = frozenset(
    {
        "apply_along_axis",
        "apply_over_axes",
        "vectorize",
        "frompyfunc",
        "piecewise",
        "fromfunction",
    }
)
#: :mod:`operator`'s callables that call what they are handed (or make a
#: caller of it); its others never do.
_OPERATOR_CALLERS = frozenset({"call", "methodcaller"})
#: Methods of builtin and NumPy objects that call what they are handed;
#: those objects' other methods never do.
_METHOD_CALLERS: dict[type, frozenset[str]] = {
    list: frozenset({"sort"}),
    re.Pattern: frozenset({"sub", "subn"}),
}
_NEVER_CALL_OBJECTS = (
    str,
    bytes,
    tuple,
    frozenset,
    int,
    float,
    complex,
    range,
    list,
    dict,
    set,
    np.ndarray,
    np.generic,
    re.Pattern,
)

#: In a set of called parameters: every one (the code can't be followed).
EVERY = "*"

#: How many objects reading what one value holds visits before giving up.
HELD_LIMIT = 20_000


def held_callables(
    value: Any, limit: int = HELD_LIMIT, read: dict[int, Any] | None = None
) -> tuple[Any, ...] | None:
    """The callables ``value`` (a resolved object) is or holds: in a
    container's items and keys, an object's own attributes, a mobject's
    members; ``None`` when what it holds can't be read (an object of a
    native kind not in :data:`.coverage.NATIVE_HOLDERS`, a module, an object
    of a class with methods of the scene's own, which are callables it
    holds, more than ``limit`` objects). A mobject's updaters are not
    counted: the engine runs them as the scene's own callbacks, which the
    plan schedules itself. Each object read whose contents may change (a
    list, a dict, a set, an object's attributes) goes into ``read``, by its
    id: what is read is a snapshot, which stands only while nothing stores
    into those (:meth:`Callers.note_snapshot`).
    """
    from arachne.mobject.mobject import Mobject

    from .ticking import _user_methods

    found: dict[int, Any] = {}
    seen: set[int] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > limit:
            return None
        if isinstance(item, FREE_KINDS) and (
            type(item) in FREE_KINDS
            or type(item).__module__.split(".")[0] in ("pathlib", "fractions", "numpy")
        ):
            continue  # a subclass of the scene's is read as any object
        if isinstance(item, (np.ndarray, np.generic)):
            if item.dtype.hasobject:
                if read is not None and isinstance(item, np.ndarray):
                    read[id(item)] = item
                stack.extend(np.asarray(item, dtype=object).flat)
            stack.extend(each for _, each in _own_state(item))  # a subclass's
            continue
        if callable(item):
            found[id(item)] = item
            continue
        if isinstance(item, enum.Enum):
            stack.append(item.value)
            continue
        if isinstance(item, types.ModuleType) or any(
            cls not in NATIVE_HOLDERS for cls in _native_bases(type(item))
        ):
            return None  # what a native kind holds is not read
        if _user_methods(type(item)):
            return None  # its class's methods are the scene's code
        if read is not None and not isinstance(item, (tuple, frozenset)):
            read[id(item)] = item
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
            if isinstance(item, collections.defaultdict):
                stack.append(item.default_factory)
        elif isinstance(item, NATIVE_HOLDERS):
            stack.extend(item)  # type: ignore[arg-type]
        for name, each in _own_state(item):
            if name == "updaters" and isinstance(item, Mobject):
                continue
            stack.append(each)
    return tuple(found.values())


_HEAP_TYPE = 1 << 9  # ``Py_TPFLAGS_HEAPTYPE``: a class Python code defines


def _native_bases(cls: type) -> list[type]:
    """The classes in ``cls``'s order defined natively, ``object`` aside."""
    return [
        each
        for each in cls.__mro__
        if each is not object and not each.__flags__ & _HEAP_TYPE
    ]


def _own_state(item: Any) -> list[tuple[str, Any]]:
    """The attributes ``item`` holds itself: its ``__dict__``, its slots."""
    state: list[tuple[str, Any]] = []
    with contextlib.suppress(TypeError):
        state.extend(vars(item).items())
    missing = object()
    for cls in type(item).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            each = getattr(item, name, missing)
            if each is not missing:
                state.append((name, each))
    return state


def _keyed(call: ast.Call) -> bool:
    return any(keyword.arg in ("key", None) for keyword in call.keywords)


def native_calls(target: Any, call: ast.Call) -> bool | None:
    """Whether native callable ``target`` (a C function or method, a ufunc,
    a builtin or NumPy class, one of NumPy's functions) calls what ``call``
    hands it: ``True`` when listed as calling it, ``False`` when listed as
    never calling it, ``None`` when listed neither way.
    """
    if isinstance(target, np.ufunc):
        return False
    name = getattr(target, "__name__", "")
    if isinstance(target, type):
        if issubclass(target, BaseException):
            return False
        module = target.__module__ or ""
        if module == "builtins":
            if name in _CALLING_BUILTIN_CLASSES:
                return True
            return False if name in _NEVER_CALL_BUILTINS else None
        if module.split(".")[0] == "numpy":
            return name in _NUMPY_CALLERS
        return None
    bound = getattr(target, "__self__", None)
    owner = getattr(target, "__objclass__", None)  # a method of a class
    if owner is None and bound is not None and not isinstance(bound, types.ModuleType):
        owner = type(bound)
    if owner is None:
        module = str(
            getattr(bound, "__name__", None) or getattr(target, "__module__", "") or ""
        )
        root = module.split(".")[0]
        if module == "builtins":
            if name in _KEY_TAKERS:
                return _keyed(call)
            if name == "iter":
                return len(call.args) > 1 or any(
                    isinstance(argument, ast.Starred) for argument in call.args
                )
            return False if name in _NEVER_CALL_BUILTINS else None
        if root == "numpy":
            return name in _NUMPY_CALLERS
        if root in ("math", "cmath"):
            return False
        if root in ("operator", "_operator"):
            return name in _OPERATOR_CALLERS
        return None
    for cls, names in _METHOD_CALLERS.items():
        if issubclass(owner, cls) and name in names:
            return _keyed(call) if name == "sort" else True
    if owner is object or issubclass(owner, _NEVER_CALL_OBJECTS):
        return False  # ``object``'s own (``super().__init__()``'s last)
    return None


def _numpy_function(function: types.FunctionType) -> bool:
    return (getattr(function, "__module__", "") or "").split(".")[0] == "numpy"


class Callers:
    """Which parameters each function may call, remembered for one plan."""

    def __init__(
        self,
        bindings: Callable[[types.FunctionType], dict[str, Any]],
        max_depth: int,
    ) -> None:
        #: What each name a function's code uses is bound to.
        self._bindings = bindings
        self._max_depth = max_depth
        #: Each function's called parameters, by its identity and the class
        #: of the object it is called on (the function is kept so that its
        #: id is not reused); ``None`` while it is being read.
        self._called: dict[tuple[int, type | None], tuple[Any, frozenset[str] | None]]
        self._called = {}
        #: What each resolved object holds (:func:`held_callables`), and the
        #: objects reading it read, by its id (the object kept so that its id
        #: is not reused).
        self._held: dict[int, tuple[Any, tuple[Any, ...] | None, dict[int, Any]]] = {}
        #: The objects whose contents the plan's proofs read as they were
        #: when it was made, by id, each with the attributes read (``None``:
        #: all it holds): snapshots, which stand only while no code the play
        #: reaches stores into what they read (:meth:`.ticking._Writes.stores`).
        self.snapshots: dict[int, tuple[Any, set[str] | None]] = {}

    def held_by(self, value: Any) -> tuple[Any, ...] | None:
        """The callables ``value`` is or holds (:func:`held_callables`),
        noting what was read as a snapshot.
        """
        found = self._held.get(id(value))
        if found is None or found[0] is not value:
            read: dict[int, Any] = {}
            found = (value, held_callables(value, read=read), read)
            self._held[id(value)] = found
        for key, each in found[2].items():
            self.snapshots[key] = (each, None)
        return found[1]

    def note_snapshot(self, value: Any, attribute: str | None = None) -> None:
        """Note that a proof read ``value``'s own ``attribute`` (all of it,
        when ``None``) as it is.
        """
        from .ticking import _MEMBERS_OF, _UNKNOWN

        if isinstance(value, _MEMBERS_OF):
            value = value.mobject
        if value is _UNKNOWN or isinstance(value, FREE_KINDS):
            return
        _, read = self.snapshots.get(id(value), (value, set()))
        if read is not None and attribute is not None:
            read = read | {attribute}
        else:
            read = None
        self.snapshots[id(value)] = (value, read)

    def calls_argument(
        self,
        target: Any,
        call: ast.Call,
        argument: ast.expr,
        bound: bool,
        cls: type | None,
        depth: int,
    ) -> str | None:
        """Why calling ``target`` through ``call`` may call ``argument``, one
        of its arguments, or ``None`` when it never does. ``bound`` says
        whether ``target``'s first parameter is the object it is called on
        (of class ``cls``).
        """
        from .ticking import _PARAMETER, _UNKNOWN

        if target is None or target is _UNKNOWN or target is _PARAMETER:
            return UNKNOWN_CALLEE
        if isinstance(target, types.MethodType):
            this = target.__self__
            cls = this if isinstance(this, type) else type(this)
            target, bound = target.__func__, True
        if isinstance(target, type):
            return self._constructor_calls(target, call, argument, depth)
        if isinstance(target, types.FunctionType):
            if _numpy_function(target):
                return CALLED_NATIVE if target.__name__ in _NUMPY_CALLERS else None
            return self._function_calls(target, call, argument, bound, cls, depth)
        if target is functools.reduce:
            # It calls its function on the items, never an item itself.
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Starred) or any(
                keyword.arg is None for keyword in call.keywords
            ):
                return CALLED_NATIVE  # which argument is its function is not told
            return CALLED_NATIVE if argument is first else None
        if isinstance(
            target,
            (
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
                types.MethodWrapperType,
                types.ClassMethodDescriptorType,
                np.ufunc,
            ),
        ):
            verdict = native_calls(target, call)
            if verdict is None:
                return NATIVE_UNLISTED
            return CALLED_NATIVE if verdict else None
        method = inspect.getattr_static(type(target), "__call__", None)
        if isinstance(method, types.FunctionType):
            return self._function_calls(
                method, call, argument, True, type(target), depth
            )
        return UNKNOWN_CALLEE

    def _constructor_calls(
        self, target: type, call: ast.Call, argument: ast.expr, depth: int
    ) -> str | None:
        """Why making an object of class ``target`` may call ``argument``."""
        if inspect.getattr_static(type(target), "__call__", None) is not type.__call__:
            return UNKNOWN_CALLEE  # a metaclass of its own decides
        verdict = native_calls(target, call)
        if verdict is not None or target.__module__ == "builtins":
            if verdict is None:
                return NATIVE_UNLISTED
            return CALLED_NATIVE if verdict else None
        for name in ("__new__", "__init__"):
            member = inspect.getattr_static(target, name, None)
            member = getattr(member, "__func__", member)  # a static __new__
            if isinstance(member, types.FunctionType):
                reason = self._function_calls(
                    member, call, argument, True, target, depth
                )
                if reason is not None:
                    return reason
            elif member is not getattr(object, name):
                # A builtin base's: a list's or a dict's never calls it.
                owner = getattr(member, "__objclass__", None)
                if not (owner is not None and issubclass(owner, _NEVER_CALL_OBJECTS)):
                    return NATIVE_UNLISTED
        return None

    def _function_calls(
        self,
        function: types.FunctionType,
        call: ast.Call,
        argument: ast.expr,
        bound: bool,
        cls: type | None,
        depth: int,
    ) -> str | None:
        from .ticking import _arguments_of

        called = self.called_parameters(function, cls if bound else None, depth)
        if not called:
            return None
        if EVERY in called:
            return FOLLOWING_STOPS
        binding = _arguments_of(call, function, bound, user_only=False)
        if binding is None:
            return FOLLOWING_STOPS
        varkw = _varkw(function)
        key = next(
            (keyword.arg for keyword in call.keywords if keyword.value is argument),
            None,
        )
        for name, handed in binding.items():
            if not any(
                each is argument
                or (isinstance(argument, ast.Starred) and each is argument.value)
                for each in handed
            ):
                continue
            if name in called:
                return CALLS_IT
            if name == varkw and (
                f"{name}:{key}" in called
                if key is not None
                else any(each.startswith(f"{name}:") for each in called)
            ):
                return CALLS_IT  # what it gathers under that keyword is called
        return None

    def _forwarded_keys(
        self, target: Any, bound: bool, cls: type | None, depth: int
    ) -> set[str] | None:
        """The keywords of a mapping handed on as ``**`` whose values
        ``target`` may call (:data:`EVERY` for any); ``None`` when ``target``
        is no function whose code is read.
        """
        if isinstance(target, types.MethodType):
            this = target.__self__
            cls = this if isinstance(this, type) else type(this)
            target, bound = target.__func__, True
        if isinstance(target, type):
            if (
                inspect.getattr_static(type(target), "__call__", None)
                is not type.__call__
            ):
                return None
            keys: set[str] = set()
            for name in ("__new__", "__init__"):
                member = inspect.getattr_static(target, name, None)
                member = getattr(member, "__func__", member)
                if isinstance(member, types.FunctionType):
                    found = self._forwarded_keys(member, True, target, depth)
                    if found is None:
                        return None
                    keys |= found
                elif member is not getattr(object, name):
                    return None
            return keys
        if not isinstance(target, types.FunctionType) or _numpy_function(target):
            return None
        called = self.called_parameters(target, cls if bound else None, depth)
        if EVERY in called:
            return {EVERY}
        code = target.__code__
        named = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        by_keyword = set(named[code.co_posonlyargcount :]) - (
            set(named[:1]) if bound else set()
        )
        varkw = _varkw(target)
        keys = set()
        for each in called:
            name, _, key = each.partition(":")
            if key:
                if name == varkw:
                    keys.add(key)
            elif name == varkw:
                return {EVERY}  # whatever it gathers may be called
            elif name in by_keyword:
                keys.add(name)
        return keys

    def called_parameters(
        self, function: types.FunctionType, cls: type | None, depth: int
    ) -> frozenset[str]:
        """The parameters ``function`` may call (:data:`EVERY` for any): its
        first one bound to an object of class ``cls`` when given. Its ``**``
        parameter ``kw`` handed on as ``**kw`` is called key by key:
        ``"kw:key"`` for each keyword whose value may be called, ``"kw"``
        for every one.
        """
        key = (id(function), cls)
        found = self._called.get(key)
        if found is not None:
            return found[1] if found[1] is not None else frozenset({EVERY})
        if depth > self._max_depth:
            return frozenset({EVERY})
        self._called[key] = (function, None)  # a call back into itself
        result = self._read(function, cls, depth)
        self._called[key] = (function, result)
        return result

    def _read(
        self, function: types.FunctionType, cls: type | None, depth: int
    ) -> frozenset[str]:
        from . import purity
        from .ticking import _parameters, _source_node, _walk

        try:
            node = _source_node(function)
        except purity.Refused:
            return frozenset({EVERY})
        parameters = set(_parameters(function))
        if not parameters:
            return frozenset()
        bindings = self._bindings(function)
        first = function.__code__.co_varnames[0] if cls is not None else None
        varkw = _varkw(function)
        nodes = _walk(node)
        carried = _carriers(nodes, parameters)
        called: set[str] = set()

        def carries(expr: ast.AST | None) -> set[str]:
            return _carried(expr, carried)

        for item in nodes:
            # Kept where a later call can reach it: an attribute or an item.
            for target, value in _stores(item):
                if any(
                    isinstance(part, (ast.Attribute, ast.Subscript))
                    for part in ast.walk(target)
                ):
                    called |= carries(value)
            if isinstance(item, ast.Subscript):
                # An item of it runs its class's code (a ``defaultdict``'s
                # factory, a ``__getitem__``): judged where it is handed.
                called |= carries(item.value)
            if not isinstance(item, ast.Call):
                continue
            func = item.func
            if not isinstance(func, ast.Attribute):
                called |= carries(func)  # it calls the parameter itself
            arguments = [*item.args, *(keyword.value for keyword in item.keywords)]
            handed = [(argument, carries(argument)) for argument in arguments]
            if not any(names - called for _, names in handed):
                continue
            callees = self._callees(func, function, bindings, first, cls)
            for argument, names in handed:
                if not names - called:
                    continue
                if callees is None:
                    called |= names
                    continue
                forwarded = (
                    varkw is not None
                    and isinstance(argument, ast.Name)
                    and argument.id == varkw
                    and any(
                        keyword.arg is None and keyword.value is argument
                        for keyword in item.keywords
                    )
                )
                for target, target_bound, target_cls in callees:
                    if forwarded:
                        # ``**kwargs`` handed on: called key by key.
                        keys = self._forwarded_keys(
                            target, target_bound, target_cls, depth + 1
                        )
                        if keys is not None and EVERY not in keys:
                            called |= {f"{varkw}:{key}" for key in keys}
                            continue
                    if self.calls_argument(
                        target, item, argument, target_bound, target_cls, depth + 1
                    ):
                        called |= names
                        break
        return frozenset(
            each
            for each in called
            if each == EVERY or each.partition(":")[0] in parameters
        )

    def _callees(
        self,
        func: ast.AST,
        function: types.FunctionType,
        bindings: dict[str, Any],
        first: str | None,
        cls: type | None,
    ) -> list[tuple[Any, bool, type | None]] | None:
        """What ``func``, in ``function``'s code, may name when called: each
        with whether it is bound to the object it is called on and that
        object's class; ``None`` when that can't be told.
        """
        from .ticking import _PARAMETER, _UNKNOWN, _module_named, _super_call

        if isinstance(func, ast.Name):
            found = bindings.get(func.id, _UNKNOWN)
            if found is _UNKNOWN or found is _PARAMETER:
                return None
            return [(found, False, None)]
        if not isinstance(func, ast.Attribute):
            return None
        receiver = func.value
        if _super_call(receiver, bindings):
            member = _after(function, cls, func.attr)
            return None if member is None else [_method(member, cls)]
        if isinstance(receiver, ast.Name) and receiver.id == first and cls is not None:
            member = inspect.getattr_static(cls, func.attr, None)
            return None if member is None else [_method(member, cls)]
        module = _module_named(receiver, bindings)
        if module is not None:
            found = getattr(module, func.attr, _UNKNOWN)
            return None if found is _UNKNOWN else [(found, False, None)]
        if isinstance(receiver, ast.Name):
            holder = bindings.get(receiver.id, _UNKNOWN)
            if holder is not _UNKNOWN and holder is not _PARAMETER:
                member = inspect.getattr_static(type(holder), func.attr, None)
                if member is None:
                    return None
                if isinstance(member, (types.FunctionType, staticmethod, classmethod)):
                    return [_method(member, type(holder))]
                found = getattr(holder, func.attr, _UNKNOWN)
                return None if found is _UNKNOWN else [(found, False, None)]
        return None


def _varkw(function: types.FunctionType) -> str | None:
    """The name of ``function``'s ``**`` parameter, if it has one."""
    code = function.__code__
    if not code.co_flags & inspect.CO_VARKEYWORDS:
        return None
    rest = code.co_argcount + code.co_kwonlyargcount
    rest += bool(code.co_flags & inspect.CO_VARARGS)
    return code.co_varnames[rest]


def _method(member: Any, cls: type | None) -> tuple[Any, bool, type | None]:
    """A class's attribute ``member``, called on an object of class ``cls``."""
    if isinstance(member, staticmethod):
        return member.__func__, False, None
    if isinstance(member, classmethod):
        return member.__func__, True, cls
    if isinstance(member, types.FunctionType):
        return member, True, cls
    return member, False, None


def _after(function: types.FunctionType, cls: type | None, name: str) -> Any:
    """What ``super().name`` names in ``function``, a method of a class in
    ``cls``'s order; ``None`` when that can't be told.
    """
    if cls is None:
        return None
    order = cls.__mro__
    for index, klass in enumerate(order):
        if any(
            getattr(value, "__func__", value) is function
            for value in vars(klass).values()
        ):
            for later in order[index + 1 :]:
                if name in vars(later):
                    return vars(later)[name]
            return None
    return None


def _stores(item: ast.AST) -> list[tuple[ast.AST, ast.AST | None]]:
    """What a statement or expression binds or stores, and the value."""
    if isinstance(item, ast.Assign):
        return [(target, item.value) for target in item.targets]
    if isinstance(item, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return [(item.target, item.value)]
    if isinstance(item, (ast.For, ast.AsyncFor, ast.comprehension)):
        return [(item.target, item.iter)]
    if isinstance(item, ast.withitem) and item.optional_vars is not None:
        return [(item.optional_vars, item.context_expr)]
    return []


def _carriers(nodes: tuple[ast.AST, ...], parameters: set[str]) -> dict[str, set[str]]:
    """Each name the code binds to something taken from a parameter, with
    the parameters it took it from; each parameter, with itself.
    """
    from .ticking import _names_bound

    carried: dict[str, set[str]] = {name: {name} for name in parameters}
    pairs = [pair for item in nodes for pair in _stores(item)]
    changed = True
    while changed:
        changed = False
        for target, value in pairs:
            names = _carried(value, carried)
            if not names:
                continue
            for bound in _names_bound(target):
                held = carried.setdefault(bound, set())
                if not names <= held:
                    held |= names
                    changed = True
    return carried


def _carried(expr: ast.AST | None, carried: dict[str, set[str]]) -> set[str]:
    """The parameters whose objects (or anything taken from them) ``expr``
    may evaluate to, or hold, by the rules of :data:`.coverage.OPERATIONS`:
    a constant and a formatted string carry nothing, nor does a lambda
    (its body's calls are read where they are); a name, what its stores
    bound it to; any other form (an operator, which may give back what its
    operands hold, a ``[f] * n`` among them) carries what its parts do.
    """
    if expr is None:
        return set()
    rule = rule_of(expr)
    if rule in (Rule.CONSTANT, Rule.LAMBDA):
        return set()
    if rule is Rule.NAME:
        assert isinstance(expr, ast.Name)
        return set(carried.get(expr.id, ()))
    found: set[str] = set()
    for child in ast.iter_child_nodes(expr):
        if isinstance(child, ast.expr_context):
            continue
        found |= _carried(child, carried)
    return found
