"""The proof that a callback is a function of the instant (route 2).

A callback is admitted to route 2 only when its code shows, statically, that
what it leaves depends on nothing but the instant and registered state. This
module reads the callback's source and proves one of three shapes, or says
why it can't:

- a *setter body* (a dt-free updater, an ``UpdateFromFunc`` or
  ``UpdateFromAlphaFunc`` function): a straight-line sequence of calls to
  admitted setters (:mod:`.setters`) on the mobject it updates, or on a scene
  mobject it names, with local assignments between them and no conditional,
  early return or loop that could leave an output unassigned;
- a *builder* (the function ``always_redraw`` rebuilds from): it returns a
  mobject it builds in the same call;
- a *readout* (a camera's centre function, an ``f_always`` argument, or a
  helper either calls): it returns a value.

Wherever it reads, it may read only registered state: mobjects (their
values are recorded), the scene clock, immutable constants of exactly the
engine's value types, and the library entries (:mod:`.library`); every
mobject it reaches must run the engine's own methods (:mod:`.dispatch`); and
it may change nothing but the outputs above. Admission is by allowlist: a
call, value or receiver the proof does not positively admit is a refusal,
and a value that may be ``None`` (an item picked from ``[None, RED]``)
counts as ``None``. A
``nonlocal`` or ``global`` write, an attribute or item assignment, a call
that changes a scene mobject other than an admitted setter, a relative
change to the mobject it writes (``shift``, ``rotate``, a setter whose
argument reads the receiver), a random draw, I/O or a clock other than the
scene's, are refusals. The proof records what the callback reads and
writes, for the dependency order, and the bindings it read its names from,
which route 2's planned restore compares with what they hold then.
"""

from __future__ import annotations

__all__ = [
    "Binding",
    "Proof",
    "Refused",
    "prove_builder",
    "prove_readout",
    "prove_setter_body",
]

import ast
import builtins
import dis
import inspect
import linecache
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any

import numpy as np

from . import library, setters
from .dispatch import family_refusal
from .frozen import engine_code
from .identity import ByIdentity, one_of

#: How deep the proof follows calls into helper functions.
MAX_DEPTH = 4


class Refused(Exception):
    """The callback is not a provable function of the instant; the message
    says why.
    """


@dataclass(frozen=True)
class Binding:
    """A name the callback read, and what it held when the play began.

    ``where`` is ``"cell"`` (a closure cell), ``"global"``, ``"attribute"``
    (an attribute of the scene) or ``"default"`` (a default argument);
    ``holder`` is what holds the name: the cell itself, the globals
    namespace, the scene, or the function whose default it is. Two names
    alike in different holders are two bindings. ``contents`` is a detached,
    read-only snapshot of everything mutable the value holds (a container's
    items, nested ones included, and every array's numbers), so a later
    restore can tell whether it still holds them.
    """

    where: str
    name: str
    value: Any
    contents: Any = None
    holder: Any = None


@dataclass
class Proof:
    """What a proven callback reads and writes."""

    #: The scene mobjects it reads, at the root it reaches them from.
    reads: list[Any] = field(default_factory=list)
    #: The mobjects it writes.
    writes: list[Any] = field(default_factory=list)
    bindings: list[Binding] = field(default_factory=list)
    #: Whether it reads the scene clock.
    clock: bool = False

    def read(self, mobjects: Iterable[Any]) -> None:
        """Record reads of ``mobjects``. Whatever a callback does with a
        mobject it reaches (a read, a setter, a group's setter on each
        child) runs the methods of its family, so each member must run the
        engine's own (:mod:`.dispatch`).
        """
        for mob in mobjects:
            if not any(mob is seen for seen in self.reads):
                reason = family_refusal([mob])
                if reason is not None:
                    raise Refused(f"it reaches {reason}")
                self.reads.append(mob)

    def bind(self, binding: Binding) -> None:
        if not any(
            b.where == binding.where
            and b.holder is binding.holder
            and b.name == binding.name
            for b in self.bindings
        ):
            self.bindings.append(binding)


# Abstract values: what an expression evaluates to, as far as the proof knows.


class _Value:
    """A plain value computed at run time (a number, array, string, colour)."""


VALUE = _Value()


class _External(_Value):
    """A plain value the callback did not make: an array (or a view of one)
    read from a closure cell, a global, a default, the scene or a mobject.
    Reading it is fine; changing it in place is not.
    """


EXTERNAL = _External()


class _NoneValue(_Value):
    """``None``: given to a setter, it leaves what it sets as it was."""


NONE = _NoneValue()
_PLAIN = (VALUE, EXTERNAL, NONE)


def _is_plain(value: Any) -> bool:
    return any(value is plain for plain in _PLAIN)


@dataclass(frozen=True)
class _Scene:
    """A scene mobject (or several), reached from these roots; reading it
    reads them.
    """

    roots: tuple[Any, ...]


@dataclass(frozen=True)
class _Fresh:
    """A mobject the callback built in this call; ``owned`` when it holds no
    scene mobject, so changing it changes nothing else. ``cls`` is its class
    when the proof can tell it.
    """

    owned: bool
    cls: type | None = None


@dataclass(frozen=True)
class _Known:
    """An object known when the play began: a module, function or class, the
    scene, its renderer or camera.
    """

    obj: Any


@dataclass(frozen=True)
class _Items:
    """A container whose items are these abstract values. ``exact`` when
    they are its items in order (a literal, a tuple the callback read);
    otherwise each stands for any number of items (a comprehension, a
    library function's result), and an item taken out of it is their join.
    """

    items: tuple[Any, ...]
    exact: bool = False


@dataclass(frozen=True)
class _LambdaValue:
    """A lambda written inside the callback, with the names it closes over."""

    node: ast.Lambda
    env: dict[str, Any]


@dataclass(frozen=True)
class _Method:
    """A bound method, not yet called."""

    on: Any
    name: str


class _Role(Enum):
    SETTER = "setter body"
    BUILDER = "builder"
    READOUT = "readout"


# Finding a function's source.

_TREES: dict[tuple[str, int], ast.Module] = {}
_PROLOGUE = frozenset(
    {"RESUME", "COPY_FREE_VARS", "MAKE_CELL", "RETURN_GENERATOR", "NOP"}
)
_RETURNS = frozenset({"RETURN_VALUE", "RETURN_CONST"})


def _function_node(function: Callable[..., Any]) -> ast.Lambda | ast.FunctionDef:
    code = getattr(function, "__code__", None)
    if code is None:
        raise Refused(f"{function!r} has no Python code to read")
    filename = code.co_filename
    lines = linecache.getlines(filename, function.__globals__)
    if not lines:
        raise Refused(f"the source of {code.co_name} is not available")
    text = "".join(lines)
    key = (filename, hash(text))
    tree = _TREES.get(key)
    if tree is None:
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            raise Refused(f"the source of {code.co_name} does not parse") from error
        _TREES[key] = tree
    # The prologue (RESUME, the closure's cells) sits at column 0 of the
    # first line, outside a lambda: only the body's instructions locate it.
    # So, on CPython 3.11 and some 3.12 releases (3.12.3 among them), does a
    # lambda's implicit return; later releases place it on the body.
    line = code.co_firstlineno
    unplaced_return = (line, line, 0, 0) if code.co_name == "<lambda>" else None
    positions = [
        (pos.lineno, pos.col_offset)
        for instruction in dis.get_instructions(code)
        if instruction.opname not in _PROLOGUE
        and (pos := instruction.positions) is not None
        and pos.lineno is not None
        and pos.col_offset is not None
        and not (instruction.opname in _RETURNS and tuple(pos) == unplaced_return)
    ]
    candidates: list[ast.Lambda | ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and code.co_name == "<lambda>":
            first = node.lineno
        elif isinstance(node, ast.FunctionDef) and node.name == code.co_name:
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        else:
            continue
        if first != code.co_firstlineno or not _same_arguments(node, code):
            continue
        if all(_contains(node, line, column) for line, column in positions):
            candidates.append(node)
    if not candidates:
        raise Refused(f"the source of {code.co_name} does not match its code")
    if len(candidates) == 1:
        return candidates[0]
    # Nodes nest or are disjoint, so candidates holding a located instruction
    # lie one inside another, and the narrowest is the source unless the code
    # is a wider one's: then it makes the narrower's function, a code of its
    # own name, first line and arguments (``lambda: lambda: 1``). With no
    # located instruction, any candidate could be.
    if not positions or _makes_its_like(code, code):
        raise Refused(f"the source of {code.co_name} is ambiguous")
    return min(candidates, key=_span)


def _makes_its_like(code: types.CodeType, like: types.CodeType) -> bool:
    """Whether ``code`` makes, at any depth, a function a node alike to
    ``like``'s could be the source of.
    """
    for const in code.co_consts:
        if isinstance(const, types.CodeType) and (
            _signature(const) == _signature(like) or _makes_its_like(const, like)
        ):
            return True
    return False


def _signature(code: types.CodeType) -> tuple[Any, ...]:
    count = code.co_argcount + code.co_kwonlyargcount
    return (code.co_name, code.co_firstlineno, code.co_varnames[:count])


def _span(node: ast.AST) -> tuple[int, int]:
    end_line = getattr(node, "end_lineno", node.lineno) or node.lineno  # type: ignore[attr-defined]
    end_col = getattr(node, "end_col_offset", 0) or 0
    return (end_line - node.lineno, end_col - node.col_offset)  # type: ignore[attr-defined]


def _contains(node: ast.AST, line: int, column: int) -> bool:
    start = (node.lineno, node.col_offset)  # type: ignore[attr-defined]
    end = (node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
    return start <= (line, column) <= end


def _same_arguments(node: ast.Lambda | ast.FunctionDef, code: types.CodeType) -> bool:
    arguments = node.args
    names = [a.arg for a in (*arguments.posonlyargs, *arguments.args)]
    count = code.co_argcount + code.co_kwonlyargcount
    expected = list(code.co_varnames[:count])
    return names + [a.arg for a in arguments.kwonlyargs] == expected


# The proof itself.


class _Prover:
    def __init__(self, scene: Any, proof: Proof) -> None:
        self.scene = scene
        self.proof = proof
        self.depth = 0
        #: Proving a setter body: no conditional in it may choose what it
        #: sets.
        self.setter_mode = False
        self._callables_seen: set[int] = set()
        self._mobjects_seen: set[int] = set()

    # Names

    def resolve_global(self, function: Any, name: str) -> Any:
        """What a free name of ``function`` holds: its closure cell, its
        global, or a builtin.
        """
        code = function.__code__
        if name in code.co_freevars:
            cell = function.__closure__[code.co_freevars.index(name)]
            try:
                value = cell.cell_contents
            except ValueError as error:
                raise Refused(f"{name} is not bound yet") from error
            return self.classify(value, "cell", name, cell)
        if name in function.__globals__:
            return self.classify(
                function.__globals__[name], "global", name, function.__globals__
            )
        if hasattr(builtins, name):
            return self.classify(getattr(builtins, name), None, name)
        raise Refused(f"{name} is not defined")

    def classify(
        self, value: Any, where: str | None, name: str, holder: Any = None
    ) -> Any:
        """The abstract value of an object a callback reads by name, bound in
        ``holder`` (a cell, a namespace, the scene or a function).
        """
        abstract = self._classify(value, name)
        if where is not None:
            self.proof.bind(Binding(where, name, value, _contents(value), holder))
        if isinstance(abstract, _Scene):
            self.proof.read(abstract.roots)
        elif isinstance(abstract, _Items):
            for item in abstract.items:
                if isinstance(item, _Scene):
                    self.proof.read(item.roots)
        return abstract

    def _classify(self, value: Any, name: str) -> Any:
        from arachne.mobject.mobject import Mobject
        from arachne.scene.scene import Scene

        if isinstance(value, Mobject):
            return _Scene((value,))
        if value is self.scene or isinstance(value, Scene):
            if value is not self.scene:
                raise Refused(f"{name} is another scene")
            return _Known(value)
        if value is None:
            return NONE
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray or value.dtype == object:
                raise Refused(
                    f"{name} is a {type(value).__name__} of {value.dtype}, not an"
                    " array of numbers"
                )
            return EXTERNAL
        if _constant(value):
            return VALUE
        if isinstance(value, types.ModuleType):
            if not setters.pure_module(value):
                # Its name is not read: a module of a scene's may compute it.
                raise Refused(f"{name} is a module a callback may not read")
            return _Known(value)
        if one_of(type(value), (list, tuple)):
            return _Items(
                tuple(self._classify(item, f"{name}[]") for item in value), exact=True
            )
        if isinstance(value, (types.FunctionType, types.MethodType, type)) or callable(
            value
        ):
            if isinstance(value, types.MethodType):
                return _Method(
                    self._classify(value.__self__, f"{name}.__self__"), value.__name__
                )
            return _Known(value)
        raise Refused(
            f"{name} holds a {type(value).__name__}, which a record does not register"
        )

    # Statements

    def run_body(
        self,
        node: ast.Lambda | ast.FunctionDef,
        function: Any,
        env: dict[str, Any],
        role: _Role,
    ) -> Any:
        """The abstract value the body returns."""
        if isinstance(node, ast.Lambda):
            if role is _Role.SETTER:
                return self.setter_expression(node.body, function, env)
            return self.expression(node.body, function, env)
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        result: Any = NONE
        for index, statement in enumerate(body):
            last = index == len(body) - 1
            if isinstance(statement, ast.Return):
                if not last:
                    raise Refused("it returns early")
                if statement.value is None:
                    return NONE
                if role is _Role.SETTER:
                    return self.setter_expression(statement.value, function, env)
                return self.expression(statement.value, function, env)
            if isinstance(statement, ast.Expr):
                if role is _Role.SETTER:
                    self.setter_expression(statement.value, function, env)
                else:
                    self.expression(statement.value, function, env)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                if statement.value is None:
                    raise Refused("it declares a name without a value")
                value = self.expression(statement.value, function, env)
                for target in targets:
                    self.assign(target, value, env)
            elif isinstance(statement, ast.AugAssign):
                # In place: only a value the callback computed itself may
                # change (an array it read may be the scene's, or a closure's).
                if (
                    not isinstance(statement.target, ast.Name)
                    or env.get(statement.target.id) is not VALUE
                ):
                    raise Refused("it changes in place a name or value it did not make")
                self._plain(
                    self.expression(statement.value, function, env), "computes with"
                )
                env[statement.target.id] = VALUE
            elif isinstance(statement, ast.Pass):
                continue
            elif isinstance(statement, (ast.Global, ast.Nonlocal)):
                raise Refused(f"it declares {type(statement).__name__.lower()} names")
            else:
                raise Refused(
                    f"its body has a {type(statement).__name__} statement, which can leave"
                    " an output unassigned"
                )
        return result

    def assign(self, target: ast.AST, value: Any, env: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if (
                isinstance(value, _Items)
                and value.exact
                and len(value.items) == len(target.elts)
            ):
                for element, item in zip(target.elts, value.items, strict=True):
                    self.assign(element, item, env)
                return
            item = _element(value)
            for element in target.elts:
                self.assign(element, item, env)
        else:
            raise Refused("it assigns to an attribute or an item")

    # Setter calls

    def setter_expression(
        self, node: ast.AST, function: Any, env: dict[str, Any]
    ) -> Any:
        """A setter body's statement or lambda: a chain of admitted setters."""
        if isinstance(node, ast.Constant) and node.value is None:
            return VALUE
        if isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self.setter_expression(element, function, env)
            return VALUE
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            if isinstance(node, ast.IfExp):
                raise Refused("a conditional decides which setters run")
            raise Refused("its body is not a sequence of setter calls")
        method = node.func.attr
        on = node.func.value
        if isinstance(on, ast.Call):
            receiver = self.setter_expression(on, function, env)
        else:
            # Naming the receiver is not reading it: what a setter reads of
            # its own receiver is the setter's, and its property test's.
            reads = len(self.proof.reads)
            receiver = self.expression(on, function, env)
            del self.proof.reads[reads:]
        if not isinstance(receiver, _Scene) or len(receiver.roots) != 1:
            raise Refused(
                f"it calls {method} on something that is not one scene mobject"
            )
        mobject = receiver.roots[0]
        setter = setters.setter_for(mobject, method)
        if setter is None:
            raise Refused(
                f"{type(mobject).__name__}.{method} is not an admitted setter"
            )
        before = list(self.proof.reads)
        params = setter.params
        if (
            any(isinstance(a, ast.Starred) for a in node.args)
            or any(k.arg is None for k in node.keywords)
            or len(node.args) > len(params)
            or any(k.arg not in params[len(node.args) :] for k in node.keywords)
        ):
            raise Refused(
                f"{method} is given arguments its property test does not cover"
                f" (it covers {', '.join(params)})"
            )
        if setter.fresh_argument:
            if node.keywords or len(node.args) != 1:
                raise Refused(f"{method} is given options that read its receiver")
            value = self.expression(node.args[0], function, env)
            if not (isinstance(value, _Fresh) and value.owned):
                raise Refused(f"{method} is given a mobject it would change")
        else:
            named = [
                *zip(params, node.args, strict=False),
                *((k.arg, k.value) for k in node.keywords),
            ]
            for param, argument in named:
                value = self.expression(argument, function, env)
                if value is NONE or _holds_none(value):
                    raise Refused(
                        f"{method} is given None, which leaves what it sets as it was"
                    )
                takes_mobject = "mobject" in (param or "")
                if not (
                    _all_plain(value)
                    or (
                        takes_mobject
                        and isinstance(value, _Scene)
                        and len(value.roots) == 1
                    )
                ):
                    raise Refused(f"{method} is given {_describe(value)} as {param}")
        family = {id(m) for m in mobject.get_family()}
        read_now = self.proof.reads[len(before) :]
        if any(
            id(read) in family or id(mobject) in {id(m) for m in read.get_family()}
            for read in read_now
        ):
            raise Refused(
                f"{method} reads the mobject it writes, so each call changes it from"
                " where the last left it"
            )
        if not any(m is mobject for m in self.proof.writes):
            self.proof.writes.append(mobject)
        return receiver

    # Expressions

    def expression(self, node: ast.AST, function: Any, env: dict[str, Any]) -> Any:
        method = getattr(self, f"_{type(node).__name__}", None)
        if method is None:
            raise Refused(
                f"it uses {type(node).__name__}, which the proof does not read"
            )
        return method(node, function, env)

    def _Constant(self, node: ast.Constant, function: Any, env: dict[str, Any]) -> Any:
        return NONE if node.value is None else VALUE

    def _JoinedStr(
        self, node: ast.JoinedStr, function: Any, env: dict[str, Any]
    ) -> Any:
        for value in node.values:
            self.expression(value, function, env)
        return VALUE

    def _FormattedValue(
        self, node: ast.FormattedValue, function: Any, env: dict[str, Any]
    ) -> Any:
        self._plain(self.expression(node.value, function, env), "formats")
        if node.format_spec is not None:
            self.expression(node.format_spec, function, env)
        return VALUE

    def _Name(self, node: ast.Name, function: Any, env: dict[str, Any]) -> Any:
        if node.id in env:
            value = env[node.id]
            if isinstance(value, _Scene):
                self.proof.read(value.roots)
            return value
        return self.resolve_global(function, node.id)

    def _Attribute(
        self, node: ast.Attribute, function: Any, env: dict[str, Any]
    ) -> Any:
        on = self.expression(node.value, function, env)
        return self.attribute(on, node.attr)

    def attribute(self, on: Any, name: str) -> Any:
        from arachne.mobject.mobject import Mobject

        if name.startswith("__"):
            raise Refused(f"it reads {name}")
        if isinstance(on, _Known):
            obj = on.obj
            if obj is self.scene:
                if name == "time":
                    self.proof.clock = True
                    return VALUE
                if name in ("renderer", "camera"):
                    return _Known(getattr(obj, name))
                value = getattr(type(obj), name, None)
                if isinstance(value, types.FunctionType):
                    return _Method(on, name)
                value = getattr(obj, name)
                return self.classify(value, "attribute", f"self.{name}", obj)
            if obj is getattr(self.scene, "renderer", None):
                if name == "time":
                    self.proof.clock = True
                    return VALUE
                if name == "camera":
                    return _Known(self.scene.renderer.camera)
                raise Refused(f"it reads the renderer's {name}")
            if obj is getattr(self.scene, "camera", None):
                value = getattr(obj, name)
                if isinstance(value, Mobject):
                    return self.classify(value, "attribute", f"camera.{name}")
                if isinstance(value, types.MethodType):
                    return _Method(on, name)
                raise Refused(f"it reads the camera's {name}")
            if isinstance(obj, types.ModuleType):
                value = getattr(obj, name, None)
                if value is None:
                    raise Refused(f"{obj.__name__}.{name} is not defined")
                return self.classify(value, None, f"{obj.__name__}.{name}")
            if isinstance(obj, type):
                raise Refused(f"it reads {obj.__name__}.{name}")
            raise Refused(f"it reads {name} off {obj!r}")
        if isinstance(on, _Scene):
            return self.mobject_attribute(on, name, fresh=False)
        if isinstance(on, _Fresh):
            return self.mobject_attribute(on, name, fresh=True)
        if on is VALUE or on is EXTERNAL:
            if name in _VIEW_ATTRIBUTES:
                return on
            return VALUE if name in _VALUE_ATTRIBUTES else _Method(on, name)
        raise Refused(f"it reads {name} off {type(on).__name__.strip('_')}")

    def mobject_attribute(self, on: _Scene | _Fresh, name: str, *, fresh: bool) -> Any:
        from arachne.mobject.mobject import Mobject

        cls = on.cls if isinstance(on, _Fresh) else type(on.roots[0])
        if cls is None:
            return _Method(on, name)
        cls_value = inspect.getattr_static(cls, name, None)
        if isinstance(cls_value, (types.FunctionType, staticmethod, classmethod)):
            return _Method(on, name)
        if isinstance(cls_value, property):
            if not setters.builtin_class(cls):
                raise Refused(f"it reads the property {name} of a user class")
            # The engine's property: a value of the mobject's, which may be a
            # live view of its data (``points``).
            return EXTERNAL
        if fresh:
            raise Refused(f"it reads {name} off a mobject it built")
        if isinstance(on, _Scene) and len(on.roots) == 1:
            value = on.roots[0].__dict__.get(name, _MISSING)
            if value is _MISSING:
                raise Refused(f"{cls.__name__} has no attribute {name}")
            if isinstance(value, Mobject):
                self.proof.read([value])
                return _Scene((value,))
            if isinstance(value, (list, tuple)) and all(
                isinstance(v, Mobject) for v in value
            ):
                if not value:
                    return VALUE
                self.proof.read(value)
                return _Scene(tuple(value))
            if type(value) is np.ndarray and value.dtype != object:
                return EXTERNAL
            if value is None:
                return NONE
            if _constant(value):
                return VALUE
            raise Refused(
                f"it reads {name}, a {type(value).__name__} the record does not register"
            )
        raise Refused(f"it reads {name} off several mobjects at once")

    def _Subscript(
        self, node: ast.Subscript, function: Any, env: dict[str, Any]
    ) -> Any:
        on = self.expression(node.value, function, env)
        self._plain(self.expression(node.slice, function, env), "indexes with")
        if isinstance(on, _Scene):
            return on
        if isinstance(on, _Fresh):
            return on
        if isinstance(on, _Items):
            index = node.slice
            if isinstance(index, ast.Slice):
                return _Items(on.items)
            if (
                on.exact
                and isinstance(index, ast.Constant)
                and type(index.value) is int
                and -len(on.items) <= index.value < len(on.items)
            ):
                return on.items[index.value]
            return _element(on)
        if on is VALUE or on is EXTERNAL:
            # An index of an array it read is a view of it.
            return on
        raise Refused(f"it indexes {_describe(on)}")

    def _Slice(self, node: ast.Slice, function: Any, env: dict[str, Any]) -> Any:
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                self._plain(self.expression(part, function, env), "slices with")
        return VALUE

    def _Starred(self, node: ast.Starred, function: Any, env: dict[str, Any]) -> Any:
        return self.expression(node.value, function, env)

    def _BinOp(self, node: ast.BinOp, function: Any, env: dict[str, Any]) -> Any:
        left = self.expression(node.left, function, env)
        right = self.expression(node.right, function, env)
        self._plain(left, "computes with")
        self._plain(right, "computes with")
        operands = [v for v in (left, right) if isinstance(v, _Items)]
        if operands:
            # ``rows + []`` is a new list holding the same items.
            return _Items(tuple(item for v in operands for item in v.items))
        return VALUE

    def _UnaryOp(self, node: ast.UnaryOp, function: Any, env: dict[str, Any]) -> Any:
        self._plain(self.expression(node.operand, function, env), "computes with")
        return VALUE

    def _BoolOp(self, node: ast.BoolOp, function: Any, env: dict[str, Any]) -> Any:
        if self.setter_mode and self.depth == 0:
            raise Refused("a conditional (and/or) chooses what is set")
        values = [self.expression(value, function, env) for value in node.values]
        for value in values:
            self._plain(value, "tests")
        # ``a or b`` is one of its operands itself.
        for plain in (NONE, EXTERNAL):
            if any(v is plain for v in values):
                return plain
        return VALUE

    def _Compare(self, node: ast.Compare, function: Any, env: dict[str, Any]) -> Any:
        self._plain(self.expression(node.left, function, env), "compares")
        for value in node.comparators:
            self._plain(self.expression(value, function, env), "compares")
        return VALUE

    def _IfExp(self, node: ast.IfExp, function: Any, env: dict[str, Any]) -> Any:
        if self.setter_mode and self.depth == 0:
            # In the body itself; a helper's conditional value is followed
            # (a None it may return reaches the setter as None).
            raise Refused("a conditional decides what is set")
        self._plain(self.expression(node.test, function, env), "tests")
        body = self.expression(node.body, function, env)
        orelse = self.expression(node.orelse, function, env)
        return _join([body, orelse])

    def _Tuple(self, node: ast.Tuple, function: Any, env: dict[str, Any]) -> Any:
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return _Items(tuple(self.expression(e, function, env) for e in node.elts))
        return _Items(
            tuple(self.expression(e, function, env) for e in node.elts), exact=True
        )

    _List = _Tuple

    def _Set(self, node: ast.Set, function: Any, env: dict[str, Any]) -> Any:
        for element in node.elts:
            self._plain(self.expression(element, function, env), "collects")
        return VALUE

    def _Dict(self, node: ast.Dict, function: Any, env: dict[str, Any]) -> Any:
        held = []
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                self._plain(self.expression(key, function, env), "keys with")
            item = self.expression(value, function, env)
            self._plain(item, "maps to")
            held.append(_flat_plain(item))
        # What it maps to is what an item taken out of it is.
        return _join(held)

    def _comprehension(
        self, node: Any, function: Any, env: dict[str, Any]
    ) -> dict[str, Any]:
        inner = dict(env)
        for generator in node.generators:
            if generator.is_async:
                raise Refused("it iterates asynchronously")
            iterable = self.expression(generator.iter, function, inner)
            self.assign(generator.target, _element(iterable), inner)
            for condition in generator.ifs:
                self._plain(self.expression(condition, function, inner), "tests")
        return inner

    def _ListComp(self, node: ast.ListComp, function: Any, env: dict[str, Any]) -> Any:
        inner = self._comprehension(node, function, env)
        return _Items((self.expression(node.elt, function, inner),))

    _GeneratorExp = _ListComp
    _SetComp = _ListComp

    def _DictComp(self, node: ast.DictComp, function: Any, env: dict[str, Any]) -> Any:
        inner = self._comprehension(node, function, env)
        self._plain(self.expression(node.key, function, inner), "keys with")
        value = self.expression(node.value, function, inner)
        self._plain(value, "maps to")
        return _flat_plain(value)

    def _Lambda(self, node: ast.Lambda, function: Any, env: dict[str, Any]) -> Any:
        return _LambdaValue(node, dict(env))

    def _Call(self, node: ast.Call, function: Any, env: dict[str, Any]) -> Any:
        callee = self.expression(node.func, function, env)
        _check_form(callee, node)
        arguments = [self.expression(a, function, env) for a in node.args]
        keywords = {
            k.arg: self.expression(k.value, function, env) for k in node.keywords
        }
        if None in keywords:
            raise Refused("it passes keyword arguments it unpacks")
        return self.call(callee, arguments, keywords, function)

    # Calls

    def call(
        self,
        callee: Any,
        arguments: list[Any],
        keywords: dict[str | None, Any],
        function: Any,
    ) -> Any:
        values = [*arguments, *keywords.values()]
        if isinstance(callee, _LambdaValue):
            return self.call_lambda(callee, arguments, keywords, function)
        if isinstance(callee, _Method):
            return self.call_method(callee, values, function)
        if not isinstance(callee, _Known):
            raise Refused(f"it calls a {type(callee).__name__.strip('_')}")
        target = callee.obj
        kind = setters.constructor_kind(target)
        if kind is not None:
            owned = True
            self.prove_stored_callables(_roots(values))
            for value in values:
                self.check_callable_argument(value, function)
                if isinstance(value, _Scene) or _holds_scene(value):
                    if kind != "reads":
                        raise Refused(f"{target.__name__} would hold a scene mobject")
                elif isinstance(value, _Fresh) or _holds_fresh(value):
                    owned = owned and _all_owned(value)
            return _Fresh(owned, target)
        found = library.entry(target)
        if found is not None:
            return self.library_call(found, target, arguments, keywords, function)
        if _is_rate_function(target):
            for value in values:
                self._plain(value, "times with")
            return VALUE
        if isinstance(target, types.FunctionType) and not _engine_function(target):
            return self.call_user(target, None, arguments, keywords)
        raise Refused(f"it calls {_qualified(target)}, which no library entry admits")

    def library_call(
        self,
        found: library.Entry,
        target: Any,
        arguments: list[Any],
        keywords: dict[str | None, Any],
        function: Any,
    ) -> Any:
        """A call to a library entry, in its admitted form (checked by
        :func:`_check_form`): what it may be given, and what it returns.
        """
        given = [*arguments, *keywords.values()]
        if target is isinstance:
            # Its second argument is a class it compares with, not calls.
            given = given[:1]
        callables = [value for value in given if _is_callable_value(value)]
        others = [value for value in given if not _is_callable_value(value)]
        if callables and not found.calls:
            raise Refused(f"{found.name} is given a callable")
        results = [self.check_callable_argument(v, function) for v in callables]
        for result in results:
            if not _all_plain(result):
                raise Refused(
                    f"{found.name} is given a function that returns {_describe(result)}"
                )
        if found.takes_mobjects:
            return _container_result(target, arguments, others)
        for value in others:
            if not _all_plain(value):
                raise Refused(f"{found.name} is given {_describe(value)}")
        for index in found.content:
            if index < len(arguments) and (
                arguments[index] is NONE or _holds_none(arguments[index])
            ):
                raise Refused(
                    f"{found.name} is given None, which its result would hold"
                    " (an array of objects)"
                )
        if found.allocates:
            return VALUE
        # It may hand back what it was given (or a view of it): an array the
        # callback did not make, or None.
        return _join([_flat_plain(value) for value in [*others, *results]] or [VALUE])

    def check_callable_argument(self, value: Any, function: Any) -> Any:
        """A callable passed on (``plot(f)``, ``sorted(key=f)``) must itself
        be a readout, or a library entry: the abstract value it returns.
        Anything else that may be called is refused; a value that is not
        callable is returned as it is.
        """
        if isinstance(value, _LambdaValue):
            return self.call_lambda(value, None, {}, function)
        if isinstance(value, _Method):
            return self.call_method(value, None, function)
        if isinstance(value, _Items):
            for item in value.items:
                self.check_callable_argument(item, function)
            return value
        if isinstance(value, _Known) and _is_callable_value(value):
            from arachne.mobject.mobject import Mobject

            target = value.obj
            if library.entry(target) is not None or _is_rate_function(target):
                return VALUE
            if (
                isinstance(target, type)
                and issubclass(target, Mobject)
                and setters.builtin_class(target)
            ):
                # An engine mobject class the engine instantiates
                # (``tip_shape=ArrowSquareTip``).
                return VALUE
            if isinstance(target, types.FunctionType) and not _engine_function(target):
                return self.call_user(target, None, None, {})
            raise Refused(
                f"it passes on {_qualified(target)}, which no library entry admits"
            )
        return value

    def call_lambda(
        self,
        callee: _LambdaValue,
        arguments: list[Any] | None,
        keywords: dict[str | None, Any],
        function: Any,
    ) -> Any:
        node = callee.node
        env = dict(callee.env)
        names = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
        # What the caller passes, unknown, is a value it did not make.
        for index, name in enumerate(names):
            env[name] = (
                arguments[index]
                if arguments is not None and index < len(arguments)
                else EXTERNAL
            )
        for argument in node.args.kwonlyargs:
            env[argument.arg] = keywords.get(argument.arg, EXTERNAL)
        if node.args.vararg:
            env[node.args.vararg.arg] = _Items((EXTERNAL,))
        if node.args.kwarg:
            env[node.args.kwarg.arg] = EXTERNAL
        return self.within(lambda: self.expression(node.body, function, env))

    def call_user(
        self,
        target: types.FunctionType,
        bound: Any,
        arguments: list[Any] | None,
        keywords: dict[str | None, Any],
    ) -> Any:
        node = _function_node(target)
        if isinstance(node, ast.FunctionDef) and (node.args.vararg or node.args.kwarg):
            raise Refused(f"{target.__name__} takes arbitrary arguments")
        env: dict[str, Any] = {}
        names = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
        given = ([bound] if bound is not None else []) + (arguments or [])
        defaults = target.__defaults__ or ()
        first_default = len(names) - len(defaults)
        for index, name in enumerate(names):
            if index < len(given):
                env[name] = given[index]
            elif arguments is None:
                env[name] = EXTERNAL
            elif name in keywords:
                env[name] = keywords[name]
            elif index >= first_default:
                env[name] = self.classify(
                    defaults[index - first_default], "default", name, target
                )
            else:
                raise Refused(f"{target.__name__} is called without {name}")
        for argument in node.args.kwonlyargs:
            if argument.arg in keywords:
                env[argument.arg] = keywords[argument.arg]
            else:
                defaults_ = target.__kwdefaults__ or {}
                env[argument.arg] = (
                    self.classify(
                        defaults_[argument.arg], "default", argument.arg, target
                    )
                    if argument.arg in defaults_
                    else EXTERNAL
                )
        return self.within(lambda: self.run_body(node, target, env, _Role.READOUT))

    def within(self, step: Callable[[], Any]) -> Any:
        if self.depth >= MAX_DEPTH:
            raise Refused("its helpers call each other too deep to follow")
        self.depth += 1
        try:
            return step()
        finally:
            self.depth -= 1

    def call_method(
        self, callee: _Method, values: list[Any] | None, function: Any
    ) -> Any:
        on, name = callee.on, callee.name
        values = values or []
        for value in values:
            self.check_callable_argument(value, function)
        if on is VALUE or on is EXTERNAL:
            form = library.value_method(name)
            if form is None:
                raise Refused(f"it calls {name} on a value, which no entry admits")
            for value in values:
                if not _all_plain(value):
                    raise Refused(f"{name} is given {_describe(value)}")
            return on if form[2] else VALUE
        if isinstance(on, _Known):
            obj = on.obj
            if obj is self.scene:
                method = getattr(type(obj), name)
                if _engine_function(method):
                    raise Refused(f"it calls the scene's {name}")
                return self.call_user(method, on, values, {})
            if obj is getattr(self.scene, "camera", None):
                raise Refused(f"it calls the camera's {name}")
            raise Refused(f"it calls {name} on {obj!r}")
        if isinstance(on, _Scene):
            kinds = {setters.read_kind(root, name) for root in on.roots}
            if None in kinds or len(kinds) != 1:
                raise Refused(
                    f"it calls {name} on {type(on.roots[0]).__name__}, which may change it"
                )
            kind = kinds.pop()
            if any(isinstance(v, _Fresh) or _holds_fresh(v) for v in values):
                raise Refused(f"{name} is given a mobject the callback built")
            if any(isinstance(v, _Scene) or _holds_scene(v) for v in values):
                self.proof.read(_roots(values))
            # A read may call what the mobjects store (a graph's function).
            self.prove_stored_callables([*on.roots, *_roots(values)])
            if kind == "value":
                return VALUE
            if kind == "member":
                return on
            return _Fresh(
                True, setters.returned_class(setters.method_function(on.roots[0], name))
            )
        if isinstance(on, _Fresh):
            cls = on.cls
            if cls is None:
                raise Refused(
                    f"it calls {name} on a mobject of a kind the proof can't tell"
                )
            kind = setters.fresh_method(cls, name)
            if kind is None:
                raise Refused(f"it calls {cls.__name__}.{name}")
            reads = kind.endswith("+reads")
            self.prove_stored_callables(_roots(values))
            for value in values:
                if isinstance(value, _Scene) or _holds_scene(value):
                    if not reads:
                        raise Refused(f"{name} is given a scene mobject it may change")
                elif isinstance(value, _Fresh) and not value.owned:
                    raise Refused(
                        f"{name} is given a mobject that holds scene mobjects"
                    )
            if not on.owned:
                raise Refused(f"it calls {name} on a mobject that holds scene mobjects")
            if not kind.startswith("mobject"):
                return VALUE
            method = setters.method_function(cls, name)
            returned = setters.returned_class(method)
            return _Fresh(True, cls if returned is None else returned)
        raise Refused(f"it calls {name} on {type(on).__name__.strip('_')}")

    def prove_stored_callables(self, mobjects: Iterable[Any]) -> None:
        """Every callable the mobjects (their families included) store, which
        a read of them may call: ``underlying_function`` and ``function`` of a
        graph, a vector field's ``func``. Each must be proven a readout, or be
        a library entry; the engine's own wrappers are proven through what
        they hold (:meth:`prove_engine_closure`).
        """
        for mob in mobjects:
            if id(mob) in self._mobjects_seen:
                continue
            self._mobjects_seen.add(id(mob))
            for member in mob.get_family():
                for key, value in vars(member).items():
                    if _stored_callable(value):
                        name = f"{type(member).__name__}.{key}"
                        try:
                            self.prove_callable(value, name)
                        except Refused as refused:
                            raise Refused(
                                f"a read calls {name}, and {refused}"
                            ) from None

    def prove_callable(self, value: Any, name: str) -> None:
        if id(value) in self._callables_seen:
            return
        self._callables_seen.add(id(value))
        if library.entry(value) is not None or _is_rate_function(value):
            return
        if isinstance(value, types.MethodType):
            receiver, target = value.__self__, value.__func__
            if _engine_function(target):
                self.prove_engine_method(receiver, value.__name__, target, name)
                return
            if not isinstance(target, types.FunctionType):
                raise Refused(f"{name} is a callable the proof can't read")
            self.call_user(target, self.classify(receiver, None, "self"), None, {})
            return
        if isinstance(value, types.FunctionType):
            if _engine_function(value):
                self.prove_engine_closure(value, name)
                return
            self.call_user(value, None, None, {})
            return
        raise Refused(f"{name} is a {type(value).__name__}, which the proof can't read")

    def prove_engine_method(
        self, receiver: Any, method: str, target: Any, name: str
    ) -> None:
        """An engine method bound to its receiver (``line.n2p``): calling it
        reads the receiver, so it must be a read the engine vouches for, and
        the receiver is read, for the dependency order.
        """
        from arachne.mobject.mobject import Mobject

        if not (
            isinstance(receiver, Mobject)
            and setters.read_kind(receiver, method) == "value"
            and setters.method_function(receiver, method) is target
        ):
            raise Refused(
                f"{name} is {type(receiver).__name__}.{method}, which is not a read"
                " the engine vouches for"
            )
        self.proof.read([receiver])
        self.prove_stored_callables([receiver])

    def prove_engine_closure(self, function: Any, name: str) -> None:
        """An engine function a mobject stores (``plot``'s wrapper of the
        user's function): what it holds is what it calls and reads. A
        mobject it holds is read; a callable is proven; plain values and
        containers of them are its data; anything else is refused.
        """
        held = [
            *(c.cell_contents for c in function.__closure__ or () if _filled(c)),
            *(function.__defaults__ or ()),
            *(function.__kwdefaults__ or {}).values(),
        ]
        for item in held:
            self._engine_held(item, name)

    def _engine_held(self, item: Any, name: str) -> None:
        from arachne.mobject.mobject import Mobject

        if isinstance(item, Mobject):
            self.proof.read([item])
            self.prove_stored_callables([item])
        elif isinstance(item, (list, tuple)):
            for inner in item:
                self._engine_held(inner, name)
        elif isinstance(item, dict):
            for inner in item.values():
                self._engine_held(inner, name)
        elif _stored_callable(item):
            self.prove_callable(item, f"{name} (what it calls)")
        elif not (
            item is None
            or _constant(item)
            or (type(item) is np.ndarray and item.dtype != object)
        ):
            raise Refused(f"{name} holds a {type(item).__name__} the proof can't read")

    def _plain(self, value: Any, verb: str) -> None:
        if _is_plain(value):
            return
        if isinstance(value, _Items) and all(_is_plain(item) for item in value.items):
            return
        if isinstance(value, _Scene):
            raise Refused(f"it {verb} a scene mobject itself")
        if isinstance(value, _Fresh):
            raise Refused(f"it {verb} a mobject it built")
        raise Refused(f"it {verb} {_describe(value)}")


_MISSING = object()

#: Attributes of plain values that are values themselves.
_VALUE_ATTRIBUTES = frozenset({"T", "real", "imag", "shape", "size", "ndim", "dtype"})
#: Attributes of an array that are views of it: what it was, a value the
#: callback made or one it read, they are.
_VIEW_ATTRIBUTES = frozenset({"T", "real", "imag"})


def _constant(value: Any) -> bool:
    """Whether ``value`` can't change and runs no code of a scene's when
    computed with: a number, string, colour or engine enum of exactly its
    type (a subclass may define its own arithmetic), or a tuple of them.
    """
    from arachne.utils.color.core import ManimColor

    from .setters import builtin_class

    kind = type(value)
    if value is None or one_of(kind, (bool, int, float, complex, str, bytes, Fraction)):
        return True
    if isinstance(value, np.generic):
        return kind in _NUMPY_SCALARS and value.dtype != object
    if isinstance(value, (ManimColor, Enum)):
        return builtin_class(kind)
    if one_of(kind, (tuple, frozenset)):
        return all(_constant(item) for item in value)
    return False


#: numpy's own scalar types, by identity: the ones numpy names (a class of a
#: scene's may say its ``__module__`` is ``"numpy"``).
_NUMPY_SCALARS = ByIdentity.of(
    value
    for value in vars(np).values()
    if isinstance(value, type) and issubclass(value, np.generic)
)


def _contents(value: Any) -> Any:
    """What a binding's value holds that could change behind it, detached
    and read-only: an array's numbers (whatever its flags: a read-only view
    changes with its base), and a container's items with theirs, nested
    containers included. ``None`` when nothing it holds can change.
    """
    if isinstance(value, np.ndarray):
        copy = np.array(value, copy=True)
        copy.flags.writeable = False
        return copy
    if isinstance(value, (list, tuple)):
        items = tuple(_snapshot(item) for item in value)
        if isinstance(value, tuple) and all(
            a is b for a, b in zip(items, value, strict=True)
        ):
            return None
        return items
    if isinstance(value, dict):
        return tuple((key, _snapshot(item)) for key, item in value.items())
    return None


def _snapshot(value: Any) -> Any:
    contents = _contents(value)
    return value if contents is None else contents


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return False
    return True


def _stored_callable(value: Any) -> bool:
    from arachne.mobject.mobject import Mobject

    return callable(value) and not isinstance(value, (type, Mobject, np.ndarray))


def _check_form(callee: Any, node: ast.Call) -> None:
    """Refuse a call to a library entry, or to a method of a plain value,
    outside its admitted form (:mod:`.library`): more arguments by position
    than it admits (an output array, or options its contract does not
    cover), a keyword it does not admit (``out=``, ``where=``), or a
    ``dtype`` other than a number type written out.
    """
    starred = any(isinstance(a, ast.Starred) for a in node.args)
    keywords: frozenset[str] | None
    dtype_at: int | None
    if isinstance(callee, _Known):
        found = library.entry(callee.obj)
        if found is None:
            return
        name, positional, keywords = found.name, found.positional, found.keywords
        dtype_at = found.dtype_at
        if starred and (positional is not None or found.takes_mobjects or found.calls):
            raise Refused(f"it unpacks arguments into {name}")
    elif isinstance(callee, _Method) and _is_plain(callee.on):
        form = library.value_method(callee.name)
        if form is None:
            return
        positional, keywords, _ = form
        name = callee.name
        dtype_at = 0 if name == "astype" else None
        if starred and positional is not None:
            raise Refused(f"it unpacks arguments into {name}")
    else:
        return
    if positional is not None and len(node.args) > positional:
        raise Refused(
            f"{name} is given {len(node.args)} arguments by position, past what its"
            " entry admits (an output array, or options its contract does not cover)"
        )
    for keyword in node.keywords:
        if keywords is not None and keyword.arg not in keywords:
            raise Refused(
                f"{name} is given {keyword.arg}=, which its entry does not admit"
            )
    dtypes = [k.value for k in node.keywords if k.arg == "dtype"]
    if dtype_at is not None and len(node.args) > dtype_at:
        dtypes.append(node.args[dtype_at])
    for dtype in dtypes:
        _check_dtype(name, dtype)


def _check_dtype(name: str, node: ast.expr) -> None:
    """Refuse a ``dtype`` other than a number type written out: an array of
    objects holds whatever it is given, None included.
    """
    value = node.value if isinstance(node, ast.Constant) else _MISSING
    if value is None:
        return
    try:
        kind = np.dtype(value).kind if isinstance(value, str) else None
    except TypeError:
        kind = None
    if kind is None or kind not in "biufc":
        raise Refused(
            f"{name} is given dtype={ast.unparse(node)}: the proof admits a number"
            ' type written as a string ("float"), never an array of objects,'
            " which holds whatever it is given"
        )


def _is_callable_value(value: Any) -> bool:
    if isinstance(value, (_LambdaValue, _Method)):
        return True
    return isinstance(value, _Known) and callable(value.obj)


def _container_result(target: Any, arguments: list[Any], given: list[Any]) -> Any:
    """What ``len``, ``isinstance`` and the container builtins return, given
    mobjects or values: a count or a truth, or a container of what they were
    given.
    """
    for value in given:
        if (
            isinstance(value, (_Known, _LambdaValue, _Method))
            and target is not isinstance
        ):
            raise Refused(f"{_qualified(target)} is given {_describe(value)}")
    if target in (len, isinstance):
        return VALUE
    if target is zip:
        return _Items((_Items(tuple(_element(a) for a in arguments), exact=True),))
    if target is enumerate:
        return _Items((_Items((VALUE, _element(arguments[0])), exact=True),))
    return _Items(tuple(_element(a) for a in arguments[:1]))


def _join(values: Iterable[Any]) -> Any:
    """The abstract value that stands for any of ``values``: one picked by
    an index, a branch or a loop. A value that may be None is None; one that
    may be an array the callback did not make is one; mobjects are any of
    them. Values of different kinds are refused.
    """
    values = list(values)
    if not values:
        return VALUE
    if all(_is_plain(v) for v in values):
        for plain in (NONE, EXTERNAL):
            if any(v is plain for v in values):
                return plain
        return VALUE
    if all(isinstance(v, _Scene) for v in values):
        roots: list[Any] = []
        for value in values:
            roots.extend(r for r in value.roots if not any(r is s for s in roots))
        return _Scene(tuple(roots))
    if all(isinstance(v, _Fresh) for v in values):
        classes = {v.cls for v in values}
        return _Fresh(
            all(v.owned for v in values), classes.pop() if len(classes) == 1 else None
        )
    if all(isinstance(v, _Items) for v in values):
        return _Items(tuple(item for value in values for item in value.items))
    raise Refused(
        "it picks one of things of different kinds ("
        + ", ".join(sorted({_describe(v) for v in values}))
        + ")"
    )


def _element(value: Any) -> Any:
    """The abstract value of an item of ``value`` (iterated or unpacked)."""
    if isinstance(value, _Items):
        return _join(value.items)
    if isinstance(value, (_Scene, _Fresh)) or value is VALUE or value is EXTERNAL:
        return value
    raise Refused(f"it iterates {_describe(value)}")


def _all_plain(value: Any) -> bool:
    """Whether ``value`` is a plain value or a container of them only."""
    if _is_plain(value):
        return True
    return isinstance(value, _Items) and all(_all_plain(item) for item in value.items)


def _flat_plain(value: Any) -> Any:
    """The join of the plain values ``value`` is or holds."""
    if isinstance(value, _Items):
        return _join([_flat_plain(item) for item in value.items] or [VALUE])
    return value


def _describe(value: Any) -> str:
    if value is NONE:
        return "None"
    if value is EXTERNAL or value is VALUE:
        return "a value"
    if isinstance(value, _Scene):
        return "a scene mobject"
    if isinstance(value, _Fresh):
        return "a mobject it built"
    if isinstance(value, _Items):
        return "a container"
    if isinstance(value, _Known):
        return f"{_qualified(value.obj)}"
    if isinstance(value, _LambdaValue):
        return "a lambda"
    if isinstance(value, _Method):
        return f"the method {value.name}"
    return type(value).__name__.strip("_")


def _holds_none(value: Any) -> bool:
    return isinstance(value, _Items) and any(
        item is NONE or _holds_none(item) for item in value.items
    )


def _holds_scene(value: Any) -> bool:
    return isinstance(value, _Items) and any(
        isinstance(item, _Scene) or _holds_scene(item) for item in value.items
    )


def _holds_fresh(value: Any) -> bool:
    return isinstance(value, _Items) and any(
        isinstance(item, _Fresh) or _holds_fresh(item) for item in value.items
    )


def _all_owned(value: Any) -> bool:
    if isinstance(value, _Fresh):
        return value.owned
    if isinstance(value, _Items):
        return all(_all_owned(item) for item in value.items)
    return True


def _roots(values: Iterable[Any]) -> tuple[Any, ...]:
    found: list[Any] = []
    for value in values:
        if isinstance(value, _Scene):
            found.extend(value.roots)
        elif isinstance(value, _Items):
            found.extend(_roots(value.items))
    return tuple(found)


def _engine_function(function: Any) -> bool:
    """Whether ``function`` is the engine's own code, by its code object
    (:func:`.engine_code`, which compares code by equality): no
    ``__module__`` or ``functools.wraps`` makes a scene's function the
    engine's.
    """
    return engine_code(function)


def _is_rate_function(function: Any) -> bool:
    from .rates import admitted_rate

    return admitted_rate(function)


def _name(target: Any) -> str:
    return str(
        getattr(target, "__qualname__", None)
        or getattr(target, "__name__", repr(target))
    )


def _qualified(target: Any) -> str:
    """``target``'s module and name (``numpy.put_along_axis``)."""
    module = getattr(target, "__module__", None)
    if isinstance(target, np.ufunc):
        module = "numpy"
    name = _name(target)
    return f"{module}.{name}" if module and module != "builtins" else name


# Entry points


def prove_setter_body(
    function: Callable[..., Any],
    receiver: Any,
    scene: Any,
    *,
    takes_alpha: bool = False,
) -> Proof:
    """Prove that ``function(receiver)`` (``function(receiver, alpha)`` with
    ``takes_alpha``) sets ``receiver`` and scene mobjects it names through
    admitted setters only, from registered state.
    """
    proof = Proof()
    prover = _Prover(scene, proof)
    prover.setter_mode = True
    node = _function_node(function)
    names = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
    expected = 2 if takes_alpha else 1
    if len(names) < expected:
        raise Refused(f"it takes {len(names)} arguments")
    env: dict[str, Any] = {names[0]: _Scene((receiver,))}
    if takes_alpha:
        env[names[1]] = VALUE
    defaults = function.__defaults__ or ()
    for index, name in enumerate(names[expected:], start=expected):
        offset = index - (len(names) - len(defaults))
        if offset < 0:
            raise Refused(f"it takes an argument {name} nobody passes")
        env[name] = prover.classify(defaults[offset], "default", name, function)
    prover.run_body(node, function, env, _Role.SETTER)
    if not proof.writes:
        raise Refused("it sets nothing")
    return proof


def prove_builder(function: Callable[..., Any], scene: Any) -> Proof:
    """Prove that ``function()`` builds and returns a mobject of its own from
    registered state.
    """
    proof = Proof()
    prover = _Prover(scene, proof)
    if isinstance(function, types.MethodType):
        target = function.__func__
        on = prover.classify(function.__self__, None, "self")
        if _engine_function(target) or not isinstance(target, types.FunctionType):
            # The engine's method (``ax.copy``): one of the reads it vouches for.
            result = prover.call_method(_Method(on, function.__name__), [], function)
        else:
            result = prover.call_user(target, on, [], {})
    else:
        node = _function_node(function)
        if node.args.args or node.args.posonlyargs:
            raise Refused("it takes arguments")
        result = prover.run_body(node, function, {}, _Role.BUILDER)
    if not (isinstance(result, _Fresh) and result.owned):
        raise Refused("it does not return a mobject it built")
    return proof


def prove_readout(function: Callable[..., Any], scene: Any) -> Proof:
    """Prove that ``function()`` returns a value read from registered state
    and changes nothing.
    """
    proof = Proof()
    prover = _Prover(scene, proof)
    if isinstance(function, types.MethodType):
        on = prover._classify(function.__self__, "self")
        if isinstance(on, _Scene):
            proof.read(on.roots)
        result = prover.call_method(_Method(on, function.__name__), [], function)
    else:
        node = _function_node(function)
        if node.args.args or node.args.posonlyargs:
            raise Refused("it takes arguments")
        result = prover.run_body(node, function, {}, _Role.READOUT)
    if not _all_plain(result) or result is NONE or _holds_none(result):
        raise Refused("it does not return a plain value")
    return proof
