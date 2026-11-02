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
values are recorded), the scene clock, immutable constants and the functions
the engine vouches for; and it may change nothing but the outputs above. A
``nonlocal`` or ``global`` write, an attribute or item assignment, a call
that changes a scene mobject other than an admitted setter, a relative
change to the mobject it writes (``shift``, ``rotate``, a setter whose
argument reads the receiver), a random draw, I/O or a clock other than the
scene's, are refusals. The proof records what the callback reads and
writes, for the dependency order, and the bindings it read its names from,
which route 2's restore (M5) compares with what they hold then.
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
from numbers import Number
from typing import Any

import numpy as np

from . import setters

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
        for mob in mobjects:
            if not any(mob is seen for seen in self.reads):
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
    """A container whose items are these abstract values."""

    items: tuple[Any, ...]


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
    positions = [
        (pos.lineno, pos.col_offset)
        for instruction in dis.get_instructions(code)
        if instruction.opname not in _PROLOGUE
        and (pos := instruction.positions) is not None
        and pos.lineno is not None
        and pos.col_offset is not None
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
    candidates.sort(key=_span)
    if len(candidates) > 1 and _span(candidates[0]) == _span(candidates[1]):
        raise Refused(f"the source of {code.co_name} is ambiguous")
    return candidates[0]


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
            if value.dtype == object:
                raise Refused(f"{name} is an array of objects")
            return EXTERNAL
        if _constant(value):
            return VALUE
        if isinstance(value, types.ModuleType):
            if not setters.pure_module(value):
                raise Refused(f"{name} is the module {value.__name__}")
            return _Known(value)
        if isinstance(value, (list, tuple)):
            return _Items(tuple(self._classify(item, f"{name}[]") for item in value))
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
            for element in target.elts:
                self.assign(element, _element(value), env)
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
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
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
            for argument in arguments:
                value = self.expression(argument, function, env)
                if value is NONE or _holds_none(value):
                    raise Refused(
                        f"{method} is given None, which leaves what it sets as it was"
                    )
                if isinstance(value, (_Fresh, _LambdaValue)) or (
                    isinstance(value, _Known) and callable(value.obj)
                ):
                    raise Refused(
                        f"{method} is given a {type(value).__name__.strip('_')}"
                    )
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
            return VALUE
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
                return _Scene(tuple(value)) if value else VALUE
            if isinstance(value, np.ndarray) and value.dtype != object:
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
            return _element(on)
        if on is VALUE or on is EXTERNAL:
            # An index of an array it read is a view of it.
            return on
        raise Refused(f"it indexes {type(on).__name__.strip('_')}")

    def _Slice(self, node: ast.Slice, function: Any, env: dict[str, Any]) -> Any:
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                self._plain(self.expression(part, function, env), "slices with")
        return VALUE

    def _Starred(self, node: ast.Starred, function: Any, env: dict[str, Any]) -> Any:
        return self.expression(node.value, function, env)

    def _BinOp(self, node: ast.BinOp, function: Any, env: dict[str, Any]) -> Any:
        self._plain(self.expression(node.left, function, env), "computes with")
        self._plain(self.expression(node.right, function, env), "computes with")
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
        if isinstance(body, _Fresh) and isinstance(orelse, _Fresh):
            return _Fresh(body.owned and orelse.owned)
        if _is_plain(body) and _is_plain(orelse):
            for plain in (EXTERNAL, NONE):
                if plain in (body, orelse):
                    return plain
            return VALUE
        raise Refused("a conditional chooses between things that are not plain values")

    def _Tuple(self, node: ast.Tuple, function: Any, env: dict[str, Any]) -> Any:
        return _Items(tuple(self.expression(e, function, env) for e in node.elts))

    _List = _Tuple

    def _Set(self, node: ast.Set, function: Any, env: dict[str, Any]) -> Any:
        for element in node.elts:
            self._plain(self.expression(element, function, env), "collects")
        return VALUE

    def _Dict(self, node: ast.Dict, function: Any, env: dict[str, Any]) -> Any:
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                self._plain(self.expression(key, function, env), "keys with")
            self._plain(self.expression(value, function, env), "maps to")
        return VALUE

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
        self._plain(self.expression(node.value, function, inner), "maps to")
        return VALUE

    def _Lambda(self, node: ast.Lambda, function: Any, env: dict[str, Any]) -> Any:
        return _LambdaValue(node, dict(env))

    def _Call(self, node: ast.Call, function: Any, env: dict[str, Any]) -> Any:
        callee = self.expression(node.func, function, env)
        _check_outputs(callee, node)
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
        if setters.pure_function(target):
            for value in values:
                self.check_callable_argument(value, function)
                if (
                    isinstance(value, _Fresh) or _holds_fresh(value)
                ) and target not in (
                    len,
                    isinstance,
                    list,
                    tuple,
                    reversed,
                    zip,
                    enumerate,
                ):
                    raise Refused(f"{_name(target)} is given a mobject")
            if any(isinstance(v, _Scene) or _holds_scene(v) for v in values):
                return _Scene(_roots(values))
            if target in (
                list,
                tuple,
                reversed,
                zip,
                enumerate,
                sorted,
                filter,
            ) and any(isinstance(v, _Fresh) or _holds_fresh(v) for v in values):
                return _Items(
                    tuple(v for v in values if isinstance(v, (_Fresh, _Items)))
                )
            if any(v is EXTERNAL or _holds_external(v) for v in values) and not (
                _allocates(target)
            ):
                # It may hand back what it was given, or a view of it.
                return EXTERNAL
            return VALUE
        if isinstance(target, types.FunctionType) and not _engine_function(target):
            return self.call_user(target, None, arguments, keywords)
        if isinstance(target, types.FunctionType) and _is_rate_function(target):
            return VALUE
        raise Refused(f"it calls {_name(target)}, which the engine does not vouch for")

    def check_callable_argument(self, value: Any, function: Any) -> None:
        """A callable passed on (``plot(f)``) must itself be a readout."""
        if isinstance(value, _LambdaValue):
            self.call_lambda(value, None, {}, function)
        elif isinstance(value, _Known) and isinstance(value.obj, types.FunctionType):
            target = value.obj
            if setters.pure_function(target) or _is_rate_function(target):
                return
            if _engine_function(target):
                raise Refused(f"it passes on {_name(target)}")
            self.call_user(target, None, None, {})
        elif isinstance(value, _Method):
            self.call_method(value, None, function)

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
        for index, name in enumerate(names):
            env[name] = (
                arguments[index]
                if arguments is not None and index < len(arguments)
                else VALUE
            )
        for argument in node.args.kwonlyargs:
            env[argument.arg] = keywords.get(argument.arg, VALUE)
        if node.args.vararg:
            env[node.args.vararg.arg] = VALUE
        if node.args.kwarg:
            env[node.args.kwarg.arg] = VALUE
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
                env[name] = VALUE
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
                default = (target.__kwdefaults__ or {}).get(argument.arg, VALUE)
                env[argument.arg] = (
                    default
                    if default is VALUE
                    else self.classify(default, "default", argument.arg, target)
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
            if name in _VALUE_METHODS:
                if any(isinstance(v, (_Scene, _Fresh)) for v in values):
                    raise Refused(f"{name} is given a mobject")
                return on if name in _VIEW_METHODS else VALUE
            raise Refused(f"it calls {name} on a value")
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
        a pure library function; the engine's own wrappers are vouched for and
        what they close over is proven in turn.
        """
        for mob in mobjects:
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
        if setters.pure_function(value) or _is_rate_function(value):
            return
        if isinstance(value, types.MethodType):
            target = value.__func__
            if _engine_function(target) and setters.builtin_class(type(value.__self__)):
                return
            if not isinstance(target, types.FunctionType):
                raise Refused(f"{name} is a callable the proof can't read")
            self.call_user(target, self._classify(value.__self__, "self"), None, {})
            return
        if isinstance(value, types.FunctionType):
            if _engine_function(value):
                inner = [
                    *(c.cell_contents for c in value.__closure__ or () if _filled(c)),
                    *(value.__defaults__ or ()),
                ]
                for item in inner:
                    if _stored_callable(item):
                        self.prove_callable(item, f"{name} (what it calls)")
                return
            self.call_user(value, None, None, {})
            return
        raise Refused(f"{name} is a {type(value).__name__}, which the proof can't read")

    def _plain(self, value: Any, verb: str) -> None:
        if _is_plain(value):
            return
        if isinstance(value, _Items) and all(_is_plain(item) for item in value.items):
            return
        if isinstance(value, _Scene):
            raise Refused(f"it {verb} a scene mobject itself")
        if isinstance(value, _Fresh):
            raise Refused(f"it {verb} a mobject it built")
        if isinstance(value, _Known) and not callable(value.obj):
            return
        raise Refused(f"it {verb} a {type(value).__name__.strip('_')}")


_MISSING = object()

#: Attributes of plain values that are values themselves.
_VALUE_ATTRIBUTES = frozenset({"T", "real", "imag", "shape", "size", "ndim", "dtype"})
#: Attributes and methods of an array that are views of it: what it was, a
#: value the callback made or one it read, they are.
_VIEW_ATTRIBUTES = frozenset({"T", "real", "imag"})
_VIEW_METHODS = frozenset({"reshape", "T", "real", "imag", "astype"})
#: Methods of plain values that write into an array given at this position
#: (``out``), counted from the first argument after the receiver.
_OUT_POSITIONS = {
    "dot": 1,
    "clip": 2,
    "round": 1,
    "sum": 2,
    "mean": 2,
    "max": 1,
    "min": 1,
}

#: Methods of plain values (arrays, numbers, strings) with no effect beyond
#: their result.
_VALUE_METHODS = frozenset(
    {
        "copy",
        "dot",
        "sum",
        "mean",
        "max",
        "min",
        "reshape",
        "flatten",
        "tolist",
        "astype",
        "round",
        "clip",
        "conjugate",
        "real",
        "imag",
        "format",
        "upper",
        "lower",
        "strip",
        "split",
        "join",
        "replace",
        "item",
        "T",
        "__abs__",
    }
)


def _constant(value: Any) -> bool:
    """Whether ``value`` can't change: a number, string, colour or other
    immutable value, or a tuple of them.
    """
    from arachne.utils.color.core import ManimColor

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (Fraction, Number, np.generic, ManimColor, Enum)):
        return True
    if isinstance(value, np.ndarray):
        return not value.flags.writeable and value.dtype != object
    if isinstance(value, (tuple, frozenset)):
        return all(_constant(item) for item in value)
    return False


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


def _allocates(target: Any) -> bool:
    """Whether calling ``target`` always returns a new value, never one of
    its arguments or a view of one.
    """
    import math

    if isinstance(target, np.ufunc):
        return True
    if target in (
        int,
        float,
        complex,
        bool,
        len,
        abs,
        round,
        str,
        sum,
        any,
        all,
        pow,
        divmod,
        isinstance,
    ):
        return True
    return getattr(target, "__module__", None) == math.__name__


def _check_outputs(callee: Any, node: ast.Call) -> None:
    """Refuse a call that may write into an array it is given: ``out=`` or
    ``where=`` (which leaves outputs unassigned), or an output array passed
    positionally to a numpy function or an array method.
    """
    names = {keyword.arg for keyword in node.keywords}
    if names & {"out", "where"}:
        raise Refused("it passes out= or where=, which write into an array")
    positional = len(node.args)
    starred = any(isinstance(a, ast.Starred) for a in node.args)
    if isinstance(callee, _Known):
        target = callee.obj
        if isinstance(target, np.ufunc):
            if starred or positional > target.nargs - target.nout:
                raise Refused(f"{_name(target)} is given an output array")
            return
        module = getattr(target, "__module__", None) or ""
        if module.startswith("numpy"):
            index = _out_index(target)
            if starred or (index is not None and positional > index):
                raise Refused(f"{_name(target)} is given an output array")
    elif isinstance(callee, _Method) and _is_plain(callee.on):
        position = _OUT_POSITIONS.get(callee.name)
        if position is not None and (starred or positional > position):
            raise Refused(f"{callee.name} is given an output array")


def _out_index(target: Any) -> int | None:
    """The position of ``target``'s ``out`` parameter: ``None`` when it has
    none, 1 (anything after the first argument) when its signature can't be
    read.
    """
    try:
        parameters = list(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return 1
    return parameters.index("out") if "out" in parameters else None


def _element(value: Any) -> Any:
    """The abstract value of an item of ``value`` (iterated or unpacked)."""
    if isinstance(value, _Items):
        if not value.items:
            return VALUE
        scenes = [item for item in value.items if isinstance(item, _Scene)]
        if scenes:
            return _Scene(tuple(root for item in scenes for root in item.roots))
        fresh = [item for item in value.items if isinstance(item, _Fresh)]
        if fresh:
            return _Fresh(all(f.owned for f in fresh), fresh[0].cls)
        nested = [item for item in value.items if isinstance(item, _Items)]
        if nested:
            return nested[0]
        if any(item is EXTERNAL for item in value.items):
            return EXTERNAL
        return VALUE
    if isinstance(value, (_Scene, _Fresh)) or value is EXTERNAL:
        return value
    return VALUE


def _holds_none(value: Any) -> bool:
    return isinstance(value, _Items) and any(
        item is NONE or _holds_none(item) for item in value.items
    )


def _holds_external(value: Any) -> bool:
    return isinstance(value, _Items) and any(
        item is EXTERNAL or _holds_external(item) for item in value.items
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
    """Whether ``function`` is the engine's own code."""
    module = getattr(function, "__module__", None) or ""
    return module == "arachne" or module.startswith("arachne.")


def _is_rate_function(function: Any) -> bool:
    from .rates import admitted_rate

    return admitted_rate(function)


def _name(target: Any) -> str:
    return str(
        getattr(target, "__qualname__", None)
        or getattr(target, "__name__", repr(target))
    )


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
        assert isinstance(target, types.FunctionType)
        result = prover.call_user(
            target, prover._classify(function.__self__, "self"), [], {}
        )
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
    if not (result is VALUE or result is EXTERNAL) and not (
        isinstance(result, _Items)
        and all(item is VALUE or item is EXTERNAL for item in result.items)
    ):
        raise Refused("it does not return a plain value")
    return proof
