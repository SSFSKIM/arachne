"""Which work of a ticked play runs at a tick no frame reads: the stepping tick.

A ticked play (route 3) is evaluated on the 60 Hz tick, so that what
integrates over time does so the same way whatever the frame rate. Only that
work needs the tick (a dt updater receives exactly 1/60 at each), and
everything else is evaluated where CE evaluates it, at the instants a frame
reads. So a ticked
play runs a **full tick** (every callback, in CE's phase order) at each
instant a frame of the render's grid reads, at its event instants, at its
last tick and throughout a play with a stop condition, and every other tick
only **steps**:

- the updaters that integrate: those with a ``dt`` parameter whose code
  loads it (one that never loads it is dt-free, as it is frame-rate
  dependent in CE exactly like one without the parameter);
- the scene's own updaters;
- every animation's interpolation, the functions of ``UpdateFromFunc`` and
  ``UpdateFromAlphaFunc`` included;
- the camera's inputs (as every tick evaluates them);
- and the closure of the dt-free callbacks those may read, in either phase
  order (a reader that runs before its producer reads the previous tick's
  value, so the producer must have run there too).

The principle: a callback runs at a stepping tick only if a stepping
callback (a dt updater that reads ``dt``, a scene updater, a camera input)
reads, directly or transitively, what it writes; its own reads matter only
when it runs there. Interpolation always runs, but what it reads joins the
closure only when stepping work reads what it writes (its mobject's family).

What a callback may read is bounded by what its code can reach
(:func:`reach`): its closure cells, the globals and attributes its code
names, its defaults, the object a bound method holds, the mobject it updates
with its family, and what those hold (``TracedPath``'s
``traced_point_func`` included). What a dt-free callback, or the function of
an ``UpdateFromFunc``/``UpdateFromAlphaFunc``, may write is the mobject it
updates (``always_redraw``'s target), unless its source shows a change
anywhere else (``_Writes``): then it writes everything it reaches, which
stepping work may read whenever the play has any. The play ticks everything,
under the reason ``tick.unbounded_read``, when a reach that matters can't be
bounded (dynamic attribute access, more objects than the bound follows): a
stepping callback's, one pulled into the closure, or one whose writes are
everything it reaches in a play with stepping work. An unbounded read of a
callback nothing pulls, whose writes are bounded, leaves the play stepping.

:class:`TickPlan` is the outcome; :func:`skipping` installs its skipped
updaters for one stepping tick, which :meth:`.Mobject.update` consults.
"""

from __future__ import annotations

__all__ = [
    "UNBOUNDED",
    "TickPlan",
    "loads_dt",
    "plan_ticks",
    "reach",
    "skipping",
]

import ast
import builtins
import contextlib
import dis
import functools
import inspect
import types
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .frozen import engine_class, engine_code
from .source import _internal, relative_path

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

#: The reason family of a play that ticks everything because a callback's
#: reads can't be bounded.
UNBOUNDED = "tick.unbounded_read"

#: How many objects a reach follows before it gives up.
REACH_LIMIT = 50_000

#: How deep the write check follows calls into the scene's helpers.
MAX_DEPTH = 4

#: Names whose use reads or writes by a name the code does not spell.
_DYNAMIC = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "locals",
        "globals",
        "eval",
        "exec",
        "compile",
        "__dict__",
        "__import__",
    }
)


@dataclass(frozen=True)
class TickPlan:
    """What a ticked play skips at its stepping ticks."""

    #: ``(id(mobject), id(updater))`` of every updater a stepping tick skips.
    skipped: frozenset[tuple[int, int]] = frozenset()
    #: Every updater the plan weighed: one it did not (a mobject a callback
    #: added, a copy a part made as it began) was not in its closure.
    known: frozenset[tuple[int, int]] = frozenset()
    #: Why the play ticks everything, when it does (``tick.unbounded_read: …``).
    unbounded: str | None = None
    #: How many updaters step, and how many dt-free ones step because
    #: stepping work may read them.
    stepping: int = 0
    pulled: int = 0

    @property
    def steps_only(self) -> bool:
        """Whether a stepping tick does less than a full one."""
        return self.unbounded is None and bool(self.skipped)


@contextlib.contextmanager
def skipping(plan: TickPlan) -> Iterator[None]:
    """Run a stepping tick: :meth:`.Mobject.update` leaves out ``plan``'s
    skipped updaters and flags one the plan does not know
    (:func:`met_unforeseen`).
    """
    from arachne.mobject.mobject import Mobject

    Mobject._tick_skips = plan.skipped
    Mobject._tick_known = plan.known
    try:
        yield
    finally:
        Mobject._tick_skips = None


def met_unforeseen() -> bool:
    """Whether a stepping tick ran an updater the plan did not weigh (and
    forget it).
    """
    from arachne.mobject.mobject import Mobject

    met = Mobject._tick_unforeseen
    Mobject._tick_unforeseen = False
    return met


def current_keys(
    scene: Scene, animations: Iterable[Animation]
) -> frozenset[tuple[int, int]]:
    """Every updater the scene's tick may run now."""
    return frozenset(
        (id(member), id(updater))
        for member, updater in _updaters((*_copies(animations), *scene.mobjects))
    )


# Which updaters integrate


def _code_of(function: Any) -> types.CodeType | None:
    code = getattr(function, "__code__", None)
    if code is None:
        code = getattr(getattr(function, "__func__", None), "__code__", None)
    return code if isinstance(code, types.CodeType) else None


def takes_dt(updater: Any) -> bool:
    """Whether ``updater`` has a ``dt`` parameter, as :meth:`.Mobject.update`
    decides which form to call.
    """
    try:
        return "dt" in inspect.signature(updater).parameters
    except (TypeError, ValueError):
        return True


@functools.cache
def _code_loads(code: types.CodeType, name: str) -> bool:
    if name in code.co_cellvars:
        return True  # an inner function reads it
    if "locals" in code.co_names:
        return True
    for instruction in dis.get_instructions(code):
        # Any use of the local but a plain store or delete may read it (the
        # superinstructions pair a load with a store: STORE_FAST_LOAD_FAST).
        opname = instruction.opname
        if opname != "LOAD_DEREF" and ("FAST" not in opname or opname in _NOT_READS):
            continue
        value = instruction.argval
        if value == name or (isinstance(value, tuple) and name in value):
            return True
    return False


_NOT_READS = frozenset({"STORE_FAST", "DELETE_FAST"})


def loads_dt(updater: Any) -> bool:
    """Whether ``updater`` integrates: it has a ``dt`` parameter and its
    code reads it. One without Python code (a builtin, a callable object)
    counts as reading it.
    """
    if not takes_dt(updater):
        return False
    code = _code_of(updater)
    if code is None:
        return True
    return _code_loads(code, "dt")


# Reach: what a callback may read


class Unbounded(Exception):
    """The reach can't be bounded; the message says why, of ``code`` when
    a function's code is the cause.
    """

    def __init__(self, why: str, code: types.CodeType | None = None) -> None:
        super().__init__(why)
        self.code = code

    def describe(self, base_dir: str | None) -> str:
        code = self.code
        if code is None:
            return str(self)
        where = relative_path(code.co_filename, base_dir)
        return f"{code.co_name} at {where}:{code.co_firstlineno} {self}"


def _user_code(code: types.CodeType) -> bool:
    return not _internal(code.co_filename)


def _names_of(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names_of(const)
    return names


_IMMUTABLE = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    Fraction,
    range,
    slice,
    type(Ellipsis),
    np.generic,
    np.dtype,
    # The C slots of builtin types: they hold nothing.
    type(int.__add__),
    type((0).__add__),
)


def _user_methods(cls: type) -> dict[str, types.FunctionType]:
    """The functions a scene's own class and its own bases define, by name
    (the nearest definition), properties' getters included; none for the
    engine's classes.
    """
    try:
        return _USER_METHODS[cls]
    except (KeyError, TypeError):
        pass
    found: dict[str, types.FunctionType] = {}
    for klass in reversed(cls.__mro__):
        if engine_class(klass) or klass is object:
            continue
        for name, member in vars(klass).items():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if isinstance(member, property):
                member = member.fget
            if isinstance(member, types.FunctionType):
                found[name] = member
    with contextlib.suppress(TypeError):
        _USER_METHODS[cls] = found
    return found


_USER_METHODS: weakref.WeakKeyDictionary[type, dict[str, types.FunctionType]] = (
    weakref.WeakKeyDictionary()
)


class _Reach:
    def __init__(self) -> None:
        self.ids: set[int] = set()
        self.names: set[str] = set()
        self._seen: set[int] = set()
        self._codes: set[int] = set()
        #: Objects followed only through the attributes code names: revisited
        #: as more names turn up.
        self._filtered: dict[int, tuple[Any, frozenset[str]]] = {}
        self._count = 0

    def done(self) -> frozenset[int]:
        while True:
            pending = [
                obj for obj, names in self._filtered.values() if not self.names <= names
            ]
            if not pending:
                return frozenset(self.ids)
            for obj in pending:
                self._attributes(obj)

    def _tick(self) -> None:
        self._count += 1
        if self._count > REACH_LIMIT:
            raise Unbounded(f"it reaches more than {REACH_LIMIT} objects")

    def visit(self, value: Any) -> None:
        from arachne.animation.animation import Animation
        from arachne.camera.camera import Camera
        from arachne.mobject.mobject import Mobject
        from arachne.scene.scene import Scene

        if isinstance(value, _IMMUTABLE):
            return
        key = id(value)
        if key in self._seen:
            return
        self._seen.add(key)
        self._tick()
        if isinstance(value, Mobject):
            self.mobject(value)
        elif isinstance(value, Scene):
            self._attributes(value)
        elif isinstance(value, (types.FunctionType, types.MethodType)):
            self.callable(value)
        elif isinstance(value, functools.partial):
            self.ids.add(key)
            self.visit(value.func)
            for item in (*value.args, *value.keywords.values()):
                self.visit(item)
        elif isinstance(value, types.BuiltinFunctionType):
            owner = getattr(value, "__self__", None)
            if owner is not None and not isinstance(owner, types.ModuleType):
                self.visit(owner)
        elif isinstance(value, types.CellType):
            self.ids.add(key)
            with contextlib.suppress(ValueError):  # an empty cell
                self.visit(value.cell_contents)
        elif isinstance(value, types.ModuleType):
            filename = getattr(value, "__file__", None)
            if filename and not _internal(filename):
                self._attributes(value)
        elif isinstance(value, type):
            if not _internal(getattr(inspect.getmodule(value), "__file__", "") or ""):
                self.ids.add(key)  # its class attributes are shared state
                for member in vars(value).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if isinstance(member, (types.FunctionType, property)):
                        self.visit(
                            member.fget if isinstance(member, property) else member
                        )
        elif isinstance(value, np.ndarray):
            if value.flags.writeable:
                self.ids.add(key)
            if value.dtype == object:
                for item in value.flat:
                    self.visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if not isinstance(value, (tuple, frozenset)):
                self.ids.add(key)
            for item in value:
                self.visit(item)
        elif isinstance(value, dict):
            self.ids.add(key)
            for k, item in value.items():
                self.visit(k)
                self.visit(item)
        elif isinstance(value, Animation):
            self.ids.add(key)
            for item in vars(value).values():
                self.visit(item)
        elif isinstance(value, Camera):
            for item in vars(value).values():
                if isinstance(item, Mobject):
                    self.visit(item)
        elif isinstance(value, property):
            self.visit(value.fget)
        elif engine_class(type(value)):
            return  # the engine's values (colours, paints, frozen arrays)
        else:
            # A plain object (the scene's own helper, a generator, a random
            # state): shared state, followed through the attributes code names.
            self.ids.add(key)
            self._attributes(value)

    def mobject(self, mob: Mobject) -> None:
        for member in mob.get_family():
            self.ids.add(id(member))
            self._seen.add(id(member))
        for member in mob.get_family():
            for name, item in vars(member).items():
                if name in ("updaters", "submobjects"):
                    continue
                self.visit(item)
            if not engine_class(type(member)):
                self._class_code(type(member))

    def _class_code(self, cls: type) -> None:
        """A user class's methods: what its instances' methods may read."""
        for function in _user_methods(cls).values():
            self.visit(function)

    def _attributes(self, obj: Any) -> None:
        """Follow the attributes of ``obj`` that the code reached so far names,
        those it has not followed yet.
        """
        from arachne.scene.scene import Scene

        done = self._filtered.get(id(obj), (obj, frozenset()))[1]
        names = self.names - done
        self._filtered[id(obj)] = (obj, frozenset(self.names))
        try:
            own = vars(obj)
        except TypeError:
            own = {}
            for name in getattr(type(obj), "__slots__", ()):
                if hasattr(obj, name):
                    own[name] = getattr(obj, name)
        for name in names:
            if name in own:
                self.visit(own[name])
        if isinstance(obj, Scene):
            if names & {"camera", "renderer"}:
                self.visit(obj.renderer.camera)
            if names & {
                "mobjects",
                "foreground_mobjects",
                "get_mobject_family_members",
            }:
                self.visit(list(obj.mobjects))
                self.visit(list(obj.foreground_mobjects))
            if "updaters" in names:
                self.visit(list(obj.updaters))
        if not isinstance(obj, types.ModuleType):
            methods = _user_methods(type(obj))
            for name in names & methods.keys():
                self.visit(methods[name])

    def callable(self, function: Any) -> None:
        if isinstance(function, types.MethodType):
            self.visit(function.__self__)
            self.visit(function.__func__)
            return
        code = function.__code__
        for cell in function.__closure__ or ():
            self.visit(cell)
        for item in (
            *(function.__defaults__ or ()),
            *(function.__kwdefaults__ or {}).values(),
        ):
            self.visit(item)
        if engine_code(function) or not _user_code(code):
            return
        if id(code) in self._codes:
            return
        self._codes.add(id(code))
        names = _names_of(code)
        if names & _DYNAMIC:
            names |= _attribute_constants(function, code)
        self.names |= names
        namespace = function.__globals__
        for name in names:
            # The binding itself, which a ``global`` statement rebinds.
            self.ids.add(_global_key(namespace, name))
            if name in namespace:
                self.visit(namespace[name])


def _global_key(namespace: dict[str, Any], name: str) -> int:
    """A stand-in id for the module global ``name``: negative, so it never
    meets an object's id.
    """
    return -1 - (hash((id(namespace), name)) & (2**62 - 1))


#: Calls that name an attribute by a value: bounded when it is a constant.
_BY_NAME = frozenset({"getattr", "setattr", "delattr", "hasattr"})

#: Each code object's source node, or why it has none: finding it reads the
#: file and the code's instructions, so a scene's plays share the answer.
_SOURCE_NODES: dict[types.CodeType, ast.AST | Exception] = {}


def _source_node(function: Any) -> Any:
    """``purity._function_node``, remembered by code object."""
    from . import purity

    code = function.__code__
    found = _SOURCE_NODES.get(code)
    if found is None:
        try:
            found = purity._function_node(function)
        except purity.Refused as refused:
            found = refused
        _SOURCE_NODES[code] = found
    if isinstance(found, Exception):
        raise found
    return found


def _attribute_constants(function: Any, code: types.CodeType) -> set[str]:
    """The attributes the code names through ``getattr`` and its kin, each
    by a string constant; :class:`Unbounded` for any other dynamic access.
    """
    from . import purity

    dynamic = _names_of(code) & _DYNAMIC
    if dynamic - _BY_NAME:
        raise Unbounded(f"uses {', '.join(sorted(dynamic - _BY_NAME))}", code)
    try:
        node = _source_node(function)
    except purity.Refused as refused:
        raise Unbounded(
            f"uses {', '.join(sorted(dynamic))} ({refused})", code
        ) from None
    calls = {id(c.func): c for c in ast.walk(node) if isinstance(c, ast.Call)}
    found: set[str] = set()
    for item in ast.walk(node):
        used = (
            item.id
            if isinstance(item, ast.Name)
            else item.attr
            if isinstance(item, ast.Attribute)
            else None
        )
        if used not in _BY_NAME:
            continue
        call = calls.get(id(item))
        name = call.args[1] if call is not None and len(call.args) > 1 else None
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            raise Unbounded(f"uses {used} with a name it computes", code)
        found.add(name.value)
    return found


def reach(callables: Iterable[Any], mobjects: Iterable[Any] = ()) -> frozenset[int]:
    """The ids of everything ``callables`` may read, with ``mobjects``
    (what they update) and their families: mobjects, mutable containers and
    arrays, closure cells, global namespaces and plain objects. Raises
    :class:`Unbounded` when it can't be bounded.
    """
    walk = _Reach()
    for mob in mobjects:
        walk.visit(mob)
    for function in callables:
        walk.visit(function)
    return walk.done()


# Writes: does a dt-free callback change anything but what it updates?

#: Mobject methods that only read: called on a mobject the callback does not
#: update, they change nothing.
_READERS = frozenset(
    {
        "copy",
        "deepcopy",
        "n2p",
        "p2n",
        "c2p",
        "p2c",
        "i2gp",
        "number_to_point",
        "point_to_number",
        "coords_to_point",
        "point_to_coords",
        "input_to_graph_point",
        "input_to_graph_coords",
        "point_from_proportion",
        "proportion_from_point",
        "length_over_dim",
        "family_members_with_points",
        "plot",
        "plot_parametric_curve",
        "plot_line_graph",
        "angle_of_tangent",
        "slope_of_tangent",
        "pfp",
    }
)
_READER_PREFIXES = ("get_", "is_", "has_")
#: Library functions that change an argument.
_MUTATORS = frozenset(
    {"copyto", "put", "putmask", "place", "fill_diagonal", "put_along_axis", "shuffle"}
)
#: Builtins that change something the call does not return.
_IMPURE_BUILTINS = frozenset(
    {"setattr", "delattr", "exec", "eval", "next", "open", "input", "globals", "vars"}
)


def _root(node: ast.AST) -> ast.AST:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node


class _Writes:
    """Whether a function's source shows a change to anything but its
    ``owners`` (the parameter naming the mobject it updates) and objects it
    made itself.
    """

    def __init__(self) -> None:
        self._checked: set[tuple[int, frozenset[str]]] = set()

    def elsewhere(self, function: Any, owners: frozenset[str], depth: int = 0) -> bool:
        from . import purity

        if isinstance(function, types.MethodType):
            function = function.__func__
            owners = frozenset()
        code = _code_of(function)
        if code is None or not isinstance(function, types.FunctionType):
            return True
        if engine_code(function) or not _user_code(code):
            return True  # judged by what it reaches
        if (id(code), owners) in self._checked:
            return False
        self._checked.add((id(code), owners))
        if depth > MAX_DEPTH:
            return True
        try:
            node = _source_node(function)
        except purity.Refused:
            return True
        bindings = self._bindings(function)
        fresh = self._fresh(node, bindings)
        for item in ast.walk(node):
            if isinstance(item, (ast.Global, ast.Nonlocal)):
                return True
            targets: list[ast.AST] = []
            if isinstance(item, ast.Assign):
                targets = list(item.targets)
            elif isinstance(item, (ast.AugAssign, ast.AnnAssign)):
                targets = [item.target]
                if (
                    isinstance(item, ast.AugAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id not in fresh
                ):
                    return True  # in place on what the name holds
            elif isinstance(item, ast.Delete):
                targets = list(item.targets)
            if any(self._store_elsewhere(t, owners, fresh) for t in targets):
                return True
            if isinstance(item, ast.Call) and self._call_elsewhere(
                item, owners, fresh, bindings, depth
            ):
                return True
        return False

    def _store_elsewhere(
        self, target: ast.AST, owners: frozenset[str], fresh: set[str]
    ) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._store_elsewhere(t, owners, fresh) for t in target.elts)
        if isinstance(target, ast.Starred):
            return self._store_elsewhere(target.value, owners, fresh)
        if isinstance(target, ast.Name):
            return False
        root = _root(target)
        return not (
            isinstance(root, ast.Name) and (root.id in owners or root.id in fresh)
        )

    def _bindings(self, function: types.FunctionType) -> dict[str, Any]:
        """What each name the function's code uses is bound to when it is
        called as an updater: a parameter is the argument (unknown), unless
        a default fills it.
        """
        found: dict[str, Any] = dict(vars(builtins))
        found.update(function.__globals__)
        code = function.__code__
        for name, cell in zip(
            code.co_freevars, function.__closure__ or (), strict=True
        ):
            with contextlib.suppress(ValueError):  # an empty cell
                found[name] = cell.cell_contents
        positional = code.co_varnames[: code.co_argcount]
        found.update(
            dict.fromkeys(
                code.co_varnames[: code.co_argcount + code.co_kwonlyargcount],
                _PARAMETER,
            )
        )
        defaults = function.__defaults__ or ()
        if defaults:
            found.update(zip(positional[-len(defaults) :], defaults, strict=True))
        found.update(function.__kwdefaults__ or {})
        return found

    def _fresh(self, node: ast.AST, bindings: dict[str, Any]) -> set[str]:
        """Local names only ever bound to an object the function made."""
        assigned: dict[str, bool] = {}
        for item in ast.walk(node):
            pairs: list[tuple[ast.AST, ast.AST | None]] = []
            if isinstance(item, ast.Assign):
                pairs = [(t, item.value) for t in item.targets]
            elif isinstance(item, (ast.AnnAssign, ast.NamedExpr)):
                pairs = [(item.target, item.value)]
            elif isinstance(item, (ast.For, ast.comprehension, ast.With, ast.AsyncFor)):
                targets = (
                    [item.target]
                    if not isinstance(item, ast.With)
                    else [w.optional_vars for w in item.items if w.optional_vars]
                )
                pairs = [(t, None) for t in targets]
            for target, value in pairs:
                names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
                made = (
                    value is not None
                    and isinstance(target, ast.Name)
                    and self._made(value, bindings)
                )
                for name in names:
                    assigned[name] = assigned.get(name, True) and made
        return {name for name, made in assigned.items() if made}

    def _made(self, value: ast.AST, bindings: dict[str, Any]) -> bool:
        if isinstance(
            value,
            (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp),
        ):
            return True
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Attribute) and func.attr in ("copy", "deepcopy"):
                return True
            root = _root(func)
            if isinstance(root, ast.Call):
                return self._made(root, bindings)
            if isinstance(func, ast.Name):
                target = bindings.get(func.id)
                return isinstance(target, type)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                holder = bindings.get(func.value.id)
                if isinstance(holder, types.ModuleType):
                    return isinstance(getattr(holder, func.attr, None), type)
        return False

    def _call_elsewhere(
        self,
        call: ast.Call,
        owners: frozenset[str],
        fresh: set[str],
        bindings: dict[str, Any],
        depth: int,
    ) -> bool:
        if any(k.arg == "out" for k in call.keywords):
            return True
        func = call.func
        if isinstance(func, ast.Name):
            target = bindings.get(func.id, _UNKNOWN)
            return self._callee_elsewhere(func.id, target, depth)
        if not isinstance(func, ast.Attribute):
            return True
        name = func.attr
        root = _root(func.value)
        if isinstance(root, ast.Name):
            if root.id in owners or root.id in fresh:
                return False
            holder = bindings.get(root.id, _UNKNOWN)
            if isinstance(holder, types.ModuleType):
                module = getattr(holder, "__name__", "")
                if (
                    module.split(".")[0] == "random"
                    or ".random" in module
                    or name == "random"
                ):
                    return True
                if not _internal(getattr(holder, "__file__", "") or ""):
                    return self._callee_elsewhere(
                        name, getattr(holder, name, _UNKNOWN), depth
                    )
                return name in _MUTATORS
            if name in _READERS or name.startswith(_READER_PREFIXES):
                return False
            from . import library

            if name in library.VALUE_METHODS and not _is_mobject(holder):
                return False
            if (
                holder is not _UNKNOWN
                and holder is not _PARAMETER
                and func.value is root
            ):
                member = inspect.getattr_static(type(holder), name, None)
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if isinstance(member, types.FunctionType) and _user_code(
                    member.__code__
                ):
                    return self.elsewhere(member, frozenset(), depth + 1)
            return True
        if isinstance(root, ast.Call):
            # A method of what a call returned: a new object when the call
            # makes one; else whatever it returned may be shared.
            reader = name in _READERS or name.startswith(_READER_PREFIXES)
            return not reader and not self._made(root, bindings)
        return not isinstance(
            root,
            (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Dict, ast.ListComp),
        )

    def _callee_elsewhere(self, name: str, target: Any, depth: int) -> bool:
        if target is _UNKNOWN or target is _PARAMETER:
            return True
        if isinstance(target, type):
            return False  # it makes a new object
        if isinstance(target, types.BuiltinFunctionType):
            if (
                getattr(target, "__self__", None) is builtins
                or target.__module__ == "builtins"
            ):
                return name in _IMPURE_BUILTINS
            module = getattr(target, "__module__", "") or ""
            return (
                module.split(".")[0] == "random"
                or "random" in module
                or name in _MUTATORS
            )
        if isinstance(target, np.ufunc):
            return False
        if isinstance(target, (types.FunctionType, types.MethodType)):
            code = _code_of(target)
            if code is not None and _user_code(code):
                return self.elsewhere(target, frozenset(), depth + 1)
            module = getattr(target, "__module__", "") or ""
            return (
                module.split(".")[0] == "random"
                or ".random" in module
                or name in _MUTATORS
            )
        return True


_UNKNOWN = object()
_PARAMETER = object()


def _is_mobject(value: Any) -> bool:
    from arachne.mobject.mobject import Mobject

    return isinstance(value, Mobject)


# The plan


#: A node's writes when its source shows a change beyond what it updates:
#: anything it reaches.
_REACHED = None


@dataclass
class _Node:
    what: str
    #: "step" runs at every tick and reads there; "instant" (a dt-free
    #: updater or redraw) runs at full ticks only, unless pulled; "transparent"
    #: (an animation's interpolation) runs at every tick, but what it reads
    #: there matters only when stepping work reads what it writes.
    role: str
    #: What it may read: a set, or how to find it. A node that is not
    #: stepping is reached only once something pulls it: until then what it
    #: reads matters at no stepping tick, so an unbounded read pulled by
    #: nothing leaves the play stepping.
    reach_of: frozenset[int] | Callable[[], frozenset[int]]
    #: What it may write: a set, or ``_REACHED`` for anything it reaches.
    writes: frozenset[int] | None = frozenset()
    keys: tuple[tuple[int, int], ...] = ()
    active: bool = False

    @property
    def reads(self) -> frozenset[int]:
        if callable(self.reach_of):
            self.reach_of = self.reach_of()
        return self.reach_of

    def written(self) -> frozenset[int]:
        """What it may write. Anything it reaches is only asked for when the
        play has stepping work; reaching it unboundedly then raises
        :class:`Unbounded`, since stepping work may read any of it.
        """
        return self.reads if self.writes is _REACHED else self.writes


def _family_ids(mob: Any) -> frozenset[int]:
    return frozenset(id(member) for member in mob.get_family())


def _describe(function: Any) -> str:
    code = _code_of(function)
    if code is None:
        return repr(function)
    return f"{code.co_name} at {code.co_filename}:{code.co_firstlineno}"


def _updaters(roots: Iterable[Any]) -> Iterator[tuple[Any, Any]]:
    """Every (mobject, updater) :meth:`.Mobject.update` may run from ``roots``."""
    seen: set[tuple[int, int]] = set()
    for root in roots:
        for member in root.get_family():
            for updater in member.updaters:
                key = (id(member), id(updater))
                if key not in seen:
                    seen.add(key)
                    yield member, updater


def _copies(animations: Iterable[Animation]) -> list[Any]:
    """The mobjects the animations update before interpolating (their
    copies' updaters run there, as in CE).
    """
    from arachne.animation.composition import AnimationGroup

    from . import trusted

    found: list[Any] = []
    for animation in trusted.iter_animations(list(animations)):
        if isinstance(animation, AnimationGroup):
            continue
        with contextlib.suppress(AttributeError):
            found.extend(animation.get_all_mobjects_to_update())
    return found


def _animation_node(animation: Animation, check: _Writes) -> _Node:
    """An animation's interpolation, which runs at every tick: what it reads
    (anything it holds: a path, a rate function, a mobject it follows) joins
    the closure only when stepping work reads what it writes, its mobject's
    family. A callback animation writes what its function may change,
    bounded as an updater's writes are.
    """
    from . import trusted

    writes: frozenset[int] | None = frozenset(trusted.writes(animation))
    if isinstance(animation, trusted.CALLBACK_ANIMATIONS):
        function = vars(animation)["update_function"]
        try:
            owners = frozenset(list(inspect.signature(function).parameters)[:1])
        except (TypeError, ValueError):
            owners = frozenset()
        if check.elsewhere(function, owners):
            writes = _REACHED
    return _Node(
        type(animation).__name__,
        "transparent",
        lambda: reach([animation], trusted._mobjects_of(animation)),
        writes,
    )


def _updater_node(
    member: Any,
    updater: Any,
    check: _Writes,
    reaches: dict[tuple[int, int, int], frozenset[int]],
) -> _Node:
    from .admission import _redraw_code

    key = (id(member), id(updater))
    code = _code_of(updater)
    if code is not None and code is _redraw_code():
        cells = dict(
            zip(
                code.co_freevars,
                (c.cell_contents for c in updater.__closure__),
                strict=True,
            )
        )
        target, function = cells["mob"], cells["func"]
        owners: frozenset[str] = frozenset()
        what = "redraw " + _describe(function)
    else:
        target, function = member, updater
        try:
            parameters = list(inspect.signature(updater).parameters)
        except (TypeError, ValueError):
            parameters = []
        owners = frozenset(parameters[:1])
        what = "updater " + _describe(function)

    def reads() -> frozenset[int]:
        # The mobject it is handed is read through too: ``m.source`` is as
        # much a read as a closure cell.
        found = (id(function), id(member), id(target))
        if found not in reaches:
            reaches[found] = reach([function], [member, target])
        return reaches[found]

    if loads_dt(updater):
        return _Node(what, "step", reads(), keys=(key,))
    writes = _REACHED if check.elsewhere(function, owners) else _family_ids(target)
    return _Node(what, "instant", reads, writes, keys=(key,))


def _closure(nodes: list[_Node]) -> int:
    """Activate every node stepping work reads, directly or through what
    reads it; return how many dt-free updaters that pulls onto the stepping
    tick. With no stepping work nothing is read at a stepping tick, so
    nothing is pulled, whatever the others may write.
    """
    needed: set[int] = set()
    for node in nodes:
        if node.role == "step":
            node.active = True
            needed |= node.reads
    pulled = 0
    changed = any(node.role == "step" for node in nodes)
    while changed:
        changed = False
        for node in nodes:
            if not node.active and node.written() & needed:
                node.active = True
                needed |= node.reads
                changed = True
                pulled += node.role == "instant"
    return pulled


def plan_ticks(
    scene: Scene, animations: list[Animation], base_dir: str | None = None
) -> TickPlan:
    """The stepping plan of the ticked play ``scene`` is about to evaluate
    (its animations begun); source paths in its reason are relative to
    ``base_dir``.
    """
    from arachne.animation.composition import AnimationGroup

    from . import trusted
    from .admission import _update_replaced
    from .dispatch import user_overrides

    def unbounded(why: str) -> TickPlan:
        return TickPlan(unbounded=f"{UNBOUNDED}: {why}")

    replaced = {"update_mobjects", "update_self"} & set(user_overrides(type(scene)))
    if replaced:
        return unbounded(f"the scene's {', '.join(sorted(replaced))} is its own code")
    why = _update_replaced(scene)
    if why is not None:
        return unbounded(why)
    camera = scene.renderer.camera
    if vars(camera).get("image_mobjects_from_cameras"):
        return unbounded(
            "the camera draws other cameras, whose inputs it does not weigh"
        )
    # A scene may redefine its classes' methods between plays.
    _USER_METHODS.clear()
    nodes: list[_Node] = []
    try:
        for animation in trusted.iter_animations(animations):
            if not engine_class(type(animation)):
                raise Unbounded(
                    f"{type(animation).__name__} is the scene's own animation class"
                )
        check = _Writes()
        nodes.extend(
            _animation_node(animation, check)
            for animation in trusted.iter_animations(animations)
            if not isinstance(animation, AnimationGroup)
        )
        reaches: dict[tuple[int, int, int], frozenset[int]] = {}
        for member, updater in _updaters((*_copies(animations), *scene.mobjects)):
            nodes.append(_updater_node(member, updater, check, reaches))
        nodes.extend(
            _Node("scene updater " + _describe(function), "step", reach([function]))
            for function in scene.updaters
        )
        centres = getattr(camera, "fixed_orientation_mobjects", None) or {}
        if centres:
            nodes.append(
                _Node(
                    "camera centres",
                    "step",
                    reach(list(centres.values()), centres),
                )
            )
        pulled = _closure(nodes)
    except Unbounded as refused:
        return unbounded(refused.describe(base_dir))
    return TickPlan(
        skipped=frozenset(
            key
            for node in nodes
            if node.role == "instant" and not node.active
            for key in node.keys
        ),
        known=frozenset(key for node in nodes for key in node.keys),
        stepping=sum(1 for node in nodes if node.role == "step" and node.keys),
        pulled=pulled,
    )
