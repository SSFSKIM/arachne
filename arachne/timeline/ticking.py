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

Both bounds are default-deny: an access is bounded only along a path the
analysis resolves at plan time, and anything else counts as reaching
anything. What a callback may read is what its code can reach
(:func:`reach`): its closure cells, the globals its code names (one key per
binding), its defaults, the object a bound method holds, the mobject it
updates with its family, and what those hold, through the attributes its
code names, on the instance or on the scene's own class that holds them
(keyed by that class), into containers it follows whole, and into arrays by
the memory they share (``TracedPath``'s ``traced_point_func`` included). A
module the scene imports is read binding by binding, one key per attribute
whatever alias it is imported under. A name computed at run time
(``getattr`` with a computed name, ``getattr`` held under another name), or
an object whose contents it can't follow, is unbounded.

What a dt-free callback, or the function of an
``UpdateFromFunc``/``UpdateFromAlphaFunc``, may write is the mobject it
updates with its family (``always_redraw``'s target), unless its source
shows a change anywhere else (``_Writes``): a store or a call that reaches
something other than that family, or than what the callback made itself,
along a path it can't resolve to that family (an attribute the mobject
holds, an element of a list it made). Then it writes everything it reaches,
which stepping work may read whenever the play has any. The play ticks
everything, under the reason ``tick.unbounded_read``, when a reach that
matters can't be bounded: a stepping callback's, one pulled into the
closure, or one whose writes are everything it reaches in a play with
stepping work. An unbounded read of a callback nothing pulls, whose writes
are bounded, leaves the play stepping.

A callback that may add or remove updaters or mobjects, or suspend them, as
it plays (``add``, ``remove``, ``add_updater``, ``clear_updaters``, a
``become`` that pads its mobject with members carrying updaters) would leave
a stepping tick meeting work no plan weighed, or running work gone: a play
holding one, whichever callback it is, ticks everything under the reason
``tick.structural_mutation``. A composition's own events (a part beginning)
plan again instead.

:class:`TickPlan` is the outcome; :func:`skipping` installs its skipped
updaters for one stepping tick, which :meth:`.Mobject.update` consults.
"""

from __future__ import annotations

__all__ = [
    "STRUCTURAL",
    "UNBOUNDED",
    "TickPlan",
    "loads_dt",
    "plan_ticks",
    "reach",
    "skipping",
]

import ast
import builtins
import collections
import contextlib
import dis
import functools
import inspect
import operator
import random
import sys
import types
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
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

#: The reason family of a play that ticks everything because a callback may
#: add or remove updaters or mobjects as it plays.
STRUCTURAL = "tick.structural_mutation"

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
        "__getattribute__",
        "attrgetter",
        "methodcaller",
        "getattr_static",
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
    #: The ids of what the work a stepping tick runs may read or write.
    stepped_ids: frozenset[int] = frozenset()
    #: Whether a skipped callback may change which updaters there are: the
    #: play plans again after each full tick that changed them.
    rechecks: bool = False

    def stepping_families(self) -> frozenset[int]:
        """The ids of everything the stepping set (dt updaters, the scene's
        updaters, every interpolation, and the callbacks pulled with them)
        may read or write: each mobject among them with its family, and the
        containers, cells and bindings they hold. A difference outside these
        and outside :attr:`skipped`'s mobjects is not stepping's.
        """
        return self.stepped_ids

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
        if key in _DYNAMIC_IDS:
            # Held as a value, under a name of the code's own choosing: its
            # uses can't be told from the names the code spells.
            raise Unbounded(f"holds {value.__name__} as a value")
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
            if _array_root(value)[1]:
                self.ids.add(_ANY_ARRAY)
            storage = _storage_key(value)
            if storage is not None:
                self.ids.add(storage)
            if value.dtype == object:
                for item in value.flat:
                    self.visit(item)
        elif isinstance(value, (list, tuple, set, frozenset, collections.deque)):
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
        elif isinstance(value, _OPAQUE_STATE):
            self.ids.add(key)  # its own state, which holds no mobject
        elif hasattr(value, "__dict__") or getattr(type(value), "__slots__", None):
            # A plain object (the scene's own helper): shared state, followed
            # through the attributes code names.
            self.ids.add(key)
            self._attributes(value)
        else:
            raise Unbounded(
                f"reaches a {type(value).__name__}, whose contents it can't follow"
            )

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
                self.ids.add(id(type(member)))

    def _class_code(self, cls: type) -> None:
        """A user class's methods: what its instances' methods may read; and
        the values its classes hold, which every instance reads as its own.
        """
        for function in _user_methods(cls).values():
            self.visit(function)
        for klass in cls.__mro__:
            if not _scene_class(klass):
                continue
            for name in vars(klass):
                self._class_attribute(cls, name)

    def _class_attribute(self, cls: type, name: str) -> None:
        """What ``name`` reads on an instance of ``cls`` that does not hold
        it: the value the nearest of the scene's own classes defining it
        holds, keyed by that class (a store to it rebinds the class's).
        """
        for klass in cls.__mro__:
            if not _scene_class(klass):
                continue
            if name in vars(klass):
                value = vars(klass)[name]
                if name.startswith("__") or isinstance(
                    value, (types.FunctionType, staticmethod, classmethod, property)
                ):
                    return  # a method: _user_methods follows it
                self.ids.add(id(klass))
                self.visit(value)
                return

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
        module = isinstance(obj, types.ModuleType)
        for name in names:
            if name in own:
                if module:
                    # The binding itself, as a global's: one key for every
                    # alias the module is imported under.
                    self.ids.add(_global_key(own, name))
                self.visit(own[name])
            elif not module:
                self._class_attribute(type(obj), name)
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


def _scene_class(klass: type) -> bool:
    """Whether ``klass`` is defined by the scene's code."""
    module = sys.modules.get(klass.__module__)
    filename = getattr(module, "__file__", None)
    return bool(filename) and not engine_class(klass) and not _internal(filename)


#: The key every reach of a writable array carries, and every write in place
#: to an array the plan does not key: such a write may change any of them.
_ANY_ARRAY = -(2**62) - 1


def _array_root(array: np.ndarray) -> tuple[Any, bool]:
    """What owns ``array``'s memory, through its chain of views, and whether
    any array along it can write that memory.
    """
    owner: Any = array
    writable = array.flags.writeable
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
        writable = writable or (isinstance(owner, np.ndarray) and owner.flags.writeable)
    return owner, writable


def _storage_key(array: np.ndarray) -> int | None:
    """One key for every array sharing ``array``'s memory; ``None`` for
    memory nothing can write, and for the engine's constant vectors
    (``ORIGIN`` and its kin, which every mobject made with a default holds,
    so that a key of theirs would tie every mobject to every other): a write
    in place to one of those is a write to :data:`_ANY_ARRAY`.
    """
    owner, writable = _array_root(array)
    if not writable or id(owner) in _constant_arrays():
        return None
    return id(owner)


@functools.cache
def _constant_arrays() -> frozenset[int]:
    from arachne import constants

    return frozenset(
        id(value) for value in vars(constants).values() if isinstance(value, np.ndarray)
    )


def _global_key(namespace: dict[str, Any], name: str) -> int:
    """A stand-in id for the module global ``name``: negative, so it never
    meets an object's id.
    """
    return -1 - (hash((id(namespace), name)) & (2**62 - 1))


#: The callables behind :data:`_DYNAMIC`'s names, found by identity
#: whatever name the code binds them to.
_DYNAMIC_CALLABLES = tuple(
    getattr(builtins, name) for name in sorted(_DYNAMIC) if hasattr(builtins, name)
) + (operator.attrgetter, operator.methodcaller, inspect.getattr_static)
_DYNAMIC_IDS = frozenset(map(id, _DYNAMIC_CALLABLES))

#: Objects whose state is their own and holds no mobject: followed by id.
_OPAQUE_STATE = (random.Random, np.random.Generator, np.random.RandomState)

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


@dataclass(frozen=True)
class _Effect:
    """What a callback's source shows it may do beyond what it updates."""

    #: It may change something but the mobject it updates (with its family)
    #: and the objects it made itself.
    elsewhere: bool = False
    #: It may change which updaters run or which mobjects there are (add,
    #: remove, suspend): work no plan weighed, or work a plan weighed gone.
    restructures: bool = False
    #: It may write in place to an array the plan does not key (one it can't
    #: resolve, or the engine's constant vectors): any array.
    arrays: bool = False

    def __or__(self, other: _Effect) -> _Effect:
        return _Effect(
            self.elsewhere or other.elsewhere,
            self.restructures or other.restructures,
            self.arrays or other.arrays,
        )


_NO_EFFECT = _Effect()
_ELSEWHERE = _Effect(elsewhere=True)
_ANY_EFFECT = _Effect(elsewhere=True, restructures=True, arrays=True)
_IN_PLACE = _Effect(elsewhere=True, arrays=True)

#: Mobject methods that return a new value (a number, a new array).
_VALUE_GETTERS = frozenset(
    {
        "get_x",
        "get_y",
        "get_z",
        "get_value",
        "get_center",
        "get_width",
        "get_height",
        "get_depth",
        "get_left",
        "get_right",
        "get_top",
        "get_bottom",
        "get_corner",
        "get_angle",
        "get_length",
        "point_from_proportion",
    }
)
#: numpy functions that return a new array.
_NUMPY_MAKERS = frozenset(
    {
        "array",
        "zeros",
        "ones",
        "empty",
        "full",
        "zeros_like",
        "ones_like",
        "full_like",
        "linspace",
        "arange",
        "concatenate",
        "stack",
        "vstack",
        "hstack",
        "column_stack",
        "cross",
        "dot",
        "outer",
        "clip",
        "round",
        "copy",
    }
)

#: Methods of an array that change it in place.
_ARRAY_MUTATORS = frozenset(
    {"fill", "put", "sort", "resize", "itemset", "partition", "setfield", "byteswap"}
)

#: Methods that may change a mobject's or the scene's members or updaters,
#: or whether they run.
_RESTRUCTURERS = frozenset(
    {
        "add",
        "add_to_back",
        "add_updater",
        "add_foreground_mobject",
        "add_foreground_mobjects",
        "add_n_more_submobjects",
        "align_data",
        "align_submobjects",
        "become",
        "bring_to_back",
        "bring_to_front",
        "clear",
        "clear_updaters",
        "insert",
        "match_updaters",
        "remove",
        "remove_foreground_mobject",
        "remove_foreground_mobjects",
        "remove_updater",
        "replace_submobject",
        "resume_updating",
        "set_submobjects",
        "shuffle",
        "shuffle_submobjects",
        "sort",
        "sort_submobjects",
        "invert",
        "suspend_updating",
    }
)
#: The lists that are a mobject's or the scene's structure.
_STRUCTURE_LISTS = frozenset(
    {"updaters", "submobjects", "mobjects", "foreground_mobjects", "moving_mobjects"}
)


class _Writes:
    """What a function's source shows it may change beyond its ``owners``
    (the parameter naming the mobject it updates, ``owner`` when known) and
    the objects it made itself, and whether it may change the structure
    (``_Effect``).

    Default-deny: a change is local only when it reaches the owner, a member
    of its family or an object the function made, through a path resolved at
    plan time (the owner's own attributes, read on the instance, each to a
    mobject of its family). Anything else changes something elsewhere.
    """

    def __init__(self) -> None:
        #: Each function's effect, by its identity, owners and owner: the
        #: function is kept so that its id is not reused.
        self._effects: dict[tuple[int, frozenset[str], int], tuple[Any, _Effect]] = {}

    def elsewhere(
        self, function: Any, owners: frozenset[str], owner: Any = None
    ) -> bool:
        return self.effect(function, owners, owner).elsewhere

    def effect(
        self,
        function: Any,
        owners: frozenset[str],
        owner: Any = None,
        depth: int = 0,
        made: frozenset[str] = frozenset(),
    ) -> _Effect:
        """``made`` names parameters bound to an object being made (a
        constructor's ``self``): its own, as what the function makes is.
        """
        if isinstance(function, types.MethodType):
            function = function.__func__
            owners, owner = frozenset(), None
        code = _code_of(function)
        if code is None or not isinstance(function, types.FunctionType):
            return _ANY_EFFECT
        if engine_code(function) or not _user_code(code):
            # Judged by what it reaches, and by the names its code uses.
            names = _names_of(code)
            return _Effect(
                elsewhere=True,
                restructures=bool(names & (_RESTRUCTURERS | _STRUCTURE_LISTS)),
            )
        key = (id(function), owners | made, id(owner))
        found = self._effects.get(key)
        if found is not None:
            return found[1]
        # Calls back into itself add nothing to what it is found to do.
        self._effects[key] = (function, _NO_EFFECT)
        effect = (
            _ANY_EFFECT
            if depth > MAX_DEPTH
            else self._scan(function, owners, owner, depth, made)
        )
        self._effects[key] = (function, effect)
        return effect

    def _scan(
        self,
        function: Any,
        owners: frozenset[str],
        owner: Any,
        depth: int,
        made: frozenset[str],
    ) -> _Effect:
        from . import purity

        try:
            node = _source_node(function)
        except purity.Refused:
            return _ANY_EFFECT
        site = _Site(self, owners, owner, self._bindings(function), depth)
        site.fresh = self._fresh(node, site.bindings) | made
        site.values = self._fresh(node, site.bindings, self._new_value)
        parameters = set(_parameters(function))
        site.aliases = {
            name: value
            for name, value in _aliases(node).items()
            if name not in parameters
        }
        effect = _NO_EFFECT
        for item in ast.walk(node):
            if isinstance(item, (ast.Global, ast.Nonlocal)):
                effect |= _ELSEWHERE
            targets: list[ast.AST] = []
            if isinstance(item, ast.Assign):
                targets = list(item.targets)
            elif isinstance(item, (ast.AugAssign, ast.AnnAssign)):
                targets = [item.target]
                if isinstance(item, ast.AugAssign) and isinstance(
                    item.target, (ast.Name, ast.Attribute)
                ):
                    # In place on what the name or attribute holds.
                    effect |= site.in_place(item.target)
            elif isinstance(item, ast.Delete):
                targets = list(item.targets)
            for target in targets:
                effect |= site.store(target)
            if isinstance(item, ast.Call):
                effect |= site.call(item)
            if effect == _ANY_EFFECT:
                break
        return effect

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
        # A local is bound only as the function runs.
        found.update(dict.fromkeys((*code.co_varnames, *code.co_cellvars), _UNKNOWN))
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

    def _fresh(
        self,
        node: ast.AST,
        bindings: dict[str, Any],
        made: Callable[[ast.AST, dict[str, Any]], bool] | None = None,
    ) -> set[str]:
        """Local names only ever bound to an object the function made
        (``made`` says which values are, :meth:`_made` by default).
        """
        made_by = self._made if made is None else made
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
                new = (
                    value is not None
                    and isinstance(target, ast.Name)
                    and made_by(value, bindings)
                )
                for name in names:
                    assigned[name] = assigned.get(name, True) and new
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

    def _new_value(self, value: ast.AST, bindings: dict[str, Any]) -> bool:
        """Whether ``value`` is a value the expression makes: a number, a new
        array, a group an operator builds (what a change in place to it can
        reach is nothing shared).
        """
        if isinstance(
            value,
            (
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.JoinedStr,
            ),
        ):
            return True
        if isinstance(value, ast.Subscript):
            return self._new_value(value.value, bindings)
        if not isinstance(value, ast.Call):
            return self._made(value, bindings)
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in _VALUE_GETTERS:
            return True
        if isinstance(func, ast.Name):
            target = bindings.get(func.id)
            if any(target is maker for maker in (abs, round, sum, len)):
                return True
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            holder = bindings.get(func.value.id)
            if getattr(holder, "__name__", None) == "numpy":
                found = getattr(holder, func.attr, None)
                return isinstance(found, np.ufunc) or func.attr in _NUMPY_MAKERS
        return self._made(value, bindings)

    def callee(self, name: str, target: Any, depth: int) -> _Effect:
        """What calling ``target``, bound to ``name``, may do."""
        if target is _UNKNOWN or target is _PARAMETER:
            return _ANY_EFFECT
        if isinstance(target, type):
            # A new object; the scene's own class runs its own code to make it.
            init = vars(target).get("__init__") if not engine_class(target) else None
            if isinstance(init, types.FunctionType) and _user_code(init.__code__):
                made = frozenset(init.__code__.co_varnames[:1])
                return self.effect(init, frozenset(), None, depth + 1, made)
            return _NO_EFFECT
        if isinstance(target, types.BuiltinFunctionType):
            name = target.__name__  # whatever the code calls it
            if (
                getattr(target, "__self__", None) is builtins
                or target.__module__ == "builtins"
            ):
                if name in ("exec", "eval", "setattr"):
                    return _ANY_EFFECT
                return _ELSEWHERE if name in _IMPURE_BUILTINS else _NO_EFFECT
            module = getattr(target, "__module__", "") or ""
            impure = (
                module.split(".")[0] == "random"
                or "random" in module
                or name in _MUTATORS
            )
            return _ELSEWHERE if impure else _NO_EFFECT
        if isinstance(target, np.ufunc):
            return _NO_EFFECT
        if isinstance(target, (types.FunctionType, types.MethodType)):
            code = _code_of(target)
            if code is not None and _user_code(code):
                return self.effect(target, frozenset(), None, depth + 1)
            module = getattr(target, "__module__", "") or ""
            impure = (
                module.split(".")[0] == "random"
                or ".random" in module
                or name in _MUTATORS
            )
            return _ELSEWHERE if impure else _NO_EFFECT
        return _ANY_EFFECT


@dataclass
class _Site:
    """One function's source being read for its effect."""

    writes: _Writes
    owners: frozenset[str]
    owner: Any
    bindings: dict[str, Any]
    depth: int
    fresh: set[str] = field(default_factory=set)
    #: Local names only ever bound to a value an expression makes.
    values: set[str] = field(default_factory=set)
    #: Local names bound once, to another name or an attribute of one.
    aliases: dict[str, ast.AST] = field(default_factory=dict)
    resolving: set[str] = field(default_factory=set)

    def resolve(self, node: ast.AST) -> Any:
        """The object ``node`` names when the function runs, resolved at plan
        time through the owner's attributes (on the instance) and a mobject's
        members; :data:`_UNKNOWN` when it can't be.
        """
        from arachne.mobject.mobject import Mobject

        if isinstance(node, ast.Name):
            if node.id in self.owners:
                return _UNKNOWN if self.owner is None else self.owner
            if node.id in self.aliases and node.id not in self.resolving:
                self.resolving.add(node.id)
                try:
                    return self.resolve(self.aliases[node.id])
                finally:
                    self.resolving.discard(node.id)
            found = self.bindings.get(node.id, _UNKNOWN)
            return found if found is not _PARAMETER else _UNKNOWN
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if isinstance(base, Mobject):
                return vars(base).get(node.attr, _UNKNOWN)
            return _UNKNOWN
        if isinstance(node, ast.Subscript):
            base = self.resolve(node.value)
            if isinstance(base, Mobject) and engine_code(type(base).__getitem__):
                # One of its members, or a group of them.
                return _MEMBERS_OF(base)
        return _UNKNOWN

    def owned(self, receiver: ast.AST) -> bool:
        """Whether ``receiver`` is the owner, a member of its family, or what
        the function made: the receiver itself, not what it holds.
        """
        if isinstance(receiver, ast.Name):
            return receiver.id in self.owners or receiver.id in self.fresh
        if self.owner is None or not isinstance(_root(receiver), ast.Name):
            return False
        found = self.resolve(receiver)
        if isinstance(found, _MEMBERS_OF):
            found = found.mobject
        family = _family_ids(self.owner)
        return id(found) in family

    def restructures_by(self, name: str, receiver: ast.AST) -> bool:
        """Whether calling method ``name`` on ``receiver`` may change a
        mobject's or the scene's structure.
        """
        if (
            isinstance(receiver, ast.Attribute)
            and receiver.attr in _STRUCTURE_LISTS
            and name not in _READERS
            and name not in ("count", "index")
        ):
            return True  # the structure's own list, changed in place
        if name not in _RESTRUCTURERS:
            return False
        if isinstance(receiver, ast.Name) and receiver.id in self.fresh:
            return False
        if name == "become":
            # It pads its mobject with copies of its own members, which take
            # their updaters along.
            found = self._bound(receiver)
            if isinstance(found, _MEMBERS_OF):
                found = found.mobject
            return not _quiet_family(found)
        holder = self._bound(receiver)
        return not isinstance(holder, (types.ModuleType, list, set, dict, np.ndarray))

    def _bound(self, receiver: ast.AST) -> Any:
        return self.resolve(receiver)

    def in_place(self, target: ast.AST) -> _Effect:
        """What an operator in place (``+=``) on ``target`` may change: an
        array in place, a mobject's members (a mobject's ``+=`` adds); a
        number it only rebinds.
        """
        if isinstance(target, ast.Attribute) and isinstance(
            self._bound(target), _IMMUTABLE
        ):
            return _NO_EFFECT
        return self.into(target)

    def store(self, target: ast.AST) -> _Effect:
        if isinstance(target, (ast.Tuple, ast.List)):
            effect = _NO_EFFECT
            for item in target.elts:
                effect |= self.store(item)
            return effect
        if isinstance(target, ast.Starred):
            return self.store(target.value)
        if isinstance(target, ast.Name):
            return _NO_EFFECT
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            return _ELSEWHERE
        receiver = target.value
        fresh = isinstance(receiver, ast.Name) and receiver.id in self.fresh
        structure = (
            isinstance(target, ast.Attribute) and target.attr in _STRUCTURE_LISTS
        ) or (isinstance(receiver, ast.Attribute) and receiver.attr in _STRUCTURE_LISTS)
        effect = _Effect(
            elsewhere=not self.owned(receiver), restructures=structure and not fresh
        )
        if isinstance(target, ast.Subscript):
            effect |= self.into(receiver)
        return effect

    def into(self, receiver: ast.AST) -> _Effect:
        """What a store into an item of what ``receiver`` holds may change
        (an array's memory, a mobject's members), as a call writing to it
        in place does.
        """
        if isinstance(receiver, ast.Name) and (
            receiver.id in self.fresh or receiver.id in self.values
        ):
            return _NO_EFFECT
        found = self._bound(receiver)
        if found is _UNKNOWN or isinstance(found, _MEMBERS_OF):
            return _ANY_EFFECT
        owned = self.owned(receiver) or (
            isinstance(receiver, ast.Attribute) and self.owned(receiver.value)
        )
        if isinstance(found, np.ndarray):
            if not _array_root(found)[1]:
                return _NO_EFFECT
            if _storage_key(found) is None:
                return _IN_PLACE  # the engine's constant
            return _NO_EFFECT if owned else _ELSEWHERE
        if _is_mobject(found):
            return _Effect(elsewhere=not owned, restructures=True)
        return _NO_EFFECT if owned else _ELSEWHERE

    def call(self, call: ast.Call) -> _Effect:
        writes = self.writes
        for keyword in call.keywords:
            if keyword.arg == "out":
                return _ELSEWHERE | self.into(keyword.value)
        func = call.func
        if isinstance(func, ast.Name):
            target = self.bindings.get(func.id, _UNKNOWN)
            return writes.callee(func.id, target, self.depth)
        if not isinstance(func, ast.Attribute):
            return _ANY_EFFECT
        name = func.attr
        receiver = func.value
        restructures = _Effect(restructures=self.restructures_by(name, receiver))
        if name in _ARRAY_MUTATORS:
            restructures |= self.into(receiver)
        reader = name in _READERS or name.startswith(_READER_PREFIXES)
        root = _root(receiver)
        if self.owned(receiver):
            if isinstance(receiver, ast.Name) and receiver.id in self.fresh:
                return _NO_EFFECT
            return restructures | self._own_method(name, receiver)
        if isinstance(root, ast.Name) and (
            root.id in self.owners or root.id in self.fresh
        ):
            # What the owner or a made object holds: not its own.
            return restructures if reader else restructures | _ELSEWHERE
        if isinstance(root, ast.Name):
            holder = self.bindings.get(root.id, _UNKNOWN)
            if isinstance(holder, types.ModuleType):
                module = getattr(holder, "__name__", "")
                if (
                    module.split(".")[0] == "random"
                    or ".random" in module
                    or name == "random"
                ):
                    return _ELSEWHERE
                if receiver is not root:
                    return restructures | _ELSEWHERE
                if not _internal(getattr(holder, "__file__", "") or ""):
                    return writes.callee(
                        name, getattr(holder, name, _UNKNOWN), self.depth
                    )
                if name in _MUTATORS:
                    return _ELSEWHERE | (
                        self.into(call.args[0]) if call.args else _IN_PLACE
                    )
                return _NO_EFFECT
            if reader:
                return restructures
            from . import library

            if name in library.VALUE_METHODS and not _is_mobject(holder):
                return restructures
            if holder is not _UNKNOWN and holder is not _PARAMETER and receiver is root:
                member = inspect.getattr_static(type(holder), name, None)
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if isinstance(member, types.FunctionType) and _user_code(
                    member.__code__
                ):
                    return restructures | writes.effect(
                        member, frozenset(), None, self.depth + 1
                    )
            return restructures | _ELSEWHERE
        if isinstance(root, ast.Call):
            # A method of what a call returned: a new object when the call
            # makes one; else whatever it returned may be shared.
            if reader or writes._made(root, self.bindings):
                return _NO_EFFECT
            return restructures | _ELSEWHERE
        if isinstance(
            root,
            (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Dict, ast.ListComp),
        ):
            return _NO_EFFECT
        return restructures | _ELSEWHERE

    def _own_method(self, name: str, receiver: ast.AST) -> _Effect:
        """A method called on the owner's family: the scene's own method
        is read for what it does, with its ``self`` as the owner.
        """
        found = self._bound(receiver)
        if isinstance(found, _MEMBERS_OF):
            # Some member's: what each of their classes defines.
            for member in found.mobject.get_family():
                defined = inspect.getattr_static(type(member), name, None)
                code = _code_of(getattr(defined, "fget", defined))
                if code is not None and _user_code(code):
                    return _ANY_EFFECT
            return _NO_EFFECT
        if found is _UNKNOWN:
            return _NO_EFFECT
        member = inspect.getattr_static(type(found), name, None)
        if isinstance(member, (staticmethod, classmethod)):
            return _ANY_EFFECT
        if isinstance(member, property):
            member = member.fget
        if isinstance(member, types.FunctionType) and _user_code(member.__code__):
            first = member.__code__.co_varnames[:1]
            return self.writes.effect(member, frozenset(first), found, self.depth + 1)
        return _NO_EFFECT


def _parameters(function: types.FunctionType) -> tuple[str, ...]:
    code = function.__code__
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    return code.co_varnames[:count]


def _aliases(node: ast.AST) -> dict[str, ast.AST]:
    """The local names a function binds exactly once, each to a name or an
    attribute of one: what they hold is what that names.
    """
    bound: dict[str, list[ast.AST | None]] = {}
    for item in ast.walk(node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target, value = item.targets[0], item.value
            if isinstance(target, ast.Name):
                bound.setdefault(target.id, []).append(value)
                continue
        targets: list[ast.AST] = []
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
        elif isinstance(
            item,
            (
                ast.AugAssign,
                ast.AnnAssign,
                ast.NamedExpr,
                ast.For,
                ast.comprehension,
                ast.AsyncFor,
            ),
        ):
            targets = [item.target]
        elif isinstance(item, ast.With):
            targets = [w.optional_vars for w in item.items if w.optional_vars]
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and not (
                    isinstance(item, ast.AugAssign) and name is target
                ):
                    bound.setdefault(name.id, []).extend([None, None])
    return {
        name: values[0]
        for name, values in bound.items()
        if len(values) == 1
        and isinstance(values[0], (ast.Name, ast.Attribute))
        and isinstance(_root(values[0]), ast.Name)
        and _root(values[0]).id != name
    }


@dataclass(frozen=True)
class _MEMBERS_OF:  # noqa: N801 (a marker)
    """Some members of ``mobject`` (what indexing it gives)."""

    mobject: Any


def _quiet_family(mob: Any) -> bool:
    """Whether ``mob`` is a mobject none of whose members but itself carry
    updaters: padding it (``become``) copies nothing that runs.
    """
    if not _is_mobject(mob):
        return False
    return not any(member.updaters for member in mob.get_family()[1:])


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
    #: Whether it may change which updaters run or which mobjects there are.
    restructures: bool = False

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


def _family_writes(mob: Any) -> frozenset[int]:
    """What a change to ``mob``'s family may change: its members, and the
    memory of the arrays they hold, which another mobject's may share.
    """
    found = set(_family_ids(mob))
    for member in mob.get_family():
        for value in vars(member).values():
            if isinstance(value, np.ndarray):
                storage = _storage_key(value)
                if storage is not None:
                    found.add(storage)
    return frozenset(found)


def _describe(function: Any, base_dir: str | None) -> str:
    code = _code_of(function)
    if code is None:
        return repr(function)
    where = relative_path(code.co_filename, base_dir)
    return f"{code.co_name} at {where}:{code.co_firstlineno}"


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
    effect = _NO_EFFECT
    if isinstance(animation, trusted.CALLBACK_ANIMATIONS):
        function = vars(animation)["update_function"]
        effect = check.effect(function, _owners(function), animation.mobject)
        if effect.elsewhere:
            writes = _REACHED
    return _Node(
        type(animation).__name__,
        "transparent",
        lambda: reach([animation], trusted._mobjects_of(animation)),
        writes,
        restructures=effect.restructures,
    )


def _owners(function: Any) -> frozenset[str]:
    """The parameter naming the mobject a callback is handed."""
    try:
        return frozenset(list(inspect.signature(function).parameters)[:1])
    except (TypeError, ValueError):
        return frozenset()


def _updater_node(
    member: Any,
    updater: Any,
    check: _Writes,
    reaches: dict[tuple[int, int, int], frozenset[int]],
    base_dir: str | None,
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
        what = "redraw " + _describe(function, base_dir)
        effect = check.effect(function, frozenset())
        # ``become`` pads the target with copies of its members.
        effect |= _Effect(restructures=not _quiet_family(target))
    else:
        target, function = member, updater
        what = "updater " + _describe(function, base_dir)
        effect = check.effect(function, _owners(updater), target)

    def reads() -> frozenset[int]:
        # The mobject it is handed is read through too: ``m.source`` is as
        # much a read as a closure cell.
        found = (id(function), id(member), id(target))
        if found not in reaches:
            reaches[found] = reach([function], [member, target])
        return reaches[found]

    writes = _REACHED if effect.elsewhere else _family_writes(target)
    if loads_dt(updater):
        return _Node(
            what, "step", reads(), writes, (key,), restructures=effect.restructures
        )
    return _Node(
        what, "instant", reads, writes, (key,), restructures=effect.restructures
    )


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
            nodes.append(_updater_node(member, updater, check, reaches, base_dir))
        nodes.extend(
            _Node(
                "scene updater " + _describe(function, base_dir),
                "step",
                reach([function]),
                _REACHED,
                restructures=check.effect(function, frozenset()).restructures,
            )
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
        # Work a stepping tick runs must not change which updaters there are:
        # the tick would meet work no plan weighed, after skipping what it
        # reads, or keep running work gone. (Skipped work changes them only
        # at a full tick, after which the play plans again.)
        if any(node.role == "instant" for node in nodes):
            for node in nodes:
                if node.restructures and node.role != "instant":
                    return _structural(node)
        pulled = _closure(nodes)
        for node in nodes:
            if node.restructures and node.active:
                return _structural(node)
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
        stepped_ids=_stepped_ids(nodes),
        rechecks=any(node.restructures for node in nodes),
    )


def _structural(node: _Node) -> TickPlan:
    return TickPlan(
        unbounded=f"{STRUCTURAL}: {node.what} may add or remove updaters or mobjects"
    )


def _stepped_ids(nodes: list[_Node]) -> frozenset[int]:
    """What the work a stepping tick runs may read or write."""
    found: set[int] = set()
    for node in nodes:
        if node.active:
            found |= node.reads
        if node.active or node.role == "transparent":
            with contextlib.suppress(Unbounded):
                found |= node.written()
    return frozenset(found)
