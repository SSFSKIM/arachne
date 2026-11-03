"""Which work of a ticked play runs at a tick no frame reads: the stepping tick.

A ticked play (route 3) is evaluated on the 60 Hz tick, so that what
integrates over time does so the same way whatever the frame rate. Only that
work needs the tick (a dt updater receives exactly 1/60 at each), and
everything else is evaluated where CE evaluates it, at the instants a frame
reads. So a ticked play runs a **full tick** (every callback, in CE's phase
order) at each instant a frame of the render's grid reads, at its event
instants, at its last tick and throughout a play with a stop condition, and
every other tick only **steps**:

- the updaters whose own code loads ``dt`` (:func:`loads_dt`): one with a
  ``dt`` parameter it never loads is dt-free, frame-rate dependent in CE
  exactly like one without the parameter;
- the scene's own updaters;
- every animation's interpolation, the functions of ``UpdateFromFunc`` and
  ``UpdateFromAlphaFunc`` included;
- the camera's inputs (as every tick evaluates them).

No dt-free callback runs at a stepping tick, whoever reads what it writes: a
dt updater reading a dt-free value there sees it as of the last full tick,
as CE shows it once per frame. Each updater is judged as it is called
(:func:`skipping`, consulted by :meth:`.Mobject.update`), so one that joins
as the play runs is judged the same way. A play whose updaters do not run
through :meth:`.Mobject.update` (the scene's own ``update_mobjects`` or
``update_self``, a mobject's own ``update``, a camera drawing other cameras)
ticks everything, under the reason ``tick.unbounded_read``.

:func:`plan_ticks` makes the play's :class:`TickPlan`, which records what it
found: the updaters a stepping tick leaves out, and the structure
(:func:`current_structure`) it was made for. Before every stepping tick the
play compares the structure with its plan's and plans again when they
differ. Every plan a play stepped by is kept, with the tick it governed from
(``PlayRecord.tick_plans``).

Diagnostic only
---------------

The rest of this module (with :mod:`.handed` and :mod:`.coverage`) is an
analysis of what callbacks may read and write, which nothing that plans or
runs a play consults: a comparison of two runs asks it, through
:meth:`TickPlan.stepping_families` (worked out the first time it is asked)
and :func:`reach`.

Both bounds are default-deny: an access is bounded only along a path the
analysis resolves, and anything else counts as reaching anything. What a
callback may read is what its code can reach (:func:`reach`): its closure
cells, the globals its code names (one key per binding), its defaults, the
object a bound method holds, the mobject it updates with its family, and
what those hold, through the attributes its code names, on the instance or
on the scene's own class that holds them (keyed by that class), into
containers it follows whole, and into arrays by the memory they share
(``TracedPath``'s ``traced_point_func`` included). A module the scene
imports is read binding by binding, one key per attribute whatever alias it
is imported under. A name computed at run time (``getattr`` with a computed
name, ``getattr`` held under another name), or an object whose contents it
can't follow, is unbounded.

What a dt-free callback, or the function of an
``UpdateFromFunc``/``UpdateFromAlphaFunc``, may write is the mobject it
updates with its family (``always_redraw``'s target), unless its source
shows a change anywhere else (``_Writes``): a store or a call that reaches
something other than that family, or than what the callback made itself,
along a path it can't resolve to that family (an attribute the mobject
holds, an element of a list it made). Then it writes everything it reaches.
An array is keyed by the memory it shares with others; the engine's
constant vectors (``ORIGIN``, ``OUT``), which every mobject made with a
default holds, carry no key, and a write in place to one of them, or to an
array the write check can't resolve, is a write to every array.
"""

from __future__ import annotations

__all__ = [
    "UNBOUNDED",
    "TickPlan",
    "current_structure",
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
import pathlib
import random
import re
import sys
import types
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

from .coverage import (
    INERT_KINDS,
    INERT_RESULT_BUILTINS,
    NAMESPACE_ATTRIBUTES,
    NAMESPACE_BUILTINS,
    NATIVE_HOLDERS,
    Rule,
    inert_value,
    item_kind,
    rule_of,
)
from .frozen import engine_class, engine_code
from .source import _internal, relative_path

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

#: The reason family of a play that ticks everything, its updaters not
#: running where a stepping tick could leave one out.
UNBOUNDED = "tick.unbounded_read"

#: How many objects a reach follows before it gives up.
REACH_LIMIT = 50_000

#: How deep the write check follows calls into the scene's helpers.
MAX_DEPTH = 4

#: Names whose use reads or writes by a name the code does not spell.
_DYNAMIC = (
    frozenset(
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
    | NAMESPACE_ATTRIBUTES
    | NAMESPACE_BUILTINS
)

#: Builtins making a container or an iterator: iterating what one gives runs
#: only what the call's arguments are judged for where they are handed.
_ITERATOR_BUILTINS: tuple[Any, ...] = (
    range,
    enumerate,
    zip,
    map,
    filter,
    reversed,
    sorted,
    list,
    tuple,
    set,
    frozenset,
    dict,
    iter,
)

#: Kinds whose iteration runs no code but the interpreter's: an exact native
#: holder, a string, a range, a dict's view, a NumPy array.
_QUIET_ITERABLES: tuple[type, ...] = (
    *NATIVE_HOLDERS,
    str,
    bytes,
    range,
    type({}.keys()),
    type({}.values()),
    type({}.items()),
    np.ndarray,
)


def _iterates_quietly(value: Any) -> bool:
    """Whether iterating ``value`` (resolved) runs no code of the scene's:
    an exact :data:`_QUIET_ITERABLES` kind, or an object of the engine's
    class (whose iteration is the engine's code).
    """
    return type(value) in _QUIET_ITERABLES or engine_class(type(value))


#: Methods a class runs as it makes or names its objects, never on one in use.
_CONSTRUCTION = frozenset(
    {
        "__init__",
        "__new__",
        "__init_subclass__",
        "__set_name__",
        "__class_getitem__",
        "__subclasshook__",
        "__post_init__",
        "__del__",
    }
)


def _protocol_code_of(cls: type) -> bool:
    """Whether ``cls`` defines, in the scene's code, a method Python runs
    on its objects by syntax (an operator's, a truth test's, a formatting's,
    a hash's, an attribute hook's).
    """
    return any(
        name.startswith("__") and name.endswith("__") and name not in _CONSTRUCTION
        for name in _user_methods(cls)
    )


def _protocol_code(value: Any, truth: bool = False) -> bool:
    """Whether an operator, a formatting or (with ``truth``) a truth test on
    ``value`` (resolved) may run the scene's code: its class's own
    (:func:`_protocol_code_of`), or its items' for a container (a
    comparison or a formatting reads them; a truth test only counts them)
    and an array of objects.
    """
    if type(value) is np.ndarray or isinstance(value, np.generic):
        return bool(value.dtype.hasobject)
    if type(value) in (tuple, frozenset):
        return not truth and any(_protocol_code(item) for item in value)
    if type(value) in NATIVE_HOLDERS:
        return not truth
    return _protocol_code_of(type(value))


def _reading_runs_code(value: Any, name: str) -> bool:
    """Whether reading attribute ``name`` of ``value`` (resolved) may run the
    scene's code: its class's ``__getattribute__`` or ``__getattr__``, a
    property's getter, or a descriptor whose ``__get__`` or the function it
    wraps is the scene's.
    """
    methods = _user_methods(type(value))
    if "__getattribute__" in methods or "__getattr__" in methods:
        return True
    member = inspect.getattr_static(type(value), name, None)
    if member is None or isinstance(
        member,
        (types.FunctionType, staticmethod, classmethod, types.BuiltinFunctionType),
    ):
        return False
    getter = inspect.getattr_static(type(member), "__get__", None)
    if getter is None:
        return False
    held = [
        getter,
        *(getattr(member, part, None) for part in ("fget", "func", "__wrapped__")),
    ]
    return any(
        isinstance(each, types.FunctionType) and _user_function(each) for each in held
    )


@dataclass(frozen=True)
class TickPlan:
    """What a ticked play found to leave out at its stepping ticks."""

    #: ``(id(mobject), id(updater))`` of every updater found whose code does
    #: not load ``dt``, which a stepping tick leaves out (each is judged as
    #: it is called: one that joins later is left out alike).
    skipped: frozenset[tuple[int, int]] = frozenset()
    #: Every updater the plan found.
    known: frozenset[tuple[int, int]] = frozenset()
    #: Why the play ticks everything, when it does (``tick.unbounded_read: …``).
    unbounded: str | None = None
    #: How many of the updaters found step (their code loads ``dt``).
    stepping: int = 0
    #: Diagnostic only: what :meth:`stepping_families` reads, worked out
    #: the first time it is asked, never while the play is planned or run.
    families: Callable[[], frozenset[int]] | None = field(
        default=None, compare=False, repr=False
    )
    #: The structure the plan was made for (:func:`current_structure`): a
    #: stepping tick that finds another plans again before it steps.
    structure: tuple[int, ...] = ()
    #: Whether the play stepped by it. A run that ticks everything
    #: (``Recorder(stepping=False)``) makes and remakes its plans where a
    #: stepping run would, and applies none: every tick stays full.
    applied: bool = True

    def stepping_families(self) -> frozenset[int]:
        """Diagnostic only, for a comparison of two runs: the ids of
        everything the stepping set (dt updaters, the scene's updaters, every
        interpolation, the camera's inputs) may read or write, each mobject
        among them with its family, and the containers, cells and bindings
        they hold, by the analysis of their code
        (:func:`_stepping_families`). Worked out the first time it is asked,
        from the objects as they are then; nothing that plans or runs a play
        asks it.
        """
        return frozenset() if self.families is None else self.families()

    @property
    def steps_only(self) -> bool:
        """Whether a stepping tick does less than a full one (when the plan
        is :attr:`applied`).
        """
        return self.unbounded is None


@contextlib.contextmanager
def skipping(plan: TickPlan) -> Iterator[None]:
    """Run a stepping tick: :meth:`.Mobject.update` runs only the updaters
    whose code loads ``dt`` (:func:`loads_dt`), decided as each runs, so an
    updater that joined since the plan was made is judged the same way.
    """
    from arachne.mobject.mobject import Mobject

    Mobject._tick_steps = loads_dt
    try:
        yield
    finally:
        Mobject._tick_steps = None


def current_structure(scene: Scene, animations: Iterable[Animation]) -> tuple[int, ...]:
    """What a plan weighs that callbacks may change as a play runs: every
    updater the scene's tick may run (the scene's own and every mobject's),
    and every mobject's place, under the scene or in its parent (a member
    joining a family another callback writes changes what a reader of the
    family reads). Every stepping tick compares it with its plan's.
    """
    # One walk, in preorder: each mobject's id, its updaters (counted), how
    # many members it has, then each member's walk. Cheap enough for every
    # stepping tick: no family lists, no sets.
    found: list[int] = [len(scene.updaters), *map(id, scene.updaters)]
    pending = list(reversed((*_copies(animations), *scene.mobjects)))
    found.append(len(pending))
    while pending:
        mob = pending.pop()
        updaters, members = mob.updaters, mob.submobjects
        found.append(id(mob))
        found.append(len(updaters))
        found.extend(map(id, updaters))
        found.append(len(members))
        pending.extend(reversed(members))
    return tuple(found)


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
    if _FRAME_ACCESS & set(code.co_names):
        return True  # it may read its own frame's locals by name
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

#: Names through which code may read its own frame's locals without loading
#: them: ``locals()``, ``vars()``, ``eval``/``exec`` (their default
#: namespace), and frame objects (``sys._getframe``, ``inspect`` helpers,
#: ``f_locals``).
_FRAME_ACCESS = frozenset(
    {
        "locals",
        "vars",
        "eval",
        "exec",
        "_getframe",
        "currentframe",
        "stack",
        "getouterframes",
        "f_locals",
    }
)


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


class _UnboundedWrites(Unbounded):
    """What a callback may write can't be bounded without what objects hold
    when it is asked, which may change as the play runs: it may write any
    member of the play's families (:func:`_stepping_families`).
    """


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
    pathlib.PurePath,
    np.generic,
    np.dtype,
    # A compiled regular expression: its pattern, flags and group names are
    # fixed, its group index a read-only view, and it takes no attributes.
    re.Pattern,
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
    def __init__(self, every_attribute: bool = False) -> None:
        #: Follow every attribute of what it meets, not only those code names:
        #: what a change into an object may reach (:func:`_written_into`).
        self.every_attribute = every_attribute
        self.ids: set[int] = set()
        self.names: set[str] = set()
        self._seen: set[int] = set()
        self._codes: set[int] = set()
        #: Objects followed only through the attributes code names: revisited
        #: as more names turn up.
        self._filtered: dict[int, tuple[Any, frozenset[str]]] = {}
        self._count = 0
        #: Whether code it reaches may change the structure: the scene's code
        #: naming a way to, or a method doing so held as a value.
        self.restructures = False
        #: The code of the callables reached from: what it does itself the
        #: write check reads.
        self.roots: set[int] = set()
        #: The stand-in ids of the module globals it reaches, and their names.
        self.global_keys: set[int] = set()
        self.global_names: set[str] = set()
        #: Whether code it reaches may rebind a module global: a ``global``
        #: statement's store or deletion, or an attribute stored (or deleted,
        #: or set by ``setattr``) under a global's name, as through a module.
        self.rebinds = False
        self.stored_attributes: set[str] = set()
        #: Whether it met writable memory no key names (the engine's constants).
        self.unkeyed_arrays = False

    def done(self) -> frozenset[int]:
        while True:
            pending = [
                obj for obj, names in self._filtered.values() if not self.names <= names
            ]
            if not pending:
                found = _Reached(self.ids)
                found.restructures = self.restructures
                found.globals = frozenset(self.global_keys)
                found.rebinds = self.rebinds or bool(
                    self.stored_attributes & self.global_names
                )
                return found
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
            if value.__name__ in _RESTRUCTURERS:
                self.restructures = True
            owner = getattr(value, "__self__", None)
            if owner is not None and not isinstance(owner, types.ModuleType):
                self.visit(owner)
        elif isinstance(value, types.CellType):
            self.ids.add(key)
            with contextlib.suppress(ValueError):  # an empty cell
                self.visit(value.cell_contents)
        elif isinstance(value, types.ModuleType):
            filename = getattr(value, "__file__", None)
            if filename and not _internal(filename) and not self.every_attribute:
                # (An object holding a module does not hold its bindings.)
                self._attributes(value)
        elif isinstance(value, type):
            if not _internal(getattr(inspect.getmodule(value), "__file__", "") or ""):
                self.ids.add(key)  # its class attributes are shared state
                for member in vars(value).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if isinstance(member, (types.FunctionType, property)):
                        self.visit(member)
        elif isinstance(value, np.ndarray):
            if _array_root(value)[1]:
                self.ids.add(_ANY_ARRAY)
            storage = _storage_key(value)
            if storage is not None:
                self.ids.add(storage)
            elif _array_root(value)[1]:
                self.unkeyed_arrays = True
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
            if isinstance(value, collections.defaultdict):
                # A missing item runs its factory.
                self.visit(value.default_factory)
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
            # Reading, assigning and deleting it each run code of its own.
            self.visit(value.fget)
            self.visit(value.fset)
            self.visit(value.fdel)
        elif engine_class(type(value)):
            return  # the engine's values (colours, paints, frozen arrays)
        elif isinstance(value, _OPAQUE_STATE):
            self.ids.add(key)  # its own state, which holds no mobject
        elif hasattr(value, "__dict__") or getattr(type(value), "__slots__", None):
            # A plain object (the scene's own helper): shared state, followed
            # through the attributes code names, and what its class runs
            # unnamed: calling it, and reaching its attributes (a descriptor's
            # or the class's own hooks).
            self.ids.add(key)
            self._attributes(value)
            methods = _user_methods(type(value))
            for hook in _IMPLICIT_HOOKS & methods.keys():
                self.visit(methods[hook])
            if self.every_attribute:
                # What its class runs on it: its methods and properties.
                self.visit(type(value))
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
        holds; keyed by every one of the scene's classes the lookup passes
        on the way (a store to any of them, ``Child.name = …`` shadowing
        ``Parent.name``, rebinds what the instance reads).
        """
        for klass in cls.__mro__:
            if not _scene_class(klass):
                continue
            self.ids.add(id(klass))
            if name in vars(klass):
                value = vars(klass)[name]
                if isinstance(value, property):
                    self.visit(value)
                    return
                if name.startswith("__") or isinstance(
                    value, (types.FunctionType, staticmethod, classmethod)
                ):
                    return  # a method: _user_methods follows it
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
        if self.every_attribute and not module:
            names = set(own) | names
        for name in names:
            if name in own:
                if module:
                    # The binding itself, as a global's: one key for every
                    # alias the module is imported under.
                    self._global(own, name)
                self.visit(own[name])
            elif not module:
                self._class_attribute(type(obj), name)
        if isinstance(obj, Scene):
            if self.every_attribute or names & {"camera", "renderer"}:
                self.visit(obj.renderer.camera)
            if self.every_attribute or names & {
                "mobjects",
                "foreground_mobjects",
                "get_mobject_family_members",
            }:
                self.visit(list(obj.mobjects))
                self.visit(list(obj.foreground_mobjects))
            if self.every_attribute or "updaters" in names:
                self.visit(list(obj.updaters))
        if not isinstance(obj, types.ModuleType):
            methods = _user_methods(type(obj))
            for name in names & methods.keys():
                self.visit(methods[name])

    def callable(self, function: Any) -> None:
        if getattr(function, "__name__", None) in _RESTRUCTURERS:
            self.restructures = True  # held as a value: a call of it does
        if isinstance(function, types.MethodType):
            self.visit(function.__self__)
            self.visit(function.__func__)
            return
        code = function.__code__
        if not engine_code(function):
            # Its attributes (``follow.bias``) are state a store may change:
            # the function is keyed, and what they hold followed.
            self.ids.add(id(function))
            for item in vars(function).values():
                self.visit(item)
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
        if names & _STRUCTURE_NAMES and id(code) not in self.roots:
            self.restructures = True
        rebinds, stored = _rebinding(code)
        self.rebinds = self.rebinds or rebinds
        self.stored_attributes |= stored
        if names & {"setattr", "delattr"}:
            self.stored_attributes |= names  # with the names it sets by
        namespace = function.__globals__
        for name in names:
            # The binding itself, which a ``global`` statement rebinds.
            self._global(namespace, name)
            if name in namespace:
                self.visit(namespace[name])

    def _global(self, namespace: dict[str, Any], name: str) -> None:
        key = _global_key(namespace, name)
        self.ids.add(key)
        self.global_keys.add(key)
        if name in namespace:
            # A binding an attribute stored through its module rebinds.
            self.global_names.add(name)


@functools.cache
def _rebinding(code: types.CodeType) -> tuple[bool, frozenset[str]]:
    """Whether ``code`` (or code nested in it) stores or deletes a module
    global by a ``global`` statement, and the attributes it stores or
    deletes (a module's among them, which rebinds the global).
    """
    rebinds = False
    stored: set[str] = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
            rebinds = True
        elif instruction.opname in ("STORE_ATTR", "DELETE_ATTR"):
            stored.add(instruction.argval)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            inner, names = _rebinding(constant)
            rebinds = rebinds or inner
            stored |= names
    return rebinds, frozenset(stored)


def _scene_class(klass: type) -> bool:
    """Whether ``klass`` is defined by the scene's code."""
    module = sys.modules.get(klass.__module__)
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str) or not filename:
        return False
    return not engine_class(klass) and not _internal(filename)


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


#: Every node under a source node, in ``ast.walk`` order, by the source
#: node's id (kept with it): a play plans again as often as its structure
#: changes, and each plan reads each callback's source several times.
_WALKS: dict[int, tuple[ast.AST, tuple[ast.AST, ...]]] = {}


def _walk(node: ast.AST) -> tuple[ast.AST, ...]:
    found = _WALKS.get(id(node))
    if found is None or found[0] is not node:
        found = (node, tuple(ast.walk(node)))
        _WALKS[id(node)] = found
    return found[1]


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
    calls = {id(c.func): c for c in _walk(node) if isinstance(c, ast.Call)}
    found: set[str] = set()
    for item in _walk(node):
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


class _Reached(frozenset[int]):
    """The ids a reach found; ``restructures`` says whether code it reaches
    may change the structure (what a call the write check can't resolve may
    run is in it); ``globals`` holds the stand-in ids of the module globals
    among them, and ``rebinds`` whether code it reaches may rebind one.
    """

    restructures: bool = False
    globals: frozenset[int] = frozenset()
    rebinds: bool = True


def reach(callables: Iterable[Any], mobjects: Iterable[Any] = ()) -> frozenset[int]:
    """The ids of everything ``callables`` may read, with ``mobjects``
    (what they update) and their families: mobjects, mutable containers and
    arrays, closure cells, global namespaces and plain objects. Raises
    :class:`Unbounded` when it can't be bounded.
    """
    walk = _Reach()
    callables = list(callables)
    for function in callables:
        code = _code_of(function)
        if code is not None:
            walk.roots.add(id(code))
    for mob in mobjects:
        walk.visit(mob)
    for function in callables:
        walk.visit(function)
    return walk.done()


def _written_into(targets: Iterable[_Ref]) -> frozenset[int]:
    """What a change into each of ``targets``, or into anything it holds,
    may change: everything reachable from it, through every attribute. Not
    a module global's binding (a change into an object rebinds none), and
    not every array (:data:`_ANY_ARRAY`): an array it holds is changed
    under its own key, and the engine's constants it holds (as every
    mobject does) only by a write in place the write check could not
    resolve, which names :data:`_IN_PLACE_UNRESOLVED`.
    """
    targets = list(targets)
    walk = _Reach(every_attribute=True)
    for target in targets:
        if target is _IN_PLACE_UNRESOLVED:
            walk.ids.add(_ANY_ARRAY)
        elif not isinstance(target, _CodeRef):
            walk.visit(target.value)
            walk.ids.add(id(target.value))
    found = walk.done()
    # What a function (or a class's code) read, and so may have put into
    # what it made, is what it reads through the names its code uses.
    read_by = _Reach()
    for target in targets:
        if isinstance(target, _CodeRef):
            read_by.visit(target.value)
    read = read_by.done()
    dropped = set(getattr(found, "globals", ())) | set(getattr(read, "globals", ()))
    dropped.add(_ANY_ARRAY)
    if _IN_PLACE_UNRESOLVED in targets:
        dropped.discard(_ANY_ARRAY)
    return frozenset((found | read) - dropped)


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
#: Builtins that change nothing but what they return. Any other changes
#: something (``next`` an iterator, ``open`` a file), or anything.
_PURE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "callable",
        "chr",
        "divmod",
        "format",
        "hasattr",
        "hash",
        "hex",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "max",
        "min",
        "oct",
        "ord",
        "pow",
        "print",
        "repr",
        "round",
        "sorted",
        "sum",
    }
)
#: Modules whose C functions change nothing but what they return, but for
#: :data:`_MUTATORS` (and, in :mod:`operator`, its in-place forms).
_PURE_MODULES = frozenset({"math", "cmath", "numpy", "operator"})
#: :mod:`operator`'s functions that store into their first argument.
_OPERATOR_STORES = frozenset(
    {
        "setitem",
        "delitem",
        "iadd",
        "iand",
        "iconcat",
        "ifloordiv",
        "ilshift",
        "imatmul",
        "imod",
        "imul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
    }
)
#: The methods of builtin containers that change nothing, by the class
#: that defines them: any other changes the object it is bound to.
_PURE_METHODS: dict[type, frozenset[str]] = {
    list: frozenset(
        {"copy", "count", "index", "__getitem__", "__len__", "__contains__"}
    ),
    dict: frozenset(
        {
            "copy",
            "get",
            "items",
            "keys",
            "values",
            "__getitem__",
            "__len__",
            "__contains__",
        }
    ),
    set: frozenset(
        {
            "copy",
            "difference",
            "intersection",
            "isdisjoint",
            "issubset",
            "issuperset",
            "symmetric_difference",
            "union",
            "__contains__",
            "__len__",
        }
    ),
    np.ndarray: frozenset(
        {
            "all",
            "any",
            "argmax",
            "argmin",
            "astype",
            "conj",
            "conjugate",
            "copy",
            "cumsum",
            "diagonal",
            "dot",
            "flatten",
            "item",
            "max",
            "mean",
            "min",
            "nonzero",
            "prod",
            "ravel",
            "reshape",
            "squeeze",
            "std",
            "sum",
            "tolist",
            "trace",
            "transpose",
            "var",
            "view",
            "__getitem__",
            "__len__",
            "__contains__",
        }
    ),
}
#: Builtin values no method changes.
_IMMUTABLE_NATIVE = (str, bytes, tuple, frozenset, int, float, complex, range)


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
    #: It calls what its source can't resolve (or what it can't read): that
    #: changes the structure if anything its code reaches may.
    unknown: bool = False
    #: A change elsewhere whose object is not resolved: it may be anything
    #: the callback reaches. Implied by ``elsewhere`` with no object named.
    anywhere: bool = False
    #: The objects its changes elsewhere go into, each with everything it
    #: holds, when every one of them is resolved.
    targets: frozenset[_Ref] = frozenset()
    #: It changes what its caller hands it: the parameters (by name;
    #: :data:`_EVERY_ARGUMENT` for all of them) whose objects it may change,
    #: which a call site resolves to the arguments it binds to them.
    writes_args: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.elsewhere and not (self.targets or self.writes_args):
            object.__setattr__(self, "anywhere", True)

    def __or__(self, other: _Effect) -> _Effect:
        return _Effect(
            self.elsewhere or other.elsewhere,
            self.restructures or other.restructures,
            self.arrays or other.arrays,
            self.unknown or other.unknown,
            self.anywhere or other.anywhere,
            self.targets | other.targets,
            self.writes_args | other.writes_args,
        )

    def without_args(self) -> _Effect:
        """The effect with its writes into parameters taken out, for a call
        site to put back as writes into its own arguments.
        """
        return _Effect(
            self.elsewhere and (self.anywhere or bool(self.targets)),
            self.restructures,
            self.arrays,
            self.unknown,
            self.anywhere,
            self.targets,
        )


class _Ref:
    """An object a change goes into, compared by identity."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Ref) and other.value is self.value

    def __hash__(self) -> int:
        return id(self.value)


#: The target of a write in place to an array (or a mobject's members)
#: the write check could not resolve: any array.
_IN_PLACE_UNRESOLVED = _Ref(object())


class _CodeRef(_Ref):
    """Everything a function (or a class's code) reads: what it may have put
    into an object it made and handed back.
    """

    __slots__ = ()


#: In ``writes_args``: every parameter, as a call site binds none apart.
_EVERY_ARGUMENT = "*"

#: A made object's source standing for what its maker was handed (a
#: constructor's arguments, as its ``self`` is made), which a call site
#: resolves to its own arguments.
_ARGUMENTS = object()


def _no_call_site(effect: _Effect) -> _Effect:
    """``effect`` where no call site hands the parameters (a hook a store
    runs): what it changes through them may be anything.
    """
    if not effect.writes_args:
        return effect
    return effect.without_args() | _ELSEWHERE


def _code_writes(code: Any) -> _Effect:
    return _Effect(elsewhere=True, targets=frozenset({_CodeRef(code)}))


def _into(value: Any) -> _Effect:
    """A change into ``value`` (a resolved object) or anything it holds."""
    if isinstance(value, _MEMBERS_OF):
        value = value.mobject
    if value is None or (
        isinstance(value, _IMMUTABLE_NATIVE)
        and not isinstance(value, (tuple, frozenset))
    ):
        return _NO_EFFECT  # a number or a string: nothing changes in it
    if isinstance(value, (types.ModuleType, type)):
        # A module's bindings and a class's attributes are read through
        # names, not ids: a change there may be to anything reached.
        return _ELSEWHERE
    if isinstance(value, np.ndarray):
        if not _array_root(value)[1]:
            return _NO_EFFECT  # memory nothing can write
        if _storage_key(value) is None:
            return _IN_PLACE  # the engine's constant
    return _Effect(elsewhere=True, targets=frozenset({_Ref(value)}))


_NO_EFFECT = _Effect()
_ELSEWHERE = _Effect(elsewhere=True)
_ANY_EFFECT = _Effect(elsewhere=True, restructures=True, arrays=True)
_IN_PLACE = _Effect(elsewhere=True, arrays=True)
_UNKNOWN_CALL = _Effect(elsewhere=True, arrays=True, unknown=True)

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

#: Builtins that hand back one of their arguments, an item of one, or a
#: new list of such items.
_HANDING_BACK = (min, max, sorted, iter, next)

#: numpy.linalg functions that return a new value.
_LINALG_VALUES = frozenset(
    {
        "norm",
        "det",
        "inv",
        "solve",
        "pinv",
        "eigvals",
        "eigvalsh",
        "matrix_rank",
        "cond",
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
#: Restructurers that change the lists of the object they are called on
#: (its members, its updaters) and no member's: on an object the callback
#: made, a structure no other code holds.
_OWN_LISTS_ONLY = frozenset(
    {"add", "add_to_back", "insert", "remove", "remove_updater"}
)
#: The lists that are a mobject's or the scene's structure.
_STRUCTURE_LISTS = frozenset(
    {"updaters", "submobjects", "mobjects", "foreground_mobjects", "moving_mobjects"}
)
#: The names code that changes the structure uses.
_STRUCTURE_NAMES = _RESTRUCTURERS | _STRUCTURE_LISTS
#: The methods of an object's class that run with no name in the code that
#: sets them off: a call of the object, its attributes' reads, stores and
#: deletions (the class's own, or a descriptor's it holds).
_IMPLICIT_HOOKS = frozenset(
    {
        "__call__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__get__",
        "__set__",
        "__delete__",
    }
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
        self._effects: dict[
            tuple[int, frozenset[str], int, int, int], tuple[Any, Any, Any, _Effect]
        ] = {}
        #: The hooks a store into an attribute of a class runs, and those
        #: any store into one of its instances may: classes keep their
        #: attributes while one plan is made.
        self._hooks: dict[tuple[type, str, bool], list[tuple[Any, Any]]] = {}
        self._class_hooks: dict[type, list[tuple[Any, Any]]] = {}
        #: Whether each of the scene's functions hands back an object it made,
        #: by its identity (the function is kept so that its id is not reused).
        self._returns: dict[int, tuple[Any, bool]] = {}
        #: What a change into what each of them returns may change, by its
        #: identity and the object it is called on (both kept).
        self._returned: dict[tuple[int, int], tuple[Any, Any, _Effect]] = {}
        #: The module globals the plan read, and the attributes it read
        #: through a class (absent from the object), by name; and the
        #: attributes reached code stores a value not inert into, on objects
        #: not made by the function, each with why (:meth:`plan_reads_hold`).
        self.global_reads: set[str] = set()
        self.class_reads: set[str] = set()
        self.loose_stores: dict[str, str] = {}
        #: Why code the play reaches is not read, when some is: it may rebind
        #: what the plan read (:meth:`plan_reads_hold`).
        self.unread: str | None = None
        from .handed import Callers

        #: Which parameters each function may call (:mod:`.handed`).
        self.callers = Callers(self._bindings, MAX_DEPTH)

    def plan_reads_hold(self) -> None:
        """Raise :class:`Unbounded` unless what the plan read when it was
        made stands for the whole play: it read no value of a mutable kind
        (:attr:`.handed.Callers.snapshots`: what a list, a dict, an object
        or a class held), whatever could change it; it read all the code
        the play reaches; and no reached store into an attribute may rebind
        a module's global or a class's attribute it read, or shadow one, by
        that name.
        """
        for value, _ in self.callers.snapshots.values():
            if not inert_value(value):
                raise Unbounded(
                    f"the plan reads what a {type(value).__name__} holds, which"
                    " may change as the play runs"
                )
        if self.unread is not None:
            raise Unbounded(self.unread)
        read = self.global_reads | self.class_reads
        for name in sorted(read & self.loose_stores.keys()):
            raise Unbounded(
                f"{self.loose_stores[name]}, which may rebind the {name} the plan read"
            )

    def assignment_hooks(
        self, cls: type, name: str, deleting: bool
    ) -> list[tuple[Any, Any]]:
        """:func:`_assignment_hooks`, remembered for this plan."""
        key = (cls, name, deleting)
        found = self._hooks.get(key)
        if found is None:
            found = self._hooks[key] = _assignment_hooks(cls, name, deleting)
        return found

    def class_hooks(self, cls: type) -> list[tuple[Any, Any]]:
        """:func:`_class_hooks`, remembered for this plan."""
        found = self._class_hooks.get(cls)
        if found is None:
            found = self._class_hooks[cls] = _class_hooks(cls)
        return found

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
        this: Any = None,
        made_type: type | None = None,
    ) -> _Effect:
        """``made`` names parameters bound to an object being made (a
        constructor's ``self``, of class ``made_type``): its own, as what
        the function makes is. ``this`` is the object a method is called on,
        which its first parameter resolves to (not its own: what it changes
        there is elsewhere).
        """
        if isinstance(function, types.MethodType):
            this = function.__self__
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
                restructures=bool(names & _STRUCTURE_NAMES),
            )
        key = (id(function), owners | made, id(owner), id(this), id(made_type))
        found = self._effects.get(key)
        if found is not None:
            return found[3]
        # Calls back into itself add nothing to what it is found to do. The
        # objects keyed by id are kept, so that no id is reused.
        self._effects[key] = (function, owner, (this, made_type), _NO_EFFECT)
        if depth > MAX_DEPTH:
            # Code not read may store anything anywhere, and rebind anything.
            self.unread = self.unread or f"{function.__qualname__} is not read"
            effect = _UNKNOWN_CALL
        else:
            effect = self._scan(function, owners, owner, depth, made, this, made_type)
        self._effects[key] = (function, owner, (this, made_type), effect)
        return effect

    def _scan(
        self,
        function: Any,
        owners: frozenset[str],
        owner: Any,
        depth: int,
        made: frozenset[str],
        this: Any,
        made_type: type | None,
    ) -> _Effect:
        found = self._site(function, owners, owner, depth, made, this, made_type)
        if found is None:
            # Its code's names are the reach's to read; its stores, not read,
            # may go anywhere and rebind anything.
            self.unread = self.unread or (
                f"{function.__qualname__}'s source is not read"
            )
            return _UNKNOWN_CALL
        node, site = found
        effect = site.read_table(node)
        for item in _walk(node):
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
                if isinstance(item, ast.AugAssign) and isinstance(
                    item.target, ast.Subscript
                ):
                    effect |= site.item_in_place(item.target.value)
            elif isinstance(item, ast.Delete):
                targets = list(item.targets)
            for target in targets:
                effect |= site.store(target, deleting=isinstance(item, ast.Delete))
            if isinstance(item, ast.Call):
                effect |= site.call(item)
            if effect == _ANY_EFFECT:
                break
        return effect

    def returns(self, function: types.FunctionType, this: Any = None) -> _Effect:
        """What a change into what ``function`` (of the scene's) returns, or
        into anything that holds, may change; ``writes_args`` for what its
        caller hands it. ``this`` is the object a method is called on.
        """
        key = (id(function), id(this))
        found = self._returned.get(key)
        if found is not None:
            return found[2]
        # A call back into itself: what it returns may be anything.
        self._returned[key] = (function, this, _ELSEWHERE)
        effect = self._scan_returns(function, this)
        self._returned[key] = (function, this, effect)
        return effect

    def _scan_returns(self, function: types.FunctionType, this: Any) -> _Effect:
        found = self._site(function, frozenset(), None, 0, frozenset(), this, None)
        if found is None:
            return _ELSEWHERE
        node, site = found
        if isinstance(node, ast.Lambda):
            return site.expr_writes(node.body)
        own = _own_nodes(node)
        if any(isinstance(item, (ast.Yield, ast.YieldFrom, ast.Await)) for item in own):
            return _ELSEWHERE  # what it hands back is no value of a return
        effect = _NO_EFFECT
        for item in own:
            if isinstance(item, ast.Return) and item.value is not None:
                effect |= site.expr_writes(item.value)
        return effect

    def _site(
        self,
        function: Any,
        owners: frozenset[str],
        owner: Any,
        depth: int,
        made: frozenset[str],
        this: Any,
        made_type: type | None,
    ) -> tuple[ast.AST, _Site] | None:
        """``function``'s source and the site reading it; ``None`` when its
        source can't be read.
        """
        from . import purity

        try:
            node = _source_node(function)
        except purity.Refused:
            return None
        bindings = self._bindings(function)
        # A parameter of a function it defines is that function's own, bound
        # to whatever it is handed, under a name its own code may also use.
        # A comprehension's variable, likewise, is its own; and a store to
        # the owner's name binds it to something else.
        nested = _nested_parameters(node)
        bindings.update(dict.fromkeys(nested, _UNKNOWN))
        first = function.__code__.co_varnames[:1]
        if this is not None and first and first[0] not in nested:
            bindings[first[0]] = this
        rebound = {
            item.id
            for item in _walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
        }
        site = _Site(self, owners - nested - rebound, owner, bindings, depth)
        parameters = set(_parameters(function))
        # A parameter holds what it was handed until a store rebinds it.
        site.fresh = (self._fresh(node, site.bindings) - parameters) | made
        site.made_from = _bound_values(node, site.fresh)
        site.made_type = made_type
        site.values = self._fresh(node, site.bindings, self._new_value) - parameters
        site.aliases = {
            name: value
            for name, value in _aliases(node).items()
            if name not in parameters
        }
        site.value_from = _bound_values(node, site.values)
        site.fresh -= nested - made
        site.values -= nested
        site.aliases = {
            name: value for name, value in site.aliases.items() if name not in nested
        }
        site.local_from = _local_values(node)
        for name in parameters:
            # What it was handed, and what any store binds it to.
            site.local_from.setdefault(name, []).append(_PARAMETER)
        code = function.__code__
        site.code = code
        site.first = code.co_varnames[0] if code.co_argcount else None
        site.parameters = frozenset(parameters - rebound - nested)
        site.bound = this is not None and site.first not in nested
        site.sources = {
            name: [source for value in values for source in site.made_sources(value)]
            for name, values in site.made_from.items()
        }
        for name in made:
            # The object a constructor makes holds what it was handed, and
            # what the class's code reads.
            site.sources[name] = [_ARGUMENTS, _CodeRef(function)]
        return node, site

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
        for item in _walk(node):
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
        """Whether ``value`` is an object the expression makes: a container,
        a copy, an object a class makes, one a function of the scene's makes
        and hands back. What it holds may still be shared (its sources:
        :meth:`_Site.made_sources`).
        """
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
            target: Any = _UNKNOWN
            if isinstance(func, ast.Name):
                target = bindings.get(func.id, _UNKNOWN)
            elif isinstance(func, ast.Attribute):
                holder = _module_named(func.value, bindings)
                if holder is not None:
                    target = getattr(holder, func.attr, _UNKNOWN)
            return self.makes(target)
        return False

    def makes(self, target: Any) -> bool:
        """Whether calling ``target`` hands back a new object."""
        if any(target is maker for maker in _FRESH_RANDOM):
            return True
        if isinstance(target, type):
            # A ``__new__`` of the scene's own may hand back one it keeps.
            return not _user_function(inspect.getattr_static(target, "__new__", None))
        return _user_function(target) and self.returns_made(target)

    def returns_made(self, function: types.FunctionType) -> bool:
        """Whether every value ``function`` returns is an object it made (or
        ``None``): it is no generator, and each ``return`` gives a made value
        or a local name only ever bound to one.
        """
        found = self._returns.get(id(function))
        if found is not None:
            return found[1]
        self._returns[id(function)] = (function, False)  # a call back into itself
        result = self._returns_made(function)
        self._returns[id(function)] = (function, result)
        return result

    def _returns_made(self, function: types.FunctionType) -> bool:
        from . import purity

        try:
            node = _source_node(function)
        except purity.Refused:
            return False
        bindings = self._bindings(function)
        own = _own_nodes(node)
        if any(isinstance(item, (ast.Yield, ast.YieldFrom, ast.Await)) for item in own):
            return False
        fresh = self._fresh(node, bindings)
        values: list[ast.AST | None] = (
            [node.body]
            if isinstance(node, ast.Lambda)
            else [item.value for item in own if isinstance(item, ast.Return)]
        )

        def made(value: ast.AST | None) -> bool:
            if value is None or (
                isinstance(value, ast.Constant) and value.value is None
            ):
                return True
            if isinstance(value, ast.IfExp):
                return made(value.body) and made(value.orelse)
            if isinstance(value, ast.Name):
                return value.id in fresh
            return self._made(value, bindings)

        return all(made(value) for value in values)

    def _new_value(self, value: ast.AST, bindings: dict[str, Any]) -> bool:
        """Whether ``value`` is a value the expression makes: a number, a new
        array, a group an operator builds (what a change in place to it can
        reach is nothing shared). An expression that gives back one of its
        operands (``a or b``, ``a if c else b``, ``(x := a)``) makes only
        what each of those makes.
        """
        if isinstance(
            value,
            (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare, ast.JoinedStr),
        ):
            return True
        if isinstance(value, ast.BoolOp):
            return all(self._new_value(each, bindings) for each in value.values)
        if isinstance(value, ast.IfExp):
            return self._new_value(value.body, bindings) and self._new_value(
                value.orelse, bindings
            )
        if isinstance(value, ast.NamedExpr):
            return self._new_value(value.value, bindings)
        if isinstance(value, ast.Subscript):
            return self._new_value(value.value, bindings)
        if not isinstance(value, ast.Call):
            return self._made(value, bindings)
        return self.value_call(value, bindings) or self._made(value, bindings)

    def value_call(self, value: ast.Call, bindings: dict[str, Any]) -> bool:
        """Whether the call ``value`` gives a number or a new array: a
        mobject's getter of one, ``abs``, ``round``, ``len``, a ufunc, a
        numpy function making an array, a function of :mod:`math`.
        """
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in _VALUE_GETTERS:
            return True
        if isinstance(func, ast.Name):
            target = bindings.get(func.id)
            if any(target is maker for maker in (abs, round, sum, len)):
                return True
        if isinstance(func, ast.Attribute):
            module = getattr(_module_named(func.value, bindings), "__name__", None)
            if module in ("math", "cmath"):
                return True
            if module == "numpy.linalg":
                return func.attr in _LINALG_VALUES
            if module == "numpy":
                found = getattr(np, func.attr, None)
                return isinstance(found, np.ufunc) or func.attr in _NUMPY_MAKERS
        return False

    def hook(self, function: Any, this: Any, receiver: Any, depth: int) -> _Effect:
        """What a hook of a class (a setter, ``__setattr__``, a descriptor's
        ``__set__``) does when it runs on ``receiver`` (``None`` when it is
        not known): what it changes on its own ``self`` is the receiver's;
        a descriptor's hook runs on the descriptor (``this``), and the
        object it is handed is not its own.
        """
        if this is not None:
            return self.effect(function, frozenset(), None, depth + 1, this=this)
        first = frozenset(function.__code__.co_varnames[:1])
        return self.effect(function, first, receiver, depth + 1)

    def callee(self, name: str, target: Any, depth: int) -> _Effect:
        """What calling ``target``, bound to ``name``, may do."""
        if target is _UNKNOWN or target is _PARAMETER:
            return _UNKNOWN_CALL
        if isinstance(target, type):
            # A new object; the scene's own class runs its own code to make
            # it, and its stores into the new object may run the class's
            # setters.
            if engine_class(target) or target.__module__ == "builtins":
                return _NO_EFFECT
            effect = _NO_EFFECT
            init = inspect.getattr_static(target, "__init__", None)
            if _user_function(init):
                made = frozenset(init.__code__.co_varnames[:1])
                effect |= self.effect(
                    init, frozenset(), None, depth + 1, made, made_type=target
                )
            for hook, this in self.class_hooks(target):
                effect |= _no_call_site(self.hook(hook, this, None, depth))
            return effect

        if isinstance(target, types.BuiltinFunctionType):
            return _native_effect(target)
        if isinstance(target, np.ufunc):
            return _NO_EFFECT
        if isinstance(target, (types.FunctionType, types.MethodType)):
            code = _code_of(target)
            if code is not None and _user_code(code):
                return self.effect(target, frozenset(), None, depth + 1)
            module = getattr(target, "__module__", "") or ""
            if module.split(".")[0] == "random" or ".random" in module:
                return _ELSEWHERE
            if isinstance(target, types.MethodType):
                # The engine's method of an object, held as a value: it acts
                # on that object unless it only reads it.
                method = target.__func__.__name__
                if method in _READERS or method.startswith(_READER_PREFIXES):
                    return _NO_EFFECT
                return _Effect(
                    restructures=method in _RESTRUCTURERS,
                    arrays=True,
                    unknown=True,
                ) | _into(target.__self__)
            return _ELSEWHERE if name in _MUTATORS else _NO_EFFECT
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
    #: What each local name in ``fresh`` is bound to.
    made_from: dict[str, list[ast.AST]] = field(default_factory=dict)
    #: The class of the object a constructor makes (its ``self``).
    made_type: type | None = None
    resolving: set[str] = field(default_factory=set)
    #: What went into each made object as it was made (its sources): an
    #: expression, an object, :data:`_ARGUMENTS`, or ``None`` for anything.
    sources: dict[str, list[Any]] = field(default_factory=dict)
    #: What each local name in ``values`` is bound to.
    value_from: dict[str, list[ast.AST]] = field(default_factory=dict)
    #: The function's first parameter (a method's ``self``), if any.
    first: str | None = None
    #: The function's parameters.
    parameters: frozenset[str] = frozenset()
    #: Whether its first parameter is bound to the object it is called on.
    bound: bool = False
    #: What each other local name may be bound to: the values its stores
    #: bind (a loop's iterable for its variable), ``None`` for one not known.
    local_from: dict[str, list[Any]] = field(default_factory=dict)
    #: The function's own code.
    code: types.CodeType | None = None

    def classes(self, receiver: ast.AST) -> set[type] | None:
        """The classes of what ``receiver`` names, ``None`` when that can't be
        told: a made object's from the expression making it.
        """
        if isinstance(receiver, ast.Name) and receiver.id in self.fresh:
            values = self.made_from.get(receiver.id)
            if not values:
                # A constructor's ``self``; else a name made some other way.
                return None if self.made_type is None else {self.made_type}
            found: set[type] = set()
            for value in values:
                cls = self._class_made(value)
                if cls is None:
                    return None
                found.add(cls)
            return found
        bound = self._bound(receiver)
        if bound is _UNKNOWN:
            return None
        if isinstance(bound, _MEMBERS_OF):
            return {type(member) for member in bound.mobject.get_family()}
        return {type(bound)}

    def _class_made(self, value: ast.AST) -> type | None:
        if isinstance(value, (ast.List, ast.ListComp)):
            return list
        if isinstance(value, (ast.Dict, ast.DictComp)):
            return dict
        if isinstance(value, (ast.Set, ast.SetComp)):
            return set
        if not isinstance(value, ast.Call):
            return None
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in ("copy", "deepcopy"):
            copied = self._bound(func.value)
            if isinstance(copied, types.ModuleType) and value.args:
                copied = self._bound(value.args[0])  # copy.copy(x)
            return None if copied is _UNKNOWN else type(copied)
        root = _root(func)
        if isinstance(root, ast.Call):
            return self._class_made(root)  # Dot().shift(UP): the Dot
        made = self._bound(func)
        return made if isinstance(made, type) else None

    def assigning(self, target: ast.Attribute, deleting: bool) -> _Effect:
        """What a store into (or a deletion of) ``target`` runs beyond the
        store: the code of its receiver's class it sets off
        (:func:`_assignment_hooks`), as a method called on the receiver.
        Through a receiver not the function's own the store already writes
        elsewhere; whether the hook changes the structure is then the
        callback's reach's to say.
        """
        receiver = target.value
        owned = self.owned(receiver)
        classes = self.classes(receiver)
        if classes is None:
            return _UNKNOWN_CALL if owned else _NO_EFFECT
        effect = _NO_EFFECT
        bound = self._bound(receiver)
        if bound is _UNKNOWN or isinstance(bound, _MEMBERS_OF):
            bound = None
        for cls in classes:
            for hook, this in self.writes.assignment_hooks(cls, target.attr, deleting):
                if owned:
                    effect |= _no_call_site(
                        self.writes.hook(hook, this, bound, self.depth)
                    )
                else:
                    effect |= _Effect(unknown=True)
        return effect

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
            if found is _PARAMETER:
                return _UNKNOWN
            if (
                found is not _UNKNOWN
                and self.code is not None
                and node.id not in self.code.co_freevars
                and found is not getattr(builtins, node.id, _UNKNOWN)
            ):
                self.writes.global_reads.add(node.id)  # a global's binding
            return found
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is _UNKNOWN or isinstance(base, _MEMBERS_OF):
                return _UNKNOWN
            # On the instance only: a class attribute is shared, and resolving
            # it would take what its class holds for what this one does.
            try:
                found = vars(base).get(node.attr, _UNKNOWN)
            except TypeError:
                return _UNKNOWN
            if isinstance(base, types.ModuleType):
                self.writes.global_reads.add(node.attr)  # a global's binding
            elif found is _UNKNOWN:
                # Absent from the object, so read through its class: a
                # class's binding, which a store of that name may change.
                self.writes.class_reads.add(node.attr)
            elif not (node.attr in _STRUCTURE_LISTS and isinstance(base, Mobject)):
                # What the object holds, read as it is now: no plan rests on
                # it (a mobject's member list is compared at every stepping
                # tick).
                self.writes.callers.note_snapshot(base, node.attr)
            return found
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
            self.structure_list(receiver)
            and name not in _READERS
            and name not in ("count", "index")
        ):
            return True  # the structure's own list, changed in place
        if name not in _RESTRUCTURERS:
            return False
        if (
            isinstance(receiver, ast.Name)
            and receiver.id in self.fresh
            and (
                name in _OWN_LISTS_ONLY or not self.fresh_writes(receiver.id).elsewhere
            )
        ):
            # Its own lists, or its family, which is all its own.
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

    def structure_list(self, node: ast.AST) -> bool:
        """Whether ``node`` names one of the lists that are a mobject's or the
        scene's structure (``group.submobjects``), itself or through a local
        name bound to it (``parts = group.submobjects``).
        """
        seen: set[str] = set()
        while (
            isinstance(node, ast.Name)
            and node.id in self.aliases
            and node.id not in seen
        ):
            seen.add(node.id)
            node = self.aliases[node.id]
        return isinstance(node, ast.Attribute) and node.attr in _STRUCTURE_LISTS

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

    def item_in_place(self, container: ast.AST) -> _Effect:
        """What an operator in place on an item of ``container`` changes in
        the item itself (``held[k] += v`` adds into an array ``held`` holds):
        nothing for an array of numbers, whose memory the store into it
        keys; any array, for anything else.
        """
        if isinstance(container, ast.Name) and container.id in self.values:
            return _NO_EFFECT  # a number, or a new array
        found = self._bound(container)
        if isinstance(found, np.ndarray) and found.dtype != object:
            return _NO_EFFECT
        return _Effect(elsewhere=True, targets=frozenset({_IN_PLACE_UNRESOLVED}))

    def store(self, target: ast.AST, deleting: bool = False) -> _Effect:
        """What a store into (a deletion of) ``target`` may change."""
        if isinstance(target, (ast.Tuple, ast.List)):
            effect = _NO_EFFECT
            for item in target.elts:
                effect |= self.store(item, deleting)
            return effect
        if isinstance(target, ast.Starred):
            return self.store(target.value, deleting)
        if isinstance(target, ast.Name):
            return _NO_EFFECT
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            return _ELSEWHERE
        receiver = target.value
        fresh = isinstance(receiver, ast.Name) and receiver.id in self.fresh
        structure = (
            isinstance(target, ast.Attribute) and target.attr in _STRUCTURE_LISTS
        ) or self.structure_list(receiver)
        effect = _Effect(restructures=structure and not fresh)
        if not self.owned(receiver):
            effect |= self.write_to(receiver)
        elif fresh and isinstance(target, ast.Attribute):
            # A setter (the engine's among them) may change what it holds.
            assert isinstance(receiver, ast.Name)
            effect |= self.fresh_writes(receiver.id)
        if isinstance(target, ast.Subscript):
            effect |= self.into(receiver)
        else:
            effect |= self.assigning(target, deleting)
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
        owned = self.owned(receiver) or (
            isinstance(receiver, ast.Attribute) and self.owned(receiver.value)
        )
        if self.structure_list(receiver):
            # ``parts += [child]``: a mobject's (or the scene's) own members.
            return _Effect(restructures=True) | (
                _NO_EFFECT if owned else self.write_to(receiver)
            )
        found = self._bound(receiver)
        if found is _UNKNOWN or isinstance(found, _MEMBERS_OF):
            if owned:
                return _Effect(restructures=True, arrays=True)
            unresolved = _Effect(
                elsewhere=True,
                restructures=True,
                arrays=True,
                targets=frozenset({_IN_PLACE_UNRESOLVED}),
            )
            return unresolved | self.write_to(receiver)
        if isinstance(found, np.ndarray):
            if not _array_root(found)[1]:
                return _NO_EFFECT
            if _storage_key(found) is None:
                return _IN_PLACE  # the engine's constant
            return _NO_EFFECT if owned else _into(found)
        if _is_mobject(found):
            return _Effect(restructures=True) | (_NO_EFFECT if owned else _into(found))
        return _NO_EFFECT if owned else _into(found)

    def write_to(
        self, receiver: ast.AST, seen: frozenset[str] = frozenset()
    ) -> _Effect:
        """A change into what ``receiver`` names, or into anything it holds:
        into what went into it, for an object the function made; into its
        argument, for a parameter (the call site's to resolve); into the
        object it resolves to, or the one its path starts from; into what
        each of its stores bound, for any other local; anywhere, when none
        of these is known. ``seen`` names locals already accounted for.
        """
        root = _root(receiver)
        if isinstance(root, ast.Name):
            if root.id in seen:
                return _NO_EFFECT
            if root.id in self.fresh:
                return self.fresh_writes(root.id, seen)
            if root.id in self.values:
                return self.expr_writes(root)
            if (
                root.id in self.parameters
                and root.id not in self.owners
                and not (root.id == self.first and self.bound)
            ):
                return self.handed_to(root.id)
        found = self.resolve(receiver)
        if found is _UNKNOWN and receiver is not root and isinstance(root, ast.Name):
            found = self.resolve(root)  # what it holds is reached through it
        if found is not _UNKNOWN:
            return _into(found)
        if isinstance(root, ast.Name) and root.id in self.local_from:
            # A local: what any of its stores bound, or what that holds.
            effect = _NO_EFFECT
            for value in self.local_from[root.id]:
                if value is None:
                    effect |= _ELSEWHERE
                elif value is _PARAMETER:
                    effect |= self.handed_to(root.id)
                else:
                    effect |= self.expr_writes(value, seen | {root.id})
            return effect
        return _ELSEWHERE

    def handed_to(self, name: str) -> _Effect:
        """What the parameter ``name`` holds as the function starts: what
        its caller hands it, or the default it falls back to; the object a
        method is called on, for its first.
        """
        if name == self.first and self.bound:
            return _into(self.bindings[name])
        handed = _Effect(elsewhere=True, writes_args=frozenset({name}))
        default = self.bindings.get(name, _PARAMETER)
        if default is _PARAMETER or default is _UNKNOWN:
            return handed
        return handed | _into(default)

    def fallback(self, name: str) -> Any:
        """What the parameter ``name`` holds when its caller hands nothing:
        its default, or the object a method is called on; ``_PARAMETER``
        when only a caller fills it (and hands a value shown free of
        callables).
        """
        found = self.bindings.get(name, _PARAMETER)
        return _PARAMETER if found is _UNKNOWN else found

    def fresh_writes(self, name: str, seen: frozenset[str] = frozenset()) -> _Effect:
        """A change into what the object ``name``, which the function made,
        holds: into what went into it as it was made.
        """
        if name in seen:
            return _NO_EFFECT
        seen = seen | {name}
        effect = _NO_EFFECT
        for source in self.sources.get(name, [None]):
            effect |= self.source_writes(source, seen)
        return effect

    def source_writes(self, source: Any, seen: frozenset[str] = frozenset()) -> _Effect:
        if source is None:
            return _ELSEWHERE
        if source is _ARGUMENTS:
            return _Effect(elsewhere=True, writes_args=frozenset({_EVERY_ARGUMENT}))
        if isinstance(source, _Effect):
            return source
        if isinstance(source, _CodeRef):
            return _code_writes(source.value)
        if isinstance(source, _Ref):
            return _into(source.value)
        return self.expr_writes(source, seen)

    def expr_writes(self, expr: ast.AST, seen: frozenset[str] = frozenset()) -> _Effect:
        """A change into whatever ``expr`` evaluates to, or into anything it
        holds. ``seen`` names locals already accounted for (a made object
        being followed, a comprehension's variables, whose iterables are).
        """
        if isinstance(expr, (ast.Constant, ast.JoinedStr, ast.Compare)):
            return _NO_EFFECT
        if isinstance(expr, (ast.Starred, ast.NamedExpr)):
            return self.expr_writes(expr.value, seen)
        parts: list[ast.AST] = []
        if isinstance(expr, ast.BinOp):
            # A mobject's ``+`` groups both; arithmetic on NumPy's own arrays
            # of numbers makes a new one. Any other operand's operator, or
            # its reflected form, may hand back either operand.
            operands: list[ast.AST] = [expr.left, expr.right]
            if all(self._number_array(part) for part in operands):
                return _NO_EFFECT
            parts = operands
        elif isinstance(expr, ast.UnaryOp):
            parts = [expr.operand]
        elif isinstance(expr, ast.BoolOp):
            parts = list(expr.values)
        elif isinstance(expr, ast.IfExp):
            parts = [expr.body, expr.orelse]
        elif isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            parts = list(expr.elts)
        elif isinstance(expr, ast.Dict):
            parts = [key for key in expr.keys if key is not None]
            parts += expr.values
        elif isinstance(
            expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            bound = {
                name for comp in expr.generators for name in _names_bound(comp.target)
            }
            seen = seen | bound
            parts = [comp.iter for comp in expr.generators]
            parts += (
                [expr.key, expr.value] if isinstance(expr, ast.DictComp) else [expr.elt]
            )
        if isinstance(expr, ast.Lambda):
            parts = self._held_by_lambda(expr)
        if parts:
            effect = _NO_EFFECT
            for part in parts:
                effect |= self.expr_writes(part, seen)
            return effect
        if isinstance(expr, ast.Lambda):
            return _NO_EFFECT
        if isinstance(expr, ast.Call):
            sources = self.call_sources(expr)
            if sources is not None:
                effect = _NO_EFFECT
                for source in sources:
                    effect |= self.source_writes(source, seen)
                return effect
            if self.writes.value_call(expr, self.bindings):
                return _NO_EFFECT  # a number, or a new array
            return self.returned_by(expr)
        if isinstance(expr, ast.Name):
            if expr.id in seen:
                return _NO_EFFECT
            if expr.id in self.fresh:
                return self.fresh_writes(expr.id, seen)
            if expr.id in self.values:
                effect = _NO_EFFECT
                for value in self.value_from.get(expr.id, [None]):
                    effect |= (
                        _ELSEWHERE
                        if value is None
                        else self.expr_writes(value, seen | {expr.id})
                    )
                return effect
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            return self.write_to(expr, seen)
        return _ELSEWHERE

    def returned_by(self, call: ast.Call) -> _Effect:
        """A change into what ``call`` hands back, or anything that holds,
        when it may hand back an object it did not make: what a function or
        a method of the scene's returns; an attribute ``getattr`` reads by
        a constant name, or its default; anything, for any other call.
        """
        func = call.func
        target: Any = _UNKNOWN
        this: Any = None
        if isinstance(func, ast.Name):
            target = self.bindings.get(func.id, _UNKNOWN)
            if _by_constant_name(target, call) and target is getattr:
                obj, name, *default = call.args
                assert isinstance(name, ast.Constant)
                assert isinstance(name.value, str)
                read = ast.Attribute(value=obj, attr=name.value, ctx=ast.Load())
                effect = self.write_to(read)
                for each in default:
                    effect |= self.expr_writes(each)
                return effect
        elif isinstance(func, ast.Attribute):
            holder = self.resolve(func.value)
            own = getattr(holder, "__dict__", None)
            if isinstance(own, dict) and func.attr in own:
                target = own[func.attr]  # a function it holds (a module's)
            elif holder is not _UNKNOWN and not isinstance(
                holder, (types.ModuleType, _MEMBERS_OF)
            ):
                target = inspect.getattr_static(type(holder), func.attr, None)
                this = holder
                if isinstance(target, staticmethod):
                    this = None
                elif isinstance(target, classmethod):
                    this = type(holder)
                target = getattr(target, "__func__", target)
        if not _user_function(target):
            return _ELSEWHERE
        return self._handed(
            self.writes.returns(target, this), call, target, this is not None
        )

    def made_sources(self, value: ast.AST) -> list[Any]:
        """What went into the object ``value`` makes (:meth:`_Writes._made`)."""
        if isinstance(value, ast.Call):
            found = self.call_sources(value)
            return [None] if found is None else found
        return [value]  # a container, or a comprehension: what its items hold

    def call_sources(self, call: ast.Call) -> list[Any] | None:
        """What went into the new object ``call`` hands back, or ``None``
        when it may hand back an object it did not make: a class's object
        holds what it was handed (and what the scene's own class's code
        reads); what a function of the scene's returns, what its returns
        hold (:meth:`_Writes.returns`); a copy what the original holds; a
        method of a made object gives back that object or something it
        holds.
        """
        func = call.func
        arguments: list[Any] = [*call.args, *(k.value for k in call.keywords)]
        if isinstance(func, ast.Attribute) and func.attr in ("copy", "deepcopy"):
            if isinstance(self.resolve(func.value), types.ModuleType):
                return arguments  # copy.copy(x), copy.deepcopy(x)
            return [func.value]  # a shallow copy shares what the original holds
        if isinstance(func, ast.Attribute) and self._generator(func.value):
            return arguments  # a draw: numbers, or items of what it draws from
        root = _root(func)
        if isinstance(root, ast.Call):
            inner = self.call_sources(root)
            return None if inner is None else inner + arguments
        target = self._called(func)
        if any(target is each for each in _HANDING_BACK):
            return arguments  # one of them, an item of one, or a list of those
        if target is _UNKNOWN or not self.writes.makes(target):
            return None
        if isinstance(target, type):
            if engine_class(target) or target.__module__ == "builtins":
                return arguments
            return [*arguments, _CodeRef(target)]
        if _user_function(target):
            return [self._handed(self.writes.returns(target), call, target)]
        return arguments  # a fresh generator from a seed

    def _held_by_lambda(self, expr: ast.Lambda) -> list[ast.AST]:
        """What the function ``expr`` makes holds: its defaults, and the
        variables of this function and of those around it that its body
        uses (a module's globals it looks up as it runs, holding none).
        """
        arguments = expr.args
        own = {
            each.arg
            for each in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *filter(None, (arguments.vararg, arguments.kwarg)),
            )
        }
        own |= {
            item.id
            for item in ast.walk(expr.body)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        }
        outer = set(self.local_from) | self.fresh | self.values | self.owners
        if self.code is not None:
            outer |= set(self.code.co_freevars)
        held: list[ast.AST] = [*arguments.defaults]
        held += [each for each in arguments.kw_defaults if each is not None]
        for item in ast.walk(expr.body):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id not in own
                and item.id in outer
            ):
                held.append(item)
        return held

    def _number_array(self, node: ast.AST) -> bool:
        """Whether ``node`` is a number or names an array of numbers of
        NumPy's own class (a subclass's operators may hand back an operand),
        resolved.
        """
        if isinstance(node, ast.Constant):
            return type(node.value) in (int, float, complex)
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return False
        if isinstance(node, ast.Name) and (
            node.id in self.fresh
            or node.id in self.values
            or node.id in self.local_from
        ):
            return False  # bound as the function runs
        found = self.resolve(node)
        return type(found) is np.ndarray and found.dtype != object

    def _generator(self, receiver: ast.AST) -> bool:
        """Whether ``receiver`` names a random generator the function made."""
        if not (isinstance(receiver, ast.Name) and receiver.id in self.fresh):
            return False
        values = self.made_from.get(receiver.id) or []
        return bool(values) and all(
            isinstance(value, ast.Call)
            and any(self._called(value.func) is maker for maker in _FRESH_RANDOM)
            for value in values
        )

    def call(self, call: ast.Call) -> _Effect:
        """What ``call`` may change: what its callee does, and the arrays it
        hands for a result to be written into.
        """
        effect = self._callee_effect(call) | self._handed_callables(call)
        func = call.func
        method = None
        if isinstance(func, ast.Attribute) and self._called(func.value) is _UNKNOWN:
            method = func.attr
        written = [k.value for k in call.keywords if k.arg == "out"]
        written += _numpy_writes(self._called(func), call, method)
        for argument in written:
            items = (
                argument.elts
                if isinstance(argument, (ast.Tuple, ast.List))
                else [argument]
            )
            for item in items:
                effect |= self.into(
                    item.value if isinstance(item, ast.Starred) else item
                )
        return effect

    def _handed_callables(self, call: ast.Call) -> _Effect:
        """What the callables ``call`` hands may do: whatever it calls may
        call them in turn (a key, a pattern's replacement, a mapped
        function), with arguments of its own, so each is taken as called
        with arguments no call site binds. A value holding callables (a
        list of them, a dict unpacked as keywords) hands those where its
        callee may call what it is handed (:mod:`.handed`). One that may be
        or hold a callable the check can't resolve (a generator, an
        iterator, what an unknown call gives) makes the call an unknown one,
        whatever its callee: a callee shown never to call what it is handed
        may still run its code (taking a generator's items).
        """
        effect = _NO_EFFECT
        for argument in [*call.args, *(k.value for k in call.keywords)]:
            if isinstance(argument, ast.Starred):
                argument = argument.value
            target = self._callable_value(argument)
            if target is not None:
                effect |= self._called_effect(target, argument)
                continue
            held = self._held(argument, set())
            if held is None:
                effect |= _UNKNOWN_CALL  # a value not shown free of callables
                continue
            if held == () or not self.unresolved_call_reason(call, argument):
                continue
            for each in held:
                effect |= self._called_effect(each, None)
        # Where a callee keeps what it is handed needs no note: whatever
        # later reads it reads what an object holds, on which no plan rests.
        return effect

    def _called_effect(self, target: Any, argument: ast.AST | None) -> _Effect:
        """What calling ``target``, a callable handed (as ``argument``, or
        in what it holds), may do with arguments no call site binds.
        """
        if isinstance(target, types.FunctionType) and not _user_function(target):
            # A library's or the engine's function, called by what it
            # hands it (``Mobject.shift`` mapped over mobjects): judged
            # by what its code may reach, not by its name.
            return self.writes.effect(target, frozenset(), None, self.depth + 1)
        name = str(getattr(target, "__name__", ""))
        if isinstance(argument, ast.Attribute):
            name = argument.attr
        elif isinstance(argument, ast.Name):
            name = argument.id
        return _no_call_site(self.writes.callee(name, target, self.depth))

    def _held(self, node: ast.AST, resolving: set[str]) -> tuple[Any, ...] | None:
        """The callables ``node`` may evaluate to or hold, each one the
        check resolves; ``None`` when it may be or hold one the check can't
        resolve. Only a value's kind shows it free of them (a constant, a
        string, a number or an array of numbers, what the check resolves
        and reads through); what an operator, a display or a comprehension
        makes holds what its parts do, a ``[f] * n`` among them. Each form
        follows its rule in :data:`.coverage.OPERATIONS`.
        """
        rule = rule_of(node)
        if rule in (Rule.CONSTANT, Rule.LAMBDA):
            return ()  # a lambda's body is read where it is
        if rule in (Rule.NAME, Rule.ATTRIBUTE):
            target = self._callable_value(node)
            if target is not None:
                return (target,)
        if rule is Rule.NAME:
            assert isinstance(node, ast.Name)
            return self._name_held(node.id, resolving)
        if rule is Rule.ATTRIBUTE:
            found = self.resolve(node)
            if found is _UNKNOWN:
                return None
            return self._object_held(found)
        if rule is Rule.SUBSCRIPT:
            assert isinstance(node, ast.Subscript)
            return self._item_held(node, resolving)
        if rule is Rule.CALL:
            assert isinstance(node, ast.Call)
            return self._call_held(node, resolving)
        if rule is Rule.COMPARE:
            # A truth value, unless an operand's own comparison makes
            # something else.
            assert isinstance(node, ast.Compare)
            compared = [node.left, *node.comparators]
            if any(self._held(part, resolving) is None for part in compared):
                return None
            return ()
        if rule in (Rule.PARTS, Rule.GENERATOR):
            return self._all_held(_held_parts(node), resolving)
        return None  # SENT, FALLBACK, and no other rule is a value's

    def _all_held(
        self, parts: Iterable[ast.AST], resolving: set[str]
    ) -> tuple[Any, ...] | None:
        found: dict[int, Any] = {}
        for part in parts:
            held = self._held(part, resolving)
            if held is None:
                return None
            found.update((id(each), each) for each in held)
        return tuple(found.values())

    def _name_held(self, name: str, resolving: set[str]) -> tuple[Any, ...] | None:
        """What a name holds: the owner, what it holds; a parameter, its
        call site's argument (judged there; an updater's own others are
        numbers); a local name, what each store binds it to; any other, what
        it is bound to.
        """
        if name in self.owners:
            owner = self.resolve(ast.Name(id=name, ctx=ast.Load()))
            return None if owner is _UNKNOWN else self._object_held(owner)
        if name in self.parameters:
            return ()
        values = self.local_from.get(name)
        bound = self._outer(name)
        if values is None:
            return None if bound is _UNKNOWN else self._object_held(bound)
        if name in resolving:
            return ()  # what its other stores bind it to says the rest
        resolving.add(name)
        try:
            found: dict[int, Any] = {}
            for value in values:
                if value is _PARAMETER:
                    continue
                held = None if value is None else self._held(value, resolving)
                if held is None:
                    return None
                found.update((id(each), each) for each in held)
        finally:
            resolving.discard(name)
        if bound is not _UNKNOWN and bound is not getattr(builtins, name, _UNKNOWN):
            # The same name in another scope of the code (a function it
            # defines binds it, this one reads it): what it is bound to.
            held = self._object_held(bound)
            if held is None:
                return None
            found.update((id(each), each) for each in held)
        return tuple(found.values())

    def _outer(self, name: str) -> Any:
        """What ``name`` is bound to outside the function's own stores (a
        global, a closure's cell); :data:`_UNKNOWN` when nothing is.
        """
        bound = self.bindings.get(name, _UNKNOWN)
        return _UNKNOWN if bound is _PARAMETER else bound

    def _object_held(self, found: Any) -> tuple[Any, ...] | None:
        if isinstance(found, _MEMBERS_OF):
            found = found.mobject
        return self.writes.callers.held_by(found)

    def _item_held(
        self, node: ast.Subscript, resolving: set[str]
    ) -> tuple[Any, ...] | None:
        """What an item of a container holds: what the container holds,
        when it is a builtin one or a display (an object's own
        ``__getitem__`` may give anything).
        """
        found = self.resolve(node)
        if isinstance(found, _MEMBERS_OF):
            return self._object_held(found)
        base = node.value
        if self._inert(base, set()):
            return ()
        if isinstance(base, _DISPLAYS):
            return self._held(base, resolving)
        container = self.resolve(base)
        if item_kind(container):
            return self._object_held(container)
        return None

    def _call_held(self, node: ast.Call, resolving: set[str]) -> tuple[Any, ...] | None:
        """What a call's result holds: nothing when it is inert
        (:meth:`_inert_call`); what its arguments hold for a function making
        a number or an array of what it is handed
        (:meth:`_Writes.value_call`); anything for any other.
        """
        if self._inert_call(node, set()):
            return ()
        named = _named_attribute(node)
        if (
            named is not None
            and isinstance(node.func, ast.Name)
            and node.func.id not in self.local_from
            and self.resolve(node.func) is getattr
        ):
            return self._all_held([named, *node.args[2:]], resolving)
        if not self.writes.value_call(node, self.bindings) or (
            isinstance(node.func, ast.Attribute) and node.func.attr in _VALUE_GETTERS
        ):
            return None  # a getter not shown the engine's may give anything
        return self._all_held(
            [*node.args, *(keyword.value for keyword in node.keywords)], resolving
        )

    def _inert(self, node: ast.AST, resolving: set[str]) -> bool:
        """Whether ``node`` gives an inert value: a number, a string, an
        array of them, whose items, operators and methods give such values
        again, never a callable (the rules of :data:`.coverage.OPERATIONS`).
        """
        rule = rule_of(node)
        if rule is Rule.CONSTANT:
            return isinstance(node, ast.JoinedStr) or (
                isinstance(node, ast.Constant) and type(node.value) in INERT_KINDS
            )
        if rule is Rule.NAME:
            assert isinstance(node, ast.Name)
            values = self.local_from.get(node.id)
            if node.id in self.owners:
                return False
            if node.id in self.parameters:
                return False  # what a call site hands it may be anything
            if values is None:
                return inert_value(self.resolve(node))
            bound = self._outer(node.id)
            if not (
                bound is _UNKNOWN
                or bound is getattr(builtins, node.id, _UNKNOWN)
                or inert_value(bound)
            ):
                return False  # the same name in another scope (:meth:`_name_held`)
            if node.id in resolving:
                return True  # what its other stores bind it to says the rest
            resolving.add(node.id)
            try:
                return all(
                    value is not _PARAMETER
                    and value is not None
                    and self._inert(value, resolving)
                    for value in values
                )
            finally:
                resolving.discard(node.id)
        if rule is Rule.ATTRIBUTE:
            return inert_value(self.resolve(node))
        if rule is Rule.COMPARE:
            assert isinstance(node, ast.Compare)
            return all(
                self._inert(part, resolving) for part in (node.left, *node.comparators)
            )
        if rule is Rule.PARTS:
            # An operator on inert values gives one; a display of inert
            # items gives them back, and its methods give them or numbers.
            return all(self._inert(part, resolving) for part in _held_parts(node))
        if rule is Rule.SUBSCRIPT:
            assert isinstance(node, ast.Subscript)
            return self._inert(node.value, resolving)
        if rule is Rule.CALL:
            assert isinstance(node, ast.Call)
            return self._inert_call(node, resolving)
        return False  # LAMBDA, GENERATOR, SENT, FALLBACK

    def _inert_call(self, node: ast.Call, resolving: set[str]) -> bool:
        """Whether a call gives an inert value (:meth:`_inert`): ``len``'s
        count, a function of :mod:`math`, a mobject's getter of a number or
        an array (the engine's own), a method of an inert value, a function
        making a number or an array (:meth:`_Writes.value_call`) of inert
        values. A function of the scene's is never taken for one: what it
        returns would rest on a proof over its code, not on a kind.
        """
        func = node.func
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        if isinstance(func, ast.Name) and all(
            value is _PARAMETER for value in self.local_from.get(func.id, ())
        ):
            # Not rebound by the function: a global, a closure's, or a
            # parameter its default fills (as the engine calls a callback).
            target = self.resolve(func)
            if any(target is each for each in INERT_RESULT_BUILTINS):
                return True
            named = _named_attribute(node)
            if target is getattr and named is not None:
                return self._inert(named, resolving) and all(
                    self._inert(default, resolving) for default in node.args[2:]
                )
            if isinstance(target, np.ufunc):  # held under a name of its own
                return all(self._inert(argument, resolving) for argument in arguments)
            if (
                isinstance(target, types.MethodType)
                and target.__func__.__name__ in _VALUE_GETTERS
                and engine_code(
                    inspect.getattr_static(
                        type(target.__self__), target.__func__.__name__, None
                    )
                )
            ):
                return True  # an engine getter held bound to its mobject
        if isinstance(func, ast.Attribute):
            if func.attr in _VALUE_GETTERS:
                classes = self.classes(func.value)
                return classes is not None and all(
                    engine_code(inspect.getattr_static(cls, func.attr, None))
                    for cls in classes
                )
            module = getattr(_module_named(func.value, self.bindings), "__name__", None)
            if module in ("math", "cmath"):
                return True
            if not func.attr.startswith("_") and self._inert(func.value, resolving):
                return True  # a number's, a string's, an array's own method
        return self.writes.value_call(node, self.bindings) and all(
            self._inert(argument, resolving) for argument in arguments
        )

    def read_table(self, node: ast.AST) -> _Effect:
        """Read every node of the function's code by its rule in
        :data:`.coverage.OPERATIONS`. A form with no rule, a name or an
        attribute reaching a namespace, a global or nonlocal name bound to a
        value not shown inert, or a store that may rebind a global or a
        class's attribute makes the play tick everything (:class:`Unbounded`);
        an item read, stored or deleted on a base that may run code of its
        own, iterating a value and entering a ``with`` are that code's
        effect.
        """
        effect = _NO_EFFECT
        callees = frozenset(
            id(item.func) for item in _walk(node) if isinstance(item, ast.Call)
        )
        for item in _walk(node):
            rule = rule_of(item)
            if rule is Rule.FALLBACK:
                raise Unbounded(
                    f"its code holds a {type(item).__name__}, a form the write"
                    " check does not read",
                    self.code,
                )
            self._namespace(item)
            effect |= self._protocol(item, callees)
            if rule is Rule.OUTER_REBIND:
                assert isinstance(item, (ast.Global, ast.Nonlocal))
                for name in item.names:
                    if not self._binds_inert(node, name):
                        raise Unbounded(
                            f"binds {name} of another scope to a value not shown inert",
                            self.code,
                        )
            elif rule is Rule.STORE:
                for target, value in _stored(item):
                    self._rebinds(item, target, value)
            if rule is Rule.SUBSCRIPT:
                assert isinstance(item, ast.Subscript)
                effect |= self._item_code(item)
            effect |= self._iterating(item)
        return effect

    def _namespace(self, item: ast.AST) -> None:
        """Raise :class:`Unbounded` for a node reaching a namespace: an
        attribute of :data:`.coverage.NAMESPACE_ATTRIBUTES`, a builtin of
        :data:`.coverage.NAMESPACE_BUILTINS`.
        """
        if isinstance(item, ast.Attribute) and item.attr in NAMESPACE_ATTRIBUTES:
            raise Unbounded(f"reaches {item.attr}, a namespace", self.code)
        if (
            isinstance(item, ast.Name)
            and item.id in NAMESPACE_BUILTINS
            and self.bindings.get(item.id) is getattr(builtins, item.id)
        ):
            raise Unbounded(f"names {item.id}, which reaches a namespace", self.code)

    def _protocol(self, item: ast.AST, callees: frozenset[int]) -> _Effect:
        """What ``item`` may run of its operands' own classes (PROTOCOL in
        :class:`.coverage.Rule`): an operator's, a comparison's, a truth
        test's, a formatting's, an item's key's and an attribute read's
        code. Each is an unknown call when an operand may be an object
        whose class defines that code in the scene's (:meth:`_runs_protocol`).
        ``callees`` are the ids of the attributes that are a call's callee,
        judged as calls.
        """
        operands: list[ast.AST] = []
        truth: list[ast.AST] = []
        if isinstance(item, ast.BinOp):
            operands = [item.left, item.right]
        elif isinstance(item, ast.UnaryOp):
            if isinstance(item.op, ast.Not):
                truth = [item.operand]
            else:
                operands = [item.operand]
        elif isinstance(item, ast.AugAssign):
            operands = [item.target, item.value]
        elif isinstance(item, ast.Compare):
            operands = [item.left, *item.comparators]
        elif isinstance(item, ast.BoolOp):
            truth = list(item.values)
        elif isinstance(item, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            truth = [item.test]
        elif isinstance(item, ast.comprehension):
            truth = list(item.ifs)
        elif isinstance(item, ast.FormattedValue):
            operands = [item.value]
        elif isinstance(item, ast.Subscript):
            operands = [item.slice]
        elif isinstance(item, ast.Slice):
            operands = [part for part in (item.lower, item.upper, item.step) if part]
        elif (
            isinstance(item, ast.Attribute)
            and isinstance(item.ctx, ast.Load)
            and id(item) not in callees
        ):
            return self._attribute_read(item)
        if any(self._runs_protocol(each) for each in operands) or any(
            self._runs_protocol(each, truth=True) for each in truth
        ):
            return _UNKNOWN_CALL
        return _NO_EFFECT

    def _attribute_read(self, item: ast.Attribute) -> _Effect:
        """What reading ``item`` may run: its object's class's
        ``__getattribute__`` or ``__getattr__``, a property's getter or
        another descriptor's ``__get__``, when the scene's code defines it;
        an unknown call when the object is not resolved and not inert.
        """
        if (
            isinstance(item.value, ast.Name)
            and item.value.id in self.parameters
            and item.value.id not in self.owners
            and self.fallback(item.value.id) is _PARAMETER
        ):
            return _NO_EFFECT  # its call site hands a value shown free
        base = self.resolve(item.value)
        if isinstance(base, _MEMBERS_OF):
            base = base.mobject
        if base is _UNKNOWN:
            return _NO_EFFECT if self._inert(item.value, set()) else _UNKNOWN_CALL
        if isinstance(base, types.ModuleType):
            return _NO_EFFECT
        self.writes.class_reads.add(item.attr)  # its class's binding (STORE)
        return _UNKNOWN_CALL if _reading_runs_code(base, item.attr) else _NO_EFFECT

    def _runs_protocol(
        self, node: ast.AST, seen: frozenset[str] = frozenset(), truth: bool = False
    ) -> bool:
        """Whether the value of ``node`` may be an object whose class
        defines in the scene's code what an operator, a formatting or (with
        ``truth``) a truth test runs (:func:`_protocol_code`): not for an
        inert value, a function made in place, the result of an operation
        (its operands are judged where it is), a builtin's, NumPy's or the
        engine's result of a kind not the scene's, a parameter (its call
        site hands a value shown free of callables).
        """
        if self._inert(node, set()) or isinstance(
            node,
            (
                ast.Lambda,
                ast.Constant,
                ast.JoinedStr,
                ast.BinOp,
                ast.UnaryOp,
                ast.Compare,
            ),
        ):
            return False
        if isinstance(node, (*_DISPLAYS, ast.Set, ast.SetComp, ast.GeneratorExp)):
            return not truth  # a container's comparison or form reads its items
        if isinstance(node, ast.BoolOp):
            return any(self._runs_protocol(part, seen, truth) for part in node.values)
        if isinstance(node, ast.IfExp):
            return any(
                self._runs_protocol(part, seen, truth)
                for part in (node.body, node.orelse)
            )
        if isinstance(node, ast.NamedExpr):
            return self._runs_protocol(node.value, seen, truth)
        if isinstance(node, ast.Call):
            return not self._gives_quiet_value(node, truth)
        if isinstance(node, ast.Name) and node.id not in self.owners:
            if node.id in self.parameters:
                return self._handed_runs_protocol(node.id, truth)
            values = self.local_from.get(node.id)
            if values is not None:
                if node.id in seen:
                    return False  # its other stores say the rest
                return any(
                    value is None
                    or (
                        self._handed_runs_protocol(node.id, truth)
                        if value is _PARAMETER
                        else self._runs_protocol(value, seen | {node.id}, truth)
                    )
                    for value in values
                )
        found = self.resolve(node)
        if isinstance(found, _MEMBERS_OF):
            found = found.mobject
        return found is _UNKNOWN or _protocol_code(found, truth)

    def _handed_runs_protocol(self, name: str, truth: bool) -> bool:
        """Whether an operator, a formatting or (with ``truth``) a truth test
        on the parameter ``name`` may run the scene's code: a caller hands a
        value shown free of callables, but the value it falls back to
        (:meth:`fallback`) is judged as it is.
        """
        found = self.fallback(name)
        return found is not _PARAMETER and _protocol_code(found, truth)

    def _gives_quiet_value(self, call: ast.Call, truth: bool) -> bool:
        """Whether ``call`` gives a value whose class's own code is not the
        scene's, nor its items' where the protocol reads them: an inert
        value, an object the engine's code gives or a class with no such
        code makes; a container or an iterator a builtin makes, for a truth
        test only (a comparison or a formatting reads its items).
        """
        if self._inert(call, set()):
            return True
        func = call.func
        if isinstance(func, ast.Name):
            target = self.bindings.get(func.id, _UNKNOWN)
            if any(target is each for each in _ITERATOR_BUILTINS):
                return truth
            if isinstance(target, type) and engine_class(target):
                return True
        callees = self._callees(call)
        return callees is not None and all(
            engine_code(getattr(target, "__func__", target)) for target, _, _ in callees
        )

    def _binds_inert(self, node: ast.AST, name: str) -> bool:
        """Whether every value the function binds ``name`` to is inert, its
        value outside the function among them.
        """
        outer = self._outer(name)
        if outer is not _UNKNOWN and not inert_value(outer):
            return False
        for value in _bound_to(node, name):
            if value is None or not self._inert(value, set()):
                return False
        return True

    def _rebinds(self, item: ast.AST, target: ast.AST, value: ast.AST | None) -> None:
        """Raise :class:`Unbounded` when a store into an attribute rebinds
        what the plan reads as a binding: a module's global, a class's
        attribute (a method among them). Note any other store into an
        attribute by its name (:meth:`_Writes.plan_reads_hold`): on an
        object not resolved it may be either, and on any object it may
        shadow what the plan read through its class. A value inert by kind
        brings no code, and an object the function made was read by no
        plan.
        """
        for part in _store_targets(target):
            if not isinstance(part, ast.Attribute):
                continue
            if (
                value is not None
                and not isinstance(item, ast.AugAssign)
                and self._inert(value, set())
            ):
                continue
            if isinstance(part.value, ast.Name) and part.value.id in self.fresh:
                continue
            # Resolved through a binding only: a store through what an object
            # holds is noted by name below, whatever the object turns out to be.
            receiver = (
                self.resolve(part.value) if isinstance(part.value, ast.Name) else None
            )
            if isinstance(receiver, (types.ModuleType, type)):
                raise Unbounded(
                    f"stores into {part.attr} of the {type(receiver).__name__}"
                    f" {getattr(receiver, '__name__', '')}, a binding the plan reads",
                    self.code,
                )
            self.writes.loose_stores.setdefault(
                part.attr, f"{_describe_code(self.code)} stores into {part.attr}"
            )

    def _iterating(self, item: ast.AST) -> _Effect:
        """What the iterating ``item`` does may run (ITERATION in
        :class:`.coverage.Rule`): a loop's or a comprehension clause's
        iterable, and each item an unpacking target takes apart; the right
        operand of ``in``; a starred value; a value unpacked into names; a
        ``yield from``'s. Awaiting and entering a ``with`` run code of the
        value's kind: unknown calls.
        """
        if isinstance(item, (ast.For, ast.AsyncFor, ast.comprehension)):
            effect = self._iterated(item.iter)
            if isinstance(item.target, (ast.Tuple, ast.List)):
                effect |= self._items_iterated(item.iter)
            return effect
        if isinstance(item, ast.Compare):
            effect = _NO_EFFECT
            for op, right in zip(item.ops, item.comparators, strict=True):
                if isinstance(op, (ast.In, ast.NotIn)):  # codespell:ignore
                    effect |= self._iterated(right)
            return effect
        if isinstance(item, ast.Starred) and isinstance(item.ctx, ast.Load):
            return self._iterated(item.value)
        if isinstance(item, ast.Dict):
            effect = _NO_EFFECT
            for key, value in zip(item.keys, item.values, strict=True):
                if key is None:  # ``**mapping``
                    effect |= self._iterated(value)
            return effect
        if isinstance(item, ast.Assign) and any(
            isinstance(target, (ast.Tuple, ast.List)) for target in item.targets
        ):
            return self._iterated(item.value)
        if isinstance(item, ast.YieldFrom):
            return self._iterated(item.value)
        if isinstance(item, (ast.Await, ast.With, ast.AsyncWith)):
            return _UNKNOWN_CALL
        return _NO_EFFECT

    def _iterated(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> _Effect:
        """What iterating the value of ``node`` may run: nothing for an
        inert value, a display, a builtin making an iterator (its arguments
        are its call's to judge), a call of the engine's code, a parameter
        (its call site hands a value shown free of callables), or a value
        whose kind iterates without code of the scene's
        (:func:`_iterates_quietly`); an unknown call for anything else.
        """
        if self._inert(node, set()) or isinstance(
            node, (*_DISPLAYS, ast.Set, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            return _NO_EFFECT
        if isinstance(node, ast.Call):
            return _NO_EFFECT if self._makes_quiet_iterable(node) else _UNKNOWN_CALL
        if isinstance(node, ast.Name) and node.id not in self.owners:
            if node.id in self.parameters:
                return self._handed_iterated(node.id)
            values = self.local_from.get(node.id)
            if values is not None:
                if node.id in seen:
                    return _NO_EFFECT  # its other stores say the rest
                effect = _NO_EFFECT
                for value in values:
                    if value is None:
                        return _UNKNOWN_CALL
                    if value is _PARAMETER:
                        effect |= self._handed_iterated(node.id)
                    else:
                        effect |= self._iterated(value, seen | {node.id})
                return effect
        found = self.resolve(node)
        if isinstance(found, _MEMBERS_OF) or (
            found is not _UNKNOWN and _iterates_quietly(found)
        ):
            return _NO_EFFECT
        return _UNKNOWN_CALL

    def _handed_iterated(self, name: str) -> _Effect:
        """What iterating the parameter ``name`` may run: nothing for what a
        caller hands (a value shown free of callables); for the value it
        falls back to (:meth:`fallback`), nothing only when that iterates
        without the scene's code.
        """
        found = self.fallback(name)
        if found is _PARAMETER or inert_value(found) or _iterates_quietly(found):
            return _NO_EFFECT
        return _UNKNOWN_CALL

    def _items_iterated(self, node: ast.AST) -> _Effect:
        """What taking apart each item of ``node``'s value may run: nothing
        for the tuples a builtin pairing values gives (``zip``,
        ``enumerate``, a dict's ``items``) or the items of a display, each
        judged as iterated; an unknown call for any other.
        """
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and any(
                self.bindings.get(func.id) is each for each in (zip, enumerate)
            ):
                return _NO_EFFECT
            if isinstance(func, ast.Attribute) and func.attr == "items":
                return _NO_EFFECT if self._makes_quiet_iterable(node) else _UNKNOWN_CALL
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            effect = _NO_EFFECT
            for each in node.elts:
                effect |= self._iterated(each)
            return effect
        return _UNKNOWN_CALL

    def _makes_quiet_iterable(self, call: ast.Call) -> bool:
        """Whether ``call`` gives an iterable whose iteration runs only what
        the call itself is judged for: a builtin making a container or an
        iterator (what it consumes is its argument, judged where it is
        handed), a dict's or a list's own method, NumPy's or the engine's
        code.
        """
        func = call.func
        if isinstance(func, ast.Name):
            target = self.bindings.get(func.id, _UNKNOWN)
            if any(target is each for each in _ITERATOR_BUILTINS):
                return True
        if isinstance(func, ast.Attribute):
            receiver = self.resolve(func.value)
            if func.attr in ("items", "keys", "values", "copy") and (
                type(receiver) in NATIVE_HOLDERS
                or (
                    receiver is _UNKNOWN
                    and (self.classes(func.value) or {object}) <= {dict, list}
                )
            ):
                return True
        if self.writes.value_call(call, self.bindings):
            return True
        callees = self._callees(call)
        return callees is not None and all(
            engine_code(getattr(target, "__func__", target)) for target, _, _ in callees
        )

    def _item_code(self, item: ast.Subscript) -> _Effect:
        """What reading, storing or deleting an item of ``item.value`` may
        run (:attr:`.coverage.Rule.SUBSCRIPT`): nothing on an item kind, an
        inert value, a display, a local made as a list, tuple or dict, or a
        mobject's members; the callables a resolved base holds, called (a
        ``defaultdict``'s factory); an unknown call on any other base. A
        parameter's base is its call site's to judge (:mod:`.handed` counts
        an item of a parameter as calling it).
        """
        base = item.value
        if self._inert(base, set()) or isinstance(base, _DISPLAYS):
            return _NO_EFFECT
        if isinstance(base, ast.Name) and base.id in self.parameters:
            return _NO_EFFECT
        if isinstance(self.resolve(item), _MEMBERS_OF):
            return _NO_EFFECT  # the engine's __getitem__, giving members
        found = self.resolve(base)
        if found is _UNKNOWN:
            classes = self.classes(base) if isinstance(base, ast.Name) else None
            if classes and all(cls in (list, tuple, dict) for cls in classes):
                return _NO_EFFECT  # a local made as one
            return _UNKNOWN_CALL
        if item_kind(found):
            return _NO_EFFECT
        held = self._object_held(found)
        if held is None:
            return _UNKNOWN_CALL
        effect = _NO_EFFECT
        for each in held:
            effect |= self._called_effect(each, None)
        return effect

    def unresolved_call_reason(self, call: ast.Call, argument: ast.expr) -> str | None:
        """Why ``call`` may call ``argument``, which may be a callable the
        check can't resolve; ``None`` when its callee never does
        (:meth:`.handed.Callers.calls_argument`).
        """
        from .handed import UNKNOWN_CALLEE

        callees = self._callees(call)
        if callees is None:
            return UNKNOWN_CALLEE
        for target, bound, cls in callees:
            reason = self.writes.callers.calls_argument(
                target, call, argument, bound, cls, self.depth + 1
            )
            if reason is not None:
                return reason
        return None

    def _callees(self, call: ast.Call) -> list[tuple[Any, bool, type | None]] | None:
        """What ``call``'s callee may be: each with whether its first
        parameter is bound to the object it is called on, and that object's
        class; ``None`` when that can't be told.
        """
        from .handed import _method

        func = call.func
        if isinstance(func, ast.Name):
            found = self.resolve(func)
            return None if found is _UNKNOWN else [(found, False, None)]
        if not isinstance(func, ast.Attribute) or _super_call(
            _root(func), self.bindings
        ):
            return None
        called = self._called(func)
        if called is not _UNKNOWN:
            return [(called, False, None)]
        receiver = self._bound(func.value)
        if isinstance(receiver, types.ModuleType):
            return None  # a name the module does not hold
        if isinstance(receiver, _MEMBERS_OF):
            classes: set[type] | None = {
                type(member) for member in receiver.mobject.get_family()
            }
        elif receiver is _UNKNOWN:
            classes = self.classes(func.value)
        else:
            classes = {type(receiver)}
        if classes is None:
            return None
        found_callees: list[tuple[Any, bool, type | None]] = []
        self.writes.class_reads.add(func.attr)  # a class's binding (STORE)
        for cls in classes:
            member = inspect.getattr_static(cls, func.attr, None)
            if member is None:
                return None
            if isinstance(member, (types.FunctionType, staticmethod, classmethod)):
                found_callees.append(_method(member, cls))
            elif receiver is not _UNKNOWN and not isinstance(receiver, _MEMBERS_OF):
                found_callees.append((getattr(receiver, func.attr, None), False, None))
            else:
                found_callees.append((member, False, None))
        return found_callees

    def _callable_value(self, node: ast.AST) -> Any:
        """The callable ``node`` resolves to, a method of a resolved object
        bound to it, else ``None``.
        """
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        found = self.resolve(node)  # what an object holds is read as it is
        if found is _UNKNOWN and isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is _UNKNOWN or isinstance(base, _MEMBERS_OF):
                return None
            member = inspect.getattr_static(type(base), node.attr, None)
            if isinstance(member, staticmethod):
                return member.__func__
            if isinstance(member, classmethod):
                return types.MethodType(member.__func__, type(base))
            if isinstance(member, types.FunctionType):
                return types.MethodType(member, base)
            return None
        if found is _UNKNOWN or isinstance(found, _MEMBERS_OF) or not callable(found):
            return None
        if isinstance(
            found,
            (
                type,
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                np.ufunc,
            ),
        ):
            return found
        call = inspect.getattr_static(type(found), "__call__", None)
        if isinstance(call, types.FunctionType):
            return types.MethodType(call, found)
        return found  # a callable object the check does not read

    def _called(self, func: ast.AST) -> Any:
        """The object ``func`` names, as :meth:`resolve` finds it, and a
        ufunc's or an array's attribute (``np.add.at``, ``values.clip``).
        """
        if isinstance(func, ast.Attribute):
            base = self._called(func.value)
            if isinstance(base, (np.ufunc, np.ndarray, np.generic)):
                return getattr(base, func.attr, _UNKNOWN)
        return self.resolve(func)

    def _user_super(self, name: str) -> bool:
        """Whether ``super().name`` may run code of the scene's other than
        the function's own: some class of its ``self`` defines ``name`` in
        such code, or its ``self``'s class is not known.
        """
        if self.made_type is not None:
            cls: type = self.made_type
        elif self.bound and self.first is not None:
            cls = type(self.bindings.get(self.first))
        else:
            return True
        for klass in cls.__mro__:
            member = vars(klass).get(name)
            member = getattr(member, "__func__", getattr(member, "fget", member))
            code = _code_of(member)
            if code is not None and code is not self.code and _user_code(code):
                return True
        return False

    def _handed(
        self,
        effect: _Effect,
        call: ast.Call,
        callee: Any = None,
        bound: bool = False,
    ) -> _Effect:
        """``effect``, the callee's own, with what it changes through its
        parameters taken for changes into the arguments ``call`` binds to
        them; every argument, when the callee (a function of the scene's,
        its first parameter ``bound`` to the object it is called on; a
        method; a class of the scene's, through its constructor) is not
        known.
        """
        if not effect.writes_args:
            return effect
        found = effect.without_args()
        binding = _arguments_of(call, callee, bound)
        everything = [*call.args, *(k.value for k in call.keywords)]
        for name in effect.writes_args:
            if binding is None or name == _EVERY_ARGUMENT:
                handed = everything
            else:
                handed = binding.get(name, [])
            for argument in handed:
                found |= self.expr_writes(argument)
        return found

    def _callee_effect(self, call: ast.Call) -> _Effect:
        """What ``call``'s callee may do; what it changes through its own
        parameters as changes into the arguments, so that ``writes_args``
        here is about this function's parameters.
        """
        writes = self.writes
        func = call.func
        if isinstance(func, ast.Name):
            target = self.bindings.get(func.id, _UNKNOWN)
            if _by_constant_name(target, call):
                return _NO_EFFECT  # the attribute read it spells
            if any(target is maker for maker in _FRESH_RANDOM):
                return _NO_EFFECT
            effect = writes.callee(func.id, target, self.depth)
            return self._handed(effect, call, target)
        if not isinstance(func, ast.Attribute):
            return _UNKNOWN_CALL
        name = func.attr
        receiver = func.value
        reader = name in _READERS or name.startswith(_READER_PREFIXES)
        root = _root(receiver)
        if _super_call(root, self.bindings) and self.first is not None:
            # ``super().method()``: a method of the function's own ``self``.
            if self._user_super(name):
                return _UNKNOWN_CALL  # code of the scene's that it does not read
            if reader:
                return _NO_EFFECT
            itself = ast.Name(id=self.first, ctx=ast.Load())
            if self.first in self.fresh:
                return self.fresh_writes(self.first)
            if self.first in self.owners:
                return _NO_EFFECT
            return self.write_to(itself)
        restructures = _Effect(restructures=self.restructures_by(name, receiver))
        restructures |= self._method_code(name, receiver)
        if name in _ARRAY_MUTATORS:
            restructures |= self.into(receiver)
        if self.owned(receiver):
            if isinstance(receiver, ast.Name) and receiver.id in self.fresh:
                if reader:
                    return restructures
                return restructures | self.fresh_writes(receiver.id)
            return restructures | self._handed(self._own_method(name, receiver), call)
        if isinstance(root, ast.Name) and (
            root.id in self.owners or root.id in self.fresh
        ):
            # What the owner or a made object holds: not its own.
            return restructures if reader else restructures | self.write_to(receiver)
        if isinstance(root, ast.Name):
            holder = self.bindings.get(root.id, _UNKNOWN)
            if isinstance(holder, types.ModuleType):
                if receiver is not root:
                    inner = self._called(receiver)
                    if isinstance(inner, np.ufunc):
                        # ``np.add.at``: what it changes is what it is handed.
                        return _NO_EFFECT
                    if not isinstance(inner, types.ModuleType):
                        return restructures | self.write_to(receiver)
                    holder = inner  # ``np.linalg.norm``
                target = getattr(holder, name, _UNKNOWN)
                if any(target is maker for maker in _FRESH_RANDOM):
                    return _NO_EFFECT
                module = getattr(holder, "__name__", "")
                if (
                    module.split(".")[0] == "random"
                    or ".random" in module
                    or name == "random"
                ):
                    return _ELSEWHERE
                if not _internal(getattr(holder, "__file__", "") or ""):
                    effect = writes.callee(name, target, self.depth)
                    return self._handed(effect, call, target)
                if name in _MUTATORS:
                    return self.into(call.args[0]) if call.args else _IN_PLACE
                return _NO_EFFECT
            if reader:
                return restructures
            from . import library

            if name in library.VALUE_METHODS and not _is_mobject(holder):
                return restructures
            if holder is not _UNKNOWN and holder is not _PARAMETER and receiver is root:
                member = inspect.getattr_static(type(holder), name, None)
                this = holder
                if isinstance(member, staticmethod):
                    this = None  # its first parameter is an argument
                elif isinstance(member, classmethod):
                    this = type(holder)
                member = getattr(member, "__func__", member)
                if isinstance(member, types.FunctionType) and _user_code(
                    member.__code__
                ):
                    found = writes.effect(
                        member, frozenset(), None, self.depth + 1, this=this
                    )
                    return restructures | self._handed(
                        found, call, member, this is not None
                    )
            return restructures | self.write_to(receiver)
        if reader:
            return _NO_EFFECT
        if isinstance(root, ast.Call):
            # A method of what a call returned: of a new object when the call
            # makes one, which holds what went into it; else whatever it
            # returned may be shared.
            sources = self.call_sources(root)
            if sources is None:
                return restructures | _ELSEWHERE
            effect = _NO_EFFECT
            for source in sources:
                effect |= self.source_writes(source)
            return effect | restructures if effect.elsewhere else effect
        if isinstance(root, (ast.Constant, ast.JoinedStr)):
            return _NO_EFFECT
        if receiver is root and isinstance(
            root,
            (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp),
        ):
            return _NO_EFFECT  # the new container itself
        return restructures | self.expr_writes(root)

    def _method_code(self, name: str, receiver: ast.AST) -> _Effect:
        """A method whose code the check does not read where it is called:
        one of an object it can't resolve, or the scene's own; what it may
        change structurally is then whatever the callback's reach may.
        """
        found = self._bound(receiver)
        if found is _UNKNOWN:
            return _Effect(unknown=True)
        classes = (
            {type(member) for member in found.mobject.get_family()}
            if isinstance(found, _MEMBERS_OF)
            else {type(found)}
        )
        for cls in classes:
            member = inspect.getattr_static(cls, name, None)
            member = getattr(member, "fget", getattr(member, "__func__", member))
            code = _code_of(member)
            if code is not None and _user_code(code):
                return _Effect(unknown=True)
        return _NO_EFFECT

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


def _arguments_of(
    call: ast.Call, callee: Any, bound: bool = False, *, user_only: bool = True
) -> dict[str, list[ast.expr]] | None:
    """The arguments ``call`` binds to each parameter of ``callee`` (a
    function of the scene's, a method, a class whose constructor is one);
    ``None`` when they can't be told apart. An argument unpacked from a
    sequence or a mapping may go to any parameter from where it stands, and
    so may every positional argument after the first unpacked one.
    """
    if isinstance(callee, types.MethodType):
        callee, bound = callee.__func__, True
    elif isinstance(callee, type):
        callee, bound = inspect.getattr_static(callee, "__init__", None), True
    if not isinstance(callee, types.FunctionType) or (
        user_only and not _user_function(callee)
    ):
        return None
    code = callee.__code__
    positional = list(code.co_varnames[: code.co_argcount])
    keyword_only = list(
        code.co_varnames[code.co_argcount : code.co_argcount + code.co_kwonlyargcount]
    )
    rest = code.co_argcount + code.co_kwonlyargcount
    varargs = code.co_varnames[rest] if code.co_flags & inspect.CO_VARARGS else None
    rest += bool(varargs)
    varkw = code.co_varnames[rest] if code.co_flags & inspect.CO_VARKEYWORDS else None
    if bound:
        positional = positional[1:]
    names = [*positional, *keyword_only, *filter(None, (varargs, varkw))]
    found: dict[str, list[ast.expr]] = {name: [] for name in names}
    unpacked_at: int | None = None
    for index, argument in enumerate(call.args):
        if unpacked_at is None and isinstance(argument, ast.Starred):
            unpacked_at = index
        if unpacked_at is not None:
            # The first unpacking's length is not known: it and every
            # argument after it may land at any position from its own on.
            value = argument.value if isinstance(argument, ast.Starred) else argument
            for name in [*positional[unpacked_at:], *filter(None, (varargs,))]:
                found[name].append(value)
        elif index < len(positional):
            found[positional[index]].append(argument)
        elif varargs is not None:
            found[varargs].append(argument)
    for keyword in call.keywords:
        if keyword.arg is None:
            for name in names:
                found[name].append(keyword.value)
        elif keyword.arg in found and keyword.arg not in (varargs, varkw):
            found[keyword.arg].append(keyword.value)
        elif varkw is not None:
            found[varkw].append(keyword.value)
    return found


def _names_bound(target: ast.AST) -> list[str]:
    """The names a store to ``target`` binds (not those it stores into)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _names_bound(item)]
    if isinstance(target, ast.Starred):
        return _names_bound(target.value)
    return []


def _bound_values(node: ast.AST, names: set[str]) -> dict[str, list[ast.AST]]:
    """What each of ``names`` (locals bound only by plain stores) is bound to."""
    found: dict[str, list[ast.AST]] = {}
    for item in _walk(node):
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(item, ast.Assign):
            pairs = [(target, item.value) for target in item.targets]
        elif isinstance(item, (ast.AnnAssign, ast.NamedExpr)) and item.value:
            pairs = [(item.target, item.value)]
        for target, value in pairs:
            if isinstance(target, ast.Name) and target.id in names:
                found.setdefault(target.id, []).append(value)
    return found


def _bound_nodes(target: ast.AST) -> list[ast.Name]:
    """The name nodes a store to ``target`` binds (:func:`_names_bound`)."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _bound_nodes(item)]
    if isinstance(target, ast.Starred):
        return _bound_nodes(target.value)
    return []


def _nested_parameters(node: ast.AST) -> set[str]:
    """The parameters of the functions and lambdas ``node`` defines."""
    found: set[str] = set()
    for item in _walk(node):
        if item is not node and isinstance(
            item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            arguments = item.args
            found.update(
                each.arg
                for each in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *filter(None, (arguments.vararg, arguments.kwarg)),
                )
            )
    return found


def _local_values(node: ast.AST) -> dict[str, list[Any]]:
    """What each name the code of ``node`` binds may hold: the value each
    store binds it to (an unpacked value for each name it is unpacked into,
    an operator in place's operand, a loop's iterable for its variable);
    ``None`` for a binding whose value no expression of the code gives (a
    ``with``, an ``except``, an import, a definition, a ``global`` or
    ``nonlocal`` name, a parameter of a function it defines, a pattern).
    """
    found: dict[str, list[Any]] = {}
    covered: set[int] = set()
    for item in _walk(node):
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(item, ast.Assign):
            pairs = [(target, item.value) for target in item.targets]
        elif isinstance(item, (ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
            if item.value is not None:
                pairs = [(item.target, item.value)]
        elif isinstance(item, (ast.For, ast.AsyncFor, ast.comprehension)):
            pairs = [(item.target, item.iter)]
        for target, value in pairs:
            for bound in _bound_nodes(target):
                covered.add(id(bound))
                found.setdefault(bound.id, []).append(value)
    unknown = _nested_parameters(node)
    for item in _walk(node):
        named: str | None = None
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            if id(item) not in covered:
                named = item.id
        elif isinstance(item, (ast.Global, ast.Nonlocal)):
            unknown.update(item.names)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            unknown.update(
                (alias.asname or alias.name).split(".")[0] for alias in item.names
            )
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            named = item.name if item is not node else None
        elif isinstance(item, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            named = item.name
        elif isinstance(item, ast.MatchMapping):
            named = item.rest
        if named is not None:
            unknown.add(named)
    for name in unknown:
        found.setdefault(name, []).append(None)
    return found


#: :func:`_aliases` of each source node, by its id (kept with it).
_ALIASES: dict[int, tuple[ast.AST, dict[str, ast.AST]]] = {}


def _aliases(node: ast.AST) -> dict[str, ast.AST]:
    """The local names a function binds exactly once, each to a name or an
    attribute of one: what they hold is what that names (read from the
    source alone, so remembered).
    """
    found = _ALIASES.get(id(node))
    if found is None or found[0] is not node:
        found = (node, _find_aliases(node))
        _ALIASES[id(node)] = found
    return found[1]


def _find_aliases(node: ast.AST) -> dict[str, ast.AST]:
    bound: dict[str, list[ast.AST | None]] = {}
    for item in _walk(node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target, value = item.targets[0], item.value
            if isinstance(target, ast.Name):
                bound.setdefault(target.id, []).append(value)
                continue
            if (
                isinstance(target, ast.Tuple)
                and isinstance(value, ast.Tuple)
                and len(target.elts) == len(value.elts)
                and all(isinstance(t, ast.Name) for t in target.elts)
            ):
                # ``a, b = x.a, x.b``: each name its own value.
                for each, its in zip(target.elts, value.elts, strict=True):
                    assert isinstance(each, ast.Name)
                    bound.setdefault(each.id, []).append(its)
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
        if isinstance(item, ast.AugAssign):
            continue  # in place, or a rebinding of what it held: no new alias
        for stored in targets:
            for bound_name in _names_bound(stored):
                bound.setdefault(bound_name, []).extend([None, None])
    found: dict[str, ast.AST] = {}
    for local, values in bound.items():
        once = values[0] if len(values) == 1 else None
        if not isinstance(once, (ast.Name, ast.Attribute)):
            continue
        root = _root(once)
        if isinstance(root, ast.Name) and root.id != local:
            found[local] = once
    return found


@dataclass(frozen=True)
class _MEMBERS_OF:  # noqa: N801 (a marker)
    """Some members of ``mobject`` (what indexing it gives)."""

    mobject: Any


#: Displays whose items an item of them is.
_DISPLAYS = (ast.List, ast.Tuple, ast.Dict, ast.ListComp, ast.DictComp)


def _named_attribute(call: ast.Call) -> ast.Attribute | None:
    """``getattr(x, "name"[, default])`` as the attribute ``x.name`` it
    reads; ``None`` for any other call of it.
    """
    if (
        2 <= len(call.args) <= 3
        and not call.keywords
        and not any(isinstance(each, ast.Starred) for each in call.args)
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    ):
        found = ast.Attribute(
            value=call.args[0], attr=call.args[1].value, ctx=ast.Load()
        )
        return ast.copy_location(found, call)
    return None


def _describe_code(code: types.CodeType | None) -> str:
    return "code of the scene's" if code is None else f"{code.co_name}"


def _stored(item: ast.AST) -> list[tuple[ast.AST, ast.AST | None]]:
    """What a node of the :attr:`.coverage.Rule.STORE` or
    :attr:`.coverage.Rule.BINDING` rule binds or stores, each target with
    the value it stores (a loop's iterable for its items; ``None`` for a
    value no expression gives: a deletion, a ``with``'s entered value).
    """
    if isinstance(item, ast.Assign):
        return [(target, item.value) for target in item.targets]
    if isinstance(item, (ast.AnnAssign, ast.AugAssign)):
        return [(item.target, item.value)]
    if isinstance(item, ast.Delete):
        return [(target, None) for target in item.targets]
    if isinstance(item, (ast.For, ast.AsyncFor, ast.comprehension)):
        return [(item.target, item.iter)]
    if isinstance(item, ast.withitem) and item.optional_vars is not None:
        return [(item.optional_vars, None)]
    return []


def _store_targets(target: ast.AST) -> list[ast.Attribute | ast.Subscript]:
    """The attributes and items a store to ``target`` stores into."""
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [part for each in target.elts for part in _store_targets(each)]
    if isinstance(target, ast.Starred):
        return _store_targets(target.value)
    return []


def _bound_to(node: ast.AST, name: str) -> list[ast.AST | None]:
    """Every value the code of ``node`` binds ``name`` to: a store's value
    (a loop's iterable for its variable, the whole value for an unpacked
    target, an operator in place's operand), ``None`` for a binding no
    expression gives (a ``with``, an import, a definition, an ``except``,
    a pattern).
    """
    found: list[ast.AST | None] = []
    for item in _walk(node):
        if isinstance(item, ast.NamedExpr):
            pairs: list[tuple[ast.AST, ast.AST | None]] = [(item.target, item.value)]
        else:
            pairs = _stored(item)
        for target, value in pairs:
            if name in _names_bound(target) and not isinstance(item, ast.Delete):
                found.append(value)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if item is not node and item.name == name:
                found.append(None)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            if any(
                (alias.asname or alias.name).split(".")[0] == name
                for alias in item.names
            ):
                found.append(None)
        elif isinstance(item, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if item.name == name:
                found.append(None)
        elif isinstance(item, ast.MatchMapping) and item.rest == name:
            found.append(None)
    return found


def _held_parts(node: ast.AST) -> list[ast.expr]:
    """The parts whose callables what ``node`` (of the
    :attr:`.coverage.Rule.PARTS` or :attr:`.coverage.Rule.GENERATOR` rule)
    makes may hold: an operator's operands (a ``[f] * n``), what a
    ``... or ...`` or a ``... if ... else ...`` may give, a display's items,
    a comprehension's item; for any other form, every expression it holds.
    """
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if isinstance(node, ast.BoolOp):
        return list(node.values)
    if isinstance(node, ast.IfExp):
        return [node.body, node.orelse]
    if isinstance(node, (ast.NamedExpr, ast.Starred)):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return list(node.elts)
    if isinstance(node, ast.Dict):
        return [*filter(None, node.keys), *node.values]
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [node.elt]
    if isinstance(node, ast.DictComp):
        return [node.key, node.value]
    found: list[ast.expr] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            found.append(child)
        else:  # a part holding expressions (an interpolation)
            found.extend(_held_parts(child))
    return found


def _native_effect(target: types.BuiltinFunctionType) -> _Effect:
    """What calling a C function or a C method (held as a value, under any
    name) may do: a function of a module listed pure, nothing; a method of
    a builtin container listed pure, nothing; any other method changes the
    object it is bound to (default-deny: the name it is called by says
    nothing).
    """
    name = target.__name__
    bound = getattr(target, "__self__", None)
    if bound is None or isinstance(bound, types.ModuleType):
        module = (
            bound.__name__ if bound is not None else getattr(target, "__module__", "")
        ) or ""
        if module == "builtins":
            if name in ("exec", "eval", "setattr", "delattr"):
                return _ANY_EFFECT
            return _NO_EFFECT if name in _PURE_BUILTINS else _ELSEWHERE
        root = module.split(".")[0]
        if root not in _PURE_MODULES or ".random" in module or name in _MUTATORS:
            return _ELSEWHERE
        if root == "operator" and name in _OPERATOR_STORES:
            return _ELSEWHERE
        return _NO_EFFECT
    if isinstance(bound, _IMMUTABLE_NATIVE):
        return _NO_EFFECT
    for cls in type(bound).__mro__:
        if name in _PURE_METHODS.get(cls, ()):
            return _NO_EFFECT
    if isinstance(bound, np.ndarray):
        return _Effect(arrays=True) | _into(bound)
    # A list may be a mobject's members: whether it changes the structure
    # is the callback's reach's to say.
    return _Effect(arrays=True, unknown=True) | _into(bound)


def _numpy_callable(target: Any) -> bool:
    """Whether ``target`` is NumPy's: a ufunc, a method of an array, of a
    NumPy scalar or of a ufunc, or a function of a NumPy module. A class
    is not: what it makes it makes anew.
    """
    if isinstance(target, np.ufunc):
        return True
    if isinstance(target, type) or type(target).__module__.split(".")[0] not in (
        "numpy",
        "builtins",
    ):
        return False  # never a scene object's own attribute lookup
    if isinstance(
        getattr(target, "__self__", None), (np.ndarray, np.generic, np.ufunc)
    ):
        return True
    owner = getattr(target, "__objclass__", None)
    if isinstance(owner, type) and issubclass(owner, (np.ndarray, np.generic)):
        return True
    module = getattr(target, "__module__", None)
    return isinstance(module, str) and module.split(".")[0] == "numpy"


def _out_position(target: Any) -> tuple[int, bool] | None:
    """Where NumPy callable ``target`` takes, by position, an array it
    writes: the first such position, and whether every later one is one
    too (a ufunc's outputs follow its inputs); ``None`` when no position
    is. A signature that can't be read leaves every position one
    (default-deny).
    """
    if isinstance(target, np.ufunc):
        return target.nargs - target.nout, True  # its inputs, then its outputs
    if isinstance(getattr(target, "__self__", None), np.ufunc) and (
        target.__name__ == "at"
    ):
        return 0, False  # ``np.add.at(a, indices, b)`` changes ``a``
    try:
        parameters = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return 0, True
    for index, parameter in enumerate(parameters):
        if parameter.kind is parameter.VAR_POSITIONAL:
            return None
        if parameter.name == "out" and parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            return index, False
    return None


def _numpy_writes(target: Any, call: ast.Call, method: str | None) -> list[ast.expr]:
    """The arguments of ``call`` that NumPy may write into by position, or
    through keywords unpacked from a mapping: those at the position its
    callable ``target`` takes an output at (when ``target`` is unknown,
    the position an array's ``method`` of that name takes it at), and
    every one an unpacked ``*args`` before that position may move there.
    """
    numpy = target is not _UNKNOWN and _numpy_callable(target)
    if numpy:
        position = _out_position(target)
    else:
        descriptor = (
            getattr(np.ndarray, method, None)
            if method is not None and target is _UNKNOWN
            else None
        )
        numpy = callable(descriptor)
        found = _out_position(descriptor) if numpy else None
        # The receiver is the descriptor's first parameter.
        position = None if found is None else (max(found[0] - 1, 0), found[1])
    written: list[ast.expr] = []
    if position is not None:
        start, onward = position
        args = call.args
        star = next(
            (k for k, a in enumerate(args) if isinstance(a, ast.Starred)), len(args)
        )
        if star <= start:
            written += args[star:]
        else:
            written += args[start:] if onward else args[start : start + 1]
    if numpy:
        written += [k.value for k in call.keywords if k.arg is None]
    return written


#: Callables that hand back a new generator of random numbers and change no
#: other: what the generator draws changes only itself.
_FRESH_RANDOM = (np.random.default_rng, np.random.Generator, np.random.RandomState)


def _by_constant_name(target: Any, call: ast.Call) -> bool:
    """Whether ``call`` is ``getattr(obj, "name"[, default])`` or
    ``hasattr(obj, "name")``: the attribute read it spells.
    """
    return (
        (target is getattr or target is hasattr)
        and 2 <= len(call.args) <= 3
        and not call.keywords
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
        and not any(isinstance(a, ast.Starred) for a in call.args)
    )


def _super_call(node: ast.AST, bindings: dict[str, Any]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and bindings.get(node.func.id) is super
    )


def _module_named(node: ast.AST, bindings: dict[str, Any]) -> types.ModuleType | None:
    """The module ``node`` names through ``bindings`` (``np``,
    ``np.random``), if it names one.
    """
    if isinstance(node, ast.Name):
        found = bindings.get(node.id)
    elif isinstance(node, ast.Attribute):
        holder = _module_named(node.value, bindings)
        found = None if holder is None else getattr(holder, node.attr, None)
    else:
        return None
    return found if isinstance(found, types.ModuleType) else None


def _own_nodes(node: ast.AST) -> list[ast.AST]:
    """The nodes of a function's own body, not of the functions, lambdas or
    classes it defines.
    """
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        item = stack.pop()
        found.append(item)
        if not isinstance(
            item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            stack.extend(ast.iter_child_nodes(item))
    return found


def _user_function(value: Any) -> TypeGuard[types.FunctionType]:
    return isinstance(value, types.FunctionType) and _user_code(value.__code__)


def _assignment_hooks(
    cls: type, name: str, deleting: bool
) -> list[tuple[types.FunctionType, Any]]:
    """The scene's own code a store into (or a deletion of) attribute
    ``name`` of an instance of ``cls`` runs, each with the descriptor it runs
    on (``None`` for one that runs on the instance): the class's
    ``__setattr__`` (``__delattr__``), a property's setter (deleter), a
    descriptor's ``__set__`` (``__delete__``).
    """
    found: list[tuple[types.FunctionType, Any]] = []
    hook = inspect.getattr_static(cls, "__delattr__" if deleting else "__setattr__")
    if _user_function(hook):
        found.append((hook, None))
    descriptor = inspect.getattr_static(cls, name, None)
    if isinstance(descriptor, property):
        function = descriptor.fdel if deleting else descriptor.fset
        if _user_function(function):
            found.append((function, None))
    elif descriptor is not None:
        method = inspect.getattr_static(
            type(descriptor), "__delete__" if deleting else "__set__", None
        )
        if _user_function(method):
            found.append((method, descriptor))
    return found


def _class_hooks(cls: type) -> list[tuple[types.FunctionType, Any]]:
    """Every hook a store into or a deletion of any attribute of an
    instance of ``cls`` may run (:func:`_assignment_hooks`).
    """
    # The scene's own hooks live in the scene's own classes.
    names = {
        name
        for klass in cls.__mro__
        if not engine_class(klass) and klass is not object
        for name in vars(klass)
    }
    names.add("__dict__")  # for the class-wide __setattr__ and __delattr__
    found: dict[int, tuple[types.FunctionType, Any]] = {}
    for name in names:
        for deleting in (False, True):
            for hook, this in _assignment_hooks(cls, name, deleting):
                found[id(hook)] = (hook, this)
    return list(found.values())


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
    #: "step" runs at every tick (a dt updater, a scene updater, the camera's
    #: inputs); "instant" (a dt-free updater or redraw) at full ticks only;
    #: "transparent" (an animation's interpolation) at every tick, what it
    #: writes counting and what it reads not.
    role: str
    #: What it may read: a set, or how to find it (worked out only for the
    #: stepping work that asks).
    reach_of: frozenset[int] | Callable[[], frozenset[int]]
    #: What it may write: a set, how to find it, or ``_REACHED`` for
    #: anything it reaches.
    writes: frozenset[int] | Callable[[], frozenset[int]] | None = frozenset()
    keys: tuple[tuple[int, int], ...] = ()

    @property
    def reads(self) -> frozenset[int]:
        if callable(self.reach_of):
            self.reach_of = self.reach_of()
        return self.reach_of

    def written(self) -> frozenset[int]:
        """What it may write. Writes bounded only by what it reaches raise
        :class:`_UnboundedWrites`: what it reaches is read from what objects
        hold when it is asked, which may change as the play runs.
        """
        if self.writes is _REACHED:
            raise _UnboundedWrites(f"{self.what} may write whatever it reaches")
        if callable(self.writes):
            self.writes = self.writes()
        return self.writes


def _family_ids(mob: Any) -> frozenset[int]:
    return frozenset(id(member) for member in mob.get_family())


def _effect_writes(
    effect: _Effect, mob: Any
) -> frozenset[int] | Callable[[], frozenset[int]] | None:
    """What a callback with ``effect`` that updates ``mob`` (``None`` for
    none) may write: ``mob``'s family (whose members every stepping tick
    compares), and the arrays its changes elsewhere go into, by the memory
    each holds. A change into any other object, or anything it holds, would
    be bounded by what that object holds when the plan is made, which may
    change as the play runs: asked for it, the play ticks everything
    (:class:`Unbounded`), as for anything it reaches when a target is not
    resolved.
    """
    # (What it changes through its parameters is what it is handed: the
    # mobject it updates, and a number, dt or alpha.)
    if effect.anywhere:
        return _REACHED
    own = _family_writes(mob) if mob is not None else frozenset()
    if not effect.targets:
        return own
    targets = effect.targets

    def written() -> frozenset[int]:
        for target in targets:
            if isinstance(target, _CodeRef) or (
                target is not _IN_PLACE_UNRESOLVED
                and not isinstance(target.value, np.ndarray)
            ):
                raise _UnboundedWrites(
                    f"writes into what a {type(target.value).__name__} holds"
                )
        return own | _written_into(targets)

    return written


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

    known = frozenset(trusted.writes(animation))
    writes: frozenset[int] | Callable[[], frozenset[int]] | None = known
    effect = _NO_EFFECT
    if isinstance(animation, trusted.CALLBACK_ANIMATIONS):
        function = vars(animation)["update_function"]
        effect = check.effect(function, _owners(function), animation.mobject)
        if effect.elsewhere:
            more = _effect_writes(effect, None)
            if more is _REACHED:
                writes = _REACHED
            else:

                def both() -> frozenset[int]:
                    extra = more() if callable(more) else more
                    return known | (extra or frozenset())

                writes = both
    return _Node(
        type(animation).__name__,
        "transparent",
        lambda: reach([animation], trusted._mobjects_of(animation)),
        writes,
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

    writes = _effect_writes(effect, target)
    role = "step" if loads_dt(updater) else "instant"
    return _Node(
        what,
        role,
        reads() if role == "step" else reads,
        writes,
        (key,),
    )


def plan_ticks(
    scene: Scene, animations: list[Animation], base_dir: str | None = None
) -> TickPlan:
    """The stepping plan of the ticked play ``scene`` is about to evaluate
    (its animations begun); source paths in its reason are relative to
    ``base_dir``.

    A stepping tick runs every updater whose code loads ``dt`` and leaves out
    every other, judged as each is called (:func:`skipping`): the plan only
    records what it found, for the structure a stepping tick compares and
    the counts. The play ticks everything when its updaters do not run
    through :meth:`.Mobject.update`, where nothing could leave one out.
    """
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
    if vars(scene.renderer.camera).get("image_mobjects_from_cameras"):
        return unbounded(
            "the camera draws other cameras, whose inputs it does not weigh"
        )
    roots = (*_copies(animations), *scene.mobjects)
    known: set[tuple[int, int]] = set()
    skipped: set[tuple[int, int]] = set()
    for member, updater in _updaters(roots):
        key = (id(member), id(updater))
        known.add(key)
        if not loads_dt(updater):
            skipped.add(key)
    begun = list(animations)
    return TickPlan(
        skipped=frozenset(skipped),
        known=frozenset(known),
        stepping=len(known) - len(skipped),
        families=functools.cache(lambda: _stepping_families(scene, begun, base_dir)),
        structure=current_structure(scene, animations),
    )


# Diagnostic only: what the stepping set may read or write


def _stepping_families(
    scene: Scene, animations: list[Animation], base_dir: str | None
) -> frozenset[int]:
    """:meth:`TickPlan.stepping_families`: what the work every tick runs (dt
    updaters, the scene's updaters, every interpolation, the camera's
    inputs) may read, and what it and every interpolation may write; every
    member of the roots' families when what one writes can't be bounded, or
    when the analysis can't bound a read or rests on a value that may change
    as the play runs (:meth:`_Writes.plan_reads_hold`).
    """
    from arachne.animation.composition import AnimationGroup

    from . import trusted

    roots = (*_copies(animations), *scene.mobjects)
    members = frozenset(id(member) for root in roots for member in root.get_family())
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
        for member, updater in _updaters(roots):
            nodes.append(_updater_node(member, updater, check, reaches, base_dir))
        for function in scene.updaters:
            check.effect(function, frozenset())
            nodes.append(
                _Node(
                    "scene updater " + _describe(function, base_dir),
                    "step",
                    reach([function]),
                    _REACHED,
                )
            )
        camera = scene.renderer.camera
        centres = getattr(camera, "fixed_orientation_mobjects", None) or {}
        if centres:
            nodes.append(
                _Node("camera centres", "step", reach(list(centres.values()), centres))
            )
        check.plan_reads_hold()
        found: set[int] = set()
        for node in nodes:
            if node.role == "step":
                found |= node.reads
            if node.role != "instant":
                try:
                    found |= node.written()
                except Unbounded:
                    found |= members
    except Unbounded:
        return members
    return frozenset(found)
