"""Content keys of the values a mobject's fields hold.

:func:`value_key` keys every value a field can hold: equal for two values
that hold the same content, in this process or another, and apart for two
that do not. A mobject a value refers to stands in the key as its type
only; the key gives the mobjects it refers to beside it, in order, and the
caller keys which mobjects they are (a state's key, by their places in its
walk), as it keys a parent's children.

A value the key cannot key is a *marker* that never equals another key,
its own included, so that every comparison that meets it finds a
difference rather than agreement: a value of a type the key does not know
(:data:`UNKEYABLE`), a cache whose entries it cannot read
(:data:`UNREAD_CACHE`), a value holding more parts than one key walks
(:data:`BUDGET` parts, :data:`OVER_BUDGET`), one nested deeper than it
follows (:data:`DEPTH`, :data:`OVER_DEPTH`) and one whose reading raised
(a property of the scene's own, a library value's method: :data:`RAISED`).
A builtin kind's value (a subclass's of ``str``, ``list``, ``dict``) is
read through the kind's own methods, never ones the subclass overrides.

A class (an enum member's included) and a function are named by their
qualified names, not their modules': a scene's module is named for the load
that made it (each session imports the file under a name of its own, a
render by its path). A function is keyed by its code and what it closes
over, not by the globals it reads, which the source digest covers. A scene
and a camera key as their types: what they do reaches a frame only through
the mobjects and the camera inputs a capture keys.

Containers and objects are keyed by what they hold when the key is made: a
record keeps a field's value by reference, so a container changed in place
after a capture keys as it is now (as the record's own change detection,
by identity, sees no change either).
"""

from __future__ import annotations

__all__ = [
    "BUDGET",
    "OVER_BUDGET",
    "OVER_BUDGET_KIND",
    "RAISED",
    "READING_ENGINE_CACHES",
    "UNKEYABLE",
    "UNREAD_CACHE",
    "marker_reason",
    "unkeyable",
    "value_key",
]

import collections
import dataclasses
import enum
import functools
import hashlib
import os
import pathlib
import random
import sys
import types
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

#: How many parts one key walks (every value it meets, however deep) before
#: the value it keys becomes the marker :data:`OVER_BUDGET`.
BUDGET = 100_000

#: How deep one key follows values held inside values before the value it
#: keys becomes the marker :data:`OVER_DEPTH`: well inside the interpreter's
#: recursion limit, whatever stack the caller stands on.
DEPTH = 100

#: The head of the marker of a value of a type the key does not know.
UNKEYABLE = b"\0unkeyable "
#: The head of the marker of a cache whose entries the key cannot read.
UNREAD_CACHE = b"\0unread cache "
#: The head of the marker of a value holding more than :data:`BUDGET` parts.
OVER_BUDGET = b"\0over budget "
#: The head of the marker of a value nested deeper than :data:`DEPTH`.
OVER_DEPTH = b"\0over depth "
#: The head of the marker of a value whose reading raised.
RAISED = b"\0raised "
#: What :func:`unkeyable` names a value over the budget, and one too deep.
OVER_BUDGET_KIND = "over the key budget"
OVER_DEPTH_KIND = "over the key depth"


class _Unkeyable(Exception):  # noqa: N818 (a finding, named as one)
    """A value the key cannot key: of a type it does not know, by the type's
    name; a cache it cannot read, by its function's name (``head``
    :data:`UNREAD_CACHE`); over the budget (:data:`OVER_BUDGET_KIND`) or
    too deep (:data:`OVER_DEPTH_KIND`).
    """

    def __init__(self, kind: str, head: bytes = UNKEYABLE) -> None:
        super().__init__(kind)
        self.kind = kind
        self.head = head


class _Keying:
    """One key being made: its rounding, the mobjects referred to so far,
    the ids of the values being keyed around the current one (so that a
    value holding itself keys as a reference back), and the parts left.
    """

    __slots__ = ("active", "decimals", "left", "refers")

    def __init__(self, decimals: int | None) -> None:
        self.decimals = decimals
        self.refers: list[Mobject] = []
        self.active: list[int] = []
        self.left = BUDGET


def value_key(
    value: Any, decimals: int | None = None
) -> tuple[bytes, tuple[Mobject, ...]]:
    """``value``'s content key, its floats rounded to ``decimals`` when
    given, and the mobjects it refers to in the order they stand in it; for
    a value the key cannot key, a marker equal to no other key (and no
    mobject).
    """
    keying = _Keying(decimals)
    try:
        return _key(value, keying), tuple(keying.refers)
    except Exception as found:  # noqa: BLE001 (every value keys, or is a marker)
        # Each marker names its cause's type: the type that fell through, or
        # that of the value too large or too deep to key, or what raised.
        head, kind = _marker(_found(found, value), value)
        return head + kind.encode() + b"\0" + os.urandom(16), ()


def unkeyable(value: Any) -> str | None:
    """What in ``value`` the key cannot key, the first it meets: a type's
    qualified name, a cache's function's name, :data:`OVER_BUDGET_KIND`,
    :data:`OVER_DEPTH_KIND`, or the exception reading a value raised and
    the value's type; ``None`` when it keys all of ``value``.
    """
    try:
        _key(value, _Keying(None))
    except Exception as found:  # noqa: BLE001 (as value_key)
        return _found(found, value).kind
    return None


def _found(found: Exception, value: Any) -> _Unkeyable:
    """``found``, raised keying ``value``, as the finding it is: a
    recursion the depth did not bound (code that recurses as it is read)
    is too deep; anything else raised is a raise of reading ``value``.
    """
    if isinstance(found, _Unkeyable):
        return found
    if isinstance(found, RecursionError):
        return _Unkeyable(OVER_DEPTH_KIND)
    return _raised(found, value)


def _raised(error: Exception, value: Any) -> _Unkeyable:
    return _Unkeyable(
        f"{_qualified(type(error))}, keying a {_qualified(type(value))}", RAISED
    )


def _marker(found: _Unkeyable, value: Any) -> tuple[bytes, str]:
    """The head and kind of the marker of ``found``, raised keying
    ``value``: an overflow names ``value``'s type.
    """
    if found.kind == OVER_BUDGET_KIND:
        return OVER_BUDGET, _qualified(type(value))
    if found.kind == OVER_DEPTH_KIND:
        return OVER_DEPTH, _qualified(type(value))
    return found.head, found.kind


def marker_reason(key: bytes) -> str | None:
    """Why ``key`` is a marker, in words, naming the type it is a marker
    for, or ``None`` for a key of content.
    """
    for head in (UNKEYABLE, UNREAD_CACHE, OVER_BUDGET, OVER_DEPTH, RAISED):
        if key.startswith(head):
            kind = key[len(head) :].split(b"\0", 1)[0].decode()
            if head == UNKEYABLE:
                return f"a value of a type the key does not know ({kind})"
            if head == UNREAD_CACHE:
                return f"a cache whose entries the key cannot read ({kind})"
            if head == RAISED:
                return f"raised: {kind}"
            if head == OVER_BUDGET:
                return f"a {kind} of more than {BUDGET} parts, too large to key"
            return f"a {kind} nested deeper than {DEPTH} levels, too deep to key"
    return None


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _framed(data: bytes) -> bytes:
    """``data`` with its length before it, so that two runs of parts key
    apart wherever they split.
    """
    return len(data).to_bytes(4, "big") + data


def _digest(data: bytes) -> bytes:
    """A payload as 16 bytes, so that its key does not grow with it."""
    return hashlib.blake2b(data, digest_size=16).digest()


def _dtype_name(dtype: np.dtype[Any]) -> str:
    """``dtype`` as a key names it: its ``str`` (which is its kind and size),
    and for a structured one, whose ``str`` is only the void of its size,
    its fields' names, types and offsets too.
    """
    if dtype.fields is None and dtype.subdtype is None:
        return str(dtype.str)
    return f"{dtype}{dtype.itemsize}"


def _key(value: Any, keying: _Keying) -> bytes:
    """``value``'s key, the mobjects it refers to appended to
    ``keying.refers``.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import FrozenArray, Paint

    keying.left -= 1
    if keying.left < 0:
        raise _Unkeyable(OVER_BUDGET_KIND)
    decimals = keying.decimals
    if isinstance(value, (Paint, Placement)):
        return value.key
    if isinstance(value, FrozenArray):
        shape = _framed(f"{_dtype_name(value.dtype)}{value.shape}".encode())
        if decimals is not None and value.dtype.kind in "fc":
            # Adding zero turns -0.0, which rounding makes, into 0.0.
            value = np.round(value, decimals) + 0.0
        return shape + _digest(value.tobytes())
    if isinstance(value, enum.Enum):
        # Its class and its value, as another process names them alike.
        return (
            b"enum "
            + _framed(type(value).__qualname__.encode())
            + _key(value.value, keying)
        )
    if type(value) is float and decimals is not None:
        return repr(round(value, decimals) + 0.0).encode()
    if value is None or value is Ellipsis or type(value) in (bool, int, float, str):
        return repr(value).encode()
    if isinstance(value, Mobject):
        # Which mobject it is, the caller keys; its own state is a node's.
        keying.refers.append(value)
        return b"mobject " + _framed(type(value).__qualname__.encode())
    if isinstance(value, np.generic):
        return (
            b"scalar "
            + _framed(_dtype_name(value.dtype).encode())
            + _key(value.item(), keying)
        )
    active = keying.active
    if id(value) in active:
        return b"back " + active[::-1].index(id(value)).to_bytes(4, "big")
    if len(active) >= DEPTH:
        raise _Unkeyable(OVER_DEPTH_KIND)
    active.append(id(value))
    try:
        return _held_key(value, keying)
    except (_Unkeyable, RecursionError):
        raise
    except Exception as error:  # noqa: BLE001 (a scene's own code, read)
        # Named by the innermost value whose reading raised.
        raise _raised(error, value) from error
    finally:
        active.pop()


def _held_key(value: Any, keying: _Keying) -> bytes:
    """The key of a value that holds others: an array, a container, a colour,
    a function or an object of a kind the key knows; a value of a builtin
    kind's subclass with its own attributes too.
    """
    head = b"of " + _framed(type(value).__qualname__.encode())
    return head + _content_key(value, keying) + _own_attributes(value, keying)


def _each(parts: Any, keying: _Keying) -> bytes:
    found = [_key(part, keying) for part in parts]
    return len(found).to_bytes(4, "big") + b"".join(map(_framed, found))


def _own_attributes(value: Any, keying: _Keying) -> bytes:
    """The attributes and slots an instance of a subclass of a builtin kind
    (a ``str``, a ``list``, a ``dict``), of ``functools.partial`` or of an
    array (a masked array's mask) holds beside its value, which no rule for
    the kind reads; nothing for another value.
    """
    if (
        not isinstance(value, _OWNING_KINDS)
        or type(value) in _BUILTIN_KINDS
        or type(value) is np.ndarray
    ):
        return b""
    own = getattr(value, "__dict__", None) or {}
    slots = _slot_items(value)
    if not own and not slots:
        return b""
    return _each(sorted(own.items()), keying) + _each(slots, keying)


#: The builtin kinds a rule of :func:`_content_key` keys by their values.
_BUILTIN_KINDS = (
    int,
    float,
    str,
    tuple,
    list,
    dict,
    set,
    frozenset,
    bytes,
    bytearray,
    complex,
)


#: The kinds whose instances can hold attributes beside what a rule of
#: :func:`_content_key` keys of them.
_OWNING_KINDS = (*_BUILTIN_KINDS, functools.partial, np.ndarray)


def _content_key(value: Any, keying: _Keying) -> bytes:
    """``value``'s key past its type (:func:`_held_key`)."""
    from arachne.camera.camera import Camera
    from arachne.scene.scene import Scene
    from arachne.utils.color import ManimColor

    def each(parts: Any) -> bytes:
        return _each(parts, keying)

    # A builtin kind's value is read through the kind's own methods, never
    # ones a subclass overrides (its ``__str__``, ``__iter__``, ``items``),
    # which can report one thing for different values.
    if isinstance(value, (int, float, str)):
        # A subclass of a number or a string: its value, as the builtin kind
        # it is (whatever it mixes in); its class and what else it holds
        # are :func:`_held_key`'s.
        kind: Any = next(k for k in type(value).__mro__ if k in (int, float, str))
        return _framed(kind.__repr__(value).encode())
    if isinstance(value, np.ndarray):
        array = np.asarray(value).view(np.ndarray)
        shape = _framed(f"{_dtype_name(array.dtype)}{array.shape}".encode())
        if array.dtype.hasobject:
            return shape + each(array.ravel().tolist())
        if keying.decimals is not None and array.dtype.kind in "fc":
            array = np.round(array, keying.decimals) + 0.0
        return shape + _digest(np.ascontiguousarray(array).tobytes())
    if isinstance(value, ManimColor):
        return _key(value.to_rgba(), keying)
    if isinstance(value, tuple):
        return each(tuple.__iter__(value))
    if isinstance(value, list):
        return each(list.__iter__(value))
    if isinstance(value, (dict, set, frozenset)):
        # A dict in the order it holds its items, which code iterating it
        # reads (a colour map applied in order); a set, which has no order a
        # program can rely on, in the order of its parts' keys. Each part
        # with the mobjects it refers to.
        pairs: Any
        parts: Any
        if isinstance(value, dict):
            ordered, pairs = True, dict.items(value)
        else:
            if isinstance(value, set):
                parts = set.__iter__(value)
            else:
                parts = frozenset.__iter__(value)
            ordered, pairs = False, ((part,) for part in parts)
        keyed = []
        held = keying.refers
        for pair in pairs:
            keying.refers = []
            part_key = b"".join(_framed(_key(part, keying)) for part in pair)
            keyed.append((part_key, keying.refers))
        keying.refers = held
        if not ordered:
            keyed.sort(key=lambda item: item[0])
        for _, own in keyed:
            held.extend(own)
        found = len(keyed).to_bytes(4, "big") + b"".join(
            _framed(part_key) for part_key, _ in keyed
        )
        if isinstance(value, collections.defaultdict):
            factory = vars(collections.defaultdict)["default_factory"].__get__(value)
            found += _key(factory, keying)
        return found
    if isinstance(value, bytes):
        return _digest(bytes.__bytes__(value))
    if isinstance(value, bytearray):
        return _digest(bytes(bytearray.copy(value)))
    if isinstance(value, complex):
        return _framed(complex.__repr__(value).encode())
    if isinstance(value, Fraction):
        return _framed(Fraction.__repr__(value).encode())
    if isinstance(value, (range, slice)):
        # Neither can be subclassed: its own ``repr``.
        return _framed(repr(value).encode())
    if isinstance(value, type):
        return b"class " + _framed(value.__qualname__.encode())
    if isinstance(value, types.FunctionType):
        # What it runs, what it closes over and its own attributes, not where
        # it is: a function made twice from one definition keys alike.
        cells = [_contents(cell) for cell in value.__closure__ or ()]
        return (
            _framed(value.__qualname__.encode())
            + _code_key(value.__code__, keying)
            + each(value.__defaults__ or ())
            + _key(value.__kwdefaults__, keying)
            + each(cells)
            + each(sorted(vars(value).items()))
        )
    if isinstance(value, types.MethodType):
        return each((value.__func__, value.__self__))
    if isinstance(value, np.ufunc):
        # One numpy or scipy.special names, by that module and its name;
        # another (``np.frompyfunc``'s) is code of this process. scipy is
        # looked at only where it is loaded already.
        name = value.__name__
        for named_by in ("numpy", "scipy.special"):
            loaded = sys.modules.get(named_by)
            if loaded is not None and getattr(loaded, name, None) is value:
                return _framed(named_by.encode()) + _framed(name.encode())
        raise _Unkeyable(_qualified(type(value)))
    if isinstance(value, types.BuiltinFunctionType):
        owner = value.__self__
        bound = () if owner is None or isinstance(owner, types.ModuleType) else (owner,)
        # A builtin's module is named alike in every load: ``math`` and
        # ``cmath`` each have a ``sin``.
        name = f"{value.__module__}.{value.__qualname__}"
        return _framed(name.encode()) + each(bound)
    if isinstance(value, functools.partial):
        return each((value.func, value.args, value.keywords))
    if isinstance(value, np.vectorize):
        # The function it vectorizes and how; not the ufunc it builds and
        # caches on its first call.
        return each(
            (value.pyfunc, value.otypes, value.excluded, value.signature, value.cache)
        )
    if isinstance(value, functools._lru_cache_wrapper):
        # What a call returns is its entries, which can't be read. A cache of
        # an engine function of its arguments alone holds that function's
        # values: it keys as the function and how it caches. Any other is a
        # marker naming the function, a scene's by its qualified name only.
        cached = value.__wrapped__
        module = getattr(cached, "__module__", None) or ""
        qualname = getattr(cached, "__qualname__", None) or _qualified(type(cached))
        name = f"{module}.{qualname}"
        if not _engine_module(module):
            raise _Unkeyable(qualname, UNREAD_CACHE)
        if name in READING_ENGINE_CACHES:
            raise _Unkeyable(name, UNREAD_CACHE)
        return each((cached, sorted(value.cache_parameters().items())))
    if isinstance(value, pathlib.PurePath):
        # Where a file was written differs between executions (their media
        # directories); what it holds is keyed where it was read.
        return _framed(value.name.encode())
    if isinstance(value, (Scene, Camera, types.ModuleType)):
        # What a scene and a camera do lands in the mobjects and camera
        # inputs a capture keys; a module is code, as the source digest is.
        return b""
    drawn = _generator_state(value)
    if drawn is not None:
        # A random generator by the state its next draw reads, and a
        # subclass's own attributes and slots beside it.
        attributes = getattr(value, "__dict__", None) or {}
        return (
            _key(drawn, keying)
            + each(sorted(attributes.items()))
            + each(_slot_items(value))
        )
    foreign = _foreign_key(value)
    if foreign is not None:
        return _key(foreign, keying)
    if not _state_in_python(type(value)):
        raise _Unkeyable(_qualified(type(value)))
    if dataclasses.is_dataclass(value) and not hasattr(value, "__dict__"):
        return each(
            (field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(getattr(value, "__dict__", None), dict):
        # An object of a plain class (the scene's own helpers, a dataclass,
        # the engine's scales): its type, its attributes and any slots.
        return each(sorted(vars(value).items())) + each(_slot_items(value))
    raise _Unkeyable(_qualified(type(value)))


def _state_in_python(cls: type) -> bool:
    """Whether an instance of ``cls`` holds its state where Python reads it,
    in its attributes or its slots: no builtin or extension base of it holds
    any (``io.StringIO``'s buffer, ``collections.deque``'s parts), so its
    layout is ``object``'s, slots aside.
    """
    size = cls.__basicsize__
    for klass in cls.__mro__:
        slots = vars(klass).get("__slots__", ())
        size -= len((slots,) if isinstance(slots, str) else slots) * _POINTER
    return size == object.__basicsize__


_POINTER = np.dtype(np.intp).itemsize

#: The engine's caches of functions that read more than their arguments (the
#: fonts installed, the engine's files, the file system), by the qualified
#: name of the function each caches: keyed as anyone's cache, a marker.
READING_ENGINE_CACHES = frozenset(
    {
        "arachne.mobject.text.text_mobject.Text.font_list",
        "arachne.mobject.text.text_mobject.MarkupText.font_list",
        "arachne.server.store.engine_fingerprint",
        "arachne.timeline.source._is_file",
        "arachne.timeline.source._relative_path",
        "arachne.utils.typeset_cache._answer",
        "arachne.utils.typeset_cache._file_digest",
        "arachne.utils.typeset_cache._font_list",
        "arachne.utils.typeset_cache._harfbuzz_version",
        "arachne.utils.typeset_cache._listing",
        "arachne.utils.typeset_cache._loaded_configs",
        "arachne.utils.typeset_cache._only_tex_reads",
        "arachne.utils.typeset_fonts._loaded",
    }
)


def _engine_module(module: str) -> bool:
    """Whether ``module`` is the engine's own: ``arachne`` or one inside
    it, never a scene's (each loaded as ``arachne_session_…``).
    """
    return module == "arachne" or module.startswith("arachne.")


def _slot_items(value: Any) -> list[tuple[str, Any]]:
    """The slots ``value``'s classes declare that hold a value, each with
    its value, by name.
    """
    found = []
    missing = object()
    for klass in type(value).__mro__:
        slots = vars(klass).get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"
            held = getattr(value, name, missing)
            if held is not missing:
                found.append((name, held))
    return sorted(found, key=lambda item: item[0])


#: What an empty closure cell holds, for its key: a variable the function
#: reads that is not yet bound.
_EMPTY = ("empty cell",)


def _contents(cell: types.CellType) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY


def _code_key(code: types.CodeType, keying: _Keying) -> bytes:
    """The key of a function's code: its instructions, constants and names,
    and how it is called (its parameters, by count, kind and name: an
    updater taking ``dt`` is called with it).
    """
    signature = (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )
    consts = [
        _code_key(const, keying)
        if isinstance(const, types.CodeType)
        else _key(const, keying)
        for const in code.co_consts
    ]
    return (
        _framed(code.co_code)
        + _framed(repr(signature).encode())
        + _framed(" ".join(code.co_varnames).encode())
        + _framed(" ".join(code.co_names).encode())
        + len(consts).to_bytes(4, "big")
        + b"".join(map(_framed, consts))
    )


def _generator_state(value: Any) -> Any:
    """The state a random generator's next draw reads (Python's, numpy's
    legacy ``RandomState``, or a numpy ``Generator`` by its bit generator's
    type and state), read through the kind's own methods; ``None`` for
    another value. A bit generator of no numpy module's is a marker, and so
    is a generator whose draws need not come from that state
    (:func:`_draws_from_its_state`).
    """
    kinds: tuple[type, ...] = (
        random.Random,
        np.random.RandomState,
        np.random.Generator,
    )
    kind = next((kind for kind in kinds if isinstance(value, kind)), None)
    if kind is None:
        return None
    if not _draws_from_its_state(value, kind):
        raise _Unkeyable(_qualified(type(value)))
    if isinstance(value, random.Random):
        return random.Random.getstate(value)
    if isinstance(value, np.random.RandomState):
        return np.random.RandomState.get_state(value)
    bits = vars(np.random.Generator)["bit_generator"].__get__(value)
    if not type(bits).__module__.startswith("numpy.random."):
        raise _Unkeyable(_qualified(type(bits)))
    return (_qualified(type(bits)), bits.state)


def _draws_from_its_state(value: Any, kind: type) -> bool:
    """Whether ``value``, a generator of ``kind``, draws from the state
    ``kind`` reports: neither its classes below ``kind`` nor the instance
    replace one of ``kind``'s public methods (``random.SystemRandom``'s
    ``random`` reads the system, never its state).
    """
    drawing = {name for name in dir(kind) if not name.startswith("_")}
    for klass in type(value).__mro__:
        if klass is kind:
            break
        if drawing & vars(klass).keys():
            return False
    own = getattr(value, "__dict__", None) or {}
    return not drawing & own.keys()


def _foreign_key(value: Any) -> Any:
    """What stands for a value of a library the engine's mobjects hold (an
    image, a parsed path or document, a graph), as values the key knows, or
    ``None`` for another.
    """
    image = sys.modules.get("PIL.Image")
    if image is not None and isinstance(value, image.Image):
        # ``info`` too: its transparency is read where it converts to colours.
        # The palette in the mode it is held in, which keeps its alpha.
        palette = getattr(value, "palette", None)
        mode = getattr(palette, "mode", None) or "RGB"
        return (
            value.mode,
            value.size,
            value.tobytes(),
            mode,
            value.getpalette(mode),
            value.info,
        )
    svg = sys.modules.get("svgelements")
    if svg is not None and isinstance(value, svg.Path):
        # Its segments as they are, which iterating it reads, and the
        # transform its ``d()`` applies to them.
        return (value.d(transformed=False), str(value.transform))
    soup = sys.modules.get("bs4.element")
    if soup is not None and isinstance(value, soup.Tag):
        return str(value)
    networkx = sys.modules.get("networkx")
    if networkx is not None and isinstance(value, networkx.Graph):
        # A multigraph's edges with their keys, which tell its edges apart.
        edges = (
            value.edges(keys=True, data=True)
            if value.is_multigraph()
            else value.edges(data=True)
        )
        return (
            value.graph,
            [list(node) for node in value.nodes(data=True)],
            [list(edge) for edge in edges],
        )
    return None
