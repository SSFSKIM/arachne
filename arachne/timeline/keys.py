"""Content keys of the values a mobject's fields hold.

:func:`value_key` keys every value a field can hold: equal for two values
that hold the same content, in this process or another, and apart for two
that do not. A mobject a value refers to stands in the key as its type
only; the key gives the mobjects it refers to beside it, in order, and the
caller keys which mobjects they are (a state's key, by their places in its
walk), as it keys a parent's children.

A value the key cannot key is a *marker* that never equals another key,
its own included, so that every comparison that meets it finds a
difference rather than agreement. There are two: a value of a type the key
does not know (:data:`UNKEYABLE`), and a value holding more parts than one
key walks (:data:`BUDGET` parts, :data:`OVER_BUDGET`).

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
    "UNKEYABLE",
    "marker_reason",
    "unkeyable",
    "value_key",
]

import dataclasses
import enum
import functools
import hashlib
import os
import pathlib
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

#: The head of the marker of a value of a type the key does not know.
UNKEYABLE = b"\0unkeyable "
#: The head of the marker of a value holding more than :data:`BUDGET` parts.
OVER_BUDGET = b"\0over budget "
#: What :func:`unkeyable` names a value over the budget.
OVER_BUDGET_KIND = "over the key budget"


class _Unkeyable(Exception):  # noqa: N818 (a finding, named as one)
    """A value the key cannot key: of a type it does not know, by the type's
    name, or over the budget (:data:`OVER_BUDGET_KIND`).
    """

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


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
    except _Unkeyable as found:
        head = OVER_BUDGET if found.kind == OVER_BUDGET_KIND else UNKEYABLE
        return head + found.kind.encode() + b"\0" + os.urandom(16), ()


def unkeyable(value: Any) -> str | None:
    """What in ``value`` the key cannot key, the first it meets: a type's
    qualified name, or :data:`OVER_BUDGET_KIND`; ``None`` when it keys all
    of ``value``.
    """
    try:
        _key(value, _Keying(None))
    except _Unkeyable as found:
        return found.kind
    return None


def marker_reason(key: bytes) -> str | None:
    """Why ``key`` is a marker, in words, or ``None`` for a key of content."""
    if key.startswith(UNKEYABLE):
        kind = key[len(UNKEYABLE) :].split(b"\0", 1)[0].decode()
        return f"a value of a type the key does not know ({kind})"
    if key.startswith(OVER_BUDGET):
        return f"a value of more than {BUDGET} parts, too large to key"
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
        if decimals is not None and value.dtype.kind in "fc":
            # Adding zero turns -0.0, which rounding makes, into 0.0.
            value = np.round(value, decimals) + 0.0
        return _digest(value.tobytes())
    if isinstance(value, enum.Enum):
        # Its class and its value, as another process names them alike.
        return (
            b"enum "
            + _framed(type(value).__qualname__.encode())
            + _key(value.value, keying)
        )
    if isinstance(value, float) and decimals is not None:
        return repr(round(value, decimals) + 0.0).encode()
    if value is None or value is Ellipsis or isinstance(value, (bool, int, float, str)):
        return repr(value).encode()
    if isinstance(value, Mobject):
        # Which mobject it is, the caller keys; its own state is a node's.
        keying.refers.append(value)
        return b"mobject " + _framed(type(value).__qualname__.encode())
    if isinstance(value, np.generic):
        return (
            b"scalar " + _framed(value.dtype.str.encode()) + _key(value.item(), keying)
        )
    active = keying.active
    if id(value) in active:
        return b"back " + active[::-1].index(id(value)).to_bytes(4, "big")
    active.append(id(value))
    try:
        return _held_key(value, keying)
    finally:
        active.pop()


def _held_key(value: Any, keying: _Keying) -> bytes:
    """The key of a value that holds others: an array, a container, a colour,
    a function or an object of a kind the key knows.
    """
    from arachne.camera.camera import Camera
    from arachne.scene.scene import Scene
    from arachne.utils.color import ManimColor

    def each(parts: Any) -> bytes:
        found = [_key(part, keying) for part in parts]
        return len(found).to_bytes(4, "big") + b"".join(map(_framed, found))

    head = b"of " + _framed(type(value).__qualname__.encode())
    if isinstance(value, np.ndarray):
        array = np.asarray(value).view(np.ndarray)
        shape = _framed(f"{array.dtype.str}{array.shape}".encode())
        if array.dtype.hasobject:
            return head + shape + each(array.ravel().tolist())
        if keying.decimals is not None and array.dtype.kind in "fc":
            array = np.round(array, keying.decimals) + 0.0
        return head + shape + _digest(np.ascontiguousarray(array).tobytes())
    if isinstance(value, ManimColor):
        return head + _key(value.to_rgba(), keying)
    if isinstance(value, (tuple, list)):
        return head + each(value)
    if isinstance(value, (dict, set, frozenset)):
        # Unordered: its parts in the order of their keys, each with the
        # mobjects it refers to.
        if isinstance(value, dict):
            pairs: Any = value.items()
        else:
            pairs = ((part,) for part in value)
        keyed = []
        held = keying.refers
        for pair in pairs:
            keying.refers = []
            part_key = b"".join(_framed(_key(part, keying)) for part in pair)
            keyed.append((part_key, keying.refers))
        keying.refers = held
        keyed.sort(key=lambda item: item[0])
        for _, own in keyed:
            held.extend(own)
        return (
            head
            + len(keyed).to_bytes(4, "big")
            + b"".join(_framed(part_key) for part_key, _ in keyed)
        )
    if isinstance(value, (bytes, bytearray)):
        return head + _digest(bytes(value))
    if isinstance(value, (complex, Fraction, range, slice)):
        return head + _framed(repr(value).encode())
    if isinstance(value, type):
        return b"class " + _framed(value.__qualname__.encode())
    if isinstance(value, types.FunctionType):
        # What it runs and what it closes over, not where it is: a function
        # made twice from one definition keys alike.
        cells = [_contents(cell) for cell in value.__closure__ or ()]
        return (
            head
            + _framed(value.__qualname__.encode())
            + _code_key(value.__code__, keying)
            + each(value.__defaults__ or ())
            + _key(value.__kwdefaults__, keying)
            + each(cells)
        )
    if isinstance(value, types.MethodType):
        return head + each((value.__func__, value.__self__))
    if isinstance(value, np.ufunc):
        return head + _framed(value.__name__.encode())
    if isinstance(value, types.BuiltinFunctionType):
        owner = value.__self__
        bound = () if owner is None or isinstance(owner, types.ModuleType) else (owner,)
        return head + _framed(value.__qualname__.encode()) + each(bound)
    if isinstance(value, functools.partial):
        return head + each((value.func, value.args, value.keywords))
    if isinstance(value, np.vectorize):
        # The function it vectorizes and how; not the ufunc it builds and
        # caches on its first call.
        return head + each(
            (value.pyfunc, value.otypes, value.excluded, value.signature)
        )
    if callable(value) and hasattr(value, "__wrapped__"):
        # A function's cache or another wrapper: the function it wraps.
        return head + _key(value.__wrapped__, keying)
    if isinstance(value, pathlib.PurePath):
        # Where a file was written differs between executions (their media
        # directories); what it holds is keyed where it was read.
        return head + _framed(value.name.encode())
    if isinstance(value, (Scene, Camera, types.ModuleType)):
        # What a scene and a camera do lands in the mobjects and camera
        # inputs a capture keys; a module is code, as the source digest is.
        return head
    if dataclasses.is_dataclass(value):
        return head + each(
            (field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    foreign = _foreign_key(value)
    if foreign is not None:
        return head + _key(foreign, keying)
    if isinstance(getattr(value, "__dict__", None), dict):
        # An object of a plain class (the scene's own helpers, the engine's
        # scales): its type and its attributes.
        return head + each(sorted(vars(value).items()))
    raise _Unkeyable(_qualified(type(value)))


#: What an empty closure cell holds, for its key: a variable the function
#: reads that is not yet bound.
_EMPTY = ("empty cell",)


def _contents(cell: types.CellType) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY


def _code_key(code: types.CodeType, keying: _Keying) -> bytes:
    """The key of a function's code: its instructions, constants and names."""
    consts = [
        _code_key(const, keying)
        if isinstance(const, types.CodeType)
        else _key(const, keying)
        for const in code.co_consts
    ]
    return (
        _framed(code.co_code)
        + _framed(" ".join(code.co_names).encode())
        + len(consts).to_bytes(4, "big")
        + b"".join(map(_framed, consts))
    )


def _foreign_key(value: Any) -> Any:
    """What stands for a value of a library the engine's mobjects hold (an
    image, a parsed path or document, a graph), as values the key knows, or
    ``None`` for another.
    """
    image = sys.modules.get("PIL.Image")
    if image is not None and isinstance(value, image.Image):
        return (value.mode, value.size, value.tobytes())
    svg = sys.modules.get("svgelements")
    if svg is not None and isinstance(value, svg.Path):
        return value.d()
    soup = sys.modules.get("bs4.element")
    if soup is not None and isinstance(value, soup.Tag):
        return str(value)
    networkx = sys.modules.get("networkx")
    if networkx is not None and isinstance(value, networkx.Graph):
        return (
            [list(node) for node in value.nodes(data=True)],
            [list(edge) for edge in value.edges(data=True)],
        )
    return None
