"""Content keys of the values a mobject's fields hold.

:func:`value_key` keys every value a field can hold: equal for two values
that hold the same content, in this process or another, and apart for two
that do not. A mobject a value refers to stands in the key as its type
only; the key gives the mobjects it refers to beside it, in order, and the
caller keys which mobjects they are (a state's key, by their places in its
walk), as it keys a parent's children.

A value of a type the key does not know is *unkeyable*: it keys as a
marker that never equals another key, its own included, so that every
comparison that meets it finds a difference rather than agreement.

A class (an enum member's included) and a function are named by their
qualified names, not their modules': a scene's module is named for the load
that made it (each session imports the file under a name of its own, a
render by its path). A function is keyed by its code and what it closes
over, not by the globals it reads, which the source digest covers.

Containers and objects are keyed by what they hold when the key is made: a
record keeps a field's value by reference, so a container changed in place
after a capture keys as it is now (as the record's own change detection,
by identity, sees no change either).
"""

from __future__ import annotations

__all__ = ["unkeyable", "value_key"]

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


class _Unkeyable(Exception):  # noqa: N818 (a finding, named as one)
    """A value of a type the key does not know, by its type's name."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def value_key(
    value: Any, decimals: int | None = None
) -> tuple[bytes, tuple[Mobject, ...]]:
    """``value``'s content key, its floats rounded to ``decimals`` when
    given, and the mobjects it refers to in the order they stand in it; an
    unkeyable value's is a marker equal to no other key (and refers to
    none).
    """
    refers: list[Mobject] = []
    try:
        return _key(value, decimals, refers, []), tuple(refers)
    except _Unkeyable as found:
        return b"\0unkeyable " + found.kind.encode() + os.urandom(16), ()


def unkeyable(value: Any) -> str | None:
    """The qualified name of the type in ``value`` the key does not know,
    the first it meets, or ``None`` when it keys all of ``value``.
    """
    try:
        _key(value, None, [], [])
    except _Unkeyable as found:
        return found.kind
    return None


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _framed(data: bytes) -> bytes:
    """``data`` with its length before it, so that two runs of parts key
    apart wherever they split.
    """
    return len(data).to_bytes(4, "big") + data


def _key(value: Any, decimals: int | None, refers: list, active: list[int]) -> bytes:
    """``value``'s key, the mobjects it refers to appended to ``refers``;
    ``active`` holds the ids of the values being keyed around it, so that a
    value holding itself keys as a reference back.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import FrozenArray, Paint

    if isinstance(value, (Paint, Placement)):
        return value.key
    if isinstance(value, FrozenArray):
        if decimals is not None and value.dtype.kind in "fc":
            # Adding zero turns -0.0, which rounding makes, into 0.0.
            value = np.round(value, decimals) + 0.0
        return hashlib.blake2b(value.tobytes(), digest_size=16).digest()
    if isinstance(value, enum.Enum):
        # Its class and its value, as another process names them alike.
        return (
            b"enum "
            + _framed(type(value).__qualname__.encode())
            + _key(value.value, decimals, refers, active)
        )
    if isinstance(value, float) and decimals is not None:
        return repr(round(value, decimals) + 0.0).encode()
    if value is None or value is Ellipsis or isinstance(value, (bool, int, float, str)):
        return repr(value).encode()
    if isinstance(value, Mobject):
        # Which mobject it is, the caller keys; its own state is a node's.
        refers.append(value)
        return b"mobject " + _framed(type(value).__qualname__.encode())
    if isinstance(value, np.generic):
        return (
            b"scalar "
            + _framed(value.dtype.str.encode())
            + _key(value.item(), decimals, refers, active)
        )
    if id(value) in active:
        return b"back " + active[::-1].index(id(value)).to_bytes(4, "big")
    active.append(id(value))
    try:
        return _held_key(value, decimals, refers, active)
    finally:
        active.pop()


def _held_key(
    value: Any, decimals: int | None, refers: list, active: list[int]
) -> bytes:
    """The key of a value that holds others: an array, a container, a colour,
    a function or an object of a kind the key knows.
    """
    from arachne.camera.camera import Camera
    from arachne.mobject.graphing.scale import _ScaleBase
    from arachne.utils.color import ManimColor

    def each(parts: Any) -> bytes:
        found = [_key(part, decimals, refers, active) for part in parts]
        return len(found).to_bytes(4, "big") + b"".join(map(_framed, found))

    head = b"of " + _framed(type(value).__qualname__.encode())
    if isinstance(value, np.ndarray):
        array = np.asarray(value).view(np.ndarray)
        shape = _framed(f"{array.dtype.str}{array.shape}".encode())
        if array.dtype.hasobject:
            return head + shape + each(array.ravel().tolist())
        if decimals is not None and array.dtype.kind in "fc":
            array = np.round(array, decimals) + 0.0
        return head + shape + _framed(np.ascontiguousarray(array).tobytes())
    if isinstance(value, ManimColor):
        return head + _key(value.to_rgba(), decimals, refers, active)
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
        for pair in pairs:
            own: list[Mobject] = []
            part_key = b"".join(
                _framed(_key(part, decimals, own, active)) for part in pair
            )
            keyed.append((part_key, own))
        keyed.sort(key=lambda item: item[0])
        for _, own in keyed:
            refers.extend(own)
        return (
            head
            + len(keyed).to_bytes(4, "big")
            + b"".join(_framed(part_key) for part_key, _ in keyed)
        )
    if isinstance(value, (bytes, bytearray)):
        return head + _framed(bytes(value))
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
            + _code_key(value.__code__, decimals, refers, active)
            + each(value.__defaults__ or ())
            + _key(value.__kwdefaults__, decimals, refers, active)
            + each(cells)
        )
    if isinstance(value, types.MethodType):
        return head + each((value.__func__, value.__self__))
    if isinstance(value, np.ufunc):
        return head + _framed(value.__name__.encode())
    if isinstance(value, types.BuiltinFunctionType):
        owner = value.__self__
        bound = () if owner is None or isinstance(owner, types.ModuleType) else (owner,)
        name = value.__qualname__
        return head + _framed(name.encode()) + each(bound)
    if isinstance(value, functools.partial):
        return head + each((value.func, value.args, value.keywords))
    if isinstance(value, pathlib.PurePath):
        # Where a file was written differs between executions (their media
        # directories); what it holds is keyed where it was read.
        return head + _framed(value.name.encode())
    if isinstance(value, Camera):
        # A camera's inputs are its drawing's, keyed with the capture.
        return head
    if isinstance(value, _ScaleBase):
        return head + each(sorted(vars(value).items()))
    if dataclasses.is_dataclass(value):
        return head + each(
            (field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    foreign = _foreign_key(value)
    if foreign is not None:
        return head + _key(foreign, decimals, refers, active)
    raise _Unkeyable(_qualified(type(value)))


#: What an empty closure cell holds, for its key: a variable the function
#: reads that is not yet bound.
_EMPTY = ("empty cell",)


def _contents(cell: types.CellType) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY


def _code_key(
    code: types.CodeType, decimals: int | None, refers: list, active: list[int]
) -> bytes:
    """The key of a function's code: its instructions, constants and names."""
    consts = [
        _code_key(const, decimals, refers, active)
        if isinstance(const, types.CodeType)
        else _key(const, decimals, refers, active)
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
