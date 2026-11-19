"""Declared quantities: the numbers and names a picture was built from.

A picture that carries a quantitative claim (a body's speed, a field's value,
which drawing a slot shows) is drawn by the engine from declared quantities,
so that a check reads the value from the session and never measures it from
pixels. There are two reads.

**Summaries**, in ``state_at``. Each mobject on the picture answers a list of
:class:`Quantity`, sorted by name, under its stable id
(:attr:`.State.quantities`). They come through one hook, which the engine
calls on engine-owned code only:

- a **provider**, a function an engine module registers for one of its
  classes (:func:`provider`), serves that class and every subclass of it,
  a scene's included. It is found by the class's MRO in the registry, never
  looked up on the instance or through the class's attributes, so a scene's
  subclass cannot stand in for it;
- a **source**, an immutable object of a registered engine type kept in the
  mobject's reserved attribute ``_quantities`` (:func:`attach`), serves a
  mobject of any class: :class:`Literal` is the engine's general one, a
  declared value that does not change by itself.

Each is handed the mobject's attributes as the capture that answers the
instant holds them, read-only, and an :class:`At`: the instant, its tick and
the record's bakes (:mod:`.timeline.bake`). It reads those and nothing else:
no method of a mobject, no scene code, no file. Registration refuses a
function whose code is not in a file of the engine's package.

A name is ``<prefix>.<rest>``. The prefix has one owner, who registers it
with :func:`register` when its module is imported (this module knows of
none); the rest is one or more dot-separated segments of ASCII letters,
digits, ``_`` and ``-``, which data such as a manifest may define. A value
is a finite number, a vector (a tuple or list) of at most :data:`MAX_VECTOR`
finite numbers, none included (a list of instants with nothing in it yet),
or a name of at most :data:`MAX_NAME_BYTES` bytes from a set its object
declares. A unit is a registered string: ``1`` for a plain number and
``name`` for a name value are always registered; an owner registers its
own, and engine code that loads data registers the units the data declares
(:func:`register_unit`). A tolerance, in the value's unit, is optional and
absent where the value is exact.

A quantity that cannot be read is never dropped and never a number: it is
answered with a null value and the reason (:attr:`Quantity.unreadable`),
and the lint reports ``sim.quantity_unreadable`` for it.

**A value at a point**, through ``query``'s ``value_at`` (:func:`value_at`).
An engine class with a value at every point (a field, a grid) registers a
**point source** (:func:`point_source`): its channels and how one is read
at a position. The read is made at the request's own time, from what the
record owns (the object's attributes at the last capture kept at or before
that time, and the record's bakes by tick): it never restores a play and
never runs the scene. A time before the scene, after it, or outside the
stay of the bake a source reads answers ``outside``, ``time``; a position
outside the object's domain ``outside``, ``domain``; an unknown object or
channel ``unknown``. Each value is dated by the instant it is of.
"""

from __future__ import annotations

__all__ = [
    "MAX_NAME_BYTES",
    "MAX_VECTOR",
    "At",
    "Literal",
    "PointAnswers",
    "PointProbe",
    "PointRead",
    "Provider",
    "Quantity",
    "SOURCES",
    "attach",
    "declares",
    "point_answers",
    "point_source",
    "provider",
    "read",
    "register",
    "register_unit",
    "source",
    "value_at",
]

import bisect
import math
import os
import re
import types
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, TypeVar

from .timeline.clock import TICK_RATE, exact

if TYPE_CHECKING:
    from .mobject.mobject import Mobject
    from .session import Session
    from .timeline.bake import Bakes
    from .timeline.record import Capture, Record

#: The most numbers a vector value holds.
MAX_VECTOR = 16
#: The most bytes, in UTF-8, a name value holds.
MAX_NAME_BYTES = 64

#: Where the engine's files are: a registered function's code is in one.
_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep

#: One segment of a name: a prefix, or a part of the rest.
_SEGMENT = re.compile(r"[A-Za-z0-9_-]+")

#: The attribute of a mobject that holds its attached sources.
SOURCES = "_quantities"


@dataclass(frozen=True, slots=True)
class Quantity:
    """One declared quantity of an object at an instant.

    ``value`` is ``None`` exactly when ``unreadable`` says why it could not
    be read; ``tolerance`` is ``None`` where the value is exact.
    """

    name: str
    value: float | tuple[float, ...] | str | None
    unit: str
    tolerance: float | None = None
    unreadable: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) != (self.unreadable is not None):
            raise ValueError(
                "A quantity has a value or the reason it has none, exactly one:"
                f" {self.name!r} has value {self.value!r} and reason"
                f" {self.unreadable!r}."
            )


@dataclass(frozen=True, slots=True)
class At:
    """The instant a provider reads at: ``as_of``, the instant the installed
    capture is of; ``tick``, the last 60 Hz tick at or before it; and the
    record's bakes, read-only (``None`` for a record that made none).
    """

    as_of: Fraction
    tick: int
    bakes: Bakes | None

    @classmethod
    def of(cls, as_of: Fraction, bakes: Bakes | None) -> At:
        """The reading at ``as_of`` of a record holding ``bakes``."""
        return cls(as_of, math.floor(as_of * TICK_RATE), bakes)


@dataclass(frozen=True, slots=True)
class Literal:
    """The engine's general source: a declared value that does not change by
    itself, attached to any mobject (:func:`attach`). It is read as it is
    held, so a literal replaced between two plays answers each play's own.
    """

    name: str
    value: float | tuple[float, ...] | str
    unit: str
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.unit, str):
            raise TypeError("A literal's name and unit are strings.")
        if isinstance(self.value, list):
            # Nothing mutable sits in a capture.
            object.__setattr__(self, "value", tuple(self.value))


#: A provider: the quantities of a mobject, from its attributes as the
#: installed capture holds them and the instant.
Provider = Callable[[Mapping[str, Any], At], Iterable[Quantity]]
#: How an attached source of one kind is read: ``(source, held, at)``.
SourceReader = Callable[[Any, Mapping[str, Any], At], Iterable[Quantity]]

_P = TypeVar("_P", bound=Callable[..., Any])

# The registry: one owner per prefix, the units, and the engine's functions.
_PREFIXES: dict[str, str] = {}
_UNITS: set[str] = {"1", "name"}
_PROVIDERS: dict[type, Provider] = {}
_SOURCES_READ: dict[type, SourceReader] = {}
_POINT_SOURCES: dict[type, _PointReads] = {}
#: Per class, the provider its MRO finds, or ``None``: found once, and
#: forgotten with the class (a scene's, once its module is let go).
_FOUND: weakref.WeakKeyDictionary[type, Provider | None] = weakref.WeakKeyDictionary()
_FOUND_POINTS: weakref.WeakKeyDictionary[type, _PointReads | None] = (
    weakref.WeakKeyDictionary()
)


def _engine_function(function: Any, role: str) -> None:
    """Refuse ``function`` unless it is a Python function whose code is in a
    file of the engine's package.
    """
    if not isinstance(function, types.FunctionType):
        raise TypeError(f"A {role} is a Python function, not {function!r}.")
    path = os.path.abspath(function.__code__.co_filename)
    if not path.startswith(_ROOT):
        raise ValueError(
            f"A {role} is the engine's own code; {function.__qualname__} is in"
            f" {function.__code__.co_filename}."
        )


def register(prefix: str, *, owner: str, units: Iterable[str] = ()) -> None:
    """Register the name prefix ``prefix`` (``"sim."``: one segment and its
    dot) for ``owner``, and ``units``. A prefix has one owner: registering it
    again for the same owner adds its units, for another raises.
    """
    if not isinstance(prefix, str) or not (
        prefix.endswith(".") and _SEGMENT.fullmatch(prefix[:-1])
    ):
        raise ValueError(
            f"A prefix is one segment of letters, digits, '_' and '-' and its"
            f" dot ('sim.'), not {prefix!r}."
        )
    held = _PREFIXES.setdefault(prefix, owner)
    if held != owner:
        raise ValueError(f"The prefix {prefix!r} is {held}'s, not {owner}'s.")
    for unit in units:
        register_unit(unit)


def register_unit(unit: str) -> None:
    """Register ``unit``: a non-empty string without white space."""
    if not isinstance(unit, str) or not unit or any(c.isspace() for c in unit):
        raise ValueError(f"A unit is a non-empty string without spaces: {unit!r}.")
    _UNITS.add(unit)


def provider(cls: type) -> Callable[[_P], _P]:
    """``@provider(Cls)`` above a module-level function of the engine
    registers it as the provider of ``Cls`` and its subclasses.
    """

    def registering(function: _P) -> _P:
        _engine_function(function, "provider")
        if cls in _PROVIDERS:
            raise ValueError(f"{cls.__qualname__} has a provider already.")
        _PROVIDERS[cls] = function
        _FOUND.clear()
        return function

    return registering


def source(kind: type) -> Callable[[_P], _P]:
    """``@source(Kind)`` above a module-level function of the engine
    registers how an attached source of exactly the type ``Kind`` is read:
    ``read(source, held, at)``.
    """

    def registering(function: _P) -> _P:
        _engine_function(function, "source reader")
        if kind in _SOURCES_READ:
            raise ValueError(f"{kind.__qualname__} is read already.")
        _SOURCES_READ[kind] = function
        return function

    return registering


def attach(mobject: Mobject, *sources: Any) -> None:
    """Add ``sources``, each of a registered kind (:func:`source`), to the
    mobject's reserved attribute ``_quantities``: a tuple, which a capture
    holds like any value.
    """
    for each in sources:
        if type(each) not in _SOURCES_READ:
            raise TypeError(
                f"A {type(each).__qualname__} is no source of declared quantities."
            )
    held = mobject.__dict__.get(SOURCES, ())
    setattr(mobject, SOURCES, (*held, *sources))


def _provider_of(cls: type) -> Provider | None:
    try:
        return _FOUND[cls]
    except KeyError:
        found = next((_PROVIDERS[c] for c in cls.__mro__ if c in _PROVIDERS), None)
        _FOUND[cls] = found
        return found


def declares(cls: type, attached: Any = ()) -> bool:
    """Whether a mobject of ``cls`` whose reserved attribute holds
    ``attached`` has any quantity to read: a provider, or a source.
    """
    return _provider_of(cls) is not None or bool(attached)


def read(cls: type, held: Mapping[str, Any], at: At) -> tuple[Quantity, ...]:
    """The quantities of a mobject of ``cls`` whose attributes, as the
    installed capture holds them, are ``held``, at ``at``: its class's
    provider's, then its attached sources', each checked, sorted by name.
    The engine's one call.
    """
    found = _provider_of(cls)
    attached = held.get(SOURCES, ())
    if found is None and not attached:
        return ()
    view = held if isinstance(held, types.MappingProxyType) else _frozen(held)
    out: list[Quantity] = []
    if found is not None:
        out.extend(found(view, at))
    if isinstance(attached, tuple):
        for each in attached:
            reader = _SOURCES_READ.get(type(each))
            if reader is not None:
                out.extend(reader(each, view, at))
    return _checked(out)


def _frozen(held: Mapping[str, Any]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(held))


def _checked(found: list[Quantity]) -> tuple[Quantity, ...]:
    """``found`` with each entry checked, every one that fails answered by a
    null value and its reason, a name declared twice once, unreadable; sorted
    by name.
    """
    by_name: dict[str, Quantity] = {}
    for quantity in found:
        if not isinstance(quantity, Quantity):
            raise TypeError(f"A provider answers Quantity entries, not {quantity!r}.")
        if not isinstance(quantity.name, str) or not isinstance(quantity.unit, str):
            raise TypeError(f"A quantity's name and unit are strings: {quantity!r}.")
        if quantity.name in by_name:
            by_name[quantity.name] = _unreadable(
                quantity, "declared more than once on this object"
            )
            continue
        by_name[quantity.name] = _check(quantity)
    return tuple(by_name[name] for name in sorted(by_name))


def _unreadable(quantity: Quantity, reason: str) -> Quantity:
    return Quantity(quantity.name, None, quantity.unit, unreadable=reason)


def _check(quantity: Quantity) -> Quantity:
    """``quantity`` with its value in its stored form, or unreadable with why."""
    name, unit, value = quantity.name, quantity.unit, quantity.value
    why = _name_fault(name)
    if why is None and unit not in _UNITS:
        why = f"the unit {unit!r} is not registered"
    if why is not None:
        return _unreadable(quantity, why)
    if value is None:
        return quantity
    if isinstance(value, str):
        if unit != "name":
            return _unreadable(quantity, f"a name value's unit is 'name', not {unit!r}")
        size = len(value.encode())
        if not value or size > MAX_NAME_BYTES:
            return _unreadable(
                quantity, f"a name value holds 1 to {MAX_NAME_BYTES} bytes, not {size}"
            )
        stored: float | tuple[float, ...] | str = value
    else:
        if unit == "name":
            return _unreadable(quantity, "a value with unit 'name' is a name")
        number = _number(value)
        if number is not None:
            if not math.isfinite(number):
                return _unreadable(quantity, f"the value {number!r} is not finite")
            stored = number
        else:
            numbers = _vector(value)
            if numbers is None:
                return _unreadable(
                    quantity,
                    f"the value is not a number, a vector or a name: {value!r}",
                )
            if len(numbers) > MAX_VECTOR:
                return _unreadable(
                    quantity,
                    f"a vector holds at most {MAX_VECTOR} numbers, not {len(numbers)}",
                )
            if not all(map(math.isfinite, numbers)):
                return _unreadable(quantity, "a number of the vector is not finite")
            stored = numbers
    tolerance = quantity.tolerance
    if tolerance is not None:
        bound = _number(tolerance)
        if bound is None or not math.isfinite(bound) or bound < 0:
            return _unreadable(
                quantity, f"the tolerance is a finite number not below 0: {tolerance!r}"
            )
        tolerance = float(bound)
    return Quantity(name, stored, unit, tolerance)


def _name_fault(name: str) -> str | None:
    head, dot, rest = name.partition(".")
    if not dot or not _SEGMENT.fullmatch(head) or not rest:
        return f"the name {name!r} is not <prefix>.<rest>"
    if not all(_SEGMENT.fullmatch(part) for part in rest.split(".")):
        return (
            f"the name {name!r} is not dot-separated segments of letters,"
            " digits, '_' and '-'"
        )
    if f"{head}." not in _PREFIXES:
        return f"the prefix {head + '.'!r} is not registered"
    return None


def _number(value: Any) -> float | int | None:
    """``value`` as a Python number when it is one (an int stays an int)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return None


def _vector(value: Any) -> tuple[float, ...] | None:
    """``value`` as a tuple of floats when it is a tuple or list of numbers."""
    if not isinstance(value, (tuple, list)):
        return None
    numbers = [_number(item) for item in value]
    if any(number is None for number in numbers):
        return None
    return tuple(float(number) for number in numbers if number is not None)


@source(Literal)
def _literal(found: Literal, held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    if found.value is None:
        why = "a literal declares a value, and this one holds none"
        return (Quantity(found.name, None, found.unit, unreadable=why),)
    return (Quantity(found.name, found.value, found.unit, found.tolerance),)


# A value at a point


@dataclass(frozen=True, slots=True)
class PointProbe:
    """One point asked for: the object by stable id, its channel, the
    position and the space it is in (``"scene"``: scene units; ``"grid"``:
    the object's own grid, in cells, fractions allowed).
    """

    id: str
    channel: str
    position: tuple[float, ...]
    space: str = "scene"


@dataclass(frozen=True, slots=True)
class PointRead:
    """A point's answer. ``result`` is ``"value"`` (with ``value``, ``unit``,
    the ``interpolation`` the object declares, the grid ``cells`` read and
    ``as_of``, the instant the value is of), ``"outside"`` (``why``:
    ``"domain"`` or ``"time"``) or ``"unknown"`` (``why``: ``"id"``,
    ``"channel"`` or ``"not_point_readable"``).
    """

    result: str
    value: float | tuple[float, ...] | None = None
    unit: str | None = None
    interpolation: str | None = None
    cells: tuple[tuple[int, ...], ...] = ()
    as_of: Fraction | None = None
    why: str | None = None


#: What each result's ``why`` may be.
_WHYS = {
    "value": (None,),
    "outside": ("domain", "time"),
    "unknown": ("id", "channel", "not_point_readable"),
}


@dataclass(frozen=True, slots=True)
class _PointReads:
    channels: Callable[[Mapping[str, Any]], Mapping[str, str]]
    value_at: Callable[[Mapping[str, Any], At, str, tuple[float, ...], str], PointRead]


def point_source(
    cls: type,
    *,
    channels: Callable[[Mapping[str, Any]], Mapping[str, str]],
    value_at: Callable[[Mapping[str, Any], At, str, tuple[float, ...], str], PointRead],
) -> None:
    """Register, for the engine class ``cls`` and its subclasses, how a value
    at a point is read: ``channels(held)``, each channel's unit by name, and
    ``value_at(held, at, channel, position, space)``, a :class:`PointRead`.
    Both read what they are handed, as a provider does.
    """
    _engine_function(channels, "point source's channels")
    _engine_function(value_at, "point source's value_at")
    if cls in _POINT_SOURCES:
        raise ValueError(f"{cls.__qualname__} has a point source already.")
    _POINT_SOURCES[cls] = _PointReads(channels, value_at)
    _FOUND_POINTS.clear()


def _point_source_of(cls: type) -> _PointReads | None:
    try:
        return _FOUND_POINTS[cls]
    except KeyError:
        found = next(
            (_POINT_SOURCES[c] for c in cls.__mro__ if c in _POINT_SOURCES), None
        )
        _FOUND_POINTS[cls] = found
        return found


def value_at(
    record: Record, t: Fraction | float, probes: Sequence[PointProbe]
) -> tuple[PointRead, ...]:
    """Each of ``probes`` read at ``t`` exactly, in order: from the object's
    attributes at the last capture ``record`` kept at or before ``t``, and the
    record's bakes at the last tick at or before it. Nothing is restored and
    nothing of the scene runs.
    """
    t = exact(t)
    if t < 0 or t > record.duration:
        return tuple(PointRead("outside", why="time") for _ in probes)
    held = _held_at(record, _kept_capture(record, t), {probe.id for probe in probes})
    at = At.of(t, record.bakes)
    return tuple(_read_point(held, at, probe) for probe in probes)


def _kept_capture(record: Record, t: Fraction) -> Capture:
    """The last capture ``record`` kept at or before ``t``."""
    captures = record._captures
    index = bisect.bisect_right(captures, t, key=lambda capture: capture.instant) - 1
    assert index >= 0, "a record keeps a capture at the scene's start"
    return captures[index]


def _held_at(
    record: Record, capture: Capture, ids: set[str]
) -> dict[str, tuple[type, Mapping[str, Any]]]:
    """Each mobject of ``ids`` (stable ids) that ``capture`` reaches (the
    first of an id), with its class and its attributes there.
    """
    found: dict[str, tuple[type, Mapping[str, Any]]] = {}
    roots = [*capture.roots, *capture.foreground, *capture.moving]
    for mob, state in record._walk(capture, roots):
        node_id = record.origin(mob).id
        if node_id in ids and node_id not in found:
            attributes = dict(zip(state.keys, state.values, strict=True))
            found[node_id] = (type(mob), types.MappingProxyType(attributes))
            if len(found) == len(ids):
                break
    return found


def _read_point(
    held: Mapping[str, tuple[type, Mapping[str, Any]]], at: At, probe: PointProbe
) -> PointRead:
    entry = held.get(probe.id)
    if entry is None:
        return PointRead("unknown", why="id")
    cls, attributes = entry
    reads = _point_source_of(cls)
    if reads is None:
        return PointRead("unknown", why="not_point_readable")
    units = reads.channels(attributes)
    if probe.channel not in units:
        return PointRead("unknown", why="channel")
    found = reads.value_at(attributes, at, probe.channel, probe.position, probe.space)
    return _point_checked(found, units[probe.channel], at.as_of)


def _point_checked(found: PointRead, unit: str, as_of: Fraction) -> PointRead:
    """``found`` checked, its unit and instant filled in for a value."""
    if not isinstance(found, PointRead) or found.why not in _WHYS.get(found.result, ()):
        raise ValueError(f"A point source answered {found!r}.")
    if found.result != "value":
        return PointRead(found.result, why=found.why)
    number = _number(found.value)
    stored: float | tuple[float, ...] | None
    if number is not None:
        stored = float(number)
        finite = math.isfinite(stored)
    else:
        stored = _vector(found.value)
        finite = stored is not None and 1 <= len(stored) <= MAX_VECTOR
        finite = finite and all(map(math.isfinite, stored or ()))
    if not finite or found.interpolation not in ("nearest", "linear"):
        raise ValueError(f"A point source answered {found!r}.")
    return replace(
        found,
        value=stored,
        unit=found.unit if found.unit is not None else unit,
        as_of=as_of,
    )


@dataclass(frozen=True)
class PointAnswers:
    """A query's ``value_at``: the request's time, the play holding it, the
    picture's box (as the record's configuration gives it, in home-frame
    units), and each probe with its answer, in order.
    """

    t: Fraction
    play: int | None
    frame: tuple[float, float, float, float]
    reads: tuple[tuple[PointProbe, PointRead], ...]


def point_answers(
    session: Session, t: Fraction | float, probes: Sequence[PointProbe]
) -> PointAnswers:
    """``probes`` read at ``t`` in ``session``'s record (:func:`value_at`)."""
    record = session.record
    instant = exact(t)
    play = record.play_at(instant)
    width, height = record._scene._arachne_timeline.home
    reads = value_at(record, instant, probes)
    return PointAnswers(
        instant,
        play.index if play is not None else None,
        (-width / 2, -height / 2, width / 2, height / 2),
        tuple(zip(probes, reads, strict=True)),
    )
