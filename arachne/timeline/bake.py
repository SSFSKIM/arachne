"""Bakes: what a simulation computed, kept in the record by 60 Hz tick.

One container serves every simulation. It has two parts.

A :class:`BakeContent` is immutable and holds nothing of a scene: no mobject,
no stable id, no instant. It is the identity of what computed it
(:class:`BakeIdentity`), its row count, its **channels** (per row, an array
of one shape, one type and one unit) and its **event tables** (rows of
unsigned integers, the first column a row index). It is keyed by the
BLAKE2b-128 digest of its canonical bytes (:meth:`BakeContent.canonical_bytes`),
so two equal contents are one, wherever and whenever they were computed.

An :class:`Occurrence` is a record's use of one content: one uninterrupted
stay of what made it in the scene, from its first tick to its last (row
``i`` is tick ``first_tick + i``, shown at every instant of
``[(first_tick + i)/60, (first_tick + i + 1)/60)``), which mobject each row
of a channel's first axis is about (a channel may be about none), what was
found, and the digest of the declaration it was computed from. Occurrences
are numbered by the order their stays began in the execution.

:class:`Bakes` is the record's (``Record.bakes``): its contents by key and
its occurrences by ordinal. A producer reserves an ordinal as a stay begins
(:meth:`Bakes.begin`), holding whatever computes it until the stay is sealed
(:meth:`Bakes.seal`), when it is dropped. A read by tick (:meth:`Bakes.row`)
or by instant (:meth:`Bakes.at`, the last tick at or before it) answers a
sealed occurrence's row, or ``None`` outside its stay.
"""

from __future__ import annotations

__all__ = [
    "LAYOUT_VERSION",
    "BakeContent",
    "BakeFinding",
    "BakeIdentity",
    "BakeRef",
    "Bakes",
    "Binding",
    "Channel",
    "EventTable",
    "Occurrence",
    "Row",
    "bakes_of",
]

import hashlib
import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import TICK_RATE

if TYPE_CHECKING:
    from .record import Record

#: The version of the canonical bytes' layout.
LAYOUT_VERSION = 1
#: Each channel type and its code in the canonical bytes.
DTYPES: Mapping[str, tuple[int, np.dtype]] = MappingProxyType(
    {
        "f32": (1, np.dtype("<f4")),
        "f64": (2, np.dtype("<f8")),
        "i32": (3, np.dtype("<i4")),
        "u32": (4, np.dtype("<u4")),
    }
)
#: The one quiet NaN each float type stores a non-finite value as.
_NAN_BITS = {
    "f32": (np.dtype("<u4"), 0x7FC00000),
    "f64": (np.dtype("<u8"), 0x7FF8000000000000),
}


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class BakeIdentity:
    """What computed a content: the producer and its version, the solver and
    its version, the integrator, the build (toolchain, target and target
    features) and the features that change numbers, in any order (the
    canonical bytes sort them).
    """

    producer: str
    producer_version: str
    solver: str
    solver_version: str
    integrator: str
    build: str
    features: tuple[str, ...] = ()

    def strings(self) -> tuple[str, ...]:
        """The seven strings of the canonical bytes, in order."""
        return (
            self.producer,
            self.producer_version,
            self.solver,
            self.solver_version,
            self.integrator,
            self.build,
            ",".join(sorted(self.features)),
        )


@dataclass(frozen=True)
class Channel:
    """Per row, an array of one shape: ``values`` is ``[rows, *shape]`` of
    the type ``dtype`` (one of :data:`DTYPES`), little-endian and read-only,
    every non-finite value the type's one quiet NaN.
    """

    dtype: str
    unit: str
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"A channel's type is one of {', '.join(DTYPES)}.")
        if not isinstance(self.unit, str):
            raise TypeError("A channel's unit is a string.")
        stored = np.array(self.values, dtype=DTYPES[self.dtype][1], order="C")
        if stored.ndim < 1:
            raise ValueError("A channel's values have a row axis first.")
        if self.dtype in _NAN_BITS:
            bits, nan = _NAN_BITS[self.dtype]
            stored.view(bits)[~np.isfinite(stored)] = nan
        object.__setattr__(self, "values", _read_only(stored))

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one row."""
        return tuple(self.values.shape[1:])


@dataclass(frozen=True)
class EventTable:
    """Rows of ``columns`` unsigned 32-bit integers, the first a row index,
    sorted lexicographically, read-only.
    """

    rows: np.ndarray

    def __post_init__(self) -> None:
        given = np.asarray(self.rows)
        if given.ndim != 2 or given.shape[1] < 1:
            raise ValueError("An event table is rows of one or more columns.")
        if given.size and (given.min() < 0 or given.max() > 0xFFFFFFFF):
            raise ValueError("An event table holds unsigned 32-bit integers.")
        stored = np.array(given, dtype="<u4", order="C")
        if len(stored):
            stored = stored[np.lexsort(stored.T[::-1])]
        object.__setattr__(self, "rows", _read_only(np.ascontiguousarray(stored)))

    @property
    def columns(self) -> int:
        return int(self.rows.shape[1])


class BakeContent:
    """One simulation's result, immutable, keyed by the digest of its
    canonical bytes; holding nothing of a scene.
    """

    __slots__ = ("channels", "events", "identity", "key", "ticks")

    def __init__(
        self,
        identity: BakeIdentity,
        ticks: int,
        channels: Mapping[str, Channel],
        events: Mapping[str, EventTable] | None = None,
    ) -> None:
        if not isinstance(ticks, int) or ticks < 1:
            raise ValueError("A content holds one row or more.")
        tables = dict(events or {})
        for name, channel in channels.items():
            if channel.values.shape[0] != ticks:
                raise ValueError(
                    f"Channel {name!r} holds {channel.values.shape[0]} rows, not {ticks}."
                )
        for name, table in tables.items():
            if len(table.rows) and int(table.rows[:, 0].max()) >= ticks:
                raise ValueError(f"Event table {name!r} names a row past the last.")
        self.identity = identity
        self.ticks = ticks
        self.channels: Mapping[str, Channel] = MappingProxyType(
            {name: channels[name] for name in sorted(channels, key=str.encode)}
        )
        self.events: Mapping[str, EventTable] = MappingProxyType(
            {name: tables[name] for name in sorted(tables, key=str.encode)}
        )
        digest = hashlib.blake2b(digest_size=16)
        for chunk in self._chunks():
            digest.update(chunk)
        self.key: bytes = digest.digest()

    def canonical_bytes(self) -> bytes:
        """The content's canonical bytes, integers little-endian: ``ARBK``; the
        layout version; the identity's seven strings; the row count; the
        channels in bytewise order of name (name, type code, rank and each
        dimension of a row, unit, values in row-major order with the row axis
        first); the event tables in bytewise order of name (name, column
        count, row count, rows). A string is its UTF-8 length and bytes.
        """
        return b"".join(self._chunks())

    def _chunks(self) -> Iterator[bytes]:
        yield b"ARBK" + _u32(LAYOUT_VERSION)
        for text in self.identity.strings():
            yield _text(text)
        yield _u32(self.ticks) + _u32(len(self.channels))
        for name, channel in self.channels.items():
            shape = channel.shape
            yield (
                _text(name)
                + struct.pack("<BB", DTYPES[channel.dtype][0], len(shape))
                + b"".join(_u32(dim) for dim in shape)
                + _text(channel.unit)
            )
            yield channel.values.tobytes()
        yield _u32(len(self.events))
        for name, table in self.events.items():
            yield _text(name) + struct.pack("<B", table.columns) + _u32(len(table.rows))
            yield table.rows.tobytes()

    def __repr__(self) -> str:
        return f"BakeContent({self.key.hex()}, ticks={self.ticks})"


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _text(text: str) -> bytes:
    data = text.encode()
    return _u32(len(data)) + data


@dataclass(frozen=True, slots=True)
class BakeRef:
    """What a mobject holds of a stay: its occurrence's ordinal."""

    ordinal: int


@dataclass(frozen=True, slots=True)
class Binding:
    """Which mobject (by stable id) a channel's row (``index`` on its first
    axis), or the whole channel (``index`` ``None``), is about.
    """

    id: str
    channel: str
    index: int | None = None


@dataclass(frozen=True)
class BakeFinding:
    """What was found at a row of a stay: a lint code, the row, the mobjects
    it is about by stable id, and the code's detail fields.
    """

    code: str
    row: int
    ids: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", tuple(self.ids))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@dataclass(frozen=True)
class Occurrence:
    """A record's use of one content: the stay from ``first_tick`` to
    ``last_tick`` (row ``i`` is tick ``first_tick + i``), whether its world
    ran no scene code as it stepped (``closed``), the digest of the
    declaration it was computed from, its bindings and its findings.
    """

    ordinal: int
    content: bytes
    first_tick: int
    last_tick: int
    closed: bool
    declaration: bytes
    bindings: tuple[Binding, ...] = ()
    findings: tuple[BakeFinding, ...] = ()

    def index(self, tick: int) -> int | None:
        """The row of ``tick``, or ``None`` outside the stay."""
        if self.first_tick <= tick <= self.last_tick:
            return tick - self.first_tick
        return None


@dataclass(frozen=True)
class Row:
    """One row of an occurrence: its index in the stay, its tick, and each
    channel's array there, read-only.
    """

    index: int
    tick: int
    values: Mapping[str, np.ndarray]

    def __getitem__(self, channel: str) -> np.ndarray:
        return self.values[channel]


class Bakes:
    """A record's bakes: contents by key, occurrences by ordinal."""

    def __init__(self) -> None:
        self._contents: dict[bytes, BakeContent] = {}
        self._occurrences: dict[int, Occurrence] = {}
        #: What computes each stay not yet sealed, by ordinal.
        self._live: dict[int, Any] = {}
        self._next = 0

    @property
    def contents(self) -> Mapping[bytes, BakeContent]:
        """Every content an occurrence uses, by key."""
        return MappingProxyType(self._contents)

    @property
    def occurrences(self) -> tuple[Occurrence, ...]:
        """The sealed occurrences, by ordinal."""
        return tuple(self._occurrences[n] for n in sorted(self._occurrences))

    def begin(self, live: Any = None) -> int:
        """Reserve the next ordinal for a stay that begins now, holding
        ``live`` (what computes it) until it is sealed.
        """
        ordinal = self._next
        self._next += 1
        self._live[ordinal] = live
        return ordinal

    def live(self, ordinal: int) -> Any:
        """What computes the stay ``ordinal`` while it is not sealed."""
        return self._live[ordinal]

    def seal(
        self,
        ordinal: int,
        content: BakeContent,
        *,
        first_tick: int,
        closed: bool,
        declaration: bytes,
        bindings: tuple[Binding, ...] = (),
        findings: tuple[BakeFinding, ...] = (),
    ) -> Occurrence:
        """Fix the stay ``ordinal`` as ``content`` from ``first_tick``, and
        drop what computed it. A content equal to one held is that one.
        """
        if ordinal not in self._live:
            raise KeyError(f"No stay {ordinal} is open.")
        if not isinstance(first_tick, int) or first_tick < 0:
            raise ValueError("A stay's first tick is a tick of the scene: 0 or more.")
        for binding in bindings:
            channel = content.channels.get(binding.channel)
            if channel is None:
                raise ValueError(f"A binding names no channel {binding.channel!r}.")
            rows = channel.shape[0] if channel.shape else 0
            if binding.index is not None and not 0 <= binding.index < rows:
                raise ValueError(f"A binding's index is past {binding.channel!r}.")
        for finding in findings:
            if not 0 <= finding.row < content.ticks:
                raise ValueError(f"A finding's row {finding.row} is outside the stay.")
        content = self._contents.setdefault(content.key, content)
        occurrence = Occurrence(
            ordinal,
            content.key,
            first_tick,
            first_tick + content.ticks - 1,
            closed,
            declaration,
            tuple(bindings),
            tuple(findings),
        )
        del self._live[ordinal]
        self._occurrences[ordinal] = occurrence
        return occurrence

    def occurrence(self, ordinal: int) -> Occurrence | None:
        """The sealed occurrence ``ordinal``, or ``None``."""
        return self._occurrences.get(ordinal)

    def content_of(self, ordinal: int) -> BakeContent | None:
        """The content of the sealed occurrence ``ordinal``, or ``None``."""
        occurrence = self._occurrences.get(ordinal)
        return None if occurrence is None else self._contents[occurrence.content]

    def row(self, ordinal: int, tick: int) -> Row | None:
        """The sealed occurrence ``ordinal``'s row at ``tick``, or ``None``
        outside its stay (or for an ordinal not sealed).
        """
        occurrence = self._occurrences.get(ordinal)
        if occurrence is None:
            return None
        index = occurrence.index(tick)
        if index is None:
            return None
        content = self._contents[occurrence.content]
        return Row(
            index,
            tick,
            MappingProxyType(
                {
                    name: channel.values[index]
                    for name, channel in content.channels.items()
                }
            ),
        )

    def at(self, ordinal: int, t: Fraction) -> Row | None:
        """The row shown at the instant ``t``: its last tick at or before."""
        return self.row(ordinal, math.floor(t * TICK_RATE))


def bakes_of(record: Record) -> Bakes:
    """``record``'s bakes, made empty when it has none yet."""
    if record.bakes is None:
        record.bakes = Bakes()
    return record.bakes
