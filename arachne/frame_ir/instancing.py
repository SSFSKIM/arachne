"""Instancing: runs of like leaves sent to a core drawer as ``instances``
records.

A run is a stretch of the draw order, nothing else drawn inside it, whose
leaves repeat a group of ``k`` leaves (``k`` from 1 to :data:`MOST_PARTS`;
an arrow is a shaft and a tip) and agree, place by place in the group, on
everything but where they are and their fill and stroke colours: one
prototype shape, one paint block. It becomes one ``instances`` record at the
run's place in the order. Member ``i`` draws prototype ``i % k`` (XF 1.6's
``parts`` beyond the first), at its own placement, in its own colours, and
ties as the run's ``i``-th leaf did, so the core draws the bytes the run's
leaves draw as records of the same shapes and placements
(``core/tests/instances.rs``).

Leaves of one shape (one content key) batch as they are: the prototype is
their shape, their placements the identity. Others are **factored**: CE
builds each arrow of a field from scratch, so arrows of different lengths
and directions share no key. A small path's points ``P`` are written as
``o + M C``: ``o`` its first point, ``M`` a frame taken from its own points
(:func:`factor`), ``C`` its canonical shape. Leaves whose canonical shapes,
snapped to the frame's input grid, are one shape batch with that shape for
prototype and ``[M | o]`` for placement. A leaf is factored only when its
canonical shape mapped back reproduces its points within
:data:`RECONSTRUCTION` scene units; the snap then moves the batched points as
it moves any (:mod:`.canonical`), so a factored run draws within the parity
tolerance of its leaves, not byte for byte, and a run is factored only in a
family of at least :data:`FEWEST_FACTORED_GROUPS` groups: two squares of
different sizes stay records of their own.

Only a core drawer reads ``instances`` records: :class:`Instancing` turns
the deltas a :class:`~.extract.FrameStream` made into the batched deltas the
core is handed, and a stream for Cairo (``ir_cairo``, the overlay over the
core's frame) keeps the leaves' own records, so Cairo's frames are untouched.

Nothing here imports the core.
"""

from __future__ import annotations

__all__ = [
    "MOST_FACTORED_POINTS",
    "FEWEST_FACTORED_GROUPS",
    "MOST_PARTS",
    "RECONSTRUCTION",
    "Factored",
    "Instancing",
    "expand",
    "factor",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .xf import Delta, PaintTable, Record, Shape, StreamState

#: The longest group of leaves a run repeats.
MOST_PARTS = 4
#: The most points a path is factored with: a small path's (an arrow's shaft
#: has 4, its tip 12, a dot 32).
MOST_FACTORED_POINTS = 256

#: The fewest groups a run batches in when its shapes were factored: its
#: members then draw within the parity tolerance of their own records, not
#: byte for byte, which only a family of leaves large enough to be drawn
#: faster batched is worth.
FEWEST_FACTORED_GROUPS = 8
#: How far, in scene units, a factored leaf's canonical shape mapped back may
#: stand from its own points.
RECONSTRUCTION = 1e-9
#: Where the slots of instances records start: above every slot a stream
#: gives a leaf.
SLOT_BASE = 1 << 31
#: How far, relative to a path's extent, a point must stand from the line or
#: plane of the frame's earlier axes to give it another axis.
_RANK = 1e-6

#: The record fields a leaf may carry and still batch: a plain path's (CE's
#: own), unshaded, not facing the camera, without a term.
_BATCHABLE = frozenset(
    {
        "shape",
        "fill",
        "stroke",
        "background_stroke",
        "stroke_width",
        "background_stroke_width",
        "joint",
        "cap",
        "space",
        "z_index",
        "shade_in_3d",
        "source",
    }
)
#: The paint block a prototype shares with its instances, by field name,
#: beyond its paints: each instance strokes at a width of its own (CE thins
#: a short arrow's stroke by its length).
_BLOCK = ("background_stroke_width", "joint", "cap")


@dataclass(frozen=True)
class Factored:
    """A path's points as ``place[:, :3] @ shape + place[:, 3]``: its
    canonical ``shape`` (``(n, 3)``) and the affine ``place`` (``(3, 4)``).
    """

    shape: np.ndarray
    place: np.ndarray


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _norm(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _first(rows: np.ndarray, size: np.ndarray, bound: float) -> int | None:
    """The first row whose ``size`` exceeds ``bound``."""
    found = np.flatnonzero(size > bound)
    return int(found[0]) if len(found) else None


def factor(points: Any) -> Factored | None:
    """``points`` (``(n, 3)``) factored into a canonical shape and a
    placement, or ``None``: too many points, a value not finite, or a
    reconstruction off by more than :data:`RECONSTRUCTION`.

    The frame is taken from the points in their order, so every affine
    image of one path gets one canonical shape: its origin the first point;
    its first axis the first point apart from it; its second the first point
    off that axis's line, its third the first off their plane, each by more
    than :data:`_RANK` of the path's extent. An axis a path does not span
    (a line spans one, a flat shape two) is made perpendicular to those it
    spans, as long as the first, and the canonical shape is 0 along it.
    Elementwise IEEE arithmetic alone, no reduction a vector unit or a BLAS
    kernel orders, so the factoring is the same on every machine.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not 0 < len(p) <= MOST_FACTORED_POINTS:
        return None
    if not np.isfinite(p).all():
        return None
    origin = p[0].copy()
    d = p - origin
    size = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    extent = float(size.max())
    if extent == 0.0:
        # A path at one point: placed there, its shape at the origin.
        place = np.zeros((3, 4))
        place[:, :3] = np.eye(3)
        place[:, 3] = origin
        return Factored(np.zeros_like(p), place)
    i = _first(d, size, _RANK * extent)
    assert i is not None
    u = d[i]
    across = np.array([_norm(_cross(u, row)) for row in d])
    j = _first(d, across, _RANK * _norm(u) * extent)
    if j is None:
        # A line: any perpendicular, as long as u.
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(u)))] = 1.0
        v = _cross(u, axis)
        v = v * (_norm(u) / _norm(v))
    else:
        v = d[j]
    n = _cross(u, v)
    beyond = np.abs(d[:, 0] * n[0] + d[:, 1] * n[1] + d[:, 2] * n[2])
    k = _first(d, beyond, _RANK * _norm(n) * extent)
    w = n * (_norm(u) / _norm(n)) if k is None else d[k]
    m = np.column_stack([u, v, w])
    # m's inverse by its adjugate, entry by entry.
    cof = np.array(
        [
            [
                m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
                m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
                m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
            ],
            [
                m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2],
                m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
                m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
            ],
            [
                m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],
                m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
                m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
            ],
        ]
    )
    det = m[0, 0] * cof[0, 0] + m[0, 1] * cof[1, 0] + m[0, 2] * cof[2, 0]
    if not (det != 0.0 and np.isfinite(det)):
        return None
    inverse = cof / det
    shape = _apply(inverse, d)
    if k is None:
        shape[:, 2] = 0.0
    if j is None:
        shape[:, 1] = 0.0
    back = _apply(m, shape) + origin
    if not float(np.abs(back - p).max()) <= RECONSTRUCTION:
        return None
    place = np.zeros((3, 4))
    place[:, :3] = m
    place[:, 3] = origin
    return Factored(shape, place)


def _apply(m: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``m @ row`` for each row, as the core places a point: each output the
    sum of its three products in order.
    """
    return np.column_stack(
        [
            m[r, 0] * rows[:, 0] + m[r, 1] * rows[:, 1] + m[r, 2] * rows[:, 2]
            for r in range(3)
        ]
    )


def _snap(values: np.ndarray, grid: float) -> np.ndarray:
    return np.round(values / grid) * grid + 0.0


def _one(value: Any) -> np.ndarray | None:
    """A paint's one stop, or ``None`` for no paint; a paint of more stops is
    not batched (``ValueError``).
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        raise ValueError("a paint table")
    rows = np.asarray(value, dtype=float).reshape(-1, 4)
    if len(rows) != 1:
        raise ValueError("a gradient")
    return rows[0]


@dataclass(frozen=True)
class _Leaf:
    """What a batchable leaf record shares with its run (``block``) and what
    is its own (its colours and its stroke width), and its shape's key.
    """

    shape: bytes
    block: tuple[Any, ...]
    fill: np.ndarray | None
    stroke: np.ndarray | None
    width: float


def _leaf(record: Record) -> _Leaf | None:
    """``record`` as a run's member, or ``None`` where it cannot be one."""
    fields = record.fields
    if record.kind != "path" or not _BATCHABLE.issuperset(fields):
        return None
    if fields.get("shade_in_3d") or fields.get("space", "screen") == "overlay":
        return None
    shape = fields.get("shape")
    if not isinstance(shape, bytes):
        return None
    try:
        fill = _one(fields.get("fill"))
        stroke = _one(fields.get("stroke"))
        background = _one(fields.get("background_stroke"))
    except ValueError:
        return None
    block = (
        *(fields.get(name) for name in _BLOCK),
        fields.get("space", "screen"),
        float(fields.get("z_index", 0.0)),
        fill is None,
        stroke is None,
        None if background is None else background.tobytes(),
    )
    return _Leaf(shape, block, fill, stroke, float(fields.get("stroke_width", 0.0)))


@dataclass(frozen=True)
class _Part:
    """One place of a run's group, as drawn: its prototype's key and shape
    (a factored one, made here), and each member's placement.
    """

    key: bytes
    shape: Shape | None
    placements: list[np.ndarray]
    factored: bool


_IDENTITY = np.hstack([np.eye(3), np.zeros((3, 1))])


class Instancing:
    """Turns the deltas a :class:`~.extract.FrameStream` made into the deltas
    a core drawer is handed, its runs batched (module docstring), keeping
    what it sent. Make one per core drawer, with its stream.

    ``enabled`` false hands every delta on as it is.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        #: The stream as made.
        self._in = StreamState()
        #: What was sent: records by slot, keys held, the order.
        self._out: dict[int, Record] = {}
        self._out_keys: set[bytes] = set()
        self._out_order: list[tuple[str, int, int]] | None = None
        #: The grid the held runs were found at.
        self._grid: float | None = None
        #: Each leaf record's reading, by its id (the record held beside).
        self._leaves: dict[int, tuple[Record, _Leaf | None]] = {}
        #: Each shape's factoring, by its key and the grid its canonical
        #: shape was snapped to.
        self._factored: dict[
            tuple[bytes, float], tuple[Factored, bytes, Shape] | None
        ] = {}
        #: Each run's record, by its members' records and period.
        self._runs: dict[
            tuple[tuple[int, ...], int], tuple[tuple[Record, ...], Record]
        ] = {}
        #: Shapes and paint tables made here, by key.
        self._made: dict[bytes, Shape | PaintTable] = {}
        #: Instances records sent and leaves factored, over the stream's life.
        self.instances_sent = 0
        self.factored_leaves = 0

    def __call__(self, delta: Delta, grid: float | None = None) -> Delta:
        """The delta the core is handed for ``delta``; ``grid`` the frame's
        input grid (:func:`~.canonical.input_grid` of the views held, where
        not given), which factored prototypes are snapped to as the snap
        will snap them.
        """
        if not self.enabled:
            return delta
        from .canonical import input_grid

        before = dict(self._in.records)
        self._in.apply(delta)
        if grid is None:
            views = [] if self._in.view is None else [self._in.view]
            views += [
                r.fields["sub_frame"][1]
                for r in self._in.records.values()
                if r.fields.get("sub_frame") is not None
            ]
            grid = input_grid(views)
        found = Delta(t=delta.t, seq=delta.seq, objects=list(delta.objects))
        found.view = delta.view
        held = (
            not delta.set
            and not delta.remove
            and delta.order is None
            and not delta.source
            and not delta.upload
            and not delta.upload_paints
            and not delta.release
            and grid == self._grid
        )
        if held:
            return found
        self._grid = grid
        records, order = self._batched(grid)
        used: set[bytes] = set()
        view = self._in.view
        if view is not None and isinstance(view.get("background"), bytes):
            used.add(view["background"])
        for record in records.values():
            used.update(record.keys)
        for key in sorted(used - self._out_keys):
            item = self._made.get(key)
            if item is None:
                item = self._in.shapes.get(key)
            if item is None:
                item = self._in.paints[key]
            if isinstance(item, Shape):
                found.upload.append(item)
            else:
                found.upload_paints.append(item)
        moved = {slot for slot, _, _ in delta.source}
        for slot, record in records.items():
            if self._out.get(slot) is record:
                continue
            if (
                slot in moved
                and slot in self._out
                and self._out[slot] is before.get(slot)
                and record is self._in.records.get(slot)
            ):
                # Its drawing is the one held: its new source alone.
                found.source.append(
                    (
                        slot,
                        record.fields["source"],
                        record.fields.get("facing_group", 0),
                    )
                )
                continue
            found.set.append((slot, record))
            if record.kind == "instances":
                self.instances_sent += 1
        found.remove = [slot for slot in self._out if slot not in records]
        found.release = sorted(self._out_keys - used)
        if order != self._out_order:
            found.order = order
        self._out = records
        self._out_keys = used
        self._out_order = order
        self._made = {k: v for k, v in self._made.items() if k in used}
        return found

    # Runs

    def _reading(self, record: Record) -> _Leaf | None:
        held = self._leaves.get(id(record))
        if held is not None and held[0] is record:
            return held[1]
        leaf = _leaf(record)
        self._leaves[id(record)] = (record, leaf)
        return leaf

    def _factoring(
        self, key: bytes, grid: float
    ) -> tuple[Factored, bytes, Shape] | None:
        """The factoring of shape ``key`` and its canonical shape snapped to
        ``grid``, with that shape's key.
        """
        at = (key, grid)
        if at not in self._factored:
            found = factor(self._in.shapes[key].data)
            if found is None:
                self._factored[at] = None
            else:
                shape = Shape("path", _snap(found.shape, grid))
                self._factored[at] = (found, shape.key, shape)
        return self._factored[at]

    def _batched(
        self, grid: float
    ) -> tuple[dict[int, Record], list[tuple[str, int, int]]]:
        """The records and order the core is sent for the stream as made."""
        state = self._in
        entries = state.order
        leaves: list[_Leaf | None] = []
        signatures: list[int] = []
        names: dict[Any, int] = {}
        # A record a display's sub-frame draws is named by its slot there:
        # it is sent as it is.
        named = {
            entry[1]
            for record in state.records.values()
            if record.fields.get("sub_frame") is not None
            for entry in record.fields["sub_frame"][2]
        }
        for layer, slot, part in entries:
            record = state.records[slot]
            leaf = self._reading(record) if part == 0 and slot not in named else None
            leaves.append(leaf)
            if leaf is None:
                signatures.append(-1 - len(signatures))
                continue
            factoring = self._factoring(leaf.shape, grid)
            # A leaf that factors batches with those whose canonical shapes are
            # its own; one that does not, with those of its own shape.
            prototype = (
                ("own", leaf.shape)
                if factoring is None
                else ("canonical", factoring[1])
            )
            name = (layer, prototype, leaf.block)
            signatures.append(names.setdefault(name, len(names)))
        records: dict[int, Record] = {}
        batched: set[int] = set()
        order: list[tuple[str, int, int]] = []
        runs: dict[tuple[tuple[int, ...], int], tuple[tuple[Record, ...], Record]] = {}
        at = 0
        while at < len(entries):
            period, groups = _run_at(signatures, at)
            run = leaves[at : at + period * groups]
            factored = any(
                len({leaf.shape for leaf in run[j::period] if leaf is not None}) > 1
                for j in range(period)
            )
            if groups < 2 or (factored and groups < FEWEST_FACTORED_GROUPS):
                layer, slot, part = entries[at]
                records[slot] = state.records[slot]
                order.append((layer, slot, part))
                at += 1
                continue
            end = at + period * groups
            members = [state.records[slot] for _, slot, _ in entries[at:end]]
            ident = (tuple(id(r) for r in members), period)
            cached = self._runs.get(ident)
            if cached is not None and all(
                a is b for a, b in zip(cached[0], members, strict=True)
            ):
                record = cached[1]
            else:
                record = self._run_record(
                    [slot for _, slot, _ in entries[at:end]],
                    members,
                    leaves[at:end],
                    period,
                    grid,
                )
            runs[ident] = (tuple(members), record)
            batched.update(slot for _, slot, _ in entries[at:end])
            slot = SLOT_BASE + entries[at][1]
            records[slot] = record
            order.append((entries[at][0], slot, 0))
            at = end
        # Records outside the order (a sub-frame's) are held as they are.
        for slot, record in state.records.items():
            if slot not in batched:
                records.setdefault(slot, record)
        self._runs = runs
        live = {id(r) for r in state.records.values()}
        self._leaves = {k: v for k, v in self._leaves.items() if k in live}
        keys = {(r.fields.get("shape"), grid) for r in state.records.values()}
        self._factored = {k: v for k, v in self._factored.items() if k in keys}
        return records, order

    def _run_record(
        self,
        slots: Sequence[int],
        members: Sequence[Record],
        leaves: Sequence[_Leaf | None],
        period: int,
        grid: float,
    ) -> Record:
        """The instances record of one run: ``members`` (in ``slots``) repeat
        a group of ``period``.
        """
        parts: list[_Part] = []
        for j in range(period):
            own = [leaf for leaf in leaves[j::period] if leaf is not None]
            if len({leaf.shape for leaf in own}) == 1:
                # One shape throughout: its own key, each at the identity.
                parts.append(_Part(own[0].shape, None, [_IDENTITY] * len(own), False))
                continue
            factorings = [self._factoring(leaf.shape, grid) for leaf in own]
            assert all(f is not None for f in factorings)
            first = factorings[0]
            assert first is not None
            parts.append(
                _Part(
                    first[1],
                    first[2],
                    [f[0].place for f in factorings if f is not None],
                    True,
                )
            )
        placements = np.empty((len(members), 3, 4))
        fills = np.zeros((len(members), 4))
        strokes = np.zeros((len(members), 4))
        for i, leaf in enumerate(leaves):
            assert leaf is not None
            placements[i] = parts[i % period].placements[i // period]
            if leaf.fill is not None:
                fills[i] = leaf.fill
            if leaf.stroke is not None:
                strokes[i] = leaf.stroke
        for part in parts:
            if part.shape is not None:
                self._made[part.key] = part.shape
            if part.factored:
                self.factored_leaves += len(part.placements)
                from .extract import _count_batching

                _count_batching(factored=len(part.placements))
        fill_table = PaintTable(fills)
        stroke_table = PaintTable(strokes)
        self._made[fill_table.key] = fill_table
        self._made[stroke_table.key] = stroke_table
        head = members[0].fields
        fields: dict[str, Any] = _prototype(head, parts[0].key)
        fields.update(
            space=head.get("space", "screen"),
            z_index=head.get("z_index", 0.0),
            shade_in_3d=False,
            placements=placements,
            instance_fill=fill_table.key,
            instance_stroke=stroke_table.key,
            objects=np.asarray(slots, dtype=np.uint32),
        )
        if period > 1:
            fields["parts"] = tuple(
                _prototype(members[j].fields, parts[j].key) for j in range(1, period)
            )
        # Each member's stroke width, where one is not its prototype's.
        widths = np.array([leaf.width for leaf in leaves if leaf is not None])
        if any(
            widths[j::period].min() != widths[j::period].max() for j in range(period)
        ):
            fields["instance_widths"] = widths
        if head.get("source") is not None:
            fields["source"] = head["source"]
        from .extract import _count_batching

        _count_batching(instances=1)
        return Record("instances", fields)


def _prototype(fields: Mapping[str, Any], shape: bytes) -> dict[str, Any]:
    """A prototype's shape and paint block, from a member's fields: its
    colours stand only for whether it fills and strokes (each instance's own
    are its rows of the record's tables).
    """
    found: dict[str, Any] = {"shape": shape}
    for name in ("fill", "stroke", "background_stroke"):
        value = fields.get(name)
        if value is not None:
            found[name] = value
    for name in ("stroke_width", *_BLOCK):
        if name in fields:
            found[name] = fields[name]
    return found


def _run_at(signatures: Sequence[int], at: int) -> tuple[int, int]:
    """The run starting at ``at``: its period and how many whole groups of
    it repeat there, the most leaves covered, the shortest period among
    those; a period of 1 and one group where no run starts.
    """
    best = (1, 1)
    covered = 1
    if signatures[at] < 0:
        return best
    for period in range(1, MOST_PARTS + 1):
        group = signatures[at : at + period]
        if len(group) < period or any(s < 0 for s in group):
            break
        groups = 1
        while signatures[at + groups * period : at + (groups + 1) * period] == group:
            groups += 1
        if groups >= 2 and groups * period > covered:
            best, covered = (period, groups), groups * period
    return best


def expand(state: StreamState) -> StreamState:
    """``state`` with each instances record drawn as its members: each a
    path record under its own slot (its object), its prototype's points
    placed by its placement, painted by its prototype's block in its own
    colours, at the run's place in the order. What Cairo reads of a frame the
    core drew from instances (:meth:`~.core_painter.CorePainter.outlines`);
    ``state`` itself where it holds none.
    """
    if all(record.kind != "instances" for record in state.records.values()):
        return state
    found = StreamState()
    found.view = state.view
    found.shapes = dict(state.shapes)
    found.paints = dict(state.paints)
    found.objects = dict(state.objects)
    for layer, slot, part in state.order:
        record = state.records[slot]
        if record.kind != "instances":
            found.records[slot] = record
            found.order.append((layer, slot, part))
            continue
        fields = record.fields
        prototypes = [_prototype(fields, fields["shape"])]
        prototypes += [dict(p) for p in fields.get("parts", ())]
        fills = state.paints[fields["instance_fill"]].rows
        strokes = state.paints[fields["instance_stroke"]].rows
        place = np.asarray(fields.get("place", _IDENTITY), dtype=float).reshape(3, 4)
        for i, (member, placement) in enumerate(
            zip(fields["objects"], fields["placements"], strict=True)
        ):
            prototype = prototypes[i % len(prototypes)]
            points = state.shapes[prototype["shape"]].data
            m = np.asarray(placement, dtype=float).reshape(3, 4)
            composed = np.zeros((3, 4))
            composed[:, :3] = _apply(m[:, :3], place[:, :3].T).T
            composed[:, 3] = _apply(m[:, :3], place[:, 3][np.newaxis])[0] + m[:, 3]
            shape = Shape("path", _apply(composed[:, :3], points) + composed[:, 3])
            found.shapes[shape.key] = shape
            member_fields = {k: v for k, v in prototype.items() if k not in ("shape",)}
            member_fields["shape"] = shape.key
            if prototype.get("fill") is not None:
                member_fields["fill"] = fills[i : i + 1]
            if prototype.get("stroke") is not None:
                member_fields["stroke"] = strokes[i : i + 1]
            for name in ("space", "z_index", "shade_in_3d"):
                if name in fields:
                    member_fields[name] = fields[name]
            if fields.get("instance_widths") is not None:
                member_fields["stroke_width"] = float(fields["instance_widths"][i])
            found.records[int(member)] = Record("path", member_fields)
            found.order.append((layer, int(member), 0))
    return found
