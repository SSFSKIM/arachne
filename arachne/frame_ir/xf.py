"""XF, the Frame IR, on the Python side: records, shapes, deltas, a stream.

XF's meaning and binary form are fixed in ``core/XF.md``; the drawers in
``arachne_core`` read it. This module holds the same description as Python
values, so that the Cairo painter can draw a frame from it alone and the
extractor can build it without the core installed. A :class:`Record` holds
exactly XF's fields, by the names the core's binding takes
(``FrameDelta.set_record``), and nothing that refers back to a mobject: what a
drawer can see of a frame is what XF carries. :meth:`Delta.to_core` hands a
delta to the core, which checks it whole.
"""

from __future__ import annotations

__all__ = [
    "FEATURE_BACKGROUND_IMAGE",
    "FEATURE_PRE_PROJECTED",
    "LAYERS",
    "XF_VERSION",
    "Delta",
    "Record",
    "Shape",
    "StreamState",
    "image_key",
    "paint_key",
    "path_key",
]

import hashlib
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

#: The newest XF version this module knows. A delta is written at the lowest
#: version its content needs: 1.1 when it carries a pre-projected record, 1.2
#: when it carries snapped geometry (an input grid, a record's planes), 1.3
#: when a face group carries its faces' ends or stroke widths, 1.4 when a
#: record carries its joins, 1.5 when a record carries its ``source`` (the
#: objects it was drawn from) or its ``facing_group``, which nothing drawn
#: reads.
XF_VERSION = (1, 5)
#: The record's paint is CE's background-image fill (Cairo only).
FEATURE_BACKGROUND_IMAGE = 1
#: The record carries what a camera of the scene's own made of it
#: (``pre_projected``: its pre-display points, gradient ends and colours), so
#: a drawer places it with the frame mapping alone (XF 1.1).
FEATURE_PRE_PROJECTED = 2
#: The layers an order entry names, in XF's codes' order.
LAYERS = ("static", "moving", "overlay")

_PATH, _MESH, _IMAGE, _PAINT = 1, 2, 3, 16
_F64, _U8 = 1, 3


def _canonical(kind: int, arrays: Iterable[tuple[int, np.ndarray]]) -> bytes:
    arrays = list(arrays)
    parts = [bytes([kind, len(arrays)])]
    for dtype, array in arrays:
        parts.append(bytes([dtype, array.ndim]))
        parts.append(struct.pack(f"<{array.ndim}Q", *array.shape))
        little = "<f8" if dtype == _F64 else "u1"
        parts.append(np.ascontiguousarray(array, dtype=little).tobytes())
    return b"".join(parts)


def _key(canonical: bytes) -> bytes:
    return hashlib.blake2b(canonical, digest_size=16).digest()


def path_key(points: np.ndarray) -> bytes:
    """The content key of a path shape, f64 (n, 3)."""
    return _key(_canonical(_PATH, [(_F64, points)]))


def image_key(rgba: np.ndarray) -> bytes:
    """The content key of an image shape, u8 (height, width, 4)."""
    return _key(_canonical(_IMAGE, [(_U8, rgba)]))


def paint_key(rows: np.ndarray) -> bytes:
    """The content key of a paint table, f64 (n, 4)."""
    return _key(_canonical(_PAINT, [(_F64, rows)]))


class Shape:
    """Geometry uploaded once under its content key: a path's points (CE's
    cubic layout) or an image's straight RGBA. ``data`` is read-only.
    """

    __slots__ = ("data", "key", "kind")

    def __init__(self, kind: str, data: np.ndarray, key: bytes | None = None) -> None:
        if kind == "path":
            data = np.asarray(data, dtype=float).reshape(-1, 3)
            found = path_key(data) if key is None else key
        elif kind == "image":
            data = np.asarray(data, dtype=np.uint8)
            if data.ndim != 3 or data.shape[2] != 4:
                raise ValueError("an image shape is u8 (height, width, 4)")
            found = image_key(data) if key is None else key
        else:
            raise ValueError(f"no Python drawer holds a {kind} shape")
        if data.flags.writeable:
            data = data.copy()
            data.flags.writeable = False
        self.kind = kind
        self.data = data
        self.key = found


class PaintTable:
    """RGBA rows uploaded under a key: a paint of 3 or more stops, or one
    colour per face or point.
    """

    __slots__ = ("key", "rows")

    def __init__(self, rows: np.ndarray, key: bytes | None = None) -> None:
        rows = np.asarray(rows, dtype=float).reshape(-1, 4)
        if rows.flags.writeable:
            rows = rows.copy()
            rows.flags.writeable = False
        self.rows = rows
        self.key = paint_key(rows) if key is None else key


#: A face group's lists that, empty, are not carried (``core/XF.md``): a
#: record holds none empty, so each reader finds the list absent.
EMPTY_IS_ABSENT = ("face_ends", "face_widths")


class Record:
    """One drawn object: a kind and XF's fields, by the names the core's
    binding takes. Never changed once made; a frame that draws an object
    differently makes a new record. ``keys`` are the shapes and paint tables
    it refers to (``sub_frame`` views' backgrounds included). A list of
    :data:`EMPTY_IS_ABSENT` given empty is left out.
    """

    __slots__ = ("fields", "keys", "kind")

    def __init__(self, kind: str, fields: Mapping[str, Any]) -> None:
        self.kind = kind
        self.fields: dict[str, Any] = dict(fields)
        for name in EMPTY_IS_ABSENT:
            value = self.fields.get(name)
            if value is not None and len(value) == 0:
                del self.fields[name]
        self.keys = tuple(_references(self.fields))

    def __repr__(self) -> str:
        return f"Record({self.kind}, {sorted(self.fields)})"

    def with_source(self, source: Any, facing_group: int) -> Record:
        """This record with another source and facing group (XF 1.5): a held
        record's, given a row of :attr:`Delta.source`.
        """
        fields = dict(self.fields)
        fields["source"] = source
        fields.pop("facing_group", None)
        if facing_group:
            fields["facing_group"] = facing_group
        return Record(self.kind, fields)


def _references(fields: Mapping[str, Any]) -> Iterable[bytes]:
    for name in ("shape", "colors", "face_fill", "face_stroke", "fill_image"):
        value = fields.get(name)
        if isinstance(value, bytes):
            yield value
    blend = fields.get("blend")
    if blend is not None:
        yield blend[0]
    for name in ("fill", "stroke", "background_stroke"):
        if isinstance(fields.get(name), bytes):
            yield fields[name]
    paint_b = fields.get("paint_b")
    if paint_b:
        for value in paint_b.values():
            if isinstance(value, bytes):
                yield value
    sub_frame = fields.get("sub_frame")
    if sub_frame is not None and isinstance(sub_frame[1].get("background"), bytes):
        yield sub_frame[1]["background"]
    pre = fields.get("pre_projected")
    if pre is not None:
        yield pre["points"]
        for name in _PAINTS:
            if isinstance(pre.get(name), bytes):
                yield pre[name]


#: The paints a record states, by their field names.
_PAINTS = ("fill", "stroke", "background_stroke")


@dataclass
class Delta:
    """One drawn frame's changes against a stream: empty for a held frame."""

    t: Fraction = Fraction(0)
    seq: int = 0
    view: dict[str, Any] | None = None
    objects: list[tuple[int, str, str]] = field(default_factory=list)
    upload: list[Shape] = field(default_factory=list)
    upload_paints: list[PaintTable] = field(default_factory=list)
    release: list[bytes] = field(default_factory=list)
    set: list[tuple[int, Record]] = field(default_factory=list)
    remove: list[int] = field(default_factory=list)
    order: list[tuple[str, int, int]] | None = None
    #: Held records' new sources (XF 1.5), their drawing unchanged: (slot,
    #: source, facing group) for slots the delta neither sets nor removes.
    source: list[tuple[int, Any, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.view is not None
            or self.objects
            or self.upload
            or self.upload_paints
            or self.release
            or self.set
            or self.remove
            or self.order is not None
            or self.source
        )

    def to_core(self) -> Any:
        """This delta as the core's ``FrameDelta`` (imports ``arachne_core``)."""
        import arachne_core

        core = arachne_core.FrameDelta((self.t.numerator, self.t.denominator), self.seq)
        if self.view is not None:
            core.set_view(**_core_view(self.view))
        for ref, stable_id, source in self.objects:
            core.add_object(ref, stable_id, source)
        for shape in self.upload:
            if shape.kind == "path":
                key = core.upload_path(shape.data)
            else:
                key = core.upload_image(shape.data)
            assert key == shape.key, "the core keys a shape as XF documents"
        for table in self.upload_paints:
            key = core.upload_paint(table.rows)
            assert key == table.key, "the core keys a paint table as XF documents"
        for key in self.release:
            core.release(key)
        for slot, record in self.set:
            core.set_record(record.kind, slot, slot, **_core_fields(record.fields))
        for slot in self.remove:
            core.remove(slot)
        if self.order is not None:
            core.set_order(self.order)
        for slot, source, facing_group in self.source:
            core.set_source(slot, source=source, facing_group=facing_group)
        return core

    @classmethod
    def from_contents(cls, contents: Mapping[str, Any]) -> Delta:
        """A delta from what the core read out of XF's bytes
        (``arachne_core.FrameDelta.contents()``): the description a drawer
        holds once the wire has carried it.
        """
        num, den = contents["t"]
        delta = cls(t=Fraction(num, den), seq=contents["seq"])
        if contents["view"] is not None:
            delta.view = _python_view(contents["view"])
        delta.objects = [tuple(o) for o in contents["objects"]]
        for kind, key, data in contents["upload"]:
            if kind in ("path", "image"):
                # A mesh or a reserved kind no Cairo record draws: a record
                # that uses one is refused whole when it is drawn.
                delta.upload.append(Shape(kind, data, key))
        for key, rows in contents["upload_paints"]:
            delta.upload_paints.append(PaintTable(rows, key))
        delta.release = list(contents["release"])
        for kind, slot, _, fields in contents["set"]:
            fields = dict(fields)
            if fields.get("sub_frame") is not None:
                rank, view, order = fields["sub_frame"]
                fields["sub_frame"] = (
                    rank,
                    _python_view(view),
                    [tuple(entry) for entry in order],
                )
            delta.set.append((slot, Record(kind, fields)))
        delta.remove = list(contents["remove"])
        if contents["order"] is not None:
            delta.order = [tuple(entry) for entry in contents["order"]]
        delta.source = [(slot, path, group) for slot, path, group in contents["source"]]
        return delta


_VECTORS = ("frame_center", "mapping_centre", "light_source")


def _python_view(view: Mapping[str, Any]) -> dict[str, Any]:
    found = dict(view)
    for name in _VECTORS:
        if name in found:
            found[name] = tuple(float(v) for v in found[name])
    return found


def _core_view(view: Mapping[str, Any]) -> dict[str, Any]:
    return dict(view)


def _core_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """A record's fields as the core's ``set_record`` takes them."""
    found = dict(fields)
    if "corners" in found:
        found["corners"] = np.reshape(found["corners"], (4, 3)).tolist()
    # The fields that carry a feature set its bit themselves.
    features = int(found.pop("features", 0))
    if found.get("fill_image") is not None:
        features &= ~FEATURE_BACKGROUND_IMAGE
    if found.get("pre_projected") is not None:
        features &= ~FEATURE_PRE_PROJECTED
    if features:
        found["features"] = features
    for name in _PAINTS:
        if isinstance(found.get(name), np.ndarray):
            found[name] = found[name].tolist()
    pre = found.get("pre_projected")
    if pre is not None:
        found["pre_projected"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pre.items()
        }
    paint_b = found.get("paint_b")
    if paint_b is not None:
        found["paint_b"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in paint_b.items()
        }
    sub_frame = found.get("sub_frame")
    if sub_frame is not None:
        rank, view, order = sub_frame
        found["sub_frame"] = (rank, _core_view(view), list(order))
    return found


class StreamState:
    """What a drawer holds across frames, kept by applying deltas: shapes and
    paint tables by key, records by slot, the view and the order.
    """

    def __init__(self) -> None:
        self.shapes: dict[bytes, Shape] = {}
        self.paints: dict[bytes, PaintTable] = {}
        self.records: dict[int, Record] = {}
        self.objects: dict[int, tuple[str, str]] = {}
        self.view: dict[str, Any] | None = None
        self.order: list[tuple[str, int, int]] = []

    def apply(self, delta: Delta) -> None:
        if delta.view is not None:
            self.view = delta.view
        for ref, stable_id, source in delta.objects:
            self.objects[ref] = (stable_id, source)
        for shape in delta.upload:
            self.shapes.setdefault(shape.key, shape)
        for table in delta.upload_paints:
            self.paints.setdefault(table.key, table)
        for slot, record in delta.set:
            self.records[slot] = record
        for slot in delta.remove:
            self.records.pop(slot, None)
        for slot, source, group in delta.source:
            self.records[slot] = self.records[slot].with_source(source, group)
        for key in delta.release:
            self.shapes.pop(key, None)
            self.paints.pop(key, None)
        if delta.order is not None:
            self.order = list(delta.order)

    def paint(self, value: Any) -> np.ndarray | None:
        """A paint's RGBA rows: inline stops, or a table's rows."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return self.paints[value].rows
        return value
