"""Canonical float inputs: every geometric float the core receives, snapped
to a grid the camera sets.

The core draws the same frame from the same bytes on every machine (no fused
multiply-add, transcendental functions from the pure-Rust ``libm``). What it
is handed is not so: the geometry a scene computes with numpy's vectorised
loops (trigonometry, matrix products) differs between CPUs in its last bits,
and the core decides by comparison (depth order at a sub-sample, which
resolve a pixel takes, a plane's choice), so a difference of one ulp moves
whole pixels. Snapped, two machines' values agree unless a grid boundary lies
between them, with probability about ``|difference| / grid`` per value.

A value is snapped as ``round(x / grid) * grid``: the grid is a power of
two, so each step is exact IEEE arithmetic and gives the same bits on every
machine (``-0.0`` is made ``0.0``). Positions (scene units) are snapped to
the **input grid** (:func:`input_grid`): ``2^(e - 9)`` for ``e`` the binary
exponent of the scene units one pixel spans, ``u = frame_width /
(pixel_width * zoom)``, so a pixel holds 256 to 512 steps whatever the
camera's zoom. The grid is the finest any camera that sees the geometry
needs: the scene's camera and each display's (a sub-frame's view). When a
frame's grid differs from the last frame's, every shape and record held is
snapped again, from the values the scene made, so a snapped value is the
same function of its value and the frame's grid whatever frames came
before. Dimensionless values (angles, zoom, the linear part of an affine,
interpolation parameters) are snapped to :data:`RATIO_GRID`. Colours,
widths, opacities, z-indices and the camera's constants are not geometry the
scene computes: they reach the core as given. Every field is classified
here; a field this module does not know is refused, so nothing reaches the
core unsnapped by being new.

Whether a path is flat is decided before its points are snapped, by the
core's own rules on the values as made (``Stream.planes`` on a stream of the
unsnapped deltas), and carried with the snapped record as its ``planes``:
snapped, the points of a slightly curved face and of a flat one can be the
same points. So is what CE's lighting derives from nearby points: the
normals each shaded path is lit by at its two lit corners
(``Stream.normals``, decided for the frame's grid), carried as the record's
``normals`` and snapped to :data:`RATIO_GRID`. On a finely sampled curve
that normal is the cross product of two nearly opposite chords, which the
snap's step would turn by tens of degrees. The view carries the grid as its
``input_grid`` (XF 1.2).

Whether a path's cubics join and whether its subpaths close are decided by
the core on the snapped points, by CE's point-equality rule (each coordinate
within ``1e-6 + 1e-5 * |b|``). A comparison is sensitive only at its
threshold: a snap moves each point at most ``g / 2``, so a gap moves at most
``g``, and only a gap made within ``g`` of the tolerance can be decided
otherwise (near the origin, where the tolerance is under ``g``, a gap of up
to about ``g`` that CE leaves open can be closed). Content keeps far from it
(joins are copies of one point, closed ends a float's noise apart, breaks
pixels apart), and :data:`point_equality_disagreements` records every
boundary decided otherwise, so the test suite finds any that does not.

Only the core's inputs are snapped (:class:`~.core_painter.CorePainter`,
and the display pictures drawn alongside them on the core path): frames
drawn on ``ir_cairo`` stay what Cairo draws from the scene's own values.
"""

from __future__ import annotations

__all__ = [
    "POSITION_GRID",
    "RATIO_GRID",
    "Canonical",
    "canonical_order",
    "input_grid",
    "snap",
]

import contextlib
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

from .xf import Delta, Record, Shape

#: The fixed grid of positions that order, not place: the camera's depth
#: sort keys, and the frame width a view's input grid is read from.
POSITION_GRID = 2.0**-16
#: The grid dimensionless values are snapped to.
RATIO_GRID = 2.0**-24
#: How far below the pixel's binary exponent the input grid sits.
_STEPS_EXPONENT = 9

#: The record kinds that draw paths, whose flatness is decided before the
#: snap.
_PATH_KINDS = ("path", "face_group", "instances")


def snap(values: Any, grid: float) -> Any:
    """``values`` (a number, a sequence or an array) on ``grid``, as given:
    a float for a number, a tuple of floats for a tuple, an array for any
    other sequence.
    """
    found = np.round(np.asarray(values, dtype=np.float64) / grid) * grid + 0.0
    if isinstance(values, (int, float, np.floating)):
        return float(found)
    if isinstance(values, tuple):
        return tuple(float(v) for v in found.ravel())
    return found


def input_grid(views: list[Mapping[str, Any]]) -> float:
    """The grid positions seen through ``views`` are snapped to: the finest
    of each view's ``2^(e - 9)``, ``e`` the binary exponent (``math.frexp``,
    exact) of the scene units its pixel spans, read from its frame width
    and zoom on fixed grids.
    """
    found = math.inf
    for view in views:
        width = snap(float(view["frame_width"]), POSITION_GRID)
        zoom = snap(float(view.get("zoom", 1.0)), RATIO_GRID)
        span = width / (int(view["pixel_width"]) * zoom)
        if not (span > 0 and math.isfinite(span)):
            continue
        _, exponent = math.frexp(span)
        found = min(found, math.ldexp(1.0, exponent - _STEPS_EXPONENT))
    return POSITION_GRID if found == math.inf else found


@contextlib.contextmanager
def canonical_order(camera: Any) -> Iterator[None]:
    """Within: ``camera``, and each display camera it draws through, sorts
    what it draws by depth on snapped keys (``ThreeDCamera``'s
    ``get_mobjects_to_display``): the order is the core's input too, and a
    near-tie in it is taken as a tie, which the sort keeps in the order it
    was handed.
    """
    from arachne.timeline.camera_code import drawn_cameras

    cameras = [each for each in drawn_cameras(camera) if hasattr(each, "__dict__")]
    held = [vars(each).get(_ORDER) for each in cameras]
    for each in cameras:
        vars(each)[_ORDER] = True
    try:
        yield
    finally:
        for each, before in zip(cameras, held, strict=True):
            if before is None:
                vars(each).pop(_ORDER, None)
            else:
                vars(each)[_ORDER] = before


#: The camera attribute under which :func:`canonical_order` holds.
_ORDER = "_xf_canonical_order"


def _position(value: Any, grid: float) -> Any:
    return snap(value, grid)


def _ratio(value: Any, grid: float) -> Any:
    return snap(value, RATIO_GRID)


def _affine(value: Any, grid: float) -> np.ndarray:
    """An affine (``(3, 4)``, or ``(n, 3, 4)``): its linear part a ratio, its
    translation a position.
    """
    found = np.array(value, dtype=np.float64)
    found[..., :3] = snap(found[..., :3], RATIO_GRID)
    found[..., 3] = snap(found[..., 3], grid)
    return found


def _as_given(value: Any, grid: float) -> Any:
    return value


#: What each view field is.
_VIEW: dict[str, Callable[[Any, float], Any]] = {
    "frame_width": _position,
    "frame_height": _position,
    "frame_center": _position,
    "mapping_centre": _position,
    "focal_distance": _position,
    "light_source": _position,
    "phi": _ratio,
    "theta": _ratio,
    "gamma": _ratio,
    "zoom": _ratio,
    "pixel_width": _as_given,
    "pixel_height": _as_given,
    "camera": _as_given,
    "background": _as_given,
    "cairo_line_width_multiple": _as_given,
    "use_z_index": _as_given,
    "exponential_projection": _as_given,
    "shading_factor": _as_given,
    "should_apply_shading": _as_given,
    "light_model": _as_given,
}

#: What each record field is, but those that refer to shapes
#: (:meth:`Canonical._record`).
_RECORD: dict[str, Callable[[Any, float], Any]] = {
    "sort_point": _position,
    "fixed_orientation_centre": _position,
    "gradient": _position,
    "corners": _position,
    "place": _affine,
    "placements": _affine,
    "mix": _as_given,
    "features": _as_given,
    "layer": _as_given,
    "opacity": _as_given,
    "space": _as_given,
    "z_index": _as_given,
    "shade_in_3d": _as_given,
    "joint": _as_given,
    "cap": _as_given,
    "fill": _as_given,
    "stroke": _as_given,
    "background_stroke": _as_given,
    "stroke_width": _as_given,
    "background_stroke_width": _as_given,
    "paint_b": _as_given,
    "dash": _as_given,
    "fill_image": _as_given,
    "uv": _as_given,
    "grid": _as_given,
    "face_fill": _as_given,
    "face_stroke": _as_given,
    "colors": _as_given,
    "radius": _as_given,
    "resampling": _as_given,
    "image_opacity": _as_given,
}


def _classified(
    table: Mapping[str, Callable[[Any, float], Any]], what: str, name: str
) -> Callable[[Any, float], Any]:
    found = table.get(name)
    if found is None:
        raise ValueError(f"canonical: a {what} field {name!r} is not classified")
    return found


class Canonical:
    """Deltas as the core receives them, their geometry snapped to the
    frame's input grid, each path's flatness decided before the snap.

    A path shape is uploaded snapped, keyed by its snapped points. A path
    record that the core would make points of (a reveal window cut from its
    shape, a blend with a second shape) is made into those points here, with
    the engine's own functions (:mod:`.geometry`, which give the scene's
    points bit for bit) on the shapes as the scene made them, and snapped:
    the core draws the same snapped points from a record as from the points
    a live capture hands it, where snapping the shapes and the parameter
    apart would not. Its paint blend (``paint_b``, ``mix``) stays the core's.

    Shapes whose points snap alike are one snapped shape, uploaded while the
    first of its holders (a shape of the stream, or a record made into
    points) holds it and released with the last, as each delta ends: a shape
    held when a delta starts and when it ends is neither uploaded nor
    released by it, whatever held it in between. The stream's uploads are
    counted first, then its releases, then its records, as the core applies
    them.

    With ``planes`` (the core's input), a stream of the deltas as made, held
    by the core (``arachne_core.Stream``), decides how each path a record
    draws lies, and the snapped record carries it.
    """

    def __init__(self, *, planes: bool = True) -> None:
        #: Each path shape of the stream held, by its own key, as the scene
        #: made it.
        self._shapes: dict[bytes, Shape] = {}
        #: Each holder's snapped shape key: a stream shape's key, or
        #: ``("slot", slot)`` for a record made into points.
        self._snapped: dict[Any, bytes] = {}
        #: Each snapped shape held, by its key: its holders.
        self._holders: dict[bytes, set[Any]] = {}
        #: Within a delta: the snapped shapes it made, and the keys whose
        #: holders it changed, in the order it did (:meth:`_settle_shapes`).
        self._made: dict[bytes, Shape] = {}
        self._touched: dict[bytes, None] = {}
        #: Each record held, by slot, as the scene made it.
        self._records: dict[int, Record] = {}
        #: The view held, as the scene made it.
        self._view: dict[str, Any] | None = None
        #: The grid the held geometry is snapped to.
        self.grid: float | None = None
        #: Each held record's planarity codes, by slot.
        self._planes: dict[int, tuple[int, ...]] = {}
        self._decider: Any = None
        if planes:
            import arachne_core

            self._decider = arachne_core.Stream()

    def __call__(self, delta: Delta) -> Delta:
        found = Delta(t=delta.t, seq=delta.seq, objects=list(delta.objects))
        held = set(self._holders)
        self._made = {}
        self._touched = {}
        if delta.view is not None:
            self._view = dict(delta.view)
        for slot, record in delta.set:
            self._records[slot] = record
        for slot in delta.remove:
            self._records.pop(slot, None)
            self._planes.pop(slot, None)
        self._decide(delta)
        grid = input_grid(self._views())
        regrid = grid != self.grid
        self.grid = grid
        if self._view is not None and (delta.view is not None or regrid):
            found.view = self._view_of(self._view)
        for shape in delta.upload:
            if shape.kind == "image":
                found.upload.append(shape)
            elif shape.kind == "path":
                self._shapes[shape.key] = shape
                self._hold(shape.key, shape.data, found)
            else:
                raise ValueError(f"canonical: a {shape.kind} shape is not classified")
        if regrid:
            for key, shape in self._shapes.items():
                self._hold(key, shape.data, found)
        found.upload_paints = list(delta.upload_paints)
        for key in delta.release:
            if self._shapes.pop(key, None) is None:
                found.release.append(key)
            else:
                self._let_go(key, found)
        setting = dict(delta.set)
        if regrid:
            setting.update(
                (slot, record)
                for slot, record in sorted(self._records.items())
                if slot not in setting
            )
        normals = self._normals(setting, grid)
        found.set = [
            (slot, self._record(slot, record, found, normals.get(slot)))
            for slot, record in setting.items()
        ]
        for slot in delta.remove:
            self._let_go(("slot", slot), found)
        self._settle_shapes(held, found)
        found.remove = list(delta.remove)
        found.order = None if delta.order is None else list(delta.order)
        return found

    def _views(self) -> list[Mapping[str, Any]]:
        """Each view that sees the held geometry: the frame's, and each
        display's sub-frame view.
        """
        views: list[Mapping[str, Any]] = [] if self._view is None else [self._view]
        for record in self._records.values():
            sub_frame = record.fields.get("sub_frame")
            if sub_frame is not None:
                views.append(sub_frame[1])
        return views

    def _decide(self, delta: Delta) -> None:
        """How each path of the records ``delta`` sets lies, decided by the
        core on the values as made.
        """
        if self._decider is None:
            return
        self._decider.apply(delta.to_core())
        slots = [slot for slot, record in delta.set if record.kind in _PATH_KINDS]
        for slot in slots:
            self._planes.pop(slot, None)
        if slots and self._view is not None:
            for slot, codes in self._decider.planes(slots).items():
                self._planes[slot] = tuple(codes)

    def _normals(
        self, setting: Mapping[int, Record], grid: float
    ) -> dict[int, list[tuple[float, ...]]]:
        """The normals each shaded record of ``setting`` is lit by, decided
        by the core on the values as made for points snapped to ``grid``,
        snapped to :data:`RATIO_GRID`; by slot.
        """
        slots = [slot for slot, record in setting.items() if record.kind in _PATH_KINDS]
        if self._decider is None or not slots or self._view is None:
            return {}
        return {
            slot: [snap(tuple(n), RATIO_GRID) for n in normals]
            for slot, normals in self._decider.normals(slots, grid).items()
        }

    def _hold(self, holder: Any, points: Any, found: Delta) -> bytes:
        """``points`` snapped, held by ``holder`` (instead of what it held);
        their key. Whether the delta uploads them is decided once it is built
        (:meth:`_settle_shapes`).
        """
        assert self.grid is not None
        shape = Shape("path", _position(points, self.grid))
        _count_disagreements(np.asarray(points, dtype=np.float64), shape.data)
        if self._snapped.get(holder) == shape.key:
            return shape.key
        holders = self._holders.setdefault(shape.key, set())
        if not holders:
            self._made.setdefault(shape.key, shape)
            self._touched[shape.key] = None
        holders.add(holder)
        self._let_go(holder, found)
        self._snapped[holder] = shape.key
        return shape.key

    def _let_go(self, holder: Any, found: Delta) -> None:
        """``holder`` holds its snapped shape no longer. Whether the delta
        releases it is decided once it is built (:meth:`_settle_shapes`).
        """
        key = self._snapped.pop(holder, None)
        if key is None:
            return
        holders = self._holders[key]
        holders.discard(holder)
        if not holders:
            del self._holders[key]
            self._touched[key] = None

    def _settle_shapes(self, held: set[bytes], found: Delta) -> None:
        """The snapped shapes ``found`` uploads and releases, from what was
        ``held`` before it and what is held once it is built: a shape held at
        both ends is neither, however its holders changed in between (two
        records that swap shapes), and one nothing holds at the end is
        released in this delta.
        """
        for key in self._touched:
            if key in self._holders and key not in held:
                found.upload.append(self._made[key])
            elif key not in self._holders and key in held:
                found.release.append(key)

    def _key(self, key: bytes) -> bytes:
        """A shape's key as the core holds it: its snapped shape's, for a
        path shape of the stream.
        """
        return self._snapped.get(key, key)

    def _view_of(self, view: Mapping[str, Any]) -> dict[str, Any]:
        assert self.grid is not None
        found = {
            name: _classified(_VIEW, "view", name)(value, self.grid)
            for name, value in view.items()
        }
        found["input_grid"] = self.grid
        return found

    def _points(self, fields: dict[str, Any]) -> np.ndarray | None:
        """The points the core would make of a path record's shape (popping
        its ``reveal`` and ``blend``), or ``None`` where it makes none.
        """
        from .geometry import blended, revealed

        reveal = fields.pop("reveal", None)
        blend = fields.pop("blend", None)
        if reveal is None and blend is None:
            return None
        if fields.get("place") is not None and not np.array_equal(
            fields["place"], _IDENTITY
        ):
            raise ValueError("canonical: a placed path with a reveal or blend")
        points = self._shapes[fields["shape"]].data
        if reveal is not None:
            points = revealed(points, *reveal)
        if blend is not None:
            key, place, t = blend
            if not np.array_equal(place, _IDENTITY):
                raise ValueError("canonical: a blend with a placed second shape")
            points = blended(points, self._shapes[key].data, t)
        return points

    def _record(
        self,
        slot: int,
        record: Record,
        found: Delta,
        normals: list[tuple[float, ...]] | None = None,
    ) -> Record:
        assert self.grid is not None
        grid = self.grid
        fields = dict(record.fields)
        points = self._points(fields) if record.kind == "path" else None
        if points is not None:
            fields["shape"] = self._hold(("slot", slot), points, found)
        else:
            self._let_go(("slot", slot), found)
        snapped: dict[str, Any] = {}
        for name, value in fields.items():
            if value is None:
                snapped[name] = None
            elif name == "shape":
                snapped[name] = value if points is not None else self._key(value)
            elif name == "clip":
                key, place = value
                snapped[name] = (self._key(key), _affine(place, grid))
            elif name in ("blend", "reveal"):
                raise ValueError(f"canonical: a {record.kind} record with a {name}")
            elif name == "pre_projected":
                snapped[name] = {
                    **value,
                    "points": self._key(value["points"]),
                    "gradient": (
                        None
                        if value.get("gradient") is None
                        else _position(value["gradient"], grid)
                    ),
                }
            elif name == "sub_frame":
                rank, view, order = value
                snapped[name] = (rank, self._view_of(view), list(order))
            else:
                snapped[name] = _classified(_RECORD, "record", name)(value, grid)
        if slot in self._planes:
            snapped["planes"] = self._planes[slot]
        if normals is not None:
            snapped["normals"] = normals
        return Record(record.kind, snapped)


#: CE's ``tolerance_for_point_equality`` and the relative tolerance of its
#: ``consider_points_equals_2d``.
_POINT_EQUALITY = 1e-6
_POINT_RTOL = 1e-5

#: Every path whose snapped points CE's point-equality rule decides otherwise
#: than its points as made, where its cubics join or its subpaths close: the
#: number of points, the coordinates compared (2 as the core compares them on
#: a flat camera, 3 for the placed points of a 3D one), and how many
#: boundaries differ. Measured, never acted on; the tests assert it stays
#: empty.
point_equality_disagreements: list[tuple[int, int, int]] = []


def _equal(a: np.ndarray, b: np.ndarray, dims: int) -> np.ndarray:
    gap = np.abs(a[..., :dims] - b[..., :dims])
    within = gap <= _POINT_EQUALITY + _POINT_RTOL * np.abs(b[..., :dims])
    return np.all((gap == 0) | within, axis=-1)


def _boundaries(points: np.ndarray, splits: np.ndarray, dims: int) -> np.ndarray:
    """CE's decisions on ``points``: whether each cubic's end is the next's
    start, then, for each subpath ``splits`` (the starts of those that do
    not join) delimit, whether its end is its start.
    """
    at = np.arange(4, len(points), 4)
    joins = _equal(points[at - 1], points[at], dims)
    starts = np.concatenate([[0], splits])
    ends = np.concatenate([splits, [len(points)]])
    long = ends - starts >= 4
    closes = _equal(points[starts[long]], points[ends[long] - 1], dims)
    return np.concatenate([joins, closes])


def _count_disagreements(made: np.ndarray, snapped: np.ndarray) -> None:
    """Records ``made`` in :data:`point_equality_disagreements` where its
    snapped points' boundaries are decided otherwise.
    """
    if made.ndim != 2 or len(made) < 4:
        return
    for dims in (2, 3):
        at = np.arange(4, len(made), 4)
        splits = at[~_equal(made[at - 1], made[at], dims)]
        differ = int(
            (_boundaries(made, splits, dims) != _boundaries(snapped, splits, dims)).sum()
        )
        if differ:
            point_equality_disagreements.append((len(made), dims, differ))


#: The identity placement.
_IDENTITY = np.hstack([np.eye(3), np.zeros((3, 1))])
