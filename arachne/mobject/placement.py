"""A mobject's points as a shape under an affine map, and the bounds of points.

:meth:`~.Mobject.shift`, :meth:`~.Mobject.scale` and :meth:`~.Mobject.rotate`
on a large points value compose a :class:`Placement` instead of computing
every point again: the points are ``shape @ linear + offset``, computed once
when they are first read and kept with the placement. The shape is a points
value that never changes, so every placement of it shares it, whatever moves
it has been through.

The bounds of a points value (its least and greatest coordinates, of every
point or of the curve anchors) are computed once and kept with the value. A
placement that only translates and scales along the axes gets its bounds
from its shape's, exactly: a translation and a scale by one factor per axis
keep the order of each coordinate, so they carry the least and greatest
points to the least and greatest points.
"""

from __future__ import annotations

__all__ = ["UNKNOWN", "Placement", "anchor_bounds", "point_bounds"]

import hashlib
from typing import Any

import numpy as np

from .values import WORK, FrozenArray, FrozenPoints, adopt

#: A points value with fewer points than this is moved point by point: its
#: placement would cost more than it saves.
MIN_PLACED_POINTS = 256


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class Placement:
    """Points as ``shape @ linear + offset``.

    ``linear`` is ``None`` for the identity, a vector of three factors for a
    scale along the axes, or a 3×3 matrix acting on row vectors; ``offset``
    is a vector. A placement never changes: :meth:`then` composes a new one.
    """

    __slots__ = ("shape", "linear", "offset", "_points", "_bounds", "_key")

    shape: FrozenPoints
    linear: np.ndarray | None
    offset: np.ndarray

    def __init__(
        self,
        shape: FrozenPoints,
        linear: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> None:
        self.shape = shape
        self.linear = linear
        self.offset = _read_only(np.zeros(shape.shape[1])) if offset is None else offset
        self._points: FrozenPoints | None = None
        self._bounds: dict[int | None, Any] = {}
        self._key: bytes | None = None

    def __len__(self) -> int:
        return len(self.shape)

    def then(self, linear: np.ndarray | None, offset: np.ndarray) -> Placement:
        """This placement followed by ``points @ linear + offset``, with
        ``linear`` given as for :class:`Placement`.
        """
        if linear is None:
            return Placement(self.shape, self.linear, _read_only(self.offset + offset))
        moved = (
            self.offset * linear if linear.ndim == 1 else self.offset @ linear
        ) + offset
        if self.linear is None:
            composed = linear
        elif self.linear.ndim == 1 and linear.ndim == 1:
            composed = self.linear * linear
        else:
            composed = _as_matrix(self.linear) @ _as_matrix(linear)
        return Placement(self.shape, _read_only(composed), _read_only(moved))

    @property
    def points(self) -> FrozenPoints:
        """The points, computed on first reading and kept."""
        points = self._points
        if points is None:
            points = adopt(self._apply(np.asarray(self.shape)), FrozenPoints)
            self._points = points
        return points

    @property
    def materialized(self) -> FrozenPoints | None:
        """The points if they were computed, else ``None``."""
        return self._points

    def _apply(self, array: np.ndarray) -> np.ndarray:
        linear = self.linear
        if linear is None:
            return array + self.offset
        if linear.ndim == 1:
            return array * linear + self.offset
        return array @ linear + self.offset

    def bounds(self, anchors_every: int | None = None) -> Any:
        """The least and greatest coordinates of the points, or of the anchors
        taken every ``anchors_every`` points (:func:`anchor_bounds`).
        """
        cache = self._bounds
        if anchors_every in cache:
            return cache[anchors_every]
        found: Any
        if self._points is None and (self.linear is None or self.linear.ndim == 1):
            found = _bounds_of(self.shape, anchors_every)
            if found is not None and found is not UNKNOWN:
                lows, highs = (self._apply(bound[np.newaxis])[0] for bound in found)
                found = (
                    _read_only(np.minimum(lows, highs)),
                    _read_only(np.maximum(lows, highs)),
                )
                if not (np.isfinite(found[0]).all() and np.isfinite(found[1]).all()):
                    found = UNKNOWN
        else:
            found = _bounds_of(self.points, anchors_every)
        cache[anchors_every] = found
        return found

    @property
    def key(self) -> bytes:
        """The content key of the shape and the map, computed once."""
        key = self._key
        if key is None:
            WORK.hashes += 1
            digest = hashlib.blake2b(digest_size=16)
            for array in (self.shape, self.linear, self.offset):
                if array is None:
                    digest.update(b"N")
                else:
                    array = np.ascontiguousarray(array)
                    digest.update(f"A{array.dtype.str}{array.shape}".encode())
                    digest.update(array.tobytes())
            key = self._key = digest.digest()
        return key

    def __copy__(self) -> Placement:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Placement:
        return self

    def __reduce__(self) -> Any:
        return (Placement, (self.shape, self.linear, self.offset))

    def __repr__(self) -> str:
        kind = (
            "translation"
            if self.linear is None
            else ("axis scale" if self.linear.ndim == 1 else "affine")
        )
        return f"Placement({len(self.shape)} points, {kind})"


def _as_matrix(linear: np.ndarray) -> np.ndarray:
    return np.diag(linear) if linear.ndim == 1 else linear


#: The bounds of points that the exact reading can't give: a coordinate that
#: isn't finite, or (for anchors) points that don't split into whole curves,
#: which CE's anchor reading refuses. The caller reads the points as CE did.
UNKNOWN: Any = object()


def point_bounds(points: np.ndarray) -> Any:
    """The least and greatest coordinates of ``points`` (``None`` if it has
    none, :data:`UNKNOWN` if one isn't finite), kept with a read-only value.
    """
    return _bounds_of(points, None)


def anchor_bounds(points: np.ndarray, every: int) -> Any:
    """As :func:`point_bounds`, over the anchors of the curves of ``every``
    points that ``points`` holds (all of them for a single point), as
    :meth:`~.VMobject.get_anchors` reads them.
    """
    return _bounds_of(points, every)


def _bounds_of(points: np.ndarray, anchors_every: int | None) -> Any:
    kept = isinstance(points, FrozenArray) and not points.flags.writeable
    cache = getattr(points, "_bounds", None) if kept else None
    if cache is not None and anchors_every in cache:
        return cache[anchors_every]
    array = np.asarray(points)
    found: Any = None
    if anchors_every is not None and len(array) != 1:
        starts = array[::anchors_every]
        ends = array[anchors_every - 1 :: anchors_every]
        if len(starts) != len(ends):
            found = UNKNOWN
        else:
            array = np.concatenate((starts, ends))
    if found is None and len(array):
        lows, highs = array.min(axis=0), array.max(axis=0)
        if np.isfinite(lows).all() and np.isfinite(highs).all():
            found = (_read_only(lows), _read_only(highs))
        else:
            found = UNKNOWN
    if kept:
        if cache is None:
            cache = {}
            points._bounds = cache
        cache[anchors_every] = found
    return found
