"""Cubic Bézier paths the marks are drawn from, as point arrays a
:class:`~.VMobject` takes whole: four points per curve, a jump between
two curves starting a new subpath.
"""

from __future__ import annotations

__all__ = [
    "ArcLength",
    "arc",
    "as_point",
    "direction",
    "ellipse_arc",
    "head",
    "hermite",
    "line",
    "normal",
    "polyline",
    "rounded_polyline",
    "rounded_rect",
]

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from arachne.constants import TAU

#: The widest turn one cubic curve of an arc spans.
_ARC_STEP = TAU / 4


def as_point(value: Any) -> np.ndarray:
    """``value`` as a point in 3D: a 2D point gets ``z = 0``."""
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.size == 2:
        point = np.append(point, 0.0)
    if point.size != 3:
        raise ValueError(f"a point has two or three coordinates, not {point.size}")
    return point


def direction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The unit vector from ``start`` to ``end``."""
    course = end - start
    length = float(np.linalg.norm(course))
    if length == 0:
        raise ValueError("a mark between two points needs two different points")
    return course / length


def normal(unit: np.ndarray) -> np.ndarray:
    """``unit`` turned a quarter turn counterclockwise, in the plane."""
    return np.array([-unit[1], unit[0], 0.0])


def line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The straight curve from ``a`` to ``b``, its handles a third of the way
    in from each end, so its parameter runs at a constant speed.
    """
    return np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])


def polyline(points: Sequence[np.ndarray], *, closed: bool = False) -> np.ndarray:
    """Straight curves through ``points`` in order, back to the first when
    ``closed``.
    """
    corners = [as_point(p) for p in points]
    if closed:
        corners.append(corners[0])
    return np.concatenate(
        [line(a, b) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    )


def ellipse_arc(
    center: np.ndarray, rx: float, ry: float, start: float, angle: float
) -> np.ndarray:
    """The arc of the axis-aligned ellipse of radii ``rx`` and ``ry`` about
    ``center``, from the angle ``start`` through ``angle`` (radians,
    counterclockwise when positive), in curves of at most a quarter turn.
    """
    pieces = max(1, math.ceil(abs(angle) / _ARC_STEP - 1e-9))
    step = angle / pieces
    k = 4 / 3 * math.tan(step / 4)
    scale = np.array([rx, ry, 0.0])
    curves = []
    for index in range(pieces):
        a = start + index * step
        b = a + step
        p0 = np.array([math.cos(a), math.sin(a), 0.0])
        p3 = np.array([math.cos(b), math.sin(b), 0.0])
        p1 = p0 + k * np.array([-math.sin(a), math.cos(a), 0.0])
        p2 = p3 - k * np.array([-math.sin(b), math.cos(b), 0.0])
        curves.append(center + np.array([p0, p1, p2, p3]) * scale)
    return np.concatenate(curves)


def arc(center: np.ndarray, radius: float, start: float, angle: float) -> np.ndarray:
    """The circular arc of ``radius`` about ``center`` from the angle
    ``start`` through ``angle``.
    """
    return ellipse_arc(center, radius, radius, start, angle)


def rounded_polyline(points: Sequence[np.ndarray], radius: float) -> np.ndarray:
    """Straight runs through ``points``, each corner filleted with a circular
    arc of ``radius``, or less where a run is too short to hold it (a fillet
    takes at most half of each run it joins).
    """
    corners = [as_point(p) for p in points]
    if len(corners) < 2:
        raise ValueError("a path needs at least two points")
    # Drop repeated points: a zero-length run has no direction.
    kept = [corners[0]]
    for point in corners[1:]:
        if float(np.linalg.norm(point - kept[-1])) > 1e-9:
            kept.append(point)
    corners = kept
    if len(corners) == 1:
        raise ValueError("a path needs two different points")
    curves: list[np.ndarray] = []
    cursor = corners[0]
    for index in range(1, len(corners) - 1):
        before, corner, after = corners[index - 1], corners[index], corners[index + 1]
        u = direction(before, corner)
        v = direction(corner, after)
        turn = math.acos(max(-1.0, min(1.0, float(np.dot(u, v)))))
        if turn < 1e-9 or radius <= 0:
            continue
        reach = radius * math.tan(turn / 2)
        room = min(
            float(np.linalg.norm(corner - before)) / 2,
            float(np.linalg.norm(after - corner)) / 2,
        )
        reach = min(reach, room)
        fillet = reach / math.tan(turn / 2)
        enter, leave = corner - u * reach, corner + v * reach
        if float(np.linalg.norm(enter - cursor)) > 1e-12:
            curves.append(line(cursor, enter))
        handle = 4 / 3 * math.tan(turn / 4) * fillet
        curves.append(np.array([enter, enter + u * handle, leave - v * handle, leave]))
        cursor = leave
    curves.append(line(cursor, corners[-1]))
    return np.concatenate(curves)


def head(tip: np.ndarray, unit: np.ndarray, length: float, width: float) -> np.ndarray:
    """A closed triangle pointing along ``unit`` with its point at ``tip``."""
    base = tip - unit * length
    side = normal(unit) * width / 2
    return polyline([tip, base + side, base - side], closed=True)


def rounded_rect(
    center: np.ndarray, width: float, height: float, radius: float
) -> np.ndarray:
    """The closed outline of a ``width`` by ``height`` rectangle about
    ``center`` with corners of ``radius`` (at most half its shorter side),
    counterclockwise from the right end of its bottom side.
    """
    r = max(0.0, min(radius, width / 2, height / 2))
    w, h = width / 2 - r, height / 2 - r
    corners = [
        (np.array([w, -h, 0.0]), -TAU / 4),
        (np.array([w, h, 0.0]), 0.0),
        (np.array([-w, h, 0.0]), TAU / 4),
        (np.array([-w, -h, 0.0]), TAU / 2),
    ]
    curves = []
    for index, (centre, start) in enumerate(corners):
        arc_start = start
        if r > 0:
            curves.append(arc(center + centre, r, arc_start, TAU / 4))
        here = (
            center
            + centre
            + r * np.array([math.cos(start + TAU / 4), math.sin(start + TAU / 4), 0.0])
        )
        next_centre, next_start = corners[(index + 1) % 4]
        there = (
            center
            + next_centre
            + r * np.array([math.cos(next_start), math.sin(next_start), 0.0])
        )
        if float(np.linalg.norm(there - here)) > 1e-12:
            curves.append(line(here, there))
    return np.concatenate(curves)


def hermite(
    f: Callable[[float], np.ndarray],
    df: Callable[[float], np.ndarray],
    t0: float,
    t1: float,
    pieces: int,
) -> np.ndarray:
    """The curve ``f`` from ``t0`` to ``t1`` in ``pieces`` cubic curves, each
    matching ``f`` and its derivative ``df`` at its two ends.
    """
    curves = []
    step = (t1 - t0) / pieces
    for index in range(pieces):
        a, b = t0 + index * step, t0 + (index + 1) * step
        pa, pb = f(a), f(b)
        curves.append(np.array([pa, pa + df(a) * step / 3, pb - df(b) * step / 3, pb]))
    return np.concatenate(curves)


class ArcLength:
    """A path's points by distance along it: its cubic curves, in order,
    sampled finely enough that the distance between samples is the arc
    length to well under a thousandth of a curve's length.
    """

    #: Samples per cubic curve.
    SAMPLES = 64

    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=float)
        curves = points[: len(points) // 4 * 4].reshape(-1, 4, points.shape[1])
        if not len(curves):
            raise ValueError("a path with no curve has no length")
        t = np.linspace(0.0, 1.0, self.SAMPLES + 1)
        basis = np.stack(
            [(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3], axis=1
        )
        samples = np.einsum("tk,ckd->ctd", basis, curves).reshape(-1, points.shape[1])
        steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        # Between two curves that do not meet the path jumps; a jump is no
        # distance along it.
        steps[self.SAMPLES :: self.SAMPLES + 1] = 0.0
        self._samples = samples
        self._distance = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def length(self) -> float:
        """The path's length."""
        return float(self._distance[-1])

    def point(self, distance: float) -> np.ndarray:
        """The point ``distance`` along the path from its start (clamped to
        its ends).
        """
        d = min(max(distance, 0.0), self.length)
        index = int(np.searchsorted(self._distance, d, side="right")) - 1
        index = min(max(index, 0), len(self._samples) - 2)
        span = self._distance[index + 1] - self._distance[index]
        share = 0.0 if span <= 0 else (d - self._distance[index]) / span
        return self._samples[index] + share * (
            self._samples[index + 1] - self._samples[index]
        )
