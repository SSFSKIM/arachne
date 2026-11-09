"""A path's miter joins decided as Cairo decides them, from CE's own points.

CE strokes a path through Cairo: its points as the camera places them, in
frame units, through the context's matrix (``Camera.make_cairo_context``),
each rounded to Cairo's 24.8 fixed point. Cairo's stroker decides each join
from slopes between those rounded points: where a control arm is a few
1/256 px long, or the turn is near the miter limit, a drawer handed the
points snapped to the engine's grid (:mod:`.canonical`) cannot tell from them
what Cairo decided. This module restates, in Python's IEEE doubles and
integers, the arithmetic Cairo 1.18 applies, never calling Cairo:

- ``cairo_matrix_transform_point``: ``x' = (xx x + xy y) + x0`` (and so for
  ``y``), the context's matrix being CE's;
- ``_cairo_fixed_from_double``: the nearest 1/256, ties to even;
- ``_cairo_spline_init``: a curve leaves along its first control point that
  is not at its start (else its end), and arrives from its last control point
  that is not at its end; a curve whose four points are one is no element,
  and one with no arriving slope but from its start is stroked as a line;
- ``_cairo_slope_init`` and ``normalize_slope`` (the polygon stroker,
  ``cairo-path-stroke-polygon.c``): a slope is the difference of the fixed
  points, normalized by its length (exactly, when it lies on an axis);
- ``outer_join`` and ``outer_close``: a join is mitred where
  ``2 <= ml² (1 + in · out)``, ``ml`` the miter limit, CE's 10; faces leaving
  in one direction make no join; ``close_path`` strokes the closing line, then
  joins it to the first face.

:func:`joins` gives, per cubic of a path's points, whether Cairo joins into
it and whether it mitres there (for the first cubic of a closed subpath: the
closing join), the record's ``joins`` (XF 1.4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: The miter limit CE strokes with (Cairo's default; CE never changes it).
MITER_LIMIT = 10.0
#: A path whose joins a drawer handed its snapped points may decide otherwise
#: than Cairo: one with a join turning past this many degrees, or arriving or
#: leaving along a slope shorter than :data:`SHORT` (in 1/256 px). Elsewhere
#: the snap (at most a few thousandths of a pixel) cannot move the decision.
SHARP = 160.0
SHORT = 16

#: CE's ``tolerance_for_point_equality`` and the relative tolerance of its
#: ``consider_points_equals_2d``.
_ATOL = 1e-6
_RTOL = 1e-5

#: A join byte: Cairo joins into the cubic; it mitres there.
JOINED = 1
MITRED = 2


def _close(a: float, b: float) -> bool:
    """numpy's ``isclose(a, b)`` with CE's tolerances, for one coordinate."""
    return a == b or (abs(a - b) <= _ATOL + _RTOL * abs(b) and math.isfinite(a - b))


def _subpaths(points: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """CE's subpath ranges over ``points`` (cubic layout), compared in two
    dimensions (``get_subpath_split_indices_from_points``).
    """
    n = len(points)
    if n < 4:
        return []
    out, start = [], 0
    for b in range(4, n, 4):
        end, nxt = points[b - 1], points[b]
        if not (_close(end[0], nxt[0]) and _close(end[1], nxt[1])):
            out.append((start, b))
            start = b
    out.append((start, n))
    return out


def matrix(view: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """The context matrix CE strokes a path through (``make_cairo_context``,
    about the view's mapping centre): ``(xx, yy, x0, y0)``.
    """
    pw, ph = view["pixel_width"], view["pixel_height"]
    fw, fh = view["frame_width"], view["frame_height"]
    fc = view.get("mapping_centre") or view.get("frame_center") or (0.0, 0.0, 0.0)
    return (
        (pw / fw),
        -(ph / fh),
        (pw / 2) - fc[0] * (pw / fw),
        (ph / 2) + fc[1] * (ph / fh),
    )


def _fixed(m: tuple[float, float, float, float], p: Sequence[float]) -> tuple[int, int]:
    xx, yy, x0, y0 = m
    x, y = float(p[0]), float(p[1])
    # round() on a double is to the nearest integer, ties to even.
    return (
        round(((xx * x + 0.0 * y) + x0) * 256.0),
        round(((0.0 * x + yy * y) + y0) * 256.0),
    )


def _unit(s: tuple[int, int]) -> tuple[float, float]:
    dx, dy = s[0] / 256.0, s[1] / 256.0
    if dx == 0.0:
        return (0.0, 1.0 if dy > 0.0 else -1.0)
    if dy == 0.0:
        return (1.0 if dx > 0.0 else -1.0, 0.0)
    mag = math.hypot(dx, dy)
    return (dx / mag, dy / mag)


def mitres(slope_in: tuple[int, int], slope_out: tuple[int, int]) -> bool:
    """Whether Cairo mitres a join of a face leaving along ``slope_in`` to one
    leaving along ``slope_out`` (24.8 slopes).
    """
    a, b = _unit(slope_in), _unit(slope_out)
    return MITER_LIMIT * MITER_LIMIT * (1 + (a[0] * b[0] + a[1] * b[1])) >= 2


def _one_way(s: tuple[int, int], t: tuple[int, int]) -> bool:
    """Whether two slopes leave in one direction (``_cairo_slope_compare``
    equal: no join).
    """
    return s[0] * t[1] == s[1] * t[0] and s[0] * t[0] >= 0 and s[1] * t[1] >= 0


def _turn(s: tuple[int, int], t: tuple[int, int]) -> float:
    a, b = _unit(s), _unit(t)
    return math.degrees(math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))))


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def joins(
    points: Sequence[Sequence[float]], view: Mapping[str, Any]
) -> tuple[int, ...] | None:
    """Per cubic of ``points`` (CE's cubic layout, in frame units, as the
    camera places them), Cairo's join into it: :data:`JOINED`, with
    :data:`MITRED` where it mitres; for the first cubic of a closed subpath, the
    closing join (none where Cairo strokes a closing line first). ``None`` for
    a path whose joins a drawer cannot decide otherwise (see :data:`SHARP`).
    """
    if not all(math.isfinite(float(c)) for p in points for c in p[:2]):
        return None  # a path the drawer omits
    m = matrix(view)
    n = len(points) // 4
    out = [0] * n
    wanted = False
    for start, end in _subpaths(points):
        if end - start < 4:
            continue
        a0, z = points[start], points[end - 1]
        closed = _close(a0[0], z[0]) and _close(a0[1], z[1])
        first_f = cur = _fixed(m, a0)
        last: tuple[int, int] | None = None
        first: tuple[int, int] | None = None

        def join(into: int, s: tuple[int, int], t: tuple[int, int]) -> None:
            nonlocal wanted
            if _one_way(s, t):
                return
            out[into] = JOINED | (MITRED if mitres(s, t) else 0)
            if (
                _turn(s, t) > SHARP
                or min(max(abs(s[0]), abs(s[1])), max(abs(t[0]), abs(t[1]))) < SHORT
            ):
                wanted = True

        for i in range(start, end - 3, 4):
            b, c, d = (_fixed(m, points[i + k]) for k in (1, 2, 3))
            a = cur
            init = next((_sub(q, a) for q in (b, c, d) if q != a), None)
            fin = _sub(d, c) if c != d else (_sub(d, b) if b != d else None)
            if init is None:
                continue  # its points are one: no element
            if fin is None:
                init = fin = _sub(d, a)  # stroked as a line
            if last is not None:
                join(i // 4, last, init)
            if first is None:
                first = init
            last, cur = fin, d
        if closed and last is not None and first is not None and cur == first_f:
            join(start // 4, last, first)
    return tuple(out) if wanted else None
