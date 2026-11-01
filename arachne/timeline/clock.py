"""The scene clock: exact time, one global frame grid and the 60 Hz tick.

Scene time is a :class:`~fractions.Fraction`. A play or wait that starts at
exact ``T`` with a float duration ``d`` ends at the simplest fraction within
4 ulps of ``float(T) + d`` (:func:`play_end`, the end-window rule), so ``0.1``
from 0 ends at ``1/10``, ``0.2`` after it at ``3/10``, and a wait sized from
the float clock (``cue(t)``: ``wait(t - renderer.time)``) ends at ``t``;
play spans, lags and sections line up exactly. Frames sample one grid for
the whole scene, frame ``k`` at ``k/fps`` of the scene clock, and plays that
integrate (dt updaters, recorders) step on the 60 Hz tick, ``k/60`` of the
same clock, whatever the output rate.
"""

from __future__ import annotations

__all__ = [
    "END_WINDOW_ULPS",
    "TICK",
    "TICK_DT",
    "TICK_RATE",
    "exact",
    "frames_in",
    "play_end",
    "ticks_in",
]

import math
from collections.abc import Iterator
from fractions import Fraction
from numbers import Integral

#: The integration rate, in ticks per second of scene time.
TICK_RATE = 60
#: One tick, exactly.
TICK = Fraction(1, TICK_RATE)
#: The ``dt`` a dt updater receives at every tick.
TICK_DT = float(TICK)
#: How far, in ulps of the float end, a play's end may move to a simpler fraction.
END_WINDOW_ULPS = 4


def exact(value: float | Fraction | str) -> Fraction:
    """The exact value of ``value``, a duration or an instant.

    A float becomes the simplest fraction that rounds to it: the one with the
    smallest denominator among the rationals whose nearest float is
    ``value``. Integers and fractions are kept as they are.

    Examples
    --------
    >>> exact(0.1)
    Fraction(1, 10)
    >>> exact(1 / 3)
    Fraction(1, 3)
    >>> exact(2)
    Fraction(2, 1)
    >>> exact(0.1 + 0.2) == Fraction(3, 10)
    False
    >>> float(exact(0.1 + 0.2)) == 0.1 + 0.2
    True
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return Fraction(int(value))
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"A scene time must be finite, not {x!r}.")
    if x == 0 or x.is_integer():
        return Fraction(int(x))
    if x < 0:
        return -_simplest_rounding_to(-x)
    return _simplest_rounding_to(x)


def play_end(start: Fraction, duration: float | Fraction) -> Fraction:
    """Where a play or wait from ``start`` that lasts ``duration`` ends.

    An int or :class:`~fractions.Fraction` duration is used as given. A float
    one ends the play at the simplest fraction within
    :data:`END_WINDOW_ULPS` ulps of ``float(start) + duration`` (the ulp of
    that float), never at or before ``start``; a duration of 0 ends it at
    ``start``. The span is the exact difference.

    Snapping the duration alone leaves a wait sized from the float clock a
    rounding error off its target, and snapping the float end alone turns
    ``0.2`` after ``1/10`` into a 15-digit end; the window does neither, since
    two fractions ``a/b`` and ``p/q`` differ by at least ``1/(bq)``.

    Examples
    --------
    >>> play_end(Fraction(1, 10), 0.2)
    Fraction(3, 10)
    >>> play_end(Fraction(2, 3), 2.5 - float(Fraction(2, 3)))
    Fraction(5, 2)
    >>> play_end(Fraction(1, 7), Fraction(1, 3))
    Fraction(10, 21)
    """
    if isinstance(duration, Fraction) or (
        isinstance(duration, Integral) and not isinstance(duration, bool)
    ):
        return start + duration
    d = float(duration)
    if not math.isfinite(d):
        raise ValueError(f"A duration must be finite, not {d!r}.")
    if d == 0:
        return start
    end = float(start) + d
    ulp = Fraction(math.ulp(end))
    centre = Fraction(end)
    low = max(centre - END_WINDOW_ULPS * ulp, start)
    high = centre + END_WINDOW_ULPS * ulp
    found = _simplest_in(low, high)
    if found == start:
        found = _simplest_in(start + (high - start) / 2, high)
    return found


def _simplest_rounding_to(x: float) -> Fraction:
    """The simplest positive fraction whose nearest float is ``x`` (> 0)."""
    exact_x = Fraction(x)
    low = (exact_x + Fraction(math.nextafter(x, 0.0))) / 2
    high = (exact_x + Fraction(math.nextafter(x, math.inf))) / 2
    found = _simplest_in(low, high)
    if float(found) != x:
        # A tie on an edge of the interval rounds to the other float.
        margin = (high - low) / 10**6
        found = _simplest_in(low + margin, high - margin)
    return found


def _simplest_in(low: Fraction, high: Fraction) -> Fraction:
    """The rational with the smallest denominator in ``[low, high]``
    (``0 <= low <= high``), by continued fractions.
    """
    whole = math.floor(low)
    if whole == low:
        return Fraction(whole)
    if whole + 1 <= high:
        return Fraction(whole + 1)
    return whole + 1 / _simplest_in(1 / (high - whole), 1 / (low - whole))


def ticks_in(start: Fraction, end: Fraction) -> Iterator[Fraction]:
    """The ticks a play over ``[start, end]`` steps: every ``k/60`` with
    ``start < k/60 <= end``.

    A tick belongs to the play that ends at or after it, so every tick of the
    scene is stepped exactly once.
    """
    first = math.floor(start * TICK_RATE) + 1
    last = math.floor(end * TICK_RATE)
    for k in range(first, last + 1):
        yield Fraction(k, TICK_RATE)


def frames_in(start: Fraction, end: Fraction, fps: Fraction) -> list[Fraction]:
    """The instants of the global frame grid at ``fps`` in ``[start, end)``."""
    first = math.ceil(start * fps)
    stop = math.ceil(end * fps)
    return [Fraction(k) / fps for k in range(first, stop)]
