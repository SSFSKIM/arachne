"""The scene clock: exact time, one global frame grid and the 60 Hz tick.

Scene time is a :class:`~fractions.Fraction`. A duration authored as a float
becomes the simplest fraction that rounds to that float, so ``0.1`` is
``1/10`` and ``1/3`` is ``1/3``, and play spans, lags and sections line up
exactly. Frames sample one grid for the whole scene, frame ``k`` at ``k/fps``
of the scene clock, and plays that integrate (dt updaters, recorders) step on
the 60 Hz tick, ``k/60`` of the same clock, whatever the output rate.
"""

from __future__ import annotations

__all__ = [
    "TICK",
    "TICK_DT",
    "TICK_RATE",
    "exact",
    "frames_in",
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


def exact(value: float | Fraction | str) -> Fraction:
    """The exact scene time of ``value``.

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


def _simplest_rounding_to(x: float) -> Fraction:
    """The simplest positive fraction whose nearest float is ``x`` (> 0)."""
    exact_x = Fraction(x)
    below = Fraction(math.nextafter(x, 0.0))
    above = Fraction(math.nextafter(x, math.inf))
    # Every real strictly between the midpoints rounds to x.
    found = _simplest_between((exact_x + below) / 2, (exact_x + above) / 2)
    assert float(found) == x
    return found


def _simplest_between(low: Fraction, high: Fraction) -> Fraction:
    """The rational with the smallest denominator strictly between ``low``
    and ``high`` (``0 <= low < high``), found by continued fractions.
    """
    # Each step peels off the common integer part and inverts the remainder;
    # the convergents are accumulated as h/k.
    h0, h1 = 0, 1
    k0, k1 = 1, 0
    while True:
        whole = math.floor(low)
        if whole + 1 < high:
            # An integer lies strictly inside: the smallest one above low.
            a = whole + 1
            return Fraction(a * h1 + h0, a * k1 + k0)
        h0, h1 = h1, whole * h1 + h0
        k0, k1 = k1, whole * k1 + k0
        low_rest, high_rest = low - whole, high - whole
        if low_rest == 0:
            # Everything above whole and below high: the simplest is whole + 1/n
            # with the smallest n whose 1/n < high_rest.
            a = math.floor(1 / high_rest) + 1
            return Fraction(a * h1 + h0, a * k1 + k0)
        low, high = 1 / high_rest, 1 / low_rest


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
