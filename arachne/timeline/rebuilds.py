"""Which mobjects ``become`` rebuilt, and when.

``become`` gives a mobject's family another's values position by position,
so a rebuilt group's children each take whatever item the new list puts at
their place: a list that gains or loses an item hands its children each
other's places. The motion rule reads that as no jump
(:mod:`arachne.lint.motion`), so it needs to know that ``become`` rebuilt a
group, and between which moments of the scene.

Every call is numbered, once per process and in order, and each mobject
keeps the numbers of the calls that rebuilt it; a recorder takes the count
at the moments it marks (each play's call and the state it began in), so a
call between two of them is told by its number alone.
"""

from __future__ import annotations

__all__ = ["count", "note", "rebuilt_between"]

import weakref
from typing import Any

_COUNT = 0
#: Each mobject ``become`` rebuilt, with the numbers of those calls in order.
_CALLS: weakref.WeakKeyDictionary[Any, list[int]] = weakref.WeakKeyDictionary()


def note(mob: Any) -> None:
    """Number a ``become`` call that rebuilt ``mob``."""
    global _COUNT
    _COUNT += 1
    _CALLS.setdefault(mob, []).append(_COUNT)


def count() -> int:
    """How many ``become`` calls the process has made so far."""
    return _COUNT


def rebuilt_between(mob: Any, after: int, through: int) -> bool:
    """Whether a ``become`` call numbered past ``after`` and at most
    ``through`` (each a :func:`count`) rebuilt ``mob``.
    """
    return any(after < call <= through for call in _CALLS.get(mob, ()))
