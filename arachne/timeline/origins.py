"""Where each mobject was made: its creating source line and stable id.

While a recording runs, every mobject is tagged when it is created (copies
included) with the first frame of the scene's own code on the stack, and
with its ordinal among the mobjects made at that line. ``path:line#n`` is
then the same in two executions of the same script, so it serves as the
mobject's stable id.

The tags live in a table beside the objects, not in their ``__dict__``, so a
tag never reaches a mobject's hash, copy or recorded state.
"""

from __future__ import annotations

__all__ = ["Origin", "ORIGINS", "origin_of"]

import sys
import weakref
from dataclasses import dataclass
from typing import Any

from .source import SourceLine, scene_location


@dataclass(frozen=True)
class Origin:
    """A mobject's creating source line and its stable id."""

    id: str
    source_line: SourceLine | None


class _Origins:
    """The table of origins, keyed by object identity, and the switch that
    makes creation tag objects.
    """

    def __init__(self) -> None:
        self.active = False
        self.base_dir: str | None = None
        self._entries: dict[int, tuple[weakref.ref[Any], Origin]] = {}
        self._ordinals: dict[str, int] = {}

    def start(self, base_dir: str | None) -> None:
        """Begin tagging, with source paths relative to ``base_dir``, and
        number every line's objects from zero again.
        """
        self.active = True
        self.base_dir = base_dir
        self._entries.clear()
        self._ordinals.clear()

    def stop(self) -> None:
        """Stop tagging; the origins given so far stay readable."""
        self.active = False

    def tag(self, obj: object) -> None:
        """Give ``obj`` the origin of the scene code now creating it."""
        line = scene_location(sys._getframe(2), self.base_dir)
        self.assign(obj, line)

    def assign(self, obj: object, line: SourceLine | None) -> Origin:
        """Give ``obj`` the next stable id at ``line`` and return its origin."""
        where = str(line) if line is not None else "?"
        ordinal = self._ordinals.get(where, 0)
        self._ordinals[where] = ordinal + 1
        origin = Origin(f"{where}#{ordinal}", line)
        key = id(obj)
        entries = self._entries

        def forget(_: object, key: int = key) -> None:
            entry = entries.get(key)
            if entry is not None and entry[0]() is None:
                del entries[key]

        entries[key] = (weakref.ref(obj, forget), origin)
        return origin

    def get(self, obj: object) -> Origin | None:
        """The origin ``obj`` was given, if any."""
        entry = self._entries.get(id(obj))
        if entry is None or entry[0]() is not obj:
            return None
        return entry[1]


#: The process's origin table.
ORIGINS = _Origins()


def origin_of(obj: object) -> Origin:
    """``obj``'s origin, giving it one now (with no source line) if it was
    made while nothing was tagging.
    """
    found = ORIGINS.get(obj)
    if found is None:
        found = ORIGINS.assign(obj, None)
    return found
