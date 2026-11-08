"""Where each mobject was made: its creating source line and stable id.

While a recording runs, every mobject is tagged when it is created (copies
included) with the first frame of the scene's own code on the stack, and
with its ordinal among the mobjects made at that line. ``path:line#n`` is
then the same in two executions of the same script, so it serves as the
mobject's stable id. It is also tagged with the line of the innermost frame
in the scene file itself, which differs from the first when a helper module
made it.

The tags live in a table beside the objects, not in their ``__dict__``, so a
tag never reaches a mobject's hash, copy or recorded state.
"""

from __future__ import annotations

__all__ = ["Origin", "ORIGINS", "origin_of"]

import os
import sys
import weakref
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .source import SourceLine, scene_file_line, scene_location


@dataclass(frozen=True)
class Origin:
    """A mobject's creating source line and its stable id.

    ``scene_line`` is the line of the innermost frame in the scene file
    when it was made: ``source_line``'s own when the scene file made it, the
    call that led to a helper module otherwise; ``None`` when no frame was
    the scene file's.
    """

    id: str
    source_line: SourceLine | None
    scene_line: int | None = None


class _Origins:
    """The table of origins, keyed by object identity, and the switch that
    makes creation tag objects.
    """

    def __init__(self) -> None:
        self.active = False
        self.base_dir: str | None = None
        #: The scene file's real path, whose innermost frame is a scene line.
        self.scene_file: str | None = None
        self._entries: dict[int, tuple[weakref.ref[Any], Origin]] = {}
        self._ordinals: dict[str, int] = {}
        #: The lists :meth:`counting` hands out, open now.
        self._counts: list[list[tuple[object, Origin]]] = []

    def start(self, base_dir: str | None, scene_file: str | None = None) -> None:
        """Begin tagging, with source paths relative to ``base_dir`` and
        scene lines in ``scene_file``, and number every line's objects from
        zero again.
        """
        self.active = True
        self.base_dir = base_dir
        self.scene_file = _real(scene_file)
        self._entries.clear()
        self._ordinals.clear()

    def stop(self) -> None:
        """Stop tagging; the origins given so far stay readable."""
        self.active = False

    @contextmanager
    def resumed(
        self,
        base_dir: str | None,
        ordinals: Mapping[str, int],
        scene_file: str | None = None,
    ) -> Iterator[None]:
        """Tag what is created inside the block, relative to ``base_dir``
        and with scene lines in ``scene_file``, numbering each line on from
        ``ordinals`` (by ``path:line``, the next ordinal); afterwards tag and
        number as before: a block run again gives the objects it creates the
        same ids.
        """
        saved = (self.active, self.base_dir, self.scene_file, dict(self._ordinals))
        self.active, self.base_dir = True, base_dir
        self.scene_file = _real(scene_file)
        self._ordinals.clear()
        self._ordinals.update(ordinals)
        try:
            yield
        finally:
            self.active, self.base_dir, self.scene_file, ordinals = saved
            self._ordinals.clear()
            self._ordinals.update(ordinals)

    @contextmanager
    def kept(self) -> Iterator[None]:
        """Whatever the block does to the table (another execution starts
        and stops tagging), afterwards it holds what it held before: the
        switch, the base directory, every origin of an object still alive,
        the scene file and the numbering.
        """
        saved = (
            self.active,
            self.base_dir,
            self.scene_file,
            dict(self._entries),
            dict(self._ordinals),
        )
        try:
            yield
        finally:
            self.active, self.base_dir, self.scene_file, entries, ordinals = saved
            self._entries.clear()
            self._entries.update(
                (key, entry) for key, entry in entries.items() if entry[0]() is not None
            )
            self._ordinals.clear()
            self._ordinals.update(ordinals)

    def tag(self, obj: object) -> None:
        """Give ``obj`` the origin of the scene code now creating it."""
        frame = sys._getframe(2)
        line = scene_location(frame, self.base_dir)
        scene_line = (
            scene_file_line(frame, self.scene_file)
            if self.scene_file is not None
            else None
        )
        self.assign(obj, line, scene_line)

    def assign(
        self, obj: object, line: SourceLine | None, scene_line: int | None = None
    ) -> Origin:
        """Give ``obj`` the next stable id at ``line``, and ``scene_line``,
        and return its origin.
        """
        where = str(line) if line is not None else "?"
        ordinal = self._ordinals.get(where, 0)
        self._ordinals[where] = ordinal + 1
        origin = Origin(f"{where}#{ordinal}", line, scene_line)
        self._enter(obj, origin)
        for made in self._counts:
            made.append((obj, origin))
        return origin

    def _enter(self, obj: object, origin: Origin) -> None:
        key = id(obj)
        entries = self._entries

        def forget(_: object, key: int = key) -> None:
            entry = entries.get(key)
            if entry is not None and entry[0]() is None:
                del entries[key]

        entries[key] = (weakref.ref(obj, forget), origin)

    @contextmanager
    def counting(self) -> Iterator[list[tuple[object, Origin]]]:
        """A list of every object given an origin inside the block, with its
        origin, in order.
        """
        made: list[tuple[object, Origin]] = []
        self._counts.append(made)
        try:
            yield made
        finally:
            # Blocks nest: the list open last closes first.
            self._counts.pop()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Tag nothing created inside the block."""
        active, self.active = self.active, False
        try:
            yield
        finally:
            self.active = active

    def place(
        self, root: object, placed: Iterable[tuple[object, int]], count: int
    ) -> None:
        """Number objects as a construction of ``root`` numbered what it made:
        ``count`` objects at ``root``'s line right after ``root``, of which
        each of ``placed`` is the one at its offset among them; the line is
        numbered on after all ``count``. Nothing happens when ``root`` has no
        origin.
        """
        origin = self.get(root)
        if origin is None:
            return
        where, _, ordinal = origin.id.rpartition("#")
        first = int(ordinal) + 1
        for obj, offset in placed:
            self._enter(
                obj,
                Origin(
                    f"{where}#{first + offset}", origin.source_line, origin.scene_line
                ),
            )
        self._ordinals[where] = first + count

    def get(self, obj: object) -> Origin | None:
        """The origin ``obj`` was given, if any."""
        entry = self._entries.get(id(obj))
        if entry is None or entry[0]() is not obj:
            return None
        return entry[1]


def _real(path: str | None) -> str | None:
    return os.path.realpath(path) if path is not None else None


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
