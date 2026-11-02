"""Source lines and names from the scene's own code.

A play's source line, and a mobject's creating line, is the innermost frame
on the stack that belongs to the scene's code: the first one outside this
package, the standard library and installed packages. Names come from the
local variables of those frames and the scene's attributes, with indices
into groups three deep, so a finding can say "``title[2]``, the ``Text``
from line 42".

The stack walk out of the package follows ManimCommunity/manim PR #5006's
``_location()``, and naming by frame locals follows ManimGX's diagnostics;
both are credited in ``NOTICE.md``.
"""

from __future__ import annotations

__all__ = [
    "SourceLine",
    "names_from_frames",
    "relative_path",
    "scene_frames",
    "scene_location",
]

import os
import sysconfig
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any


@dataclass(frozen=True, order=True)
class SourceLine:
    """A line of the scene's code: its file (relative to the scene file's
    directory when it is inside it) and its line number.
    """

    path: str
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}"


def _internal_prefixes() -> tuple[str, ...]:
    package = str(Path(__file__).resolve().parent.parent)
    paths = sysconfig.get_paths()
    dirs = {package}
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(name):
            dirs.add(str(Path(paths[name]).resolve()))
    return tuple(directory.rstrip(os.sep) + os.sep for directory in dirs)


_INTERNAL = _internal_prefixes()
_IS_INTERNAL: dict[str, bool] = {}


def _internal(filename: str) -> bool:
    """Whether code from ``filename`` is the engine's or a library's rather
    than the scene's.
    """
    known = _IS_INTERNAL.get(filename)
    if known is None:
        if filename.startswith("<"):
            # Code compiled from a string names no file; it is the scene's only
            # when it is the scene (exec'd scene code keeps its own name).
            known = filename.startswith("<frozen")
        else:
            resolved = os.path.realpath(filename)
            known = resolved.startswith(_INTERNAL)
        _IS_INTERNAL[filename] = known
    return known


def scene_frames(frame: FrameType | None) -> Iterator[FrameType]:
    """The frames of the scene's own code from ``frame`` outwards."""
    while frame is not None:
        if not _internal(frame.f_code.co_filename):
            yield frame
        frame = frame.f_back


def scene_location(frame: FrameType | None, base_dir: str | None) -> SourceLine | None:
    """The source line of the innermost scene frame from ``frame`` outwards,
    or ``None`` when no frame of the stack is the scene's or it was compiled
    from a string.
    """
    for scene_frame in scene_frames(frame):
        filename = scene_frame.f_code.co_filename
        if filename.startswith("<"):
            return None
        return SourceLine(relative_path(filename, base_dir), scene_frame.f_lineno)
    return None


def relative_path(filename: str, base_dir: str | None) -> str:
    """``filename`` relative to ``base_dir`` when it is inside it."""
    if base_dir is not None:
        try:
            return Path(filename).resolve().relative_to(base_dir).as_posix()
        except ValueError:
            pass
    return Path(filename).as_posix()


#: How deep names reach into groups: ``name[i][j][k]``.
NAME_DEPTH = 3

# The scene's own bookkeeping lists name nothing a scene author wrote.
_SCENE_LISTS = frozenset(
    {"mobjects", "foreground_mobjects", "moving_mobjects", "static_mobjects"}
)


def names_from_frames(frames: list[FrameType], scene: Any) -> dict[int, str]:
    """Names for the mobjects the scene's code can reach, by object id.

    The innermost frame's locals come first, then the outer frames', then the
    scene's attributes as ``self.<name>``; a plain name wins over an index
    into a group, and the first name found for an object is kept.
    """
    from arachne.mobject.mobject import Mobject

    names: dict[int, str] = {}
    pending: list[tuple[str, Any]] = []

    def visit(name: str, value: Any, depth: int) -> None:
        if isinstance(value, Mobject):
            names.setdefault(id(value), name)
            if depth < NAME_DEPTH:
                for index, child in enumerate(value.submobjects):
                    pending.append((f"{name}[{index}]", (child, depth + 1)))
        elif isinstance(value, (list, tuple)) and depth < NAME_DEPTH:
            for index, item in enumerate(value):
                if isinstance(item, Mobject):
                    pending.append((f"{name}[{index}]", (item, depth + 1)))

    roots: list[tuple[str, Any]] = []
    for frame in frames:
        for name, value in list(frame.f_locals.items()):
            if value is not scene and not name.startswith("__"):
                roots.append((name, value))
    for name, value in list(vars(scene).items()):
        if not name.startswith("_") and name not in _SCENE_LISTS:
            roots.append((f"self.{name}", value))
    for name, value in roots:
        visit(name, value, 0)
    # Indices are named only after every plain name had its chance.
    while pending:
        batch, pending = pending, []
        for name, (value, depth) in batch:
            visit(name, value, depth)
    return names
