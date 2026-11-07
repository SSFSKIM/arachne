"""A scene's error, explained so that it carries its fix.

:func:`explain` turns what a scene raised into a :class:`SceneError`: the
error's class and message, unchanged (a message that names its fix, as a
read-only write's or a removed option's does, keeps it), the scene file's
line, the last frames of the scene's own code, where inside the engine it
was raised, and hints for a name the scene got wrong: a name from earlier
Manim, one Arachne removed, ``manim`` itself, or a misspelling, with the
nearest names.

Which frames are the scene's follows :mod:`arachne.timeline.source`:
anything outside this package, the standard library and installed
packages. The report's shape, the table of old names and the nearest-name
hints follow ManimGX's ``cli/scenes.py``, credited in ``NOTICE.md``.
"""

from __future__ import annotations

__all__ = ["Frame", "SceneError", "explain"]

import builtins
import difflib
import linecache
import os
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import FrameType, ModuleType
from typing import TYPE_CHECKING

import arachne
from arachne._removed import OPENGL, OPENGL_PREFIX, REMOVED, RENAMED, Removal
from arachne.timeline.source import _internal, relative_path

if TYPE_CHECKING:
    from arachne.typing import StrPath

#: How many of the scene's own frames an explained error keeps.
FRAMES = 3

_PACKAGE = Path(__file__).resolve().parent
_IN_PACKAGE = str(_PACKAGE) + os.sep

_MANIM_HINT = "Arachne replaces manim: write `from arachne import *`"


@dataclass(frozen=True)
class Frame:
    """A frame of a traceback: its file (relative to the scene file's
    directory when inside it; ``arachne/...`` inside the engine), line,
    function and the line's code.
    """

    path: str
    line: int
    function: str
    code: str | None

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: in {self.function}"


@dataclass(eq=False)
class SceneError(Exception):
    """What a scene raised, explained.

    ``cls`` and ``message`` are the error's own. ``line`` is the innermost
    line of the scene file (``path``) the error went through, or ``None``.
    ``t`` is the scene clock's time when it was raised, when known.
    ``frames`` are the last :data:`FRAMES` frames of the scene's own code,
    innermost last; ``raised_in`` is the innermost engine frame when the
    error was raised inside the engine. ``hints`` say what to write instead.
    """

    cls: str
    message: str
    line: int | None
    path: str
    t: Fraction | float | None
    frames: tuple[Frame, ...]
    raised_in: Frame | None
    hints: tuple[str, ...]

    def report(self) -> str:
        """The error as it went through the scene's code: its frames there,
        where the engine raised it, the error, and the hints.
        """
        lines: list[str] = []
        if self.t is not None:
            lines.append(f"at t = {float(self.t):g} s on the scene's clock")
        for frame in self.frames:
            lines.append(str(frame))
            if frame.code:
                lines.append(f"    {frame.code}")
        if self.raised_in is not None:
            where = self.raised_in
            lines.append(
                f"  (raised in {where.path}:{where.line}, in {where.function})"
            )
        lines.append(f"{self.cls}: {self.message}" if self.message else self.cls)
        lines.extend(f"hint: {hint}" for hint in self.hints)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def explain(
    error: BaseException, scene_file: StrPath, *, t: Fraction | float | None = None
) -> SceneError:
    """``error``, raised by executing the scene file ``scene_file``,
    explained; ``t`` is the scene clock's time when it was raised.
    """
    scene_path = os.path.realpath(scene_file)
    base_dir = os.path.dirname(scene_path)
    trace = list(traceback.walk_tb(error.__traceback__))
    # The scene's frames begin at the scene file's first: those before it
    # are the code that called the scene, whether the engine or not.
    start = next(
        (i for i, (f, _) in enumerate(trace) if _is(_file(f), scene_path)), None
    )
    if start is None:
        # No frame of the scene file: compiling the source raised it, or a
        # helper the engine called. The frames before the engine's first are
        # the caller's; with no engine frame, every frame is kept.
        start = next((i for i, (f, _) in enumerate(trace) if _internal(_file(f))), 0)
    frames = [
        _frame(_file(frame), line, frame.f_code.co_name, base_dir)
        for frame, line in trace[start:]
        if not _internal(_file(frame))
    ]
    line = next(
        (line for frame, line in reversed(trace) if _is(_file(frame), scene_path)),
        None,
    )
    raised_in = _raised_in(trace)
    if isinstance(error, SyntaxError):
        # Compiling the source raised it: its place is the error's own.
        raised_in = None
        if error.filename and error.lineno and not _internal(error.filename):
            frames.append(_frame(error.filename, error.lineno, "<module>", base_dir))
            if _is(error.filename, scene_path):
                line = error.lineno
    return SceneError(
        cls=type(error).__name__,
        message=str(error),
        line=line,
        path=os.fspath(scene_file),
        t=t,
        frames=tuple(frames[-FRAMES:]),
        raised_in=raised_in,
        hints=tuple(_hints(error, trace)),
    )


def _file(frame: FrameType) -> str:
    return frame.f_code.co_filename


def _is(filename: str, real_path: str) -> bool:
    return not filename.startswith("<") and os.path.realpath(filename) == real_path


def _frame(filename: str, line: int, function: str, base_dir: str) -> Frame:
    code = linecache.getline(filename, line).strip() or None
    return Frame(relative_path(filename, base_dir), line, function, code)


def _raised_in(trace: list[tuple[FrameType, int]]) -> Frame | None:
    """The innermost engine frame below the scene's last frame, if any."""
    for frame, line in reversed(trace):
        filename = _file(frame)
        if not _internal(filename):
            return None
        real = os.path.realpath(filename)
        if real.startswith(_IN_PACKAGE):
            path = Path(real).relative_to(_PACKAGE.parent).as_posix()
            code = linecache.getline(filename, line).strip() or None
            return Frame(path, line, frame.f_code.co_name, code)
    return None


def _hints(error: BaseException, trace: list[tuple[FrameType, int]]) -> list[str]:
    """What to write instead, for a name the scene got wrong. An error whose
    message is not Python's own for a missing name gets none: it says what
    is wrong already.
    """
    if isinstance(error, ModuleNotFoundError):
        return _missing_module(error.name)
    if isinstance(error, ImportError):
        name = getattr(error, "name_from", None)
        if name and error.name and _is_arachne(error.name):
            return _missing_from(error.name, name)
        return []
    if isinstance(error, NameError) and error.name:
        if f"name {error.name!r} is not defined" not in str(error):
            return []
        namespace: set[str] = set(dir(builtins))
        if trace:
            frame = trace[-1][0]
            namespace.update(frame.f_globals, frame.f_locals)
        return _missing_name(error.name, namespace)
    if isinstance(error, AttributeError) and error.name and error.obj is not None:
        if f"has no attribute {error.name!r}" not in str(error):
            return []
        owner = error.obj
        if isinstance(owner, ModuleType) and _is_arachne(owner.__name__):
            return _missing_from(owner.__name__, error.name)
        return _nearest(error.name, dir(owner))
    return []


def _is_arachne(module: str) -> bool:
    return module == "arachne" or module.startswith("arachne.")


def _ce(module: str) -> str:
    return "manim" + module[len("arachne") :]


def _missing_module(name: str | None) -> list[str]:
    if name is None:
        return []
    if name == "manim" or name.startswith("manim."):
        return [_MANIM_HINT]
    if name.startswith("arachne."):
        ce = _ce(name)
        for removed, removal in REMOVED.items():
            if ":" not in removed and (ce == removed or ce.startswith(removed + ".")):
                return [_removed(f"arachne{removed[len('manim') :]}", removal)]
    return []


def _missing_from(module: str, name: str) -> list[str]:
    """Hints for ``name``, which ``module`` (``arachne`` or under it) lacks."""
    removal = _removal(module, name)
    if removal is not None:
        return [_removed(name, removal)]
    if module == "arachne" and name in RENAMED:
        return [_renamed(name)]
    owner = sys.modules.get(module)
    return _nearest(name, dir(owner)) if owner is not None else []


def _missing_name(name: str, namespace: set[str]) -> list[str]:
    removal = _removal("arachne", name)
    if removal is not None:
        return [_removed(name, removal)]
    if name in RENAMED:
        return [_renamed(name)]
    # Arachne's own modules are no names a scene means to import.
    if (
        not name.startswith("_")
        and hasattr(arachne, name)
        and not isinstance(getattr(arachne, name), ModuleType)
    ):
        return [f"{name} is arachne.{name}: write `from arachne import *`"]
    return _nearest(name, namespace)


def _removal(module: str, name: str) -> Removal | None:
    removal = REMOVED.get(f"{_ce(module)}:{name}")
    if removal is None and module == "arachne":
        removal = REMOVED.get(f"*:{name}")
    if removal is None and name.startswith(OPENGL_PREFIX) and name != OPENGL_PREFIX:
        cairo = name[len(OPENGL_PREFIX) :]
        removal = Removal(OPENGL, cairo if hasattr(arachne, cairo) else None)
    return removal


def _removed(name: str, removal: Removal) -> str:
    hint = f"{name} {removal.by.removed_with}"
    return hint + (f": use {removal.instead}" if removal.instead else "")


def _renamed(name: str) -> str:
    return f"{name} is {RENAMED[name].text}"


def _nearest(name: str, candidates: Iterable[str]) -> list[str]:
    private = name.startswith("_")
    pool = sorted(
        {
            candidate
            for candidate in candidates
            if candidate != name and candidate.startswith("_") == private
        }
    )
    close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
    return [f"did you mean {' or '.join(close)}?"] if close else []
