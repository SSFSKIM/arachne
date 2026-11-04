"""What an execution of a scene ran and read: the identity a record carries
(:class:`.Identity`).

Its **source** is the scene file and the closure of its import
statements, resolved by path in the scene's directory (:func:`source_files`).
The source digest decides which executions are compared, so it is a
function of the files alone, whatever the process imported before and
whichever producer (a session, a render) made the record. That makes it
deterministic rather than complete: a module the scene reaches only by a
computed name (``importlib.import_module("a" + "b")``) is not in it, and a
change to such a module shows as a divergence that names no input.

Its **inputs** are the files opened for reading while the scene's code is on
the stack, from the moment its script begins to the moment it ends (files
read as the scene file loads, before any recording, are not among them:
the producers load it differently). Both are named by their path relative to
the scene's directory, so that one source staged in two places digests
alike. The files opened are seen by one :func:`sys.addaudithook` hook per
process, which can't be removed, so it is installed once and does nothing
while no recording watches.
"""

from __future__ import annotations

__all__ = ["Inputs", "active", "engine_dirs", "source_digest", "source_files"]

import ast
import hashlib
import importlib.machinery
import os
import sys
import threading
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

from .source import _INTERNAL

#: The recordings watching now, innermost last.
_WATCHING: list[Inputs] = []
_HOOKED = False
#: Set while the hook reads a file to digest it, which it must not see.
_LOCAL = threading.local()


def active() -> bool:
    """Whether a recording watches the files opened now."""
    return bool(_WATCHING)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _WATCHING or event != "open" or getattr(_LOCAL, "busy", False):
        return
    path, mode, flags = args
    if isinstance(path, int) or not _reads(mode, flags):
        return
    _LOCAL.busy = True
    try:
        frame = sys._getframe(1)
        for watcher in tuple(_WATCHING):
            watcher.opened(path, frame)
    finally:
        _LOCAL.busy = False


def _reads(mode: str | None, flags: int) -> bool:
    """Whether a file opened so may be read for what it held before."""
    if isinstance(mode, str):
        return "r" in mode or ("+" in mode and "w" not in mode)
    return (flags & os.O_ACCMODE) != os.O_WRONLY and not flags & os.O_TRUNC


class Inputs:
    """The files a recording's scene opened for reading, with the digest of
    what each held when first opened.

    A file counts when the code that opened it was called from the scene's
    code, from a module under the scene's directory, and not by an import
    (which is the source's to digest), and when it lies outside the
    ``excluded`` directories (the engine's media and caches,
    :func:`engine_dirs`), the engine's package, the standard library and
    the installed packages: those are what the engine and its environment
    read, the same for every scene they run. A directory among them that
    holds the scene's own is not excluded, since it would hold every input.
    """

    def __init__(self, scene_dir: str, excluded: Iterable[str]) -> None:
        self.scene_dir = os.path.realpath(scene_dir)
        self._scene_prefix = self.scene_dir.rstrip(os.sep) + os.sep
        prefixes = (
            *(
                os.path.realpath(directory).rstrip(os.sep) + os.sep
                for directory in excluded
            ),
            *_INTERNAL,
        )
        # A directory holding the scene's would cover its inputs too.
        self._excluded = tuple(
            prefix for prefix in prefixes if not self._scene_prefix.startswith(prefix)
        )
        #: Each file read, by real path, and the digest of what it held.
        self._read: dict[str, str] = {}
        #: Whether code from a file (by code object file name) is the scene's.
        self._scene_code: dict[str, bool] = {}

    def start(self) -> None:
        """Watch the files opened from now on."""
        global _HOOKED
        if self in _WATCHING:
            return
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True
        _WATCHING.append(self)

    def stop(self) -> None:
        """Stop watching; what was seen stays readable."""
        if self in _WATCHING:
            _WATCHING.remove(self)

    def opened(self, path: Any, frame: FrameType | None) -> None:
        """``path`` is being opened for reading, by the code of ``frame``."""
        try:
            real = os.path.realpath(os.fsdecode(os.fspath(path)))
            if real in self._read or real.startswith(self._excluded):
                return
            if not self._by_scene(frame) or not os.path.isfile(real):
                return
            with open(real, "rb") as file:
                self._read[real] = hashlib.file_digest(file, "sha256").hexdigest()
        except (OSError, TypeError, ValueError):
            # Not a path the scene can read; its own open says why.
            return

    def _is_scene_code(self, name: str) -> bool:
        known = self._scene_code.get(name)
        if known is None:
            known = self._scene_code[name] = not name.startswith(
                "<"
            ) and os.path.realpath(name).startswith(self._scene_prefix)
        return known

    def _by_scene(self, frame: FrameType | None) -> bool:
        """Whether the scene's code opens the file through ``frame``: walking
        out from the opener, a frame of the scene's code comes before any
        frame of the import system.
        """
        while frame is not None:
            name = frame.f_code.co_filename
            if self._is_scene_code(name):
                return True
            if name.startswith("<frozen importlib"):
                return False
            frame = frame.f_back
        return False

    def input_digests(self, sources: Collection[str]) -> tuple[tuple[str, str], ...]:
        """Each file read but the ``sources`` (real paths, which the source
        digest covers), as ``(path relative to the scene's directory,
        sha256)``, by path.
        """
        return tuple(
            sorted(
                (_relative(path, self.scene_dir), digest)
                for path, digest in self._read.items()
                if path not in sources
            )
        )


def engine_dirs(config: Any) -> list[str]:
    """The directories the engine writes its media and caches into, under
    ``config``, as far as they resolve without a module or scene name.
    """
    hole = "\0"
    found = []
    for key in (
        "media_dir",
        "video_dir",
        "sections_dir",
        "images_dir",
        "text_dir",
        "tex_dir",
        "partial_movie_dir",
        "log_dir",
    ):
        try:
            directory = config.get_dir(key, module_name=hole, scene_name=hole)
        except KeyError:
            continue
        if directory is None:
            continue
        text = os.fspath(directory).split(hole)[0]
        if text:
            found.append(os.path.realpath(text))
    return found


# The source


def source_files(scene_file: str) -> list[str]:
    """The source of ``scene_file``, as real paths: the file, and every
    module under its directory that an import statement of it names,
    resolved by path in that directory (:func:`_named_files`, whether or not
    the statement runs), and followed the same way. It never asks
    ``sys.modules``: what the process imported before changes nothing. A
    module named only by a computed string is not found.
    """
    scene_file = os.path.realpath(scene_file)
    scene_dir = os.path.dirname(scene_file)
    prefix = scene_dir.rstrip(os.sep) + os.sep
    found = {scene_file}
    todo = [scene_file]
    while todo:
        for path in _named_files(todo.pop(), scene_dir):
            if path not in found and path.startswith(prefix):
                found.add(path)
                todo.append(path)
    return sorted(found)


def source_digest(files: Iterable[str], scene_dir: str) -> str:
    """One sha256 over ``files``: each one's path relative to
    ``scene_dir`` and the sha256 of what it holds, by path.
    """
    entries = []
    for path in files:
        with open(path, "rb") as file:
            entries.append(
                (
                    _relative(path, scene_dir),
                    hashlib.file_digest(file, "sha256").hexdigest(),
                )
            )
    digest = hashlib.sha256()
    for relative, content in sorted(entries):
        digest.update(f"{relative}\0{content}\n".encode())
    return digest.hexdigest()


def _named_files(path: str, scene_dir: str) -> Iterator[str]:
    """The files of the modules the import statements of the source at
    ``path`` name (anywhere in it, whether or not they ran), each dotted
    name's packages included: an absolute name resolved in ``scene_dir``, a
    relative one from ``path``'s own package directory. A call of
    ``importlib.import_module`` or ``__import__`` with a literal absolute
    name counts as an import statement.
    """
    try:
        tree = ast.parse(Path(path).read_bytes(), path)
    except (OSError, SyntaxError, ValueError):
        # Not Python source (a compiled module): its file is digested as is.
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield from _resolve(scene_dir, alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                start = os.path.dirname(path)
                for _ in range(node.level - 1):
                    start = os.path.dirname(start)
            else:
                start = scene_dir
            parts = node.module.split(".") if node.module else []
            yield from _resolve(start, parts)
            for alias in node.names:
                if alias.name != "*":
                    yield from _resolve(start, [*parts, alias.name])
        elif isinstance(node, ast.Call) and _literal_import(node):
            name: str = node.args[0].value  # type: ignore[attr-defined]
            yield from _resolve(scene_dir, name.split("."))


def _resolve(directory: str, parts: list[str]) -> Iterator[str]:
    """The files of the module ``parts`` names from ``directory`` and of the
    packages along the way, as the path finder finds them there: a package
    with an ``__init__``, then an extension, source or bytecode module, then
    a namespace package (a directory, no file); the first part not found
    ends the walk.
    """
    suffixes = (
        *importlib.machinery.EXTENSION_SUFFIXES,
        *importlib.machinery.SOURCE_SUFFIXES,
        *importlib.machinery.BYTECODE_SUFFIXES,
    )
    for part in parts:
        if not part:
            return
        package = os.path.join(directory, part)
        init = os.path.join(package, "__init__.py")
        if os.path.isfile(init):
            yield os.path.realpath(init)
            directory = package
            continue
        module = next(
            (
                package + suffix
                for suffix in suffixes
                if os.path.isfile(package + suffix)
            ),
            None,
        )
        if module is not None:
            yield os.path.realpath(module)
            return
        if not os.path.isdir(package):
            return
        directory = package


def _literal_import(call: ast.Call) -> bool:
    """Whether ``call`` imports a module named by a literal absolute name."""
    func = call.func
    named = (
        isinstance(func, ast.Name) and func.id in ("__import__", "import_module")
    ) or (isinstance(func, ast.Attribute) and func.attr == "import_module")
    return (
        named
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
        and not call.args[0].value.startswith(".")
    )


def _relative(path: str, directory: str) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()
