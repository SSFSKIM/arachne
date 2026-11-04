"""What an execution of a scene ran and read: the identity a record carries
(:class:`.Identity`).

Its **source** is the scene file and every module under the scene's
directory that the scene's code imports, followed from the scene file's
import statements through the modules they name, as each was imported
(:func:`source_files`). Its **inputs** are the files opened for reading
while the scene's code is on the stack (:class:`Inputs`). Both are named by
their path relative to the scene's directory, so that one source staged in
two places digests alike.

The files opened are seen by one :func:`sys.addaudithook` hook per process.
An audit hook can't be removed, so it is installed once, when a recording
first watches, and does nothing unless one watches: a recording watches from
the moment its script begins to the moment it ends.
"""

from __future__ import annotations

__all__ = ["Inputs", "active", "engine_dirs", "source_digest", "source_files"]

import ast
import hashlib
import importlib.util
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
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True
        _WATCHING.append(self)

    def stop(self) -> None:
        """Stop watching; reading :meth:`input_digests` still works."""
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

    def _by_scene(self, frame: FrameType | None) -> bool:
        """Whether the scene's code opens the file through ``frame``: walking
        out from the opener, a frame of the scene's code comes before any
        frame of the import system.
        """
        known = self._scene_code
        while frame is not None:
            name = frame.f_code.co_filename
            scene = known.get(name)
            if scene is None:
                scene = known[name] = not name.startswith("<") and os.path.realpath(
                    name
                ).startswith(self._scene_prefix)
            if scene:
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


def source_files(scene_file: str, package: str | None) -> list[str]:
    """The scene file and the modules under its directory that its code
    imports, as real paths: each module an ``import`` statement anywhere in
    the file names (or an ``importlib.import_module`` or ``__import__`` call
    with a literal name), as ``sys.modules`` holds it, then the same for
    that module's own file. A module never imported is not there; a module
    imported only under a computed name is not found.
    """
    scene_file = os.path.realpath(scene_file)
    prefix = os.path.dirname(scene_file).rstrip(os.sep) + os.sep
    found = {scene_file}
    todo = [(scene_file, package)]
    while todo:
        path, package = todo.pop()
        for name in _imported_names(path, package):
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None)
            if not file:
                continue
            real = os.path.realpath(file)
            if real in found or not real.startswith(prefix):
                continue
            found.add(real)
            todo.append((real, getattr(module, "__package__", None)))
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


def _imported_names(path: str, package: str | None) -> Iterator[str]:
    """The absolute names of the modules the source at ``path`` imports, a
    dotted name's packages included.
    """
    try:
        tree = ast.parse(Path(path).read_bytes(), path)
    except (OSError, SyntaxError, ValueError):
        # Not Python source (a compiled module): its file is digested as is.
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield from _with_packages(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                try:
                    base = importlib.util.resolve_name(
                        "." * node.level + base, package or None
                    )
                except (ImportError, ValueError):
                    continue
            yield from _with_packages(base)
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}"
        elif isinstance(node, ast.Call) and _literal_import(node):
            yield from _with_packages(node.args[0].value)  # type: ignore[attr-defined]


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


def _with_packages(name: str) -> Iterator[str]:
    """``a.b.c`` and the packages importing it imports: ``a``, ``a.b``."""
    parts = name.split(".")
    for end in range(1, len(parts) + 1):
        yield ".".join(parts[:end])


def _relative(path: str, directory: str) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()
