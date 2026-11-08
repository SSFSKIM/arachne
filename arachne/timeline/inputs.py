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

__all__ = [
    "Inputs",
    "Read",
    "active",
    "engine_dirs",
    "source_digest",
    "source_files",
]

import ast
import hashlib
import importlib.machinery
import os
import sys
import threading
from collections.abc import Callable, Collection, Iterable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any, NamedTuple

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


class Read(NamedTuple):
    """A file an execution read: its ``path`` relative to the scene's
    directory, the sha256 of what it held when first opened, and the index
    of the first ``play`` that content could change (the one running as it
    was opened, or else the next to begin).
    """

    path: str
    digest: str
    play: int


class Inputs:
    """The files a recording's scene opened for reading, with the digest of
    what each held when first opened, and the play it was first opened for
    (``play()``, the play running or else the next to begin; ``0`` without
    it).

    A file counts when the code that opened it was called from the scene's
    code, from a module under the scene's directory, and not by an import
    (which is the source's to digest), and when it lies outside the
    ``excluded`` directories (the engine's media and caches,
    :func:`engine_dirs`), the engine's package, the standard library and
    the installed packages: those are what the engine and its environment
    read, the same for every scene they run. A directory among them that
    holds the scene's own is not excluded, since it would hold every input.
    """

    def __init__(
        self,
        scene_dir: str,
        excluded: Iterable[str],
        play: Callable[[], int] | None = None,
    ) -> None:
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
        #: Each file read, by real path, and the play it was first read for.
        self._first_play: dict[str, int] = {}
        self._play = play
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
            self._first_play[real] = 0 if self._play is None else self._play()
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

    def reads(self, sources: Collection[str]) -> tuple[Read, ...]:
        """Each file read but the ``sources``, as :meth:`input_digests` gives
        it, with the play it was first read for, by path.
        """
        return tuple(
            sorted(
                Read(_relative(path, self.scene_dir), digest, self._first_play[path])
                for path, digest in self._read.items()
                if path not in sources
            )
        )


#: The configuration's directories the engine writes media and caches into.
ENGINE_DIR_KEYS = (
    "media_dir",
    "video_dir",
    "sections_dir",
    "images_dir",
    "text_dir",
    "tex_dir",
    "typeset_dir",
    "partial_movie_dir",
    "log_dir",
)

#: Stands for the module and scene names, which a run's configuration does
#: not hold: a directory whose path has one as a whole component is excluded
#: up to it (``{media_dir}/videos/{module_name}/...``).
_HOLE = "\0"


def engine_dirs(config: Any) -> list[str]:
    """The directories the engine writes its media and caches into, under
    ``config``, resolved by ``config.get_dir`` itself, so that every field
    it fills (the other directories, ``assets_dir``, the input and output
    files, the quality) fills them here too.

    It resolves them on a view of the configuration whose path objects are
    strings (the configuration keeps one that is not a ``pathlib.Path`` as
    it was given, which ``get_dir`` refuses). A directory whose path holds a
    module or scene name as a whole component is taken up to it; one that
    does not resolve, or holds a name inside a component, is left out: it
    excludes nothing. Whatever goes wrong resolving one directory leaves
    that one out, since watching what a scene reads must never stop a
    render.
    """
    try:
        view = type(config)()
        view._d = {
            key: os.fspath(value) if isinstance(value, os.PathLike) else value
            for key, value in config._d.items()
        }
    except Exception:  # noqa: BLE001 (any configuration: see the docstring)
        return []
    found = []
    for key in ENGINE_DIR_KEYS:
        try:
            directory = _engine_dir(view, key)
        except Exception:  # noqa: BLE001, S112 (see the docstring)
            continue
        if directory is not None:
            found.append(directory)
    return found


def _engine_dir(view: Any, key: str) -> str | None:
    """The real path of the directory ``key`` of ``view`` names, up to a
    module or scene name, or ``None`` when it excludes nothing.
    """
    if not isinstance(view._d.get(key), str):
        return None
    directory = view.get_dir(key, module_name=_HOLE, scene_name=_HOLE)
    if directory is None:
        return None
    parts = Path(directory).parts
    cut = next((k for k, part in enumerate(parts) if _HOLE in part), None)
    if cut is not None:
        if parts[cut] != _HOLE or cut == 0:
            return None
        parts = parts[:cut]
    # ``Path(".").parts`` is empty, and ``Path()`` is the working directory.
    return os.path.realpath(Path(*parts))


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
    ``scene_dir`` and the sha256 of what it holds, by path
    (:func:`file_digests`).
    """
    digest = hashlib.sha256()
    for relative, content in file_digests(files, scene_dir):
        digest.update(f"{relative}\0{content}\n".encode())
    return digest.hexdigest()


def file_digests(files: Iterable[str], scene_dir: str) -> tuple[tuple[str, str], ...]:
    """Each of ``files`` as ``(path relative to scene_dir, sha256 of what it
    holds)``, by path. Raises ``OSError`` for a file that can't be read.
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
    return tuple(sorted(entries))


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
