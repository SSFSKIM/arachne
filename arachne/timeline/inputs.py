"""What an execution of a scene ran and read: the identity a record carries
(:class:`.Identity`).

Its **source** is the scene file and every module under the scene's
directory that the execution imported or could have imported
(:func:`source_files`): each module the scene's code looked up while it
loaded and ran, whether imported then or found already imported, under any
name, computed ones included; and each module an import statement of those
files names, resolved by path in the scene's directory, whether or not the
statement ran. Neither depends on what the process imported before. Its
**inputs** are the files opened for reading while the scene's code is on
the stack. Both are named by their path relative to the scene's directory,
so that one source staged in two places digests alike.

:class:`Inputs` watches while a recording runs, and from before the scene
file loads when the session that records it loads it (:meth:`.Session.open`).
The files opened are seen by one :func:`sys.addaudithook` hook per process,
which can't be removed, so it is installed once and does nothing while
nothing watches. The modules looked up are seen by wrapping
``builtins.__import__`` and ``importlib.import_module`` while something
watches.
"""

from __future__ import annotations

__all__ = ["Inputs", "active", "engine_dirs", "source_digest", "source_files"]

import ast
import builtins
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

from .source import _INTERNAL

#: The recordings watching now, innermost last.
_WATCHING: list[Inputs] = []
_HOOKED = False
#: Set while the hook reads a file to digest it, which it must not see.
_LOCAL = threading.local()
#: ``builtins.__import__`` and ``importlib.import_module`` as they were
#: before something watched, while something does.
_UNWRAPPED: tuple[Any, Any] | None = None


def active() -> bool:
    """Whether a recording watches the files opened now."""
    return bool(_WATCHING)


# Files opened


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


# Modules looked up


def _watched_import(
    name: str,
    globals: Mapping[str, Any] | None = None,  # noqa: A002 (``__import__``'s)
    locals: Mapping[str, Any] | None = None,  # noqa: A002
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> ModuleType:
    assert _UNWRAPPED is not None
    module = _UNWRAPPED[0](name, globals, locals, fromlist, level)
    if _WATCHING:
        for watcher in tuple(_WATCHING):
            watcher.imported(name, globals, fromlist or (), level)
    return module  # type: ignore[no-any-return]


def _watched_import_module(name: str, package: str | None = None) -> ModuleType:
    assert _UNWRAPPED is not None
    module = _UNWRAPPED[1](name, package)
    if _WATCHING:
        for watcher in tuple(_WATCHING):
            watcher.looked_up(_with_packages(module.__name__))
    return module  # type: ignore[no-any-return]


def _wrap_imports() -> None:
    global _UNWRAPPED
    if _UNWRAPPED is None:
        _UNWRAPPED = (builtins.__import__, importlib.import_module)
        builtins.__import__ = _watched_import
        importlib.import_module = _watched_import_module


def _unwrap_imports() -> None:
    """Put the import functions back, unless something wrapped them since
    (the wrappers then stay, doing nothing more while nothing watches).
    """
    global _UNWRAPPED
    if _UNWRAPPED is None:
        return
    if (
        builtins.__import__ is _watched_import
        and importlib.import_module is _watched_import_module
    ):
        builtins.__import__, importlib.import_module = _UNWRAPPED
        _UNWRAPPED = None


class Inputs:
    """What a scene's execution looked up and read while watched: the modules
    under the scene's directory its code looked up, and the files it opened
    for reading, with the digest of what each held when first opened.

    A module counts when the code that looked it up is the scene's (a file
    under the scene's directory), or the lookup came from no module's code
    (``importlib.import_module``, ``__import__`` called without globals).

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
        #: The files of the modules under the scene's directory looked up.
        self.modules: set[str] = set()
        #: Whether code from a file (by code object file name) is the scene's.
        self._scene_code: dict[str, bool] = {}

    def start(self) -> None:
        """Watch from now on."""
        global _HOOKED
        if self in _WATCHING:
            return
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True
        _wrap_imports()
        _WATCHING.append(self)

    def stop(self) -> None:
        """Stop watching; what was seen stays readable."""
        if self in _WATCHING:
            _WATCHING.remove(self)
        if not _WATCHING:
            _unwrap_imports()

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

    def imported(
        self,
        name: str,
        globals: Mapping[str, Any] | None,  # noqa: A002 (``__import__``'s)
        fromlist: Sequence[str],
        level: int,
    ) -> None:
        """An import statement (or ``__import__`` call) of ``name`` has run,
        from the module whose globals are ``globals``.
        """
        if globals is not None:
            file = globals.get("__file__")
            if not isinstance(file, str) or not self._is_scene_code(file):
                return
        base = name
        if level:
            package = globals.get("__package__") if globals is not None else None
            if package is None and globals is not None:
                own = str(globals.get("__name__", ""))
                package = own if "__path__" in globals else own.rpartition(".")[0]
            try:
                base = importlib.util.resolve_name("." * level + name, package)
            except (ImportError, ValueError):
                return
        names = list(_with_packages(base)) if base else []
        names.extend(f"{base}.{item}" for item in fromlist if item != "*")
        self.looked_up(names)

    def looked_up(self, names: Iterable[str]) -> None:
        """The modules named ``names`` were looked up: note those whose file
        lies under the scene's directory.
        """
        for name in names:
            file = getattr(sys.modules.get(name), "__file__", None)
            if not isinstance(file, str):
                continue
            try:
                real = os.path.realpath(file)
            except (OSError, ValueError):
                continue
            if real.startswith(self._scene_prefix):
                self.modules.add(real)

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


def source_files(scene_file: str, imported: Iterable[str] = ()) -> list[str]:
    """The source of an execution of ``scene_file``, as real paths: the scene
    file, the ``imported`` files (the modules under its directory its code
    looked up, :attr:`Inputs.modules`), and every module under its
    directory that an import statement of any of them names, resolved by
    path in that directory (:func:`_named_files`) and followed the same way.
    """
    scene_file = os.path.realpath(scene_file)
    scene_dir = os.path.dirname(scene_file)
    prefix = scene_dir.rstrip(os.sep) + os.sep
    found = {scene_file}
    found.update(path for path in imported if path.startswith(prefix))
    todo = list(found)
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


def _with_packages(name: str) -> Iterator[str]:
    """``a.b.c`` and the packages importing it imports: ``a``, ``a.b``."""
    parts = name.split(".")
    for end in range(1, len(parts) + 1):
        yield ".".join(parts[:end])


def _relative(path: str, directory: str) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()
