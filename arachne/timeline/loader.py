"""The modules of one execution of a scene file.

A scene file imports modules beside it (helpers, a package of shapes),
and a session executes it many times in one process: when opened, when
the probe records it again, for each forward run. Python's import system
keeps a module in ``sys.modules`` once imported, under its plain name, and
reuses bytecode cached beside a source file whose size and modification
time (in whole seconds) still match it. Either serves an edited helper as
it was, while the source digest, read from the files, says it is new.

:func:`imports` gives each execution modules of its own: every module of
the scene's directory it imports is compiled from what its file holds now,
and once the execution ends, however it ends, ``sys.modules`` and
``sys.path`` hold what they held before it.
"""

from __future__ import annotations

__all__ = ["FromSource", "imports"]

import contextlib
import importlib
import linecache
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import CodeType, ModuleType


class FromSource(SourceFileLoader):
    """The loader of a source file that compiles what the file holds now:
    never bytecode cached beside it, which an edit within the same second
    that keeps the size still matches, and none written.
    """

    def get_code(self, fullname: str) -> CodeType:
        # Tracebacks and the purity check read a function's lines from the
        # line cache, which may hold the file as it was.
        linecache.cache.pop(self.path, None)
        return self.source_to_code(self.get_data(self.path), self.path)


class _SceneModules(MetaPathFinder):
    """Finds, for one execution, the modules that are the scene's: a
    top-level module the import path resolves to a file or package in the
    scene's directory, and the submodules of such a package. A name the
    process had imported a module of before the execution (``arachne``,
    ``numpy``) is never the scene's, whatever the directory holds.
    """

    def __init__(self, directory: str, host: frozenset[str]) -> None:
        self.directory = directory
        self.host = host
        #: The names of the modules it found, which the execution imported.
        self.names: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        package, dot, _ = fullname.rpartition(".")
        if dot:
            if package not in self.names:
                return None
        elif fullname in self.host:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or (not dot and not self._in_directory(fullname, spec)):
            return None
        self.names.add(fullname)
        if isinstance(spec.loader, SourceFileLoader):
            spec.loader = FromSource(fullname, spec.loader.path)
        return spec

    def _in_directory(self, fullname: str, spec: ModuleSpec) -> bool:
        """Whether ``spec``, the import path's module ``fullname``, is the one
        the scene's directory holds.
        """
        local = PathFinder.find_spec(fullname, [self.directory])
        if local is None:
            return False
        if local.origin is not None:
            return spec.origin is not None and os.path.realpath(
                spec.origin
            ) == os.path.realpath(local.origin)
        # A namespace package: the directory's portion is among its portions.
        return set(local.submodule_search_locations or ()) <= set(
            spec.submodule_search_locations or ()
        )


@contextlib.contextmanager
def imports(scene_file: Path) -> Iterator[None]:
    """The block imports ``scene_file``'s modules afresh: its directory is
    first on ``sys.path`` (unless already on it), and each module of that
    directory the block imports is loaded from its source (:class:`FromSource`),
    whether or not an earlier block imported it. Afterwards, on a raise too,
    ``sys.path`` holds what it held before, and ``sys.modules`` none of the
    modules the block imported from the directory: what the process imported
    before is kept as it was, and what the block imported from elsewhere
    stays imported.
    """
    directory = str(scene_file.parent)
    path = list(sys.path)
    finder = _SceneModules(
        directory, frozenset(name.partition(".")[0] for name in sys.modules)
    )
    if directory not in sys.path:
        sys.path.insert(0, directory)
    # Before the path's own finder, so a module of the directory is found
    # here first; directory listings cached since the last block refreshed.
    sys.meta_path.insert(sys.meta_path.index(PathFinder), finder)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = path
        for name in finder.names:
            sys.modules.pop(name, None)
