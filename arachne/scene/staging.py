"""Where a render's files wait until it has ended.

A render used to write as it went: each play's segment into the shared
partial-movie cache, a PNG sequence's frames to their numbered places, then
the still, the sections and the movie. A render refused or failing late
would leave new files behind, or have overwritten old ones, and deleting
from a shared cache can't tell this render's segment from a concurrent
writer's. So everything a render writes goes first to a staging directory of
its own, laid out by the directory each file is meant for, and is moved into
place only once the render has ended unrefused (:meth:`Staging.publish`). A
render that ends otherwise removes the staging directory and touches
nothing else (:meth:`Staging.discard`).

Reading is unchanged: a file the render did not write is read where it is,
so the cache a render finds is the cache as it was, and a file the render
wrote is read from staging (:meth:`Staging.find`) until it is published.

A file writer used on its own, outside a render, writes in place
(:class:`InPlace`).
"""

from __future__ import annotations

__all__ = ["InPlace", "Staging"]

import errno
import os
import shutil
import tempfile
from pathlib import Path


class Staging:
    """One render's staging directory, made beside ``near`` at the first
    file staged.

    Each directory a final file is meant for gets one directory of its own
    in staging, and a file keeps its name there, so a file written beside a
    staged file (a sound track beside the movie) is staged beside it too,
    and published beside it.
    """

    def __init__(self, near: Path) -> None:
        self._near = Path(near)
        self._root: Path | None = None
        #: Every directory a final file is meant for, with its directory in
        #: staging, in the order first staged.
        self._dirs: dict[Path, Path] = {}
        #: The directories made to hold the staging directory, deepest last.
        self._made: list[Path] = []

    def path(self, final: Path) -> Path:
        """Where ``final`` is written while the render runs: its directory's
        place in staging, made if need be, and its own name.
        """
        final = Path(final)
        directory = final.parent
        staged = self._dirs.get(directory)
        if staged is None:
            staged = self._dirs[directory] = self._root_dir() / str(len(self._dirs))
            staged.mkdir()
        return staged / final.name

    def find(self, final: Path) -> Path:
        """Where ``final`` is read now: from staging when the render wrote
        it, else where it is.
        """
        final = Path(final)
        staged = self._dirs.get(final.parent)
        if staged is not None and (staged / final.name).exists():
            return staged / final.name
        return final

    def staged(self, final: Path) -> Path | None:
        """``final``'s place in staging, written or not, when the render
        has staged a file of its directory; else ``None``.
        """
        final = Path(final)
        staged = self._dirs.get(final.parent)
        return None if staged is None else staged / final.name

    #: Files wait here until published.
    waits = True

    def publish(self, only: Path | None = None) -> list[Path]:
        """Move every staged file onto its final path, directory by
        directory in the order they were first staged (only those meant for
        the directory ``only``, when given), then remove the staging
        directory: the final paths, in that order.
        """
        published: list[Path] = []
        for directory, staged in self._dirs.items():
            if only is not None and directory != only:
                continue
            files = sorted(path for path in staged.iterdir() if path.is_file())
            if not files:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            for path in files:
                target = directory / path.name
                _move(path, target)
                published.append(target)
        self._remove()
        return published

    def discard(self) -> None:
        """Remove the staging directory and everything in it, and the
        directories made only to hold it: nothing outside staging changes.
        """
        self._remove()
        for directory in reversed(self._made):
            try:
                directory.rmdir()
            except OSError:
                break
        self._made.clear()

    def _root_dir(self) -> Path:
        if self._root is None:
            missing = []
            for directory in (self._near, *self._near.parents):
                if directory.exists():
                    break
                missing.append(directory)
            self._near.mkdir(parents=True, exist_ok=True)
            self._made = list(reversed(missing))
            self._root = Path(tempfile.mkdtemp(prefix=".staging-", dir=self._near))
        return self._root

    def _remove(self) -> None:
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None
        self._dirs.clear()


class InPlace:
    """No staging: every file is written where it is meant to be, at once.
    A file writer used outside a render writes so.
    """

    #: Files are where they are meant to be as soon as they are written.
    waits = False

    def path(self, final: Path) -> Path:
        final = Path(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        return final

    def find(self, final: Path) -> Path:
        return Path(final)

    def staged(self, final: Path) -> Path | None:
        return Path(final)

    def publish(self, only: Path | None = None) -> list[Path]:
        return []

    def discard(self) -> None:
        pass


def _move(source: Path, target: Path) -> None:
    """Move ``source`` onto ``target``: a rename where both are on one
    filesystem, which a reader of ``target`` sees whole or not at all.
    """
    try:
        os.replace(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.move(source, target)
