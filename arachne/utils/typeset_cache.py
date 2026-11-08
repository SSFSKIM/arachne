"""The typesetting cache: Tex and Text SVGs kept in one directory the caller
names (:attr:`~.ManimConfig.typeset_dir`), shared by every scene and process
that names it, and by other hosts when it is on a shared disk.

A file's name is a digest of everything its bytes depend on: for Tex, the
whole ``.tex`` source (the template and its preamble with the expression),
the compilers and their output format, and the versions of the compilers
and of ``dvisvgm``; for Text and MarkupText, every argument Pango typesets
with (the text, its spans' fonts, slants, weights and colours, the size,
the line spacing and the layout's width), and the versions of ManimPango,
Pango, HarfBuzz and Cairo with a fingerprint of the fonts the process can
use, since shaping depends on all of them. So two processes share a file
only when they would have typeset the same bytes.

A file there is made in a scratch directory of the process's own (under
``scratch/``) and renamed into place whole, and is never written there
again, so that any number of processes can typeset into the directory at
once:

- a reader finds a whole file or none, never one another process is still
  writing or left half written when it was killed;
- what a compile leaves beside its output (``.dvi``, ``.aux``, ``.log``) is
  in its own scratch directory, which only it removes;
- Text's SVG is final when it is published: the edit CE makes to it in
  place at every construction, a cached one's too, is made once, in scratch.

(:class:`~.SVGMobject` parses every SVG through no file beside it, so
neither does loading one.) Two processes typesetting one string at once both
make it, and the later rename replaces the earlier file with the same bytes.
A failed compile leaves its scratch directory, which holds its log.

Unset, each media directory keeps its own Tex and Text files, in ``tex_dir``
and ``text_dir``, as CE does.
"""

from __future__ import annotations

__all__ = ["DIR_ENV", "directory", "made", "tex_toolchain", "text_toolchain"]

import contextlib
import ctypes
import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from .. import config
from .._config.utils import TYPESET_DIR_ENV as DIR_ENV

#: How the engine makes a file from its inputs, part of every key: raised
#: when that changes (another ``dvisvgm`` option, another edit of Pango's
#: SVG), so that no file made the old way is read.
FORMAT = 1


def directory() -> Path | None:
    """The typesetting directory the configuration names (``None`` while
    the speed levers are off), or ``None``.
    """
    return config.typeset_dir


def made(kind: str, inputs: object, make: Callable[[Path], object]) -> Path:
    """The file of ``kind`` (``"tex"`` or ``"text"``) made from ``inputs``
    (lists, strings and numbers; anything else by its ``repr``) in the
    typesetting directory: the one kept there, else the one ``make(path)``
    writes whole at ``path``, in a scratch directory of the process's own
    that it may write beside it too, renamed into place.
    """
    root = directory()
    assert root is not None
    source = json.dumps([FORMAT, kind, inputs], ensure_ascii=False, default=repr)
    target = root / kind / f"{hashlib.sha256(source.encode()).hexdigest()[:32]}.svg"
    if target.exists():
        return target
    with _scratch(root) as work:
        path = work / target.name
        make(path)
        return _publish(path, target)


@contextlib.contextmanager
def _scratch(root: Path) -> Iterator[Path]:
    """A directory of this process's own under ``root``, removed when the
    block ends, unless it raised (a failed compile's log stays) or
    ``no_latex_cleanup`` keeps what a compile left.
    """
    parent = root / "scratch"
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=parent))
    yield work
    if not config["no_latex_cleanup"]:
        shutil.rmtree(work, ignore_errors=True)


def _publish(made: Path, target: Path) -> Path:
    """Move the whole file ``made`` to ``target`` at once, its bytes on disk
    first: a reader of ``target`` finds it whole or not at all.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(made, "rb") as written:
        os.fsync(written.fileno())
    os.replace(made, target)
    return target


# The toolchain


def tex_toolchain(compilers: Sequence[str]) -> list[list[str]]:
    """Each of ``compilers`` and ``dvisvgm``, with the version it reports."""
    return [[tool, _version(tool)] for tool in (*compilers, "dvisvgm")]


def _version(tool: str) -> str:
    """The first line ``tool --version`` prints, of the program ``PATH``
    finds now; ``"unavailable"`` when there is none.
    """
    found = shutil.which(tool)
    if found is None:
        return "unavailable"
    stat = os.stat(found)
    return _version_of(found, stat.st_ino, stat.st_size, stat.st_mtime_ns)


@functools.cache
def _version_of(path: str, inode: int, size: int, mtime_ns: int) -> str:
    """``path``'s version, asked once for each build of it (by its inode,
    size and time of change).
    """
    try:
        ran = subprocess.run(
            [path, "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unavailable"
    lines = (ran.stdout or ran.stderr).strip().splitlines()
    return lines[0] if lines else "unavailable"


def text_toolchain() -> list[tuple[str, str]]:
    """What Text's bytes depend on beside its arguments: ManimPango's,
    Pango's, HarfBuzz's and Cairo's versions, and the fonts the process can
    use (:func:`_font_families`, by digest).
    """
    import manimpango

    fonts = hashlib.sha256("\n".join(_font_families()).encode()).hexdigest()
    return [
        ("manimpango", manimpango.__version__),
        ("pango", manimpango.pango_version()),
        ("harfbuzz", _harfbuzz_version()),
        ("cairo", manimpango.cairo_version()),
        ("fonts", fonts),
    ]


@functools.cache
def _harfbuzz_version() -> str:
    """The version of the HarfBuzz ManimPango shapes with, asked of the
    library it loaded; where none answers, the digest of ManimPango's own
    build, which names the HarfBuzz built into it.
    """
    from manimpango import cmanimpango

    try:
        version = ctypes.CDLL(cmanimpango.__file__).hb_version_string
    except (OSError, AttributeError):
        built = Path(cmanimpango.__file__).read_bytes()
        return "build " + hashlib.sha256(built).hexdigest()
    version.restype = ctypes.c_char_p
    found: bytes = version()
    return found.decode()


def _font_families() -> tuple[str, ...]:
    """The font families Pango can use, with the fonts registered through
    :func:`~.register_font`, each registered file by its content.
    """
    import manimpango

    registered = []
    for font in sorted(manimpango.registered_fonts, key=lambda f: f.path):
        stat = os.stat(font.path)
        registered.append((font.path, stat.st_size, stat.st_mtime_ns))
    return _families(tuple(registered))


@functools.cache
def _families(registered: tuple[tuple[str, int, int], ...]) -> tuple[str, ...]:
    """The families Pango lists while the fonts ``registered`` are, and each
    registered file's digest. (Listing takes about a tenth of a second, so
    it is done once for each set.)
    """
    import manimpango

    families = tuple(sorted(manimpango.list_fonts()))
    files = tuple(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() for path, _, _ in registered
    )
    return families + files
