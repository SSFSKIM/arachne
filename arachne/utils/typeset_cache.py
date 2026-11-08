"""The typesetting cache: Tex and Text SVGs kept in one directory the caller
names (:attr:`~.ManimConfig.typeset_dir`), shared by every scene and process
that names it, and by other hosts when it is on a shared disk.

A file's name is a digest of everything its bytes depend on.

- For Tex: the whole ``.tex`` source (the template and its preamble with the
  expression), the compilers and their output format; the versions of the
  compilers and of ``dvisvgm`` with the libraries it reports (and, for PDF
  output, of the converters it reads PDF through); the TeX distribution's
  package database, by digest; the environment's TeX search path; for
  XeLaTeX and LuaLaTeX, the system fonts they can load (fontconfig's list,
  by digest, and ``OSFONTDIR``); and, where the compile can read files
  beside the process (a file there only TeX reads, such as a package, a
  font description or metrics, and for XeLaTeX and LuaLaTeX a font file;
  the configuration of a package or class the source loads; a file the
  source reads by name), the working directory and those files.
- For Text and MarkupText: every argument Pango typesets with (the text, its
  spans' fonts, slants, weights and colours, the size, the line spacing and
  the layout's width); the versions of ManimPango, Pango, HarfBuzz and
  Cairo; and the fonts the text is shaped with, by content
  (:mod:`.typeset_fonts`).

Where what a key needs cannot be found (no package database, no font list
the compiler reads, no answer from Pango), the key names the host instead
(:func:`host`), so the file is never shared with another host. So two
processes share a file only when they would have typeset the same bytes.

A file there is made in a scratch directory of the process's own (under
``scratch/``), checked to be a whole SVG, and renamed into place, and is
never written there again, so that any number of processes can typeset into
the directory at once:

- a reader finds a whole file or none, never one another process is still
  writing, or left half written when it was killed or its tool failed;
- what a compile leaves beside its output (``.dvi``, ``.aux``, ``.log``) is
  in its own scratch directory, which only it removes;
- Text's SVG is final when it is published: the edit CE makes to it in
  place at every construction, a cached one's too, is made once, in scratch;
- :class:`~.SVGMobject` parses the SVG it modified from memory, not through
  a file beside the SVG, which two processes loading one SVG would share.

Two processes typesetting one string at once both make it, and the later
rename replaces the earlier file with the same bytes. A failed compile
leaves its scratch directory, which holds its log.

Unset, or with the speed levers off, each media directory keeps its own Tex
and Text files, in ``tex_dir`` and ``text_dir``, as CE does.
"""

from __future__ import annotations

__all__ = [
    "DIR_ENV",
    "directory",
    "host",
    "made",
    "tex_toolchain",
    "text_toolchain",
]

import contextlib
import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from stat import S_ISREG
from xml.etree import ElementTree

from .. import config
from .._config.utils import TYPESET_DIR_ENV as DIR_ENV

#: How the engine makes a file from its inputs, part of every key: raised
#: when that changes (another ``dvisvgm`` option, another edit of Pango's
#: SVG), so that no file made the old way is read.
FORMAT = 2


def directory() -> Path | None:
    """The typesetting directory the configuration names, or ``None`` (none
    named, or the speed levers off).
    """
    return config.typeset_dir


def made(kind: str, inputs: object, make: Callable[[Path], object]) -> Path:
    """The file of ``kind`` (``"tex"`` or ``"text"``) made from ``inputs``
    (lists, strings and numbers; anything else by its ``repr``) in the
    typesetting directory: the one kept there, else the one ``make(path)``
    writes at ``path``, in a scratch directory of the process's own that it
    may write beside it too, renamed into place once it is a whole SVG.
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
        _whole(path)
        return _publish(path, target)


def _whole(path: Path) -> None:
    """Refuse (``ValueError``) a file that is not a whole SVG."""
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as error:
        raise ValueError(
            f"Typesetting made no whole SVG ({path}: {error}); nothing was kept."
        ) from error
    if root.tag.rpartition("}")[2] != "svg":
        raise ValueError(f"Typesetting made no whole SVG ({path}); nothing was kept.")


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


# What Tex depends on beside its source

#: Packages and classes the source loads by name, whose configuration
#: (``<name>.cfg``) TeX finds in the working directory before the
#: distribution's.
_LOADS = re.compile(
    r"\\(?:usepackage|RequirePackage|documentclass|LoadClass)\s*"
    r"(?:\[[^\]]*\])?\s*\{([^{}]*)\}"
)

#: Suffixes of the files only TeX reads (packages, classes, their option and
#: definition files, font descriptions, metrics, Type 1 fonts and maps),
#: which count wherever they are found beside the process: a package the
#: source's packages load in turn is found there by a name the source never
#: names.
_ONLY_TEX_READS = frozenset(
    {
        ".sty",
        ".cls",
        ".clo",
        ".fd",
        ".def",
        ".ltx",
        ".tfm",
        ".vf",
        ".pfb",
        ".pfa",
        ".afm",
        ".enc",
        ".map",
    }
)

#: Suffixes of the font files XeLaTeX and LuaLaTeX load by file name
#: (``\setmainfont{Inter-Regular.otf}``), which count under those compilers.
_FONT_FILES = frozenset({".otf", ".ttf", ".ttc"})

#: Commands that read a file by the name they are given: the file is the
#: first argument, or the second for the commands named here; the suffixes
#: TeX tries where the name has none.
_READS = re.compile(
    r"\\(input|include|InputIfFileExists|IfFileExists|includegraphics|includesvg"
    r"|includepdf|includestandalone|lstinputlisting|verbatiminput|inputminted"
    r"|import|subimport)"
    r"\b\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?(?:\{([^{}]*)\})?"
)
_SECOND_ARGUMENT = frozenset({"inputminted", "import", "subimport"})
_TRIED = {
    "includegraphics": (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps", ".mps"),
    "includesvg": (".svg",),
    "includepdf": (".pdf",),
    "lstinputlisting": (),
    "verbatiminput": (),
    "inputminted": (),
}

#: Compilers that load system fonts by name (``fontspec``), outside the TeX
#: tree.
_SYSTEM_FONT_COMPILERS = frozenset({"xelatex", "lualatex", "xetex", "luatex"})

#: Whether fontconfig's list is what those compilers load fonts from: not on
#: macOS, where XeTeX asks CoreText and LuaTeX scans the system's font
#: directories itself.
_FONTS_BY_FONTCONFIG = sys.platform != "darwin"

#: What names a machine beside its hostname, which containers from one image
#: can share: its machine id and the id of its current boot (Linux).
_MACHINE_ID = Path("/etc/machine-id")
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def tex_toolchain(
    compilers: Sequence[str], output_format: str, source: str
) -> list[object]:
    """What a Tex file's bytes depend on beside ``source`` (the ``.tex``
    text) compiled by ``compilers`` to ``output_format``: the tools'
    versions, the distribution's packages, the search path, the system fonts
    the compilers can load and the files beside the process the compile can
    read.
    """
    tools = [[tool, _version(tool, "--version")] for tool in compilers]
    tools.append(["dvisvgm", _version("dvisvgm", "--version=yes", "--version")])
    if output_format == ".pdf":
        tools += [
            ["mutool", _version("mutool", "-v")],
            ["gs", _version("gs", "--version")],
        ]
    system_fonts = bool(_SYSTEM_FONT_COMPILERS.intersection(compilers))
    found = [tools, _packages(), _search_path(), _beside(source, system_fonts)]
    if system_fonts:
        found.append(_system_fonts())
    return found


def host() -> list[str]:
    """What a key names where it cannot name what a file depends on: this
    machine, by its hostname, machine id and boot id (each that it has).
    """
    return ["host", socket.gethostname(), _read_id(_MACHINE_ID), _read_id(_BOOT_ID)]


def _read_id(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _system_fonts() -> list[str]:
    """The system fonts XeLaTeX and LuaLaTeX can load: fontconfig's list
    (each font's file, index and version), by digest, and ``OSFONTDIR``;
    else, where fontconfig cannot be asked or is not what they load fonts
    through, the host.
    """
    found = shutil.which("fc-list")
    if found is None or not _FONTS_BY_FONTCONFIG:
        return host()
    status = os.stat(found)
    listed = _font_list(found, status.st_ino, status.st_mtime_ns, _fontconfig_env())
    if listed is None:
        return host()
    return ["system fonts", listed, os.environ.get("OSFONTDIR", "")]


def _fontconfig_env() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("FONTCONFIG")
        )
    )


@functools.cache
def _font_list(
    path: str, inode: int, mtime_ns: int, env: tuple[tuple[str, str], ...]
) -> str | None:
    """The digest of the font list ``path`` (``fc-list``) prints, sorted:
    asked once in a process for each build of it and fontconfig
    environment ``env``; ``None`` when it fails.
    """
    try:
        ran = subprocess.run(
            [path, "--format", r"%{file}\t%{index}\t%{fontversion}\n"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if ran.returncode != 0:
        return None
    lines = sorted(ran.stdout.splitlines())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _search_path() -> list[list[str]]:
    """The environment's TeX settings: its search paths and its trees."""
    return sorted(
        [name, value]
        for name, value in os.environ.items()
        if name.startswith(("TEX", "KPATHSEA")) or name.endswith(("INPUTS", "FONTS"))
    )


def _packages() -> list[str]:
    """The TeX distribution's package database (``tlpkg/texlive.tlpdb``
    under its root, which ``kpsewhich`` names), by digest; else the host.
    """
    for variable in ("TEXMFROOT", "SELFAUTOPARENT"):
        root = _version("kpsewhich", f"-var-value={variable}")
        if root in ("", "unavailable"):
            continue
        database = Path(root) / "tlpkg" / "texlive.tlpdb"
        if database.is_file():
            stat = database.stat()
            return [
                "packages",
                _file_digest(str(database), stat.st_size, stat.st_mtime_ns),
            ]
    return host()


@functools.cache
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    """``path``'s digest, read once for each version of it (by its size and
    time of change).
    """
    digest = hashlib.sha256()
    with open(path, "rb") as read:
        for block in iter(lambda: read.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _beside(source: str, font_files: bool) -> list[object]:
    """The files beside the process a compile of ``source`` can read: each
    file there only TeX reads (and, where ``font_files``, each font file);
    the configuration (``.cfg``) of each package and class ``source`` loads;
    and each file ``source`` reads by name. None to name when there are none
    and ``source`` reads no file by name; else the directory and each of
    those files, by size and time of change.
    """
    here = os.getcwd()
    mtime_ns = os.stat(here).st_mtime_ns
    configured = {
        name.strip() + ".cfg"
        for loads in _LOADS.finditer(source)
        for name in loads.group(1).split(",")
    }
    reads = list(_READS.finditer(source))
    candidates = (
        _only_tex_reads(here, mtime_ns, font_files)
        | (configured & _listing(here, mtime_ns))
        | {name for read in reads for name in _read_names(read)}
    )
    if not candidates and not reads:
        return ["nothing beside"]
    files = []
    for name in sorted(candidates):
        try:
            status = os.stat(os.path.join(here, name))
        except OSError:
            continue
        if S_ISREG(status.st_mode):
            files.append([name, status.st_size, status.st_mtime_ns])
    return ["beside", here, files]


@functools.lru_cache(maxsize=64)
def _only_tex_reads(here: str, mtime_ns: int, font_files: bool) -> frozenset[str]:
    """The names in directory ``here`` of files only TeX reads (font files
    too, where ``font_files``), found once for each version of its entries.
    """
    suffixes = _ONLY_TEX_READS | _FONT_FILES if font_files else _ONLY_TEX_READS
    return frozenset(
        name
        for name in _listing(here, mtime_ns)
        if os.path.splitext(name)[1].lower() in suffixes
    )


@functools.lru_cache(maxsize=64)
def _listing(here: str, mtime_ns: int) -> frozenset[str]:
    """The names in directory ``here``, listed once for each version of its
    entries (by its time of change, which a file added, removed or renamed
    there moves).
    """
    return frozenset(os.listdir(here))


def _read_names(read: re.Match[str]) -> list[str]:
    """The names TeX tries for the file one command of :data:`_READS` reads."""
    command, first, second = read.groups()
    if command in _SECOND_ARGUMENT:
        if not second:
            return []
        # ``\import{directory}{file}``; ``\inputminted{language}{file}``.
        name = second if command == "inputminted" else os.path.join(first, second)
    elif first:
        name = first
    else:
        return []
    name = name.strip()
    if Path(name).suffix:
        return [name]
    return [name] + [name + suffix for suffix in _TRIED.get(command, (".tex",))]


def _version(tool: str, *asks: str) -> str:
    """What ``tool`` (the program ``PATH`` finds now) prints asked each of
    ``asks`` in turn, the first it answers without an error: ``"unavailable"``
    when there is no such program or it answers none.
    """
    found = shutil.which(tool)
    if found is None:
        return "unavailable"
    stat = os.stat(found)
    for ask in asks:
        answer = _answer(
            found, stat.st_ino, stat.st_size, stat.st_mtime_ns, ask, _tex_env()
        )
        if answer is not None:
            return answer
    return "unavailable"


def _tex_env() -> tuple[tuple[str, str], ...]:
    return tuple((name, value) for name, value in _search_path())


@functools.cache
def _answer(
    path: str,
    inode: int,
    size: int,
    mtime_ns: int,
    ask: str,
    env: tuple[tuple[str, str], ...],
) -> str | None:
    """``path``'s answer to ``ask``, asked once for each build of it (by its
    inode, size and time of change) and TeX environment ``env``; ``None``
    when it fails.
    """
    try:
        ran = subprocess.run([path, ask], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if ran.returncode != 0:
        return None
    return (ran.stdout or ran.stderr).strip()


# What Text depends on beside its arguments and fonts


def text_toolchain() -> list[tuple[str, str]]:
    """ManimPango's, Pango's, HarfBuzz's and Cairo's versions."""
    import manimpango

    return [
        ("manimpango", manimpango.__version__),
        ("pango", manimpango.pango_version()),
        ("harfbuzz", _harfbuzz_version()),
        ("cairo", manimpango.cairo_version()),
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
