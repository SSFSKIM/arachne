"""The typesetting cache: Tex and Text SVGs kept in a directory the caller
names (``typeset_dir``), shared by every scene, process and host naming it.

Most tests typeset through stand-in TeX tools, scripts first on ``PATH``
named ``latex``, ``xelatex`` and ``dvisvgm``, which log each compile and can
pause in the middle of a step, so that a race between two processes is
staged at the instant it happens rather than hoped for. Each race is staged
twice: through the cache, where it must not happen, and through one Tex
directory that several processes share the way CE's cache can be shared
(``tex_dir``), where it does, which shows the staging reproduces it. The
tests that compare with a fresh compile use the real TeX and Pango.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest

from arachne import config, tempconfig
from arachne.mobject.svg.svg_mobject import SVGMobject
from arachne.mobject.text.tex_mobject import SingleStringMathTex
from arachne.mobject.text.text_mobject import MarkupText, Text
from arachne.utils import typeset_cache, typeset_fonts
from arachne.utils.tex import TexTemplate

#: How long a staged step waits for its signal before it gives up.
WAIT_S = 60

# The stand-in tools, each reporting its ``VERSION`` (dvisvgm, with
# ``--version=yes``, its ``EXTRA`` line too). ``FAKE_TEX_LOG`` collects one
# line per compile; ``FAKE_PAUSE`` names the step that pauses (``latex``: once
# its output is whole; ``dvisvgm``: half way through writing its SVG) until
# the file ``resume`` appears in ``FAKE_SIGNAL``, having made ``paused``
# there; ``FAKE_FAIL`` names the step that fails (``latex`` exits 1;
# ``dvisvgm`` writes half its SVG and exits 137, as a killed one does).
# ``kpsewhich`` names ``FAKE_TEXMFROOT`` as the TeX tree.
_PAUSE = f"""\
def pause(step):
    if os.environ.get("FAKE_PAUSE") != step:
        return
    signals = Path(os.environ["FAKE_SIGNAL"])
    (signals / "paused").touch()
    until = time.time() + {WAIT_S}
    while not (signals / "resume").exists() and time.time() < until:
        time.sleep(0.01)
"""

FAKE_LATEX = """\
import os, sys, time
from pathlib import Path
{pause}
args = sys.argv[1:]
name = Path(sys.argv[0]).name
if "--version" in args:
    print(name, VERSION)
    sys.exit(0)
out = Path(next(a.split("=", 1)[1] for a in args if a.startswith("-output-directory=")))
tex = Path(args[-1])
fmt = next((a.split("=", 1)[1] for a in args if a.startswith("-output-format=")), None)
suffix = "." + (fmt or ("xdv" if "-no-pdf" in args else "pdf"))
log = os.environ.get("FAKE_TEX_LOG")
if log:
    with open(log, "a") as f:
        f.write(name + " " + tex.name + "\\n")
(out / (tex.stem + suffix)).write_text(tex.read_text())
if os.environ.get("FAKE_FAIL") == "latex":
    (out / (tex.stem + ".log")).write_text("no errors to show\\n")
    sys.exit(1)
pause("latex")
"""

FAKE_DVISVGM = """\
import os, sys, time
from pathlib import Path
{pause}
args = sys.argv[1:]
if "--version" in args:
    print("dvisvgm", VERSION)
    sys.exit(0)
if "--version=yes" in args:
    print("dvisvgm", VERSION)
    print(EXTRA)
    sys.exit(0)
out = Path(next(a.split("=", 1)[1] for a in args if a.startswith("--output=")))
source = Path(args[-1])
if not source.exists():
    sys.exit(1)
width = 2 + len(source.read_text()) % 7
svg = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='10pt' height='10pt' "
    "viewBox='0 0 10 10'><path d='M0 0L" + str(width) + " 0L" + str(width)
    + " 5Z'/></svg>"
)
with open(out, "w") as f:
    f.write(svg[: len(svg) // 2])
    f.flush()
    if os.environ.get("FAKE_FAIL") == "dvisvgm":
        os._exit(137)
    pause("dvisvgm")
    f.write(svg[len(svg) // 2 :])
"""

FAKE_KPSEWHICH = """\
import os, sys
if sys.argv[1:] in (["-var-value=TEXMFROOT"], ["-var-value=SELFAUTOPARENT"]):
    print(os.environ.get("FAKE_TEXMFROOT", ""))
"""


class Tools:
    """The stand-in tools on ``PATH``, their compile log and their signals."""

    def __init__(self, root: Path) -> None:
        self.bin = root / "bin"
        self.log = root / "compiles.log"
        self.signals = root / "signals"
        self.bin.mkdir(parents=True)
        self.signals.mkdir()
        self.tree = root / "texlive"
        (self.tree / "tlpkg").mkdir(parents=True)
        self.packages("one release")
        for name in ("latex", "xelatex", "dvisvgm", "kpsewhich"):
            self.install(name, "1")

    def install(self, name: str, version: str, extra: str = "freetype 1") -> None:
        """Install the stand-in ``name``, reporting ``version`` (and dvisvgm,
        ``extra`` among its libraries).
        """
        body = {"dvisvgm": FAKE_DVISVGM, "kpsewhich": FAKE_KPSEWHICH}.get(
            name, FAKE_LATEX
        )
        made = self.bin / f".{name}"
        made.write_text(
            f"#!{sys.executable}\nVERSION = {version!r}\nEXTRA = {extra!r}\n"
            + body.format(pause=_PAUSE)
        )
        made.chmod(0o755)
        os.replace(made, self.bin / name)

    def packages(self, text: str) -> None:
        """The TeX tree's package database holds ``text``."""
        (self.tree / "tlpkg" / "texlive.tlpdb").write_text(text)

    def compiles(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def wait_paused(self) -> None:
        until = time.time() + WAIT_S
        while not (self.signals / "paused").exists():
            assert time.time() < until, "the paused step never paused"
            time.sleep(0.01)

    def resume(self) -> None:
        (self.signals / "resume").touch()


@pytest.fixture
def tools(tmp_path, monkeypatch) -> Tools:
    found = Tools(tmp_path / "tools")
    monkeypatch.setenv("PATH", f"{found.bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_TEX_LOG", str(found.log))
    monkeypatch.setenv("FAKE_SIGNAL", str(found.signals))
    monkeypatch.setenv("FAKE_TEXMFROOT", str(found.tree))
    monkeypatch.delenv("FAKE_PAUSE", raising=False)
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("TEXINPUTS", raising=False)
    monkeypatch.delenv(typeset_cache.DIR_ENV, raising=False)
    return found


# Processes sharing one directory: through the cache (``typeset_dir``), or
# as CE's cache is shared, one Tex directory (``tex_dir``).
WORKER = """\
import sys, time
from pathlib import Path
from arachne import config
from arachne.mobject.text.tex_mobject import SingleStringMathTex
how, shared, media, *strings = sys.argv[1:]
config.media_dir = media
config.verbosity = "WARNING"
if how == "typeset_dir":
    config.typeset_dir = shared
else:
    config.tex_dir = shared
for s in strings:
    assert SingleStringMathTex(s).submobjects, s
print("ok")
"""

SHARED = ["typeset_dir", "tex_dir"]


def start(
    how: str, shared: Path, media: Path, *strings: str, pause: str | None = None
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.pop("FAKE_PAUSE", None)
    if pause is not None:
        env["FAKE_PAUSE"] = pause
    return subprocess.Popen(
        [sys.executable, "-c", WORKER, how, str(shared), str(media), *strings],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def finish(process: subprocess.Popen[str]) -> tuple[int, str]:
    out, _ = process.communicate(timeout=WAIT_S * 2)
    return process.returncode, out


def succeeded(result: tuple[int, str]) -> bool:
    return result[0] == 0 and result[1].endswith("ok\n")


def failed_with(result: tuple[int, str], error: str) -> bool:
    return result[0] != 0 and error in result[1]


def published(shared: Path) -> list[Path]:
    return sorted((shared / "tex").glob("*"))


# The races, each through the cache and as CE shares one Tex directory.


@pytest.mark.parametrize("how", SHARED)
def test_a_process_never_reads_an_svg_another_is_still_writing(tools, tmp_path, how):
    shared = tmp_path / "shared"
    writer = start(how, shared, tmp_path / "a", "x", pause="dvisvgm")
    tools.wait_paused()
    # The writer is half way through the SVG; the reader typesets the same.
    reader = finish(start(how, shared, tmp_path / "b", "x"))
    tools.resume()
    wrote = finish(writer)
    if how == "tex_dir":
        assert failed_with(reader, "ParseError"), reader[1]
        return
    assert succeeded(reader), reader[1]
    assert succeeded(wrote), wrote[1]
    # Both typeset the string; one file holds it.
    assert len(tools.compiles()) == 2
    assert [p.suffix for p in published(shared)] == [".svg"]
    assert list((shared / "scratch").iterdir()) == []


@pytest.mark.parametrize("how", SHARED)
def test_an_svg_a_killed_process_left_half_written_is_never_read(tools, tmp_path, how):
    shared = tmp_path / "shared"
    writer = start(how, shared, tmp_path / "a", "x", pause="dvisvgm")
    tools.wait_paused()
    writer.send_signal(signal.SIGKILL)
    writer.wait(timeout=WAIT_S)
    after = finish(start(how, shared, tmp_path / "b", "x"))
    if how == "tex_dir":
        assert failed_with(after, "ParseError"), after[1]
        return
    assert succeeded(after), after[1]
    (svg,) = published(shared)
    assert svg.read_text().endswith("</svg>")


@pytest.mark.parametrize("how", SHARED)
def test_a_process_s_cleanup_never_removes_another_s_files(tools, tmp_path, how):
    shared = tmp_path / "shared"
    # ``y``'s compile is done and its output waits for dvisvgm, when ``x``
    # is typeset whole beside it, cleanup included.
    slow = start(how, shared, tmp_path / "a", "y", pause="latex")
    tools.wait_paused()
    fast = finish(start(how, shared, tmp_path / "b", "x"))
    tools.resume()
    slowed = finish(slow)
    assert succeeded(fast), fast[1]
    if how == "tex_dir":
        assert failed_with(slowed, "converting"), slowed[1]
        return
    assert succeeded(slowed), slowed[1]
    assert [p.suffix for p in published(shared)] == [".svg", ".svg"]
    assert list((shared / "scratch").iterdir()) == []


def _killed_mid_write(monkeypatch, module):
    """``module.manimpango.text2svg`` writes half an SVG and dies, as a
    process killed while Pango wrote does.
    """

    def half(*args, **kwargs):
        path = next(a for a in args if isinstance(a, str) and a.endswith(".svg"))
        Path(path).write_text("<svg xmlns='http://www.w3.org/2000/svg'><path d='M0")
        raise KeyboardInterrupt

    monkeypatch.setattr(module.manimpango, "text2svg", half)


@pytest.mark.parametrize("how", ["typeset_dir", "text_dir"])
def test_a_text_svg_left_half_written_is_never_read(tmp_path, monkeypatch, how):
    from arachne.mobject.text import text_mobject

    with tempconfig({how: str(tmp_path / "shared")}):
        with monkeypatch.context() as patched:
            _killed_mid_write(patched, text_mobject)
            with pytest.raises(KeyboardInterrupt):
                Text("half")
        if how == "text_dir":
            with pytest.raises(ElementTree.ParseError):
                Text("half")
            return
        assert len(Text("half").submobjects) == 4


# A Text writer paused half way through its SVG, as Pango leaves it while it
# writes, in a process of its own.
TEXT_WORKER = """\
import os, sys, time
from pathlib import Path
from arachne import config
from arachne.mobject.text import text_mobject
how, shared, media, pause = sys.argv[1:]
config.media_dir = media
config.verbosity = "WARNING"
setattr(config, how, shared)
if pause == "pause":
    write = text_mobject.manimpango.text2svg

    def paused(*args, **kwargs):
        path = next(a for a in args if isinstance(a, str) and a.endswith(".svg"))
        Path(path).write_text("<svg xmlns='http://www.w3.org/2000/svg'><path d='M0")
        signals = Path(os.environ["FAKE_SIGNAL"])
        (signals / "paused").touch()
        until = time.time() + 60
        while not (signals / "resume").exists() and time.time() < until:
            time.sleep(0.01)
        return write(*args, **kwargs)

    text_mobject.manimpango.text2svg = paused
assert text_mobject.Text("shared").submobjects
print("ok")
"""


@pytest.mark.parametrize("how", ["typeset_dir", "text_dir"])
def test_a_process_never_reads_a_text_svg_another_is_still_writing(
    tools, tmp_path, how
):
    shared = tmp_path / "shared"

    def text(media: str, pause: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-c", TEXT_WORKER, how, str(shared), media, pause],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    writer = text(str(tmp_path / "a"), "pause")
    tools.wait_paused()
    reader = finish(text(str(tmp_path / "b"), "-"))
    tools.resume()
    wrote = finish(writer)
    if how == "text_dir":
        assert failed_with(reader, "ParseError"), reader[1]
        return
    assert succeeded(reader), reader[1]
    assert succeeded(wrote), wrote[1]
    assert len(list((shared / "text").iterdir())) == 1


@pytest.mark.parametrize("step", ["dvisvgm", "latex"])
def test_a_tool_that_fails_part_way_leaves_nothing_kept(
    tools, tmp_path, monkeypatch, step
):
    shared = tmp_path / "shared"
    with tempconfig({"typeset_dir": str(shared)}):
        monkeypatch.setenv("FAKE_FAIL", step)
        with pytest.raises((ValueError, RuntimeError)):
            _typeset()
        assert not (shared / "tex").exists()
        monkeypatch.delenv("FAKE_FAIL")
        assert _typeset().read_text().endswith("</svg>")


@pytest.mark.parametrize("make", [lambda: Text("short"), lambda: MarkupText("short")])
def test_a_text_svg_pango_wrote_short_without_an_error_is_not_kept(
    tmp_path, monkeypatch, make
):
    from arachne.mobject.text import text_mobject

    def short(*args, **kwargs):
        path = next(a for a in args if isinstance(a, str) and a.endswith(".svg"))
        Path(path).write_text("<svg xmlns='http://www.w3.org/2000/svg'><path d='M0")
        return path

    shared = tmp_path / "shared"
    with tempconfig({"typeset_dir": str(shared)}):
        with monkeypatch.context() as patched:
            patched.setattr(text_mobject.manimpango, "text2svg", short)
            patched.setattr(text_mobject.MarkupUtils, "text2svg", short)
            with pytest.raises(ValueError, match="no whole SVG"):
                make()
        assert not (shared / "text").exists()
        assert make().submobjects


@pytest.mark.parametrize(
    ("how", "make"),
    [
        ("typeset_dir", lambda: Text("kept")),
        ("text_dir", lambda: Text("kept")),
        ("typeset_dir", lambda: MarkupText("<b>kept</b>")),
        ("text_dir", lambda: MarkupText("<b>kept</b>")),
    ],
)
def test_a_text_svg_once_made_is_never_written_again(tmp_path, how, make):
    """CE rewrites a Text's SVG in place at every construction, a cache hit
    included, while another process may be reading it.
    """
    shared = tmp_path / "shared"
    with tempconfig({how: str(shared)}):
        make()
        (svg,) = [p for p in shared.rglob("*.svg") if "scratch" not in p.parts]
        os.utime(svg, ns=(0, 0))
        make()
        rewritten = svg.stat().st_mtime_ns != 0
    assert rewritten == (how == "text_dir")


@pytest.mark.parametrize("named", [True, False])
def test_an_svg_is_parsed_through_no_file_another_process_shares(
    tmp_path, monkeypatch, named
):
    """CE parses an SVG through a copy beside it, ``<name>_.svg``, which two
    processes loading one SVG write, parse and remove at once; it still does
    with no typesetting directory named.
    """
    from arachne.mobject.svg import svg_mobject

    svg = tmp_path / "shape.svg"
    svg.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'>"
        "<path d='M0 0L4 0L4 5Z'/></svg>"
    )
    parse = svg_mobject.se.SVG.parse

    def other_process_done_first(source, *args, **kwargs):
        # Another process loading the same SVG removes its copy beside it.
        (tmp_path / "shape_.svg").unlink(missing_ok=True)
        return parse(source, *args, **kwargs)

    monkeypatch.setattr(svg_mobject.se.SVG, "parse", other_process_done_first)
    if not named:
        with pytest.raises(FileNotFoundError):
            SVGMobject(svg, use_svg_cache=False)
        return
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        shape = SVGMobject(svg, use_svg_cache=False)
    assert len(shape.submobjects) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shape.svg"]


# The key


def _typeset(expression: str = "x", **template: object) -> Path:
    from arachne.utils.tex_file_writing import tex_to_svg_file

    return tex_to_svg_file(expression, tex_template=TexTemplate(**template))


def test_one_expression_under_two_compilers_keys_apart(tools, tmp_path):
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        dvi = _typeset(tex_compiler="latex", output_format=".dvi")
        xdv = _typeset(tex_compiler="xelatex", output_format=".xdv")
        assert dvi != xdv
        assert _typeset(tex_compiler="latex", output_format=".dvi") == dvi
    assert [line.split()[0] for line in tools.compiles()] == ["latex", "xelatex"]


def test_another_preamble_or_tool_version_keys_apart(tools, tmp_path):
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        first = _typeset()
        assert _typeset(preamble=r"\usepackage{amsmath}") != first
        # The same process, once the compiler, then dvisvgm, was upgraded.
        tools.install("latex", "2")
        assert _typeset() != first
        tools.install("latex", "1")
        tools.install("dvisvgm", "2")
        assert _typeset() != first
        # A library dvisvgm reports beside its own version.
        tools.install("dvisvgm", "1", extra="freetype 2")
        assert _typeset() != first
        tools.install("dvisvgm", "1")
        assert _typeset() == first


def test_the_tex_tree_s_packages_and_search_path_key_apart(
    tools, tmp_path, monkeypatch
):
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        first = _typeset()
        tools.packages("one release, a package updated")
        assert _typeset() != first
        tools.packages("one release")
        monkeypatch.setenv("TEXINPUTS", "/elsewhere:")
        assert _typeset() != first
        monkeypatch.delenv("TEXINPUTS")
        assert _typeset() == first


def test_with_no_package_database_tex_is_never_shared_across_hosts(
    tools, tmp_path, monkeypatch
):
    import socket

    monkeypatch.setenv("FAKE_TEXMFROOT", str(tmp_path / "nowhere"))
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        here = _typeset()
        monkeypatch.setattr(socket, "gethostname", lambda: "another-host")
        assert _typeset() != here


def test_a_working_directory_tex_can_read_from_keys_apart(tools, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    (work / "arachne.cfg").write_text("[CLI]\n")
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        bare = _typeset()
        # A file TeX finds by its bare name, beside the process.
        (work / "amsmath.sty").write_text("% a local package\n")
        local = _typeset()
        assert local != bare
        (work / "amsmath.sty").write_text("% the local package, edited\n")
        assert _typeset() != local
        (work / "amsmath.sty").unlink()
        assert _typeset() == bare
        # A source that reads a file by name reads it from here.
        reads = _typeset(preamble=r"\input{macros}")
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        assert _typeset(preamble=r"\input{macros}") != reads
        assert _typeset() == bare


def test_two_media_directories_naming_one_typesetting_directory_share_it(
    tools, tmp_path
):
    shared = tmp_path / "shared"
    for media in ("one", "two"):
        with tempconfig(
            {"media_dir": str(tmp_path / media), "typeset_dir": str(shared)}
        ):
            SingleStringMathTex("x")
            Text("shared")
    assert len(tools.compiles()) == 1
    assert len(list((shared / "text").glob("*.svg"))) == 1
    assert not any((tmp_path / media).exists() for media in ("one", "two"))


def test_unset_the_media_directory_holds_the_typesetting_as_ce_s_does(tools, tmp_path):
    with tempconfig({"media_dir": str(tmp_path / "media")}):
        assert config.typeset_dir is None
        SingleStringMathTex("x")
        Text("own")
    assert len(list((tmp_path / "media" / "Tex").glob("*.svg"))) == 1
    assert len(list((tmp_path / "media" / "texts").glob("*.svg"))) == 1


@pytest.mark.parametrize(
    "other",
    [
        {"t2c": {"ex": "#FF0000"}},
        {"font": "Noto Sans"},
        {"line_spacing": 2},
    ],
)
def test_text_keys_apart_on_what_it_is_typeset_with(tmp_path, other):
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        Text("text", warn_missing_font=False)
        Text("text", warn_missing_font=False, **other)
    assert len(list((tmp_path / "shared" / "text").glob("*.svg"))) == 2


def test_text_keys_apart_on_the_pixel_width_its_lines_wrap_at(tmp_path):
    for width in (854, 1920):
        with tempconfig(
            {"typeset_dir": str(tmp_path / "shared"), "pixel_width": width}
        ):
            Text("wrapped")
    assert len(list((tmp_path / "shared" / "text").glob("*.svg"))) == 2


def test_the_fonts_text_is_shaped_with_are_resolved_fallbacks_included():
    """Latin and CJK text in the default font shapes with two fonts at
    least: the one Pango resolves the family to and the one it falls back
    to for the characters that font lacks; each by its content.
    """
    latin = typeset_fonts.text_fonts([("Hello", "", "NORMAL", "NORMAL")], 10.0)
    mixed = typeset_fonts.text_fonts([("Hello 世界", "", "NORMAL", "NORMAL")], 10.0)
    assert len(latin) == 1
    assert len(mixed) >= 2
    assert set(map(tuple, latin)) < set(map(tuple, mixed))
    assert all(kind == "font" and len(digest) == 64 for kind, _, digest in mixed)


def test_text_keys_apart_when_a_font_it_is_shaped_with_changes(tmp_path, monkeypatch):
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        Text("fonts")
        monkeypatch.setattr(typeset_fonts, "_DIGESTS", {})
        monkeypatch.setattr(typeset_fonts, "_digest", lambda data: "another file")
        Text("fonts")
    assert len(list((tmp_path / "shared" / "text").glob("*.svg"))) == 2


def test_without_pango_s_resolution_text_is_never_shared_across_hosts(
    tmp_path, monkeypatch
):
    import socket

    def unavailable():
        raise OSError("no Pango to ask")

    monkeypatch.setattr(typeset_fonts, "_pango", unavailable)
    fonts = typeset_fonts.text_fonts([("text", "", "NORMAL", "NORMAL")], 10.0)
    assert [kind for kind, *_ in fonts] == ["host"]
    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        Text("hosts")
        monkeypatch.setattr(socket, "gethostname", lambda: "another-host")
        Text("hosts")
    assert len(list((tmp_path / "shared" / "text").glob("*.svg"))) == 2


def test_the_text_toolchain_names_pango_harfbuzz_and_cairo():
    names = typeset_cache.text_toolchain()
    assert [name for name, _ in names] == ["manimpango", "pango", "harfbuzz", "cairo"]
    assert all(value for _, value in names)


# A hit is a fresh compile.

SCENE = """\
from arachne import MathTex, Scene, Text, VGroup


class Typeset(Scene):
    def construct(self):
        self.add(VGroup(MathTex(r"\\int_0^1 x^2\\,dx"), Text("area")).arrange())
        self.wait(0.2)
"""


def test_a_hit_builds_and_draws_what_a_fresh_compile_does(tmp_path):
    from arachne.session import Session
    from arachne.timeline.execution import play_keys

    path = tmp_path / "typeset.py"
    path.write_text(SCENE)
    shared = tmp_path / "shared"
    small = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

    def opened(**options: object) -> tuple[tuple[bytes, ...], bytes]:
        session = Session.open(path, "Typeset", config={**small, **options})
        try:
            return play_keys(session.record), session.frame(0.1).tobytes()
        finally:
            session.close()

    fresh = opened(media_dir=str(tmp_path / "fresh"))
    missed = opened(media_dir=str(tmp_path / "missed"), typeset_dir=str(shared))
    hit = opened(media_dir=str(tmp_path / "hit"), typeset_dir=str(shared))
    assert fresh == missed == hit
    # The files are CE's own, byte for byte.
    ce = sorted(
        p.read_bytes()
        for p in (tmp_path / "fresh").rglob("*.svg")
        if p.parent.name in ("Tex", "texts")
    )
    kept = sorted(p.read_bytes() for p in shared.glob("t*/*.svg"))
    assert kept == ce


def test_processes_typesetting_at_once_with_the_real_tools_all_succeed(tmp_path):
    code = textwrap.dedent(
        """\
        import sys, time
        from arachne import MathTex, Text, config
        config.media_dir = sys.argv[1]
        config.typeset_dir = sys.argv[2]
        while time.time() < float(sys.argv[3]):
            time.sleep(0.005)
        for s in [r"\\frac{a}{b}", r"e^{i\\pi}+1=0", r"\\sqrt{x^2+1}"]:
            assert MathTex(s).submobjects
            assert Text("at once " + s[:4]).submobjects
        print("ok")
        """
    )
    shared = tmp_path / "shared"
    at = time.time() + 8
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path / f"m{i}"), str(shared), str(at)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for i in range(4)
    ]
    results = [finish(p) for p in processes]
    assert all(map(succeeded, results)), results
    assert len(list((shared / "tex").iterdir())) == 3
    assert len(list((shared / "text").iterdir())) == 3
    assert list((shared / "scratch").iterdir()) == []


# Sessions


CLIPPED_TEX = """\
from arachne import Scene
from arachne.mobject.text.tex_mobject import SingleStringMathTex


class Typeset(Scene):
    def construct(self):
        self.add(SingleStringMathTex("x"))
        self.wait(0.2)
"""


@pytest.mark.parametrize("named", [False, True])
def test_a_session_s_render_typesets_nothing_its_open_did(tools, tmp_path, named):
    """Through the typesetting directory; with none named, the render
    typesets into its own media directory again, as CE's does.
    """
    from arachne.server.dispatch import Dispatcher

    path = tmp_path / "typeset.py"
    path.write_text(CLIPPED_TEX)
    settings = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}
    if named:
        settings["typeset_dir"] = str(tmp_path / "shared")
    server = Dispatcher()
    server.handle("open", {"file": str(path), "scene": "Typeset", "config": settings})
    try:
        assert len(tools.compiles()) == 1
        server.handle("render", {"out": str(tmp_path / "render.mp4")})
        assert len(tools.compiles()) == (1 if named else 2)
        assert (tmp_path / "render.mp4").exists()
    finally:
        assert server.session is not None
        server.session.close()


def test_reads_from_the_typesetting_directory_are_not_the_scene_s_inputs(tmp_path):
    from arachne.timeline import inputs

    with tempconfig({"typeset_dir": str(tmp_path / "shared")}):
        dirs = inputs.engine_dirs(config)
    assert os.path.realpath(tmp_path / "shared") in dirs


def test_with_the_speed_levers_off_the_named_directory_is_not_used(tools, tmp_path):
    with tempconfig(
        {
            "media_dir": str(tmp_path / "media"),
            "typeset_dir": str(tmp_path / "shared"),
            "speed_levers": False,
        }
    ):
        SingleStringMathTex("x")
        Text("own")
    assert not (tmp_path / "shared").exists()
    assert len(list((tmp_path / "media" / "Tex").glob("*.svg"))) == 1
    assert len(list((tmp_path / "media" / "texts").glob("*.svg"))) == 1
