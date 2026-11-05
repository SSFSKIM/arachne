"""An execution's identity (its source, the files it read, its seed), and
the comparison of two executions of one source at the instants they share.

Every scene here is written into the test's own directory, with the data
files it reads, so that editing them between executions is the test's to do.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import arachne
from arachne.session import Session
from arachne.timeline import inputs
from arachne.timeline.execution import (
    ExecutionSummary,
    Incomparable,
    compare,
    summarize,
)
from arachne.timeline.source import SourceLine
from arachne.utils.module_ops import get_module

#: A small frame at 4 fps, so a session costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}

HEADER = """\
from pathlib import Path

from arachne import *

HERE = Path(__file__).parent
"""

READS = (
    HEADER
    + """
class Reads(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        offset = float((HERE / "data.txt").read_text())
        self.play(dot.animate.shift(offset * UP), run_time=0.5)  # @offset
        self.wait(0.25)
"""
)


@pytest.fixture
def scene_dir(tmp_path):
    """A directory for scenes, whose modules are forgotten afterwards: a
    helper module stays in ``sys.modules`` under its plain name, where the
    next test's helper of that name would find it.
    """
    directory = tmp_path / "scenes"
    directory.mkdir()
    yield directory
    _forget_modules(tmp_path)


def _forget_modules(directory: Path) -> None:
    root = os.path.realpath(directory) + os.sep
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and os.path.realpath(file).startswith(root):
            del sys.modules[name]


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def _open(path: Path, scene: str) -> Session:
    media = path.parent.parent / "media"
    session = Session.open(path, scene, config={**SMALL, "media_dir": str(media)})
    return session


def _line(path: Path, tag: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The identity


def test_the_files_a_scene_reads_are_its_inputs_and_the_engines_caches_are_not(
    scene_dir,
):
    _write(scene_dir, "data.txt", "0.5")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class ReadsAndWrites(Scene):
    def construct(self):
        offset = float((HERE / "data.txt").read_text())
        (HERE / "written.txt").write_text("for nobody")
        # The engine writes these into its media directory and reads them
        # back: its own caches, not the scene's inputs.
        label = Text("label")
        formula = MathTex("x^2")
        dot = Dot()
        self.add(dot, label, formula)
        self.play(dot.animate.shift(offset * RIGHT), run_time=0.5)
""",
    )
    session = _open(scene, "ReadsAndWrites")
    identity = session.record.identity
    assert identity.input_digests == (("data.txt", _sha256(scene_dir / "data.txt")),)
    assert identity.seed == session.scene.random_seed
    # The caches were read: the scene's text and formula are drawn from them.
    media = scene_dir.parent / "media"
    assert any(media.rglob("*.svg"))


def test_a_file_the_scene_reads_inside_the_media_directory_is_not_an_input(
    scene_dir,
):
    media = scene_dir.parent / "media"
    media.mkdir()
    (media / "made_by_the_engine.txt").write_text("1")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class ReadsMedia(Scene):
    def construct(self):
        (HERE.parent / "media" / "made_by_the_engine.txt").read_text()
        self.wait(0.25)
""",
    )
    assert _open(scene, "ReadsMedia").record.identity.input_digests == ()


def test_a_media_directory_given_as_a_path_object_is_excluded_too(scene_dir):
    """The configuration keeps a directory as it was given when it is not a
    ``pathlib.Path`` (a ``py.path.local``, say), which ``get_dir`` refuses:
    the scene still records, and the directory's files are no inputs.
    """
    py_path = pytest.importorskip("py.path")
    media = scene_dir.parent / "media"
    media.mkdir()
    (media / "made_by_the_engine.txt").write_text("1")
    data = _write(scene_dir, "data.txt", "1")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class ReadsBoth(Scene):
    def construct(self):
        (HERE.parent / "media" / "made_by_the_engine.txt").read_text()
        (HERE / "data.txt").read_text()
        self.wait(0.25)
""",
    )
    session = Session.open(
        scene, "ReadsBoth", config={**SMALL, "media_dir": py_path.local(media)}
    )
    assert session.record.identity.input_digests == (("data.txt", _sha256(data)),)


def test_a_directory_that_does_not_resolve_is_left_out_of_the_exclusions():
    class Unresolvable:
        def __getattr__(self, name):
            raise ValueError(f"{name} does not resolve")

    assert inputs.engine_dirs(Unresolvable()) == []


def _resolved(directory: Path) -> str:
    return os.path.realpath(directory)


def test_a_cache_directory_built_from_another_field_is_excluded(tmp_path):
    from arachne import config, tempconfig

    assets = tmp_path / "assets"
    with tempconfig({"assets_dir": str(assets), "tex_dir": "{assets_dir}/cache/Tex"}):
        dirs = inputs.engine_dirs(config)
    assert _resolved(assets / "cache" / "Tex") in dirs
    # The assets themselves are the scene's to read.
    assert _resolved(assets) not in dirs


def test_a_cache_directory_built_from_the_quality_is_excluded(tmp_path):
    from arachne import config, tempconfig

    with tempconfig(
        {
            "pixel_height": 480,
            "pixel_width": 854,
            "frame_rate": 15,
            "text_dir": str(tmp_path / "texts" / "{quality}"),
        }
    ):
        dirs = inputs.engine_dirs(config)
    assert _resolved(tmp_path / "texts" / "480p15") in dirs


def test_a_directory_cut_at_a_module_or_scene_name_is_excluded_up_to_it(tmp_path):
    from arachne import config, tempconfig

    with tempconfig(
        {"partial_movie_dir": str(tmp_path / "parts" / "{module_name}" / "x")}
    ):
        dirs = inputs.engine_dirs(config)
    assert _resolved(tmp_path / "parts") in dirs


def test_a_template_that_does_not_resolve_excludes_nothing(tmp_path):
    from arachne import config, tempconfig

    data = tmp_path / "data"
    with tempconfig(
        {
            "text_dir": str(data / "{missing.attribute}"),
            "tex_dir": str(data / "{module_name}_cache"),
        }
    ):
        dirs = inputs.engine_dirs(config)
    assert not [path for path in dirs if path.startswith(_resolved(data))]


@pytest.mark.parametrize("here", [".", Path(".")], ids=["str", "path"])
def test_a_media_directory_that_is_the_working_directory(
    scene_dir, tmp_path, monkeypatch, here
):
    """``get_dir`` gives ``Path(".")`` for it, a path of no parts. Elsewhere
    it is excluded as the working directory; where it holds the scene's
    directory, it is not, and the scene's inputs stay inputs.
    """
    from arachne import config, tempconfig

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with tempconfig({"media_dir": here}):
        assert _resolved(elsewhere) in inputs.engine_dirs(config)

    data = _write(scene_dir, "data.txt", "1")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class ReadsHere(Scene):
    def construct(self):
        (HERE / "data.txt").read_text()
        self.wait(0.25)
""",
    )
    monkeypatch.chdir(scene_dir)
    session = Session.open(scene, "ReadsHere", config={**SMALL, "media_dir": here})
    assert session.record.identity.input_digests == (("data.txt", _sha256(data)),)


HELPED = (
    HEADER
    + """
from shapes_helper import make_dot
from helpers_pkg import sub


class Helped(Scene):
    def construct(self):
        import offsets_helper

        dot = make_dot()
        self.add(dot)
        self.play(dot.animate.shift(offsets_helper.OFFSET * sub.UNIT), run_time=0.5)
"""
)


def _helped(directory: Path) -> Path:
    _write(
        directory,
        "shapes_helper.py",
        """\
        from arachne import Dot


        def make_dot():
            return Dot()
        """,
    )
    _write(directory, "offsets_helper.py", "OFFSET = 0.5\n")
    _write(directory, "helpers_pkg/__init__.py", "")
    _write(
        directory,
        "helpers_pkg/sub.py",
        """\
        from arachne import RIGHT

        UNIT = RIGHT
        """,
    )
    _write(directory, "unrelated.py", "VALUE = 1\n")
    return _write(directory, "scene.py", HELPED)


def test_the_source_digest_covers_the_helper_modules_the_scene_imported(scene_dir):
    scene = _helped(scene_dir)
    first = _open(scene, "Helped").record.identity
    assert first.source_digest is not None
    # Their source is not among the files read.
    assert first.input_digests == ()

    def digest_after_editing(name: str) -> str | None:
        path = scene_dir / name
        original = path.read_text()
        path.write_text(original + "# edited\n")
        try:
            return _open(scene, "Helped").record.identity.source_digest
        finally:
            path.write_text(original)

    # A module the scene imported at its top, one it imported in construct,
    # a package and its submodule: editing any of them changes the digest.
    for name in (
        "scene.py",
        "shapes_helper.py",
        "offsets_helper.py",
        "helpers_pkg/__init__.py",
        "helpers_pkg/sub.py",
    ):
        assert digest_after_editing(name) != first.source_digest, name
    # A module beside it that it never imported does not.
    assert digest_after_editing("unrelated.py") == first.source_digest


def test_the_same_source_in_another_directory_has_the_same_digest(scene_dir, tmp_path):
    first = _open(_helped(scene_dir), "Helped").record.identity
    # The first directory's helpers stay imported under their names, as
    # they would in a long-lived process: the copy's digest is its own.
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(scene_dir, elsewhere)
    second = _open(elsewhere / "scene.py", "Helped").record.identity
    assert second.source_digest == first.source_digest


COMPUTED = (
    HEADER
    + """
import importlib

TOP = importlib.import_module("top_" + "helper")


class Computed(Scene):
    def construct(self):
        late = importlib.import_module("late_" + "helper")
        dunder = __import__("dunder_" + "helper")
        dot = Dot()
        self.add(dot)
        offset = TOP.VALUE + late.VALUE + dunder.VALUE
        self.play(dot.animate.shift(offset * RIGHT), run_time=0.5)
"""
)


def test_a_module_reached_only_by_a_computed_name_is_outside_the_source_digest(
    scene_dir,
):
    """The digest decides which executions are compared, so it is a function
    of the files alone: a module the import statements do not name is not in
    it, whether or not the process has imported it, and a change to it shows
    as a divergence that names no input.
    """
    for name in ("top_helper", "late_helper", "dunder_helper"):
        _write(scene_dir, f"{name}.py", "VALUE = 0.25\n")
    scene = _write(scene_dir, "scene.py", COMPUTED)
    _forget_modules(scene_dir)
    first = _open(scene, "Computed").record.identity.source_digest
    # Imported now, each is found in the cache the second time.
    assert _open(scene, "Computed").record.identity.source_digest == first
    for name in ("top_helper", "late_helper", "dunder_helper"):
        path = scene_dir / f"{name}.py"
        original = path.read_text()
        path.write_text(original + "# edited\n")
        try:
            edited = _open(scene, "Computed").record.identity.source_digest
        finally:
            path.write_text(original)
        assert edited == first, name


MODULE_LEVEL = (
    HEADER
    + """
from produced_helper import UNIT

SIZE = float((HERE / "loaded.txt").read_text())


class Produced(Scene):
    def construct(self):
        offset = float((HERE / "data.txt").read_text())
        dot = Dot(radius=SIZE)
        self.add(dot)
        self.play(dot.animate.shift(offset * UNIT), run_time=0.5)
"""
)


def test_a_session_and_a_render_of_one_scene_have_one_identity(scene_dir):
    """Whichever producer recorded it, an execution's identity is the same,
    so a render compares with a session's record: the files read while the
    scene file loads are no one's inputs.
    """
    from arachne import tempconfig

    _write(scene_dir, "produced_helper.py", "from arachne import RIGHT as UNIT\n")
    _write(scene_dir, "loaded.txt", "0.1")
    data = _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", MODULE_LEVEL)
    session = _open(scene, "Produced").record.identity

    # Loaded as the command line loads a scene file.
    path = list(sys.path)
    try:
        with warnings.catch_warnings():
            module = get_module(scene)
    finally:
        sys.path[:] = path
    media = scene_dir.parent / "media"
    with tempconfig(
        {**SMALL, "dry_run": True, "disable_caching": True, "media_dir": str(media)}
    ):
        rendered = module.Produced()
        rendered.render()
    assert rendered.record.identity == session
    assert session.input_digests == (("data.txt", _sha256(data)),)


def test_the_source_digest_does_not_depend_on_what_the_process_imported_before(
    scene_dir,
):
    _write(scene_dir, "unused_helper.py", "VALUE = 1\n")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
if False:
    import unused_helper


class Guarded(Scene):
    def construct(self):
        self.wait(0.25)
""",
    )
    before = _open(scene, "Guarded").record.identity.source_digest
    sys.path.insert(0, str(scene_dir))
    try:
        import unused_helper  # noqa: F401
    finally:
        sys.path.remove(str(scene_dir))
    assert _open(scene, "Guarded").record.identity.source_digest == before


def test_outside_a_recording_the_hook_records_nothing(scene_dir):
    opener = _write(
        scene_dir,
        "opener.py",
        """\
        def read(path):
            with open(path) as file:
                return file.read()
        """,
    )
    first = _write(scene_dir, "first.txt", "1")
    second = _write(scene_dir, "second.txt", "2")
    sys.path.insert(0, str(scene_dir))
    try:
        import opener as module
    finally:
        sys.path.remove(str(scene_dir))
    assert module.__file__ == str(opener)
    watcher = inputs.Inputs(str(scene_dir), excluded=())
    watcher.start()
    try:
        assert inputs.active()
        module.read(first)
    finally:
        watcher.stop()
    assert not inputs.active()
    module.read(second)
    assert watcher.input_digests(sources=()) == (("first.txt", _sha256(first)),)


def test_a_scene_that_fails_leaves_the_hook_inactive(scene_dir):
    _write(scene_dir, "data.txt", "1")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Fails(Scene):
    def construct(self):
        (HERE / "data.txt").read_text()
        raise RuntimeError("the scene's own error")
""",
    )
    unwatched = (builtins.__import__, importlib.import_module)
    with pytest.raises(RuntimeError, match="the scene's own error"):
        _open(scene, "Fails")
    assert not inputs.active()
    # The process's import functions are untouched.
    assert (builtins.__import__, importlib.import_module) == unwatched


# Comparing executions


def test_the_same_script_executed_twice_does_not_diverge(scene_dir):
    _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", READS)
    first = _open(scene, "Reads")
    second = _open(scene, "Reads")
    assert first.compare(second.summary()) is None
    assert first.divergences == []


def test_an_edited_data_file_diverges_at_the_first_play_it_changed_and_is_named(
    scene_dir,
):
    data = _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", READS)
    before = _open(scene, "Reads")
    data.write_text("1.5")
    after = _open(scene, "Reads")
    divergence = before.compare(after.summary())
    assert divergence is not None
    assert divergence.code == "replay.diverged"
    assert divergence.kind == "execution"
    assert divergence.play == 1
    assert divergence.source_line == SourceLine("scene.py", _line(scene, "offset"))
    assert divergence.changed_inputs == ("data.txt",)
    assert "Dot" in divergence.reason
    assert before.divergences == [divergence]


def test_an_unseeded_random_generator_diverges_at_the_first_play_it_moves(
    scene_dir,
):
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
import random


class Unseeded(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        offset = random.Random().random()
        self.play(dot.animate.shift(offset * UP), run_time=0.5)  # @random
""",
    )
    first = _open(scene, "Unseeded")
    divergence = first.compare(_open(scene, "Unseeded").summary())
    assert divergence is not None
    assert divergence.play == 1
    assert divergence.source_line == SourceLine("scene.py", _line(scene, "random"))
    assert divergence.changed_inputs == ()


def test_a_handoff_file_is_the_reading_scenes_input_and_names_its_divergence(
    scene_dir,
):
    seed = _write(scene_dir, "seed.txt", "1")
    writer = _write(
        scene_dir,
        "writer.py",
        HEADER
        + """
class Writer(Scene):
    def construct(self):
        value = float((HERE / "seed.txt").read_text())
        (HERE / "handoff.txt").write_text(str(2 * value))
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(value * RIGHT), run_time=0.5)
""",
    )
    reader = _write(
        scene_dir,
        "reader.py",
        HEADER
        + """
class Reader(Scene):
    def construct(self):
        value = float((HERE / "handoff.txt").read_text())
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(value * UP), run_time=0.5)  # @handoff
""",
    )
    wrote = _open(writer, "Writer")
    read = _open(reader, "Reader")
    # What a scene writes is not its input; what the next one reads is.
    assert [path for path, _ in wrote.record.identity.input_digests] == ["seed.txt"]
    assert [path for path, _ in read.record.identity.input_digests] == ["handoff.txt"]
    # The reader's source is its own file, not the writer's beside it.
    assert read.record.identity.source_digest != wrote.record.identity.source_digest

    seed.write_text("3")
    wrote_again = _open(writer, "Writer")
    read_again = _open(reader, "Reader")
    assert wrote.compare(wrote_again.summary()).changed_inputs == ("seed.txt",)
    divergence = read.compare(read_again.summary())
    assert divergence is not None
    assert divergence.play == 0
    assert divergence.source_line == SourceLine("reader.py", _line(reader, "handoff"))
    assert divergence.changed_inputs == ("handoff.txt",)


def test_a_value_either_side_of_a_rounding_boundary_is_no_divergence(scene_dir):
    """Content keys round to 9 decimals, so two values 2e-13 apart that
    round apart give different keys; the raw values agree within 1e-9.
    """
    weight = _write(scene_dir, "weight.txt", "1.0000000004999")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Boundary(Scene):
    def construct(self):
        dot = Dot()
        dot.weight = float((HERE / "weight.txt").read_text())
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
""",
    )
    below = _open(scene, "Boundary")
    weight.write_text("1.0000000005001")
    above = _open(scene, "Boundary")
    mine, theirs = below.summary(), above.summary()
    assert [state.key for state in mine.plays[0].instants.values()] != [
        state.key for state in theirs.plays[0].instants.values()
    ]
    assert below.compare(theirs) is None
    # Beyond the tolerance, the same scene diverges.
    weight.write_text("1.000001")
    assert below.compare(_open(scene, "Boundary").summary()) is not None


def test_two_sources_are_not_compared(scene_dir):
    _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", READS)
    first = _open(scene, "Reads").summary()
    scene.write_text(scene.read_text() + "# edited\n")
    second = _open(scene, "Reads").summary()
    with pytest.raises(ValueError, match="source"):
        compare(first, second)


def _unknown(summary: ExecutionSummary) -> ExecutionSummary:
    return dataclasses.replace(
        summary, identity=dataclasses.replace(summary.identity, source_digest=None)
    )


def test_executions_whose_source_is_unknown_are_not_compared(scene_dir):
    _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", READS)
    known = _open(scene, "Reads").summary()
    unknown = _unknown(known)
    for mine, theirs in ((unknown, unknown), (known, unknown), (unknown, known)):
        with pytest.raises(Incomparable, match="unknown"):
            compare(mine, theirs)


def test_a_source_gone_by_the_end_of_its_run_is_unknown_and_not_compared(
    scene_dir,
):
    """A scene file removed while it runs leaves its record's source unknown:
    the render goes on, and the record is compared with no execution.
    """
    _write(scene_dir, "remove.txt", "0")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Vanishing(Scene):
    def construct(self):
        if int((HERE / "remove.txt").read_text()):
            Path(__file__).unlink()
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
""",
    )
    known = _open(scene, "Vanishing")
    (scene_dir / "remove.txt").write_text("1")
    gone = _open(scene, "Vanishing")
    assert not scene.exists()
    assert gone.record.identity.source_digest is None
    with pytest.raises(Incomparable, match="unknown"):
        gone.compare(known.summary())
    with pytest.raises(Incomparable, match="unknown"):
        known.compare(gone.summary())
    # Its summary carries the unknown source to another process as such.
    carried = ExecutionSummary.from_json(gone.summary().to_json())
    assert carried.identity.source_digest is None
    with pytest.raises(Incomparable, match="unknown"):
        compare(carried, carried)
    assert gone.divergences == []
    assert known.divergences == []


def _diverged_after_editing(scene: Path, name: str, data: Path, value: str):
    """The divergence of ``scene``'s execution with ``data`` holding ``value``
    from its execution as ``data`` held before, through JSON as another
    process would hand it over.
    """
    before = _open(scene, name)
    original = data.read_text()
    data.write_text(value)
    try:
        after = _open(scene, name).summary()
    finally:
        data.write_text(original)
    return before.compare(ExecutionSummary.from_json(after.to_json()))


@pytest.mark.parametrize(
    "move",
    [
        # The frame moved before the play: only the camera holds it.
        "self.camera.frame.shift(shift * RIGHT)\n        self.wait(0.5)",
        # The frame moved by the play, which holds it as its mobject.
        "self.play(self.camera.frame.animate.shift(shift * RIGHT), run_time=0.5)",
    ],
    ids=["moved_before", "animated"],
)
def test_a_camera_that_differs_alone_diverges(scene_dir, move):
    data = _write(scene_dir, "camera.txt", "0.5")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + f"""
class Framed(MovingCameraScene):
    def construct(self):
        self.add(Dot())
        shift = float((HERE / "camera.txt").read_text())
        {move}
""",
    )
    divergence = _diverged_after_editing(scene, "Framed", data, "1.5")
    assert divergence is not None
    assert divergence.play == 0
    assert divergence.changed_inputs == ("camera.txt",)


def test_a_camera_value_and_the_draw_order_are_compared(scene_dir):
    data = _write(scene_dir, "drawing.txt", "0")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Drawn(Scene):
    def construct(self):
        flag = int((HERE / "drawing.txt").read_text())
        square, dot = Square(), Dot()
        self.add(dot, square)
        if flag == 1:
            self.camera.background_color = BLUE
        if flag == 2:
            self.add_foreground_mobject(square)
        self.wait(0.25)
""",
    )
    background = _diverged_after_editing(scene, "Drawn", data, "1")
    assert background is not None
    assert "background_color" in background.reason
    order = _diverged_after_editing(scene, "Drawn", data, "2")
    assert order is not None
    assert "foreground" in order.reason


def test_an_array_of_another_layout_but_the_same_bytes_diverges(scene_dir):
    data = _write(scene_dir, "layout.txt", "2,3 int64")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
import numpy as np

from arachne.mobject.values import FrozenArray, freeze


class Laid(Scene):
    def construct(self):
        shape, dtype = (HERE / "layout.txt").read_text().split()
        dot = Dot()
        dot.table = freeze(
            np.zeros(tuple(map(int, shape.split(","))), dtype=dtype), FrozenArray
        )
        self.add(dot)
        self.wait(0.25)
""",
    )
    for value in ("3,2 int64", "2,3 float64"):
        divergence = _diverged_after_editing(scene, "Laid", data, value)
        assert divergence is not None, value
        assert "table" in divergence.reason, value


def test_a_zero_dimensional_array_is_not_a_one_element_one(scene_dir):
    data = _write(scene_dir, "dims.txt", "0")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
import numpy as np

from arachne.mobject.values import FrozenArray, freeze


class Scalar(Scene):
    def construct(self):
        dims = int((HERE / "dims.txt").read_text())
        dot = Dot()
        dot.weight = freeze(np.zeros((1,) * dims), FrozenArray)
        self.add(dot)
        self.wait(0.25)
""",
    )
    session = _open(scene, "Scalar")
    weights = [
        value
        for node in ExecutionSummary.from_json(session.summary().to_json()).nodes
        for name, value in node.fields
        if name == "weight"
    ]
    assert weights
    assert all(weight.shape == () for weight in weights)
    divergence = _diverged_after_editing(scene, "Scalar", data, "1")
    assert divergence is not None
    assert "weight" in divergence.reason


def _background(scene_dir: Path, red: float) -> tuple[Path, Path]:
    data = _write(scene_dir, "red.txt", repr(red))
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Tinted(Scene):
    def construct(self):
        red = float((HERE / "red.txt").read_text())
        self.camera.background_color = ManimColor((red, 0.0, 0.0))
        self.add(Dot())
        self.wait(0.25)
""",
    )
    return scene, data


def test_a_camera_colour_is_compared_by_its_numbers_not_its_hex(scene_dir):
    from arachne import ManimColor

    # Either side of a hex boundary, 2e-13 apart: one colour.
    edge = 128 / 255
    below, above = edge - 1e-13, edge + 1e-13
    assert (
        ManimColor((below, 0.0, 0.0)).to_hex() != ManimColor((above, 0.0, 0.0)).to_hex()
    )
    scene, data = _background(scene_dir, below)
    assert _diverged_after_editing(scene, "Tinted", data, repr(above)) is None
    # Inside one hex bucket, 1e-4 apart: two colours.
    assert (
        ManimColor((0.3, 0.0, 0.0)).to_hex() == ManimColor((0.3001, 0.0, 0.0)).to_hex()
    )
    data.write_text(repr(0.3))
    divergence = _diverged_after_editing(scene, "Tinted", data, repr(0.3001))
    assert divergence is not None
    assert "background_color" in divergence.reason


def test_the_fixed_orientation_mobjects_are_compared_as_a_set(scene_dir):
    data = _write(scene_dir, "order.txt", "0")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Oriented(ThreeDScene):
    def construct(self):
        dot, square = Dot(), Square().shift(RIGHT)
        self.add(dot, square)
        pair = (dot, square)
        if int((HERE / "order.txt").read_text()):
            pair = pair[::-1]
        self.renderer.camera.add_fixed_orientation_mobjects(*pair)
        self.wait(0.25)
""",
    )
    assert _diverged_after_editing(scene, "Oriented", data, "1") is None


def test_what_the_script_does_after_its_last_play_is_compared(scene_dir):
    data = _write(scene_dir, "tail.txt", "0.5")
    after_last = _write(
        scene_dir,
        "after.py",
        HEADER
        + """
class AfterLast(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.wait(0.25)
        dot.shift(float((HERE / "tail.txt").read_text()) * UP)
""",
    )
    no_plays = _write(
        scene_dir,
        "still.py",
        HEADER
        + """
class NoPlays(Scene):
    def construct(self):
        self.add(Dot().shift(float((HERE / "tail.txt").read_text()) * UP))
""",
    )
    for scene, name in ((after_last, "AfterLast"), (no_plays, "NoPlays")):
        divergence = _diverged_after_editing(scene, name, data, "1.5")
        assert divergence is not None, name
        assert divergence.play is None, name
        assert divergence.source_line is None, name
        assert divergence.changed_inputs == ("tail.txt",), name
        assert "after its last play" in divergence.reason, name


SUMMARIZE = """\
import json
import sys

from arachne.session import Session

session = Session.open(
    sys.argv[1], "Reads", config={**json.loads(sys.argv[2]), "media_dir": sys.argv[3]}
)
with open(sys.argv[4], "w") as out:
    out.write(session.summary().to_json())
"""


def test_a_summary_from_another_process_compares_with_a_session(scene_dir, tmp_path):
    data = _write(scene_dir, "data.txt", "0.5")
    scene = _write(scene_dir, "scene.py", READS)
    out = tmp_path / "summary.json"
    root = Path(arachne.__file__).resolve().parent.parent
    subprocess.run(
        [
            sys.executable,
            "-c",
            SUMMARIZE,
            str(scene),
            json.dumps(SMALL),
            str(tmp_path / "media"),
            str(out),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
    )
    text = out.read_text()
    theirs = ExecutionSummary.from_json(text)
    assert theirs.to_json() == text

    same = _open(scene, "Reads")
    assert summarize(same.record).to_json() == text
    assert same.compare(theirs) is None

    data.write_text("1.5")
    edited = _open(scene, "Reads")
    divergence = edited.compare(theirs)
    assert divergence is not None
    assert divergence.play == 1
    assert divergence.changed_inputs == ("data.txt",)
    assert edited.divergences == [divergence]


SVG = Path(__file__).parents[1] / "test_graphical_units" / "img_svg_resources" / "arcs01.svg"


def test_an_svg_drawn_again_in_one_process_is_the_execution_it_was(scene_dir):
    """The SVG cache carries a parsed image from one execution to the next,
    so a second execution in one process must build what the first built.
    """
    shutil.copy(SVG, scene_dir / "image.svg")
    scene = _write(
        scene_dir,
        "scene.py",
        HEADER
        + """
class Drawn(Scene):
    def construct(self):
        image = SVGMobject(HERE / "image.svg")
        self.add(image)
        self.play(image.animate.shift(RIGHT), run_time=0.5)
""",
    )
    first = _open(scene, "Drawn").summary()
    again = _open(scene, "Drawn").summary()
    assert compare(first, again) is None
