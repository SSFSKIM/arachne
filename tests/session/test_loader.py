"""Every execution a session makes imports the scene's own modules afresh,
from their source, and leaves the process's modules and path as it found
them: a helper edited between two executions is the edited helper in the
second.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import smooth
from arachne.session import Session
from arachne.timeline.fallback import visit_frames

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}


@pytest.fixture
def scene_dir(tmp_path):
    """A directory for a scene and its helpers, whose modules are forgotten
    afterwards whatever the session left imported.
    """
    directory = tmp_path / "scenes"
    directory.mkdir()
    yield directory
    root = os.path.realpath(tmp_path) + os.sep
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
    return Session.open(path, scene, config={**SMALL, "media_dir": str(media)})


def _x_at_end(session: Session, name: str = "dot") -> float:
    node = session.state_at(session.record.duration).named(name)
    return (node.box[0] + node.box[2]) / 2


SHIFTED = """\
from arachne import *
from shift_helper import OFFSET


class Shifted(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(OFFSET * RIGHT), run_time=0.5)
"""


def test_a_helper_edited_between_two_executions_is_imported_again(scene_dir):
    helper = _write(scene_dir, "shift_helper.py", "OFFSET = 0.5\n")
    scene = _write(scene_dir, "scene.py", SHIFTED)
    first = _open(scene, "Shifted")
    assert _x_at_end(first) == pytest.approx(0.5)
    first.close()
    helper.write_text("OFFSET = 1.5\n")
    second = _open(scene, "Shifted")
    assert _x_at_end(second) == pytest.approx(1.5)
    second.close()


HELPED = """\
from arachne import *
from start_helper import START

#: Read by the first play's updater, and rebound after it.
SHIFT = 0.0


class Helped(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        tracker = ValueTracker(START)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)
"""


def test_a_forward_run_after_a_helper_edit_executes_the_edited_helper(scene_dir):
    """The record can't answer an instant the global rebound after it was
    sampled, so a forward run executes the scene again: after the helper
    was edited, that execution is of the source as it is now, the edited
    helper included, and says so.
    """
    helper = _write(scene_dir, "start_helper.py", "START = 0.5\n")
    scene = _write(scene_dir, "scene.py", HELPED)
    session = _open(scene, "Helped")
    try:
        helper.write_text("START = 1.5\n")
        t = Fraction(1, 7)
        (answer,) = session.answers([t])
        assert answer.state.answer == "forward"
        assert answer.fallback.kind == "source-changed"
        assert "start_helper.py" in answer.fallback.detail
        dot = answer.state.named("dot")
        expected = 1.5 + (2 - 1.5) * float(smooth(float(t)))
        assert (dot.box[0] + dot.box[2]) / 2 == pytest.approx(expected)
    finally:
        session.close()


PACKAGED = """\
from arachne import *
from shift_pkg import SCALE
from shift_pkg.steps import STEP


class Packaged(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(SCALE * STEP * RIGHT), run_time=0.5)
"""


def test_a_package_and_its_submodule_are_imported_again(scene_dir):
    package = _write(scene_dir, "shift_pkg/__init__.py", "SCALE = 1.0\n")
    steps = _write(
        scene_dir, "shift_pkg/steps.py", "from . import SCALE\n\nSTEP = 0.5\n"
    )
    scene = _write(scene_dir, "scene.py", PACKAGED)
    first = _open(scene, "Packaged")
    assert _x_at_end(first) == pytest.approx(0.5)
    first.close()
    package.write_text("SCALE = 2.0\n")
    steps.write_text("from . import SCALE\n\nSTEP = 0.75\n")
    second = _open(scene, "Packaged")
    assert _x_at_end(second) == pytest.approx(1.5)
    second.close()


CACHED = """\
from arachne import *
from cached_helper import OFFSET

SCENE_OFFSET = 0.25


class Cached(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift((OFFSET + SCENE_OFFSET) * RIGHT), run_time=0.5)
"""


def _stale_bytecode(path: Path, now: str) -> Path:
    """Compile ``path`` as it is, then write ``now`` in its place with the
    same size and modification time: the bytecode left beside it matches
    the file by both, and holds another program.
    """
    before = path.stat()
    cached = Path(importlib.util.cache_from_source(os.fspath(path)))
    py_compile.compile(
        os.fspath(path),
        cfile=os.fspath(cached),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    assert len(now.encode()) == before.st_size
    path.write_text(now)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    return cached


def test_bytecode_matching_a_file_by_size_and_time_is_not_trusted(scene_dir):
    """The scene file and its helper each have bytecode beside them that
    matches them by size and modification time and holds another program:
    each is compiled from its source, and no bytecode is written.
    """
    helper = _write(scene_dir, "cached_helper.py", "OFFSET = 0.5\n")
    scene = _write(scene_dir, "scene.py", CACHED)
    cached = {
        _stale_bytecode(helper, "OFFSET = 1.5\n"),
        _stale_bytecode(scene, CACHED.replace("0.25", "0.75")),
    }
    bytecode = {path: path.read_bytes() for path in cached}
    session = _open(scene, "Cached")
    assert _x_at_end(session) == pytest.approx(1.5 + 0.75)
    session.close()
    # Nothing written beside the sources, and nothing rewritten.
    assert set((scene_dir / "__pycache__").iterdir()) == cached
    assert {path: path.read_bytes() for path in cached} == bytecode


RAISES = """\
import sys
from pathlib import Path

from arachne import *
import raise_helper

HERE = Path(__file__).parent


class Passes(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(raise_helper.OFFSET * RIGHT), run_time=0.5)


class Raises(Scene):
    def construct(self):
        import raise_late_helper

        sys.path.append(str(HERE / "lib"))
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(raise_late_helper.OFFSET * RIGHT), run_time=0.5)
        raise ValueError("raised on purpose")
"""


def test_an_execution_that_raised_leaves_the_modules_and_path_as_they_were(
    scene_dir,
):
    _write(scene_dir, "raise_helper.py", "OFFSET = 0.5\n")
    _write(scene_dir, "raise_late_helper.py", "OFFSET = 0.5\n")
    scene = _write(scene_dir, "scene.py", RAISES)
    # Whatever the engine imports the first time a scene runs, imported.
    _open(scene, "Passes").close()
    modules, path, meta_path = dict(sys.modules), list(sys.path), list(sys.meta_path)
    with pytest.raises(ValueError, match="raised on purpose"):
        _open(scene, "Raises")
    assert sys.path == path
    assert sys.meta_path == meta_path
    assert set(sys.modules) == set(modules)
    assert all(sys.modules[name] is module for name, module in modules.items())


SHADOWING = """\
from arachne import *
import numpy

SCENE_NUMPY = numpy


class Shadowing(Scene):
    def construct(self):
        self.wait(0.25)
"""


def test_a_scene_module_named_like_an_imported_one_does_not_replace_it(scene_dir):
    """A module beside the scene named as one the process already imported
    (``numpy``) is not the scene's to load: the scene gets the process's,
    and the process keeps it.
    """
    _write(scene_dir, "numpy.py", "raise ImportError('the scene directory\\'s')\n")
    scene = _write(scene_dir, "scene.py", SHADOWING)
    session = _open(scene, "Shadowing")
    try:
        assert session.module.SCENE_NUMPY is np
        assert sys.modules["numpy"] is np
    finally:
        session.close()


SHARED = """\
from arachne import *
import shared_helper


class Shared(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(shared_helper.OFFSET * RIGHT), run_time=0.5)
"""


def test_two_sessions_of_one_file_share_no_module(scene_dir):
    _write(scene_dir, "shared_helper.py", "OFFSET = 0.5\n")
    scene = _write(scene_dir, "scene.py", SHARED)
    first, second = _open(scene, "Shared"), _open(scene, "Shared")
    assert first.module is not second.module
    assert first.module.shared_helper is not second.module.shared_helper
    assert "shared_helper" not in sys.modules
    names = (first.module.__name__, second.module.__name__)
    assert all(name in sys.modules for name in names)
    first.close()
    second.close()
    assert not any(name in sys.modules for name in names)


VISITED = """\
from arachne import *


class Visited(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)

    def helper_value(self):
        import visit_helper

        return visit_helper.VALUE
"""


def _visit_past_the_end(session: Session) -> list[int]:
    """Execute the session's scene once more and visit it past its end:
    what its ``helper_value`` gave at each visit after the script ended.
    """
    seen: list[int] = []

    def visit(t, record, play, capture):
        if play is None:
            seen.append(record._scene.helper_value())

    visit_frames(session, [session.record.duration + 1], visit, to_end=True)
    return seen


def test_a_visit_past_the_end_imports_the_scene_s_helpers(scene_dir):
    """A forward run's visits after the scene's end still belong to that
    execution: the scene's code they call imports the scene's helpers.
    """
    _write(scene_dir, "visit_helper.py", "VALUE = 1\n")
    scene = _write(scene_dir, "scene.py", VISITED)
    session = _open(scene, "Visited")
    try:
        assert _visit_past_the_end(session) == [1]
        assert "visit_helper" not in sys.modules
    finally:
        session.close()


def test_each_execution_s_visits_past_the_end_see_its_own_helper(scene_dir):
    helper = _write(scene_dir, "visit_helper.py", "VALUE = 1\n")
    scene = _write(scene_dir, "scene.py", VISITED)
    session = _open(scene, "Visited")
    try:
        assert _visit_past_the_end(session) == [1]
        helper.write_text("VALUE = 2\n")
        assert _visit_past_the_end(session) == [2]
    finally:
        session.close()
