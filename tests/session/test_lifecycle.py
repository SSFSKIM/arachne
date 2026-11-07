"""A session's life in process: it notices an edit to its source or to a
file its scene read, executes again as a new generation when reloaded,
raises what its scene raised as it was, its explanation on it, lints what its scene armed, and
lets go of everything when closed.
"""

from __future__ import annotations

import builtins
import gc
import hashlib
import os
import sys
import textwrap
import traceback
import weakref
from fractions import Fraction
from pathlib import Path

import pytest

from arachne import api_json
from arachne.errors import SceneError
from arachne.lint import armed, lint
from arachne.lint.departures import departures
from arachne.lint.serialise import violation
from arachne.lint.watch import hold_watch
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.hooks import SceneHookRefused
from arachne.timeline.routes import RecordRefused
from tests.lint import armed_scenes
from tests.timeline import camera_scenes, hook_scenes

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}
ARMED = Path(armed_scenes.__file__)
HOOK_SCENES = Path(hook_scenes.__file__)
CAMERA_SCENES = Path(camera_scenes.__file__)


@pytest.fixture
def scene_dir(tmp_path):
    """A directory for a scene, its helpers and its data, whose modules are
    forgotten afterwards, each test starting and ending with no demotion.
    """
    directory = tmp_path / "scenes"
    directory.mkdir()
    admission.DEMOTED.clear()
    yield directory
    admission.DEMOTED.clear()
    root = os.path.realpath(tmp_path) + os.sep
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and os.path.realpath(file).startswith(root):
            del sys.modules[name]


@pytest.fixture
def opened():
    """Sessions a test opens, closed after it."""
    sessions: list[Session] = []
    yield sessions
    for session in sessions:
        session.close()


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


def _open(opened: list[Session], path: Path, scene: str, **settings) -> Session:
    media = path.parent.parent / "media"
    session = Session.open(
        path, scene, config={**SMALL, "media_dir": str(media), **settings}
    )
    opened.append(session)
    return session


def _line(path: Path, tag: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def _x_at_end(session: Session, name: str = "dot") -> float:
    node = session.state_at(session.record.duration).named(name)
    return (node.box[0] + node.box[2]) / 2


def _same_size_same_time(path: Path, now: str) -> None:
    """Write ``now`` over ``path``, of the same size, and give the file back
    its modification time: an edit made within the same second.
    """
    before = path.stat()
    assert len(now.encode()) == before.st_size
    path.write_text(now)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns


# Reloading --------------------------------------------------------------------

TWO_PLAYS = """\
from arachne import *


class TwoPlays(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        self.play(dot.animate.shift(LEFT), run_time=1)  # @last
"""


def test_an_edit_is_seen_and_a_reload_executes_it_as_a_new_generation(
    scene_dir, opened
):
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    session = _open(opened, scene, "TwoPlays")
    record = session.record
    assert session.generation == 1
    assert record.duration == 2
    assert not session.changed()
    scene.write_text(TWO_PLAYS.replace("run_time=1)  # @last", "run_time=2)  # @last"))
    # Seen, and nothing else: the record stands, and is not called stale
    # until something asks the source again.
    assert session.changed()
    assert not session.stale
    assert session.record is record
    assert session.generation == 1
    assert session.reload()
    assert session.record is not record
    assert session.record.duration == 3
    assert session.generation == 2
    assert session.divergences == []
    assert not session.stale
    assert session.probe is not None
    assert not session.changed()
    assert api_json.inspect(session)["generation"] == 2


def test_a_reload_with_nothing_changed_executes_only_when_forced(scene_dir, opened):
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    session = _open(opened, scene, "TwoPlays")
    record = session.record
    assert not session.reload()
    assert session.record is record
    assert session.generation == 1
    assert session.reload(force=True)
    assert session.record is not record
    assert session.generation == 2
    assert session.record.duration == 2


SHIFTED = """\
from arachne import *
from shift_helper import OFFSET


class Shifted(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(OFFSET * RIGHT), run_time=0.5)
"""


def test_a_helper_edited_to_its_size_within_the_same_second_is_seen(scene_dir, opened):
    """No file's size or modification time says it changed: its content
    does, and the next generation imports the edited helper.
    """
    helper = _write(scene_dir, "shift_helper.py", "OFFSET = 0.5\n")
    scene = _write(scene_dir, "scene.py", SHIFTED)
    session = _open(opened, scene, "Shifted")
    assert _x_at_end(session) == pytest.approx(0.5)
    _same_size_same_time(helper, "OFFSET = 1.5\n")
    assert session.changed()
    assert session.reload()
    assert session.generation == 2
    assert _x_at_end(session) == pytest.approx(1.5)


READS_DATA = """\
from pathlib import Path

from arachne import *

HERE = Path(__file__).parent

#: Read by the first play's updater, and rebound after it.
SHIFT = 0.0


class ReadsData(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        start = float((HERE / "data.txt").read_text())
        tracker = ValueTracker(start)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)  # @play
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_data_file_edited_then_reloaded_is_a_new_generation_not_a_divergence(
    scene_dir, opened
):
    """An edit to a file the scene read, followed by a reload, is a new
    execution of what the files hold now. Within a generation, a forward
    run that meets a file changed under it still diverges.
    """
    scene = _write(scene_dir, "scene.py", READS_DATA)
    data = _write(scene_dir, "data.txt", "0.5")
    session = _open(opened, scene, "ReadsData")
    assert not session.changed()
    _same_size_same_time(data, "1.5")
    assert session.changed()
    assert session.reload()
    assert session.generation == 2
    assert session.divergences == []
    assert session.forward_runs == []
    assert not session.stale
    assert session.record.identity.input_digests == (("data.txt", _sha256(data)),)
    assert not session.changed()
    # Within the generation: the record's restore can't answer 1/7 (the
    # global the updater reads was rebound), so a forward run does, and it
    # read another data file than the record's execution did.
    data.write_text("2.5")
    (answer,) = session.answers([Fraction(1, 7)])
    assert answer.state.answer == "forward"
    (run,) = session.forward_runs
    assert run.divergence is not None
    assert run.divergence.code == "replay.diverged"
    assert run.divergence.changed_inputs == ("data.txt",)
    assert run.divergence.source_line.line == _line(scene, "play")
    assert session.divergences == [run.divergence]


def test_a_data_file_removed_is_a_change(scene_dir, opened):
    scene = _write(scene_dir, "scene.py", READS_DATA)
    data = _write(scene_dir, "data.txt", "0.5")
    session = _open(opened, scene, "ReadsData")
    data.unlink()
    assert session.changed()


def test_a_helper_newly_imported_is_a_change(scene_dir, opened):
    """The source is the closure of the scene's imports as the files are
    now: a helper the edited scene imports for the first time is part of it.
    """
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    _write(scene_dir, "shift_helper.py", "OFFSET = 0.5\n")
    session = _open(opened, scene, "TwoPlays")
    scene.write_text("import shift_helper\n" + TWO_PLAYS)
    assert session.changed()


# Errors -------------------------------------------------------------------

RAISES = """\
from arachne import *


class RaisesAtOne(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        dot.shiftt(RIGHT)  # @raise
"""


def _innermost_line(error: BaseException, path: Path) -> int:
    """The innermost line of ``path`` the error's traceback went through."""
    found = [
        frame.lineno
        for frame in traceback.extract_tb(error.__traceback__)
        if os.path.realpath(frame.filename) == os.path.realpath(path)
    ]
    return found[-1]


def test_a_scene_s_error_is_raised_as_it_was_and_carries_its_explanation(scene_dir):
    """The error is the scene's own, its type, message and traceback as
    executing it gave them; its explanation rides on it, at the scene
    clock's time, and its hint is a note a printed traceback shows.
    """
    scene = _write(scene_dir, "scene.py", RAISES)
    with pytest.raises(AttributeError) as info:
        Session.open(scene, "RaisesAtOne", config=SMALL)
    error = info.value
    assert type(error) is AttributeError
    assert str(error) == "Dot object has no attribute 'shiftt'"
    assert _innermost_line(error, scene) == _line(scene, "raise")
    explained = error.scene_error
    assert isinstance(explained, SceneError)
    assert explained.cls == "AttributeError"
    assert explained.message == str(error)
    assert explained.t == 1
    assert explained.line == _line(scene, "raise")
    assert explained.path == str(scene)
    [hint] = [hint for hint in explained.hints if "shift" in hint]
    printed = "".join(traceback.format_exception(error))
    assert hint in printed
    assert "t = 1 s" in printed


def test_a_scene_that_raises_before_it_starts_has_no_time(scene_dir):
    scene = _write(scene_dir, "scene.py", "from arachne import Nothing  # @import\n")
    with pytest.raises(ImportError) as info:
        Session.open(scene, "S", config=SMALL)
    assert info.value.scene_error.cls == "ImportError"
    assert info.value.scene_error.t is None
    assert info.value.scene_error.line == _line(scene, "import")


def test_an_engine_refusal_raises_as_itself_with_its_reason():
    with pytest.raises(SceneHookRefused) as info:
        Session.open(HOOK_SCENES, "UpdateMobjectsTwice", config=SMALL)
    assert info.value.reason == "hook.update_mobjects.twice"
    assert info.value.scene_error.cls == "SceneHookRefused"


def test_a_record_s_refusal_is_raised_as_itself(opened):
    session = Session.open(CAMERA_SCENES, "Mapped", config=SMALL)
    opened.append(session)
    with pytest.raises(RecordRefused, match="mapping_func"):
        session.frame(Fraction(1, 2))


def test_a_reload_that_raises_keeps_the_generation_and_is_stale(scene_dir, opened):
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    session = _open(opened, scene, "TwoPlays")
    record = session.record
    scene.write_text(
        TWO_PLAYS.replace(
            "self.play(dot.animate.shift(LEFT), run_time=1)  # @last",
            'raise RuntimeError("edited to raise")  # @last',
        )
    )
    with pytest.raises(RuntimeError) as info:
        session.reload()
    assert str(info.value) == "edited to raise"
    assert info.value.scene_error.t == 1
    assert info.value.scene_error.line == _line(scene, "last")
    assert session.record is record
    assert session.generation == 1
    assert session.stale


def test_a_reload_that_exits_still_leaves_the_session_stale(scene_dir, opened):
    """Whatever ends the new execution, the source changed: the record held
    is of a program the file no longer is.
    """
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    session = _open(opened, scene, "TwoPlays")
    record = session.record
    scene.write_text(
        TWO_PLAYS.replace(
            "self.play(dot.animate.shift(LEFT), run_time=1)  # @last",
            "raise SystemExit(3)  # @last",
        )
    )
    with pytest.raises(SystemExit):
        session.reload()
    assert session.record is record
    assert session.generation == 1
    assert session.stale


SECOND_RUN_RAISES = """\
import builtins

from arachne import *


class SecondRunRaises(Scene):
    def construct(self):
        runs = builtins.__dict__.setdefault("_second_run_raises", [])
        runs.append(1)
        self.add(Dot())
        self.wait(1)
        if len(runs) > 1:
            raise RuntimeError("raised the second time")
"""


def test_a_forced_reload_of_an_unchanged_source_that_raises_is_not_stale(
    scene_dir, opened
):
    """A scene that raises on its second execution only: its source is
    still what the record was made from.
    """
    scene = _write(scene_dir, "scene.py", SECOND_RUN_RAISES)
    try:
        session = _open(opened, scene, "SecondRunRaises")
        with pytest.raises(RuntimeError, match="the second time"):
            session.reload(force=True)
        assert not session.stale
        assert session.generation == 1
    finally:
        builtins.__dict__.pop("_second_run_raises", None)


def test_a_reload_executes_the_file_opened_whatever_the_working_directory(
    scene_dir, opened, monkeypatch
):
    """A file opened by a relative path is that file: a reload after the
    working directory moved executes it, and names it as it was given.
    """
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    elsewhere = scene_dir.parent / "elsewhere"
    elsewhere.mkdir()
    # A file of the same name where the working directory moves to, which
    # the reload must not execute.
    _write(elsewhere, "scene.py", TWO_PLAYS.replace("TwoPlays", "Other"))
    monkeypatch.chdir(scene_dir)
    session = _open(opened, Path("scene.py"), "TwoPlays")
    monkeypatch.chdir(elsewhere)
    scene.write_text(TWO_PLAYS.replace("run_time=1)  # @last", "run_time=2)  # @last"))
    assert session.changed()
    assert session.reload()
    assert session.record.duration == 3
    scene.write_text(
        TWO_PLAYS.replace(
            "self.play(dot.animate.shift(LEFT), run_time=1)  # @last",
            'raise RuntimeError("edited to raise")  # @last',
        )
    )
    with pytest.raises(RuntimeError) as info:
        session.reload()
    assert info.value.scene_error.path == "scene.py"


# What the scene armed -----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["ExemptionGrowsBetweenRemovals", "RearmedWithoutTextGap", "CaptionLeavesTheGroup"],
)
def test_linting_what_the_scene_armed_is_the_rules_composed_from_its_first_arming(
    opened, name
):
    session = Session.open(ARMED, name, config={**SMALL, "frame_rate": 15})
    opened.append(session)
    assert armed(session.record) == (hold_watch.where(), departures.where())
    by_hand = lint(session, rules=(hold_watch.where(), departures.where()))
    asked = lint(session, rules="armed")
    assert [violation(f) for f in asked.findings] == [
        violation(f) for f in by_hand.findings
    ]
    assert asked.findings
    through_session = session.lint(rules="armed")
    assert [violation(f) for f in through_session.findings] == [
        violation(f) for f in by_hand.findings
    ]


ARMS_LATE = """\
from arachne import *
from arachne.lint.bridge import arm


class ArmsLate(Scene):
    def construct(self):
        self.add(Dot())
        self.wait(1)
        arm(text_gap=0.5)
        self.wait(1)
        arm()
        self.wait(1)


class NeverArms(Scene):
    def construct(self):
        self.add(Dot())
        self.wait(1)
"""


def test_the_armed_rules_start_at_the_first_arming_and_sample_as_asked(
    scene_dir, opened
):
    scene = _write(scene_dir, "scene.py", ARMS_LATE)
    session = _open(opened, scene, "ArmsLate")
    assert armed(session.record) == (
        hold_watch.where(since=1),
        departures.where(since=1),
    )
    assert armed(session.record, sample_fps=8) == (
        hold_watch.where(since=1, sample_fps=8),
        departures.where(since=1),
    )
    never = _open(opened, scene, "NeverArms")
    assert armed(never.record) == ()


def test_a_sample_rate_is_the_armed_rules_and_no_others(scene_dir, opened):
    scene = _write(scene_dir, "scene.py", ARMS_LATE)
    session = _open(opened, scene, "NeverArms")
    with pytest.raises(ValueError, match="sample_fps"):
        lint(session, rules=(hold_watch.where(),), sample_fps=8)


TELEPORTS = """\
from arachne import *
from arachne.lint.bridge import arm


class Teleports(Scene):
    arms = False

    def construct(self):
        if self.arms:
            arm()
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT * 2))
        puck.move_to(RIGHT * 3)
        self.wait()


class ArmedTeleports(Teleports):
    arms = True
"""


def test_a_teleport_is_a_jump_only_where_the_scene_armed_the_motion_rule(
    scene_dir, opened
):
    scene = _write(scene_dir, "scene.py", TELEPORTS)

    def jumps(name: str) -> list:
        session = _open(opened, scene, name, frame_rate=15)
        result = session.lint(rules="armed")
        return [f for f in result.findings if f.code == "motion.jump"]

    assert jumps("Teleports") == []
    assert len(jumps("ArmedTeleports")) == 1


# Closing ------------------------------------------------------------------


def test_closing_lets_go_of_the_scene_its_record_and_its_module(scene_dir):
    scene = _write(scene_dir, "scene.py", TWO_PLAYS)
    session = Session.open(scene, "TwoPlays", config=SMALL)
    executed = weakref.ref(session.scene)
    record = weakref.ref(session.record)
    module = session.module.__name__
    assert module in sys.modules
    session.close()
    gc.collect()
    assert module not in sys.modules
    assert executed() is None
    assert record() is None
    # Closing again is nothing.
    session.close()


def test_the_armed_rules_function_is_no_module_s_name():
    """``arachne.lint.armed`` is the function: no submodule of that name
    stands behind it for an import to find.
    """
    import importlib.util

    import arachne.lint

    assert callable(arachne.lint.armed)
    assert importlib.util.find_spec("arachne.lint.armed") is None
