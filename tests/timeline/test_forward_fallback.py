"""The forward fallback: an instant the record can't answer (a state play
that can't be restored there, or a record that refuses drawing) is answered
by executing the scene again in process, once for a whole batch of instants,
and the answer says so. The new execution is compared with the record's,
and leaves the session as it found it.
"""

from __future__ import annotations

import os
import random
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import session as session_module
from arachne.session import Session
from arachne.timeline.origins import ORIGINS

from . import camera_scenes, restore_scenes
from .conftest import render
from .test_restore import _canonical, _unsampled

CAMERA_SCENES = Path(camera_scenes.__file__)
RESTORE_SCENES = Path(restore_scenes.__file__)


@pytest.fixture
def executions(monkeypatch):
    """How many times a session imports a scene file to execute it, from
    here on. (Not by patching the engine's classes: admission proves no
    play while one of their methods is replaced.)
    """
    counted = [0]
    load = session_module._import

    def counting(path):
        counted[0] += 1
        return load(path)

    monkeypatch.setattr(session_module, "_import", counting)
    return counted


def _pixels(image) -> np.ndarray:
    return np.asarray(image)


# A state play that can't be restored


def test_an_unrestorable_instant_is_answered_by_one_forward_run(session_of, executions):
    """The global the first play's updater reads is rebound after it, so a
    restore there is refused; a forward run evaluates the play at each
    instant asked, with the global as it was, as an execution asked for
    those instants evaluates it. The second play restores as before.
    """
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    first, second = session.record.plays
    unrestorable = _unsampled(first)
    restorable = _unsampled(second)[0]
    oracle = session_of(
        "ReboundGlobal", RESTORE_SCENES, frame_rate=15, instants=unrestorable
    )
    executions[0] = 0

    answers = session.answers([*unrestorable, restorable])

    assert executions[0] == 1
    assert [answer.t for answer in answers] == [*unrestorable, restorable]
    *forward, restored = answers
    assert restored.state.answer == "restored"
    assert restored.fallback is None
    for answer in forward:
        expected = oracle.state_at(answer.t)
        assert expected.answer == "recorded"
        assert answer.state.answer == "forward"
        assert answer.state.route == "state"
        assert answer.state.as_of == answer.t
        assert _canonical(answer.state) == _canonical(expected), answer.t
        assert np.array_equal(_pixels(answer.frame), _pixels(oracle.frame(answer.t))), (
            answer.t
        )
        assert answer.fallback.kind == "unrestorable"
        assert answer.fallback.reason == "restore.global_rebound"
    # One execution, stopped after the play holding the last instant asked,
    # and the same as the record's wherever both kept an instant.
    (run,) = session.forward_runs
    assert run.instants == tuple(unrestorable)
    assert run.plays == 1
    assert run.compared_until == unrestorable[-1]
    assert run.divergence is None
    assert session.divergences == []


HEADER = """\
from pathlib import Path

from arachne import *

HERE = Path(__file__).parent

#: Read by the first play's updater, and rebound after it.
SHIFT = 0.0
"""

READS_AND_REBINDS = (
    HEADER
    + """
class ReadsAndRebinds(Scene):
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
        # Read once the second play has run: a forward run asked about the
        # first stops before it.
        (HERE / "late.txt").read_text()
"""
)


@pytest.fixture
def scene_dir(tmp_path):
    """A directory for a scene and the data it reads, whose modules are
    forgotten afterwards.
    """
    directory = tmp_path / "scenes"
    directory.mkdir()
    yield directory
    root = os.path.realpath(tmp_path) + os.sep
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and os.path.realpath(file).startswith(root):
            del sys.modules[name]


def _open_written(directory: Path) -> tuple[Session, Path]:
    scene = directory / "scene.py"
    scene.write_text(textwrap.dedent(READS_AND_REBINDS))
    (directory / "data.txt").write_text("0.5")
    (directory / "late.txt").write_text("a")
    session = Session.open(
        scene,
        "ReadsAndRebinds",
        config={
            "pixel_width": 96,
            "pixel_height": 54,
            "frame_rate": 4,
            "media_dir": str(directory.parent / "media"),
        },
    )
    return session, scene


def _line(path: Path, tag: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


@pytest.mark.parametrize("edited", [False, True])
def test_a_forward_run_is_compared_with_the_record(scene_dir, edited):
    """Deterministic, the forward run agrees with the record wherever both
    kept an instant. With the data files edited between the two, it is
    another execution: the divergence is at the first play, at its line,
    and names the file both executions read before it whose content
    differed, not one the record's execution read only after the plays
    compared.
    """
    session, scene = _open_written(scene_dir)
    try:
        if edited:
            (scene_dir / "data.txt").write_text("1.5")
            (scene_dir / "late.txt").write_text("b")
        (answer,) = session.answers([Fraction(1, 7)])
        assert answer.state.answer == "forward"
        assert answer.fallback.reason == "restore.global_rebound"
        (run,) = session.forward_runs
        if not edited:
            assert run.divergence is None
            assert session.divergences == []
            return
        divergence = run.divergence
        assert divergence is not None
        assert divergence.code == "replay.diverged"
        assert divergence.kind == "execution"
        assert divergence.play == 0
        assert divergence.source_line.line == _line(scene, "play")
        assert divergence.changed_inputs == ("data.txt",)
        assert session.divergences == [divergence]
        # The answer is the new execution's: the tracker began at 1.5 there.
        dot = answer.state.named("dot")
        expected = 1.5 + (2 - 1.5) * _smooth(Fraction(1, 7))
        assert (dot.box[0] + dot.box[2]) / 2 == pytest.approx(expected)
    finally:
        session.close()


HELPED = (
    HEADER
    + """
from fallback_helper import START


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
)


def test_a_source_edited_since_the_record_makes_it_stale_not_divergent(scene_dir):
    """A helper the scene imports, edited after the session opened: the
    forward run executes another program than the record's. It still
    answers, saying so; the record is stale, and nothing is compared, so no
    divergence is reported (an edit is no replay that diverged).
    """
    scene = scene_dir / "scene.py"
    scene.write_text(textwrap.dedent(HELPED))
    helper = scene_dir / "fallback_helper.py"
    helper.write_text("START = 0.5\n")
    session = Session.open(
        scene,
        "Helped",
        config={
            "pixel_width": 96,
            "pixel_height": 54,
            "frame_rate": 4,
            "media_dir": str(scene_dir.parent / "media"),
        },
    )
    try:
        assert not session.stale
        helper.write_text("START = 1.5\n")
        (answer,) = session.answers([Fraction(1, 7)])
        assert answer.state.answer == "forward"
        assert answer.fallback.kind == "source-changed"
        assert answer.fallback.reason == "source.changed"
        assert "fallback_helper.py" in answer.fallback.detail
        assert session.stale
        (run,) = session.forward_runs
        assert run.source_changed
        assert run.divergence is None
        assert session.divergences == []
    finally:
        session.close()


READS_IF_THERE = (
    HEADER
    + """
class ReadsIfThere(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        data = HERE / "data.txt"
        start = float(data.read_text()) if data.exists() else 1.5
        tracker = ValueTracker(start)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)
"""
)


def _open_in(
    directory: Path, source: str, name: str, *, frame_rate: int = 8
) -> Session:
    scene = directory / "scene.py"
    scene.write_text(textwrap.dedent(source))
    return Session.open(
        scene,
        name,
        config={
            "pixel_width": 96,
            "pixel_height": 54,
            "frame_rate": frame_rate,
            "media_dir": str(directory.parent / "media"),
        },
    )


def test_an_input_the_forward_run_never_read_is_named_when_it_changed(scene_dir):
    """The data file is deleted after the session opened: the forward run,
    stopped after the first play, reads no file at all, yet the file the
    record's execution read is what changed.
    """
    data = scene_dir / "data.txt"
    data.write_text("0.5")
    session = _open_in(scene_dir, READS_IF_THERE, "ReadsIfThere")
    try:
        data.unlink()
        session.answers([Fraction(1, 7)])
        (run,) = session.forward_runs
        assert run.divergence is not None
        assert run.divergence.changed_inputs == ("data.txt",)
    finally:
        session.close()


READS_THEN_WRITES_BACK = (
    HEADER
    + """
class ReadsThenWritesBack(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        data = HERE / "data.txt"
        start = float(data.read_text())
        data.write_text("0.5")
        tracker = ValueTracker(start)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)
"""
)


def test_an_input_is_named_by_what_each_execution_read(scene_dir):
    """The scene reads its data file and writes the record's value back:
    after the forward run, the file holds what the record's execution read,
    but the forward run read something else, and that is the change.
    """
    data = scene_dir / "data.txt"
    data.write_text("0.5")
    session = _open_in(scene_dir, READS_THEN_WRITES_BACK, "ReadsThenWritesBack")
    try:
        data.write_text("1.5")
        session.answers([Fraction(1, 7)])
        (run,) = session.forward_runs
        assert data.read_text() == "0.5"
        assert run.divergence is not None
        assert run.divergence.changed_inputs == ("data.txt",)
    finally:
        session.close()


READS_ALONG = (
    HEADER
    + """
class ReadsAlong(Scene):
    def construct(self):
        (HERE / "before.txt").read_text()
        dot = Dot().add_updater(lambda m: (HERE / "during.txt").read_text())
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        (HERE / "between.txt").read_text()
        self.play(dot.animate.shift(LEFT), run_time=1)
        (HERE / "after.txt").read_text()
"""
)


def test_an_input_is_kept_with_the_play_it_was_first_read_for(scene_dir):
    """Each file the scene reads is kept with the first play its content
    could change: the one running as it is read, or else the next to begin
    (one past the last, after the script's last play).
    """
    for name in ("before", "during", "between", "after"):
        (scene_dir / f"{name}.txt").write_text(name)
    session = _open_in(scene_dir, READS_ALONG, "ReadsAlong")
    try:
        assert {read.path: read.play for read in session._source.reads} == {
            "before.txt": 0,
            "during.txt": 0,
            "between.txt": 1,
            "after.txt": 2,
        }
    finally:
        session.close()


@pytest.mark.parametrize(
    ("mine", "theirs", "bounds", "changed"),
    [
        # Both read it before the bound: named when they read different content.
        ([("a", "x", 0)], [("a", "x", 0)], (1, 1), ()),
        ([("a", "x", 0)], [("a", "y", 0)], (1, 1), ("a",)),
        # Only one read it before the bound (a file gone, a conditional read).
        ([("a", "x", 0)], [], (1, 1), ("a",)),
        ([], [("a", "x", 0)], (1, 1), ("a",)),
        ([("a", "x", 0)], [("a", "x", 1)], (1, 1), ("a",)),
        # First read at or after the bound in both: never named.
        ([("a", "x", 1)], [("a", "y", 1)], (1, 1), ()),
        ([("a", "x", 2)], [], (1, 1), ()),
        # Each execution's reads against its own bound: a read for a play
        # only the other compared is not one.
        ([], [("a", "x", 1)], (2, 1), ()),
        ([("a", "x", 1)], [], (2, 1), ("a",)),
        ([("a", "x", 1)], [("a", "x", 1)], (2, 1), ("a",)),
        # No bound (both compared to their end): every read counts.
        ([("a", "x", 5)], [("a", "y", 5)], (None, None), ("a",)),
        ([("b", "x", 0), ("a", "x", 0)], [("a", "y", 0)], (None, None), ("a", "b")),
    ],
)
def test_changed_inputs_are_those_of_the_compared_plays(mine, theirs, bounds, changed):
    from arachne.timeline.fallback import _changed_inputs
    from arachne.timeline.inputs import Read

    assert (
        _changed_inputs(
            tuple(Read(*read) for read in mine),
            tuple(Read(*read) for read in theirs),
            bounds,
        )
        == changed
    )


LONGER_FIRST_PLAY = (
    HEADER
    + """
class LongerFirstPlay(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        span = float((HERE / "span.txt").read_text())
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=span)
        SHIFT = 1.0
        if span > 0.75:
            (HERE / "late.txt").read_text()
        self.play(tracker.animate.set_value(0), run_time=1)
        # Rebound after the second play, which the record then can't restore.
        SHIFT = 2.0
"""
)


def test_a_read_for_a_play_only_the_other_execution_compared_is_not_named(
    scene_dir,
):
    """The record's first play is shorter, so both its plays began by the
    instant asked; the forward run's first play is longer, and the file it
    reads after it is for its second play, which it never compared: only the
    file that made the plays differ is named.
    """
    span = scene_dir / "span.txt"
    span.write_text("0.5")
    (scene_dir / "late.txt").write_text("late")
    session = _open_in(scene_dir, LONGER_FIRST_PLAY, "LongerFirstPlay")
    try:
        span.write_text("1.0")
        session.answers([Fraction(3, 5)], frames=False)
        (run,) = session.forward_runs
        assert run.plays == 1
        assert run.divergence is not None
        assert run.divergence.changed_inputs == ("span.txt",)
    finally:
        session.close()


BEGIN_FAILS_ONCE = (
    HEADER
    + """
class BeginFailsOnce(Scene):
    def begin_animations(self):
        failing = getattr(self, "failing", False)
        self.failing = False
        if failing:
            raise RuntimeError("caught by the scene")
        return super().begin_animations()

    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        self.failing = True
        try:
            self.play(dot.animate.shift(UP), run_time=0.5)
        except RuntimeError:
            pass
        (HERE / "after_failure.txt").read_text()
        self.play(dot.animate.shift(LEFT), run_time=0.5)
"""
)


def test_a_read_after_a_play_that_failed_to_begin_is_for_the_next(scene_dir):
    """A play whose animations failed to begin, the failure caught by the
    scene: a file read after it is for the next play, not for it.
    """
    (scene_dir / "after_failure.txt").write_text("x")
    session = _open_in(scene_dir, BEGIN_FAILS_ONCE, "BeginFailsOnce")
    try:
        plays = len(session.record.plays)
        reads = {read.path: read.play for read in session._source.reads}
        assert reads == {"after_failure.txt": plays - 1}
    finally:
        session.close()


DEFAULTS = (
    HEADER
    + """
class Defaults(Scene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)

    def construct(self):
        Dot.set_default(color=RED)
        if (HERE / "flag.txt").read_text() == "raise":
            raise RuntimeError("forward error")
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)
        self.play(dot.animate.shift(LEFT), run_time=1)
        Dot.set_default()
"""
)


@pytest.mark.parametrize("raises", [False, True])
def test_an_engine_default_the_scene_sets_is_put_back(scene_dir, raises):
    """The scene sets a default of an engine class and resets it at its end,
    which a forward run that stops, or raises, never reaches.
    """
    from arachne import Dot

    flag = scene_dir / "flag.txt"
    flag.write_text("ok")
    original = Dot.__init__
    session = _open_in(scene_dir, DEFAULTS, "Defaults")
    try:
        assert Dot.__init__ is original
        if raises:
            flag.write_text("raise")
            with pytest.raises(RuntimeError, match="forward error"):
                session.answers([Fraction(1, 7)])
        else:
            session.answers([Fraction(1, 7)])
        assert Dot.__init__ is original
    finally:
        Dot.__init__ = original
        session.close()


def _smooth(t: Fraction) -> float:
    from arachne import smooth

    return float(smooth(float(t)))


# A record that refuses drawing


def test_a_mapping_camera_scene_answers_its_grid_instants_with_one_execution(
    session_of, executions
):
    """Its record answers no instant: every one asked is answered by one
    forward run, its frames drawn as the render draws them, while the scene
    runs. An instant between frames is answered with the frame on screen.
    """
    fps = 6
    _, frames = render(camera_scenes.Mapped, fps)
    session = session_of("Mapped", CAMERA_SCENES, frame_rate=fps)
    assert session.record.refusal is not None
    grid = [Fraction(k, fps) for k in range(len(frames))]
    between = Fraction(1, 4)
    executions[0] = 0

    answers = session.answers([*grid, between])

    assert executions[0] == 1
    for k, answer in enumerate(answers[: len(grid)]):
        assert answer.state.answer == "forward"
        assert answer.state.route == "ticked"
        assert answer.fallback.kind == "refused"
        assert answer.fallback.reason == "camera.mapping_func"
        assert np.array_equal(_pixels(answer.frame), frames[k]), k
    on_screen = answers[-1]
    assert on_screen.state.as_of == Fraction(1, 6)
    assert np.array_equal(_pixels(on_screen.frame), frames[1])
    (run,) = session.forward_runs
    assert run.divergence is None
    # Every play is drawn as it runs, from the first: what its drawing
    # writes the record's execution never holds, so only the plays before
    # the first are compared, which are none.
    assert run.compared_plays == 0


def test_a_record_refusing_from_a_later_play_answers_as_the_render_draws(
    session_of,
):
    """The zoomed display leaves its camera at 1/2 s: the render draws the
    first play from its record and the second as it runs, and the forward
    run answers each the same way.
    """
    fps = 8
    _, frames = render(camera_scenes.DisplayLeaves, fps)
    session = session_of("DisplayLeaves", CAMERA_SCENES, frame_rate=fps)
    assert session.record.refusal.startswith("camera.display_left:")
    answers = session.answers([Fraction(k, fps) for k in range(len(frames))])
    assert [answer.state.route for answer in answers[:: fps // 2]] == [
        "trusted",
        "ticked",
    ]
    for k, answer in enumerate(answers):
        assert answer.state.answer == "forward"
        assert answer.fallback.reason == "camera.display_left"
        assert np.array_equal(_pixels(answer.frame), frames[k]), k
    (run,) = session.forward_runs
    # The display's picture, which the second play draws as it runs, is
    # written into the scene: the executions are compared before it.
    assert run.compared_plays == 1
    assert run.divergence is None


def test_a_forward_run_executes_under_the_configuration_the_session_opened_in():
    """The session opened inside a configuration that has since been left:
    its forward run renders as that configuration did, white background
    and all, not as the process's configuration is now.
    """
    from arachne import WHITE, tempconfig

    settings = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}
    with tempconfig({**settings, "background_color": WHITE}):
        session = Session.open(CAMERA_SCENES, "Mapped")
    try:
        _, frames = render(camera_scenes.Mapped, 4, background_color=WHITE)
        (answer,) = session.answers([Fraction(1, 4)])
        assert np.array_equal(_pixels(answer.frame), frames[1])
    finally:
        session.close()


INTRUDING = (
    HEADER
    + """
class Intruding(Scene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)
        # Its drawing inserts curves into the very mobjects it draws.
        self.renderer.camera.allow_object_intrusion = True

    def construct(self):
        static = Circle(radius=0.5)
        dot = Dot().add_updater(lambda m, dt: m.set_x(len(static.points) / 100))
        self.add(static, dot)
        self.play(dot.animate.shift(UP), run_time=1)
        self.wait(0.5)
"""
)


def test_what_a_live_play_draws_into_the_scene_is_compared_with_nothing(
    scene_dir,
):
    """The camera's drawing writes into the static circle, even its static
    picture drawn as the play begins: the record's execution, which draws
    nothing, never holds that, so no sample of the play is compared.
    """
    session = _open_in(scene_dir, INTRUDING, "Intruding")
    try:
        session.answers([Fraction(1, 7)])
        (run,) = session.forward_runs
        assert run.compared_plays == 0
        assert run.divergence is None
    finally:
        session.close()


def test_states_without_frames_are_those_of_the_render(scene_dir):
    """A record refusing drawing is answered by a render whether or not
    frames are asked for: what its drawing writes is part of the execution.
    """
    session = _open_in(scene_dir, INTRUDING, "Intruding")
    try:
        (bare,) = session.answers([Fraction(1, 7)], frames=False)
        (drawn,) = session.answers([Fraction(1, 7)])
        assert bare.frame is None
        assert bare.state.named("dot").box == drawn.state.named("dot").box
    finally:
        session.close()


def test_an_instant_after_the_script_ended_is_answered_from_its_end(session_of):
    session = session_of("Mapped", CAMERA_SCENES, frame_rate=6)
    end = session.record.duration
    (answer,) = session.answers([end])
    assert answer.state.answer == "forward"
    assert answer.state.play is None
    _, frames = render(camera_scenes.Mapped, 6)
    # The scene as its script left it, which the held wait already showed.
    assert np.array_equal(_pixels(answer.frame), frames[-1])
    (run,) = session.forward_runs
    assert run.plays == len(session.record.plays)
    assert run.divergence is None


# What a forward run leaves


def _origins() -> tuple:
    entries = {
        key: (ref(), origin)
        for key, (ref, origin) in ORIGINS._entries.items()
        if ref() is not None
    }
    return ORIGINS.active, ORIGINS.base_dir, entries, dict(ORIGINS._ordinals)


def test_a_forward_run_leaves_the_session_as_it_was(session_of):
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    record = session.record
    first, second = record.plays
    asked = [
        Fraction(0),
        Fraction(4, 15),
        _unsampled(second)[0],
        Fraction(3, 2),
        record.duration,
    ]
    before = session.answers(asked)
    assert {answer.state.answer for answer in before} == {"recorded", "restored"}
    summary = session.summary().to_json()
    captures = list(record._captures)
    origins = _origins()
    randoms = (random.getstate(), np.random.get_state())  # noqa: NPY002
    module_shift = session.module.SHIFT
    modules = set(sys.modules)

    forward = session.answers(_unsampled(first))
    assert {answer.state.answer for answer in forward} == {"forward"}

    assert session.record is record
    assert record._captures == captures
    assert session.summary().to_json() == summary
    assert _origins() == origins
    after_random = (random.getstate(), np.random.get_state())  # noqa: NPY002
    assert after_random[0] == randoms[0]
    assert all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(after_random[1], randoms[1], strict=True)
    )
    assert session.module.SHIFT == module_shift == 1.0
    assert set(sys.modules) == modules
    after = session.answers(asked)
    for earlier, again in zip(before, after, strict=True):
        assert again.state == earlier.state
        assert np.array_equal(_pixels(again.frame), _pixels(earlier.frame))


@pytest.mark.parametrize(
    ("name", "path"),
    [("ReboundGlobal", RESTORE_SCENES), ("Mapped", CAMERA_SCENES)],
)
def test_forward_answers_without_frames_and_at_a_quality(session_of, name, path):
    """Asked for states alone, a forward run answers the same states; asked
    at a quality, its frames have that quality's size.
    """
    session = session_of(name, path, frame_rate=15)
    asked = _unsampled(session.record.plays[0])
    states = session.answers(asked, frames=False)
    assert all(answer.frame is None for answer in states)
    drawn = session.answers(asked, "low_quality")
    assert [answer.state for answer in states] == [answer.state for answer in drawn]
    assert {answer.state.answer for answer in drawn} == {"forward"}
    assert {answer.frame.size for answer in drawn} == {(854, 480)}
    assert len(session.forward_runs) == 2


# After the probe recorded the scene again

DEMOTED_THEN_FORWARD = (
    HEADER
    + """
class DemotedThenForward(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        data = HERE / "data.txt"
        start = float(data.read_text())
        # What the next execution reads.
        data.write_text("1.5")
        self.count = DecimalNumber(0, num_decimal_places=0).to_corner(UP + RIGHT)
        tracker = ValueTracker(start)
        dot = always_redraw(lambda: Dot([tracker.get_value(), 0, 0]))
        self.add(self.count, dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)
        self.counting = False
        later = Dot().add_updater(lambda m: m.set_y(tracker.get_value() + SHIFT))
        self.add(later)
        self.play(tracker.animate.set_value(0), run_time=1)
        SHIFT = 1.0

    def update_to_time(self, t):
        super().update_to_time(t)
        if getattr(self, "counting", True) and hasattr(self, "count"):
            self.count.set_value(self.count.number + 1)
"""
)


def test_a_forward_run_runs_as_the_execution_the_session_kept(scene_dir, monkeypatch):
    """The probe demotes the counted play, and the scene is recorded again,
    reading what the first execution wrote: the session holds that
    execution. A forward run ticks the demoted play as its record does, and
    the input it names is the one it read otherwise than the kept execution
    (the same as the first). Its divergence has the cause the recording
    again already kept, so it is kept once.
    """
    from arachne.timeline import admission, fallback

    made = []

    class Kept(fallback._ForwardRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(fallback, "_ForwardRecorder", Kept)
    admission.DEMOTED.clear()
    data = scene_dir / "data.txt"
    data.write_text("0.5")
    session = _open_in(scene_dir, DEMOTED_THEN_FORWARD, "DemotedThenForward")
    try:
        assert session.probe.demoted == (0,)
        assert session.record.plays[0].route == "ticked"
        # What the first execution read, which the kept one did not.
        data.write_text("0.5")
        (answer,) = session.answers([Fraction(8, 7)])
        assert answer.state.answer == "forward"
        (forward,) = made
        play = forward.record.plays[0]
        assert play.route == "ticked"
        assert play.admission.reasons[0].startswith(admission.PROBE_DEMOTED)
        (run,) = session.forward_runs
        assert run.divergence is not None
        assert run.divergence.play == 0
        assert run.divergence.changed_inputs == ("data.txt",)
        causes = [divergence.cause for divergence in session.divergences]
        assert causes.count(run.divergence.cause) == 1
    finally:
        session.close()
        admission.DEMOTED.clear()


EDITS_ITS_HELPER = (
    HEADER
    + """
import probe_edited_helper


class EditsItsHelper(Scene):
    def construct(self):
        # Each execution leaves the helper, part of the source, edited.
        helper = HERE / "probe_edited_helper.py"
        helper.write_text(helper.read_text() + "# executed\\n")
        self.count = DecimalNumber(0, num_decimal_places=0).to_corner(UP + RIGHT)
        tracker = ValueTracker(0)
        dot = always_redraw(lambda: Dot([tracker.get_value(), 0, 0]))
        self.add(self.count, dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=linear)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "count"):
            self.count.set_value(self.count.number + 1)
"""
)


def test_a_source_changed_while_recording_again_makes_the_record_stale(scene_dir):
    """The probe demotes a play, and the scene is recorded again from a
    source its first execution edited: the record is stale, as a forward
    run of a changed source finds it, and no divergence is kept.
    """
    from arachne.timeline import admission

    admission.DEMOTED.clear()
    (scene_dir / "probe_edited_helper.py").write_text("VALUE = 1\n")
    session = _open_in(scene_dir, EDITS_ITS_HELPER, "EditsItsHelper")
    try:
        assert session.stale
        assert not session.probe.settled
        assert session.record.plays[0].route == "ticked"
        assert [d.kind for d in session.divergences] == ["probe"]
    finally:
        session.close()
        admission.DEMOTED.clear()
        sys.modules.pop("probe_edited_helper", None)


EDITED_AFTER_THE_SHIFT = (
    HEADER
    + """
import shift_edited_helper


class EditedAfterTheShift(Scene):
    def construct(self):
        counter = HERE / "counter.txt"
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        if count == 2:
            # The third execution, which only ticks a line whole, is of
            # another source.
            (HERE / "shift_edited_helper.py").write_text("VALUE = 2\\n")
        for _ in range(count + 1):
            tracker = ValueTracker(0)
            self.mark = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
            self.add(self.mark)
            self.play(tracker.animate.set_value(2), run_time=0.4)
            self.mark.clear_updaters()

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "mark"):
            self.mark.set_y(t)
"""
)


def test_a_source_changed_after_probing_stopped_makes_the_record_stale(scene_dir):
    """The plays shift, so the probe stops, and the scene is recorded once
    more to tick the shifted line whole: that execution is of an edited
    source, so the record is stale, though the probe no longer runs.
    """
    from arachne.timeline import admission

    admission.DEMOTED.clear()
    (scene_dir / "shift_edited_helper.py").write_text("VALUE = 1\n")
    session = _open_in(
        scene_dir, EDITED_AFTER_THE_SHIFT, "EditedAfterTheShift", frame_rate=15
    )
    try:
        assert (scene_dir / "counter.txt").read_text() == "3"
        assert [play.route for play in session.record.plays] == ["ticked"] * 3
        assert not session.probe.settled
        assert session.stale
    finally:
        session.close()
        admission.DEMOTED.clear()
        sys.modules.pop("shift_edited_helper", None)


ROLES_THEN_REBOUND = (
    HEADER
    + """
class RolesThenRebound(Scene):
    def construct(self):
        global SHIFT
        SHIFT = 0.0
        self.role = None
        for role in (HERE / "roles.txt").read_text().split(","):
            self.role = role
            tracker = ValueTracker(0)
            self.dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
            self.add(self.dot)
            self.play(tracker.animate.set_value(2), run_time=0.4)
            self.dot.clear_updaters()
        self.role = None
        tracker = ValueTracker(0)
        later = Dot().add_updater(lambda m: m.set_y(tracker.get_value() + SHIFT))
        self.add(later)
        self.play(tracker.animate.set_value(1), run_time=0.4)
        SHIFT = 1.0

    def update_to_time(self, t):
        super().update_to_time(t)
        if self.role == "bad":
            self.dot.set_y(t)
"""
)


@pytest.fixture
def forward_recorders(monkeypatch):
    """The recorders of the forward runs from here on, as they are made."""
    from arachne.timeline import fallback

    made = []

    class Kept(fallback._ForwardRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(fallback, "_ForwardRecorder", Kept)
    return made


@pytest.mark.parametrize(
    ("roles", "routes"),
    [
        # Another occurrence before the demoted one: which play was demoted is
        # unknown, so every play of its line ticks.
        ("new,clean,bad", ["ticked", "ticked", "ticked"]),
        # The same plays: ticking the line whole changes no state compared.
        ("clean,bad", ["ticked", "ticked", "state"]),
    ],
)
def test_a_forward_run_ticks_every_play_of_a_demoted_line(
    scene_dir, forward_recorders, roles, routes
):
    """The probe demotes the second play of a line. A forward run ticks every
    play of that line, wherever the line's plays fall in it: it answers as
    a session opened on the scene as it is now does, and, where the scene
    plays the same plays, finds no divergence.
    """
    from arachne.timeline import admission

    admission.DEMOTED.clear()
    path = scene_dir / "roles.txt"
    path.write_text("clean,bad")
    session = _open_in(scene_dir, ROLES_THEN_REBOUND, "RolesThenRebound", frame_rate=15)
    try:
        assert [play.route for play in session.record.plays] == [
            "state",
            "ticked",
            "state",
        ]
        assert session.probe.settled
        path.write_text(roles)
        # In the record's last play, which can't be restored there.
        t = Fraction(20, 21)
        (answer,) = session.answers([t], frames=False)
        assert answer.state.answer == "forward"
        (forward,) = forward_recorders
        assert [play.route for play in forward.record.plays] == routes
        (run,) = session.forward_runs
        if roles == "clean,bad":
            assert run.divergence is None
            assert [d.kind for d in session.divergences] == ["probe"]
            return
        oracle = _open_in(
            scene_dir, ROLES_THEN_REBOUND, "RolesThenRebound", frame_rate=15
        )
        try:
            expected = oracle.state_at(t)
            assert answer.state.named("self.dot").box == expected.named("self.dot").box
        finally:
            oracle.close()
    finally:
        session.close()
        admission.DEMOTED.clear()
