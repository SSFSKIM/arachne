"""Content keys cover every field a mobject holds: two states that draw
differently never key alike, in the record's key, in an execution's summary
or in the probe.

A value of a type the key does not know keys as a marker that never equals
another, so every comparison that meets one finds a difference; one of
every mobject class the engine exports keys without it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import threading
from enum import Enum, IntEnum
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.constants import CapStyleType, LineJointType
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.execution import ExecutionSummary, compare, play_keys
from arachne.timeline.keys import OVER_BUDGET_KIND, unkeyable
from arachne.timeline.record import (
    IDENTITY_FIELDS,
    STRUCTURE_FIELDS,
    _value_key,
    covered_fields,
)

from . import key_scenes

#: A small frame at 4 fps, so a session costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}

HEADER = """\
from pathlib import Path

from arachne import *

HERE = Path(__file__).parent
"""


@pytest.fixture
def scene_dir(tmp_path):
    """A directory for scenes, whose modules are forgotten afterwards."""
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


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


def _open(path: Path, scene: str, **settings) -> Session:
    media = path.parent.parent / "media"
    return Session.open(
        path, scene, config={**SMALL, **settings, "media_dir": str(media)}
    )


def _pixels(session: Session, t: Fraction) -> np.ndarray:
    return np.asarray(session.frame(t), dtype=np.int16)


def _two_executions(scene_dir, source, scene, data, first, second, **settings):
    """Two executions of ``source``, the file ``data`` holding ``first`` for
    the first and ``second`` for the other.
    """
    path = _write(scene_dir, "scene.py", source)
    _write(scene_dir, data, first)
    before = _open(path, scene, **settings)
    _write(scene_dir, data, second)
    after = _open(path, scene, **settings)
    return before, after


def _across(summary: ExecutionSummary) -> ExecutionSummary:
    """``summary`` as another process reads it."""
    return ExecutionSummary.from_json(summary.to_json())


# Values


class Shade(Enum):
    LIGHT = 0
    DARK = 1


class Level(IntEnum):
    LOW = 0


class Note:
    """A value of a type the content key does not know: no attributes of
    its own to key it by.
    """

    __slots__ = ()


class Helper:
    """An object of a plain class, keyed by its attributes."""

    def __init__(self, k, held=None):
        self.k = k
        self.held = held


def test_an_enum_keys_by_its_class_and_its_members_value():
    assert _value_key(CapStyleType.ROUND) == _value_key(CapStyleType.ROUND)
    assert _value_key(CapStyleType.ROUND) != _value_key(CapStyleType.BUTT)
    # The same value in another class keys apart, an int enum's included.
    assert CapStyleType.AUTO.value == LineJointType.AUTO.value == Shade.LIGHT.value
    keys = {
        _value_key(member)
        for member in (CapStyleType.AUTO, LineJointType.AUTO, Shade.LIGHT, Level.LOW)
    }
    assert len(keys) == 4
    assert _value_key(Level.LOW) != _value_key(0)


def test_an_enum_keys_alike_in_another_process():
    code = (
        "from arachne.constants import CapStyleType;"
        "from arachne.timeline.record import _value_key;"
        "print(_value_key(CapStyleType.ROUND).hex())"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert run.stdout.split()[-1] == _value_key(CapStyleType.ROUND).hex()


def test_containers_key_by_what_they_hold():
    assert _value_key((1, 2.5)) == _value_key((1, 2.5))
    assert _value_key((1, 2.5)) != _value_key([1, 2.5])
    assert _value_key((1, 2.5)) != _value_key((1, 2.25))
    assert _value_key({"a": 1, "b": 2}) == _value_key({"b": 2, "a": 1})
    assert _value_key({"a": 1}) != _value_key({"a": 2})
    assert _value_key(np.array([1.0, 2.0])) != _value_key(np.array([1.0, 3.0]))
    assert _value_key(np.int64(3)) != _value_key(np.int32(3))
    looped: list = [1]
    looped.append(looped)
    assert _value_key(looped) == _value_key(looped)
    # A large array keys as a digest, however deep it sits.
    large = np.arange(100_000, dtype=float)
    assert len(_value_key(large)) < 100
    assert len(_value_key([[large]])) < 200
    assert _value_key(large) != _value_key(large + 1)


def test_a_function_keys_by_its_code_and_what_it_closes_over():
    def scaled(factor):
        return lambda x: factor * x

    assert _value_key(scaled(2)) == _value_key(scaled(2))
    assert _value_key(scaled(2)) != _value_key(scaled(3))
    assert _value_key(lambda x: x + 1) != _value_key(lambda x: x + 2)


def test_a_value_of_a_type_the_key_does_not_know_never_keys_alike():
    note = Note()
    assert unkeyable(note) == f"{__name__}.Note"
    assert _value_key(note) != _value_key(note)
    # Inside a container too: the whole value is the marker.
    assert unkeyable({"note": note}) == f"{__name__}.Note"
    assert _value_key([note]) != _value_key([note])
    assert unkeyable(CapStyleType.ROUND) is None


def test_an_object_of_a_plain_class_keys_by_its_attributes():
    assert _value_key(Helper(1.0)) == _value_key(Helper(1.0))
    assert _value_key(Helper(1.0)) != _value_key(Helper(2.0))
    # What it holds of this process alone is still the marker.
    held = Helper(1.0, held=threading.Lock())
    assert (
        unkeyable(held)
        == f"{type(held.held).__module__}.{type(held.held).__qualname__}"
    )
    assert _value_key(held) != _value_key(held)


def test_the_fields_a_key_leaves_out_are_named():
    assert {"original_id"} == IDENTITY_FIELDS
    assert {"submobjects", "updaters"} == STRUCTURE_FIELDS


# Two executions


# The line also holds an enum and a function the scene defines: each session
# imports the scene under a module name of its own, which keys nothing.
CAPS = (
    HEADER
    + """
from enum import Enum


class Shade(Enum):
    DARK = 1


def tint(x):
    return 0.5 * x


class Caps(Scene):
    def construct(self):
        style = (HERE / "style.txt").read_text().strip()
        line = Line(LEFT * 3, RIGHT * 3, stroke_width=200)
        line.cap_style = CapStyleType[style]
        line.shade = Shade.DARK
        line.tint = tint
        self.add(line)
        self.wait(0.5)
"""
)


@pytest.mark.parametrize(("first", "second"), [("BUTT", "ROUND"), ("ROUND", "ROUND")])
def test_executions_that_differ_only_in_a_lines_cap_style_compare_apart(
    scene_dir, first, second
):
    before, after = _two_executions(scene_dir, CAPS, "Caps", "style.txt", first, second)
    try:
        t = Fraction(1, 4)
        drawn_apart = bool((_pixels(before, t) != _pixels(after, t)).any())
        assert drawn_apart == (first != second)
        found = before.compare(_across(after.summary()))
        if first == second:
            assert found is None
        else:
            assert found is not None
            assert "cap_style" in found.reason
            assert found.changed_inputs == ("style.txt",)
    finally:
        before.close()
        after.close()


# The walk meets each child first as a root, so every count and every
# node's values agree whichever child the parent holds.
HELD = (
    HEADER
    + """
class Held(Scene):
    def construct(self):
        which = (HERE / "which.txt").read_text().strip()
        a = Square(side_length=3, color=RED, fill_opacity=1)
        b = Square(side_length=3, color=BLUE, fill_opacity=1)
        self.add(a, b)
        parent = VGroup(a if which == "a" else b)
        self.add(parent)
        self.wait(0.5)
"""
)

# The parent made empty with the roots, and pointed at a child before the
# play (its start state's draw lists hold it) or after the last play (the
# state the script left, which has no draw lists of a play).
POINTED = (
    HEADER
    + """
class Pointed(Scene):
    def construct(self):
        which = (HERE / "which.txt").read_text().strip()
        a = Square(side_length=3, color=RED, fill_opacity=1)
        b = Square(side_length=3, color=BLUE, fill_opacity=1)
        parent = VGroup()
        self.add(a, b, parent)
        if WHEN == "before":
            parent.add(a if which == "a" else b)
        self.wait(0.5)
        if WHEN == "after":
            parent.add(a if which == "a" else b)
"""
)


@pytest.mark.parametrize(
    ("source", "scene"),
    [
        (HELD, "Held"),
        (POINTED.replace("WHEN", repr("before")), "Pointed"),
        (POINTED.replace("WHEN", repr("after")), "Pointed"),
    ],
    ids=["made-holding", "pointed-before-the-play", "pointed-after-the-last-play"],
)
@pytest.mark.parametrize(("first", "second"), [("a", "b"), ("b", "b")])
def test_executions_whose_parent_holds_another_child_compare_apart(
    scene_dir, source, scene, first, second
):
    before, after = _two_executions(
        scene_dir, source, scene, "which.txt", first, second
    )
    try:
        final = before.record.final.instant
        drawn_apart = bool((_pixels(before, final) != _pixels(after, final)).any())
        assert drawn_apart == (first != second)
        mine, theirs = before.summary(), after.summary()
        found = compare(mine, _across(theirs))
        if first == second:
            assert found is None
            assert mine.final.key == theirs.final.key
        else:
            assert found is not None
            assert mine.final.key != theirs.final.key
            if scene == "Pointed":
                # Every count and every node's values agree: only which
                # child the parent holds tells the two apart.
                assert "submobjects are other mobjects" in found.reason
    finally:
        before.close()
        after.close()


# The probe

#: A frame hook writing a field by how many frames it has seen: history
#: no restore can know.
HOOK = """
from arachne import *


class Hook(Scene):
    def construct(self):
        self.line = Line(LEFT * 3, RIGHT * 3, stroke_width=200)
        self.frames = 0
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(self.line, mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "line"):
            self.frames += 1
            on = self.frames >= 3
            WRITE
"""


@pytest.mark.parametrize(
    ("write", "demoted"),
    [
        ("self.line.width_note = 2.0 if on else 1.0", True),
        ("self.line.cap_style = CapStyleType.ROUND if on else CapStyleType.BUTT", True),
        ("self.line.cap_style = CapStyleType.ROUND", False),
    ],
    ids=["a-number-by-history", "an-enum-by-history", "an-enum-always"],
)
def test_the_probe_demotes_a_hook_writing_an_enum_by_history_as_it_does_a_number(
    scene_dir, write, demoted
):
    path = _write(scene_dir, "hook.py", HOOK.replace("WRITE", write))
    session = _open(path, "Hook", frame_rate=15)
    try:
        outcomes = [(play.outcome, play.reason) for play in session.probe.plays]
        if demoted:
            assert session.probe.demoted == (0,), outcomes
            assert session.record.plays[0].route == "ticked"
        else:
            assert session.probe.demoted == (), outcomes
            assert session.record.plays[0].route == "state"
    finally:
        session.close()


# The marker in every comparison

#: A line holding a value the key cannot key (``NOTE``), through a state
#: play.
NOTED = """
import threading

from arachne import *


class Note:
    def __init__(self):
        self.lock = threading.Lock()


class Noted(Scene):
    def construct(self):
        line = Line(LEFT, RIGHT)
        line.note = NOTE
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(line, mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)
"""


@pytest.mark.parametrize(
    ("note", "why"),
    [
        ("Note()", "a value of a type the key does not know (_thread.lock)"),
        ("[[0] * 1000 for _ in range(200)]", "too large to key"),
    ],
    ids=["of-a-type-the-key-does-not-know", "over-the-budget"],
)
def test_a_value_the_key_cannot_key_is_a_difference_wherever_it_is_compared(
    scene_dir, note, why
):
    path = _write(scene_dir, "noted.py", NOTED.replace("NOTE", note))
    first = _open(path, "Noted", frame_rate=15)
    second = _open(path, "Noted", frame_rate=15)
    try:
        # Two executions of one unchanged source: never found alike, and
        # why is said, each marker apart from the other.
        assert play_keys(first.record) != play_keys(second.record)
        found = first.compare(_across(second.summary()))
        assert found is not None
        assert "note" in found.reason
        assert why in found.reason
        # The probe: a restored state never agrees with the kept one.
        assert first.probe.demoted == (0,)
        assert why in (first.probe.plays[0].reason or "")
    finally:
        first.close()
        second.close()


# A scene's own attribute, read later

#: The scene keeps ``k`` on itself and reads it only as play 1 runs, in the
#: code ``READ`` names: an updater, a redrawn mobject, or a function of a
#: tracker that is a method of the scene. Play 0 runs with the tracker at 0,
#: where ``k`` changes nothing.
LATER = (
    HEADER
    + """
class Later(Scene):
    def at(self, value):
        return RIGHT * self.k * value

    def construct(self):
        self.k = float((HERE / "k.txt").read_text())
        tracker = ValueTracker(0)
        READ
        self.add(shown)
        self.wait(0.25)
        self.play(tracker.animate.set_value(1), run_time=0.5)
        self.wait(0.25)
"""
)

READS_LATER = {
    "an-updater": (
        "shown = Dot().add_updater(lambda m: m.set_x(self.k * tracker.get_value()))"
    ),
    "a-redrawn-mobject": (
        "shown = always_redraw(lambda: Dot(RIGHT * self.k * tracker.get_value()))"
    ),
    "a-method-of-the-scene": (
        "shown = Dot().add_updater(lambda m: m.move_to(self.at(tracker.get_value())))"
    ),
}


@pytest.mark.parametrize("read", list(READS_LATER.values()), ids=list(READS_LATER))
@pytest.mark.parametrize(("first", "second"), [("1.0", "2.0"), ("2.0", "2.0")])
def test_a_scene_attribute_read_during_a_play_diverges_at_that_play(
    scene_dir, read, first, second
):
    """Keying a scene by its type rests on this: what the scene's own
    attributes do reaches a frame only through state the record keys, so
    two executions apart in one diverge at the first play it changes, and
    each answers that play from its own record.
    """
    source = LATER.replace("READ", read)
    before, after = _two_executions(
        scene_dir, source, "Later", "k.txt", first, second, frame_rate=15
    )
    try:
        found = before.compare(_across(after.summary()))
        inside = Fraction(1, 2)  # inside play 1, its tracker past 0
        mine, theirs = _pixels(before, inside), _pixels(after, inside)
        assert bool((mine != theirs).any()) == (first != second)
        if first == second:
            assert found is None
        else:
            assert found is not None
            assert found.play == 1, found.reason
        # No restore stands in for another execution's state: play 1 is a
        # state play in each, its restores built from that execution's own
        # record, and the probe finds them agreeing with it.
        for session in (before, after):
            assert session.record.plays[1].route == "state"
            outcomes = {probe.play: probe.outcome for probe in session.probe.plays}
            assert outcomes == {1: "agreed"}, session.probe.plays
    finally:
        before.close()
        after.close()


# Ordinary ways of handing a mobject a function

#: Graphs of a method of the scene, a lambda over the scene, a cached
#: function, a wrapped one and a vectorized one, and a dot holding an
#: object of the scene's own class, through a state play.
IDIOMS = (
    HEADER
    + """
import functools

import numpy as np


class Gain:
    def __init__(self, k):
        self.k = k
        self.label = "gain"


@functools.lru_cache(maxsize=None)
def cached(x):
    return x * x


def logged(function):
    @functools.wraps(function)
    def inner(x):
        return function(x)

    return inner


@logged
def wrapped(x):
    return x / 2


class Idioms(Scene):
    def f(self, x):
        return 0.5 * x

    def construct(self):
        k = float((HERE / "k.txt").read_text())
        axes = Axes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], x_length=4, y_length=3)
        graphs = VGroup(
            axes.plot(self.f),
            axes.plot(lambda x: self.f(x)),
            axes.plot(cached),
            axes.plot(wrapped),
            axes.plot(np.vectorize(lambda x: x / 3)),
        )
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        dot.gain = Gain(k)
        self.add(axes, graphs, dot)
        self.play(tracker.animate.set_value(1), run_time=0.4)
"""
)


@pytest.mark.parametrize(("first", "second"), [("1.0", "1.0"), ("1.0", "2.0")])
def test_ordinary_ways_of_handing_a_mobject_a_function_key_alike(
    scene_dir, first, second
):
    before, after = _two_executions(
        scene_dir, IDIOMS, "Idioms", "k.txt", first, second, frame_rate=15
    )
    try:
        for session in (before, after):
            for mob, node in session.record.plays[0].start_state.nodes():
                for name, value in covered_fields(node):
                    assert unkeyable(value) is None, (type(mob).__name__, name)
            assert session.record.plays[0].route == "state"
            assert session.probe.demoted == (), session.probe.plays
        found = before.compare(_across(after.summary()))
        if first == second:
            assert found is None
            assert play_keys(before.record) == play_keys(after.record)
        else:
            # The scene's own object changed, and nothing drawn: still apart.
            assert found is not None
            assert "gain" in found.reason
    finally:
        before.close()
        after.close()


def test_ordinary_ways_of_handing_a_mobject_a_function_key_alike_in_another_process(
    scene_dir,
):
    path = _write(scene_dir, "scene.py", IDIOMS)
    _write(scene_dir, "k.txt", "1.0")
    media = scene_dir.parent / "media"
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        [sys.executable, "-c", SUMMARY, str(path), "Idioms", str(media)],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    theirs = json.loads(run.stdout.splitlines()[-1])
    session = _open(path, "Idioms")
    try:
        assert theirs["play_keys"] == [key.hex() for key in play_keys(session.record)]
        assert session.compare(ExecutionSummary.from_json(theirs["summary"])) is None
    finally:
        session.close()


# A field that refers to a mobject

#: A dot whose ``target`` is one square or the other: nothing drawn differs.
TARGET = (
    HEADER
    + """
class Target(Scene):
    def construct(self):
        which = (HERE / "which.txt").read_text().strip()
        a, b = Square(), Square().shift(RIGHT)
        dot = Dot()
        dot.target_square = a if which == "a" else b
        self.add(a, b, dot)
        self.wait(0.5)
"""
)


@pytest.mark.parametrize(("first", "second"), [("a", "b"), ("a", "a")])
def test_a_field_pointed_at_another_mobject_keys_apart(scene_dir, first, second):
    before, after = _two_executions(
        scene_dir, TARGET, "Target", "which.txt", first, second
    )
    try:
        apart = first != second
        assert (play_keys(before.record) != play_keys(after.record)) == apart
        mine, theirs = before.summary(), after.summary()
        assert (
            mine.plays[0].start_state.key != theirs.plays[0].start_state.key
        ) == apart
        found = compare(mine, _across(theirs))
        if apart:
            assert found is not None
            assert "fields refer to other mobjects" in found.reason
        else:
            assert found is None
    finally:
        before.close()
        after.close()


def test_the_probe_demotes_a_hook_pointing_a_field_at_another_mobject_by_history(
    scene_dir,
):
    write = "self.line.target_mobject = self.mover if on else self.line"
    source = HOOK.replace("WRITE", write).replace(
        "self.add(self.line, mover)",
        "self.mover = mover\n        self.add(self.line, mover)",
    )
    path = _write(scene_dir, "hook.py", source)
    session = _open(path, "Hook", frame_rate=15)
    try:
        assert session.probe.demoted == (0,), session.probe.plays
        assert "target_mobject" in (session.probe.plays[0].reason or "")
    finally:
        session.close()


# What the summary records


def test_a_summary_names_its_interpreter_and_refuses_another(every_mobject):
    from arachne.timeline.execution import Incomparable

    mine = every_mobject.summary()
    assert mine.interpreter == sys.implementation.cache_tag
    assert _across(mine).interpreter == mine.interpreter
    other = dataclasses.replace(_across(mine), interpreter="other-99")
    with pytest.raises(Incomparable, match="other-99"):
        compare(mine, other)


def test_an_enum_field_never_reads_as_a_string_of_its_name():
    from arachne.timeline.execution import _field_leaf

    leaf, _ = _field_leaf(CapStyleType.ROUND)
    assert leaf != "CapStyleType.ROUND"
    assert _field_leaf(leaf)[0] != leaf


def test_a_value_too_large_to_key_is_a_marker_of_its_own():
    big = [[0] * 1000 for _ in range(200)]
    assert unkeyable(big) == OVER_BUDGET_KIND
    assert _value_key(big) != _value_key(big)
    assert _value_key(big).startswith(b"\0over budget")
    assert _value_key(Note()).startswith(b"\0unkeyable")


# Every mobject the engine exports


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """One media directory for this module's sessions and processes: what
    one of them typeset, the others find.
    """
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="module")
def every_mobject(media):
    admission.DEMOTED.clear()
    session = Session.open(
        Path(key_scenes.__file__),
        "EveryMobject",
        config={**SMALL, "media_dir": str(media)},
    )
    yield session
    session.close()
    admission.DEMOTED.clear()


def test_every_mobject_the_engine_exports_is_in_the_scene(every_mobject):
    made = {type(mob) for mob in every_mobject.record._scene.made}
    missing = {cls.__qualname__ for cls in key_scenes.public_mobject_classes()} - {
        cls.__qualname__ for made_class in made for cls in made_class.__mro__
    }
    assert missing == set()


def test_every_field_of_every_mobject_the_engine_exports_has_a_key(every_mobject):
    record = every_mobject.record
    states = [play.start_state for play in record.plays]
    found: dict[tuple[str, str], str] = {}
    for state in states:
        for mob, node in state.nodes():
            for name, value in zip(node.keys, node.values, strict=True):
                if name in IDENTITY_FIELDS or name in STRUCTURE_FIELDS:
                    continue
                kind = unkeyable(value)
                if kind is not None:
                    found.setdefault((type(mob).__qualname__, name), kind)
    assert found == {}
    assert len(list(states[0].nodes())) > len(key_scenes.public_mobject_classes())


#: One process's summary and play keys of a scene, as JSON.
SUMMARY = """
import json, sys
from pathlib import Path
from arachne.session import Session
from arachne.timeline.execution import play_keys

session = Session.open(Path(sys.argv[1]), sys.argv[2], config={
    "pixel_width": 96, "pixel_height": 54, "frame_rate": 4,
    "progress_bar": "none", "verbosity": "ERROR", "media_dir": sys.argv[3]})
print(json.dumps({
    "summary": session.summary().to_json(),
    "play_keys": [key.hex() for key in play_keys(session.record)],
}))
session.close()
"""


def test_every_mobject_the_engine_exports_keys_alike_in_another_process(media):
    """Every mobject whose points one process makes as another does
    (:data:`.key_scenes.UNSTABLE_ACROSS_PROCESSES`).
    """
    scene = "EveryMobjectMadeAlike"
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        [sys.executable, "-c", SUMMARY, key_scenes.__file__, scene, str(media)],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    theirs = json.loads(run.stdout.splitlines()[-1])
    session = Session.open(
        Path(key_scenes.__file__), scene, config={**SMALL, "media_dir": str(media)}
    )
    try:
        assert theirs["play_keys"] == [key.hex() for key in play_keys(session.record)]
        assert session.compare(ExecutionSummary.from_json(theirs["summary"])) is None
    finally:
        session.close()
