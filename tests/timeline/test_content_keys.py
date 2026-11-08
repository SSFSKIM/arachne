"""Content keys cover every field a mobject holds: two states that draw
differently never key alike, in the record's key, in an execution's summary
or in the probe.

A value of a type the key does not know keys as a marker that never equals
another, so every comparison that meets one finds a difference; one of
every mobject class the engine exports keys without it.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import json
import os
import random
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
from arachne.timeline.keys import (
    OVER_BUDGET_KIND,
    OVER_DEPTH_KIND,
    marker_reason,
    unkeyable,
)
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
    # A dict by its order too; a set by its parts alone.
    assert _value_key({"a": 1, "b": 2}) != _value_key({"b": 2, "a": 1})
    assert _value_key({"a", "b"}) == _value_key({"b", "a"})
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
import functools
import threading

from arachne import *


@functools.lru_cache(maxsize=None)
def squared(x):
    return x * x


class Note:
    def __init__(self):
        self.lock = threading.Lock()


class Link:
    def __init__(self, after):
        self.after = after


def chain(length):
    link = None
    for _ in range(length):
        link = Link(link)
    return link


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
        (
            "[[0] * 1000 for _ in range(200)]",
            "a builtins.list of more than 100000 parts, too large to key",
        ),
        ("chain(1000)", "Link nested deeper than"),
        ("squared", "a cache whose entries the key cannot read (squared)"),
    ],
    ids=[
        "of-a-type-the-key-does-not-know",
        "over-the-budget",
        "nested-too-deep",
        "a-scene-s-own-cache",
    ],
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
#: code ``READ`` names. Play 0 runs with the tracker at 0, where ``k``
#: changes nothing. ``CHANGE`` is what the scene does to ``k`` after play 1
#: (play 2 then reads the new value).
LATER = (
    HEADER
    + """
class Later(Scene):
    def at(self, value):
        return RIGHT * self.k * value

    def point(self):
        return self.at(self.tracker.get_value())

    def construct(self):
        self.k = float((HERE / "k.txt").read_text())
        tracker = self.tracker = ValueTracker(0)
        READ
        self.add(shown)
        self.wait(0.25)
        self.play(tracker.animate.set_value(1), run_time=0.5)
        CHANGE
        self.play(tracker.animate.set_value(0.5), run_time=0.5)
"""
)

#: How the scene reads ``k`` as play 1 runs. The first three read it from
#: a closure held in the mobject's ``updaters``, which no key covers (they
#: are code, ``STRUCTURE_FIELDS``): they show that what the read does lands
#: in keyed state. The traced path reads it through a field the key covers,
#: ``traced_point_func``, a method of the scene, which keys as its function
#: and the scene's type: it is the case that rests on keying a scene by its
#: type.
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
    "a-traced-path-of-a-method": "shown = TracedPath(self.point, stroke_width=12)",
}


def _forward_frames(path: Path, scene: str, fps: int) -> list[np.ndarray]:
    """The frames a render of ``scene`` in ``path`` draws, executing it
    forward from its first line.
    """
    import importlib.util

    from .conftest import render

    spec = importlib.util.spec_from_file_location(f"forward_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _, frames = render(getattr(module, scene), fps)
    return [np.asarray(frame, dtype=np.int16) for frame in frames]


def _answers_as_forward(session: Session, frames: list[np.ndarray], fps: int) -> None:
    """Every frame ``session`` answers is the one a forward render draws, and
    the probe either agrees with each state play or demotes it.
    """
    for index, frame in enumerate(frames):
        t = Fraction(index, fps)
        assert np.array_equal(_pixels(session, t), frame), t
    for probe in session.probe.plays:
        assert probe.outcome in {"agreed", "demoted"}, session.probe.plays


@pytest.mark.parametrize("read", list(READS_LATER.values()), ids=list(READS_LATER))
@pytest.mark.parametrize(("first", "second"), [("1.0", "2.0"), ("2.0", "2.0")])
def test_a_scene_attribute_read_during_a_play_diverges_at_that_play(
    scene_dir, read, first, second
):
    """Keying a scene by its type rests on this: what the scene's own
    attributes do reaches a frame only through state the record keys, so
    two executions apart in one diverge at the first play it changes, and
    each answers its frames as a forward render draws them, never from the
    other's state. (Which cases exercise the scene's key: ``READS_LATER``.)
    """
    source = LATER.replace("READ", read).replace("CHANGE", "pass")
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
        # The session answering now is the second execution's, its input
        # as a forward render reads it.
        frames = _forward_frames(scene_dir / "scene.py", "Later", 15)
        _answers_as_forward(after, frames, 15)
    finally:
        before.close()
        after.close()


#: ``k`` changed by the scene where a restore of its plays cannot see it:
#: after play 1 (play 2 reads the new value), or during play 1, by a frame
#: hook counting its frames (history no restore knows).
STALE = {
    "after-play-1": ("CHANGE", "self.k = 3 * self.k"),
    "during-play-1-by-a-hook": (
        "    def construct(self):",
        "    frames = 0\n\n"
        "    def update_to_time(self, t):\n"
        "        super().update_to_time(t)\n"
        "        if hasattr(self, 'k'):\n"
        "            self.frames += 1\n"
        "            self.k = 1.0 + 0.1 * self.frames\n\n"
        "    def construct(self):",
    ),
}


def _on_a_class(text: str) -> str:
    """``text`` with the scene's ``k`` kept on a class of the scene's file
    instead (``Holder.k``): a class keys as its name, so what its attribute
    does must reach a frame through keyed state, or the restore must see
    it rebound.
    """
    return text.replace("self.k", "Holder.k").replace(
        "hasattr(self, 'k')", "hasattr(self, 'tracker')"
    )


ON_A_CLASS = _on_a_class(LATER).replace(
    "class Later(Scene):", "class Holder:\n    k = 0.0\n\n\nclass Later(Scene):"
)


@pytest.mark.parametrize("read", list(READS_LATER.values()), ids=list(READS_LATER))
@pytest.mark.parametrize(("first", "second"), [("1.0", "2.0"), ("2.0", "2.0")])
def test_a_class_attribute_read_during_a_play_diverges_at_that_play(
    scene_dir, read, first, second
):
    """A class keys as its name, not its attributes: two executions whose
    class attribute differs diverge at the play that reads it, and each
    answers as a forward render draws.
    """
    source = ON_A_CLASS.replace("READ", _on_a_class(read)).replace("CHANGE", "pass")
    before, after = _two_executions(
        scene_dir, source, "Later", "k.txt", first, second, frame_rate=15
    )
    try:
        found = before.compare(_across(after.summary()))
        if first == second:
            assert found is None
        else:
            assert found is not None
            assert found.play == 1, found.reason
        _answers_as_forward(
            after, _forward_frames(scene_dir / "scene.py", "Later", 15), 15
        )
    finally:
        before.close()
        after.close()


@pytest.mark.parametrize("read", list(READS_LATER.values()), ids=list(READS_LATER))
@pytest.mark.parametrize("stale", list(STALE.values()), ids=list(STALE))
def test_a_class_attribute_changed_where_no_restore_sees_it_is_answered_as_forward(
    scene_dir, read, stale
):
    """The class attribute changed after play 1, or during it by a frame
    hook: every frame is the forward render's, and the probe agrees with
    each state play or demotes it. (Admission ticks a play whose callbacks
    read a class's attribute, so no restore meets one.)
    """
    old, new = (_on_a_class(part) for part in stale)
    source = (
        ON_A_CLASS.replace("READ", _on_a_class(read))
        .replace(old, new)
        .replace("CHANGE", "pass")
    )
    path = _write(scene_dir, "scene.py", source)
    _write(scene_dir, "k.txt", "1.0")
    session = _open(path, "Later", frame_rate=15)
    try:
        # The plays reading it tick from the first execution: for that read
        # (a traced path's ticks for its dt updater first).
        for play in session.record.plays[1:]:
            assert play.route == "ticked", play.admission.reasons
            assert any(
                "reads Holder.k" in reason or "takes dt" in reason
                for reason in play.admission.reasons
            ), play.admission.reasons
        assert session.probe.plays == ()
        _answers_as_forward(session, _forward_frames(path, "Later", 15), 15)
    finally:
        session.close()


@pytest.mark.parametrize("read", list(READS_LATER.values()), ids=list(READS_LATER))
@pytest.mark.parametrize("stale", list(STALE.values()), ids=list(STALE))
def test_a_scene_attribute_changed_where_no_restore_sees_it_is_answered_as_forward(
    scene_dir, read, stale
):
    old, new = stale
    source = LATER.replace("READ", read).replace(old, new).replace("CHANGE", "pass")
    path = _write(scene_dir, "scene.py", source)
    _write(scene_dir, "k.txt", "1.0")
    session = _open(path, "Later", frame_rate=15)
    try:
        _answers_as_forward(session, _forward_frames(path, "Later", 15), 15)
    finally:
        session.close()


# Ordinary ways of handing a mobject a function

#: Graphs of a method of the scene, a lambda over the scene, a wrapped
#: function and a vectorized one, and a dot holding an object of the
#: scene's own class, through a state play. (A scene's own cache is a
#: marker: ``NOTED``.)
IDIOMS = (
    HEADER
    + """
import functools

import numpy as np


class Gain:
    def __init__(self, k):
        self.k = k
        self.label = "gain"


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


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class Link:
    """One link of a chain of plain objects."""

    def __init__(self, after):
        self.after = after


class Anything:
    """An object that makes a fresh callable for every name asked of it."""

    def __getattr__(self, name):
        return Anything()

    def __call__(self):
        return None


class Endless:
    """An object whose attributes recurse without end as they are read."""

    @property
    def __dict__(self):
        return self.__dict__


def test_a_value_nested_too_deep_to_key_is_a_marker_naming_its_type():
    chain = None
    for _ in range(1000):
        chain = Link(chain)
    # A chain deeper than the bound, and code that recurses as the key reads
    # it, which no bound on the value's depth sees.
    for value in (chain, Endless()):
        key = _value_key(value)  # never a RecursionError
        assert key != _value_key(value)
        assert unkeyable(value) == OVER_DEPTH_KIND
        why = marker_reason(key)
        assert why is not None
        assert f"{__name__}.{type(value).__qualname__}" in why
        assert "too deep to key" in why
    # A shallow chain keys.
    short = Link(Link(None))
    assert _value_key(short) == _value_key(Link(Link(None)))
    # An object making a fresh callable for every name asked of it holds no
    # state: it keys, as two of it behave alike.
    assert _value_key(Anything()) == _value_key(Anything())


class Scaled:
    """A callable object of its own state wrapping a function."""

    def __init__(self, function, k):
        self.__wrapped__ = function
        self.k = k

    def __call__(self, x):
        return self.k * self.__wrapped__(x)


class Seeded(random.Random):
    """Python's generator, subclassed."""


class SeededArray(np.random.Generator):
    """numpy's generator, subclassed."""


class Buffered(io.BytesIO):
    """An object of a class whose state a builtin base holds."""


#: The generators a scene seeds, each made from a seed.
GENERATORS = {
    "numpy-generator": np.random.default_rng,
    "numpy-random-state": np.random.RandomState,
    "python-random": random.Random,
    "numpy-generator-subclassed": lambda seed: SeededArray(np.random.PCG64(seed)),
    "python-random-subclassed": Seeded,
}


@pytest.mark.parametrize("make", list(GENERATORS.values()), ids=list(GENERATORS))
def test_a_random_generator_keys_by_its_state(make):
    """Seeded alike, two generators key alike; seeded apart, or one drawn
    from since, they key apart. A subclass's own attributes are keyed too.
    """
    assert _value_key(make(1)) == _value_key(make(1))
    assert _value_key(make(1)) != _value_key(make(2))
    drawn = make(1)
    before = _value_key(drawn)
    drawn.random()
    assert _value_key(drawn) != before
    first, second = make(1), make(1)
    if hasattr(first, "__dict__"):
        first.label, second.label = "a", "b"
        assert _value_key(first) != _value_key(second)
        second.label = "a"
        assert _value_key(first) == _value_key(second)


class Urandom(random.Random):
    """Python's generator, drawing from the system instead of its state."""

    def random(self):
        return int.from_bytes(os.urandom(7), "big") / 2**56


class UrandomArray(np.random.Generator):
    """numpy's generator, drawing from the system instead of its state."""

    def random(self, *args, **kwargs):
        return int.from_bytes(os.urandom(7), "big") / 2**56


def _overridden_on_the_instance():
    found = random.Random(1)
    found.random = lambda: int.from_bytes(os.urandom(7), "big") / 2**56
    return found


#: Generators whose draws do not come from the state they report.
DRAWING_ELSEWHERE = {
    "system-random": (random.SystemRandom, "random.SystemRandom"),
    "a-subclass-overriding-random": (lambda: Urandom(1), f"{__name__}.Urandom"),
    "a-numpy-subclass-overriding-random": (
        lambda: UrandomArray(np.random.PCG64(1)),
        f"{__name__}.UrandomArray",
    ),
    "an-instance-overriding-random": (_overridden_on_the_instance, "random.Random"),
}


@pytest.mark.parametrize(
    ("make", "name"), list(DRAWING_ELSEWHERE.values()), ids=list(DRAWING_ELSEWHERE)
)
def test_a_generator_drawing_elsewhere_than_its_state_is_a_marker(make, name):
    """Its state does not change as it draws, so it says nothing of what it
    draws next: the unknown-type marker, never a key of that state.
    """
    drawn = make()
    assert unkeyable(drawn) == name
    assert _value_key(drawn) != _value_key(drawn)
    assert marker_reason(_value_key(drawn)) == (
        f"a value of a type the key does not know ({name})"
    )


#: A dot holding a generator drawing from the system (``MAKE``), drawn from
#: between its plays.
DRAWS_ELSEWHERE = (
    HEADER
    + """
import os
import random


class Urandom(random.Random):
    def random(self):
        return int.from_bytes(os.urandom(7), "big") / 2**56


class DrawsElsewhere(Scene):
    def construct(self):
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        dot.rng = MAKE
        self.add(dot)
        self.play(tracker.animate.set_value(1), run_time=0.4)
        dot.rng.random()
        self.play(tracker.animate.set_value(0), run_time=0.4)
"""
)


@pytest.mark.parametrize(
    "make", ["random.SystemRandom()", "Urandom(1)"], ids=["system-random", "a-subclass"]
)
def test_a_generator_drawing_elsewhere_on_a_mobject_never_agrees(scene_dir, make):
    path = _write(scene_dir, "scene.py", DRAWS_ELSEWHERE.replace("MAKE", make))
    first = _open(path, "DrawsElsewhere", frame_rate=15)
    second = _open(path, "DrawsElsewhere", frame_rate=15)
    try:
        found = first.compare(_across(second.summary()))
        assert found is not None
        assert "a value of a type the key does not know" in found.reason
        assert play_keys(first.record) != play_keys(second.record)
        assert first.probe.demoted, first.probe.plays
    finally:
        first.close()
        second.close()


def test_a_random_generator_keys_alike_in_another_process():
    script = (
        "import random\n"
        "import numpy as np\n"
        "from arachne.timeline.keys import value_key\n"
        "for value in (np.random.default_rng(3), np.random.RandomState(3),"
        " random.Random(3)):\n"
        "    print(value_key(value)[0].hex())\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    mine = [
        _value_key(value).hex()
        for value in (
            np.random.default_rng(3),
            np.random.RandomState(3),
            random.Random(3),
        )
    ]
    assert found == mine


#: A dot holding a generator the scene seeds from ``seed.txt`` and draws
#: from between its plays, the draws drawn nowhere.
SEEDED = (
    HEADER
    + """
import numpy as np


class Seeded(Scene):
    def construct(self):
        seed = int((HERE / "seed.txt").read_text())
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        dot.rng = np.random.default_rng(seed)
        self.add(dot)
        self.play(tracker.animate.set_value(1), run_time=0.4)
        dot.rng.random()
        self.play(tracker.animate.set_value(0), run_time=0.4)
"""
)


@pytest.mark.parametrize(("first", "second"), [("4", "4"), ("4", "5")])
def test_a_generator_on_a_mobject_agrees_seeded_alike_and_diverges_apart(
    scene_dir, first, second
):
    before, after = _two_executions(
        scene_dir, SEEDED, "Seeded", "seed.txt", first, second, frame_rate=15
    )
    try:
        found = before.compare(_across(after.summary()))
        if first == second:
            assert found is None, found.reason
            assert play_keys(before.record) == play_keys(after.record)
        else:
            assert found is not None
            assert "rng" in found.reason
    finally:
        before.close()
        after.close()


class Tagged(list):
    """A list with an attribute of its own."""


@dataclasses.dataclass
class Spec:
    width: float


class Attributed:
    """An object with attributes."""


class Slotted(Attributed):
    """An object with attributes and a slot."""

    __slots__ = ("slot",)


class Mixin:
    """A class with nothing of a builtin's, mixed into one."""


class Word(str, Mixin):
    """A string with a mixin after it, so ``str`` is not its last base."""


class Unloaded:
    """An object whose attributes raise when read, as a lazy load can."""

    @property
    def __dict__(self):
        raise RuntimeError("not loaded")


class Unready:
    """An object whose attributes raise another error when read."""

    @property
    def __dict__(self):
        raise ValueError("not ready")


class Toned(list):
    """A list with a slot of its own."""

    __slots__ = ("tone",)


def _reporting_alike(kind: type, *methods: str) -> type:
    """A subclass of ``kind`` whose ``methods`` report one thing for every
    instance, whatever it holds: a string, a number, nothing to iterate.
    """
    same: dict[type, object] = {
        str: "same",
        int: 7,
        float: 7.0,
        complex: 7j,
        bytes: b"same",
        Fraction: Fraction(7),
    }

    def reported(name: str):
        if name in ("__iter__", "items", "keys", "values"):
            return lambda self: iter(())
        if name == "__repr__":
            return lambda self: "same"
        if name == "__str__":
            return lambda self: "same"
        return lambda self: same[{"__bytes__": bytes}.get(name, kind)]

    return type(
        f"Reporting{kind.__name__.title()}",
        (kind,),
        {name: reported(name) for name in methods},
    )


#: Two of a builtin kind's subclass that hold different values, whose
#: overridden methods report the same for both.
REPORTING_ALIKE = {
    "str": (_reporting_alike(str, "__str__", "__repr__"), "a", "b"),
    "int": (_reporting_alike(int, "__int__", "__index__", "__repr__"), 1, 2),
    "float": (_reporting_alike(float, "__float__", "__repr__"), 1.0, 2.0),
    "complex": (_reporting_alike(complex, "__complex__", "__repr__"), 1j, 2j),
    "bytes": (_reporting_alike(bytes, "__bytes__", "__iter__"), b"a", b"b"),
    "bytearray": (_reporting_alike(bytearray, "__iter__"), b"a", b"b"),
    "tuple": (_reporting_alike(tuple, "__iter__"), (1,), (2,)),
    "list": (_reporting_alike(list, "__iter__"), [1], [2]),
    "set": (_reporting_alike(set, "__iter__"), {1}, {2}),
    "frozenset": (_reporting_alike(frozenset, "__iter__"), {1}, {2}),
    "dict": (
        _reporting_alike(dict, "__iter__", "items", "keys", "values"),
        {"a": 1},
        {"a": 2},
    ),
    "fraction": (_reporting_alike(Fraction, "__repr__", "__str__"), 1, 2),
}


@pytest.mark.parametrize(
    ("kind", "first", "second"),
    list(REPORTING_ALIKE.values()),
    ids=list(REPORTING_ALIKE),
)
def test_a_builtin_kind_s_subclass_keys_by_what_it_holds_not_what_it_reports(
    kind, first, second
):
    """Its value is read as the builtin kind reads it, not through methods
    the subclass overrides: two that hold different values key apart
    however alike they report themselves.
    """
    assert _value_key(kind(first)) != _value_key(kind(second))
    assert _value_key(kind(first)) == _value_key(kind(first))


def test_a_builtin_kind_s_subclass_keys_as_that_kind_and_all_it_holds_beside():
    """Its value, as the builtin kind it is (whatever it mixes in), its
    class, and its own attributes and slots: two that differ in any of
    them key apart, two alike in all key alike.
    """
    assert unkeyable(Word("a")) is None
    assert _value_key(Word("a")) == _value_key(Word("a"))
    assert _value_key(Word("a")) != _value_key(Word("b"))
    assert _value_key(Word("a")) != _value_key("a")
    toned, other = Word("a"), Word("a")
    toned.tone, other.tone = 1, 2
    assert _value_key(toned) != _value_key(other)
    other.tone = 1
    assert _value_key(toned) == _value_key(other)
    slotted, other_slotted = Toned([1]), Toned([1])
    slotted.tone, other_slotted.tone = 1, 2
    assert _value_key(slotted) != _value_key(other_slotted)
    other_slotted.tone = 1
    assert _value_key(slotted) == _value_key(other_slotted)


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (
            Unloaded(),
            f"raised: builtins.RuntimeError, keying a {__name__}.Unloaded",
        ),
        (
            {"parts": [Unready()]},
            f"raised: builtins.ValueError, keying a {__name__}.Unready",
        ),
    ],
    ids=["a-property-that-raises", "one-inside-other-values"],
)
def test_a_value_whose_key_raises_is_a_marker_naming_it(value, why):
    """The key reads a scene's own objects (their attributes, their parts),
    and what they raise is a marker naming the value and the exception,
    never an error of the summary, the probe or the record's keys.
    """
    key = _value_key(value)
    assert marker_reason(key) == why
    assert _value_key(value) != key
    assert unkeyable(value) is not None


def test_a_ufunc_keys_by_its_module_and_name_alike_in_every_process():
    """numpy's and ``scipy.special``'s ufuncs key by the module that names
    them and their name, alike in another process; one numpy makes of a
    Python function is a marker.
    """
    special = pytest.importorskip(
        "scipy.special", reason="scipy is not installed in this environment"
    )
    assert _value_key(special.erf) != _value_key(special.gamma)
    assert _value_key(np.sin) != _value_key(special.erf)
    made = np.frompyfunc(lambda x: x, 1, 1)
    assert marker_reason(_value_key(made)).startswith(
        "a value of a type the key does not know"
    )
    script = (
        "import numpy as np\n"
        "from arachne.timeline.keys import value_key\n"
        "print(value_key(np.sin)[0].hex())\n"
        "import scipy.special\n"
        "print(value_key(scipy.special.erf)[0].hex())\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    assert found == [_value_key(np.sin).hex(), _value_key(special.erf).hex()]


def test_values_that_differ_anywhere_key_apart():
    """Each rule keys all of its value: two that differ in any part a later
    read could see never key alike.
    """
    import cmath
    import collections
    import math

    import networkx
    import svgelements
    from PIL import Image

    from arachne.mobject.values import FrozenPoints, freeze

    def apart(a, b):
        assert _value_key(a) != _value_key(b), (a, b)

    # An array's dtype and shape, not only its bytes.
    flat = np.arange(6, dtype=float)
    apart(
        freeze(flat.reshape(2, 3), FrozenPoints),
        freeze(flat.reshape(3, 2), FrozenPoints),
    )
    # A container's subclass and its own attributes; a default dict's default.
    apart(collections.defaultdict(list), collections.defaultdict(int))
    word, other = type("Word", (str,), {})("a"), type("Word", (str,), {})("a")
    word.note, other.note = 1, 2
    apart(word, other)
    # A dataclass's attributes beyond its fields; a function's own.
    spec, wider = Spec(1.0), Spec(1.0)
    wider.extra = 2
    apart(spec, wider)

    def f(x):
        return x

    def g(x):
        return x

    g.factor = 2
    apart(f, g)
    # A function's parameters: an updater taking ``dt`` is called with it.
    apart(lambda m, dt: None, lambda m: None)
    apart(lambda m, dt: m, lambda m, t: m)
    apart(lambda m, dt: m, lambda m, *, dt: m)
    apart(lambda *m: m, lambda m: m)
    # A builtin by its module; a ufunc numpy did not name is no numpy ufunc.
    apart(math.sin, cmath.sin)
    made = np.frompyfunc(lambda x: x, 1, 1)
    assert unkeyable(made) == _qualified(type(made))
    assert _value_key(np.sin) == _value_key(np.sin)
    # A picture's palette; a graph's own attributes.
    first, second = Image.new("P", (2, 2)), Image.new("P", (2, 2))
    first.putpalette([0, 0, 0] * 256)
    second.putpalette([255, 0, 0] * 256)
    apart(first, second)
    one, two = networkx.Graph(name="one"), networkx.Graph(name="two")
    apart(one, two)
    # A multigraph's edges by their keys too.
    keyed, other_keyed = networkx.MultiGraph(), networkx.MultiGraph()
    keyed.add_edge(0, 1, key="a")
    other_keyed.add_edge(0, 1, key="b")
    apart(keyed, other_keyed)
    # A picture's transparency, which its conversion to colours reads.
    clear = Image.new("P", (2, 2))
    clear.info["transparency"] = 0
    apart(Image.new("P", (2, 2)), clear)
    # A parsed path's own segments and its transform, each read apart.
    apart(
        svgelements.Path("M 0 0 L 2 2"),
        svgelements.Path("M 0 0 L 1 1", transform="scale(2)"),
    )
    # The slots of an object that has attributes too.
    held, other_held = Slotted(), Slotted()
    held.slot, other_held.slot = 1, 2
    apart(held, other_held)
    # An engine function's cache by its options too.
    from arachne.utils.simple_functions import choose

    apart(
        functools.lru_cache(maxsize=1)(choose.__wrapped__),
        functools.lru_cache(maxsize=2)(choose.__wrapped__),
    )
    # A dict's order: its items are read in it (a colour map applied in order).
    apart({"x": 1, "xy": 2}, {"xy": 2, "x": 1})


def test_an_object_keys_by_all_of_its_state_or_is_a_marker():
    # A callable object wrapping a function keys by its own state too.
    assert _value_key(Scaled(abs, 1)) != _value_key(Scaled(abs, 5))
    assert _value_key(Scaled(abs, 1)) == _value_key(Scaled(abs, 1))
    # A cache of the engine's keys as the function it caches.
    from arachne.utils.simple_functions import choose

    assert _value_key(choose) == _value_key(choose)
    # One of a function of anyone else's is a marker: its entries can't be
    # read, and they are what a call returns.
    for value in (functools.lru_cache(abs), functools.cache(Scaled(abs, 1))):
        assert unkeyable(value) is not None
        assert marker_reason(_value_key(value)).startswith(
            "a cache whose entries the key cannot read"
        ), value
    # State a builtin base holds is not in the attributes: the marker.
    for value in (io.StringIO("a"), Buffered(b"a")):
        assert unkeyable(value) == _qualified(type(value)), value
    assert _value_key(Buffered(b"a")) != _value_key(Buffered(b"a"))
    # A builtin container's subclass keys its own attributes beside its parts.
    first, second = Tagged([1]), Tagged([1])
    first.tag, second.tag = "a", "b"
    assert _value_key(first) != _value_key(second)
    second.tag = "a"
    assert _value_key(first) == _value_key(second)


#: Every cache the engine makes of a function, by its function's qualified
#: name, whose result is a function of its arguments alone: each keys as
#: the function it caches. The engine's others read more than their
#: arguments (the fonts installed, files) and are keyed as anyone's cache.
PURE_ENGINE_CACHES = {
    "arachne.camera.projection._cairo",
    "arachne.camera.projection._path_type",
    "arachne.lint.parts._text_types",
    "arachne.lint.parts._backdrop_type",
    "arachne.lint.parts._unit_types",
    "arachne.lint.parts._text_like_types",
    "arachne.mobject.types.vectorized_mobject._bezier_parameters",
    "arachne.timeline.camera_code._nothing_code",
    "arachne.timeline.dispatch._all_engine_classes",
    "arachne.timeline.dispatch.draw_methods",
    "arachne.timeline.dispatch.camera_draw_methods",
    "arachne.timeline.dispatch._mobject_definitions",
    "arachne.timeline.dispatch._reads",
    "arachne.timeline.dispatch.draw_protocols",
    "arachne.timeline.dispatch._protocols",
    "arachne.timeline.library._table",
    "arachne.timeline.setters._setter_functions",
    "arachne.timeline.setters.engine_classes",
    "arachne.timeline.setters._functions_named",
    "arachne.timeline.setters._constructors",
    "arachne.timeline.ticking._slot_loaded",
    "arachne.utils.simple_functions.choose",
}


def _engine_caches() -> dict[str, object]:
    """Every function the engine's source decorates with a ``functools``
    cache, by its qualified name, with the cache.
    """
    import ast
    import importlib

    import arachne

    root = Path(arachne.__file__).parent
    found: dict[str, object] = {}
    for path in sorted(root.rglob("*.py")):
        module = ".".join(("arachne", *path.relative_to(root).with_suffix("").parts))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scopes = [(node, "") for node in tree.body]
        while scopes:
            node, outer = scopes.pop()
            if isinstance(node, ast.ClassDef):
                scopes.extend((child, f"{outer}{node.name}.") for child in node.body)
                continue
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {
                ast.unparse(d.func if isinstance(d, ast.Call) else d)
                for d in node.decorator_list
            }
            if names & {"cache", "lru_cache", "functools.cache", "functools.lru_cache"}:
                value: object = importlib.import_module(module)
                for part in f"{outer}{node.name}".split("."):
                    value = getattr(value, part)
                found[f"{module}.{outer}{node.name}"] = value
    return found


def test_every_engine_cache_is_known_pure_or_keyed_as_anyone_s():
    """A cache the engine adds is either a function of its arguments
    (listed here, and keyed as the function it caches) or one that reads
    more (the key's own list, and a marker); a cache of a function a scene
    defines, under a module whose name begins as the engine's does, is a
    marker too.
    """
    from arachne.timeline.keys import READING_ENGINE_CACHES

    caches = _engine_caches()
    assert set(caches) == PURE_ENGINE_CACHES | READING_ENGINE_CACHES
    assert not PURE_ENGINE_CACHES & READING_ENGINE_CACHES
    for name, cache in caches.items():
        if name in PURE_ENGINE_CACHES:
            assert unkeyable(cache) is None, name
            assert _value_key(cache) == _value_key(cache), name
        else:
            why = marker_reason(_value_key(cache))
            assert why == f"a cache whose entries the key cannot read ({name})"

    def scene_function(x):
        return x

    scene_function.__module__ = "arachne_session_1_scene"
    assert unkeyable(functools.cache(scene_function)) is not None


def test_a_dict_keys_by_its_order_alike_in_every_process():
    """A dict keys its items in the order it holds them, which is the
    program's and no hash seed's: one built alike keys alike in processes
    under other string hash seeds.
    """
    value = {"zeta": 1, "alpha": {"y": 2.5, "b": [1, "x"]}, "mid": None}
    script = (
        "from arachne.timeline.keys import value_key\n"
        "value = {'zeta': 1, 'alpha': {'y': 2.5, 'b': [1, 'x']}, 'mid': None}\n"
        "print(value_key(value)[0].hex())\n"
    )
    mine = _value_key(value).hex()
    for seed in ("0", "1", "12345"):
        theirs = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        assert theirs == mine, seed


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


def _scaled_partial(factor):
    class Scaled(functools.partial):
        """A partial whose own call reads an attribute it holds."""

    made = Scaled(max)
    made.factor = factor
    return made


def test_a_partial_subclass_keys_the_attributes_it_holds():
    """What its own ``__call__`` could read, beside the function and the
    arguments the partial holds.
    """
    assert _value_key(_scaled_partial(2)) != _value_key(_scaled_partial(3))
    assert _value_key(_scaled_partial(2)) == _value_key(_scaled_partial(2))


def test_an_array_subclass_keys_the_state_it_holds_beside_its_data():
    """A masked array's mask is read by every operation on it."""
    data = np.array([1.0, 2.0, 3.0])
    one = np.ma.masked_array(data, mask=[False, True, False])
    other = np.ma.masked_array(data, mask=[False, False, True])
    assert _value_key(one) != _value_key(other)
    assert _value_key(one) == _value_key(
        np.ma.masked_array(data.copy(), mask=[False, True, False])
    )


def test_a_structured_array_is_keyed_by_the_fields_it_reads_its_bytes_as():
    """The same bytes under another field's type or name are another value
    (``dtype.str`` of both is the void of their size).
    """
    floats = np.array([(1.0,), (2.0,)], dtype=[("a", "<f8")])
    integers = floats.view([("a", "<i8")])
    renamed = floats.view([("b", "<f8")])
    for first, second in ((floats, integers), (floats, renamed)):
        assert _value_key(first) != _value_key(second)
        assert _value_key([first]) != _value_key([second])
    assert _value_key(floats) == _value_key(floats.copy())
    assert _value_key(floats[0]) != _value_key(integers[0])


def test_an_enum_member_is_a_summary_leaf_by_its_value_too():
    """Two enums of one name whose members' values differ (a definition that
    reads its input) are two leaves; one enum's member is one leaf.
    """
    from arachne.timeline.execution import _field_leaf

    low = Enum("Level", {"ONE": 1})
    high = Enum("Level", {"ONE": 2})
    assert _field_leaf(low.ONE) == _field_leaf(low.ONE)
    assert _field_leaf(low.ONE) != _field_leaf(high.ONE)


def test_a_palette_image_is_keyed_by_its_palettes_alpha_too():
    """Palettes of one set of colours that differ only in alpha convert to
    other pixels: two images, two keys.
    """
    from PIL import Image

    def palette_image(alpha: int) -> Image.Image:
        image = Image.new("P", (2, 2), 0)
        image.putpalette([255, 0, 0, alpha], rawmode="RGBA")
        return image

    assert _value_key(palette_image(0)) != _value_key(palette_image(255))
    assert _value_key(palette_image(255)) == _value_key(palette_image(255))
