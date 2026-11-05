"""The states a record keeps for the lint's temporal rules (``Record.states``):
the scene as the script called each play, as each play ended, and as the
script removed from it; none of them drawn, and none in the history of
drawn captures.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from arachne.timeline.fallback import answer_instants

from .conftest import render
from .state_scenes import FadesAndRemoves

STATES = Path(__file__).parent / "state_scenes.py"
UNDRAWN = ("call", "exit", "removal")


def _line(tag: str) -> int:
    for number, line in enumerate(STATES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def _types(record, mobs):
    return sorted(type(m).__name__ for m in mobs)


def test_an_exit_is_the_scene_after_the_play_s_cleanup(session_of):
    record = session_of("FadesAndRemoves", STATES, probe=False).record
    states = record.states
    assert states is not None
    assert (states.since, states.armed) == (0, 0)
    assert set(states.exits) == set(states.calls) == {0, 1, 2}
    fade = record.plays[0]
    left = states.exits[0]
    assert left.kind == "exit"
    assert left.instant == fade.end
    # Its last evaluated instant still holds the word; the fade's cleanup
    # had taken it out by the play's end.
    assert "Text" in _types(record, fade.samples[-1].roots)
    assert "Text" not in _types(record, left.roots)


def test_a_call_is_the_scene_before_the_play_adds_anything(session_of):
    record = session_of("FadesAGroupMadeForIt", STATES, probe=False).record
    fade = record.plays[1]
    called = record.states.calls[1]
    assert (called.kind, called.instant) == ("call", fade.start)
    # (And the empty mobject the wait before it left, as in CE.)
    assert _types(record, called.roots) == ["Line", "Mobject", "Text"]
    # As it began, the group made for it held both.
    assert _types(record, fade.samples[0].roots) == ["Mobject", "VGroup"]


def test_a_removal_keeps_the_scene_before_it_and_its_line(session_of):
    record = session_of("FadesAndRemoves", STATES, probe=False).record
    remove, clear = record.states.removals
    assert (remove.kind, remove.plays, remove.t) == ("remove", 1, record.plays[0].end)
    assert _types(record, remove.operands) == ["Square"]
    assert _types(record, remove.capture.roots) == ["Dot", "Square"]
    assert _types(record, remove.after) == ["Dot"]
    assert remove.source_line.line == _line("remove")
    assert remove.capture.kind == "removal"
    assert (clear.kind, clear.plays, clear.after) == ("clear", 2, ())
    assert _types(record, clear.operands) == ["Dot"]
    assert clear.source_line.line == _line("clear")


def test_a_removal_keeps_its_operands_as_they_were(session_of):
    # The group is no part of the scene, and holds nothing by the end: the
    # removal's state holds what it held as the script removed it.
    record = session_of("RemovesThroughAGroup", STATES, probe=False).record
    [removal] = record.states.removals
    [group] = removal.operands
    assert group.submobjects == []
    held = record.node_state(group, removal.capture)
    assert held is not None
    assert _types(record, held.submobjects) == ["Text"]


def test_states_are_kept_from_the_arming_on(session_of):
    record = session_of("ArmsStatesLate", STATES, probe=False).record
    states = record.states
    assert (states.since, states.armed) == (1, Fraction(1, 2))
    assert set(states.exits) == set(states.calls) == {1}
    [removal] = states.removals
    assert removal.plays == 1
    assert not states.covers(record.plays, Fraction(0))
    assert states.covers(record.plays, Fraction(1, 2))


def test_a_scene_that_never_arms_keeps_none(session_of):
    record = session_of("FadesAndRemovesUnarmed", STATES, probe=False).record
    assert record.states is None


def test_no_state_is_in_the_drawn_history(session_of):
    record = session_of("FadesAndRemoves", STATES, probe=False).record
    kept = record.states
    undrawn = [*kept.calls.values(), *kept.exits.values()]
    undrawn += [removal.capture for removal in kept.removals]
    assert {c.kind for c in undrawn} == set(UNDRAWN)
    assert all(c.kind not in UNDRAWN for c in record._captures)
    drawn = {id(c) for c in record._captures}
    assert not any(id(c) in drawn for c in undrawn)
    # Each sits between the drawn captures either side of it.
    seqs = [c.seq for c in record._captures]
    assert seqs == list(range(len(seqs)))
    assert all(int(c.seq) != c.seq for c in undrawn)


def test_keeping_states_changes_no_frame(session_of):
    armed = session_of("FadesAndRemoves", STATES, probe=False)
    unarmed = session_of("FadesAndRemovesUnarmed", STATES, probe=False)
    assert [p.sample_keys() for p in armed.record.plays] == [
        p.sample_keys() for p in unarmed.record.plays
    ]
    assert [p.draw_refusal for p in armed.record.plays] == [
        p.draw_refusal for p in unarmed.record.plays
    ]
    for t in (0, 0.5, 1, 1.5, 2, 2.4):
        assert np.array_equal(np.asarray(armed.frame(t)), np.asarray(unarmed.frame(t)))


def test_keeping_states_changes_no_camera_refusal_or_picture(session_of):
    # Unarmed, the display the camera left refuses drawing from the record;
    # a state kept after it must not hide that (it did: 208 pixels apart
    # at 0.6 s, and no refusal).
    unarmed = session_of("DisplayLeftBetweenPlays", STATES, probe=False)
    armed = session_of("DisplayLeftBetweenPlaysArmed", STATES, probe=False)
    assert unarmed.record.refusal is not None
    assert "display" in unarmed.record.refusal
    assert armed.record.refusal == unarmed.record.refusal
    assert [p.draw_refusal for p in armed.record.plays] == [
        p.draw_refusal for p in unarmed.record.plays
    ]
    at = [Fraction(3, 5)]
    (plain,) = answer_instants(unarmed, at, None, frames=True)
    (kept,) = answer_instants(armed, at, None, frames=True)
    assert np.array_equal(np.asarray(plain.frame), np.asarray(kept.frame))


def test_a_render_keeps_no_states_even_armed():
    scene, _ = render(FadesAndRemoves, 6)
    record = scene.timeline.record
    assert record.states is None
    assert all(c.kind not in UNDRAWN for c in record._captures)
