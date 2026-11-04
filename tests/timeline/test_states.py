"""The states a record keeps for the lint's temporal rules (``Record.states``):
the scene as each play ended, and as the script removed from it.
"""

from __future__ import annotations

from pathlib import Path

from .conftest import render
from .state_scenes import FadesAndRemoves

STATES = Path(__file__).parent / "state_scenes.py"


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
    assert states.since == 0
    assert set(states.exits) == {0, 1, 2}
    fade = record.plays[0]
    left = states.exits[0]
    assert left.kind == "exit"
    assert left.instant == fade.end
    # Its last evaluated instant still holds the word; the fade's cleanup
    # had taken it out by the play's end.
    assert "Text" in _types(record, fade.samples[-1].roots)
    assert "Text" not in _types(record, left.roots)
    # No frame draws it.
    assert all(left is not capture for play in record.plays for capture in play.samples)


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


def test_states_are_kept_from_the_arming_on(session_of):
    record = session_of("ArmsStatesLate", STATES, probe=False).record
    states = record.states
    assert states.since == 1
    assert set(states.exits) == {1}
    [removal] = states.removals
    assert removal.plays == 1


def test_a_scene_that_never_arms_keeps_none(session_of):
    record = session_of("FadesAndRemovesUnarmed", STATES, probe=False).record
    assert record.states is None


def test_keeping_states_changes_no_frame(session_of):
    armed = session_of("FadesAndRemoves", STATES, probe=False).record
    unarmed = session_of("FadesAndRemovesUnarmed", STATES, probe=False).record
    assert [p.sample_keys() for p in armed.plays] == [
        p.sample_keys() for p in unarmed.plays
    ]
    assert [p.draw_refusal for p in armed.plays] == [
        p.draw_refusal for p in unarmed.plays
    ]


def test_a_render_keeps_no_states_even_armed():
    scene, _ = render(FadesAndRemoves, 6)
    assert scene.timeline.record.states is None


def test_a_play_keeps_the_top_level_it_was_called_on(session_of):
    record = session_of("FadesAGroupMadeForIt", STATES, probe=False).record
    fade = record.plays[1]
    # (And the empty mobject the wait before it left, as in CE.)
    assert _types(record, fade.called) == ["Line", "Mobject", "Text"]
    # As it began, the group made for it held both.
    assert _types(record, fade.samples[0].roots) == ["Mobject", "VGroup"]
