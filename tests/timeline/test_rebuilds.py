"""Which mobjects ``become`` rebuilt, and when (:mod:`arachne.timeline.rebuilds`),
and the counts each play records of it.
"""

from __future__ import annotations

from pathlib import Path

from arachne import Dot, VGroup
from arachne.timeline import rebuilds

from .test_exits import _key_of

EXITS = Path(__file__).parent / "exit_scenes.py"


def test_a_call_is_told_by_its_number():
    group, other = VGroup(Dot()), VGroup(Dot())
    start = rebuilds.count()
    group.become(VGroup(Dot(), Dot()))
    after_one = rebuilds.count()
    assert after_one == start + 1
    assert rebuilds.rebuilt_between(group, start, after_one)
    assert not rebuilds.rebuilt_between(other, start, after_one)
    # The span is open at its start: a call numbered at it came before it.
    assert not rebuilds.rebuilt_between(group, after_one, after_one + 5)
    other.become(Dot())
    assert rebuilds.rebuilt_between(other, after_one, rebuilds.count())
    assert not rebuilds.rebuilt_between(group, after_one, rebuilds.count())


def _first(record, type_name, where=lambda mob: True):
    return next(
        m for m in record.mobjects() if type(m).__name__ == type_name and where(m)
    )


def test_each_play_records_the_count_at_its_call_and_its_start(session_of):
    record = session_of("Rebuilds", EXITS, probe=False).record
    first, second, third = record.plays
    pair = _first(record, "VGroup")
    square = _first(record, "Square")
    dot = _first(record, "Dot", lambda mob: mob.get_center()[1] > 0.5)
    # Rebuilt by the script between the first play's call and the second's
    # start, and never after.
    assert rebuilds.rebuilt_between(
        pair, first.rebuilds_called, second.rebuilds_entered
    )
    assert not rebuilds.rebuilt_between(
        pair, second.rebuilds_called, third.rebuilds_entered
    )
    assert not rebuilds.rebuilt_between(
        square, first.rebuilds_called, third.rebuilds_entered
    )
    # An updater's call as a play begins falls between its call and its start.
    assert third.rebuilds_entered > third.rebuilds_called
    assert rebuilds.rebuilt_between(dot, third.rebuilds_called, third.rebuilds_entered)
    assert not rebuilds.rebuilt_between(
        dot, first.rebuilds_called, second.rebuilds_entered
    )


def test_a_child_without_points_is_read_again(session_of):
    exits = session_of("GainsAnEmptyChild", EXITS, probe=False).record.exits
    pair = _key_of(exits[0], "VGroup")
    was, now = exits[0].bodies[pair], exits[1].bodies[pair]
    assert (was.members, was.children) == (2, 2)
    assert (now.members, now.children) == (2, 3)
