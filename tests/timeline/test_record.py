"""The forward record: plays, markers, stable ids, names and ``state_at``."""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne.timeline import SourceLine

from .conftest import SCENES, line_of


def _line(tag: str) -> SourceLine:
    return SourceLine(SCENES.name, line_of(tag))


def test_plays_have_exact_spans_kinds_lines_and_held_flags(session_of):
    record = session_of("Timeline").record
    spans = [(play.kind, play.start, play.end, play.held) for play in record.plays]
    assert spans == [
        ("play", Fraction(0), Fraction(1, 2), False),
        ("wait", Fraction(1, 2), Fraction(3, 4), True),
        ("wait", Fraction(3, 4), Fraction(1), False),
        # wait_until ends at the tick where its condition holds: 1.
        ("wait", Fraction(1), Fraction(1), False),
        ("play", Fraction(1), Fraction(1) + Fraction(1, 7), False),
    ]
    lines = [play.source_line for play in record.plays]
    assert lines[:4] == [
        _line("create"),
        _line("held"),
        _line("moving-wait"),
        _line("wait-until"),
    ]
    # The Create and the frozen wait are trusted; once the dot's dt updater is
    # in the scene, every play ticks.
    assert [play.route for play in record.plays] == [
        "trusted",
        "trusted",
        "ticked",
        "ticked",
        "ticked",
    ]
    assert record.duration == Fraction(8, 7)
    assert record.identity.seed == 0
    assert record.identity.source_digest is not None


def test_a_held_wait_is_one_sample(session_of):
    record = session_of("Timeline").record
    held = record.plays[1]
    assert len(held.samples) == 1
    assert held.capture_at(Fraction(7, 10)) is held.samples[0]


def test_markers_carry_their_instants_and_lines(session_of):
    record = session_of("Timeline").record
    kinds = [(marker.kind, marker.t) for marker in record.markers]
    assert kinds == [
        ("section", Fraction(0)),
        ("mark", Fraction(1, 2)),
        ("subcaption", Fraction(3, 4)),
        ("wait_until", Fraction(1)),
        ("subcaption", Fraction(1)),
    ]
    section, mark, caption, until, fade = record.markers
    assert section.data == {"name": "intro", "type": "default.normal", "skip": False}
    assert mark.data == {"label": "beat", "n": 1}
    assert mark.source_line == _line("mark")
    assert caption.data == {"content": "hello", "duration": 1}
    assert until.data == {"requested": Fraction(60), "actual": Fraction(0)}
    assert until.source_line == _line("wait-until")
    assert fade.data["content"] == "fade"
    # Each play carries the markers placed in its span.
    assert [marker.kind for marker in record.plays[1].markers] == ["mark"]


def test_stable_ids_are_creating_lines_and_agree_across_executions(session_of):
    first = session_of("Timeline").state_at(Fraction(1, 4))
    second = session_of("Timeline").state_at(Fraction(1, 4))
    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
    square = first.named("square")
    assert square.id == f"{SCENES.name}:{line_of('square')}#0"
    assert square.source_line == _line("square")
    assert square.type == "Square"


def test_a_copy_has_an_id_of_its_own(session_of):
    state = session_of("Copies").state_at(Fraction(1, 20))
    twin = state.named("twin")
    assert twin.id == f"{SCENES.name}:{line_of('copy')}#0"
    assert state.named("square").id != twin.id


def test_names_reach_into_groups(session_of):
    state = session_of("Timeline").state_at(Fraction(1) + Fraction(1, 14))
    group = state.named("group")
    assert [state.node(child).name for child in group.children] == [
        "group[0]",
        "group[1]",
    ]
    # The ordinal counts the mobjects made at that line, in order.
    assert group.id.startswith(f"{SCENES.name}:{line_of('group')}#")


def test_state_at_reads_the_last_tick_at_or_before(session_of):
    session = session_of("Timeline")
    # The dot shifts right one unit a second from 3/4 on, on the tick.
    dot_at = lambda t: session.state_at(t).named("dot").box  # noqa: E731
    assert dot_at(Fraction(3, 4) + Fraction(1, 120)) == dot_at(Fraction(3, 4))
    x_min = dot_at(Fraction(3, 4) + Fraction(6, 60))[0]
    assert x_min == pytest.approx(dot_at(Fraction(3, 4))[0] + 0.1)
    state = session.state_at(Fraction(9, 10))
    assert state.play == 2
    assert state.route == "ticked"
    assert state.answer == "recorded"


def test_state_at_leaves_out_trackers_and_transparent_padding(session_of):
    session = session_of("Padding")
    record = session.record
    play = record.plays[0]
    one = next(mob for mob in play.moving if type(mob).__name__ == "VGroup")
    held = record.node_state(one, play.samples[0])
    assert len(held.submobjects) == 3  # one square and two padding copies

    start = session.state_at(0)
    group = start.named("one")
    assert len(group.children) == 1
    assert all(node.type != "ValueTracker" for node in start.nodes)
    # Midway the padding is drawn, so it is reported.
    assert len(session.state_at(Fraction(1, 2)).named("one").children) == 3


def test_record_samples_follow_a_frame_grid(session_of):
    record = session_of("Timeline").record
    samples = record.samples(4)
    assert [sample.t for sample in samples] == [Fraction(k, 4) for k in range(5)]
    assert [sample.play for sample in samples] == [0, 0, 1, 2, 4]


def test_a_render_keeps_only_what_its_grid_reads():
    from .conftest import render
    from .scenes import Timeline

    scene, frames = render(Timeline, 4)
    record = scene.record
    assert record.grid == 4
    # A trusted play is evaluated at its frames, and analytic between them.
    first = record.plays[0]
    assert first.route == "trusted"
    assert [capture.instant for capture in first.samples] == [
        Fraction(0),
        Fraction(1, 4),
    ]
    assert record.state_at(Fraction(1, 8)).answer == "analytic"
    # A ticked play keeps only the ticks a frame reads.
    ticked = record.plays[2]
    assert ticked.route == "ticked"
    assert [capture.instant for capture in ticked.samples] == [Fraction(3, 4)]
    with pytest.raises(LookupError, match="kept only"):
        ticked.capture_at(Fraction(4, 5))
    assert len(frames) == 5


def test_a_skipped_section_runs_and_is_recorded():
    """A skipped section's plays run as any other, so what follows them does
    not depend on whether they were drawn; the record says they were skipped.
    """
    from .conftest import render
    from .scenes import Sections

    scene, frames = render(Sections, 6)
    record = scene.record
    assert [play.skipped for play in record.plays] == [False, True, False]
    sections = [m.data for m in record.markers if m.kind == "section"]
    assert [(s["name"], s["skip"]) for s in sections] == [
        ("skipped", True),
        ("shown", False),
    ]
    # Drawn: the first play's three frames and the wait's three.
    assert len(frames) == 6
    # The dot moved through the skipped play and the wait, on the tick.
    dot = next(mob for mob in scene.mobjects if type(mob).__name__ == "Dot")
    assert dot.get_center()[0] == pytest.approx(1.0)
