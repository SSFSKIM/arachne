"""Lint instants: frames of the record's grid, so that every one is a
capture the record kept and a frame the render shows.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from arachne.lint.instants import lint_instants
from arachne.timeline.clock import frames_in
from arachne.timeline.routes import _resolve


def _expected(record, fps: int) -> dict[Fraction, set[str]]:
    """The lint instants by their definition: the first grid frame at or
    after each k/fps, each play's first and last grid frame, each frozen
    play's frame, and the final capture.
    """
    grid = record.grid
    found: dict[Fraction, set[str]] = {}
    for k in range(math.ceil(record.duration * fps)):
        frame = Fraction(math.ceil(Fraction(k, fps) * grid)) / grid
        if frame < record.duration:
            found.setdefault(frame, set()).add("sampled")
    for play in record.plays:
        frames = frames_in(play.start, play.end, grid)
        if not frames:
            continue
        found.setdefault(frames[0], set()).add("first")
        found.setdefault(frames[-1], set()).add("last")
        if play.held:
            found[frames[0]].add("frozen")
    found.setdefault(record.duration, set()).add("final")
    return found


@pytest.mark.parametrize("rate", [15, 30, 60])
def test_lint_instants_are_kept_frames(session_of, rate):
    session = session_of("Instants", frame_rate=rate)
    record = session.record
    assert [p.held for p in record.plays] == [False, True, False, True, False, False]
    assert record.plays[4].route == "ticked"
    instants = lint_instants(record, 4)
    assert {i.t: set(i.why) for i in instants} == _expected(record, 4)
    assert [i.t for i in instants] == sorted(i.t for i in instants)
    for instant in instants:
        if "final" not in instant.why:
            assert (instant.t * rate).denominator == 1
        resolved = _resolve(record, instant.t)
        assert resolved.kind == "recorded"
        assert resolved.prepare is None
        assert resolved.capture.instant <= instant.t
        held = record.play_at(instant.t)
        assert instant.play == (held.index if held is not None else None)


def test_a_play_holding_no_frame_gives_no_instant(session_of):
    """At 15 fps the 0.01 s play and the 0.005 s wait hold no frame: no
    frame of the render shows them, so the lint reads neither.
    """
    session = session_of("Instants", frame_rate=15)
    record = session.record
    short = {2, 3}
    for index in short:
        play = record.plays[index]
        assert frames_in(play.start, play.end, Fraction(15)) == []
    assert not {i.play for i in lint_instants(record, 4)} & short


def test_sampled_instants_are_the_first_frame_at_or_after(session_of):
    session = session_of("Instants", frame_rate=15)
    by_t = {i.t: i.why for i in lint_instants(session.record, 4)}
    # 1/4 s lies between frames 3 and 4 of the 15 fps grid.
    assert "sampled" in by_t[Fraction(4, 15)]
    assert Fraction(1, 4) not in by_t


def test_instants_asked_for_are_the_frames_on_screen_then(session_of):
    session = session_of("Instants", frame_rate=15)
    record = session.record
    instants = lint_instants(record, 4, at=[0.1, 0.1, 99])
    assert [(i.t, set(i.why)) for i in instants] == [
        (Fraction(1, 15), {"asked"}),
        (record.duration, {"asked", "final"}),
    ]


def test_held_and_sampled_names_the_default(session_of):
    record = session_of("Instants", frame_rate=15).record
    assert lint_instants(record, 4, at="held+sampled") == lint_instants(record, 4)
    with pytest.raises(ValueError, match="held\\+sampled"):
        lint_instants(record, 4, at="sometimes")
