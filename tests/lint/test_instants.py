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


def _linted(session) -> list:
    from arachne.lint import lint

    seen = []

    def keep(context):
        seen.extend(context.parts)
        return ()

    lint(session, rules=(keep,))
    return seen


def test_a_session_without_a_grid_is_linted_at_its_render_s_frames(session_of):
    """Recorded with every instant it evaluated kept and no grid, a session
    is linted at the frames its render shows, and sees there what a session
    on the render's grid sees: an instant the record holds no capture of is
    read from one forward run on that grid, never evaluated between
    captures.
    """
    session = session_of("Instants", grid=None, frame_rate=15)
    gridded = session_of("Instants", frame_rate=15)
    assert session.record.grid is None
    instants = lint_instants(session.record, 4)
    assert instants == lint_instants(gridded.record, 4)
    for instant in instants:
        if "final" not in instant.why:
            assert (instant.t * 15).denominator == 1
    seen, expected = _linted(session), _linted(gridded)
    assert [p.t for p in seen] == [i.t for i in instants]
    assert {p.answer for p in seen} == {"recorded", "forward"}
    assert {p.answer for p in expected} == {"recorded"}
    assert len(session.forward_runs) == 1
    for found, wanted in zip(seen, expected, strict=True):
        assert [p.id for p in found.parts] == [p.id for p in wanted.parts]
        for part, other in zip(found.parts, wanted.parts, strict=True):
            assert part.screen_box == pytest.approx(other.screen_box, abs=1e-9)


@pytest.mark.parametrize("rate", [15, 30, 60])
def test_a_refused_record_is_read_at_its_render_s_frames(session_of, rate):
    session = session_of("RefusedInstants", frame_rate=rate)
    assert session.record.refusal is not None
    seen = _linted(session)
    record = session.record
    assert len(session.forward_runs) == 1
    assert {p.t for p in seen} == set(_expected(record, 4))
    assert all(p.answer == "forward" for p in seen)
    assert all(p.view.pixel_width == 96 for p in seen)
