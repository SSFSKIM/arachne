"""The rational clock and the one global frame grid (deliberate break 3)."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from arachne import Scene, Square, Succession, Wait
from arachne.timeline.clock import TICK_DT, exact, frames_in, play_end, ticks_in

from .conftest import render
from .scenes import Composed, Cue, Grid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.1, Fraction(1, 10)),
        (1 / 3, Fraction(1, 3)),
        (1 / 7, Fraction(1, 7)),
        (2.5, Fraction(5, 2)),
        (Fraction(2, 9), Fraction(2, 9)),
        (3, Fraction(3)),
    ],
)
def test_exact_is_the_simplest_fraction_that_rounds_to_the_float(value, expected):
    assert exact(value) == expected
    assert float(exact(value)) == float(value)


def test_exact_keeps_floats_that_no_simpler_fraction_reaches():
    value = 0.1 + 0.2  # 0.30000000000000004, which 3/10 does not round to
    assert float(exact(value)) == value
    assert exact(value) != Fraction(3, 10)


def test_ticks_and_frames_are_one_global_grid():
    assert list(ticks_in(Fraction(0), Fraction(1, 20))) == [
        Fraction(k, 60) for k in (1, 2, 3)
    ]
    # The ticks after the start, up to and with the end.
    assert list(ticks_in(Fraction(1, 7), Fraction(1, 6))) == [
        Fraction(9, 60),
        Fraction(10, 60),
    ]
    assert list(ticks_in(Fraction(1, 6), Fraction(1, 6) + Fraction(1, 100))) == []
    assert float(Fraction(1, 60)) == TICK_DT
    # Frames are k/fps in [start, end): the end is the next play's.
    assert frames_in(Fraction(1, 10), Fraction(3, 10), Fraction(15)) == [
        Fraction(2, 15),
        Fraction(3, 15),
        Fraction(4, 15),
    ]


def test_plays_share_one_frame_grid():
    """Three plays of 0.1 s at 15 fps show in the five frames of [0, 0.3):
    CE gave each play its own grid, two frames each, six in all.
    """
    scene, frames = render(Grid, 15)
    plays = scene.record.plays
    assert [(play.start, play.end) for play in plays] == [
        (Fraction(i, 10), Fraction(i + 1, 10)) for i in range(3)
    ]
    assert len(frames) == 5
    assert scene.time == Fraction(3, 10)


def test_no_one_frame_minimum():
    """A play shorter than a frame keeps its length."""
    scene = Scene()
    scene.play(Wait(run_time=0.01, frozen_frame=False))
    scene.add(Square())
    scene.wait(1e-3)
    assert scene.time == Fraction(1, 100) + Fraction(1, 1000)


def test_a_play_ends_at_the_simplest_fraction_near_its_float_end():
    """The end-window rule: plain durations add up exactly, and a wait sized
    from the float clock ends at its target.
    """
    tenth = play_end(Fraction(0), 0.1)
    assert tenth == Fraction(1, 10)
    assert play_end(tenth, 0.2) == Fraction(3, 10)
    end = Fraction(0)
    for _ in range(3):
        end = play_end(end, 0.1)
    assert end == Fraction(3, 10)
    two_thirds = Fraction(2, 3)
    assert play_end(two_thirds, 2.5 - float(two_thirds)) == Fraction(5, 2)
    # Exact durations are used as given, and nothing lasts no time but 0.
    assert play_end(Fraction(1, 7), Fraction(1, 3)) == Fraction(10, 21)
    assert play_end(Fraction(1, 7), 2) == Fraction(15, 7)
    assert play_end(Fraction(1, 7), 0.0) == Fraction(1, 7)


def test_a_positive_duration_never_ends_at_or_before_its_start():
    ugly = Fraction(10**9 + 7, 3 * 10**9)
    for d in (1e-20, 1e-17, 5e-17, 1e-16, 1e-12, 0.1, 1 / 3):
        assert play_end(ugly, d) > ugly
    rng = random.Random(7)
    for _ in range(2000):
        start = Fraction(rng.randrange(1, 10**7), rng.choice([7, 97, 10**6 + 3]))
        d = rng.choice([1e-18, 1e-15, 1e-9, rng.random(), rng.random() * 100])
        assert play_end(start, d) > start


@pytest.mark.parametrize(("fps", "count"), [(15, 38), (30, 75), (60, 150)])
def test_a_cue_lands_on_its_target(fps, count):
    """``wait(2.5 - renderer.time)`` after 2/3 ends at 5/2 exactly, so it
    holds exactly the frames of [0, 5/2) at a rate whose grid holds 5/2:
    with the duration alone made exact it ended 4e-16 later and drew one
    frame more at 30 and 60 fps.
    """
    scene, frames = render(Cue, fps)
    assert [play.end for play in scene.record.plays] == [
        Fraction(1, 3),
        Fraction(2, 3),
        Fraction(5, 2),
    ]
    assert len(frames) == count


def test_scene_time_is_exact_and_renderer_time_its_float():
    scene, _ = render(Cue, 15)
    assert isinstance(scene.time, Fraction)
    assert scene.time == Fraction(5, 2)
    assert isinstance(scene.renderer.time, float)
    assert scene.renderer.time == 2.5


@pytest.mark.parametrize("fps", [10, 30])
def test_a_composition_lasts_its_parts_exact_sum(fps):
    """``Succession(Wait(0.1), Wait(0.2))`` lasts 3/10, not the float sum
    0.30000000000000004, so it holds 3 frames at 10 fps (not 4); a run time
    given to a composition is the authored float, read by the end-window rule.
    """
    scene, frames = render(Composed, fps)
    spans = [play.end - play.start for play in scene.record.plays]
    assert spans == [Fraction(3, 10), Fraction(1, 2), Fraction(1, 2)]
    assert len(frames) == len(frames_in(Fraction(0), Fraction(13, 10), Fraction(fps)))
    assert len(frames) == 13 * fps // 10


def test_a_run_time_given_to_a_composition_is_used_as_given():
    """A composition lasts its parts' exact sum only when no run time was
    given to it: an exact run time equal to the parts' float sum is kept, both
    when given at construction and when set afterwards.
    """
    float_sum = Fraction(0.1 + 0.2)
    assert float_sum != Fraction(3, 10)

    implicit = Succession(Wait(0.1), Wait(0.2))
    assert implicit._exact_run_time() == Fraction(3, 10)

    given = Succession(Wait(0.1), Wait(0.2), run_time=float_sum)
    assert given._exact_run_time() == float_sum

    assigned = Succession(Wait(0.1), Wait(0.2))
    assigned.run_time = float_sum
    assert assigned._exact_run_time() == float_sum

    set_later = Succession(Wait(0.1), Wait(0.2)).set_run_time(float_sum)
    assert set_later._exact_run_time() == float_sum
