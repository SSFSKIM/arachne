"""The rational clock and the one global frame grid (deliberate break 3)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne import Scene, Square, Wait
from arachne.timeline.clock import TICK_DT, exact, frames_in, ticks_in

from .conftest import render
from .scenes import Cue, Grid


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


def test_scene_time_is_exact_and_renderer_time_its_float():
    """``scene.time`` is the exact clock and ``renderer.time`` its float. A
    wait sized from the float clock, as a cue sizes it, lasts the simplest
    fraction its float duration rounds to, so it lands within a rounding
    error of its target; each grid frame still belongs to exactly one play.
    """
    scene, frames = render(Cue, 15)
    plays = scene.record.plays
    assert [play.end for play in plays[:2]] == [Fraction(1, 3), Fraction(2, 3)]
    assert plays[2].end == Fraction(2, 3) + exact(2.5 - float(Fraction(2, 3)))
    assert isinstance(scene.time, Fraction)
    assert scene.time == plays[2].end
    assert isinstance(scene.renderer.time, float)
    assert scene.renderer.time == float(scene.time)
    assert abs(scene.renderer.time - 2.5) < 1e-15
    assert len(frames) == len(frames_in(Fraction(0), scene.time, Fraction(15)))
