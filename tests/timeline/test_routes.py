"""The routes across frame rates: samples identical at the instants 4, 15,
30 and 60 fps share, and ``state_at`` answered by each play's route.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.timeline.routes import Unavailable

from . import route_scenes
from .conftest import render

ROUTE_SCENES = Path(route_scenes.__file__)
RATES = (4, 15, 30, 60)


def _assert_shared_frames_agree(scene_class) -> None:
    frames = {fps: render(scene_class, fps)[1] for fps in RATES}
    for slow, fast in itertools.combinations(RATES, 2):
        shared = [k for k in range(len(frames[slow])) if k * fast % slow == 0]
        assert shared
        for k in shared:
            assert np.array_equal(frames[slow][k], frames[fast][k * fast // slow]), (
                slow,
                fast,
                k,
            )


@pytest.mark.parametrize(
    "scene_class",
    [
        route_scenes.Compositions,
        route_scenes.Readouts,
        route_scenes.RefusedOverlappingTransforms,
        route_scenes.Conditional,
    ],
    ids=lambda c: c.__name__,
)
def test_samples_agree_at_the_instants_frame_rates_share(scene_class):
    """History-free plays only: a late reader shows the previous frame's
    value, as in CE (test_admission).
    """
    _assert_shared_frames_agree(scene_class)


def test_compositions_are_trusted_with_their_events_at_exact_instants(session_of):
    record = session_of("Compositions", ROUTE_SCENES).record
    assert [play.route for play in record.plays] == ["trusted"] * 4
    first, second, third, fourth = record.plays
    # Linear parts: their boundaries at thirds (on the tick) and at 1/7 of
    # 7/6 s after the play's start (off it).
    assert first.events[:2] == (Fraction(1, 3), Fraction(2, 3))
    assert second.events[0] - second.start == Fraction(7, 12)
    # smooth crosses the part boundaries at instants bisection finds, off
    # the tick grid.
    assert len(third.events) >= 2
    assert all((event * 60).denominator != 1 for event in third.events[:2])
    assert fourth.events


def test_state_at_answers_each_route(session_of):
    session = session_of("Readouts", ROUTE_SCENES)
    record = session.record
    play = record.plays[0]
    assert play.route == "state"
    on_tick = play.start + Fraction(7, 60)
    state = session.state_at(on_tick)
    assert (state.route, state.answer, state.play) == ("state", "recorded", 0)
    with pytest.raises(Unavailable, match="needs restore"):
        session.state_at(play.start + Fraction(1, 7))
    with pytest.raises(Unavailable):
        session.frame(play.start + Fraction(1, 7))
    # What restoring it will start from: the start capture and the
    # animations' state there.
    start = play.sample_at(play.start)
    assert start is not None
    assert list(play._animation_states) == [start.seq]

    trusted = session_of("TrustedTransform", ROUTE_SCENES)
    state = trusted.state_at(Fraction(1, 7))
    assert (state.route, state.answer) == ("trusted", "analytic")

    ticked = session_of("Conditional", ROUTE_SCENES)
    between = ticked.state_at(Fraction(1, 7))
    assert (between.route, between.answer) == ("ticked", "recorded")
    assert between.nodes == ticked.state_at(Fraction(8, 60)).nodes


def test_a_render_samples_a_state_function_play_at_its_own_frames():
    scene, _ = render(route_scenes.Readouts, 15)
    record = scene.record
    play = record.plays[0]
    assert play.route == "state"
    assert record.state_at(Fraction(2, 15)).answer == "recorded"
    with pytest.raises(Unavailable):
        record.state_at(Fraction(1, 60))


def test_a_trusted_answer_between_frames_is_the_one_a_render_there_records(session_of):
    """The session answers 1/7 analytically; a 7 fps render evaluates the
    play there and records it.
    """
    session = session_of("Compositions", ROUTE_SCENES)
    scene, frames = render(route_scenes.Compositions, 7)
    for k in range(1, 7):
        t = Fraction(k, 7)
        analytic = session.state_at(t)
        recorded = scene.record.state_at(t)
        assert analytic.answer == "analytic"
        assert recorded.answer == "recorded"
        assert analytic.nodes == recorded.nodes, k
        assert np.array_equal(np.asarray(session.frame(t)), frames[k]), k
