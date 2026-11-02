"""The forward run: the 60 Hz tick (break 4), the event instants and the
phase order (break 5), play() inside an updater (break 6), and seeding.
"""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

from arachne.timeline.clock import TICK_DT

from .conftest import SCENES, line_of, render
from .scenes import (
    DtSteps,
    LazyTargetOffTick,
    LazyTargetOffTickTicked,
    LazyTargetOnTick,
    PlayInUpdater,
    Seeded,
)


@pytest.mark.parametrize("fps", [4, 15, 30])
def test_dt_updaters_step_on_the_tick_whatever_the_frame_rate(fps):
    scene, _ = render(DtSteps, fps)
    steps = [dt for dt in scene.steps if dt != 0]
    # 30 ticks in the half-second play, then 8 in the 1/7 s wait: the ticks
    # k/60 in (1/2, 1/2 + 1/7].
    assert steps == [TICK_DT] * 38
    assert all(type(dt) is float for dt in steps)


def test_scenes_are_seeded_whatever_the_configuration():
    first, _ = render(Seeded, 15)
    second, _ = render(Seeded, 15)
    assert first.drawn == second.drawn
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002
    assert first.drawn == (random.random(), np.random.random())  # noqa: NPY002


def test_play_inside_an_updater_raises_with_its_line():
    inner = f"{SCENES.name}:{line_of('inner')}"
    outer = f"{SCENES.name}:{line_of('outer')}"
    with pytest.raises(RuntimeError, match="from an updater") as raised:
        render(PlayInUpdater, 15)
    message = str(raised.value)
    assert f"at {inner}" in message
    assert f"the one at {outer}" in message


@pytest.mark.parametrize("fps", [4, 30])
def test_a_succession_part_captures_its_target_after_the_parallel_animations(fps):
    """The second part, ``Transform(a, b)``, begins at 1/2 and copies ``b``
    there after the parallel animation moved it: ``a`` ends where ``b`` was
    at 1/2, x = 3. CE begins the part before the later animation of the same
    sample runs, so it copies ``b`` as the previous sample left it, which
    depends on the frame rate (x = 1.5 at 4 fps).
    """
    scene, _ = render(LazyTargetOnTick, fps)
    assert scene.a.get_center()[0] == pytest.approx(3.0)
    assert scene.b.get_center()[0] == pytest.approx(6.0)


@pytest.mark.parametrize("fps", [4, 30])
def test_a_succession_event_between_ticks_runs_at_the_next_tick(fps):
    """The part boundary is at 1/7 s; a ticked play takes it at the first
    tick at or after it, 9/60, whatever the frame rate.
    """
    scene, _ = render(LazyTargetOffTickTicked, fps)
    assert scene.record.plays[0].route == "ticked"
    assert scene.a.get_center()[0] == pytest.approx(6 * 9 / 60)


@pytest.mark.parametrize("fps", [4, 30])
def test_a_trusted_succession_part_begins_at_its_exact_instant(fps):
    """The same boundary in a trusted play is taken at 1/7 itself, between
    ticks and frames: ``a`` copies ``b`` where it was then, x = 6/7.
    """
    scene, _ = render(LazyTargetOffTick, fps)
    play = scene.record.plays[0]
    assert play.route == "trusted"
    assert play.events[0] == Fraction(1, 7)
    assert scene.a.get_center()[0] == pytest.approx(6 / 7)


def test_a_trusted_play_answers_between_its_instants(session_of):
    """Between the event and the end the record evaluates the Transform at
    the instant asked: ``a`` runs from x = -3 at 1/7 to 6/7 at 1, linearly.
    """
    session = session_of("LazyTargetOffTick", grid=None)
    start = session.state_at(0).named("a").box
    assert session.state_at(Fraction(1, 7)).named("a").box == start
    for t in (Fraction(2, 7), Fraction(1, 2), Fraction(17, 20)):
        state = session.state_at(t)
        assert state.answer == "analytic"
        box = state.named("a").box
        x = (box[0] + box[2]) / 2
        expected = -3 + (6 / 7 + 3) * float((t - Fraction(1, 7)) / Fraction(6, 7))
        assert x == pytest.approx(expected, abs=1e-9)


def test_the_record_holds_the_part_begun_at_its_tick(session_of):
    """Sampled without a grid, the record shows ``a`` at its start up to the
    tick before the event, and moving from the tick of the event on, whose
    sample (as CE's) evaluates the part it began.
    """
    session = session_of("LazyTargetOffTickTicked")
    start = session.state_at(0).named("a").box
    assert session.state_at(Fraction(8, 60)).named("a").box == start
    # Between ticks the record answers with the last tick at or before.
    assert session.state_at(Fraction(17, 120)).named("a").box == start
    assert session.state_at(Fraction(9, 60)).named("a").box != start
