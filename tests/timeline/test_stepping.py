"""The stepping tick: a ticked play runs every callback at the ticks a frame
reads, its events, its last tick (and throughout a play with a stop
condition); every other tick steps only what integrates, the scene's
updaters, all interpolation, and the dt-free callbacks those may read.

What a skipped callback computes at a frame is then what CE computes: CE
calls it once per frame. History-free work comes out the same either way;
work that keeps history comes out as CE's per-frame evaluation does.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.timeline.clock import frames_in
from arachne.timeline.ticking import UNBOUNDED, loads_dt

from . import stepping_scenes
from .conftest import render

STEPPING_SCENES = Path(stepping_scenes.__file__)

COLLAPSE_WARNING = "ignore:put_start_and_end_on has been called:UserWarning"


def _play(scene):
    (play,) = [p for p in scene.timeline.ensure_record().plays if p.kind == "play"]
    return play


def _same_as_every_tick(session_of, name: str, frames, fps: int) -> None:
    """``frames`` equal those of a session that runs every callback at every
    tick (no grid): skipping changed no frame.
    """
    session = session_of(name, STEPPING_SCENES, grid=None, frame_rate=fps)
    assert all(p.stepping_ticks == 0 for p in session.record.plays)
    for k, frame in enumerate(frames):
        pixels = np.asarray(session.frame(Fraction(k, fps)))
        assert np.array_equal(frame, pixels), (name, fps, k)


# What a stepping tick runs


@pytest.mark.parametrize("fps", [4, 15])
def test_dt_free_work_that_keeps_no_history_steps_and_changes_no_frame(session_of, fps):
    scene, frames = render(stepping_scenes.Unchanged, fps)
    play = _play(scene)
    assert play.route == "ticked"
    plan = play.tick_plan
    # The follower and the redraw wait for the frames; the clock and the
    # spinner integrate.
    assert len(plan.skipped) == 2
    assert plan.stepping == 2
    assert plan.pulled == 0
    assert play.full_ticks == fps
    assert play.stepping_ticks == 60 - fps
    _same_as_every_tick(session_of, "Unchanged", frames, fps)


@pytest.mark.parametrize("fps", [4, 15])
def test_a_trace_keeps_the_updaters_it_reads_on_the_tick(session_of, fps):
    """One trace reads its dot through a bound method, the other through a
    lambda: both dots' dt-free updaters step with them.
    """
    scene, frames = render(stepping_scenes.TracedReader, fps)
    plan = _play(scene).tick_plan
    assert plan.pulled == 2
    assert plan.skipped == frozenset()
    _same_as_every_tick(session_of, "TracedReader", frames, fps)


@pytest.mark.filterwarnings(COLLAPSE_WARNING)
def test_a_collapse_a_stepping_reader_sees_is_kept_at_every_rate():
    """``route_scenes.Collapse`` misses the collapse at 4 fps, as CE does; a
    trace of the path's end keeps its updater on the tick, so 4 fps sees
    the path collapse at 0.3 s as 60 fps does.
    """
    slow, _ = render(stepping_scenes.TracedCollapse, 4)
    fast, _ = render(stepping_scenes.TracedCollapse, 60)
    path = slow.mobjects[0]
    assert np.ptp(path.points, axis=0).max() == 0
    assert np.array_equal(path.points, fast.mobjects[0].points)


@pytest.mark.parametrize("fps", [4, 15])
def test_interpolation_runs_at_every_tick(fps):
    calls = stepping_scenes.EveryTick.calls
    calls.clear()
    scene, _ = render(stepping_scenes.EveryTick, fps)
    play = _play(scene)
    assert play.stepping_ticks == 60 - fps
    # Beginning the animation, the play's start, its 60 ticks, finishing it.
    assert len(calls) == 1 + 1 + 60 + 1


# A dt parameter the code never reads


def test_an_updater_integrates_only_when_its_code_reads_dt():
    def reads(m, dt):
        m.shift(dt)

    def reads_inside(m, dt):
        return [dt * k for k in range(2)]

    def ignores(m, dt):
        return m

    assert loads_dt(reads)
    assert loads_dt(reads_inside)
    assert loads_dt(lambda m, dt: m.shift(dt))
    assert not loads_dt(ignores)
    assert not loads_dt(lambda m, dt: m)
    assert not loads_dt(lambda m: m)
    # Code it can't read counts as reading it.
    assert loads_dt(functools.partial(reads, None))


@pytest.mark.parametrize("fps", [4, 15])
def test_a_guard_that_never_reads_dt_runs_with_the_redraw_it_checks(fps):
    """The guard runs after the redraw it checks and fails on a stale one:
    stepping alone, it would see a dot interpolation moved and a redraw
    left for the frame. Dead in dt, it runs at the full ticks only.
    """
    calls = stepping_scenes.StaleGuard.calls
    calls.clear()
    scene, _ = render(stepping_scenes.StaleGuard, fps)
    play = _play(scene)
    assert play.stepping_ticks == 60 - fps
    assert play.full_ticks == fps
    # The play's start, its full ticks, and the update after it finishes.
    assert len(calls) == 1 + fps + 1


# Callbacks that keep history


def _shown(state) -> list[tuple]:
    return [
        (node.type, node.z, node.box, node.color)
        for node in state.nodes
        if node.box is not None
    ]


def test_redraws_that_keep_history_are_built_at_the_frames_as_in_ce(session_of):
    """Each staircase's target keeps the members ``become`` padded it with, so
    what it shows depends on the instants it was built at. At 4 fps the
    record builds them at the frames, as CE does (the reference builds
    them by hand at the frames); built at every tick, they show otherwise.
    """
    fps = 4
    stepped = session_of("Staircases", STEPPING_SCENES, frame_rate=fps)
    everything = session_of("Staircases", STEPPING_SCENES, grid=None, frame_rate=fps)
    reference = session_of("StaircasesPerFrame", STEPPING_SCENES, frame_rate=fps)
    play = stepped.record.plays[-1]
    assert play.stepping_ticks == 60 - fps
    instants = frames_in(play.start, play.end, Fraction(fps))
    by_hand = [p.start for p in reference.record.plays]
    assert len(instants) == len(by_hand) == fps
    moved = {"low": False, "high": False}
    for t, at in zip(instants, by_hand, strict=True):
        shown = _shown(stepped.state_at(t))
        assert shown == _shown(reference.state_at(at)), t
        other = _shown(everything.state_at(t))
        for name, upper in (("low", True), ("high", False)):
            part = [n for n in shown if (n[2][1] > 0) == upper]
            if part != [n for n in other if (n[2][1] > 0) == upper]:
                moved[name] = True
    assert moved == {"low": True, "high": True}


# Reads that can't be bounded


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("DynamicRead", "uses getattr"),
        ("OwnAnimationScene", "OwnAnimation is the scene's own animation class"),
    ],
)
def test_a_play_whose_reads_cant_be_bounded_ticks_everything(session_of, name, why):
    play = session_of(name, STEPPING_SCENES, frame_rate=4).record.plays[-1]
    assert play.route == "ticked"
    (reason,) = [r for r in play.admission.reasons if r.startswith(UNBOUNDED)]
    assert reason.startswith(f"{UNBOUNDED}: ")
    assert why in reason
    assert play.tick_plan.unbounded == reason
    assert play.stepping_ticks == 0
    assert play.full_ticks == 60


def test_an_updater_that_joins_at_an_event_is_planned_for(session_of):
    """A Succession part begins at 0.5 s and copies a dot whose updater a
    trace reads: the event tick plans again, and stepping goes on.
    """
    fps = 4
    scene, frames = render(stepping_scenes.LateJoiner, fps)
    play = _play(scene)
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.stepping_ticks > 0
    _same_as_every_tick(session_of, "LateJoiner", frames, fps)


# Sessions


def test_a_session_records_on_its_render_s_grid(session_of):
    """By default a session steps as the render does, so its frames are the
    render's; without a grid every tick is full.
    """
    fps = 4
    _, frames = render(stepping_scenes.Staircases, fps)
    session = session_of("Staircases", STEPPING_SCENES, frame_rate=fps)
    assert session.record.plays[-1].stepping_ticks == 60 - fps
    for k, frame in enumerate(frames):
        assert np.array_equal(frame, np.asarray(session.frame(Fraction(k, fps)))), k
