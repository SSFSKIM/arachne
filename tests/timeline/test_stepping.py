"""The stepping tick: a ticked play runs every callback at the ticks a frame
reads, its events, its last tick (and throughout a play with a stop
condition); every other tick steps only the updaters whose code loads dt,
the scene's updaters, the camera's inputs and all interpolation.

What a dt-free callback computes at a frame is then what CE computes: CE
calls it once per frame. History-free work comes out the same either way;
work that keeps history comes out as CE's per-frame evaluation does. Every
scene of the stepping fixtures is held against the same semantics computed
by brute force (``test_brute_force.py``).
"""

from __future__ import annotations

import functools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.timeline.clock import frames_in
from arachne.timeline.reference import BRUTE_FORCE
from arachne.timeline.ticking import UNGOVERNED, loads_dt

from . import soundness_scenes, stepping_scenes
from .conftest import render

STEPPING_SCENES = Path(stepping_scenes.__file__)
SOUNDNESS_SCENES = Path(soundness_scenes.__file__)


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
    assert play.full_ticks == fps
    assert play.stepping_ticks == 60 - fps
    _same_as_every_tick(session_of, "Unchanged", frames, fps)


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


# History kept at the frames


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
        assert shown == _shown(reference.state_at(at))
        other = _shown(everything.state_at(t))
        for name, upper in (("low", True), ("high", False)):
            part = [n for n in shown if (n[2][1] > 0) == upper]
            if part != [n for n in other if (n[2][1] > 0) == upper]:
                moved[name] = True
    assert moved == {"low": True, "high": True}


def test_the_captures_tell_history_kept_at_the_frames_from_every_tick(session_of):
    """Redraws that keep history differ between the two runs by design: CE
    builds them at its frames, which stepping keeps.
    """
    stepped = session_of("Staircases", STEPPING_SCENES, frame_rate=4)
    full = session_of("Staircases", STEPPING_SCENES, frame_rate=4, stepping=False)
    ours, theirs = stepped.record.plays[-1], full.record.plays[-1]
    assert [t for t, _ in ours.sample_keys()] == [t for t, _ in theirs.sample_keys()]
    assert ours.sample_keys() != theirs.sample_keys()


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


def test_a_fade_beside_a_dt_updater_that_reads_it_not_stays_stepping():
    scene, _ = render(stepping_scenes.FadeBeside, 4)
    play = _play(scene)
    assert not any(r.startswith(UNGOVERNED) for r in play.admission.reasons)
    assert play.tick_plan.skipped
    assert play.stepping_ticks == 60 - 4


# Who is handed dt, and whose code reads it


@pytest.mark.parametrize(
    "name", ["WrapsDt", "WrapsDtMethod", "DtInAnotherSlot", "DtInAnotherSlotMethod"]
)
def test_an_integrator_handed_dt_under_another_name_steps_at_every_tick(
    session_of, name
):
    """An integrator whose ``dt`` lands where its code does not call it dt
    (a ``functools.wraps`` wrapper forwarding ``*args``, or a parameter of
    another name in the second positional place) counts as reading it, and
    the dot integrates as when every tick is full.
    """
    stepped = session_of(name, STEPPING_SCENES, frame_rate=4)
    full = session_of(name, STEPPING_SCENES, frame_rate=4, stepping=False)
    assert stepped.record.plays[-1].stepping_ticks > 0
    for ours, theirs in zip(stepped.record.plays, full.record.plays, strict=True):
        assert ours.sample_keys() == theirs.sample_keys(), ours.index


@pytest.mark.parametrize(
    ("name", "loads"), [("DeadDtSceneUpdater", False), ("DtSceneUpdater", True)]
)
def test_a_scene_updater_runs_between_frames_only_when_it_reads_dt(
    session_of, name, loads
):
    """Every mode runs the scene updater at every full tick; between frames
    the stepping run and brute force run it only when its code reads the dt
    it is handed.
    """
    counts = {}
    for mode in (True, "brute_force", False):
        session = session_of(name, STEPPING_SCENES, frame_rate=4, stepping=mode)
        play = session.record.plays[-1]
        counts[mode] = (len(session.scene.calls), play.stepping_ticks, play.full_ticks)
    calls, stepping, full = counts[True]
    assert counts["brute_force"] == counts[True]
    assert stepping > 0
    assert counts[False][0] == (calls if loads else calls + stepping)


# An approved break (2026-11-03)


def test_approved_break_a_dt_check_sees_a_dt_free_output_a_frame_old(session_of):
    """A dt-loading callback sees a dt-free output one frame old between
    frames (approved 2026-11-03): a dt updater checking a dt-free updater's
    output against a tracker interpolation moves fails at a stepping tick,
    stepping and brute force alike, while ticking everything it holds.
    """
    for mode in (True, BRUTE_FORCE):
        with pytest.raises(AssertionError):
            session_of("GetterGuard", STEPPING_SCENES, frame_rate=4, stepping=mode)
    full = session_of("GetterGuard", STEPPING_SCENES, frame_rate=4, stepping=False)
    assert full.record.plays[-1].full_ticks == 60


# Plays whose updaters a stepping tick could not leave out


def test_a_scene_running_its_own_update_mobjects_ticks_everything(session_of):
    play = session_of("OwnUpdating", STEPPING_SCENES, frame_rate=4).record.plays[-1]
    assert play.route == "ticked"
    (reason,) = [r for r in play.admission.reasons if r.startswith(UNGOVERNED)]
    assert "update_mobjects is its own code" in reason
    assert play.tick_plan.unbounded == reason
    assert play.stepping_ticks == 0
    assert play.full_ticks == 60


# Plans: made from the structure, again when it changes


def test_an_updater_that_joins_at_an_event_is_planned_for(session_of):
    """A Succession part begins at 0.5 s and copies a dot whose updater a
    trace reads: the event tick plans again, and stepping goes on.
    """
    fps = 4
    scene, frames = render(stepping_scenes.LateJoiner, fps)
    play = _play(scene)
    assert not any(r.startswith(UNGOVERNED) for r in play.admission.reasons)
    assert play.stepping_ticks > 0
    _same_as_every_tick(session_of, "LateJoiner", frames, fps)


def test_become_padding_with_copies_that_update_plans_again(session_of):
    """``become`` pads a dot with copies of itself, its updater with them:
    the play plans for the copies before the next stepping tick, which skips
    them, so they move once a frame (as in CE) rather than once a tick, and
    no stepping tick meets work it never weighed.
    """
    session = session_of("BecomeRoot", STEPPING_SCENES, frame_rate=4)
    play = session.record.plays[-1]
    assert not any(r.startswith("tick.") for r in play.admission.reasons)
    assert [t for t, _ in play.tick_plans] == [play.start, Fraction(16, 60)]
    # Made at the frame at 0.25 s, the copies move there and at the frame at
    # 0.5 s: as a scene doing the same per frame by hand has them.
    box = session.state_at(Fraction(1, 2)).named("root").box
    assert (box[0] + box[2]) / 2 == pytest.approx(2 * 0.01)


@pytest.mark.parametrize("fps", [4, 15])
def test_stepping_off_ticks_everything_on_the_same_grid(session_of, fps):
    """With stepping off every tick is full and no plan is applied; dt-free
    work keeping no history leaves every capture as stepping does.
    """
    stepped = session_of("Unchanged", STEPPING_SCENES, frame_rate=fps)
    full = session_of("Unchanged", STEPPING_SCENES, frame_rate=fps, stepping=False)
    assert stepped.record.plays[-1].stepping_ticks == 60 - fps
    assert all(
        p.stepping_ticks == 0 and all(not plan.applied for _, plan in p.tick_plans)
        for p in full.record.plays
    )
    for ours, theirs in zip(stepped.record.plays, full.record.plays, strict=True):
        assert ours.sample_keys() == theirs.sample_keys(), ours.index


@pytest.mark.parametrize("name", ["SkippedInstaller", "SubmobjectAdded", "Unchanged"])
def test_a_run_that_ticks_everything_plans_where_stepping_would(session_of, name):
    """With stepping off, a play still makes its plans, and makes them
    again at the ticks a stepping run would, for a comparison to read; it
    applies none (every tick full, no reason recorded on its admission).
    """
    stepped = session_of(name, STEPPING_SCENES, frame_rate=15).record.plays[-1]
    full = session_of(
        name, STEPPING_SCENES, frame_rate=15, stepping=False
    ).record.plays[-1]
    assert [t for t, _ in full.tick_plans] == [t for t, _ in stepped.tick_plans]
    for (_, ours), (_, theirs) in zip(stepped.tick_plans, full.tick_plans, strict=True):
        assert ours.applied
        assert not theirs.applied
        assert len(theirs.skipped) == len(ours.skipped)
        assert theirs.unbounded == ours.unbounded
    assert full.stepping_ticks == 0
    assert not any(r.startswith("tick.") for r in full.admission.reasons)


# Diagnostic hooks


def test_on_step_sees_every_stepping_tick_and_no_full_one(session_of):
    """The diagnostic hook is called after each stepping tick, with the
    scene and the tick, and after no full tick.
    """
    seen: list[Fraction] = []
    session = session_of(
        "Unchanged",
        STEPPING_SCENES,
        frame_rate=4,
        on_step=lambda scene, tick: seen.append(tick),
    )
    play = session.record.plays[-1]
    assert len(seen) == play.stepping_ticks == 60 - 4
    assert not set(seen) & {Fraction(k, 4) for k in range(5)}
    full = session_of(
        "Unchanged",
        STEPPING_SCENES,
        frame_rate=4,
        stepping=False,
        on_step=lambda scene, tick: seen.append(tick),
    )
    assert full.record.plays[-1].stepping_ticks == 0
    assert len(seen) == 60 - 4


@pytest.mark.parametrize("stepping", [True, False])
def test_on_tick_sees_every_tick_and_whether_it_was_full(session_of, stepping):
    """The diagnostic hook is called after every tick of a ticked play,
    stepping or full, and says which: with stepping off, every one is full.
    """
    seen: list[tuple[Fraction, bool]] = []
    session = session_of(
        "Unchanged",
        STEPPING_SCENES,
        frame_rate=4,
        stepping=stepping,
        on_tick=lambda scene, tick, full: seen.append((tick, full)),
    )
    play = session.record.plays[-1]
    assert [t for t, _ in seen] == [Fraction(k, 60) for k in range(1, 61)]
    assert sum(not full for _, full in seen) == play.stepping_ticks
    assert sum(full for _, full in seen) == play.full_ticks
    assert play.stepping_ticks == (60 - 4 if stepping else 0)


@pytest.mark.parametrize("stepping", [True, False])
def test_on_start_sees_each_ticked_play_once_after_its_start_and_before_its_ticks(
    session_of, stepping
):
    """``on_start`` is called once for each ticked play, at its start (on a
    tick or between two), after the start's evaluation ran the dt-free
    updater there and before any tick; ``on_tick`` never sees the start.
    """
    seen: list[tuple[str, Fraction, float | None]] = []

    def on_start(scene, start):
        last = scene.evaluated[-1] if scene.evaluated else None
        seen.append(("start", start, last))

    session = session_of(
        "EvaluationClock",
        STEPPING_SCENES,
        frame_rate=4,
        stepping=stepping,
        on_tick=lambda scene, tick, full: seen.append(("tick", tick, None)),
        on_start=on_start,
    )
    ticked = [p for p in session.record.plays if p.route == "ticked"]
    assert [p.start for p in ticked] == [Fraction(0), Fraction(101, 200)]
    starts = [(t, float(last)) for kind, t, last in seen if kind == "start"]
    assert starts == [(p.start, float(p.start)) for p in ticked]
    begins = [k for k, (kind, _, _) in enumerate(seen) if kind == "start"]
    assert begins[0] == 0
    for play, begin, after in zip(
        ticked, begins, [*begins[1:], len(seen)], strict=True
    ):
        ticks = [t for _, t, _ in seen[begin + 1 : after]]
        assert all(play.start < t <= play.end for t in ticks)
        assert len(ticks) == play.full_ticks + play.stepping_ticks


def test_a_hook_that_only_observes_leaves_every_frame_as_it_was(session_of):
    """The same scene with an observing ``on_start`` hook and with none:
    the same plan, the same stepping, the same pixels.
    """
    seen: list[Fraction] = []
    plain = session_of("HookRebound", SOUNDNESS_SCENES, frame_rate=4)
    observed = session_of(
        "HookRebound",
        SOUNDNESS_SCENES,
        frame_rate=4,
        on_start=lambda scene, start: seen.append(start),
    )
    assert seen == [Fraction(0)]
    ours, theirs = observed.record.plays[-1], plain.record.plays[-1]
    assert ours.stepping_ticks == theirs.stepping_ticks > 0
    assert ours.sample_keys() == theirs.sample_keys()
    for k in range(5):
        at = Fraction(k, 4)
        assert np.array_equal(
            np.asarray(observed.frame(at)), np.asarray(plain.frame(at))
        ), at


# Diagnostic only: what the stepping set reaches, worked out when asked


def test_the_stepping_families_hold_what_stepping_reads_and_writes(session_of):
    session = session_of("MethodRead", STEPPING_SCENES, frame_rate=4)
    _, other, output, idle = session.scene.mobjects
    families = session.record.plays[-1].tick_plan.stepping_families()
    assert {id(other), id(output)} <= families
    assert id(idle) not in families
