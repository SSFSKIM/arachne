"""Scene hooks: Arachne calls CE's Scene methods where CE calls them, so a
scene's overrides and the wrappers set on its object run as they run on CE.

CE calls ``play_internal()`` once per play that is not a frozen frame, and,
inside it, ``update_to_time(t)`` once per frame with the play-local frame
time, which calls ``update_mobjects``, ``update_meshes`` and ``update_self``
with the frame's dt; after the last frame, ``update_mobjects(0)``. Arachne
calls these frame hooks at the instants a frame of its grid reads (a
gridless session reads every 60 Hz tick), never at a stepping tick or an
event between frames, and does its own work in the methods' bodies: a dt
updater integrates once per tick with the tick's dt, whatever dt a hook is
shown. Overrides it cannot honour are refused by name. The expected
sequences are CE's loop (``hook_scenes.ce_play``); every run time here is a
whole number of frames, so they hold exactly.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.timeline.clock import frames_in
from arachne.timeline.reference import BRUTE_FORCE

from . import hook_scenes
from .conftest import render
from .hook_scenes import ce_guard_samples, ce_play

HOOK_SCENES = Path(hook_scenes.__file__)
FPS = 15


def _play_calls(session) -> list[list[tuple[str, object]]]:
    return session.scene.log.plays()


# (a) every route


@pytest.mark.parametrize("fps", [15, 30])
@pytest.mark.parametrize(
    ("name", "route"),
    [("TrustedHooks", "trusted"), ("StateHooks", "state"), ("TickedHooks", "ticked")],
)
def test_every_route_calls_ces_hooks_at_its_frames(session_of, name, route, fps):
    session = session_of(name, HOOK_SCENES, frame_rate=fps)
    (play,) = session.record.plays
    assert play.route == route
    assert _play_calls(session) == [ce_play(fps, fps)]
    if route == "ticked":
        # The hooks run at the frames' ticks, which are full; the others
        # still only step.
        assert play.full_ticks == fps
        assert play.stepping_ticks == 60 - fps


# (b) a scene clock kept by class overrides


def test_class_overrides_keep_the_scene_clock_ce_shows_at_every_frame(session_of):
    session = session_of("PlayClock", HOOK_SCENES, frame_rate=FPS)
    plays = session.record.plays
    assert [p.held for p in plays] == [False, True, False]
    grid = Fraction(FPS)
    shown = hook_scenes.PlayClock.SCALE

    def x(now) -> float:
        return float(now) * shown - 3

    previous = None
    for play in plays:
        instants = frames_in(play.start, play.end, grid)
        for j, t in enumerate(instants):
            if play.held:
                # A frozen wait shows what the last play left: CE's clock at
                # its end, plus the last frame's local time its overrides kept.
                end, frames = previous
                expected = end + Fraction(frames - 1, FPS)
            else:
                expected = play.start + 2 * Fraction(j, FPS)
            x0, _, x1, _ = session.state_at(t).named("marker").box
            assert (x0 + x1) / 2 == pytest.approx(x(expected), abs=1e-9), (
                play.index,
                j,
            )
        if not play.held:
            previous = (play.end, len(instants))


# (c) a checker on the scene object


def test_a_checker_armed_on_the_scene_object_sees_a_defect_held_mid_play(session_of):
    guard = session_of("DriftingCaption", HOOK_SCENES, frame_rate=FPS).scene.guard
    assert guard.samples == ce_guard_samples(
        [(Fraction(0), 4 * FPS, Fraction(4)), (Fraction(4), 0, Fraction(23, 5))], FPS
    )
    assert guard.report == [(float(Fraction(16, FPS)), float(Fraction(36, FPS)))]


# (d) waits


def test_waits_call_the_hooks_ce_calls(session_of):
    session = session_of("Waits", HOOK_SCENES, frame_rate=FPS)
    frozen, running, stopped = session.record.plays
    assert [p.held for p in (frozen, running, stopped)] == [True, False, False]
    calls = _play_calls(session)
    # A frozen wait calls nothing.
    assert calls[0] == []
    # A wait a dt updater keeps running is a play of its frames.
    assert calls[1] == ce_play(9, FPS, wait_updates=True)
    # The stop condition is read at every tick and first holds at the 26th
    # (0.4333 s), where the wait ends: its length never depends on the frame
    # rate. The frame hooks run at the frames before that end, 0 to 6.
    assert calls[2] == ce_play(7, FPS, wait_updates=True)
    assert stopped.end - stopped.start == Fraction(26, 60)
    assert (stopped.full_ticks, stopped.stepping_ticks) == (26, 0)


# (e) an event between frames


def test_an_event_between_frames_calls_no_hook(session_of):
    session = session_of("OffGridEvent", HOOK_SCENES, frame_rate=FPS)
    (play,) = session.record.plays
    assert play.route == "trusted"
    # The second part begins at 7/20 s, between frames 5 and 6.
    assert Fraction(7, 20) in play.events
    assert _play_calls(session) == [ce_play(FPS, FPS)]


# (f) overrides Arachne refuses


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("ProgressionOverride", "hook.time_progression"),
        ("PlayInternalSkipped", "hook.play_internal.skipped"),
        ("PlayInternalTwice", "hook.play_internal.twice"),
        ("PlayInternalSkipRendering", "hook.play_internal.skip_rendering"),
        ("UpdateToTimeWarp", "hook.update_to_time.time"),
        ("UpdateToTimeReplaced", "hook.update_to_time.replaced"),
        ("UpdateMobjectsReplaced", "hook.update_mobjects.replaced"),
        ("UpdateSelfReplaced", "hook.update_self.replaced"),
        ("UpdateToTimeTwice", "hook.update_to_time.twice"),
        ("UpdateMobjectsTwice", "hook.update_mobjects.twice"),
    ],
)
def test_an_override_that_cannot_be_dispatched_is_refused_by_name(
    session_of, name, reason
):
    with pytest.raises(Exception) as caught:
        session_of(name, HOOK_SCENES, frame_rate=FPS)
    assert type(caught.value).__name__ == "SceneHookRefused", caught.value
    assert caught.value.reason == reason


# (g) a run time the scene decides


def test_a_get_run_time_override_sets_the_span(session_of):
    session = session_of("RunTimeOverride", HOOK_SCENES, frame_rate=FPS)
    (play,) = session.record.plays
    assert play.end - play.start == 2
    assert _play_calls(session) == [ce_play(2 * FPS, FPS)]


@pytest.mark.parametrize("stepping", [True, BRUTE_FORCE])
@pytest.mark.parametrize(
    ("name", "run_times"),
    [
        ("RunTimeChangesInPlay", [1.0, 2.0]),
        ("RunTimeScaledInPlay", [1.0, 2.0]),
        ("AnimationLengthenedInPlay", []),
    ],
)
def test_the_run_time_play_internal_asks_for_sets_the_span(
    session_of, name, run_times, stepping
):
    """CE plays the length ``play_internal``'s body takes, which may differ
    from the one compiling the play took: the play is planned again for it.
    """
    fps = 4
    session = session_of(name, HOOK_SCENES, frame_rate=fps, stepping=stepping)
    (play,) = session.record.plays
    scene = session.scene
    assert scene.run_times == run_times
    assert play.route == "ticked"
    assert (play.requested_end, play.end) == (2, 2)
    assert scene.frames == [j / fps for j in range(2 * fps)]
    # Its frames' ticks and its last are full; the event at 1.5 s falls on a
    # frame.
    assert (play.full_ticks, play.stepping_ticks) == (2 * fps, 120 - 2 * fps)
    if name == "RunTimeChangesInPlay":
        assert play.events == (Fraction(3, 2), 2)


@pytest.mark.parametrize("stepping", [True, BRUTE_FORCE])
def test_a_play_is_planned_from_the_run_times_its_body_begins_with(
    session_of, stepping
):
    """``play_internal`` retimes an animation without moving the span: the
    play's events are the retimed ones.
    """
    session = session_of(
        "SameSpanRetimed", HOOK_SCENES, frame_rate=4, stepping=stepping
    )
    (play,) = session.record.plays
    assert play.route == "ticked"
    assert play.end - play.start == 3
    assert play.events == (Fraction(3, 2), 3)
    # The first part moves up over 1.5 s: at 1.25 s it is 5/6 of the way.
    x0, y0, x1, y1 = session.state_at(Fraction(5, 4)).named("first").box
    assert (y0 + y1) / 2 == pytest.approx(5 / 6, abs=1e-9)


def test_a_live_play_draws_the_frames_of_the_span_its_body_takes():
    scene, frames = render(hook_scenes.LiveSpanGrows, 4)
    record = scene.record
    assert record.refusal is not None
    wait, play = record.plays
    assert (play.start, play.end) == (Fraction(1, 10), Fraction(11, 10))
    # The wait's frame at 0, then the play's at 0.25, 0.5, 0.75 and 1.
    assert len(frames) == 1 + 4


# (h) the dt an override sees, the dt an updater integrates


def test_an_update_mobjects_override_sees_the_frame_dt_while_updaters_integrate_the_tick(
    session_of,
):
    session = session_of("UpdateOverride", HOOK_SCENES, frame_rate=FPS)
    (play,) = session.record.plays
    scene = session.scene
    assert scene.seen == [0.0] + [float(Fraction(1, FPS))] * (FPS - 1) + [0.0]
    # Every tick once, with the tick's dt: the play's exact length, not CE's
    # (FPS - 1) / FPS and never a frame's dt on top.
    assert scene.total == pytest.approx(1.0, abs=1e-9)
    # An override of a frame hook runs at the frames; the play still steps.
    assert play.route == "ticked"
    assert play.stepping_ticks == 60 - FPS


# (i) sessions


def test_a_session_calls_the_hooks_at_its_grid(session_of):
    session = session_of("TrustedHooks", HOOK_SCENES, grid=Fraction(4))
    assert _play_calls(session) == [ce_play(4, 4)]


def test_a_gridless_session_calls_them_at_every_tick(session_of):
    session = session_of("TrustedHooks", HOOK_SCENES, grid=None)
    assert _play_calls(session) == [ce_play(60, 60)]


# (j) re-execution


def test_wrappers_set_in_setup_see_one_tear_down_per_completed_execution():
    hook_scenes.LOGS.clear()
    scene, _ = render(hook_scenes.Reexecuted, 6)
    assert scene.record.execution.number == 2
    assert len(hook_scenes.LOGS) == 2
    # CE never tears down a run that raised: the abandoned execution's
    # wrappers see none.
    assert [log.calls.count(("tear_down", None)) for log in hook_scenes.LOGS] == [0, 1]
    assert hook_scenes.LOGS[-1].plays()[0] == ce_play(6, 6)


# (k) a class patch of a hook


def test_a_class_patch_of_update_to_time_runs_at_the_frames_and_ticks_nothing(
    session_of, monkeypatch
):
    from arachne import Scene

    seen: list[float] = []
    original = Scene.update_to_time

    def patched(self, t):
        seen.append(float(t))
        return original(self, t)

    monkeypatch.setattr(Scene, "update_to_time", patched)
    session = session_of("TrustedHooks", HOOK_SCENES, frame_rate=FPS)
    (play,) = session.record.plays
    assert play.route == "trusted"
    assert seen == [float(Fraction(j, FPS)) for j in range(FPS)]


# The frame hook and dt updaters


def test_a_checker_on_the_frame_hook_changes_nothing_a_dt_updater_integrates():
    bare, _ = render(hook_scenes.DtIntegrator, FPS)
    guarded, _ = render(hook_scenes.DtIntegratorGuarded, FPS)
    assert guarded.state() == bare.state()
    # 1 s, a 1 s wait and 0.6 s, each integrated tick by tick.
    assert bare.total == pytest.approx(2.6, abs=1e-9)
    # The checker ran at the frames (so the equality is not for want of it).
    assert guarded.guard.samples == ce_guard_samples(
        [
            (Fraction(0), FPS, Fraction(1)),
            (Fraction(1), FPS, Fraction(2)),
            (Fraction(2), 9, Fraction(13, 5)),
        ],
        FPS,
    )


# What the record keeps of the hooks, and what the ticks do around them


def test_a_play_records_the_hooks_the_scene_held(session_of):
    def first(name):
        return session_of(name, HOOK_SCENES, frame_rate=FPS).record.plays[0]

    plain = first("DtIntegrator")
    assert (plain.frame_hooks, plain.play_internal_overridden) == ((), False)
    # A wrapper set on the scene object.
    wrapped = first("DriftingCaption")
    assert (wrapped.frame_hooks, wrapped.play_internal_overridden) == (
        ("update_to_time",),
        False,
    )
    # An override in the scene's class.
    overriding = first("UpdateOverride")
    assert overriding.frame_hooks == ("update_mobjects",)
    assert first("PlayClock").play_internal_overridden


@pytest.mark.parametrize("stepping", [False, BRUTE_FORCE])
def test_the_hooks_run_at_the_same_frames_however_the_ticks_run(session_of, stepping):
    session = session_of("TickedHooks", HOOK_SCENES, frame_rate=FPS, stepping=stepping)
    (play,) = session.record.plays
    assert play.route == "ticked"
    assert _play_calls(session) == [ce_play(FPS, FPS)]


def test_a_frame_hook_that_adds_an_updater_has_the_play_plan_again(session_of):
    stepped = session_of("HookAddsReader", HOOK_SCENES, frame_rate=FPS)
    brute = session_of(
        "HookAddsReader", HOOK_SCENES, frame_rate=FPS, stepping=BRUTE_FORCE
    )
    (play,) = stepped.record.plays
    # Planned at the start, then again at the stepping tick after the first
    # frame from 0.5 s on (8/15 s, the 32nd tick).
    assert [first for first, _ in play.tick_plans] == [0, Fraction(33, 60)]
    assert play.stepping_ticks == 60 - FPS
    assert play.sample_keys() == brute.record.plays[0].sample_keys()
    assert stepped.scene.reader.get_value() == brute.scene.reader.get_value() != 0
