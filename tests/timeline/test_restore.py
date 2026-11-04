"""Restoring a state-function play at an instant its record did not sample:
the answer equals a forward run's evaluation of the play there, a non-frame
instant of the same execution; restores change nothing they leave; and a
ticked play answers an instant its record kept no tick of with the last
capture it kept.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.timeline.draw import Drawing
from arachne.timeline.origins import ORIGINS
from arachne.timeline.record import NodeState
from arachne.timeline.restore import RestorePoint, Unrestorable, restored
from arachne.timeline.routes import Unavailable, snapshot_animations

from . import camera_scenes, restore_scenes, route_scenes
from .conftest import render

CAMERA_SCENES = Path(camera_scenes.__file__)
RESTORE_SCENES = Path(restore_scenes.__file__)
ROUTE_SCENES = Path(route_scenes.__file__)


def _unsampled(play, parts=(1, 3, 5)) -> list[Fraction]:
    """Instants at ``k/7`` of the play's span, none the record sampled."""
    found = [play.start + play.duration * Fraction(k, 7) for k in parts]
    assert all(play.sample_at(t) is None for t in found)
    return found


def _state_plays(session) -> list:
    return [p for p in session.record.plays if p.route == "state" and not p.held]


def _canonical(state) -> tuple:
    """A state with each node's id replaced by its place among the nodes: a
    mobject a callback builds at the instant takes the next ordinal of the
    execution evaluating it, so two executions that evaluated other instants
    before number it apart.
    """
    place = {node.id: index for index, node in enumerate(state.nodes)}
    nodes = tuple(
        (
            node.source_line,
            node.name,
            node.type,
            node.box,
            node.z,
            node.color,
            node.text,
            tuple(place[child] for child in node.children),
        )
        for node in state.nodes
    )
    return state.play, state.route, tuple(place[r] for r in state.roots), nodes


def _assert_restored_equals_forward(session, oracle, t) -> None:
    restored_state, forward = session.state_at(t), oracle.state_at(t)
    assert (restored_state.answer, forward.answer) == ("restored", "recorded"), t
    assert restored_state.as_of == forward.as_of == t
    assert _canonical(restored_state) == _canonical(forward), t
    assert np.array_equal(np.asarray(session.frame(t)), np.asarray(oracle.frame(t))), t


def _restore_against_forward(session_of, path, name, fps) -> tuple:
    session = session_of(name, path, frame_rate=fps)
    instants = [t for play in _state_plays(session) for t in _unsampled(play)]
    assert instants
    oracle = session_of(name, path, frame_rate=fps, instants=instants)
    for t in instants:
        _assert_restored_equals_forward(session, oracle, t)
    return session, oracle, instants


@pytest.mark.parametrize("fps", [4, 15, 30])
def test_a_pure_centre_function_restored_equals_a_forward_run(session_of, fps):
    """The camera's fixed-orientation centre is evaluated at the restored
    instant: the label follows the dot between frames.
    """
    session, oracle, instants = _restore_against_forward(
        session_of, CAMERA_SCENES, "PureCentre", fps
    )
    first, second = instants[:2]
    assert (
        session.state_at(first).nodes[-1].box != session.state_at(second).nodes[-1].box
    )


@pytest.mark.parametrize(
    "name",
    [
        "ClosureCell",
        "DetachedTracker",
        "SuccessionReadout",
        "FrameCounter",
    ],
)
def test_a_state_play_restored_equals_a_forward_run(session_of, name):
    _restore_against_forward(session_of, RESTORE_SCENES, name, 15)


def test_readouts_restored_equal_a_forward_run(session_of):
    """``always_redraw``, setters driven by a tracker, a readout of another
    mobject, ``DecimalNumber`` and ``UpdateFromAlphaFunc``.
    """
    session, _, instants = _restore_against_forward(
        session_of, ROUTE_SCENES, "Readouts", 15
    )
    texts = {session.state_at(t).named("number").text for t in instants[:3]}
    assert len(texts) == 3


def test_a_closure_cell_holds_its_value_from_the_play_while_restored(session_of):
    """The cell holds 1.5 once the scene has run; the first play's restore
    evaluates with the 0.0 it held then, and gives the cell back.
    """
    session = session_of("ClosureCell", RESTORE_SCENES, frame_rate=15)
    first, second = session.record.plays
    (dot,) = [m for m in session.scene.mobjects if m.updaters]
    (updater,) = dot.updaters
    cell = updater.__closure__[updater.__code__.co_freevars.index("offset")]
    assert cell.cell_contents == 1.5
    t = _unsampled(first)[0]
    with restored(session.record, t):
        # Linear from 0 to 2 over the first second, offset by nothing.
        assert dot.get_x() == pytest.approx(2 * float(t))
    assert cell.cell_contents == 1.5
    t = _unsampled(second)[0]
    with restored(session.record, t) as scene:
        (tracker,) = [m for m in scene.mobjects if m is not dot]
        assert dot.get_x() == pytest.approx(tracker.get_value() + 1.5)


def test_a_rebound_global_makes_the_play_unrestorable(session_of):
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    first, second = session.record.plays
    t = _unsampled(first)[0]
    with pytest.raises(Unrestorable, match="SHIFT") as raised:
        session.state_at(t)
    assert raised.value.reason == "restore.global_rebound"
    assert isinstance(raised.value, Unavailable)
    with pytest.raises(Unrestorable):
        session.frame(t)
    # The second play began with the global as it is now.
    assert session.state_at(_unsampled(second)[0]).answer == "restored"


def test_a_tracker_outside_the_scene_is_restored_as_the_play_held_it(session_of):
    """The tracker is set to 0.5 before the second play and to 3 after it;
    that play's captures never reach it.
    """
    session = session_of("DetachedTracker", RESTORE_SCENES, frame_rate=15)
    play = session.record.plays[1]
    assert play.route == "state"
    (tracker,) = {
        id(read): read
        for callback in play.admission.callbacks
        for read in callback.proof.reads
        if type(read).__name__ == "ValueTracker"
    }.values()
    assert all(
        session.record.node_state(tracker, capture) is None for capture in play.samples
    )
    for t in _unsampled(play):
        assert session.state_at(t).named("number").text == "0.50"


def test_a_suspended_traced_path_keeps_its_time_while_restored(session_of):
    session = session_of("SuspendedTrace", RESTORE_SCENES, frame_rate=15)
    routes = [play.route for play in session.record.plays]
    assert routes == ["ticked", "state", "ticked"]
    middle = session.record.plays[1]
    t = _unsampled(middle)[0]
    oracle = session_of("SuspendedTrace", RESTORE_SCENES, frame_rate=15, instants=[t])
    forward = oracle.record.plays[1].sample_at(t)
    expected = oracle.record.node_state(oracle.scene.path, forward).get("time")
    live = session.scene.path.time
    assert live > expected
    with restored(session.record, t) as scene:
        assert scene.path.time == expected
    assert session.scene.path.time == live
    _assert_restored_equals_forward(session, oracle, t)


def test_a_frame_hooks_writes_are_restored_from_the_latest_capture(session_of):
    """The hook counts frames; between two frames a forward run shows the
    count of the frames so far, which the play's start capture does not hold.
    """
    session = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    assert play.frame_hooks == ("update_to_time",)
    t = play.start + Fraction(1, 2) + Fraction(1, 60)
    frames_so_far = len([k for k in range(16) if Fraction(k, 15) <= t])
    assert session.state_at(t).named("self.count").text == str(frames_so_far)
    start = session.state_at(play.start).named("self.count").text
    assert start != str(frames_so_far)


def test_a_binding_a_frame_hook_rebinds_makes_the_play_unrestorable(session_of):
    session = session_of("HookRebinds", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    with pytest.raises(Unrestorable) as raised:
        session.state_at(_unsampled(play)[0])
    assert raised.value.reason == "restore.rebound_in_play"


def test_restores_leave_every_answer_and_the_live_scene_as_they_were(session_of):
    session = session_of("Readouts", ROUTE_SCENES, frame_rate=15)
    scene = session.scene
    sampled = [Fraction(k, 15) for k in range(0, 60, 4)]
    states = [session.state_at(t) for t in sampled]
    frames = [np.asarray(session.frame(t)) for t in sampled]
    lists = (list(scene.mobjects), list(scene.foreground_mobjects))
    clock = scene.renderer.clock
    keys = [play.sample_keys() for play in session.record.plays]
    instants = [t for play in _state_plays(session) for t in _unsampled(play)]
    first = [session.state_at(t) for t in instants]
    for t in instants:
        session.frame(t)
    assert [session.state_at(t) for t in instants] == first
    assert [session.state_at(t) for t in sampled] == states
    for t, frame in zip(sampled, frames, strict=True):
        assert np.array_equal(np.asarray(session.frame(t)), frame), t
    assert (list(scene.mobjects), list(scene.foreground_mobjects)) == lists
    assert scene.renderer.clock == clock
    assert [play.sample_keys() for play in session.record.plays] == keys


def test_a_restored_frame_reads_no_kept_static_picture(session_of):
    """It draws the play's static mobjects afresh, into the pixels a forward
    run's frame there has.
    """
    session = session_of("DetachedTracker", RESTORE_SCENES, frame_rate=15)
    play = session.record.plays[2]
    assert play.static
    t = _unsampled(play)[0]
    session.frame(t)
    assert session._static is None
    oracle = session_of("DetachedTracker", RESTORE_SCENES, frame_rate=15, instants=[t])
    assert np.array_equal(np.asarray(session.frame(t)), np.asarray(oracle.frame(t)))
    # A recorded frame of the play keeps its static picture for the next.
    session.frame(play.start)
    assert session._static is not None


def test_a_state_play_keeps_a_restore_point_at_every_kept_sample(session_of):
    session = session_of("Readouts", ROUTE_SCENES, frame_rate=15)
    for play in _state_plays(session):
        assert sorted(play._animation_states) == [c.seq for c in play.samples]
        points = list(play._animation_states.values())
        assert all(isinstance(point, RestorePoint) for point in points)
        animations = [a for a, _ in snapshot_animations(play._animations)]
        assert all([a for a, _ in p.animations] == animations for p in points)


def test_extra_instants_are_non_frame_evaluations_of_trusted_and_state_plays(
    session_of,
):
    """A ticked play evaluates none; the frame hook runs at the frames only."""
    plain = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15)
    extra = [Fraction(1, 7), Fraction(3, 7)]
    asked = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15, instants=extra)
    (play,) = asked.record.plays
    assert all(play.sample_at(t) is not None for t in extra)
    assert asked.scene.count.number == plain.scene.count.number
    for k in range(15):
        t = Fraction(k, 15)
        assert _canonical(asked.state_at(t)) == _canonical(plain.state_at(t)), k

    trusted = session_of(
        "TrustedTransform", ROUTE_SCENES, frame_rate=15, instants=extra
    )
    assert trusted.record.plays[0].route == "trusted"
    assert trusted.state_at(extra[0]).answer == "recorded"

    ticked = session_of("Conditional", ROUTE_SCENES, frame_rate=15, instants=extra)
    (play,) = ticked.record.plays
    assert play.route == "ticked"
    assert all(play.sample_at(t) is None for t in extra)


def test_a_ticked_play_between_kept_ticks_answers_the_frame_on_screen(session_of):
    """At 15 fps the record keeps one tick in four; between them the answer
    is the last kept one, the frame a render shows there.
    """
    session = session_of("Conditional", ROUTE_SCENES, frame_rate=15)
    _, frames = render(route_scenes.Conditional, 15)
    (play,) = session.record.plays
    assert play.route == "ticked"
    assert not play.events
    for k in range(len(frames) - 1):
        frame_instant = Fraction(k, 15)
        for t in (frame_instant + Fraction(1, 60), frame_instant + Fraction(1, 31)):
            state = session.state_at(t)
            assert (state.answer, state.as_of) == ("recorded", frame_instant), t
            assert state.nodes == session.state_at(frame_instant).nodes
            assert np.array_equal(np.asarray(session.frame(t)), frames[k]), t


def test_a_ticked_answer_names_the_capture_it_reads(session_of):
    session = session_of("Conditional", ROUTE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    t = Fraction(1, 120)
    assert play.sample_at(t) is None
    assert session.state_at(t).as_of == play.base_at(t).instant == 0


def test_a_ticked_record_keeps_no_event_tick_so_an_answer_reads_the_frame(
    session_of,
):
    """The event tick at 0.65 s lies between the frames at 0.6 and 2/3 s. It
    is a full tick, but no frame reads it and the record keeps none: after
    it, the answer is the frame at 0.6 s.
    """
    session = session_of("TickedSuccession", RESTORE_SCENES, frame_rate=15)
    _, frames = render(restore_scenes.TickedSuccession, 15)
    (play,) = session.record.plays
    assert play.route == "ticked"
    event = Fraction(13, 20)
    assert event in play.events
    assert play.sample_at(event) is None
    t = event + Fraction(1, 200)
    state = session.state_at(t)
    assert state.as_of == Fraction(9, 15)
    assert state.nodes == session.state_at(Fraction(9, 15)).nodes
    assert np.array_equal(np.asarray(session.frame(t)), frames[9])


def test_a_ticked_play_before_its_first_frame_answers_the_frame_before(session_of):
    """The ticked play begins at 0.97 s, after the frame at 14/15 s and
    before the next: until that one, the frame on screen is the wait's.
    """
    session = session_of("TickedOffGrid", RESTORE_SCENES, frame_rate=15)
    _, frames = render(restore_scenes.TickedOffGrid, 15)
    wait, ticked = session.record.plays
    assert ticked.route == "ticked"
    assert ticked.start == Fraction(97, 100)
    t = Fraction(98, 100)
    state = session.state_at(t)
    assert (state.answer, state.play, state.as_of) == ("recorded", 0, wait.start)
    assert np.array_equal(np.asarray(session.frame(t)), frames[14])


def test_a_restore_never_runs_a_callback_admission_did_not_prove(session_of):
    """The hook attaches its counter at the first frame: a restore refuses,
    and the counter never runs.
    """
    session = session_of("HookAddsUpdater", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    before = session.scene.calls
    with pytest.raises(Unrestorable) as raised:
        session.state_at(Fraction(3, 7))
    assert raised.value.reason == "restore.unadmitted_callback"
    with pytest.raises(Unrestorable):
        session.frame(Fraction(3, 7))
    assert session.scene.calls == before


def test_a_binding_changed_in_place_during_the_play_is_unrestorable(session_of):
    """The hook changes the list at frames and the scene puts it back after
    the play: what the list holds now is no evidence of what it held there.
    """
    session = session_of("HookMutates", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    with pytest.raises(Unrestorable) as raised:
        session.state_at(Fraction(31, 60))
    assert raised.value.reason == "restore.rebound_in_play"


def test_a_failing_restore_puts_everything_back(session_of):
    session = session_of("FailingUpdater", RESTORE_SCENES, frame_rate=15)
    scene = session.scene
    (play,) = session.record.plays
    assert play.route == "state"
    point = play._animation_states[play.samples[0].seq]
    mobjects = {id(m): m for m in session.record.mobjects()}
    mobjects.update((id(m), m) for m, _ in point.nodes)
    states = {key: NodeState.of(m) for key, m in mobjects.items()}
    names = ("mobjects", "foreground_mobjects", "moving_mobjects", "animations")
    lists = [getattr(scene, name) for name in names]
    clock = scene.time
    tagging = (ORIGINS.active, ORIGINS.base_dir, dict(ORIGINS._ordinals))
    updater = scene.dot.updaters[0]
    cells = [cell.cell_contents for cell in updater.__closure__]
    with pytest.raises(ZeroDivisionError):
        session.state_at(Fraction(1, 7))
    assert scene.time == clock
    assert [cell.cell_contents for cell in updater.__closure__] == cells
    assert all(getattr(scene, n) is v for n, v in zip(names, lists, strict=True))
    assert all(states[key].matches(m.__dict__) for key, m in mobjects.items())
    assert (ORIGINS.active, ORIGINS.base_dir, dict(ORIGINS._ordinals)) == tagging
    assert session.state_at(Fraction(3, 7)).answer == "restored"


def test_the_restored_scene_holds_the_instant_through_the_block(session_of):
    session = session_of("FailingUpdater", RESTORE_SCENES, frame_rate=15)
    updater = session.scene.dot.updaters[0]
    cell = updater.__closure__[updater.__code__.co_freevars.index("offset")]
    with restored(session.record, Fraction(3, 7)) as scene:
        assert scene.time == Fraction(3, 7)
        assert cell.cell_contents == 0.0
        assert scene.dot.get_x() == pytest.approx(0.5)
    assert scene.time == session.record.duration
    assert cell.cell_contents == 1.5


def test_a_restored_frame_draws_no_static_layer(session_of, monkeypatch):
    """The dot stands still at every frame, as anything a static layer kept
    from the captures would; restored between frames it has moved, and the
    frame shows it there, over the square a render draws under it.
    """
    session = session_of("FramePeriodic", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    assert play.static
    t = Fraction(1, 2) + Fraction(1, 60)
    oracle = session_of("FramePeriodic", RESTORE_SCENES, frame_rate=15, instants=[t])
    expected = np.asarray(oracle.frame(t))

    def refused(self, play):
        raise AssertionError("a restored frame drew a static picture")

    monkeypatch.setattr(Drawing, "static", refused)
    pixels = np.asarray(session.frame(t))
    monkeypatch.undo()
    assert np.array_equal(pixels, expected)
    assert not np.array_equal(pixels, np.asarray(session.frame(Fraction(7, 15))))
    (dot,) = [n for n in session.state_at(t).nodes if n.type == "Dot"]
    assert dot.box[1] > 0
