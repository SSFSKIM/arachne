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

from arachne import tempconfig
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
    # The restore itself is under test, so no probe demotes the play first
    # (a frame hook writing what its frames show is demoted, test_probe.py).
    session = session_of(name, path, frame_rate=fps, probe=False)
    instants = [t for play in _state_plays(session) for t in _unsampled(play)]
    assert instants
    oracle = session_of(name, path, frame_rate=fps, instants=instants, probe=False)
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
    # Unprobed: the probe demotes the play (test_probe.py).
    session = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15, probe=False)
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
    # Unprobed: the probe demotes the play (test_probe.py).
    plain = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15, probe=False)
    extra = [Fraction(1, 7), Fraction(3, 7)]
    asked = session_of(
        "FrameCounter", RESTORE_SCENES, frame_rate=15, instants=extra, probe=False
    )
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


def test_a_ticked_play_before_its_first_frame_skips_a_capture_no_frame_drew(
    session_of,
):
    """Between the frame at 14/15 s and the ticked play's first frame lies a
    trusted play too short for any frame: its capture is the latest kept,
    but no frame drew it, so the answer is the wait's, the frame a render
    shows there.
    """
    session = session_of("FramelessBeforeTicked", RESTORE_SCENES, frame_rate=15)
    _, frames = render(restore_scenes.FramelessBeforeTicked, 15)
    wait, trusted, ticked = session.record.plays
    assert [p.route for p in session.record.plays] == ["trusted", "trusted", "ticked"]
    assert trusted.start == Fraction(97, 100)
    assert ticked.start < Fraction(1)
    t = Fraction(98, 100)
    state = session.state_at(t)
    assert (state.answer, state.play, state.as_of) == ("recorded", 0, wait.start)
    assert np.array_equal(np.asarray(session.frame(t)), frames[14])


def test_a_gridless_ticked_answer_before_its_first_tick_evaluates_the_frame(
    session_of,
):
    """Without a grid every 60 Hz tick is a frame, but a trusted play with
    no frame hook keeps only its start: before the ticked play's first tick
    the answer is the trusted play evaluated at the last tick, 29/30 s, as
    a 60 fps render draws it, not its capture at the start.
    """
    session = session_of("TrustedBeforeTicked", RESTORE_SCENES, grid=None)
    oracle = session_of("TrustedBeforeTicked", RESTORE_SCENES, frame_rate=60)
    trusted, ticked = session.record.plays
    assert (trusted.route, ticked.route) == ("trusted", "ticked")
    assert ticked.start == Fraction(97, 100)
    frame, t = Fraction(29, 30), Fraction(39, 40)
    state = session.state_at(t)
    assert (state.play, state.as_of) == (0, frame)
    assert _canonical(state) == _canonical(oracle.state_at(frame))
    (dot,) = [n for n in state.nodes if n.box[1] > -1]
    assert dot.box[0] + 0.08 == pytest.approx(2 * float(frame) / 0.97)
    assert np.array_equal(np.asarray(session.frame(t)), np.asarray(oracle.frame(frame)))


def test_a_restore_refuses_a_callback_animation_pointed_elsewhere(session_of):
    """The hook points the alpha function at a dot outside the scene: a
    restore refuses, and that dot keeps where the scene left it.
    """
    session = session_of("HookRetargets", RESTORE_SCENES, frame_rate=15)
    (play,) = session.record.plays
    assert play.route == "state"
    before = session.scene.other.get_x()
    with pytest.raises(Unrestorable) as raised:
        session.state_at(Fraction(3, 7))
    assert raised.value.reason == "restore.unadmitted_callback"
    assert session.scene.other.get_x() == before


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
    # Unprobed: the probe sees the hook's change and demotes the play.
    session = session_of("HookMutates", RESTORE_SCENES, frame_rate=15, probe=False)
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


@pytest.mark.parametrize("static_layer", [True, False])
def test_a_restored_frame_shows_a_static_mobject_where_its_driver_put_it(
    session_of, static_layer
):
    """The square is one of the play's static mobjects, and the driver after
    it moves it: a restored frame draws it at its state at the instant, as
    the forward run's frame there does with the static layer on or off.
    """
    layer = {"static_layer": static_layer}
    session = session_of("DrivenSquare", RESTORE_SCENES, frame_rate=15, **layer)
    (play,) = _state_plays(session)
    assert play.static
    instants = _unsampled(play)
    oracle = session_of(
        "DrivenSquare", RESTORE_SCENES, frame_rate=15, instants=instants, **layer
    )

    def square_x(state) -> float:
        (square,) = [n for n in state.nodes if n.type == "Square"]
        return square.box[0]

    start = square_x(session.state_at(play.start))
    for t in instants:
        state = session.state_at(t)
        assert state.answer == "restored", t
        assert square_x(state) > start, t
        expected = np.asarray(oracle.frame(t))
        assert np.array_equal(np.asarray(session.frame(t)), expected), t


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_session_s_frames_do_not_depend_on_the_process_s_configuration(
    session_of, draw_path
):
    """A camera draws a point cloud at a thickness it reads from the
    configured frame size: a session's frames, recorded, restored and
    analytic, are drawn under the configuration it was opened with,
    whatever the process holds then.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    session = session_of(
        "CloudPlays",
        RESTORE_SCENES,
        frame_rate=15,
        pixel_width=160,
        pixel_height=90,
        draw_path=draw_path,
    )
    driven, moved = session.record.plays
    assert (driven.route, moved.route) == ("state", "trusted")
    asked = {
        "recorded": driven.start,
        "restored": _unsampled(driven)[0],
        "analytic": _unsampled(moved)[0],
    }
    for kind, t in asked.items():
        assert session.state_at(t).answer == kind, t
        drawn = []
        for size in ((1920, 1080), (160, 90)):
            with tempconfig({"pixel_width": size[0], "pixel_height": size[1]}):
                drawn.append(np.asarray(session.frame(t)))
        assert np.array_equal(*drawn), kind


def test_a_restored_frame_on_a_described_path_is_drawn_from_its_description(
    session_of,
):
    """On ``ir_cairo`` a restored frame is described in XF and drawn from that
    description, as the session's other frames are, and equals the forward
    run's frame there.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.painter import IrCairoPainter

    path = {"draw_path": "ir_cairo"}
    session = session_of("DrivenSquare", RESTORE_SCENES, frame_rate=15, **path)
    (play,) = _state_plays(session)
    instants = _unsampled(play)
    oracle = session_of(
        "DrivenSquare", RESTORE_SCENES, frame_rate=15, instants=instants, **path
    )
    for t in instants:
        assert session.state_at(t).answer == "restored", t
        drawn = IrCairoPainter.drawn_total
        pixels = np.asarray(session.frame(t))
        assert IrCairoPainter.drawn_total == drawn + 1, t
        assert np.array_equal(pixels, np.asarray(oracle.frame(t))), t


@pytest.mark.parametrize(
    ("name", "path", "kind"),
    [
        ("DrivenSquare", RESTORE_SCENES, "restored"),
        ("TrustedTransform", ROUTE_SCENES, "analytic"),
    ],
)
def test_an_evaluated_frame_is_described_at_its_instant(
    session_of, monkeypatch, name, path, kind
):
    """A frame evaluated between captures (restored, or a trusted play's
    analytic evaluation) is described at the instant evaluated, as the
    forward run labels its frame there, not at the capture it began from.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.painter import IrCairoPainter

    labels: list[Fraction] = []
    original = IrCairoPainter.draw

    def draw(self, delta):
        labels.append(delta.to_core().contents()["t"])
        return original(self, delta)

    monkeypatch.setattr(IrCairoPainter, "draw", draw)
    described = {"draw_path": "ir_cairo"}
    session = session_of(name, path, frame_rate=15, **described)
    play = session.record.plays[0]
    instants = _unsampled(play)
    oracle = session_of(name, path, frame_rate=15, instants=instants, **described)
    for t in instants:
        assert session.state_at(t).answer == kind, t
        found = []
        for answering in (session, oracle):
            labels.clear()
            answering.frame(t)
            found.append(labels[-1])
        assert found[0] == found[1] == (t.numerator, t.denominator), t


def _kept_by(painter) -> tuple[int, ...]:
    """What a painter keeps across frames: its stream's mobjects and
    extractions, the records it holds, and the frames it described.
    """
    stream = painter.stream
    return (
        len(stream._slots),
        len(stream._cache),
        len(stream._held),
        len(painter.state.records),
        len(painter.state.shapes),
        stream._seq,
        painter.frames,
    )


def test_restores_leave_nothing_in_the_record_s_stream(session_of):
    """What a restore builds (``always_redraw``'s new line at each restore)
    and the state it reaches belong to no other frame: however often a
    session restores, the record's painter keeps what its own frames left.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.painter import painter_for

    session = session_of(
        "DetachedTracker", RESTORE_SCENES, frame_rate=15, draw_path="ir_cairo"
    )
    play = _state_plays(session)[0]
    t = _unsampled(play)[0]
    session.frame(play.start)
    painter = painter_for(session.record, "ir_cairo")
    assert painter is not None
    kept = _kept_by(painter)
    assert kept[0] > 0
    for _ in range(11):
        assert session.state_at(t).answer == "restored"
        session.frame(t)
    assert _kept_by(painter) == kept


def test_a_frame_after_a_restore_is_described_as_without_it(session_of):
    """The frame drawn after restores is the delta, and the pixels, the same
    frame has in a session that restored nothing: nothing of a restore
    reaches what later frames are described against.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.painter import painter_for

    drawn = []
    # Each session drawn before the next opens: an execution's objects are
    # named by the origins of the execution that ran last.
    for restoring in (True, False):
        session = session_of(
            "DetachedTracker", RESTORE_SCENES, frame_rate=15, draw_path="ir_cairo"
        )
        play = _state_plays(session)[0]
        later = play.start + Fraction(4, 15)
        assert session.state_at(later).answer == "recorded"
        session.frame(play.start)
        if restoring:
            for t in _unsampled(play):
                session.frame(t)
        pixels = np.asarray(session.frame(later))
        painter = painter_for(session.record, "ir_cairo")
        drawn.append((pixels, painter.last.to_core().to_bytes()))
    (restored_pixels, restored_delta), (plain_pixels, plain_delta) = drawn
    assert np.array_equal(restored_pixels, plain_pixels)
    assert restored_delta == plain_delta


def test_a_restored_frame_asked_first_maps_around_the_render_s_centre(session_of):
    """A restored frame is drawn with the centre the render froze at its
    first frame, whether it is the first frame the session draws or not.
    """
    first = session_of("CentreMovedRestored")
    (play,) = _state_plays(first)
    t = _unsampled(play)[1]
    asked_first = np.asarray(first.frame(t))
    second = session_of("CentreMovedRestored")
    second.frame(0)
    assert np.array_equal(asked_first, np.asarray(second.frame(t)))
