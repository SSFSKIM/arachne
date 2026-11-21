"""Worlds in a scene: stepped on the 60 Hz tick whatever the frame rate, each
stay kept as one occurrence of the record's bakes with every tick of it, an
open world's callbacks called once per new tick with the state they are
handed.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arachne_sim")

from arachne.lint.raised import RaisedFinding  # noqa: E402
from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402

from . import world_scenes  # noqa: E402

SCENES = Path(world_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, *, fps: int = 15, **options):
        session = Session.open(
            SCENES, name, config={**SMALL, "frame_rate": fps}, **options
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _occurrences(session):
    bakes = session.record.bakes
    return bakes, bakes.occurrences


def _centre(state, node_id):
    box = state.node(node_id).scene_box
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def test_every_rate_bakes_the_same_worlds_and_calls_once_per_tick(opened):
    """A closed world added before an open one whose force reads it: at 4,
    15, 24, 30 and 60 fps one content key per world, one declaration, the
    same callback calls (one per new tick) and the same bodies where the
    grids meet: the start, a frame inside the first play (1 s), the end.
    """
    found = {}
    for fps in (4, 15, 24, 30, 60):
        session = opened("ClosedThenOpen", fps=fps)
        bakes, occurrences = _occurrences(session)
        calls = list(session.module.CALLS)
        states = {t: session.state_at(t) for t in (0, 1, 2)}
        found[fps] = (
            [
                (o.content, o.declaration, o.closed, o.first_tick, o.last_tick)
                for o in occurrences
            ],
            calls,
            {
                t: [(n.id, n.scene_box) for n in state.nodes]
                for t, state in states.items()
            },
        )
    first = found[4]
    assert len(first[0]) == 2
    assert [closed for _, _, closed, _, _ in first[0]] == [True, False]
    assert all(o[3:] == (0, 120) for o in first[0])
    # One call per new tick: ticks 1 to 120 step from the state at 0 to 119.
    assert first[1] == list(range(120))
    for fps in (15, 24, 30, 60):
        assert found[fps] == first, fps


def test_a_frame_between_ticks_shows_the_last_tick_before_it(opened):
    """At 24 fps every other frame falls between two ticks: 25/24 s is tick
    62.5, and shows tick 62.
    """
    session = opened("ClosedThenOpen", fps=24)
    bakes, (closed, _) = _occurrences(session)
    pose = bakes.contents[closed.content].channels["pose"].values
    ball = [b for b in closed.bindings if b.channel == "pose"][1]
    centre = _centre(session.state_at(Fraction(25, 24)), ball.id)
    assert centre == pytest.approx(pose[62, 1, :2].astype(float), abs=1e-5)
    assert centre != pytest.approx(pose[63, 1, :2].astype(float), abs=1e-5)


def test_a_render_s_occurrence_holds_every_tick(opened):
    """At 15 fps the record keeps a frame every fourth tick, and the bake
    every tick: a second is 60 rows, and a session that keeps every tick
    captures each row's placement.
    """
    rendered = opened("Falling", fps=15)
    bakes, (occurrence,) = _occurrences(rendered)
    content = bakes.contents[occurrence.content]
    assert (occurrence.first_tick, occurrence.last_tick) == (0, 60)
    assert content.ticks == 61
    pose = content.channels["pose"].values
    # The point falls from (0, 2) at 1 u/s across, under 9.8 down.
    assert pose[0, 0, :2].tolist() == [0.0, 2.0]
    assert pose[60, 0, 0] == pytest.approx(1.0, abs=1e-4)
    assert pose[60, 0, 1] == pytest.approx(2 - 9.8 / 2, abs=0.02)
    every_tick = opened("Falling", fps=15, grid=None)
    tick_bakes, (tick_occurrence,) = _occurrences(every_tick)
    assert tick_occurrence.content == occurrence.content
    (binding,) = [b for b in tick_occurrence.bindings if b.channel == "pose"]
    for tick in range(61):
        state = every_tick.state_at(Fraction(tick, 60))
        assert _centre(state, binding.id) == pytest.approx(
            pose[tick, 0, :2].astype(float), abs=1e-6
        ), tick


def test_a_wait_over_a_closed_world_moves(opened):
    session = opened("Falling")
    _, (occurrence,) = _occurrences(session)
    body = occurrence.bindings[0].id
    start = _centre(session.state_at(0), body)
    end = _centre(session.state_at(1), body)
    assert end[1] < start[1] - 4
    assert session.record.plays[0].route == "ticked"
    assert not session.record.plays[0].held


def test_a_play_s_start_its_entry_and_its_cleanup_add_no_row_and_no_call(opened):
    """The updater runs at each play's start with dt 0, for the entry state
    and after each play: none of these is a new tick.
    """
    session = opened("Counted")
    bakes, (occurrence,) = _occurrences(session)
    assert list(range(45)) == session.module.CALLS
    assert bakes.contents[occurrence.content].ticks == 46
    assert not occurrence.closed
    velocity = bakes.contents[occurrence.content].channels["velocity"].values
    # One unit of force on two of mass for 45 ticks.
    assert velocity[45, 0, 0] == pytest.approx(45 / 60 / 2, rel=1e-5)


def test_a_full_tick_run_and_a_stepping_run_key_every_sample_alike(opened):
    stepping = opened("ClosedThenOpen", stepping=True)
    full = opened("ClosedThenOpen", stepping=False)
    for one, other in zip(stepping.record.plays, full.record.plays, strict=True):
        assert one.sample_keys() == other.sample_keys()
    assert sum(p.stepping_ticks for p in stepping.record.plays) > 0
    assert sum(p.stepping_ticks for p in full.record.plays) == 0


def test_two_worlds_declared_alike_share_one_content(opened):
    session = opened("TwoAlike")
    bakes, occurrences = _occurrences(session)
    assert len(occurrences) == 2
    assert len(bakes.contents) == 1
    one, other = occurrences
    assert one.content == other.content
    assert one.declaration == other.declaration
    assert {b.id for b in one.bindings}.isdisjoint({b.id for b in other.bindings})


def test_a_world_away_and_back_has_two_stays(opened):
    session = opened("AwayAndBack")
    bakes, occurrences = _occurrences(session)
    first, second = occurrences
    # Removed at 0.51 s: its last tick is 30 (0.5 s). Back at 1.01 s: tick 60.
    assert (first.first_tick, first.last_tick) == (0, 30)
    assert (second.first_tick, second.last_tick) == (60, 90)
    assert bakes.row(first.ordinal, 30) is not None
    assert bakes.at(first.ordinal, Fraction(51, 100)) is not None
    assert bakes.row(first.ordinal, 31) is None
    assert bakes.at(first.ordinal, Fraction(3, 4)) is None
    assert bakes.at(second.ordinal, Fraction(3, 4)) is None
    left, back = bakes.row(first.ordinal, 30), bakes.row(second.ordinal, 60)
    assert back is not None
    assert back.index == 0
    # The second stay begins from the state the first left the world in.
    assert back["pose"].tolist() == left["pose"].tolist()
    assert back["velocity"].tolist() == left["velocity"].tolist()


def test_a_force_law_declared_between_plays_acts_from_the_next_tick(opened):
    session = opened("Steered")
    bakes, (occurrence,) = _occurrences(session)
    velocity = bakes.contents[occurrence.content].channels["velocity"].values
    assert velocity[30, 0, 0] == 0.0
    assert velocity[31, 0, 0] == pytest.approx(2 / 60, rel=1e-5)
    assert velocity[60, 0, 0] == pytest.approx(1.0, rel=1e-5)


def test_a_body_declared_into_a_world_in_the_scene_is_refused(opened):
    with pytest.raises(ValueError, match="before the world first enters the scene"):
        opened("BodyTooLate")


def test_a_mobject_read_through_an_object_lags_the_tick(opened):
    """What the check cannot see is not supported: a body's mobject, reached
    through another object, shows the last frame between frames, while the
    state the world hands is the tick's, the same at every frame rate.
    """
    seen = {}
    for fps in (15, 60):
        seen[fps] = list(opened("ReadThroughAnObject", fps=fps).module.READS)
    assert [(t, h) for t, h, _ in seen[15]] == [(t, h) for t, h, _ in seen[60]]
    drawn = {fps: [d for _, _, d in reads] for fps, reads in seen.items()}
    assert drawn[15] != drawn[60]
    lagging = [(t, h, d) for t, h, d in seen[15] if not math.isclose(h, d)]
    assert lagging


def test_a_body_in_a_3d_world_moves_rigidly(opened):
    session = opened("Tumbling", grid=None)
    bakes, (occurrence,) = _occurrences(session)
    pose = bakes.contents[occurrence.content].channels["pose"].values
    world = next(
        m
        for m in session.record.plays[-1].samples[-1].roots
        if type(m).__name__ == "World"
    )
    (mob,) = world.submobjects
    points = np.concatenate([m.points for m in mob.family_members_with_points()])
    assert points.mean(axis=0) == pytest.approx(pose[60, 0, :3].astype(float), abs=1e-5)
    # Rigid: every point keeps its distance from the centre (half a diagonal of
    # a unit cube's face points, at most).
    distances = np.linalg.norm(points - pose[60, 0, :3].astype(float), axis=1)
    assert distances.max() <= math.sqrt(3) / 2 + 1e-5
    assert not np.allclose(pose[60, 0, 3:], pose[0, 0, 3:])


def _spans(session):
    return [(o.first_tick, o.last_tick) for o in session.record.bakes.occurrences]


def test_a_remove_and_an_add_with_no_time_between_are_one_stay(opened):
    """Between plays no scene time passes: the world never left. Bringing it
    to the front or the back of the scene is no leave either.
    """
    assert _spans(opened("Blinked")) == [(0, 60)]


@pytest.mark.parametrize("fps", [15, 60])
def test_a_world_removed_between_frames_ends_at_that_tick(opened, fps):
    """Removed at tick 19, which no 15 fps frame reads: the closed world is
    computed through it, at every rate alike.
    """
    session = opened("LeftBetweenFrames", fps=fps)
    bakes, (occurrence,) = _occurrences(session)
    assert (occurrence.first_tick, occurrence.last_tick) == (0, 19)
    pose = bakes.contents[occurrence.content].channels["pose"].values
    assert pose[19, 0, 0] == pytest.approx(19 / 60, rel=1e-5)


def test_a_world_away_is_never_computed_on(opened):
    session = opened("ReadWhileAway")
    found = session.module.FOUND
    assert "reached" in found["ahead"]
    assert "in the scene" in found["away"]
    assert _spans(session) == [(0, 30)]
    bakes, (occurrence,) = _occurrences(session)
    assert bakes.contents[occurrence.content].ticks == 31
    assert bakes.row(occurrence.ordinal, 31) is None


@pytest.mark.parametrize(
    ("name", "spans"),
    [
        ("Grouped", [(0, 30), (60, 90)]),
        ("Replaced", [(0, 30), (60, 90)]),
        ("Assigned", [(0, 30)]),
    ],
)
def test_every_way_into_and_out_of_the_scene_makes_the_stays(opened, name, spans):
    assert _spans(opened(name)) == spans


def test_a_world_leaving_by_an_unknown_path_mid_play_is_refused(opened):
    with pytest.raises(ValueError, match="instant it did is not"):
        opened("AssignedDuringAPlay")


def test_an_animation_s_copies_of_a_world_are_inert(opened):
    """FadeIn copies the worlds it fades and runs the copies' updaters: they
    begin no stay and call no callback.
    """
    session = opened("FadedIn")
    assert _spans(session) == [(0, 60), (0, 60)]
    assert list(range(60)) == session.module.CALLS


def test_resuming_updating_calls_no_callback(opened):
    session = opened("Resumed")
    assert list(range(30)) == session.module.CALLS
    assert _spans(session) == [(0, 30)]


def test_a_change_mid_play_takes_effect_at_its_tick_at_every_rate(opened):
    """A force law added by a ``dt`` updater at tick 19 acts from tick 20,
    whether or not a frame reads tick 19.
    """
    found = []
    for fps in (15, 60):
        session = opened("SteeredMidPlay", fps=fps)
        bakes, (occurrence,) = _occurrences(session)
        velocity = bakes.contents[occurrence.content].channels["velocity"].values
        assert velocity[19, 0, 0] == 0.0
        assert velocity[20, 0, 0] == pytest.approx(3 / 60, rel=1e-5)
        found.append((occurrence.content, occurrence.declaration))
    assert found[0] == found[1]


def test_a_jointed_world_away_and_back_moves_as_if_it_stayed(opened):
    """A pin with limits and a slide with a motor keep their declared frames
    across a stay: the world that was away from 0.5 s to 1 s moves on from
    where it left as the world that stayed moved on from there.
    """
    session = opened("Hinged")
    bakes, (steady, left, back) = _occurrences(session)
    assert [(o.first_tick, o.last_tick) for o in (steady, left, back)] == [
        (0, 90),
        (0, 30),
        (60, 90),
    ]
    channels = {o.ordinal: bakes.contents[o.content].channels for o in (steady, back)}
    for channel in ("pose", "velocity"):
        assert (
            channels[back.ordinal][channel].values.tobytes()
            == channels[steady.ordinal][channel].values[30:61].tobytes()
        )


def test_the_declaration_keeps_the_order_across_kinds(opened):
    first = _occurrences(opened("GravityFirst"))[1][0].declaration
    after = _occurrences(opened("BodyFirst"))[1][0].declaration
    assert first != after


@pytest.mark.parametrize(
    ("name", "message"),
    [("PushedOutOfPlane", "plane of a 2D world"), ("PushedNotANumber", "finite")],
)
def test_a_push_out_of_the_plane_is_refused(opened, name, message):
    with pytest.raises(ValueError, match=message):
        opened(name)


def test_a_body_whose_class_overrides_the_family_walk_is_placed_whole(opened):
    session = opened("DrawnByAnOverride")
    bakes, (occurrence,) = _occurrences(session)
    pose = bakes.contents[occurrence.content].channels["pose"].values
    world = next(
        m
        for m in session.record.plays[-1].samples[-1].roots
        if type(m).__name__ == "World"
    )
    (drawn,) = world.submobjects
    assert drawn.points.mean(axis=0)[:2] == pytest.approx(
        pose[30, 0, :2].astype(float), abs=1e-5
    )


def test_a_callback_closing_over_a_world_is_refused():
    from arachne import Dot, World

    world, other = World(), World()
    body = world.body(Dot(), shape=None, mass=1.0)
    there = other.body(Dot(), shape=None, mass=1.0)

    def reads_the_world(state):
        return {body: other.at_tick(state.tick).position(there)}

    with pytest.raises(RaisedFinding) as raised:
        world.add_force(reads_the_world)
    assert raised.value.detail["names"] == ["other"]

    def reads_the_handle(state):
        return {body: state.position(there) - state.position(body)}

    world.add_force(reads_the_handle)


def test_a_callback_that_reads_a_mobject_is_refused_as_declared():
    from arachne import Dot, World

    world = World()
    drawn = Dot()
    body = world.body(drawn, shape=None, mass=1.0)

    def reads_the_mobject(state):
        return {body: drawn.get_center()}

    with pytest.raises(RaisedFinding) as raised:
        world.add_force(reads_the_mobject)
    assert raised.value.code == "sim.force_input"
    assert raised.value.detail["names"] == ["drawn"]

    def reads_the_handle(state):
        return {body: -state.position(body)}

    world.add_force(reads_the_handle)
    assert world.is_open


def test_a_callback_that_reads_a_mapping_keyed_by_a_mobject_is_refused():
    from arachne import Dot, World

    world = World()
    drawn = Dot()
    body = world.body(drawn, shape=None, mass=1.0)
    weights = {drawn: 2.0}

    def reads_the_keys(state):
        (mark,) = weights
        return {body: -mark.get_center()}

    with pytest.raises(RaisedFinding) as raised:
        world.add_force(reads_the_keys)
    assert raised.value.code == "sim.force_input"
    assert raised.value.detail["names"] == ["weights"]

    by_body = {body: 2.0}

    def reads_a_mapping_of_handles(state):
        return {each: -by_body[each] * state.position(each) for each in by_body}

    world.add_force(reads_a_mapping_of_handles)
    assert world.is_open


def test_a_withheld_feature_is_refused_as_declared(monkeypatch):
    from arachne import Circle, World
    from arachne.simulation import inputs

    monkeypatch.setattr(inputs, "WITHHELD", {"shape.ball": "not portable yet"})
    world = World()
    with pytest.raises(RaisedFinding) as raised:
        world.body(Circle())
    assert raised.value.code == "sim.withheld"
    assert raised.value.detail == {"feature": "shape.ball"}
    world.body(Circle(), shape=None, mass=1.0)


def test_a_world_routes_its_integrator_by_what_it_holds():
    from arachne import BallShape, Circle, Dot, PinJoint, World

    points = World()
    a = points.body(Dot(), shape=None, mass=1.0)
    points.body(Dot(RIGHT_UNIT), shape=None, mass=1.0)
    assert points.integrator == "point_masses"
    rigid = World()
    rigid.body(Circle())
    assert rigid.integrator == "solver"
    jointed = World()
    b = jointed.body(Dot(), shape=None, mass=1.0)
    c = jointed.body(Dot(RIGHT_UNIT), shape=BallShape(0.1))
    jointed.add_joint(PinJoint(b, c, at=(0.5, 0, 0)))
    assert jointed.integrator == "solver"
    assert a.index == 0


RIGHT_UNIT = (1.0, 0.0, 0.0)


def test_a_body_whose_class_replaces_its_points_is_refused():
    from arachne import Circle, World

    class Drawn(Circle):
        @property
        def points(self):
            return super().points

        @points.setter
        def points(self, value):
            Circle.points.fset(self, value)

    with pytest.raises(TypeError, match="Drawn replaces them"):
        World().body(Drawn())


def test_a_world_holds_no_callable_of_the_scene():
    """The callback lives apart from the world's mobject, which holds only
    its name: nothing a capture keeps runs the scene's code.
    """
    import dataclasses

    from arachne import Dot, World

    world = World()
    body = world.body(Dot(), shape=None, mass=1.0)
    seen: list[int] = []

    def push(state):
        seen.append(state.tick)
        return {body: (1.0, 0.0)}

    world.add_force(push)

    def callables(value, depth=0):
        if callable(value) and not isinstance(value, type):
            yield value
        if depth > 6:
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                yield from callables(getattr(value, field.name), depth + 1)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from callables(item, depth + 1)

    held = {
        name: list(callables(value))
        for name, value in vars(world).items()
        if name != "updaters"
    }
    assert not any(held.values()), held
    assert [item.name for item in world._declaration.callbacks] == [
        "test_a_world_holds_no_callable_of_the_scene.<locals>.push"
    ]


def _interleaved(ticks):
    """Each tick's callback, handed the tick before, then the tick's scene
    updater.
    """
    return [event for k in ticks for event in (("call", k - 1), ("tick", k))]


def test_a_world_faded_in_by_the_last_play_steps_at_every_tick(opened):
    session = opened("FadedInLast")
    assert _interleaved(range(1, 31)) == session.module.EVENTS
    assert _spans(session) == [(0, 30)]


@pytest.mark.parametrize("fps", [15, 60])
@pytest.mark.parametrize("name", ["FadedInGrouped", "FadedInGroupedQuiet"])
def test_a_world_faded_in_with_a_label_steps_at_every_tick(opened, name, fps):
    """The world's group is the animation's: its callbacks run at the
    animation's ticks as when the world itself is faded in, with a scene
    updater reading dt or without one.
    """
    session = opened(name, fps=fps)
    events = session.module.EVENTS
    if name == "FadedInGrouped":
        assert _interleaved(range(1, 31)) == events
    else:
        assert [("call", k) for k in range(30)] == events
    assert _spans(session) == [(0, 30)]


def test_worlds_faded_in_with_a_label_move_as_their_twins_added_plainly(opened):
    """An open and a closed world faded in under one group with a label,
    beside alike ones added: the same stays and the same rows.
    """
    _, (added, dropped, faded, falling) = _occurrences(opened("GroupedBeside"))
    assert not added.closed
    assert dropped.closed
    for plain, grouped in ((added, faded), (dropped, falling)):
        assert (grouped.first_tick, grouped.last_tick) == (0, 60)
        assert (plain.first_tick, plain.last_tick) == (0, 60)
        assert grouped.content == plain.content


@pytest.mark.parametrize("fps", [15, 60])
def test_a_closed_world_faded_in_with_a_label_last_keeps_the_clock_s_rows(opened, fps):
    """Its placement holds while the group's updating is suspended, as in CE;
    its rows are computed through the scene's end all the same.
    """
    _, (added, faded) = _occurrences(opened("GroupedClosedLast", fps=fps))
    assert faded.closed
    assert (faded.first_tick, faded.last_tick) == (0, 30)
    assert faded.content == added.content


def test_an_open_world_in_a_group_suspended_by_hand_is_refused(opened):
    """No animation of the play holds the group, so the play does not step
    on the tick: the world is refused as the play advances.
    """
    with pytest.raises(ValueError, match="Suspend the world's own updating"):
        opened("SuspendedByHand")


def test_an_open_world_suspended_by_hand_keeps_stepping(opened):
    session = opened("WorldSuspendedByHand")
    assert _interleaved(range(1, 31)) == session.module.EVENTS
    assert _spans(session) == [(0, 30)]


@pytest.mark.parametrize("fps", [15, 60])
@pytest.mark.parametrize(
    ("name", "last"),
    [("RemovedBeforeItsUpdater", 18), ("RemovedAfterItsUpdater", 19)],
)
def test_an_open_world_s_stay_ends_at_the_last_tick_its_callbacks_ran_for(
    opened, name, last, fps
):
    """Removed by a dot's dt updater at tick 19: before the world's own
    updater ran there (the dot first in the scene), the stay ends at 18;
    after it ran, at 19. The order of updaters within a tick does not depend
    on the frame grid, so every rate ends alike.
    """
    session = opened(name, fps=fps)
    assert _spans(session) == [(0, last)]
    assert [("call", k) for k in range(last)] == session.module.EVENTS


def test_a_world_faded_in_moves_as_its_twin_added_plainly(opened):
    session = opened("FadedBeside")
    _, (added, faded) = _occurrences(session)
    assert (faded.first_tick, faded.last_tick) == (added.first_tick, added.last_tick)
    assert faded.content == added.content


def test_a_world_faded_out_calls_nothing_as_it_leaves(opened):
    session = opened("FadedOut")
    assert _interleaved(range(1, 61)) == session.module.EVENTS
    assert _spans(session) == [(0, 60)]


def test_a_world_is_released_with_its_session(opened):
    import gc

    session = Session.open(SCENES, "Dropped", config={**SMALL, "frame_rate": 15})
    world = session.module.SHARED.world
    assert world() is not None
    session.close()
    del session
    gc.collect()
    assert world() is None


def test_a_world_is_released_when_its_scene_raises():
    import gc

    world_scenes.SHARED.world = None
    with pytest.raises(RuntimeError, match="own error"):
        Session.open(SCENES, "RaisesAfterEntering", config={**SMALL, "frame_rate": 15})
    world = world_scenes.SHARED.world
    gc.collect()
    assert world() is None


def test_two_equal_stays_of_an_accelerating_world_declare_apart(opened):
    _, (first, second) = _occurrences(opened("AwayAndBack"))
    assert first.last_tick - first.first_tick == second.last_tick - second.first_tick
    assert first.declaration != second.declaration


def _paired(events):
    """Whether every contact that ends began before, in one occurrence."""
    touching = set()
    for _, a, b, kind in events.tolist():
        if kind:
            touching.add((a, b))
        elif (a, b) not in touching:
            return False
        else:
            touching.discard((a, b))
    return True


def test_a_resting_pair_begins_touching_in_each_stay(opened):
    session = opened("Resting")
    bakes, (first, second) = _occurrences(session)
    for occurrence in (first, second):
        events = bakes.contents[occurrence.content].events["contact"].rows
        assert _paired(events)
    events = bakes.contents[second.content].events["contact"].rows
    assert events[0].tolist() == [0, 0, 1, 1]


def test_a_copy_of_an_open_world_is_refused_and_a_closed_one_runs(opened):
    with pytest.raises(ValueError, match="copy of an open world"):
        opened("CopiedOpen")
    assert _spans(opened("CopiedClosed")) == [(0, 30)]


def test_a_capture_s_key_changes_with_the_world_and_its_bodies(opened):
    """c54's content key reads a world by its declaration's digest and its
    bodies by their placements; the declaration, the bodies' list and the
    occurrence's reference are named as not content.
    """
    from arachne import World

    assert World._content_key_omits == {"_declaration", "_body_mobjects", "_bake"}
    falling, heavier = opened("Falling"), opened("Heavier")
    starts = [s.record.plays[0].start_state.content_key() for s in (falling, heavier)]
    assert starts[0] != starts[1]
    later = opened("AwayAndBack").record.plays
    assert later[0].start_state.content_key() != later[2].start_state.content_key()


def _made_at(marker):
    """Where the scenes' file makes the mobject whose line ends in ``marker``."""
    lines = SCENES.read_text().splitlines()
    (number,) = [i for i, line in enumerate(lines, 1) if line.endswith(marker)]
    return f"{SCENES.name}:{number}"


@pytest.mark.parametrize(
    ("name", "marker", "entered"),
    [
        ("BallShiftedBeforeEntry", "# the ball", "0"),
        ("FloorShiftedBeforeEntry", "# the floor", "0"),
        ("BallShiftedBetweenStays", "# the ball", "0.75"),
        ("FloorShiftedBetweenStays", "# the floor", "0.75"),
        ("HeldShiftedThenBack", "# the held ball", "0.75"),
        ("IntegerPointsMovedBeforeEntry", "# the integer mark", "0"),
    ],
)
def test_a_body_drawn_away_from_where_its_world_left_it_is_refused(
    opened, name, marker, entered
):
    """A moving body would be drawn back at its world's first step, and a
    fixed one would part from what its world simulates: the stay that would
    begin so is refused, naming the body by the line that made it and the
    fix.
    """
    with pytest.raises(ValueError) as raised:
        opened(name)
    message = str(raised.value)
    assert f"made at {_made_at(marker)}" in message
    assert f"as its world enters the scene at {entered} s" in message
    assert "Arrange the drawing first, then declare the body" in message
    assert "between stays, leave a body where the world left it" in message
    if "floor" in marker:
        assert "it is fixed, so its world would simulate it apart from" in message
    else:
        assert "the simulation drives the stay's first placement" in message


@pytest.mark.parametrize(
    "name", ["BallMovedToItsPlace", "BallShiftedAndBack", "FloorShiftedAndBack"]
)
def test_a_body_moved_back_where_its_world_left_it_enters_as_if_unmoved(opened, name):
    """A move to the centre a body has, and a shift undone, leave it where
    its world drew it: both stays begin, and the bodies move as in the twin
    left alone.
    """
    drawn = []
    contents = []
    for session in (opened(name), opened("Rearranged")):
        _, occurrences = _occurrences(session)
        contents.append([occurrence.content for occurrence in occurrences])
        world = next(
            m
            for m in session.record.plays[-1].samples[-1].roots
            if type(m).__name__ == "World"
        )
        drawn.append(
            np.concatenate([m.points for m in world.family_members_with_points()])
        )
    assert len(contents[0]) == 2
    assert contents[0] == contents[1]
    assert drawn[0] == pytest.approx(drawn[1], abs=1e-9)


def test_a_body_held_then_away_is_compared_with_its_last_placement(opened):
    """The ball's world held for a quarter second before it left: the second
    stay begins at a later row than the one the ball was last placed at, and
    the ball, left where it was placed, enters.
    """
    session = opened("HeldThenAwayAndBack")
    bakes, (first, second) = _occurrences(session)
    assert (first.last_tick, second.first_tick) == (30, 45)
    left = bakes.row(first.ordinal, 30)["pose"]
    held = bakes.row(first.ordinal, 15)["pose"]
    assert left[0, 1] < held[0, 1] - 0.1
