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
    15, 30 and 60 fps one content key per world, one declaration, the same
    callback calls (one per new tick) and the same bodies where the grids
    meet.
    """
    found = {}
    for fps in (4, 15, 30, 60):
        session = opened("ClosedThenOpen", fps=fps)
        bakes, occurrences = _occurrences(session)
        calls = list(session.module.CALLS)
        states = {t: session.state_at(t) for t in (0, 1)}
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
    assert all(o[3:] == (0, 90) for o in first[0])
    # One call per new tick: ticks 1 to 90 step from the state at 0 to 89.
    assert first[1] == list(range(90))
    for fps in (15, 30, 60):
        assert found[fps] == first, fps


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
