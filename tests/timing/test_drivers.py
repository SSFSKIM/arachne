"""One clock: a scene that declares timing has one engine updater that runs
last, its clock, which takes over the scene's drivers (a diagram's flows,
and any driver registered by the contract) and its pose settle, and runs
them at the start of each instant, before its hold.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from arachne import Flow
from arachne.assets import timing
from arachne.assets.motion import pose_settle
from arachne.mobject.diagram import flow_driver
from arachne.mobject.marks._paths import ArcLength

from .reading import RATES, boundary, frame_captures
from .timing_scenes import OPEN


def _scene_updaters(session) -> set[tuple]:
    return {
        tuple(capture.scene_updaters)
        for play in session.record.plays
        for capture in play.samples
    }


@pytest.mark.parametrize("fps", RATES)
def test_flows_made_before_and_after_the_declaration_keep_the_scene_s_time(opened, fps):
    session = opened("FlowBesideSteppedBox", fps)
    scene = session.scene
    clock = timing.clock_of(scene)
    driver = flow_driver(scene)
    assert clock is not None
    assert driver is not None
    # The clock alone is among the scene's updaters, and runs the driver.
    assert _scene_updaters(session) == {(clock.updater,)}
    assert clock.drivers == [driver]
    record = session.record
    flows = [m for m in record.mobjects() if isinstance(m, Flow)]
    assert {f.name for f in flows} == {"early", "late"}
    for flow in flows:
        assert driver.holds(flow)
        path = ArcLength(flow.edge.points)
        for t, capture in frame_captures(session, fps):
            token = np.asarray(
                record.node_state(flow.submobjects[0], capture).get("_points")
            ).mean(axis=0)
            along = (flow.offset + flow.speed * (float(t) - flow.start)) % path.length
            assert np.linalg.norm(token - path.point(along)) < 1e-6, (flow.name, t)


@pytest.mark.parametrize("fps", RATES)
def test_a_pose_is_settled_by_the_clock_and_registers_nothing(opened, fps):
    session = opened("PosedBox", fps)
    scene = session.scene
    clock = timing.clock_of(scene)
    settle = pose_settle(scene)
    assert settle is not None
    assert settle.registers is False
    assert _scene_updaters(session) == {(clock.updater,)}


def test_registered_drivers_run_first_in_the_clock_s_instant_in_their_order(opened):
    session = opened("StubDrivers", 60)
    scene = session.scene
    # The session runs the scene file as a module of its own.
    scenes = sys.modules[type(scene).__module__]
    clock = timing.clock_of(scene)
    first, second = (scenes.STUBS[kind][scene] for kind in ("first", "second"))
    # Taken over: neither updater is the scene's, and the clock is last.
    assert _scene_updaters(session) == {(clock.updater,)}
    assert clock.drivers[-2:] == [first, second]
    assert first.held == ["later"]
    seen = scenes.SEEN["StubDrivers"]
    instants = sorted({t for _, t, _ in seen})
    assert len(instants) >= 15
    for t in instants:
        names = [name for name, at, _ in seen if at == t]
        assert names == ["first", "second"], t
    # Each ran before the hold: it saw the pose true at the instant, which
    # the drawing on display shows only from the next boundary.
    for name, t, value in seen:
        assert value == pytest.approx(OPEN * float(t) / 0.25), (name, t)
    held = [t for t in instants if t != boundary(t)]
    assert held


def test_a_scene_without_a_declaration_keeps_its_drivers_as_they_were(opened):
    session = opened("Undeclared", 30)
    scene = session.scene
    assert timing.clock_of(scene) is None
    driver = flow_driver(scene)
    settle = pose_settle(scene)
    assert settle.registers is True
    # The flow driver among the scene's updaters, joined by the pose's
    # settle while it plays.
    assert _scene_updaters(session) == {
        (driver.updater, settle.updater),
        (driver.updater,),
    }


def test_set_timing_outside_a_running_scene_is_a_plain_error():
    from arachne import Square, set_timing

    with pytest.raises(RuntimeError, match="running scene"):
        set_timing(Square(), 12)


@pytest.mark.parametrize("rate", [0, 7, 60, 12.0, True, "12"])
def test_a_rate_that_does_not_divide_60_lists_the_divisors(rate):
    from arachne import Boil, Timing

    with pytest.raises(ValueError, match="1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30"):
        Timing(rate)
    with pytest.raises(ValueError, match="1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30"):
        Boil(rate=rate)


def test_a_timing_is_an_immutable_value():
    from dataclasses import FrozenInstanceError

    from arachne import Boil, Timing

    mark = Timing(12, Boil(6))
    assert mark == Timing(12, Boil(rate=6, amplitude=1.0, seed=0))
    assert hash(mark) == hash(Timing(12, Boil(6)))
    with pytest.raises(FrozenInstanceError):
        mark.drawings_per_second = 6
    for amplitude in (-1, float("nan"), float("inf"), "2"):
        with pytest.raises(ValueError, match="amplitude"):
            Boil(6, amplitude)
    with pytest.raises(ValueError, match="seed"):
        Boil(6, seed=-1)
    with pytest.raises(TypeError):
        Timing(12, boil=6)
