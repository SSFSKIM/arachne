"""A flow's tokens keep the declared speed along their connector, by arc
length, at 15, 30 and 60 frames per second, through waits and a play of
something else.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import RIGHT, Diagram, Flow, Node
from arachne.mobject.diagram import flow_driver
from arachne.session import Session
from arachne.timeline import admission

from . import diagram_scenes
from .diagram_scenes import (
    ELBOW_SPEED,
    MOVING_SPEED,
    MOVING_TOKENS,
    STRAIGHT_SPEED,
    flowing_diagram,
)

SCENES = Path(diagram_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_at(fps: int) -> Session:
        session = Session.open(
            SCENES, "FlowThroughPlaysAndWaits", config={**SMALL, "frame_rate": fps}
        )
        sessions.append(session)
        return session

    yield open_at
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _token(session: Session, t: Fraction, flow: str, index: int = 0):
    state = session.state_at(t)
    token = state.node(state.named(flow).children[index])
    x0, y0, x1, y1 = token.scene_box
    return state.as_of, np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def _polyline(points: np.ndarray, per_curve: int = 2000) -> np.ndarray:
    """The path through a connector's cubic curves, sampled densely: the
    test's own reading of its length.
    """
    curves = points.reshape(-1, 4, 3)
    t = np.linspace(0, 1, per_curve)[:, None]
    samples = [
        (1 - t) ** 3 * c[0]
        + 3 * (1 - t) ** 2 * t * c[1]
        + 3 * (1 - t) * t**2 * c[2]
        + t**3 * c[3]
        for c in curves
    ]
    return np.vstack(samples)[:, :2]


def _along(path: np.ndarray, point: np.ndarray) -> float:
    """How far along ``path`` the point of it nearest ``point`` lies."""
    starts, ends = path[:-1], path[1:]
    course = ends - starts
    lengths = np.linalg.norm(course, axis=1)
    share = np.clip(
        np.einsum("ij,ij->i", point - starts, course) / np.maximum(lengths**2, 1e-18),
        0.0,
        1.0,
    )
    feet = starts + course * share[:, None]
    nearest = int(np.argmin(np.linalg.norm(feet - point, axis=1)))
    before = float(lengths[:nearest].sum())
    return before + float(share[nearest] * lengths[nearest])


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize(
    ("flow", "edge_index", "speed"),
    [("straight_flow", 0, STRAIGHT_SPEED), ("elbow_flow", 1, ELBOW_SPEED)],
)
def test_a_flow_keeps_its_declared_speed(opened, fps, flow, edge_index, speed):
    session = opened(fps)
    diagram, _, _ = flowing_diagram()
    path = _polyline(np.asarray(diagram.edges[edge_index].points))
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    length = float(steps.sum())
    duration = session.record.duration
    assert duration == 3
    readings = [_token(session, Fraction(k, fps), flow) for k in range(3 * fps)]
    # Every frame instant is answered as itself: the flow ticks its plays.
    assert [as_of for as_of, _ in readings] == [
        Fraction(k, fps) for k in range(3 * fps)
    ]
    positions = [_along(path, p) for _, p in readings]
    # Unwrap each pass from the end back to the start.
    unwrapped = [positions[0]]
    for before, after in zip(positions, positions[1:], strict=False):
        step = after - before
        if step < -length / 2:
            step += length
        unwrapped.append(unwrapped[-1] + step)
    times = np.array([float(t) for t, _ in readings])
    measured = np.polyfit(times, np.array(unwrapped), 1)[0]
    assert measured == pytest.approx(speed, rel=0.01)
    # And frame by frame, wherever the token is not rounding a corner.
    per_frame = np.diff(unwrapped) * fps
    assert np.median(per_frame) == pytest.approx(speed, rel=0.01)


def test_the_same_instant_shows_the_same_token_at_every_rate(opened):
    found = {}
    for fps in (15, 30, 60):
        session = opened(fps)
        found[fps] = [
            _token(session, Fraction(k, 15), "elbow_flow", 1)[1] for k in range(45)
        ]
    for fps in (30, 60):
        assert np.allclose(found[fps], found[15], atol=1e-6)


def test_a_flow_wraps_and_spaces_its_tokens_by_arc_length():
    a, b = Node("a"), Node("b").shift(RIGHT * 4)
    diagram = Diagram(a, b)
    edge = diagram.connect((a, "right"), (b, "left"), route="straight", tip=False)
    flow = Flow(edge, speed=2.0, tokens=4, offset=0.5)
    length = 4 - 2 * a.disc.radius
    start = edge.points[0]
    flow.show_at(1.0)
    expected = [(0.5 + 2.0 + i * length / 4) % length for i in range(4)]
    got = [float(np.linalg.norm(token.get_center() - start)) for token in flow]
    assert got == pytest.approx(expected, abs=1e-3)
    assert flow.travelled() == pytest.approx(2.5)
    # A shorter wire spaces the tokens by its own length.
    edge.scale(0.5, about_point=start)
    flow.show_at(1.0)
    half = length / 2
    got = sorted(float(np.linalg.norm(t.get_center() - start)) for t in flow)
    assert got == pytest.approx(
        sorted((2.5 + i * half / 4) % half for i in range(4)), abs=1e-3
    )


@pytest.fixture
def moving():
    admission.DEMOTED.clear()
    opened: list[Session] = []

    def open_at(fps: int) -> Session:
        session = Session.open(
            SCENES, "FlowOnAMovingEdge", config={**SMALL, "frame_rate": fps}
        )
        opened.append(session)
        return session

    yield open_at
    for session in opened:
        session.close()
    admission.DEMOTED.clear()


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_flow_holds_speed_and_spacing_whatever_animates_it(moving, fps):
    """The diagram holding the flow moves, then the wire shrinks: at every
    frame each token is where the declared speed and the wire's length then
    put it.
    """
    session = moving(fps)
    for k in range(3 * fps):
        t = Fraction(k, fps)
        state = session.state_at(t)
        assert state.as_of == t
        x0, y0, x1, _ = state.named("edge").scene_box
        length = x1 - x0
        flow = state.named("flow")
        travelled = 0.2 + MOVING_SPEED * float(t)
        expected = sorted(
            (travelled + i * length / MOVING_TOKENS) % length
            for i in range(MOVING_TOKENS)
        )
        found = []
        for child in flow.children:
            bx0, by0, bx1, by1 = state.node(child).scene_box
            assert (by0 + by1) / 2 == pytest.approx(y0, abs=1e-6)
            found.append((bx0 + bx1) / 2 - x0)
        assert sorted(found) == pytest.approx(expected, abs=1e-6), float(t)


def test_a_scene_s_flow_driver_moves_the_flows_it_holds():
    admission.DEMOTED.clear()
    session = Session.open(
        SCENES, "FlowsThatLeaveAndStop", config={**SMALL, "frame_rate": 15}
    )
    try:
        scene = session.record._scene
        driver = flow_driver(scene)
        assert driver is not None
        # Every flow made in the scene, in the order made, and nothing else.
        assert driver.flows == (scene.kept, scene.left, scene.halted)
        assert scene.updaters.count(driver.updater) == 1
        # Held, not found in the scene: the flow that left is still moved.
        assert scene.kept.elapsed == pytest.approx(1.0)
        assert scene.left.elapsed == pytest.approx(1.0)
        assert scene.halted.elapsed == pytest.approx(0.5)
        # Called as one unit, it moves the flows that are not stopped.
        before = [m.get_center().copy() for m in scene.halted]
        driver()
        assert all(
            np.allclose(m.get_center(), c)
            for m, c in zip(scene.halted, before, strict=True)
        )
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_a_scene_without_a_flow_has_no_flow_driver():
    admission.DEMOTED.clear()
    session = Session.open(SCENES, "LegibleZoom", config={**SMALL, "frame_rate": 15})
    try:
        assert flow_driver(session.record._scene) is None
    finally:
        session.close()
        admission.DEMOTED.clear()
