"""The schematic marks: each builds as a vector object with points, and
each that carries a number draws it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import arachne
from arachne import (
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    RIGHT,
    UP,
    Block,
    CellGrid,
    ChargedParticle,
    Coil,
    Dot,
    DottedTrail,
    ElbowConnector,
    FlowArrow,
    LayerStack,
    LogicGate,
    Magnifier,
    Mobject,
    MultiLine,
    Node,
    Particles,
    Switch,
    Vessel,
    VMobject,
    WavefrontArcs,
    WavyLine,
    Zigzag,
)
from arachne.mobject import marks
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.frozen import engine_class

from . import trail_scenes
from .trail_scenes import INTERVAL, SPEED, START


def _built(name: str) -> Mobject:
    cls = getattr(marks, name)
    return cls(Dot()) if cls is DottedTrail else cls()


def test_there_are_twenty_nine_marks_all_public():
    assert len(marks.__all__) == 29
    assert all(getattr(arachne, name) is getattr(marks, name) for name in marks.__all__)


@pytest.mark.parametrize("name", marks.__all__)
def test_every_mark_builds_with_points(name):
    mark = _built(name)
    assert isinstance(mark, VMobject)
    assert engine_class(type(mark))
    assert sum(len(member.points) for member in mark.get_family()) > 0
    assert mark.width > 0
    assert mark.height > 0


def _curves(mob: VMobject) -> np.ndarray:
    return np.asarray(mob.points).reshape(-1, 4, 3)


def _subpath_starts(mob: VMobject) -> int:
    curves = _curves(mob)
    jumps = np.linalg.norm(curves[1:, 0] - curves[:-1, -1], axis=1) > 1e-9
    return 1 + int(jumps.sum())


def test_particles_number_their_count_inside_their_region_and_repeat():
    scatter = Particles(count=37, width=3.0, height=2.0, radius=0.05, seed=7)
    assert scatter.count == 37
    assert len(scatter.submobjects) == 37
    centres = np.array([p.get_center() for p in scatter])
    assert np.all(np.abs(centres[:, 0]) <= 1.5 - 0.05 + 1e-9)
    assert np.all(np.abs(centres[:, 1]) <= 1.0 - 0.05 + 1e-9)
    again = Particles(count=37, width=3.0, height=2.0, radius=0.05, seed=7)
    assert np.allclose(centres, [p.get_center() for p in again])
    other = Particles(count=37, width=3.0, height=2.0, radius=0.05, seed=8)
    assert not np.allclose(centres, [p.get_center() for p in other])


def test_wavefronts_are_spaced_as_declared():
    fronts = WavefrontArcs(count=5, spacing=0.4, start_radius=0.3, center=UP)
    assert fronts.spacing == 0.4
    curves = _curves(fronts)
    radii = sorted(
        {round(float(np.linalg.norm(c[0] - UP)), 9) for c in curves}
        | {round(float(np.linalg.norm(c[-1] - UP)), 9) for c in curves}
    )
    assert radii == pytest.approx([0.3 + 0.4 * i for i in range(5)])
    # Each arc is circular between its ends too.
    for curve in curves:
        middle = (
            0.125 * curve[0] + 0.375 * curve[1] + 0.375 * curve[2] + 0.125 * curve[3]
        )
        assert np.linalg.norm(middle - UP) == pytest.approx(
            np.linalg.norm(curve[0] - UP), rel=1e-3
        )


def test_a_vessel_holds_fluid_to_its_level():
    vessel = Vessel(width=1.6, height=2.0, level=0.7).shift(RIGHT * 2 + UP)
    assert vessel.level == 0.7
    assert vessel.fluid.height == pytest.approx(0.7)
    assert vessel.fluid.get_bottom()[1] == pytest.approx(vessel.walls.get_bottom()[1])
    vessel.set_level(1.5)
    assert vessel.fluid.height == pytest.approx(1.5)
    vessel.set_level(0)
    assert len(vessel.fluid.points) == 0
    with pytest.raises(ValueError, match="holds fluid"):
        vessel.set_level(2.5)


def test_a_wavy_line_has_its_wavelengths():
    wave = WavyLine(LEFT * 2, RIGHT * 2, wavelengths=3, amplitude=0.2)
    t = np.linspace(0, 1, 2001)
    samples = np.vstack(
        [
            np.outer((1 - t) ** 3, c[0])
            + np.outer(3 * (1 - t) ** 2 * t, c[1])
            + np.outer(3 * (1 - t) * t**2, c[2])
            + np.outer(t**3, c[3])
            for c in _curves(wave)
        ]
    )
    peaks = samples[:, 1] > 0.199
    rising = int(np.sum(peaks[1:] & ~peaks[:-1]))
    assert wave.wavelengths == 3
    assert rising == 3
    assert samples[:, 1].max() == pytest.approx(0.2, abs=1e-3)


def test_a_flow_arrow_is_as_wide_as_its_magnitude():
    arrow = FlowArrow(LEFT * 2, RIGHT * 2, magnitude=2.0, width_per_unit=0.25)
    corners = np.asarray(arrow.points)[::4]
    at_start = corners[np.isclose(corners[:, 0], -2.0)]
    assert arrow.magnitude == 2.0
    assert at_start[:, 1].max() - at_start[:, 1].min() == pytest.approx(0.5)


def test_a_zigzag_and_a_coil_have_their_turns():
    zigzag = Zigzag(LEFT * 2, RIGHT * 2, turns=5, amplitude=0.2)
    corners = np.asarray(zigzag.points)[::4]
    assert int(np.sum(np.isclose(corners[:, 1], 0.2))) == 5
    coil = Coil(LEFT * 2, RIGHT * 2, turns=4, radius=0.3)
    ys = np.asarray(coil.points)[::4, 1]
    assert int(np.sum(np.isclose(ys, 0.3))) == 4


def test_multi_lines_count_one_to_three():
    assert _subpath_starts(MultiLine(count=3, spacing=0.1)) == 3
    with pytest.raises(ValueError, match="one to three"):
        MultiLine(count=4)


def test_an_elbow_runs_at_right_angles_from_port_to_port():
    elbow = ElbowConnector(
        LEFT * 3 + DOWN, RIGHT * 2 + UP * 2, start_direction=RIGHT, end_direction=UP
    )
    curves = _curves(elbow)
    assert np.allclose(curves[0, 0], LEFT * 3 + DOWN)
    straight = [c for c in curves if np.allclose(np.cross(c[3] - c[0], c[1] - c[0]), 0)]
    for c in straight:
        run = c[3] - c[0]
        assert min(abs(run[0]), abs(run[1])) < 1e-9
    (tip,) = elbow.submobjects
    # The head's point is on the port, entering it from above; the wire ends
    # at the head's base.
    tip_points = np.asarray(tip.points)
    point = tip_points[np.argmin(tip_points[:, 1])]
    assert np.allclose(point, RIGHT * 2 + UP * 2)
    assert np.allclose(curves[-1, -1], RIGHT * 2 + UP * 2.2)


def test_ports_follow_their_mark():
    block = Block("b", width=2, height=1)
    assert np.allclose(block.port("right"), RIGHT)
    assert np.allclose(block.port_direction("top"), UP)
    block.rotate(PI / 2, about_point=ORIGIN).shift(RIGHT * 3)
    assert np.allclose(block.port("right"), RIGHT * 3 + UP)
    assert np.allclose(block.port_direction("right"), UP, atol=1e-6)
    custom = Block(ports={"in": (LEFT, 0.25)}, width=2, height=1)
    assert np.allclose(custom.port("in"), LEFT + DOWN * 0.25)
    with pytest.raises(ValueError, match="ports in"):
        custom.port("out")
    node = Node("n", radius=0.5, ports={"ne": UP + RIGHT})
    assert np.allclose(node.port("ne"), (UP + RIGHT) / math.sqrt(2) * 0.5)
    gate = LogicGate("nand", inputs=3)
    assert gate.port_names == ("in0", "in1", "in2", "out")
    assert gate.port("out")[0] > gate.body.get_right()[0]
    # A port widens nothing.
    assert block.width == pytest.approx(1.0)
    assert block.height == pytest.approx(2.0)


def test_a_switch_opens_and_closes_in_place():
    switch = Switch(LEFT, RIGHT, closed=True)
    assert np.allclose(np.asarray(switch.blade.points)[-1], RIGHT)
    switch.shift(UP).set_closed(False)
    end = np.asarray(switch.blade.points)[-1]
    assert end[1] > 1
    assert np.linalg.norm(end - (LEFT + UP)) == pytest.approx(2)


def test_magnifier_lines_touch_both_circles():
    lens = Magnifier(ORIGIN, RIGHT * 3 + UP, radius=0.3, large_radius=1.0)
    for curve in _curves(lens.lines):
        a, b = curve[0], curve[3]
        u = (b - a) / np.linalg.norm(b - a)
        for centre, radius in ((ORIGIN, 0.3), (RIGHT * 3 + UP, 1.0)):
            offset = centre - a
            distance = np.linalg.norm(offset - np.dot(offset, u) * u)
            assert distance == pytest.approx(radius)


def test_cells_layers_and_charges_are_addressed():
    grid = CellGrid(rows=2, cols=3, values=[["a", None, "c"]])
    assert grid.entry(0, 0).text == "a"
    assert grid.entry(0, 1) is None
    assert grid.cell(1, 2).get_center()[0] > grid.cell(1, 0).get_center()[0]
    stack = LayerStack(["air", "film", "glass"], thickness=[0.4, 0.2, 0.6])
    assert stack.layer(0).get_center()[1] > stack.layer(2).get_center()[1]
    assert stack.height == pytest.approx(1.2)
    assert len(ChargedParticle(-2).sign.points) == 4
    assert len(ChargedParticle(0).sign.points) == 0


def test_a_trail_drops_a_dot_every_interval():
    dot = Dot(LEFT)
    trail = DottedTrail(dot, interval=0.5)
    for _ in range(60):
        dot.shift(RIGHT * 0.05)
        trail.update(dt=1 / 60)
    assert trail.interval == 0.5
    assert len(trail.dots) == 3
    assert np.allclose([d[0] for d in trail.dots], [-1.0, -1.0 + 1.5, -1.0 + 3.0])
    assert _subpath_starts(trail) == 3
    trail.stop()
    trail.update(dt=1.0)
    assert len(trail.dots) == 3


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_at(fps: int) -> Session:
        session = Session.open(
            Path(trail_scenes.__file__),
            "TrailBehindADot",
            config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
        )
        sessions.append(session)
        return session

    yield open_at
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_trail_keeps_time_at_any_frame_rate(opened, fps):
    session = opened(fps)
    record = session.record
    trail = record._scene.trail
    # Instants every rate's grid holds, in the play and in the wait after it.
    for seconds in (Fraction(1, 3), Fraction(1), Fraction(2), Fraction(43, 15)):
        _, capture = record.capture_at(seconds)
        dots = record.node_state(trail, capture).get("dots")
        assert len(dots) == math.floor(seconds / Fraction(INTERVAL)) + 1
        # Where the dot was at each drop: moving for 2 s, then at rest.
        expected = [START + SPEED * min(k * INTERVAL, 2.0) for k in range(len(dots))]
        assert [d[0] for d in dots] == pytest.approx(expected, abs=1e-6)
