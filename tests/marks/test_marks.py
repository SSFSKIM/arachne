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
from .trail_scenes import INTERVAL, LATE_INTERVAL, LATE_START, SPEED, START


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


def test_a_shallow_fill_follows_the_bottom_corners():
    vessel = Vessel(width=1.6, height=2.0, level=0.05, corner_radius=0.2)
    r, depth = 0.2, 0.05
    # The surface meets each corner's arc, short of the walls.
    reach = 0.8 - r + math.sqrt(r**2 - (r - depth) ** 2)
    assert vessel.fluid.width == pytest.approx(2 * reach, abs=1e-3)
    assert vessel.fluid.height == pytest.approx(depth, abs=1e-6)
    # Every anchor of the fill lies inside the walls' rounded bottom.
    centres = np.array([[-0.6, -0.8], [0.6, -0.8]])
    points = np.asarray(vessel.fluid.points)
    anchors = np.vstack([points[::4], points[3::4]])
    for x, y, _ in anchors:
        if abs(x) > 0.6:
            corner = centres[0] if x < 0 else centres[1]
            assert np.hypot(x - corner[0], y - corner[1]) <= r + 1e-6


def test_a_turned_vessel_fills_in_its_own_frame():
    upright = Vessel(width=1.6, height=2.0, level=0.4)
    turned = Vessel(width=1.6, height=2.0, level=1.0)
    turned.rotate(PI / 2, about_point=ORIGIN).scale(1.5, about_point=ORIGIN)
    turned.shift(RIGHT * 2)
    turned.set_level(0.4)
    expected = upright.fluid.copy()
    expected.rotate(PI / 2, about_point=ORIGIN).scale(1.5, about_point=ORIGIN)
    expected.shift(RIGHT * 2)
    assert np.allclose(turned.fluid.points, expected.points, atol=1e-9)


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


def test_a_trail_drops_each_dot_where_its_object_was_when_it_was_due():
    """Dots due between two ticks, several to a tick, fall along the course
    between the readings, at their own times.
    """
    dot = Dot(LEFT)
    trail = DottedTrail(dot, interval=0.004)
    for _ in range(6):
        dot.shift(RIGHT * 0.05)
        trail.update(dt=1 / 60)
    # 0.1 s at 3 units a second: a dot every 0.004 s, every 0.012 units.
    assert len(trail.dots) == 26
    assert [d[0] for d in trail.dots] == pytest.approx(
        [-1.0 + 0.012 * k for k in range(26)], abs=1e-9
    )


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
    assert Fraction(INTERVAL).limit_denominator(100) == Fraction(3, 25)
    # Instants every rate's grid holds, in the play and in the wait after it;
    # the interval is on no rate's grid.
    for seconds in (Fraction(1, 3), Fraction(1), Fraction(2), Fraction(43, 15)):
        _, capture = record.capture_at(seconds)
        dots = record.node_state(trail, capture).get("dots")
        assert len(dots) == math.floor(seconds / Fraction(3, 25)) + 1
        # Where the dot was at each drop: moving for 2 s, then at rest.
        expected = [START + SPEED * min(k * INTERVAL, 2.0) for k in range(len(dots))]
        assert [d[0] for d in dots] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_trail_made_between_two_ticks_keeps_the_scene_s_time(fps):
    """A trail made off the tick's grid drops each dot at its own scene
    time, counted from when the trail was made, not from a tick before.
    """
    admission.DEMOTED.clear()
    session = Session.open(
        Path(trail_scenes.__file__),
        "TrailMadeOffTheGrid",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
    )
    try:
        record = session.record
        scene = record._scene
        made = float(scene.made_at)
        assert made == pytest.approx(LATE_START)
        checked = 0
        for capture in (c for play in record.plays for c in play.samples):
            state = record.node_state(scene.trail, capture)
            if state is None or capture.instant < scene.made_at:
                continue
            t = capture.instant
            dots = state.get("dots")
            due = math.floor((float(t) - made) / LATE_INTERVAL + 1e-9) + 1
            assert len(dots) == due, float(t)
            expected = [START + SPEED * (made + n * LATE_INTERVAL) for n in range(due)]
            assert [d[0] for d in dots] == pytest.approx(expected, abs=1e-6)
            checked += 1
        assert checked >= 0.6 * fps
    finally:
        session.close()
        admission.DEMOTED.clear()


@pytest.mark.parametrize("fps", [15, 60])
def test_a_trail_leaves_no_dots_for_time_it_was_out_of_the_scene(fps):
    admission.DEMOTED.clear()
    session = Session.open(
        Path(trail_scenes.__file__),
        "TrailLeftAndRejoined",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
    )
    try:
        xs = [d[0] for d in session.record._scene.trail.dots]
    finally:
        session.close()
        admission.DEMOTED.clear()
    away, back = START + SPEED, START + 2 * SPEED
    # Watched for 1 s, then for 1 s again: a dot every interval of each.
    watched = [x for x in xs if x <= away + 1e-6]
    assert len(watched) == math.floor(1 / INTERVAL) + 1
    # Nothing where the dot went while the trail was out.
    assert not [x for x in xs if away + 1e-6 < x < back - 1e-6]
    rejoined = [x for x in xs if x >= back - 1e-6]
    assert rejoined
    # From where it came back, a dot each interval of the time it watched.
    steps = np.diff(rejoined)
    assert steps == pytest.approx([SPEED * INTERVAL] * len(steps), abs=1e-6)


def test_a_diagram_s_elbow_takes_a_stub_and_still_keeps_out_of_the_other_piece():
    diagram = arachne.Diagram()
    a = diagram.place(Block("a"), 0, 0)
    b = diagram.place(Block("b"), 1, 0)
    c = diagram.place(Block("c"), 0, 2)
    # Asked for, and given where there is room: the route runs that far
    # right out of the port before it turns down to the piece below.
    reach = {}
    for stub in (0.2, 0.6):
        edge = diagram.connect((a, "right"), (c, "top"), stub=stub, end_stub=0.1)
        reach[stub] = float(_samples(edge)[:, 0].max()) - a.port("right")[0]
        assert stub - 0.15 < reach[stub] <= stub + 1e-9
    assert reach[0.6] > reach[0.2]
    # Asked for, and cut where the other piece is in the way.
    edge = diagram.connect((a, "right"), (b, "right"), stub=0.6)
    path = _samples(edge)
    assert not _inside(path, a.body)
    assert not _inside(path, b.body)


#: Each mark made of several parts, and the parts that keep a style of their
#: own whatever style the mark is given.
GROUPED = {
    "Vessel": lambda m: [m.fluid],
    "Drum": lambda m: [],
    "LayerStack": lambda m: list(m.labels),
    "Block": lambda m: [m.strip, m.title],
    "CellGrid": lambda m: list(m.entries),
    "Node": lambda m: [m.label],
    "ChargedParticle": lambda m: [m.sign],
    "Particles": lambda m: [],
    "Magnifier": lambda m: [],
    "Switch": lambda m: [],
    "LogicGate": lambda m: [],
}
MADE = {
    "Block": lambda **k: Block("title", **k),
    "Node": lambda **k: Node("n", **k),
    "LayerStack": lambda **k: LayerStack(**k),
    "CellGrid": lambda **k: CellGrid(values=[["a"]], **k),
    "LogicGate": lambda **k: LogicGate("nand", **k),
}


def _drawn(mark: Mobject, own: list[Mobject]) -> list[VMobject]:
    """The members of ``mark`` that draw, but those with a style of their
    own (whole families) and the texts.
    """
    kept = {id(m) for part in own for m in part.get_family()}
    return [
        m
        for m in mark.get_family()
        if isinstance(m, VMobject)
        and len(m.points) > 1
        and id(m) not in kept
        and not isinstance(m, arachne.Text)
    ]


@pytest.mark.parametrize("name", sorted(GROUPED))
def test_a_grouped_mark_hands_its_style_to_the_parts_that_draw(name):
    make = MADE.get(name, getattr(marks, name))
    assert all(isinstance(m, arachne.VGroup) for m in [make()])
    mark = make(stroke_width=7)
    drawn = _drawn(mark, GROUPED[name](mark))
    assert drawn
    for member in drawn:
        assert float(member.get_stroke_width()) == 7
    # The parts with a style of their own keep it.
    plain = make()
    for kept, before in zip(GROUPED[name](mark), GROUPED[name](plain), strict=True):
        assert [m.get_stroke_width() for m in kept.get_family()] == [
            b.get_stroke_width() for b in before.get_family()
        ]


def test_a_keyword_a_mark_cannot_take_is_refused():
    with pytest.raises(TypeError):
        Block("b", not_a_style=1)


def _inside(points: np.ndarray, box: Mobject) -> bool:
    x0, y0 = box.get_corner(DOWN + LEFT)[:2]
    x1, y1 = box.get_corner(UP + RIGHT)[:2]
    return bool(
        np.any(
            (points[:, 0] > x0 + 1e-6)
            & (points[:, 0] < x1 - 1e-6)
            & (points[:, 1] > y0 + 1e-6)
            & (points[:, 1] < y1 - 1e-6)
        )
    )


def _samples(edge: VMobject) -> np.ndarray:
    curves = _curves(edge)
    t = np.linspace(0, 1, 50)[:, None]
    return np.vstack(
        [
            (1 - t) ** 3 * c[0]
            + 3 * (1 - t) ** 2 * t * c[1]
            + 3 * (1 - t) * t**2 * c[2]
            + t**3 * c[3]
            for c in curves
        ]
    )


@pytest.mark.parametrize(
    ("first_port", "second_port", "rise"),
    [
        ("right", "right", 0.0),  # one way, on one row
        ("left", "left", 0.0),
        ("left", "right", 0.0),  # facing away, on one row
        ("right", "right", 0.4),  # one way, nearly on one row
        ("top", "top", 0.0),  # one way, in one column
    ],
)
def test_side_by_side_ports_are_joined_round_both_pieces(first_port, second_port, rise):
    a = Block("a", width=2.2, height=1.4)
    b = Block("b", width=2.2, height=1.4)
    if first_port == "top":
        a.shift(DOWN * 2)
        b.shift(UP * 1)
    else:
        a.shift(LEFT * 3)
        b.shift(RIGHT * 3 + UP * rise)
    diagram = arachne.Diagram(a, b)
    edge = diagram.connect((a, first_port), (b, second_port))
    path = _samples(edge)
    assert np.allclose(path[0], a.port(first_port))
    assert not _inside(path, a.body)
    assert not _inside(path, b.body)
    # Every run is straight along one axis or a corner's arc.
    for curve in _curves(edge):
        run = curve[3] - curve[0]
        straight = np.allclose(np.cross(curve[1] - curve[0], run), 0)
        assert not straight or min(abs(run[0]), abs(run[1])) < 1e-9


@pytest.mark.parametrize(
    ("first_port", "second_port"),
    [("right", "right"), ("left", "left"), ("left", "right"), ("right", "left")],
)
def test_neighbours_at_the_diagram_s_own_spacing_are_joined_clear_of_both(
    first_port, second_port
):
    diagram = arachne.Diagram()
    a = diagram.place(Block("a"), 0, 0)
    b = diagram.place(Block("b"), 1, 0)
    # The defaults leave a narrow gap between the two.
    gap = b.body.get_left()[0] - a.body.get_right()[0]
    assert 0 < gap < 0.3
    edge = diagram.connect((a, first_port), (b, second_port))
    path = _samples(edge)
    assert not _inside(path, a.body)
    assert not _inside(path, b.body)


def test_a_port_the_other_piece_covers_is_refused():
    a = Block("a")
    b = Block("b").next_to(a, RIGHT, buff=0)
    diagram = arachne.Diagram(a, b)
    with pytest.raises(ValueError, match="move the two pieces apart"):
        diagram.connect((a, "right"), (b, "right"))
