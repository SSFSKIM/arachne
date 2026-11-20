"""A labelled connector that carries a flow keeps its label beside the
line, clear of the tokens and of the line, on the side with more room;
a connector without a flow, or without a label, is as it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import (
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    Block,
    Diagram,
    Flow,
    Line,
    Node,
    Square,
    Text,
)
from arachne.mobject.marks._paths import ArcLength

#: Positions read along a flow's path, per token.
SAMPLES = 240


def _box(mob) -> tuple[np.ndarray, np.ndarray]:
    return mob.get_critical_point(DL)[:2], mob.get_critical_point(UR)[:2]


def _meet(a, b) -> bool:
    """Whether two boxes share any point."""
    (a0, a1), (b0, b1) = a, b
    return bool(np.all(a0 <= b1) and np.all(b0 <= a1))


def _wire(edge) -> np.ndarray:
    """Points along everything ``edge`` draws: its wire and its head."""
    t = np.linspace(0, 1, 33)[:, None]
    found = []
    for m in edge.family_members_with_points():
        curves = m.points[: len(m.points) // 4 * 4].reshape(-1, 4, 3)
        for p0, p1, p2, p3 in curves:
            found.append(
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t**2 * p2
                + t**3 * p3
            )
    return np.concatenate(found)[:, :2]


def branching() -> tuple[Diagram, list]:
    """Four blocks: a labelled join from the first to the second, which
    branches to the other two, one above and one below.
    """
    diagram = Diagram(cell=(3.2, 1.8))
    first = diagram.place(Block("p", width=1.6, height=1.0), 0, 0.5)
    second = diagram.place(Block("q", width=1.6, height=1.0), 1, 0.5)
    upper = diagram.place(Block("r", width=1.6, height=1.0), 2, 0)
    lower = diagram.place(Block("s", width=1.6, height=1.0), 2, 1)
    edges = [
        diagram.connect(first, second, label="wire"),
        diagram.connect(second, upper),
        diagram.connect(second, lower),
    ]
    diagram.move_to(0.3 * DOWN + 0.2 * LEFT)
    return diagram, edges


def _level():
    diagram, edges = branching()
    return diagram, edges[0]


def _vertical():
    a, b = Node("A").move_to(UP * 1.5), Node("B").move_to(DOWN * 1.5)
    diagram = Diagram(a, b)
    return diagram, diagram.connect(a, b, route="straight", label="down")


def _diagonal():
    a, b = Node("A").move_to(LEFT * 2 + DOWN), Node("B").move_to(RIGHT * 2 + UP)
    diagram = Diagram(a, b)
    return diagram, diagram.connect(a, b, route="straight", label="across")


def _elbow():
    source = Block("source", width=1.6, height=1.0).move_to(LEFT * 3 + UP * 2)
    sink = Block("sink", width=1.6, height=1.0).move_to(RIGHT * 2 + DOWN * 0.6)
    diagram = Diagram(source, sink)
    return diagram, diagram.connect((source, "right"), (sink, "top"), label="feed")


GEOMETRIES = {
    "level": _level,
    "vertical": _vertical,
    "diagonal": _diagonal,
    "elbow": _elbow,
}


@pytest.mark.parametrize("name", GEOMETRIES)
def test_no_token_meets_the_label_anywhere_on_its_path(name):
    _, edge = GEOMETRIES[name]()
    flow = Flow(edge, speed=1.2, tokens=2)
    pill = _box(edge.label)
    length = ArcLength(edge.points).length
    for step in range(SAMPLES):
        # The first token through one whole length of the edge.
        flow.offset = step / SAMPLES * length
        flow.show_at(0.0)
        for token in flow.submobjects:
            assert not _meet(_box(token), pill), (name, step)


@pytest.mark.parametrize("name", GEOMETRIES)
def test_the_label_of_a_flowing_connector_is_clear_of_its_line(name):
    _, edge = GEOMETRIES[name]()
    on_line = edge.label.get_center()
    Flow(edge, speed=1.2, tokens=2)
    (low, high) = _box(edge.label)
    inside = np.all((_wire(edge) >= low) & (_wire(edge) <= high), axis=1)
    assert not inside.any()
    assert not np.allclose(edge.label.get_center(), on_line)


def _points(diagram) -> list[np.ndarray]:
    return [np.array(m.points) for m in diagram.get_family()]


def test_a_connector_without_a_flow_or_a_label_stays_as_it_was():
    plain, _ = branching()
    flowing, edges = branching()
    # Flows on every unlabelled connector; none on the labelled one.
    for edge in edges[1:]:
        Flow(edge, speed=1.2, tokens=2)
    for got, want in zip(_points(flowing), _points(plain), strict=True):
        assert np.array_equal(got, want)
    # A flow on the labelled one moves its label alone.
    Flow(edges[0], speed=1.2, tokens=2)
    moved = {id(m) for m in edges[0].label.get_family()}
    for member, want in zip(flowing.get_family(), _points(plain), strict=True):
        if id(member) not in moved:
            assert np.array_equal(member.points, want)


def test_the_label_s_place_is_the_same_whatever_came_first():
    once, edge = _level()
    Flow(edge, speed=1.2, tokens=2)
    twice, edge = _level()
    Flow(edge, speed=1.2, tokens=2)
    Flow(edge, speed=0.5, tokens=3)
    for a, b in zip(_points(twice), _points(once), strict=True):
        assert np.array_equal(a, b)
    # Moved, then a flow; or a flow, then moved.
    moved_first, edge = _level()
    moved_first.shift(RIGHT * 1.3 + UP * 0.7)
    Flow(edge, speed=1.2, tokens=2)
    flow_first, edge = _level()
    Flow(edge, speed=1.2, tokens=2)
    flow_first.shift(RIGHT * 1.3 + UP * 0.7)
    for a, b in zip(_points(moved_first), _points(flow_first), strict=True):
        assert np.allclose(a, b, atol=1e-12)


def test_the_label_goes_to_the_side_with_more_room():
    # Level ground: above.
    diagram, edge = _level()
    Flow(edge, speed=1.2, tokens=2)
    line_y = edge.points[0][1]
    assert edge.label.get_center()[1] > line_y
    # A piece just above the line's middle: below.
    _, edge = _obstructed()
    Flow(edge, speed=1.2, tokens=2)
    assert edge.label.get_center()[1] < edge.points[0][1]


def _segments_meet(edge, low, high) -> bool:
    """Whether any straight stretch of ``edge``'s drawing passes through
    the box ``low``-``high``, sampled densely.
    """
    points = _wire(edge)
    dense = np.concatenate(
        [np.linspace(a, b, 200) for a, b in zip(points[:-1], points[1:], strict=True)]
    )
    return bool(np.all((dense >= low) & (dense <= high), axis=1).any())


def _parallel(flow_first: bool = False):
    """A short labelled join, a long straight wire close above it, and a
    square close below it: below has less room than the join's own line
    gives, above has none. With ``flow_first`` the flow is made before the
    long wire.
    """
    a, b = Square(0.4).move_to(LEFT * 1.4), Square(0.4).move_to(RIGHT * 1.4)
    c, d = (
        Square(0.4).move_to(LEFT * 4.7 + UP * 0.32),
        Square(0.4).move_to(RIGHT * 9.7 + UP * 0.32),
    )
    under = Square(0.4).move_to(DOWN * 0.75)
    diagram = Diagram(a, b, c, d, under)
    edge = diagram.connect(a, b, route="straight", tip=False, label="w")
    if flow_first:
        Flow(edge, speed=1.0, tokens=2)
    wire = diagram.connect(c, d, route="straight", tip=False)
    if not flow_first:
        Flow(edge, speed=1.0, tokens=2)
    return diagram, edge, wire


def test_a_long_parallel_wire_takes_the_room_on_its_side():
    _, edge, wire = _parallel()
    low, high = _box(edge.label)
    assert not _segments_meet(wire, low, high)
    assert edge.label.get_center()[1] < 0


def _crossed(flow_first: bool = False):
    """A labelled join and a short wire just above its middle, joined
    after it; with ``flow_first`` the flow is made in between.
    """
    a, b = Square(0.4).move_to(LEFT * 1.5), Square(0.4).move_to(RIGHT * 1.5)
    c, d = (
        Square(0.4).move_to(LEFT * 0.7 + UP * 0.3),
        Square(0.4).move_to(RIGHT * 0.7 + UP * 0.3),
    )
    diagram = Diagram(a, b, c, d)
    edge = diagram.connect(a, b, route="straight", tip=False, label="w")
    if flow_first:
        Flow(edge, speed=1.0, tokens=2)
    wire = diagram.connect(c, d, route="straight", tip=False)
    if not flow_first:
        Flow(edge, speed=1.0, tokens=2)
    return diagram, edge, wire


@pytest.mark.parametrize("layout", [_parallel, _crossed])
def test_a_flow_made_before_the_rest_of_the_diagram_places_its_label_alike(layout):
    later, edge, wire = layout()
    first, _, _ = layout(flow_first=True)
    assert edge.label.get_center()[1] < 0
    assert not _segments_meet(wire, *_box(edge.label))
    # The same geometry, to rounding: the label moved there by another way.
    for a, b in zip(_points(first), _points(later), strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def _obstructed():
    """A labelled join with a block close above its middle."""
    a, b = Square(0.4).move_to(LEFT * 1.5), Square(0.4).move_to(RIGHT * 1.5)
    over = Square(0.4).move_to(UP * 0.5)
    diagram = Diagram(a, b, over)
    return diagram, diagram.connect(a, b, route="straight", tip=False, label="w")


def test_a_copy_places_its_label_as_the_original_does():
    flowed, edge = _obstructed()
    Flow(edge, speed=1.0, tokens=2)
    flowed_then_copied = flowed.copy()
    original, _ = _obstructed()
    copied = original.copy()
    Flow(copied.edges[0], speed=1.0, tokens=2)
    assert copied.edges[0].label.get_center()[1] < 0
    for a, b in zip(_points(copied), _points(flowed_then_copied), strict=True):
        assert np.array_equal(a, b)


class _ComparedLine(Line):
    """A line that compares by value, and so has no hash."""

    def __eq__(self, other):
        return isinstance(other, Line) and np.array_equal(self.points, other.points)

    __hash__ = None


def test_a_flow_on_a_labelled_edge_of_any_kind_sets_its_label_beside():
    edge = _ComparedLine(LEFT * 2, RIGHT * 2)
    edge.label = Text("w", font_size=24).move_to(ORIGIN)
    Flow(edge, speed=1.0, tokens=2)
    assert edge.label.get_center()[1] > 0
