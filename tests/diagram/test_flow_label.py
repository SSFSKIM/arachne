"""A labelled connector that carries a flow keeps its label beside the
line, clear of the tokens and of the line, on the side with more room;
a connector without a flow, or without a label, is as it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import DL, DOWN, LEFT, RIGHT, UP, UR, Block, Diagram, Flow, Node
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
    diagram, edge = _level()
    middle = (edge.points[0] + edge.points[-1]) / 2
    diagram.add(Node("x", radius=0.2).move_to(middle + UP * 0.45))
    Flow(edge, speed=1.2, tokens=2)
    assert edge.label.get_center()[1] < line_y
