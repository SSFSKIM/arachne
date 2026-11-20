"""A zoom fades a connector, with its label, no later than the earliest
fading of the blocks it joins, so no arrow shows without its ends; a
return brings each back on the same window reversed.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LEFT, RIGHT, UP, Block, Diagram, Dot, VMobject
from arachne.mobject.diagram import _joins
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing

from . import zoom_out_scenes

SCENES = Path(zoom_out_scenes.__file__)
TICKS = 60
#: The scene's moves, by name: where each begins and how long it runs.
MOVES = {
    "in": (Fraction(3, 5), 2.5),
    "back": (Fraction(41, 10), 2.5),
    "again": (Fraction(33, 5), 2.5),
}


def _shows(mob) -> float:
    return max(
        float(np.max(opacities, initial=0.0))
        for m in mob.get_family()
        if len(m.points)
        for opacities in (m.get_fill_opacities(), m.get_stroke_opacities())
    )


@pytest.fixture(scope="module")
def shares():
    """Each instant of the scene at 60 Hz, with how much of itself each
    connector, its label and each block it joins shows then.
    """
    admission.DEMOTED.clear()
    session = Session.open(
        SCENES,
        "IntoAHubAndBack",
        config={"pixel_width": 480, "pixel_height": 270, "frame_rate": TICKS},
    )
    try:
        record = session.record
        joins = record._scene.joins
        parts = [
            mob
            for edge, a, b in joins
            for mob in (edge, edge.label, a, b)
            if mob is not None
        ]
        full = None
        seen = []
        with Drawing(record) as drawing:
            for k in range(int(record.duration * TICKS) + 1):
                t = Fraction(k, TICKS)
                _, capture = record.capture_at(t)
                drawing.show(capture, capture.draw_list())
                now = {id(m): _shows(m) for m in parts}
                if full is None:
                    full = now
                seen.append((t, {key: now[key] / full[key] for key in now}))
        yield joins, seen
    finally:
        session.close()
        admission.DEMOTED.clear()


def _during(seen, move):
    start, run = MOVES[move]
    return [(t, share) for t, share in seen if start <= t <= start + Fraction(run)]


@pytest.mark.parametrize("move", MOVES)
def test_no_connector_shows_more_than_the_blocks_it_joins(shares, move):
    joins, seen = shares
    for t, share in _during(seen, move):
        for edge, a, b in joins:
            ends = min(share[id(a)], share[id(b)])
            for part in (edge, edge.label):
                if part is not None:
                    assert share[id(part)] <= ends + 1e-9, (move, float(t))


@pytest.mark.parametrize("move", MOVES)
def test_the_connector_to_the_target_goes_with_its_other_end(shares, move):
    joins, seen = shares
    (data, source, target) = joins[0]
    assert data.label is not None
    during = _during(seen, move)
    # The zoom's target never fades; the block at the other end does, and
    # the connector and its label with it, hidden once it is.
    assert all(share[id(target)] == 1.0 for _, share in during)
    hidden = [t for t, share in during if share[id(source)] == 0.0]
    assert hidden
    for t, share in during:
        if share[id(source)] == 0.0:
            assert share[id(data)] == 0.0, float(t)
            assert share[id(data.label)] == 0.0, float(t)
    # And the connector does fade: partly shown on the way.
    assert any(0.0 < share[id(data)] < 1.0 for _, share in during)


def test_a_return_brings_each_connector_back_on_its_window_reversed(shares):
    joins, seen = shares
    by_t = dict(seen)
    (in_start, run), (back_start, _) = MOVES["in"], MOVES["back"]
    for k in range(int(run * TICKS) + 1):
        going = by_t[in_start + Fraction(k, TICKS)]
        coming = by_t[back_start + Fraction(run) - Fraction(k, TICKS)]
        for edge, a, b in joins:
            for part in (edge, edge.label, a, b):
                if part is not None:
                    assert coming[id(part)] == pytest.approx(going[id(part)], abs=1e-9)
    # Back to full by the return's end.
    end = by_t[back_start + Fraction(run)]
    assert all(value == pytest.approx(1.0) for value in end.values())


def _tipped() -> tuple[Diagram, Block, Block, VMobject]:
    """Two blocks joined by a tipped connector, whose wire stops short of
    the far block's port, and a small dot nearer the wire's end than that
    block.
    """
    near = Block("a", width=2.0, height=1.2).move_to(LEFT * 3)
    far = Block("b", width=2.0, height=1.2).move_to(RIGHT * 1)
    diagram = Diagram(near, far)
    edge = diagram.connect(near, far, label="go")
    diagram.add(Dot(radius=0.08).move_to(edge.points[-1] + UP * 0.2))
    return diagram, near, far, edge


def test_a_connector_s_ends_are_the_pieces_it_was_made_to_join():
    diagram, near, far, edge = _tipped()
    joined = _joins([diagram])
    assert joined[id(edge)] == (near, far)
    assert joined[id(edge.label)] == (edge, near, far)
    # And on a copy, its own pieces.
    copy = diagram.copy()
    joined = _joins([copy])
    assert joined[id(copy.edges[0])] == (copy[0], copy[1])
