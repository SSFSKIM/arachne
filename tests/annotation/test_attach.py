"""One attachment for every annotation: an object kept at its anchor plus
its offset, a leader from its body to the anchor, ``follow=False`` leaving
the object where it stands and stretching the leader; anchors on any drawn
object; an anchor that names nothing draws nothing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    BarGauge,
    Callout,
    ColumnGauge,
    DialGauge,
    LabelPill,
    LeaderLine,
    Line,
    Readout,
    Rig,
    Square,
    Text,
    VectorizedPoint,
    anchor,
)
from arachne.assets import Anchor
from arachne.lint.raised import RaisedFinding
from arachne.mobject.annotation.attach import Annotation
from arachne.timeline.frozen import engine_class


def _update(mobject) -> None:
    mobject.update(0)


def test_anchor_reads_a_box_point_a_place_along_the_outline_or_a_point():
    square = Square(2).shift(RIGHT)
    corner, along, point = (
        anchor(square, at=UR),
        anchor(square, along=0.25),
        anchor(square, index=3),
    )
    np.testing.assert_allclose(corner.point(), [2, 1, 0])
    np.testing.assert_allclose(along.point(), square.point_from_proportion(0.25))
    np.testing.assert_allclose(point.point(), square.points[3])
    np.testing.assert_allclose(anchor(square).point(), [1, 0, 0])
    square.rotate(np.pi / 2, about_point=ORIGIN)
    # Each reads the object as it is now.
    np.testing.assert_allclose(corner.point(), [1, 2, 0], atol=1e-12)
    np.testing.assert_allclose(along.point(), square.point_from_proportion(0.25))
    assert isinstance(corner, Anchor)
    assert corner.resolved
    assert copy.deepcopy(corner) is corner


def test_anchor_takes_one_way_of_attaching():
    square = Square()
    with pytest.raises(ValueError, match="one of"):
        anchor(square, at=UP, along=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        anchor(square, along=1.5)
    with pytest.raises(IndexError):
        anchor(square, index=len(square.points))


def test_a_callout_sits_at_its_anchor_plus_its_offset_and_follows_it():
    square = Square(2)
    callout = Callout("corner", at=anchor(square, at=UR), offset=UR)
    assert isinstance(callout.pill, LabelPill)
    assert isinstance(callout.leader, LeaderLine)
    np.testing.assert_allclose(callout.pill.get_center(), [2, 2, 0], atol=1e-9)
    np.testing.assert_allclose(callout.leader.points[-1], [1, 1, 0], atol=1e-12)
    # The leader leaves the pill at its edge, toward the anchor.
    start = callout.leader.points[0]
    x0, y0 = callout.pill.get_corner(np.array([-1, -1, 0]))[:2]
    assert start[0] == pytest.approx(x0) or start[1] == pytest.approx(y0)
    square.shift(3 * LEFT)
    _update(callout)
    np.testing.assert_allclose(callout.pill.get_center(), [-1, 2, 0], atol=1e-9)
    np.testing.assert_allclose(callout.leader.points[-1], [-2, 1, 0], atol=1e-12)


def test_an_annotation_that_does_not_follow_stays_and_its_leader_stretches():
    square = Square(2)
    callout = Callout(
        "corner", at=anchor(square, at=UR), offset=UR, follow=False
    ).shift(DOWN)
    np.testing.assert_allclose(callout.pill.get_center(), [2, 1, 0], atol=1e-9)
    square.shift(2 * LEFT)
    _update(callout)
    np.testing.assert_allclose(callout.pill.get_center(), [2, 1, 0], atol=1e-9)
    np.testing.assert_allclose(callout.leader.points[-1], [-1, 1, 0], atol=1e-12)


def test_an_annotation_without_a_leader_has_none():
    square = Square(2)
    reading = Readout(1, at=anchor(square, at=UP), offset=UP)
    assert reading.leader is None
    np.testing.assert_allclose(reading.label.get_center(), [0, 2, 0], atol=1e-9)
    with_leader = Readout(1, at=anchor(square, at=UP), offset=UP, leader=True)
    assert isinstance(with_leader.leader, LeaderLine)
    np.testing.assert_allclose(with_leader.leader.points[-1], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize(
    "make",
    [
        lambda at: Callout("x", at=at),
        lambda at: Readout(1, at=at, leader=True),
        lambda at: BarGauge(0.5, at=at),
        lambda at: ColumnGauge(0.5, at=at),
        lambda at: DialGauge(0.5, at=at, leader=True),
    ],
)
def test_every_annotation_attaches_the_same_way(make):
    dot = Line(LEFT, RIGHT)
    made = make(anchor(dot, at=RIGHT))
    dot.shift(UP)
    _update(made)
    body = made.body.get_center()
    np.testing.assert_allclose(body - made.attached.offset, [1, 1, 0], atol=1e-9)
    if made.leader is not None:
        np.testing.assert_allclose(made.leader.points[-1], [1, 1, 0], atol=1e-12)


def test_what_an_unknown_anchor_holds_draws_nothing():
    box = Rig("fixture:lidded-box")
    with pytest.raises(RaisedFinding) as raised:
        box.anchor("lid_tip")
    assert raised.value.code == "annotation.anchor_unknown"
    # In a scene the lint reports it; the anchor it gives has no point.
    nowhere = Anchor(box, None, "lid_tip")
    callout = Callout("lid", at=nowhere)
    assert not any(len(m.points) for m in callout.get_family())
    _update(callout)
    assert not any(len(m.points) for m in callout.get_family())


def test_a_label_pill_is_a_text_on_its_own_plate():
    pill = LabelPill("valve")
    assert isinstance(pill, Text)
    plate = pill.background_rectangle
    assert plate in pill.submobjects
    glyphs = [
        m for m in pill.family_members_with_points() if m not in plate.get_family()
    ]
    lows = np.min([m.get_corner(np.array([-1, -1, 0])) for m in glyphs], axis=0)
    highs = np.max([m.get_corner(np.array([1, 1, 0])) for m in glyphs], axis=0)
    # The plate holds the glyphs with room to spare on every side.
    assert np.all(plate.get_corner(np.array([-1, -1, 0]))[:2] < lows[:2])
    assert np.all(plate.get_corner(np.array([1, 1, 0]))[:2] > highs[:2])


def test_annotation_classes_are_the_engine_s():
    import arachne.mobject.annotation as annotation

    for name in annotation.__all__:
        found = getattr(annotation, name)
        if isinstance(found, type):
            assert engine_class(found), name
    assert engine_class(Annotation)


def _draws_nothing(annotation) -> bool:
    """Whether no member of ``annotation`` but a point drawing nothing holds
    points.
    """
    return not any(
        len(m.points)
        for m in annotation.get_family()
        if not isinstance(m, VectorizedPoint)
    )


@pytest.mark.parametrize("follow", [True, False])
def test_an_annotation_on_a_drawing_hidden_at_first_appears_with_it(follow):
    figure = Rig("fixture:figure")
    (pointing,) = [d for d in figure.part("hand").submobjects if d.name == "point"]
    callout = Callout("points", at=anchor(pointing, at=UR), offset=UR, follow=follow)
    # Hidden while what it names is.
    assert _draws_nothing(callout)
    _update(callout)
    assert _draws_nothing(callout)
    figure.show(hand="point")
    _update(callout)
    # Shown with it, as if made while it showed.
    shown = Rig("fixture:figure").show(hand="point")
    (also,) = [d for d in shown.part("hand").submobjects if d.name == "point"]
    made = Callout("points", at=anchor(also, at=UR), offset=UR, follow=follow)
    corner = pointing.get_critical_point(UR)
    np.testing.assert_allclose(callout.pill.get_center(), corner + UR, atol=1e-9)
    np.testing.assert_allclose(callout.leader.points[-1], corner, atol=1e-12)
    assert [len(m.points) for m in callout.get_family()] == [
        len(m.points) for m in made.get_family()
    ]
    assert callout.pill.width == pytest.approx(made.pill.width, abs=1e-12)
    np.testing.assert_allclose(callout.points_at(), corner, atol=1e-9)
    # Placed once, it follows, or stays, as any annotation does.
    figure.shift(LEFT)
    _update(callout)
    moved = corner + (LEFT if follow else ORIGIN)
    np.testing.assert_allclose(callout.pill.get_center(), moved + UR, atol=1e-9)


def test_a_readout_changed_while_its_target_is_hidden_appears_with_its_value():
    figure = Rig("fixture:figure")
    (pointing,) = [d for d in figure.part("hand").submobjects if d.name == "point"]
    reading = Readout(4, "{:.0f}", at=anchor(pointing, at=UR), offset=UR)
    reading.set_value(12)
    _update(reading)
    assert _draws_nothing(reading)
    figure.show(hand="point")
    _update(reading)
    made = Readout(12, "{:.0f}", at=anchor(pointing, at=UR), offset=UR)
    assert reading.label.original_text == "12"
    for got, want in zip(reading.get_family(), made.get_family(), strict=True):
        np.testing.assert_allclose(got.points, want.points, atol=1e-9)


def _pointing_hand():
    """The figure, its hand's ``point`` drawing hidden, and that drawing."""
    figure = Rig("fixture:figure")
    (pointing,) = [d for d in figure.part("hand").submobjects if d.name == "point"]
    return figure, pointing


def _shape(mobject) -> list[np.ndarray]:
    """Each member's points of ``mobject``'s family, about its centre."""
    centre = mobject.get_center()
    return [np.asarray(m.points) - centre for m in mobject.get_family()]


def _assert_same_shape(got, want) -> None:
    got, want = _shape(got), _shape(want)
    assert [len(p) for p in got] == [len(p) for p in want]
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize(
    "make",
    [
        lambda at: BarGauge(0.6, at=at),
        lambda at: ColumnGauge(0.6, at=at),
        lambda at: DialGauge(0.6, at=at),
    ],
)
def test_a_gauge_on_a_target_hidden_at_first_shows_its_value_with_it(make):
    figure, pointing = _pointing_hand()
    gauge = make(anchor(pointing, at=UR))
    gauge.set_value(0.3)
    _update(gauge)
    figure.show(hand="point")
    _update(gauge)
    # Drawn at its value, as one made while the target showed.
    shown, also = _pointing_hand()
    shown.show(hand="point")
    made = make(anchor(also, at=UR)).set_value(0.3)
    _assert_same_shape(gauge.body, made.body)


@pytest.mark.parametrize(
    "make",
    [
        lambda at: Callout("points", at=at, offset=UR),
        lambda at: Readout(4, "{:.0f} mV", at=at, offset=UR),
    ],
)
def test_an_annotation_turned_and_scaled_while_hidden_appears_so(make):
    figure, pointing = _pointing_hand()
    hidden = make(anchor(pointing, at=UR)).scale(2).rotate(np.pi / 6)
    figure.show(hand="point")
    _update(hidden)
    # At the size and angle given it while it waited, as one so changed
    # while its target showed.
    shown, also = _pointing_hand()
    shown.show(hand="point")
    made = make(anchor(also, at=UR)).scale(2).rotate(np.pi / 6)
    _assert_same_shape(hidden.body, made.body)
    np.testing.assert_allclose(
        hidden.points_at(), pointing.get_critical_point(UR), atol=1e-9
    )
