"""``become`` is value assignment with CE's family alignment.

Each case runs CE's ``become`` (its body, written out below with the copies
it made) on one pair of families and Arachne's on an identical pair, then
compares both families member by member: CE pads the family with fewer
members, the argument included, with faded copies, never drops one, and moves
the points and the seven style fields position by position.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Dot,
    Line,
    Mobject,
    Square,
    Triangle,
    VGroup,
    VMobject,
)
from arachne.mobject.values import Paint, count_work

NO_WORK = {"arrays": 0, "paints": 0, "hashes": 0, "deep_copies": 0}


def placed_curve() -> VMobject:
    """A mobject large enough for shift, scale and rotate to place it."""
    points = np.random.default_rng(0).random((10_000, 3))
    return VMobject().set_points(points).set_fill(RED, 0.5)


def ce_become(receiver: Mobject, mobject: Mobject) -> Mobject:
    """CE 0.22.1's ``become`` without its ``match_*`` options, as it was written
    (``mobject.py:3426-3451``): align, then copy points and interpolate the
    colour to alpha 1, which copied each of the seven fields.
    """
    receiver.align_data(mobject, skip_point_alignment=True)
    for sm1, sm2 in zip(receiver.get_family(), mobject.get_family(), strict=True):
        sm1.points = np.array(sm2.points)
        sm1.interpolate_color(sm1, sm2, 1)
    return receiver


def assert_same_families(mine: Mobject, theirs: Mobject) -> None:
    family, reference = mine.get_family(), theirs.get_family()
    assert [type(m) for m in family] == [type(m) for m in reference]
    for a, b in zip(family, reference, strict=True):
        assert len(a.submobjects) == len(b.submobjects)
        np.testing.assert_array_equal(a.points, b.points)
        if isinstance(a, VMobject):
            for name in Paint.FIELDS:
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


CASES = {
    "equal families": lambda: (
        VGroup(Square(), Circle()),
        VGroup(Square().shift(UP).set_fill(RED, 1), Circle().set_stroke(BLUE, 8)),
    ),
    "argument has more members": lambda: (
        VGroup(Square()),
        VGroup(Circle(), Dot().shift(RIGHT), Triangle().set_fill(GREEN, 0.5)),
    ),
    "receiver has more members": lambda: (
        VGroup(Square(), Circle(), Dot(), Triangle()),
        VGroup(Circle().set_color(YELLOW)),
    ),
    "nested, uneven at two levels": lambda: (
        VGroup(VGroup(Square(), Dot()), Circle()),
        VGroup(VGroup(Triangle()), VGroup(Dot(), Dot(), Dot()), Line(DOWN, UP)),
    ),
    "a pointed group onto an empty one": lambda: (
        VGroup(Square(), Circle()),
        Square().add(Dot()),
    ),
    "a leaf onto a group": lambda: (
        Square(),
        VGroup(Circle(), Dot()),
    ),
    "uneven point counts in members": lambda: (
        VGroup(Line(), Square()),
        VGroup(Circle(), Line()),
    ),
    "gradients that broadcast": lambda: (
        Square().set_fill([RED, BLUE], 1),
        Circle().set_fill(YELLOW, 1),
    ),
}


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_become_matches_ces_alignment_and_values(make):
    receiver, argument = make()
    ce_receiver, ce_argument = make()
    receiver.become(argument)
    ce_become(ce_receiver, ce_argument)
    assert_same_families(receiver, ce_receiver)
    # CE pads the argument too: it is changed the same way.
    assert_same_families(argument, ce_argument)


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_become_shares_the_arguments_values(make):
    receiver, argument = make()
    receiver.become(argument)
    for mine, theirs in zip(receiver.get_family(), argument.get_family(), strict=True):
        assert mine.points is theirs.points
        assert mine.paint is theirs.paint
        assert mine is not theirs


def test_become_never_drops_a_member():
    receiver = VGroup(Square(), Circle(), Dot())
    receiver.become(VGroup(Triangle()))
    assert len(receiver) == 3
    # The padding is faded copies of the argument's member (become moves the
    # rgbas, not the scalar opacity attributes, as in CE).
    for padded in receiver[1:]:
        assert not padded.fill_rgbas[:, 3].any()
        assert not padded.stroke_rgbas[:, 3].any()


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda: (VGroup(Square(), Circle().shift(UP)), None), id="its own copy"
        ),
        pytest.param(
            lambda: (
                VGroup(Square().set_fill(RED, 1), Circle()),
                VGroup(Square().shift(DOWN), Circle().set_stroke(GREEN, 6)),
            ),
            id="an equal family",
        ),
        pytest.param(
            lambda: (lambda placed: (placed, placed.copy()))(
                placed_curve().shift(RIGHT)
            ),
            id="a placed family and its copy",
        ),
        pytest.param(
            lambda: (placed_curve(), placed_curve().shift(UP).rotate(0.3)),
            id="onto a placed mobject",
        ),
    ],
)
def test_become_between_equal_families_does_no_work(make):
    receiver, argument = make()
    if argument is None:
        argument = receiver.copy()
    with count_work() as work:
        receiver.become(argument)
    assert work.as_dict() == NO_WORK


class SetsPointsAbove(VMobject):
    """A subclass whose points setter moves what it is given up, as a user's
    override might.
    """

    @VMobject.points.setter
    def points(self, value):
        VMobject.points.fset(self, np.asarray(value) + UP)


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param(lambda: placed_curve().shift(RIGHT), id="placed"),
        pytest.param(lambda: Square().shift(RIGHT), id="stored"),
    ],
)
def test_become_goes_through_an_overridden_points_setter(argument):
    # CE's become assigned each member's points through the property.
    source = argument()
    receiver = SetsPointsAbove().set_points(np.zeros((len(source.points), 3)))
    receiver.become(source)
    np.testing.assert_allclose(receiver.points, source.points + UP, rtol=0, atol=1e-12)


def test_an_always_redraw_style_rebuild_does_no_value_work():
    # A rebuilt mobject of the same family shape costs its own construction
    # and nothing in become.
    redrawn = VGroup(Square(), Circle())
    rebuilt = VGroup(Square().shift(RIGHT), Circle().set_fill(RED, 1))
    with count_work() as work:
        redrawn.become(rebuilt)
    assert work.as_dict() == NO_WORK


def test_gradients_that_dont_broadcast_raise_as_in_ce():
    receiver = Square().set_fill([RED, BLUE], 1)
    argument = Circle().set_fill([RED, GREEN, YELLOW], 1)
    with pytest.raises(ValueError, match="broadcast"):
        ce_become(Square().set_fill([RED, BLUE], 1), argument.copy())
    with pytest.raises(ValueError, match="broadcast"):
        receiver.become(argument)


def test_a_subclass_that_interpolates_its_own_way_keeps_it():
    calls = []

    class Recolouring(Square):
        def interpolate_color(self, mobject1, mobject2, alpha):
            calls.append(alpha)
            return super().interpolate_color(mobject1, mobject2, alpha)

    Recolouring().become(Circle())
    assert calls == [1]


def test_restore_becomes_the_saved_state():
    square = Square().set_fill(RED, 1)
    square.save_state()
    square.shift(RIGHT).set_fill(BLUE, 0.2)
    square.restore()
    assert square.points is square.saved_state.points
    assert square.paint is square.saved_state.paint


def test_become_takes_a_placement_without_computing_its_points():
    receiver, argument = placed_curve(), placed_curve().shift(UP).scale(2)
    placement = argument._current_placement()
    receiver.become(argument)
    assert receiver._current_placement() is placement
    assert placement.materialized is None
    np.testing.assert_array_equal(receiver.points, argument.points)
    assert receiver.points is argument.points
