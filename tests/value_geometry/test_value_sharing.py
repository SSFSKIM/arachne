"""Copies share values: copying a mobject, or restating a value it already has,
builds, hashes and deep-copies nothing, and the two sides part only when one of
them changes.

The counts come from the instrumented value constructors (:class:`Work`).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import (
    BLUE,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    Circle,
    Dot,
    ImageMobject,
    PMobject,
    Square,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
)
from arachne.mobject.values import FrozenPoints, Paint, adopt, count_work

NO_WORK = {"arrays": 0, "paints": 0, "hashes": 0, "deep_copies": 0}


def big_vmobject(n: int = 10_000) -> VMobject:
    points = np.random.default_rng(0).random((n, 3))
    return VMobject().set_points(points).set_fill(RED, 0.5).set_stroke(BLUE, 3)


@pytest.mark.parametrize(
    "copier", [lambda m: m.copy(), copy.deepcopy], ids=["copy", "deepcopy"]
)
def test_a_copy_of_a_10000_point_mobject_shares_its_values(copier):
    original = big_vmobject()
    before = np.array(original.points)
    clone = copier(original)
    assert clone is not original
    assert clone.points is original.points
    assert clone.paint is original.paint

    # Until one side changes: the change replaces that side's value only.
    clone.shift(RIGHT)
    assert clone.points is not original.points
    np.testing.assert_array_equal(original.points, before)
    np.testing.assert_array_equal(clone.points, before + RIGHT)

    paint = original.paint
    original.set_fill(BLUE)
    assert clone.paint is paint
    assert original.paint is not paint


@pytest.mark.parametrize(
    "copier", [lambda m: m.copy(), copy.deepcopy], ids=["copy", "deepcopy"]
)
def test_copying_a_family_does_no_value_work(copier):
    group = VGroup(big_vmobject(), Square().set_fill(RED, 1), Circle(), VGroup(Dot()))
    with count_work() as work:
        clone = copier(group)
    assert work.as_dict() == NO_WORK
    for mine, theirs in zip(clone.get_family(), group.get_family(), strict=True):
        assert mine is not theirs
        assert mine.points is theirs.points
        assert mine.paint is theirs.paint
        # And the Bézier parameters, one constant for every VMobject.
        assert mine._bezier_t_values is theirs._bezier_t_values


def test_a_copy_keeps_its_own_tree():
    group = VGroup(Square(), Circle())
    clone = group.copy()
    clone.add(Dot())
    clone[0].set_fill(RED, 1)
    assert len(group) == 2
    assert group[0].paint is not clone[0].paint


@pytest.mark.parametrize(
    "restate",
    [
        pytest.param(lambda m: setattr(m, "points", m.points), id="points ="),
        pytest.param(lambda m: m.set_points(m.points), id="set_points"),
        pytest.param(lambda m: setattr(m, "paint", m.paint), id="paint ="),
        pytest.param(lambda m: m.shift(ORIGIN), id="shift by zero"),
        pytest.param(lambda m: m.shift(np.zeros(3), 0 * UP), id="shift by zeros"),
    ],
)
def test_restating_a_value_does_no_work(restate):
    mob = big_vmobject()
    points, paint = mob.points, mob.paint
    with count_work() as work:
        restate(mob)
    assert work.as_dict() == NO_WORK
    assert mob.points is points
    assert mob.paint is paint


def test_restating_point_colours_and_pixels_does_no_work():
    cloud = PMobject().add_points(np.random.default_rng(1).random((100, 3)))
    image = ImageMobject(np.zeros((4, 4, 4), dtype=np.uint8))
    rgbas, pixels = cloud.rgbas, image.pixel_array
    with count_work() as work:
        cloud.rgbas = cloud.rgbas
        image.pixel_array = image.pixel_array
    assert work.as_dict() == NO_WORK
    assert cloud.rgbas is rgbas
    assert image.pixel_array is pixels


def test_a_tracker_copy_shares_its_point():
    tracker = ValueTracker(2)
    with count_work() as work:
        clone = tracker.copy()
    assert work.as_dict() == NO_WORK
    assert clone.points is tracker.points
    clone.set_value(5)
    assert tracker.get_value() == 2


def test_transform_begin_deep_copies_no_values():
    start = big_vmobject(400)
    target = big_vmobject(400).shift(UP).set_fill(BLUE)
    animation = Transform(start, target)
    with count_work() as work:
        animation.begin()
    assert work.deep_copies == 0
    # The target's copy and the starting copy share what they copied.
    assert animation.target_copy.points is target.points
    assert animation.target_copy.paint is target.paint


def test_padding_by_alignment_shares_the_padded_members_points():
    group = VGroup(Square())
    group.align_submobjects(VGroup(Square(), Circle(), Dot()))
    assert len(group) == 3
    assert group[1].points is group[0].points
    assert group[2].points is group[0].points
    assert group[1].fill_opacity == 0


def test_an_always_redraw_rebuild_shares_the_rebuilt_values():
    tracker = ValueTracker(0)
    built = []

    def build():
        square = big_vmobject(400).shift(tracker.get_value() * RIGHT)
        built.append(square)
        return square

    redrawn = always_redraw(build)
    tracker.set_value(1)
    with count_work() as work:
        redrawn.update()
    newest = built[-1]
    assert redrawn.points is newest.points
    assert redrawn.fill_rgbas is newest.fill_rgbas
    assert redrawn.stroke_rgbas is newest.stroke_rgbas
    # Building the new square is the scene's own work; become copies nothing.
    assert work.deep_copies == 0


def test_interpolating_to_the_end_shares_the_end_values():
    start, end = Square(), Square().set_fill(RED, 1).set_stroke(BLUE, 8)
    mid = Square()
    mid.interpolate(start, end, 1.0)
    assert mid.fill_rgbas is end.fill_rgbas
    assert mid.sheen_direction is end.sheen_direction


def test_a_read_only_view_of_writable_memory_is_copied_not_shared():
    # A camera's framebuffer is such memory (ImageMobjectFromCamera shows it
    # through a view like this one): a copy must keep what it saw.
    memory = np.zeros((4, 3))
    view = memory.view(FrozenPoints)
    view.flags.writeable = False
    clone = copy.deepcopy(view)
    assert clone is not view
    memory[0, 0] = 7
    assert view[0, 0] == 7
    assert clone[0, 0] == 0


def test_an_adopted_view_of_a_fresh_buffer_is_shared():
    # np.linspace returns a view of the buffer it builds; adopting it freezes
    # the buffer as well, so the value is immutable and copies share it.
    value = adopt(np.linspace(0, 1, 4), FrozenPoints)
    assert copy.deepcopy(value) is value


def test_a_paint_is_shared_unless_a_field_holds_a_mutable_object():
    paint = Paint(stroke_width=4.0, sheen_direction=np.array([1.0, 0, 0]))
    assert copy.deepcopy(paint) is paint
    widths = [1.0, 2.0]
    holding_a_list = Paint(stroke_width=widths)
    clone = copy.deepcopy(holding_a_list)
    assert clone is not holding_a_list
    assert clone.stroke_width == widths
    assert clone.stroke_width is not widths
