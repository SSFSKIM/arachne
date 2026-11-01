"""A VMobject's style is one immutable Paint: its colour arrays are read-only, the
style setters build a new paint, and paints compare by content.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from arachne import BLUE, RED, UP, Circle, Square, VGroup, VMobject
from arachne.mobject.values import EMPTY_PAINT, FrozenColors, Paint
from arachne.utils.bezier import interpolate
from arachne.utils.color import ManimColor


@pytest.mark.parametrize(
    "name", ["fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"]
)
def test_colour_arrays_are_read_only_naming_the_setters(name):
    square = Square().set_fill(RED, 0.5)
    array = getattr(square, name)
    assert isinstance(array, FrozenColors)
    before = np.array(array)
    with pytest.raises(ValueError, match="set_fill"):
        array[:, 3] = 0
    with pytest.raises(ValueError, match="set_fill"):
        np.copyto(array, 0)
    with pytest.raises(ValueError, match="set_color"):
        getattr(square, name)[:, 3] *= 2
    np.testing.assert_array_equal(getattr(square, name), before)


def test_getters_hand_out_read_only_arrays():
    square = Square().set_fill(RED, 0.5)
    for array in (
        square.get_fill_rgbas(),
        square.get_stroke_rgbas(),
        square.get_fill_opacities(),
        square.get_stroke_opacities(background=True),
    ):
        with pytest.raises(ValueError, match="set_fill"):
            array[0] = 0


def test_the_sheen_direction_is_read_only():
    square = Square().set_sheen_direction(UP)
    with pytest.raises(ValueError, match="set_sheen_direction"):
        square.sheen_direction[0] = 1
    np.testing.assert_array_equal(square.get_sheen_direction(), UP)


def test_a_style_setter_builds_a_new_paint_and_leaves_the_old_one():
    square = Square().set_fill(RED, 0.5)
    old_paint, old_fill = square.paint, square.fill_rgbas
    snapshot = np.array(old_fill)
    square.set_fill(BLUE, 1)
    assert square.paint is not old_paint
    np.testing.assert_array_equal(old_paint.fill_rgbas, snapshot)
    np.testing.assert_array_equal(old_fill, snapshot)
    np.testing.assert_allclose(square.fill_rgbas, [[*ManimColor(BLUE).to_rgb(), 1]])
    # The fields the setter didn't touch are shared, not copied.
    assert square.paint.stroke_rgbas is old_paint.stroke_rgbas


def test_colour_only_and_opacity_only_updates_keep_the_other_channel():
    square = Square().set_fill(RED, 0.25)
    square.set_fill(BLUE)
    np.testing.assert_allclose(square.fill_rgbas, [[*ManimColor(BLUE).to_rgb(), 0.25]])
    square.set_fill(opacity=0.75)
    np.testing.assert_allclose(square.fill_rgbas, [[*ManimColor(BLUE).to_rgb(), 0.75]])
    square.set_stroke(width=7)
    assert square.stroke_width == 7


def test_a_gradient_stretches_the_rgbas_as_in_ce():
    square = Square().set_fill([RED, BLUE], 1)
    assert square.fill_rgbas.shape == (2, 4)
    square.set_fill(opacity=0.5)
    np.testing.assert_allclose(square.fill_rgbas[:, 3], [0.5, 0.5])
    np.testing.assert_allclose(square.fill_rgbas[0, :3], ManimColor(RED).to_rgb())
    square.set_fill([RED, BLUE, RED])
    assert square.fill_rgbas.shape == (3, 4)


def test_paints_compare_by_content():
    a, b = Square().set_fill(RED, 0.5), Square(side_length=3).set_fill(RED, 0.5)
    assert a.paint is not b.paint
    assert a.paint == b.paint
    assert hash(a.paint) == hash(b.paint)
    b.set_fill(opacity=0.6)
    assert a.paint != b.paint
    assert Paint(stroke_width=4) == Paint(stroke_width=4.0)
    assert Paint(stroke_width=4) != Paint(stroke_width=5)


def test_a_paint_is_immutable():
    paint = Square().paint
    with pytest.raises(AttributeError, match="replace"):
        paint.stroke_width = 9
    with pytest.raises(TypeError, match="no field"):
        paint.replace(fill_color=RED)
    assert paint.replace(stroke_width=9).stroke_width == 9
    assert paint.stroke_width != 9


def test_a_paint_refuses_deletion_and_keeps_its_key():
    paint = Square().paint
    key = paint.key
    with pytest.raises(AttributeError, match="replace"):
        del paint.stroke_width
    with pytest.raises(AttributeError, match="replace"):
        del paint.sheen_factor
    assert paint.stroke_width == Square().paint.stroke_width
    assert paint.key == key == Paint(**paint.fields()).key


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("dtype", np.int64, id="dtype"),
        pytest.param("shape", (-1,), id="shape"),
        pytest.param("strides", (8, 8), id="strides"),
    ],
)
def test_a_paint_array_cant_be_reinterpreted_under_its_key(name, value):
    paint = Square().paint
    key = paint.key
    before = np.array(paint.fill_rgbas)
    with pytest.raises(ValueError, match="set_fill"):
        setattr(paint.fill_rgbas, name, value)
    np.testing.assert_array_equal(paint.fill_rgbas, before)
    assert paint.fill_rgbas.dtype == before.dtype
    assert paint.key == key == Paint(**paint.fields()).key


def test_an_unset_field_reads_as_missing():
    assert not hasattr(EMPTY_PAINT, "fill_rgbas")
    mob = VMobject.__new__(VMobject)
    assert not hasattr(mob, "fill_rgbas")
    np.testing.assert_array_equal(mob.get_fill_rgbas(), np.zeros((1, 4)))


def test_none_is_a_value_a_field_can_hold():
    mob = VMobject()
    mob.stroke_width = None  # SVGMobject does this, then reads it back
    assert mob.stroke_width is None
    mob.stroke_width = 0
    assert mob.paint.stroke_width == 0


def test_interpolate_color_matches_the_field_by_field_rule():
    start = Square().set_fill(RED, 0.2).set_stroke(BLUE, 2)
    end = Square().set_fill(BLUE, 1.0).set_stroke(RED, 6)
    mid = Square()
    mid.interpolate_color(start, end, 0.25)
    for name in Paint.FIELDS:
        a, b = getattr(start, name), getattr(end, name)
        np.testing.assert_array_equal(getattr(mid, name), interpolate(a, b, 0.25))
    mid.interpolate_color(start, end, 1.0)
    assert mid.paint == end.paint
    # At alpha 1 CE copied the end's arrays; values are shared instead.
    for name in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
        assert getattr(mid, name) is getattr(end, name)


def test_match_style_and_family_setters():
    group = VGroup(Square(), Circle()).set_fill(RED, 0.5)
    target = Square().set_fill(BLUE, 0.9).set_stroke(width=8)
    group[0].match_style(target)
    assert group[0].paint == target.paint
    assert group[1].paint != target.paint


def test_deepcopy_shares_the_paint_until_one_side_changes():
    square = Square().set_fill(RED, 0.5)
    clone = copy.deepcopy(square)
    assert clone.paint is square.paint
    clone.set_fill(BLUE)
    assert clone.paint != square.paint
    np.testing.assert_array_equal(
        square.fill_rgbas, Square().set_fill(RED, 0.5).fill_rgbas
    )
