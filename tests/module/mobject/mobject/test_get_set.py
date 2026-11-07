from __future__ import annotations

import types

import pytest

from arachne import Circle, Line, MathTex
from arachne.mobject.mobject import Mobject


def test_generic_set():
    m = Mobject()
    m.set(test=0)

    assert m.test == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_compat_layer():
    m = Mobject()

    assert isinstance(m.get_test, types.MethodType)
    with pytest.raises(AttributeError):
        m.get_test()

    m.test = 0
    assert m.get_test() == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_set_compat_layer():
    m = Mobject()
    m.test = 1

    assert isinstance(m.set_test, types.MethodType)
    m.set_test(0)

    assert m.test == 0


# Deliberate break 9 (a misspelt synthesised setter raises), approved
# 2026-11-07: CE's test set an attribute the mobject didn't have through
# set_test; that now raises, and it is set on an existing one above.


def test_a_synthesised_setter_of_no_attribute_raises_naming_the_nearest():
    with pytest.raises(AttributeError) as raised:
        Circle().set_colr
    message = str(raised.value)
    assert "set_colr" in message
    assert "did you mean set_color" in message
    assert "mob.colr = value" in message


def test_a_setter_of_a_name_nothing_has_raises():
    with pytest.raises(AttributeError, match="set_max_width") as raised:
        Line().set_max_width(6.2)
    assert "did you mean" in str(raised.value)


def test_a_synthesised_setter_of_no_attribute_never_sets_it():
    line = Line()
    with pytest.raises(AttributeError):
        line.set_max_width(6.2)
    assert "max_width" not in vars(line)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_synthesised_setter_of_a_property_still_sets_it():
    tex = MathTex("x")
    tex.set_height(1)
    assert tex.height == pytest.approx(1)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_synthesised_setter_of_an_instance_attribute_still_sets_it():
    m = Mobject()
    m.offset = 1
    m.set_offset(2)
    assert m.offset == 2


def test_a_subclass_s_own_setter_is_untouched():
    class Shifted(Mobject):
        def set_offset(self, value):
            self.shifted_by = value
            return self

    m = Shifted().set_offset(3)
    assert m.shifted_by == 3


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_synthesised_getter_is_unchanged():
    m = Mobject()
    assert isinstance(m.get_test, types.MethodType)
    with pytest.raises(AttributeError, match="'test'"):
        m.get_test()


def test_nonexistent_attr():
    m = Mobject()

    with pytest.raises(AttributeError, match="object has no attribute"):
        m.test
