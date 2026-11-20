"""``carriable`` answers, before anything is read, what a read would refuse
about a quantity's name and unit, from the same code and in its words.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import arachne.simulation  # noqa: F401  (registers the sim. prefix)
from arachne import Square
from arachne import quantities as hook
from arachne.quantities import At, Literal, attach, carriable, read

AT_ZERO = At.of(Fraction(0), None)

# A unit that is well formed and that nothing registers.
UNREGISTERED = "furlong/fortnight"


@pytest.fixture(autouse=True)
def units(monkeypatch):
    """The unit registry as a copy, so registering one here stays here."""
    monkeypatch.setattr(hook, "_UNITS", set(hook._UNITS))


def _reason_of_a_read(name: str, unit: str) -> str | None:
    """Why a read refuses a number named ``name`` in ``unit``, or None."""
    square = Square()
    attach(square, Literal(name, 1.0, unit))
    (quantity,) = read(Square, square.__dict__, AT_ZERO)
    return quantity.unreadable


def test_a_good_name_is_carriable():
    for name in ("sim.angle", "sim.a.b-c_d.9", "sim.A"):
        assert carriable(name, "1") is None
        assert _reason_of_a_read(name, "1") is None


NAME_FAULTS = [
    "sim",  # no prefix
    "sim.",  # empty rest
    ".angle",  # empty prefix
    "sim..angle",  # empty segment
    "sim.angle.",  # trailing empty segment
    "sim.an gle",  # white space
    "sim.angle!",  # a character outside letters, digits, '_' and '-'
    "sim.angleé",  # not ASCII
    "no word.angle",  # a bad prefix segment
    "",
]


@pytest.mark.parametrize("name", NAME_FAULTS)
def test_a_name_is_refused_with_the_words_of_the_read(name):
    reason = carriable(name, "1")
    assert reason is not None
    assert reason == _reason_of_a_read(name, "1")


def test_a_prefix_nobody_registered_is_refused_as_a_read_refuses_it():
    reason = carriable("nobody.angle", "1")
    assert reason == _reason_of_a_read("nobody.angle", "1")
    assert "not registered" in reason


def test_a_segment_asks_for_no_prefix():
    for segment in ("angle", "a.b", "Z-9_", "nobody"):
        assert carriable(segment, segment=True) is None
    for segment in ("", "a b", "a..b", ".a", "a.", "a!", "é"):
        reason = carriable(segment, segment=True)
        assert reason is not None
        assert "segments of letters" in reason
    # A whole name, which has a prefix, is not a segment, and the reverse.
    assert carriable("angle") is not None
    assert carriable("sim.angle", segment=True) is None


def test_a_segment_the_read_accepts_in_a_name_is_a_segment():
    for segment in ("angle", "a.b-c_d", "9"):
        assert carriable(segment, segment=True) is None
        assert carriable(f"sim.{segment}") is None
        assert _reason_of_a_read(f"sim.{segment}", "1") is None
    for segment in ("", "a b", "a..b", "a!"):
        assert carriable(segment, segment=True) is not None
        assert carriable(f"sim.{segment}") is not None
        assert _reason_of_a_read(f"sim.{segment}", "1") is not None


def test_a_unit_need_not_be_registered_to_be_carriable():
    assert UNREGISTERED not in hook._UNITS
    assert carriable("sim.angle", UNREGISTERED) is None
    assert carriable(unit=UNREGISTERED) is None
    # A read still refuses it until someone registers it; then the two agree.
    assert "not registered" in _reason_of_a_read("sim.angle", UNREGISTERED)
    hook.register_unit(UNREGISTERED)
    assert _reason_of_a_read("sim.angle", UNREGISTERED) is None


@pytest.mark.parametrize("unit", ["", " ", "per second", "m\ts", "m\n", " "])
def test_a_unit_register_unit_refuses_is_refused_in_its_words(unit):
    reason = carriable("sim.angle", unit)
    assert reason is not None
    with pytest.raises(ValueError) as refused:
        hook.register_unit(unit)
    assert reason == str(refused.value)
    # And a read can never carry it: it is not, and cannot be, registered.
    assert _reason_of_a_read("sim.angle", unit) is not None


@pytest.mark.parametrize("unit", ["1", "name", "u/s", "rad", "m^2", UNREGISTERED])
def test_a_unit_register_unit_accepts_is_carriable(unit):
    assert carriable(unit=unit) is None
    hook.register_unit(unit)


def test_what_is_not_a_string_is_a_reason_not_an_exception():
    assert carriable() is None  # nothing asked
    for bad in (3, b"sim.a", ("sim", "a")):
        assert carriable(bad) is not None
        assert carriable(bad, segment=True) is not None
        assert carriable(unit=bad) is not None


def test_the_name_is_answered_before_the_unit_as_a_read_checks_them():
    # A bad name and a bad unit together: the name's reason, as in a read.
    assert carriable("sim", " ") == carriable("sim")
    assert carriable("sim.angle", " ") == carriable(unit=" ")


def test_the_length_of_a_name_is_no_part_of_it():
    # The module limits a name VALUE's bytes, not a quantity's name.
    long_name = "sim." + "a" * (hook.MAX_NAME_BYTES * 4)
    assert carriable(long_name) is None
    assert _reason_of_a_read(long_name, "1") is None
