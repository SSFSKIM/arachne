"""The number a mark carries, declared: ``state_at`` answers it at the
session server, read from what the record captured; each is in the content
keys, a charge given as any kind of number included.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import (
    LEFT,
    RIGHT,
    ChargedParticle,
    Dot,
    DottedTrail,
    FlowArrow,
    Particles,
    Vessel,
    WavefrontArcs,
    WavyLine,
)

from ..helpers.declared import DeclaredServer, apart, keyed, values
from . import declared_scenes

SCENES = Path(declared_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}


@pytest.fixture
def marks(tmp_path):
    server = DeclaredServer(
        SCENES, "NumberedMarks", {**SMALL, "media_dir": str(tmp_path / "media")}
    )
    yield server
    server.close()


@pytest.mark.parametrize(
    ("name", "quantity", "value", "unit"),
    [
        ("vessel", "mark.level", 0.5, "u"),
        ("scatter", "mark.count", 12, "1"),
        ("fronts", "mark.spacing", 0.25, "u"),
        ("trail", "mark.interval", 0.25, "s"),
        ("wave", "mark.wavelengths", 3.0, "1"),
        ("arrow", "mark.magnitude", 2.5, "1"),
        ("ion", "mark.charge", -2, "1"),
    ],
)
def test_a_mark_answers_the_number_it_carries(marks, name, quantity, value, unit):
    found = marks.quantities(0.25, name)
    assert values(found) == {quantity: value}
    assert found[quantity]["unit"] == unit


def test_a_vessel_answers_its_level_as_the_scene_s_code_changes_it(marks):
    assert values(marks.quantities(0.25, "vessel")) == {"mark.level": 0.5}
    assert values(marks.quantities(0.75, "vessel")) == {"mark.level": 1.5}


def test_a_plain_dot_declares_nothing(marks):
    assert values(marks.quantities(0.25, "walker")) == {}


@pytest.mark.parametrize(
    ("made", "attribute", "changed"),
    [
        (lambda: Vessel(level=0.5), "level", 1.5),
        (lambda: Particles(count=3), "count", 4),
        (lambda: WavefrontArcs(spacing=0.25), "spacing", 0.5),
        (lambda: DottedTrail(Dot(), interval=0.25), "interval", 0.5),
        (lambda: WavyLine(LEFT, RIGHT, wavelengths=3), "wavelengths", 4.0),
        (lambda: FlowArrow(magnitude=1.0), "magnitude", 2.0),
        (lambda: ChargedParticle(1), "charge", -1),
    ],
    ids=["level", "count", "spacing", "interval", "wavelengths", "magnitude", "charge"],
)
def test_a_mark_s_number_is_in_the_content_keys(made, attribute, changed):
    mark = made()
    before = keyed(mark, attribute)
    setattr(mark, attribute, changed)
    after = keyed(mark, attribute)
    assert apart(before, after)


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        (np.int64(2), 2),
        (np.float32(0.5), 0.5),
        (Fraction(1, 2), 0.5),
        (-1, -1),
        (2**60 + 1, 2**60 + 1),
    ],
    ids=["numpy-int", "numpy-float", "fraction", "int", "large-int-exact"],
)
def test_a_charge_is_kept_as_a_plain_number(given, kept):
    ion = ChargedParticle(given)
    assert ion.charge == kept
    assert type(ion.charge) is type(kept)
    keyed(ion, "charge")


@pytest.mark.parametrize(
    "given",
    [
        Fraction(1, 10**400),
        Fraction(-1, 10**400),
        float("nan"),
        float("inf"),
        10**309,
        -(10**309),
    ],
    ids=[
        "tiny-positive",
        "tiny-negative",
        "nan",
        "inf",
        "huge-int",
        "huge-negative-int",
    ],
)
def test_a_charge_no_plain_number_holds_is_refused(given):
    # As a float the tiny fractions are 0: the sign drawn and the number
    # declared would not be the charge given.
    with pytest.raises(ValueError, match="charge"):
        ChargedParticle(given)


@pytest.mark.parametrize(
    ("given", "strokes", "shade"),
    [(np.float32(-0.5), 1, "blue"), (Fraction(3, 2), 2, "red"), (0, 0, "grey")],
    ids=["negative", "positive", "neutral"],
)
def test_a_particle_draws_its_kept_charge(given, strokes, shade):
    from arachne import BLUE, GREY, RED

    ion = ChargedParticle(given)
    # Each stroke of the sign is one straight segment of four points.
    assert len(ion.sign.points) == 4 * strokes
    want = {"blue": BLUE, "grey": GREY, "red": RED}[shade]
    assert ion.disc.get_color().to_hex() == want.to_hex()
