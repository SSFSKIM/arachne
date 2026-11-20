"""A rig's parameters and slots and an illustration's timing, declared:
``state_at`` answers each at the session server, read from the values the
record captured, so a stepped rig answers the value on display; and each
value is in the content keys.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pytest

from arachne import Boil, Rig, Square
from arachne.assets.timing import Timing
from arachne.timeline.clock import frames_in

from ..helpers.declared import DeclaredServer, apart, keyed, values
from . import declared_scenes
from .declared_scenes import BOIL, OPEN, RATE, REACH, TURN

SCENES = Path(declared_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}


@pytest.fixture
def server(tmp_path):
    opened: list[DeclaredServer] = []

    def open_scene(scene: str, fps: int = 4) -> DeclaredServer:
        config = {**SMALL, "frame_rate": fps, "media_dir": str(tmp_path / "media")}
        opened.append(DeclaredServer(SCENES, scene, config))
        return opened[-1]

    yield open_scene
    for each in opened:
        each.close()


def test_a_rig_answers_every_parameter_and_slot(server):
    figure = server("PosedFigure")
    at_rest = figure.quantities(0.25, "figure")
    assert values(at_rest) == {
        "rig.param.reach": 0.0,
        "rig.param.head_turn": 0.0,
        "rig.slot.eyes": "open",
        "rig.slot.hand": "open",
    }
    # Each in its manifest's unit; a slot's value is its drawing's name.
    assert {name: each["unit"] for name, each in at_rest.items()} == {
        "rig.param.reach": "1",
        "rig.param.head_turn": "deg",
        "rig.slot.eyes": "name",
        "rig.slot.hand": "name",
    }
    assert not [each for each in at_rest.values() if "tolerance" in each]
    # Half way through the linear pose, the hand switched as it began.
    assert values(figure.quantities(1.0, "figure")) == {
        "rig.param.reach": pytest.approx(REACH / 2),
        "rig.param.head_turn": pytest.approx(TURN / 2),
        "rig.slot.eyes": "open",
        "rig.slot.hand": "point",
    }
    # At the end, the targets exactly, and the slot the code switched after.
    assert values(figure.quantities(2.0, "figure")) == {
        "rig.param.reach": REACH,
        "rig.param.head_turn": TURN,
        "rig.slot.eyes": "closed",
        "rig.slot.hand": "point",
    }


def _true_opening(t: Fraction) -> float:
    """The stepped box's true opening at ``t``: from 0.37 s to 0.87 s."""
    start, end = Fraction(37, 100), Fraction(87, 100)
    share = min(max((t - start) / (end - start), Fraction(0)), Fraction(1))
    return float(OPEN * share)


def _boundary(t: Fraction) -> Fraction:
    return Fraction(math.floor(t * RATE), RATE)


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_stepped_rig_answers_the_value_on_display(server, fps):
    box = server("SteppedBoxEndingOffGrid", fps)
    end = Fraction(87, 100)
    drawn = list(frames_in(Fraction(0), end, Fraction(fps)))
    assert drawn[-1] < end
    for t in drawn:
        found = values(box.quantities(t, "box"))
        # The drawing on display is the pose true at the last boundary.
        assert found["rig.param.lid_open"] == pytest.approx(
            _true_opening(_boundary(t))
        ), t
        assert found["timing.rate"] == RATE
    # The pair the end of the scene pins: the last drawn instant answers the
    # value held from 10/12, and the scene's end the script's final state,
    # which no delivered frame shows. The difference is intended.
    held = _true_opening(Fraction(10, 12))
    assert held != OPEN
    last = values(box.quantities(drawn[-1], "box"))
    assert last["rig.param.lid_open"] == pytest.approx(held)
    final = values(box.quantities(end, "box"))
    assert final["rig.param.lid_open"] == OPEN


def test_an_illustration_answers_its_timing_from_its_declaration(server):
    group = server("DeclaredLater")
    # Not declared yet: no timing.
    assert values(group.quantities(0.25, "shapes")) == {}
    with_boil = group.quantities(0.75, "shapes")
    assert values(with_boil) == {
        "timing.rate": RATE,
        "timing.boil_rate": BOIL.rate,
        "timing.boil_amplitude": BOIL.amplitude,
    }
    assert {name: each["unit"] for name, each in with_boil.items()} == {
        "timing.rate": "1/s",
        "timing.boil_rate": "1/s",
        "timing.boil_amplitude": "px",
    }
    # Declared again, at another rate and without boil.
    assert values(group.quantities(1.25, "shapes")) == {"timing.rate": 4}


def test_a_rig_s_parameter_values_are_in_the_content_keys():
    rig = Rig("fixture:figure")
    before = keyed(rig, "_values")
    rig.write_values(rig.checked_values({"reach": 0.5}))
    after = keyed(rig, "_values")
    assert apart(before, after)
    rig.write_values(rig.checked_values({"reach": 0.0}))
    assert keyed(rig, "_values") == before


def test_a_rig_s_slot_choices_are_in_the_content_keys():
    rig = Rig("fixture:figure")
    before = keyed(rig, "_shown")
    rig.write_choices(rig.checked_drawings({"hand": "point"}))
    after = keyed(rig, "_shown")
    assert apart(before, after)
    rig.write_choices(rig.checked_drawings({"hand": "open"}))
    assert keyed(rig, "_shown") == before


@pytest.mark.parametrize(
    "other",
    [Timing(4), Timing(RATE, BOIL), Timing(RATE, Boil(rate=6, amplitude=2.5, seed=3))],
    ids=["rate", "boil", "amplitude"],
)
def test_an_illustration_s_timing_is_in_the_content_keys(other):
    shape = Square()
    shape._timing = Timing(RATE).mark
    before = keyed(shape, "_timing")
    shape._timing = other.mark
    after = keyed(shape, "_timing")
    assert apart(before, after)
