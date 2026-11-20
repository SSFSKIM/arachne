"""A rig's parameters and slots and an illustration's timing, declared:
``state_at`` answers each at the session server, read from the values the
record captured, so a stepped rig answers the value on display; each value
is in the content keys; and a manifest declaring a name or a unit no
declared quantity can carry is refused when it loads, while its own units
are registered when it does.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne import Boil, Rig, Square, quantities
from arachne.assets import load_asset
from arachne.assets.timing import Timing
from arachne.quantities import MAX_NAME_BYTES, At, read
from arachne.timeline.clock import frames_in

from ..helpers.declared import DeclaredServer, apart, keyed, values
from ..timing.reading import RATES, SMALL, boundary
from ..timing.reading import SCENES as TIMING_SCENES
from ..timing.timing_scenes import OPEN
from ..timing.timing_scenes import RATE as BOX_RATE
from . import declared_scenes
from .declared_scenes import BOIL, RATE, REACH, TURN
from .helpers import admitted, copied, edited, refused

SCENES = Path(declared_scenes.__file__)


@pytest.fixture
def server(tmp_path):
    opened: list[DeclaredServer] = []

    def open_scene(scene: str, fps: int = 4, scenes: Path = SCENES) -> DeclaredServer:
        config = {**SMALL, "frame_rate": fps, "media_dir": str(tmp_path / "media")}
        opened.append(DeclaredServer(scenes, scene, config))
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
    """The box's true opening at ``t`` in the timing tests' off-grid ending:
    linear from 0.37 s to 0.87 s.
    """
    start, end = Fraction(37, 100), Fraction(87, 100)
    share = min(max((t - start) / (end - start), Fraction(0)), Fraction(1))
    return float(OPEN * share)


@pytest.mark.parametrize("fps", RATES)
def test_a_stepped_rig_answers_the_value_on_display(server, fps):
    box = server("OffGridEnding", fps, TIMING_SCENES)
    end = Fraction(87, 100)
    drawn = list(frames_in(Fraction(0), end, Fraction(fps)))
    assert drawn[-1] < end
    for t in drawn:
        found = values(box.quantities(t, "box"))
        # The drawing on display is the pose true at the last boundary.
        assert found["rig.param.lid_open"] == pytest.approx(
            _true_opening(boundary(t))
        ), t
        assert found["timing.rate"] == BOX_RATE
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


#: The declaration the timing key tests start from: boil on.
KEYED = Timing(RATE, Boil(rate=6, amplitude=2.0, seed=3))


@pytest.mark.parametrize(
    "other",
    [
        Timing(4, KEYED.boil),
        Timing(RATE, Boil(rate=10, amplitude=2.0, seed=3)),
        Timing(RATE, Boil(rate=6, amplitude=2.5, seed=3)),
    ],
    ids=["rate", "boil_rate", "boil_amplitude"],
)
def test_an_illustration_s_timing_is_in_the_content_keys(other):
    # One number changed at a time, boil on, every other field the same.
    shape = Square()
    shape._timing = KEYED.mark
    before = keyed(shape, "_timing")
    shape._timing = other.mark
    after = keyed(shape, "_timing")
    assert apart(before, after)
    shape._timing = KEYED.mark
    assert keyed(shape, "_timing") == before


def _drawing_named(name: str):
    """An edit of the dial's manifest naming its lamp's drawing "on" ``name``."""

    def edit(document):
        (lamp,) = document["slots"]
        lamp["drawings"][1]["name"] = name

    return edit


def _parameter_unit(unit: str):
    """An edit of the dial's manifest giving its parameter the unit ``unit``."""

    def edit(document):
        document["parameters"][0]["unit"] = unit

    return edit


@pytest.mark.parametrize(
    ("edit", "member"),
    [
        (_drawing_named("o" * (MAX_NAME_BYTES + 1)), "slots[0].drawings[1].name"),
        (_parameter_unit("name"), "parameters[0].unit"),
    ],
    ids=["drawing-name-too-long", "number-in-the-unit-of-names"],
)
def test_a_manifest_no_declared_quantity_can_carry_is_refused(tmp_path, edit, member):
    directory = edited(copied(tmp_path, "dial"), edit)
    found = refused("asset.manifest_invalid", lambda: load_asset(directory))
    assert found.detail["member"] == member
    assert found.detail["reason"].startswith("a declared quantity cannot carry it (")


def test_the_longest_drawing_name_a_quantity_carries_loads_and_reads_back(tmp_path):
    longest = "o" * MAX_NAME_BYTES
    directory = admitted(edited(copied(tmp_path, "dial"), _drawing_named(longest)))
    rig = Rig(directory).show(lamp=longest)
    found = read(Rig, rig.__dict__, At.of(Fraction(0), None))
    assert {q.name: q.value for q in found}["rig.slot.lamp"] == longest


def test_a_custom_unit_is_registered_when_its_asset_loads(tmp_path):
    unit = "dial-turns"
    assert unit not in quantities._UNITS
    directory = admitted(edited(copied(tmp_path, "dial"), _parameter_unit(unit)))
    # Admitting it drew a rig of it, and registered nothing.
    assert unit not in quantities._UNITS
    load_asset(directory)
    # Registered by the load alone, before any rig is made.
    assert unit in quantities._UNITS
    rig = Rig(directory)
    (reading,) = (
        q
        for q in read(Rig, rig.__dict__, At.of(Fraction(0), None))
        if "param" in q.name
    )
    assert (reading.name, reading.unit, reading.unreadable) == (
        "rig.param.reading",
        unit,
        None,
    )
