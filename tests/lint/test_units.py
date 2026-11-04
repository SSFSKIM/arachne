"""Each drawn leaf knows the thing a sweep measures it as part of (its
unit), and the strokes it is drawn with, foreground and background apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.parts import Leaf, Part, Parts

from . import unit_scenes
from .conftest import line_of

SCENES = Path(unit_scenes.__file__)


@pytest.fixture
def parts(session_of) -> Parts:
    session = session_of("Units", SCENES, frame_rate=15)
    seen: list[Parts] = []

    def keep(context):
        seen.extend(context.parts)
        return ()

    lint(session, rules=(keep,), at=[session.record.duration])
    (last,) = seen
    return last


def _leaves(parts: Parts, line: str) -> list[tuple[Part, Leaf]]:
    """Every leaf made by the tagged line, or inside a mobject it made."""
    number = line_of(line, SCENES)
    return [
        (part, leaf)
        for part in parts.parts
        for leaf in part.leaves
        if part.scene_line == number or any(h.line == number for h in part.within)
    ]


def _units(parts: Parts, line: str) -> set[str | None]:
    return {leaf.unit.id if leaf.unit else None for _, leaf in _leaves(parts, line)}


def test_a_compound_path_is_one_unit_with_its_tip(parts):
    found = _leaves(parts, "arrow")
    assert {part.type for part, _ in found} >= {"Arrow"}
    assert len(found) >= 2
    (arrow,) = (part for part, _ in found if part.type == "Arrow")
    assert _units(parts, "arrow") == {arrow.id}
    unit = found[0][1].unit
    assert unit.type == "Arrow"
    assert not unit.text
    assert unit.furniture is None
    assert unit.name == "arrow"


def test_a_vector_group_that_draws_nothing_opens_into_its_members(parts):
    found = _leaves(parts, "row")
    assert {leaf.unit.id for _, leaf in found} == {part.id for part, _ in found}
    assert len(found) == 2


def test_a_group_that_is_no_vector_group_is_one_unit(parts):
    found = _leaves(parts, "pile")
    assert len(found) == 2
    (unit,) = {leaf.unit for _, leaf in found}
    assert unit.type == "Group"
    assert unit.name == "pile"
    assert unit.id == found[0][0].within[0].id


def test_a_container_of_the_scene_s_own_is_one_unit_and_no_text(parts):
    found = _leaves(parts, "card")
    assert {part.kind for part, _ in found} == {"text", "path"}
    (unit,) = {leaf.unit for _, leaf in found}
    assert unit.type == "Card"
    assert not unit.text


def test_a_paragraph_opens_down_to_its_glyphs_which_are_no_text(parts):
    # Its lines are vector groups of glyphs, not texts: a sweep opens both.
    found = _leaves(parts, "lines")
    units = [leaf.unit for _, leaf in found]
    assert len(set(units)) == len(found) > 2
    assert not any(unit.text for unit in units)


def test_a_dashed_line_is_one_unit(parts):
    found = _leaves(parts, "dashes")
    assert len(found) > 2
    (unit,) = {leaf.unit for _, leaf in found}
    assert unit.type == "DashedLine"


def test_axes_are_their_furniture_and_what_is_mounted_is_not_measured(parts):
    found = _leaves(parts, "axes")
    roles: dict[str, set[str]] = {}
    for _, leaf in found:
        if leaf.unit is not None:
            roles.setdefault(leaf.unit.furniture, set()).add(leaf.unit.id)
    assert set(roles) == {"line", "tip", "ticks", "number"}
    # Two axes: a line, a tip and one set of ticks each, and a unit per numeral.
    assert len(roles["line"]) == len(roles["tip"]) == len(roles["ticks"]) == 2
    assert len(roles["number"]) >= 2
    numbers = {
        leaf.unit for _, leaf in found if leaf.unit and leaf.unit.furniture == "number"
    }
    assert all(unit.text for unit in numbers)
    assert _units(parts, "mounted") == {None}


def test_a_text_s_own_plate_is_in_its_unit(parts):
    found = _leaves(parts, "plated")
    assert "BackgroundRectangle" in {leaf.type for _, leaf in found}
    (unit,) = {leaf.unit for _, leaf in found}
    assert unit.type == "Text"
    assert unit.text
    assert unit.name == "plated"
    # The plate is known by its role, the text's background rectangle.
    assert {leaf.type for _, leaf in found if leaf.plate} == {"BackgroundRectangle"}
    assert all(
        not leaf.plate for _, leaf in found if leaf.type != "BackgroundRectangle"
    )


def test_a_unit_s_family_is_what_it_is_made_from(parts):
    (unit,) = {leaf.unit for _, leaf in _leaves(parts, "pile")}
    assert {m.type for m in unit.family} == {"Group", "Square"}
    furniture = {leaf.unit for _, leaf in _leaves(parts, "axes") if leaf.unit}
    # Every piece of furniture is made from its axes, whole.
    (family,) = {unit.family for unit in furniture}
    assert family[0].type == "Axes"
    assert {"NumberLine", "Dot"} <= {m.type for m in family}


def test_a_fill_is_read_by_its_first_row(parts):
    ((_, faded),) = _leaves(parts, "faded")
    assert faded.first_fill_opacity == 0
    assert faded.fill_opacity == 1
    ((_, ring),) = _leaves(parts, "ringed")
    assert ring.first_fill_opacity == 0


def test_strokes_are_kept_apart_foreground_and_background(parts):
    glyphs = [leaf for part, leaf in _leaves(parts, "haloed") if part.kind == "text"]
    assert glyphs
    for leaf in glyphs:
        assert leaf.line_width == 0
        assert leaf.halo_width == pytest.approx(0.06)
        assert leaf.halo_opacity == pytest.approx(1.0)
        assert leaf.stroke_width == pytest.approx(0.06)
    ((_, ring),) = _leaves(parts, "ringed")
    assert ring.line_width == pytest.approx(0.08)
    assert ring.halo_width == 0
    assert ring.stroke_width == pytest.approx(0.08)
