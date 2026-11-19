"""A declared quantity a check cannot read is a named finding, never a
silent wrong number: ``state_at`` answers it as null with the reason, and
the lint reports ``sim.quantity_unreadable`` about its object.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.session import Session
from arachne.timeline import admission

from . import quantity_scenes

SCENES = Path(quantity_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


@pytest.fixture
def opened():
    sessions = []
    admission.DEMOTED.clear()

    def open_scene(name: str) -> Session:
        session = Session.open(SCENES, name, config=SMALL)
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _line(tag: str) -> int:
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def test_each_fault_is_null_with_its_reason_and_a_finding(opened):
    session = opened("Unreadables")
    state = session.state_at(Fraction(1, 2))
    found = {
        node.name: state.quantities[node.id]
        for node in state.nodes
        if node.id in state.quantities
    }
    reasons = {name: quantity.unreadable for name, (quantity,) in found.items()}
    assert all(q.value is None for (q,) in found.values())
    assert "prefix 'nothing.' is not registered" in reasons["prefix"]
    assert "unit 'furlong' is not registered" in reasons["unit"]
    assert "not finite" in reasons["finite"]
    assert reasons["missing"] == "no bake holds tick 0"

    result = lint(session, rules=())
    unreadable = [f for f in result.findings if f.code == "sim.quantity_unreadable"]
    by_object = {f.objects[0].name: f for f in unreadable}
    assert sorted(by_object) == ["finite", "missing", "prefix", "unit"]
    assert not result.passed
    for name, finding in by_object.items():
        assert finding.severity == "error"
        assert finding.detail["reason"] == reasons[name]
        assert finding.objects[0].bbox is not None
    assert by_object["prefix"].detail["name"] == "nothing.x"
    assert by_object["prefix"].objects[0].source_line == _line("prefix") - 1
    # Each held over the run of instants it stayed unreadable: the missing
    # source until its bake's stay begins at 1 s.
    assert (by_object["prefix"].t_start, by_object["prefix"].t_end) == (0, 1)
    assert by_object["missing"].t_start == 0
    assert by_object["missing"].t_end < 1


def test_the_clean_twin_has_no_finding(opened):
    session = opened("Readables")
    state = session.state_at(Fraction(1, 2))
    assert len(state.quantities) == 4
    assert all(q.unreadable is None for each in state.quantities.values() for q in each)
    result = lint(session, rules=())
    assert not [f for f in result.findings if f.code == "sim.quantity_unreadable"]
