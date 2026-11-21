"""``diagram_clear`` reports a flowing connector's label, set beside its
line, that lands on a piece of its diagram, and names ``label_side=`` as
the fix; set on the other side, it is clear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.rules import diagram_clear
from arachne.session import Session
from arachne.timeline import admission

from . import beside_scenes

SCENES = Path(beside_scenes.__file__)


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str) -> Session:
        session = Session.open(
            SCENES,
            name,
            config={"pixel_width": 192, "pixel_height": 108, "frame_rate": 15},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def test_a_label_beside_its_line_on_a_block_is_reported_with_its_fix(opened):
    findings = lint(opened("BesideUnderABlock"), rules=(diagram_clear,)).findings
    beside = [f for f in findings if "label_side=" in f.message]
    assert beside, [f.message for f in findings]
    assert {f.code for f in beside} == {"overlap.text"}
    assert "'feed'" in beside[0].message


def test_set_on_the_other_side_it_is_clear(opened):
    assert not lint(opened("BesideOnTheOtherSide"), rules=(diagram_clear,)).findings


def _beside(session) -> list:
    return [
        f
        for f in lint(session, rules=(diagram_clear,)).findings
        if "label_side=" in f.message
    ]


def test_what_the_label_lies_on_is_read_at_each_instant(opened):
    found = _beside(opened("CoveredThenRemoved"))
    assert any(f.t_start < 1 for f in found)
    assert not any(f.t_start > 1 for f in found)


def test_a_connector_passing_near_the_label_is_measured_by_its_line(opened):
    assert _beside(opened("PassingNear")) == []
    crossing = _beside(opened("Crossing"))
    assert crossing
    assert {f.code for f in crossing} == {"overlap.text"}
