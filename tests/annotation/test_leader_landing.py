"""A callout's leader lands on its anchor, inside the boxes of the object
the anchor belongs to: the sweep lets that landing pass, and only that.

The pass is the declaration's, never the shape's: a leader is let land on
the object its declared anchor belongs to, and the same line over any other
object, or with no annotation declaring it, is judged as any stroke. It is
the landing alone: a leader that crosses its target to reach its anchor, or
runs on past an outline anchor into the object, still touches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.rules import sweep
from arachne.session import Session
from arachne.timeline import admission

from . import annotation_scenes

SCENES = Path(annotation_scenes.__file__)


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


def _touching(session: Session, kind: str) -> list:
    """The ``contact.held`` findings the sweep makes about a ``kind``."""
    return [
        f
        for f in lint(session, rules=(sweep,)).findings
        if f.code == "contact.held" and kind in {o.kind for o in f.objects}
    ]


@pytest.mark.parametrize("scene", ["LeaderOnTiltedLid", "LeaderToItsCentre"])
def test_a_leader_landing_on_its_anchor_s_object_is_let_pass(opened, scene):
    assert _touching(opened(scene), "LeaderLine") == []


def test_the_same_line_declaring_no_anchor_still_touches(opened):
    found = _touching(opened("LineOnTiltedLid"), "Line")
    assert found
    assert all("VMobjectFromSVGPath" in {o.kind for o in f.objects} for f in found)


@pytest.mark.parametrize("scene", ["LeaderThroughTarget", "LeaderPastItsOutlineAnchor"])
def test_a_leader_deeper_than_its_landing_still_touches_its_target(opened, scene):
    (finding,) = _touching(opened(scene), "LeaderLine")
    assert {o.kind for o in finding.objects} == {"LeaderLine", "Square"}
    assert "target" in {o.label_or_text for o in finding.objects}


def test_a_leader_over_another_object_still_touches_it(opened):
    (finding,) = _touching(opened("LeaderOverAnotherObject"), "LeaderLine")
    assert "other" in {o.label_or_text for o in finding.objects}
