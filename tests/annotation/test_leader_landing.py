"""A callout's leader lands on its anchor, inside the boxes of the object
the anchor belongs to: the sweep lets that landing pass, and only that.

The pass is the declaration's, never the shape's: a leader is let land on
the object its declared anchor belongs to, and the same line over any other
object, or with no annotation declaring it, is judged as any stroke. It is
the landing alone: a leader that crosses its target to reach its anchor, or
runs on past an outline anchor into the object, still touches it.
"""

from __future__ import annotations

from fractions import Fraction
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


def test_a_leader_through_a_fill_its_path_leaves_open_still_touches_it(opened):
    # The fill closes the side its path leaves open: the leader crosses it
    # there and again at the top, and ends outside the fill.
    (finding,) = _touching(opened("LeaderThroughOpenFill"), "LeaderLine")
    assert "target" in {o.label_or_text for o in finding.objects}


def _touching_other(session: Session) -> list:
    return [
        f
        for f in _touching(session, "LeaderLine")
        if "other" in {o.label_or_text for o in f.objects}
    ]


def test_a_landing_is_read_as_the_frame_holds_the_anchor_s_object(opened):
    # Before the obstacle joins the anchor's group, the leader ending in it
    # touches it; the later membership excuses no earlier frame.
    found = _touching_other(opened("ObstacleJoinsTheTarget"))
    assert found
    assert all(f.t_start < Fraction(1, 4) for f in found)
    assert all(f.t_end is not None and f.t_end < Fraction(1, 4) for f in found)


def test_an_anchor_object_the_frame_does_not_hold_declares_no_landing(opened):
    # The anchor's group is never in the scene, so no frame holds its
    # members as they were: no landing is declared, and the leader ending in
    # the obstacle touches it before the obstacle joins the group too.
    found = _touching_other(opened("ObstacleJoinsAnOffSceneGroup"))
    assert any(f.t_start < Fraction(1, 4) for f in found)


def _pairs_of(objects) -> frozenset[str]:
    return frozenset(o["id"] if isinstance(o, dict) else o.id for o in objects)


def test_the_scene_s_sweep_lets_a_landing_pass_as_the_lint_does(opened):
    session = opened("SweptLanding")
    swept = {
        _pairs_of(f["objects"])
        for f in session.scene.found
        if f["code"] == "contact.held"
    }
    linted = {
        _pairs_of(f.objects)
        for f in lint(session, rules=(sweep.where(at=[Fraction(1, 4)]),)).findings
        if f.code == "contact.held"
    }
    assert len(swept) == 1
    assert swept == linted
