"""The existing lint covers annotation's labels: two callouts are two
objects of the scene, so the sweep measures their pills against each other
and against the frame's edge.
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


def _texts(session: Session, code: str) -> list:
    return [f for f in lint(session, rules=(sweep,)).findings if f.code == code]


def test_two_overlapping_callouts_overlap(opened):
    found = _texts(opened("OverlappingCallouts"), "overlap.text")
    (pills,) = [f for f in found if {o.kind for o in f.objects} == {"LabelPill"}]
    assert sorted(o.label_or_text for o in pills.objects) == ["first", "second"]


def test_a_callout_past_the_frame_s_edge_is_cut(opened):
    (finding,) = _texts(opened("CalloutPastTheEdge"), "clip.frame_edge")
    assert finding.detail["edge"] == "right"


def test_callouts_apart_are_clean(opened):
    found = lint(opened("CalloutsApart"), rules=(sweep,)).findings
    assert [
        f.code for f in found if f.code in ("overlap.text", "clip.frame_edge")
    ] == []
