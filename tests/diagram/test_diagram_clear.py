"""The rule ``diagram_clear``: labels inside one diagram, against each
other, the frame's edge and their height; nothing outside a diagram.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.rules import diagram_clear, sweep
from arachne.session import Session
from arachne.timeline import admission

from ..lint import defect_scenes, lint_scenes
from . import clear_scenes, own_diagram_scenes

SCENES = Path(clear_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, path: Path = SCENES) -> Session:
        session = Session.open(path, name, config={**SMALL, "frame_rate": 15})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _codes(session: Session, rule=diagram_clear) -> list[str]:
    return sorted(f.code for f in lint(session, rules=(rule,)).findings)


@pytest.mark.parametrize(
    ("scene", "codes"),
    [
        ("LabelsApart", []),
        ("LabelsOnLabels", ["overlap.text"]),
        ("PillPastTheEdge", ["clip.frame_edge"]),
        ("TinyLabel", ["text.small"]),
        ("LabelsNearlyTouching", ["gap.text"]),
    ],
)
def test_a_diagram_label_s_clearance(opened, scene, codes):
    assert _codes(opened(scene)) == codes


def test_labels_keep_the_gap_a_scene_declares(opened):
    session = opened("LabelsApart")
    wide = diagram_clear.where(text_gap=3.0)
    findings = lint(session, rules=(wide,)).findings
    assert {f.code for f in findings} == {"gap.text"}
    assert all(f.detail["required"] == 3.0 for f in findings)


def test_a_scene_s_own_class_named_diagram_is_not_a_diagram(opened):
    session = opened("OwnClassNamedDiagram", Path(own_diagram_scenes.__file__))
    assert _codes(session) == []


def test_the_sweep_does_not_see_two_labels_of_one_diagram(opened):
    session = opened("LabelsOnLabels")
    assert "overlap.text" not in _codes(session, sweep)


def test_a_pill_is_measured_by_its_boundary_not_its_glyphs(opened):
    session = opened("PillPastTheEdge")
    assert "clip.frame_edge" not in _codes(session, sweep)
    (finding,) = lint(session, rules=(diagram_clear,)).findings
    assert finding.detail["edge"] == "bottom"
    assert finding.detail["past"] == pytest.approx(0.03, abs=0.005)


@pytest.mark.parametrize(
    ("path", "scene"),
    [
        (Path(defect_scenes.__file__), "WordsOnWords"),
        (Path(defect_scenes.__file__), "CaptionPushedOut"),
        (Path(defect_scenes.__file__), "PlatedCaptionAtTheTop"),
        (Path(lint_scenes.__file__), "Backdropped"),
    ],
)
def test_a_scene_with_no_diagram_has_no_finding_from_it(opened, path, scene):
    assert _codes(opened(scene, path)) == []
