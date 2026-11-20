"""What a scene holds arms rules too: a lint with ``rules="armed"`` runs
``anchored`` for a scene holding an attached annotation and
``diagram_clear`` for one holding an engine diagram, whether or not the
scene armed anything; a scene holding neither is linted as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.arming import armed
from arachne.lint.departures import departures
from arachne.lint.rules import anchored, diagram_clear
from arachne.lint.watch import hold_watch
from arachne.session import Session
from arachne.timeline import admission

from ..annotation import annotation_scenes
from ..diagram import clear_scenes
from . import armed_scenes, lint_scenes

ANNOTATED = Path(annotation_scenes.__file__)
DIAGRAMS = Path(clear_scenes.__file__)


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(path: Path, name: str) -> Session:
        session = Session.open(
            path, name, config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _found(session: Session, rules, code: str) -> list:
    return [
        (f.code, f.t_start, f.t_end, tuple(o.id for o in f.objects))
        for f in lint(session, rules=rules).findings
        if f.code == code
    ]


@pytest.mark.parametrize(
    ("path", "scene", "rule", "code"),
    [
        (ANNOTATED, "CalloutBeforeMovingDot", anchored, "annotation.detached"),
        (DIAGRAMS, "LabelsOnLabels", diagram_clear, "overlap.text"),
    ],
)
def test_what_a_scene_holds_arms_its_rule_once(opened, path, scene, rule, code):
    session = opened(path, scene)
    named = _found(session, (rule,), code)
    assert named
    # The armed lint finds what the rule named alone finds, each once.
    assert _found(session, "armed", code) == named


def test_a_scene_holding_neither_arms_only_what_it_armed(opened):
    session = opened(Path(armed_scenes.__file__), "ExemptedNoteFades")
    since = next(c.t for c in session.record.bridge.calls if c.kind == "arm")
    assert armed(session.record) == (
        hold_watch.where(since=since, sample_fps=None),
        departures.where(since=since),
    )
    assert armed(opened(Path(lint_scenes.__file__), "Backdropped").record) == ()
