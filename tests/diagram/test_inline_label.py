"""``diagram_clear`` reports a connector's label, on its plate at the
line's middle, that leaves too little of its own line showing on either
side to see a line runs there, measured on screen: at the frame of each
instant it is shown at.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import config
from arachne.lint import lint
from arachne.lint.rules import diagram_clear, text_small
from arachne.lint.rules.diagram_clear import (
    INLINE_FIX,
    MIN_STUB,
    DiagramRule,
    diagram_findings,
    line_stubs,
)
from arachne.session import Session
from arachne.timeline import admission

from . import inline_scenes
from .inline_scenes import HIDDEN

SCENES = Path(inline_scenes.__file__)
#: A frame 1080 tall, so the zoom's legible height is the lint's own.
LINT_SIZE = {"pixel_width": 480, "pixel_height": 270, "frame_rate": 15}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str) -> Session:
        session = Session.open(SCENES, name, config=LINT_SIZE)
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def test_a_label_hiding_its_own_line_is_reported_with_its_fix(opened):
    findings = lint(opened("LineHiddenByItsLabel"), rules=(diagram_clear,)).findings
    (finding,) = findings
    assert finding.code == "overlap.text"
    assert finding.message.endswith(INLINE_FIX)
    assert "'link'" in finding.message
    assert "label_side" not in finding.message
    # Under a still camera, scene units and screen units agree.
    for side in ("before", "after"):
        assert finding.detail[side] == pytest.approx(HIDDEN, abs=1e-3)
        assert finding.detail[f"{side}_screen"] == pytest.approx(HIDDEN, abs=1e-3)
    assert finding.objects[0].label_or_text == "link"


def test_a_label_leaving_its_line_showing_is_clear(opened):
    assert not lint(opened("LineShowingPastItsLabel"), rules=(diagram_clear,)).findings


def test_a_label_a_flow_set_beside_its_line_is_not_inline(opened):
    assert not lint(opened("HiddenLineWithAFlow"), rules=(diagram_clear,)).findings


def test_a_zoom_reports_the_inside_s_label_until_its_line_shows_on_screen(opened):
    """The inside fades in once its labels read, the frame still wide, and
    the label is judged from its first instant shown, as its legible height
    is: there its line's stubs, the same in scene units throughout, are too
    short on screen; as the frame closes in they grow on screen, by the home
    frame's width over the frame's, past the least, and the finding ends.
    """
    session = opened("ZoomOntoAHiddenLine")
    result = lint(session, fps=15, rules=(text_small, diagram_clear))
    (finding,) = result.findings
    assert finding.code == "overlap.text"
    assert finding.message.endswith(INLINE_FIX)
    assert "'signal'" in finding.message
    # Each instant: whether the label is drawn, the frame's width, and the
    # finding's stubs (scene, screen) there, if any.
    seen: list[tuple[Fraction, bool, float, tuple[float, float] | None]] = []

    def probe(parts, terms):
        found = diagram_findings(parts, terms)
        hidden = [f for f in found if f.message.endswith(INLINE_FIX)]
        seen.append(
            (
                parts.t,
                any(part.text == "signal" for part in parts.texts),
                parts.view.frame_width,
                (hidden[0].detail["before"], hidden[0].detail["before_screen"])
                if hidden
                else None,
            )
        )
        return found

    lint(session, fps=15, rules=(DiagramRule(probe),))
    first_shown = next(t for t, shown, _, _ in seen if shown)
    reported = [(t, width, stubs) for t, _, width, stubs in seen if stubs]
    # From the label's first instant shown, while the zoom still moves (it
    # stands on the block at 2 s), and then not again though still shown.
    assert finding.t_start == first_shown == reported[0][0]
    assert finding.t_end == reported[-1][0] < Fraction(2)
    assert all(shown for t, shown, _, _ in seen if t > finding.t_end)
    for _, width, (scene, screen) in reported:
        assert scene == pytest.approx(HIDDEN, abs=1e-3)
        assert screen == pytest.approx(scene * config.frame_width / width, abs=1e-3)
        assert screen < MIN_STUB
    # At the next instant the frame is narrow enough to show the least.
    after = next(width for t, _, width, _ in seen if t > finding.t_end)
    assert HIDDEN * config.frame_width / after >= MIN_STUB


def test_a_line_s_stubs_run_along_its_bends():
    """An L of two straight curves, 1 then 1, the box over its corner: the
    stubs are measured along the path, not straight to the box.
    """
    corner = np.array([[0.0, 0.0], [1 / 3, 0.0], [2 / 3, 0.0], [1.0, 0.0]])
    up = np.array([[1.0, 0.0], [1.0, 1 / 3], [1.0, 2 / 3], [1.0, 1.0]])
    curves = np.stack([corner, up])
    assert line_stubs(curves, (0.8, -0.1, 1.1, 0.3)) == pytest.approx((0.8, 0.7))
    # A box off the line covers none of it.
    assert line_stubs(curves, (2.0, 2.0, 3.0, 3.0)) is None
