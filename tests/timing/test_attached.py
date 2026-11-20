"""A callout attached to a declared rig's anchor ends, at every delivered
frame, on the anchor of the drawing on screen, not of the pose that is true
between two drawings; the lint's ``anchored`` stays quiet on it; and the
callout itself, as exact content, is never stepped.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import Callout, Rig
from arachne.lint import lint
from arachne.lint.rules import anchored

from .reading import RATES, boundary, frame_captures
from .timing_scenes import MOVE, OPEN


def _knob(value: float, moved: float) -> np.ndarray:
    rig = Rig("fixture:lidded-box").pose(lid_open=value).shift(moved * MOVE)
    return rig.anchor("knob").point()


@pytest.mark.parametrize("fps", RATES)
def test_the_leader_ends_on_the_anchor_of_the_drawing_on_screen(opened, fps):
    session = opened("CalloutOnSteppedBox", fps)
    record = session.record
    callout = next(m for m in record.mobjects() if isinstance(m, Callout))
    leader = callout.leader
    off_the_drawing = 0
    ends = set()
    for t, capture in frame_captures(session, fps):
        at = min(boundary(t), Fraction(1))
        end = np.asarray(record.node_state(leader, capture).get("_points"))[-1]
        on_screen = _knob(float(OPEN * at), float(at))
        assert np.linalg.norm(end - on_screen) < 1e-9, t
        ends.add(end.tobytes())
        if t < 1:
            true = _knob(float(OPEN * t), float(t))
            off_the_drawing += bool(np.linalg.norm(true - on_screen) > 1e-3)
    # The callout followed the drawing, which moved, and away from the
    # boundaries the true pose was elsewhere.
    assert len(ends) > 8
    assert off_the_drawing > 0


@pytest.mark.parametrize("fps", RATES)
def test_the_lint_finds_the_callout_attached(opened, fps):
    found = lint(opened("CalloutOnSteppedBox", fps), rules=(anchored,)).findings
    assert [f.code for f in found] == [], [f.message for f in found]


def test_the_callout_s_label_moves_with_the_drawing_it_names(opened):
    session = opened("CalloutOnSteppedBox", 60)
    record = session.record
    callout = next(m for m in record.mobjects() if isinstance(m, Callout))
    pill = callout.pill
    # Between two boundaries the drawing is held, and so is the label on it:
    # it is not stepped itself, it follows what is shown.
    centres: dict[Fraction, list[np.ndarray]] = {}
    for t, capture in frame_captures(session, 60):
        points = np.asarray(
            record.node_state(pill.submobjects[0], capture).get("_points")
        )
        centres.setdefault(boundary(t), []).append(points.mean(axis=0))
    for found in centres.values():
        assert np.ptp(np.array(found), axis=0).max() < 1e-9
    # Each drawing of the move puts it somewhere new.
    held = [found[0] for at, found in centres.items() if at < 1]
    assert (
        min(np.linalg.norm(a - b) for a, b in zip(held, held[1:], strict=False)) > 1e-3
    )
