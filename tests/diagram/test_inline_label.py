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

from arachne import DL, UR, config
from arachne.lint import lint
from arachne.lint.rules import diagram_clear, text_small
from arachne.lint.rules.diagram_clear import (
    MIN_STUB,
    DiagramRule,
    _shows_too_little,
    _step_distance,
    diagram_findings,
    line_stubs,
)
from arachne.session import Session
from arachne.timeline import admission

from . import inline_scenes
from .inline_scenes import HIDDEN, INSIDE_GLYPHS, flowing_hidden_line

SCENES = Path(inline_scenes.__file__)
#: A small 16:9 target: the home frame's shape. The lint and the zoom read
#: heights on the 1080-pixel reference picture, whatever the target's size.
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


#: The fix a label hiding its line is told, as the ruling words it.
FIX = "widen the gap or drop the label"


def _findings(session, **kwargs) -> list:
    return lint(session, rules=(diagram_clear,), **kwargs).findings


def test_a_label_hiding_its_own_line_is_reported_with_its_fix(opened):
    (finding,) = _findings(opened("LineHiddenByItsLabel"))
    assert finding.code == "overlap.text"
    assert finding.message.endswith(FIX)
    assert finding.fix == FIX
    assert "'link'" in finding.message
    assert "label_side" not in finding.message
    # Under a still camera, scene units and screen units agree.
    for side in ("before", "after"):
        assert finding.detail[side] == pytest.approx(HIDDEN, abs=1e-6)
        assert finding.detail[f"{side}_screen"] == pytest.approx(HIDDEN, abs=1e-6)
    assert [o.label_or_text for o in finding.objects[:1]] == ["link"]


def test_a_label_leaving_its_line_showing_is_clear(opened):
    assert not _findings(opened("LineShowingPastItsLabel"))


@pytest.mark.parametrize(
    ("scene", "reported"),
    [
        ("StubJustUnderTheLeast", True),
        ("StubAtTheLeast", False),
        ("StubJustOverTheLeast", False),
    ],
)
def test_a_line_showing_the_least_each_side_is_clear(opened, scene, reported):
    assert bool(_findings(opened(scene))) is reported


def test_only_the_label_that_hides_its_line_is_reported(opened):
    (finding,) = _findings(opened("OneOfTwoLabelsHidesItsLine"))
    assert finding.message.endswith(FIX)
    assert finding.objects[0].label_or_text == "hides"


@pytest.mark.parametrize("scene", ["LineDrawnOverItsLabel", "LabelOnAClearPlate"])
def test_a_plate_that_does_not_paint_over_its_line_hides_none_of_it(opened, scene):
    """The same geometry as the hiding label's, but the line drawn after
    the plate, or the plate painting nothing: the line shows over it.
    """
    assert not _findings(opened(scene))


def test_a_label_a_flow_set_beside_its_line_is_not_inline(opened):
    # Its plate lies on its line as the hiding label's does...
    diagram, _ = flowing_hidden_line()
    edge = diagram.edges[0]
    plate = edge.label.background_rectangle
    box = (*plate.get_corner(DL)[:2], *plate.get_corner(UR)[:2])
    curves = edge.points[:, :2].reshape(-1, 4, 2)
    assert line_stubs(curves, box) == pytest.approx((HIDDEN, HIDDEN), abs=1e-6)
    # ...but its connector has carried a flow.
    assert not _findings(opened("HiddenLineWithAFlow"))


@pytest.mark.parametrize(
    ("scene", "scene_stub"),
    [
        ("UprightLineInAFrameHalfAsWide", 0.05),
        ("UprightHiddenLineInAFrameHalfAsWide", HIDDEN),
    ],
)
def test_a_stub_is_scaled_to_the_screen_by_the_frame_s_width(opened, scene, scene_stub):
    """A frame squeezed to half its width, its height kept: an upright stub
    counts twice its scene length on screen, though it is drawn no longer.
    """
    found = _findings(opened(scene))
    if scene_stub * 2 >= MIN_STUB:
        assert not found
        return
    (finding,) = found
    assert finding.detail["before"] == pytest.approx(scene_stub, abs=1e-6)
    assert finding.detail["before_screen"] == pytest.approx(2 * scene_stub, abs=1e-6)


def test_a_zoom_reports_the_inside_s_label_until_its_line_shows_on_screen(opened):
    """The inside fades in once its glyphs (every one set
    :data:`~.inline_scenes.INSIDE_GLYPHS` tall) read, the frame 6 units
    tall, and the label is judged from its first instant shown, as its
    legible height is: there its line's stubs, the same in scene units
    throughout, are too short on screen; as the frame closes in they grow on
    screen, by the home frame's width over the frame's, past the least, and
    the finding ends.
    """
    session = opened("ZoomOntoAHiddenLine")
    result = lint(session, fps=15, rules=(text_small, diagram_clear))
    (finding,) = result.findings
    assert finding.code == "overlap.text"
    assert finding.message.endswith(FIX)
    assert "'signal'" in finding.message
    # Each instant: whether the label is drawn, the frame's width and
    # height, and the finding's stubs (scene, screen) there, if any.
    seen: list[tuple[Fraction, bool, float, float, tuple[float, float] | None]] = []

    def probe(parts, terms):
        found = diagram_findings(parts, terms)
        hidden = [f for f in found if f.message.endswith(FIX)]
        seen.append(
            (
                parts.t,
                any(part.text == "signal" for part in parts.texts),
                parts.view.frame_width,
                parts.view.frame_height,
                (hidden[0].detail["before"], hidden[0].detail["before_screen"])
                if hidden
                else None,
            )
        )
        return found

    lint(session, fps=15, rules=(DiagramRule(probe),))
    first_shown = next(t for t, shown, *_ in seen if shown)
    reported = [(t, width, stubs) for t, _, width, _, stubs in seen if stubs]
    # The fade starts where the pinned glyphs read: the frame about 6 tall,
    # its glyphs 20 pixels with the zoom's margin.
    height = next(h for t, _, _, h, _ in seen if t == first_shown)
    assert INSIDE_GLYPHS * 1080 / height == pytest.approx(22, abs=2)
    # From the label's first instant shown, for many instants while the zoom
    # still moves (it stands on the block at 2 s), and then not again though
    # still shown.
    assert finding.t_start == first_shown == reported[0][0]
    assert finding.t_end == reported[-1][0] < Fraction(2)
    assert len(reported) >= 5
    assert all(shown for t, shown, *_ in seen if t > finding.t_end)
    for _, width, (scene, screen) in reported:
        assert scene == pytest.approx(HIDDEN, abs=1e-4)
        assert screen == pytest.approx(scene * config.frame_width / width, abs=1e-4)
        assert screen < MIN_STUB
    # Shown well short of the least at first, and past it at the next
    # instant after the last reported.
    assert reported[0][2][1] < 0.75 * MIN_STUB
    after = next(width for t, _, width, _, _ in seen if t > finding.t_end)
    assert HIDDEN * config.frame_width / after >= MIN_STUB


def _straight(*corners) -> np.ndarray:
    """Straight cubic curves through ``corners``, ``(curves, 4, 2)``."""
    points = [np.array(c, dtype=float) for c in corners]
    return np.stack(
        [
            np.array([a + (b - a) * k / 3 for k in range(4)])
            for a, b in zip(points[:-1], points[1:], strict=False)
        ]
    )


def test_a_line_s_stubs_run_along_its_bends():
    """An L of two straight curves, 1 then 1, the box over the middle of its
    upright leg: the stub before it runs along the path, round the corner.
    """
    curves = _straight((0, 0), (1, 0), (1, 1))
    assert line_stubs(curves, (0.9, 0.4, 1.1, 0.6)) == pytest.approx((1.4, 0.4))
    # A box off the line covers none of it.
    assert line_stubs(curves, (2.0, 2.0, 3.0, 3.0)) is None


def test_a_box_that_only_touches_a_line_covers_none_of_it():
    """A box whose corner meets a diagonal line's end covers no length of
    it, though for a label beside a line that touch is a contact.
    """
    curves = _straight((0, 0), (1, 1))
    box = (1.0, 1.0, 2.0, 2.0)
    assert line_stubs(curves, box) is None
    low, high = np.array(box[:2]), np.array(box[2:])
    corners = np.array([low, high, [low[0], high[1]], [high[0], low[1]]])
    assert _step_distance(curves[0][0], curves[0][-1], low, high, corners) == 0.0
    # Nor does it cover a curve that stands still at a point inside it.
    still = np.array([[[1.5, 1.5]] * 4])
    assert line_stubs(np.concatenate([curves, still]), box) is None


def test_a_stub_is_measured_in_scene_units_from_the_screen():
    """An upright screen line 1 long, its middle fifth boxed, drawn by a
    frame showing scene heights at half their size: each stub is 0.8 scene
    units.
    """
    curves = _straight((0, 0), (0, 1))
    stubs = line_stubs(curves, (-0.1, 0.4, 0.1, 0.6), to_scene=(1.0, 2.0))
    assert stubs == pytest.approx((0.8, 0.8))


def test_a_stub_of_the_least_shows_enough_on_any_rounding():
    """A unit line boxed from 0.08 to 0.92: its stubs come out a hair under
    0.08 in floating point, and show enough; a ten-thousandth under does
    not.
    """
    curves = _straight((0, 0), (1, 0))
    stubs = line_stubs(curves, (0.08, -0.1, 0.92, 0.1))
    assert stubs == pytest.approx((0.08, 0.08))
    assert not _shows_too_little(stubs)
    assert _shows_too_little(line_stubs(curves, (0.0799, -0.1, 0.92, 0.1)))
    assert not _shows_too_little(line_stubs(curves, (0.0801, -0.1, 0.92, 0.1)))


def test_a_stub_is_the_line_the_frame_shows():
    """A line 10 long across a frame whose right edge is at 1, boxed from
    0.5 to 0.9: past the box the frame shows 0.1 of it; a line entering the
    frame from its left edge, at -1, shows from there.
    """
    curves = _straight((-3, 0), (10, 0))
    box = (0.5, -0.1, 0.9, 0.1)
    assert line_stubs(curves, box) == pytest.approx((3.5, 9.1))
    frame = (-1.0, -1.0, 1.0, 1.0)
    assert line_stubs(curves, box, frame=frame) == pytest.approx((1.5, 0.1))
    # Measured in scene units as before, and none for a box off the line.
    stubs = line_stubs(curves, box, to_scene=(2.0, 1.0), frame=frame)
    assert stubs == pytest.approx((3.0, 0.2))
    assert line_stubs(curves, (0.5, 0.4, 0.9, 0.6), frame=frame) is None


@pytest.mark.parametrize(
    ("scene", "reported"),
    [("PillJustInsideTheFrame", True), ("PillWellInsideTheFrame", False)],
)
def test_a_label_whose_line_runs_off_the_screen_is_judged_by_what_shows(
    opened, scene, reported
):
    found = [f for f in _findings(opened(scene)) if f.code == "overlap.text"]
    assert bool(found) is reported
    if reported:
        (finding,) = found
        assert finding.message.endswith(FIX)
        assert finding.detail["after"] == pytest.approx(0.02, abs=1e-6)
