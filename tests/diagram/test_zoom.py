"""A zoom between a diagram and the inside of one of its blocks keeps every
label legible at every instant the lint samples, and a planted illegible
twin is found; labels inside one diagram are measured against each other,
which the sweep's pair rules leave unasked.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne import Block, Diagram, Node, Text, ZoomInto
from arachne.lint import lint
from arachne.lint.rules import diagram_clear, sweep, text_small
from arachne.session import Session
from arachne.timeline import admission

from . import diagram_scenes

SCENES = Path(diagram_scenes.__file__)
#: A frame 1080 tall, so heights in pixels are the threshold's own.
LINT_SIZE = {"pixel_width": 480, "pixel_height": 270}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, **settings) -> Session:
        session = Session.open(SCENES, name, config={**LINT_SIZE, **settings})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def test_a_legible_zoom_is_clean_at_every_instant(opened):
    session = opened("LegibleZoom", frame_rate=15)
    result = lint(session, fps=15, rules=(sweep, text_small, diagram_clear))
    assert [(f.code, f.message) for f in result.findings] == []
    # The inside was shown, and the neighbours faded, by the end.
    state = session.state_at(Fraction(5, 2))
    shown = {node.text for node in state.nodes if node.text}
    assert {"in", "out", "signal"} <= shown


def test_an_illegible_zoom_is_found(opened):
    session = opened("IllegibleZoom", frame_rate=15)
    result = lint(session, fps=15, rules=(sweep, text_small, diagram_clear))
    codes = {f.code for f in result.findings}
    assert {"overlap.text", "text.small"} <= codes
    assert not result.passed
    # The collision is inside the one diagram: the sweep alone is silent on it.
    alone = lint(session, fps=15, rules=(sweep,))
    assert "overlap.text" not in {f.code for f in alone.findings}


def test_a_zoom_needs_a_camera_whose_frame_moves(opened):
    with pytest.raises(Exception, match="MovingCameraScene"):
        opened("ZoomWithoutMovingCamera")


def test_a_zoom_outside_a_scene_names_the_moving_camera():
    with pytest.raises(ValueError, match="MovingCameraScene"):
        ZoomInto(Block("a"))


def test_a_zoom_asks_for_a_block():
    with pytest.raises(TypeError, match="Block"):
        ZoomInto(Node("a"), frame=Text("frame"))


def test_a_block_holds_its_inside_hidden_until_opened():
    inside = Diagram(Node("x"), Node("y").shift(2))
    block = Block("b", inside=inside)
    assert block.inside is inside
    assert all(
        o == (0.0, 0.0, 0.0)
        for o in [
            (
                float(m.get_fill_opacity()),
                float(m.get_stroke_opacity()),
                float(m.get_stroke_opacity(background=True)),
            )
            for m in inside.get_family()
            if hasattr(m, "get_fill_opacity") and len(m.points)
        ]
    )
    assert max(shown[0] for shown in block.inside_shown) > 0
    # It fits inside the body, below the header.
    assert inside.width <= block.body.width
    assert inside.get_top()[1] <= block.body.get_top()[1] - block.header + 1e-9
