"""A zoom between a diagram and the inside of one of its blocks keeps every
label legible at every instant the lint samples, and a planted illegible
twin is found; labels inside one diagram are measured against each other,
which the sweep's pair rules leave unasked.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import (
    BLUE,
    Block,
    Diagram,
    Node,
    ScreenRectangle,
    Square,
    Text,
    ZoomInto,
    config,
    linear,
    smooth,
)
from arachne.lint import lint
from arachne.lint.codes import THRESHOLD_HEIGHT
from arachne.lint.rules import diagram_clear, sweep, text_small
from arachne.lint.rules.text_small import SMALL_PIXELS
from arachne.mobject.diagram import FADE, LEGIBLE_MARGIN
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing

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


def _shows(mob) -> float:
    """The most opaque any colour of ``mob``'s family is."""
    return max(
        float(np.max(opacities, initial=0.0))
        for m in mob.get_family()
        if len(m.points)
        for opacities in (m.get_fill_opacities(), m.get_stroke_opacities())
    )


def _cut(piece, frame) -> bool:
    """Whether the frame's edge cuts or leaves out ``piece``."""
    low, high = (
        piece.get_corner(np.array([-1, -1, 0])),
        piece.get_corner(np.array([1, 1, 0])),
    )
    f_low, f_high = (
        frame.get_corner(np.array([-1, -1, 0])),
        frame.get_corner(np.array([1, 1, 0])),
    )
    return bool(
        np.any(low[:2] < f_low[:2] - 1e-9) or np.any(high[:2] > f_high[:2] + 1e-9)
    )


def test_a_zoom_fades_a_neighbour_out_before_the_frame_cuts_it(opened):
    session = opened("LegibleZoom", frame_rate=15)
    record = session.record
    scene = record._scene
    seen: list[tuple[Fraction, float, bool]] = []
    with Drawing(record) as drawing:
        for k in range(38):
            t = Fraction(k, 15)
            _, capture = record.capture_at(t)
            drawing.show(capture, capture.draw_list())
            seen.append(
                (t, _shows(scene.neighbour), _cut(scene.neighbour, scene.camera.frame))
            )
    assert seen[0][1:] == (1.0, False)
    cut = [t for t, _, edge in seen if edge]
    assert cut, "the zoom never cuts the neighbour"
    # Hidden from the first instant the frame's edge cuts it on.
    assert all(shown == 0.0 for t, shown, _ in seen if t >= cut[0])
    # And fading, partly shown, in the instants just before.
    assert any(0.0 < shown < 1.0 for t, shown, _ in seen if t < cut[0])


def _zoom(block: Block, **kwargs) -> tuple[ZoomInto, ScreenRectangle]:
    """A zoom onto ``block`` with a frame of its own, begun outside a scene."""
    frame = ScreenRectangle(height=config.frame_height)
    zoom = ZoomInto(block, frame=frame, **kwargs)
    zoom.begin()
    return zoom, frame


def _pixels(height: float, frame_height: float) -> float:
    """How many pixels of the lint's frame a height shows as."""
    width, tall = int(config.pixel_width), int(config.pixel_height)
    return height * THRESHOLD_HEIGHT * tall / min(width, tall) / frame_height


def _glyphs(text) -> float:
    plate = {id(m) for m in text.background_rectangle.get_family()}
    ys = np.concatenate(
        [
            m.points[:, 1]
            for m in text.get_family()
            if id(m) not in plate and len(m.points)
        ]
    )
    return float(ys.max() - ys.min())


def test_a_label_on_a_pill_shows_once_its_glyphs_read():
    inside = Diagram(cell=(2.6, 1.6))
    first = inside.place(Node("in", font_size=48), 0, 0)
    second = inside.place(Node("out", font_size=48), 1, 0)
    edge = inside.connect(first, second, label="signal", font_size=12)
    block = Block("b", width=2.2, height=1.4, inside=inside)
    zoom, frame = _zoom(block, rate_func=linear)
    legible = SMALL_PIXELS * LEGIBLE_MARGIN
    hidden, shown = [], []
    for step in range(401):
        zoom.interpolate(step / 400)
        (shown if _shows(edge.label) > 0 else hidden).append(frame.height)
    assert hidden
    assert shown
    # The pill reads before the glyphs do, and the inside waits for the glyphs.
    pill = edge.label.height
    assert _pixels(pill, min(hidden)) >= legible
    assert _pixels(_glyphs(edge.label), min(hidden)) < legible
    assert _pixels(_glyphs(edge.label), max(shown)) >= legible * (1 - 1e-9)


def _opened_at(deep_size: float) -> float:
    """The first share of a zoom at which a block's inside shows, the inside
    holding a block whose own (hidden) inside is labelled ``deep_size``.
    """
    deep = Block(
        "deep", width=1.4, height=1.0, inside=Diagram(Node("x", font_size=deep_size))
    )
    inside = Diagram(Node("a", font_size=24), deep.shift(np.array([1.6, 0, 0])))
    block = Block("b", width=3.2, height=1.6, inside=inside)
    zoom, _ = _zoom(block, rate_func=linear)
    for step in range(401):
        zoom.interpolate(step / 400)
        if _shows(inside[0]) > 0:
            return step / 400
    return 1.0


def test_a_hidden_inside_deeper_down_does_not_hold_the_fade_back():
    assert _opened_at(4) == _opened_at(48)
    assert _opened_at(4) < 1 - FADE


def test_a_zoom_eases_its_frame_by_its_rate_func():
    heights = {}
    for name, rate in (("linear", linear), ("smooth", smooth)):
        zoom, frame = _zoom(Block("b", width=2.2, height=1.4), rate_func=rate)
        h0 = frame.height
        zoom.interpolate(0.25)
        h1 = zoom._to[1]
        heights[name] = (frame.height, h0 * (h1 / h0) ** rate(0.25))
    for got, expected in heights.values():
        assert got == pytest.approx(expected, rel=1e-9)
    assert heights["linear"][0] != pytest.approx(heights["smooth"][0])


def test_a_zoom_keeps_each_stop_of_a_gradient_it_fades_in():
    patch = Square(0.6).set_fill(BLUE, opacity=(0.2, 0.8)).set_stroke(width=0)
    inside = Diagram(Node("a", font_size=48), patch.shift(np.array([1.2, 0, 0])))
    block = Block("b", width=3.2, height=1.6, inside=inside)
    assert float(np.max(patch.get_fill_opacities())) == 0
    zoom, _ = _zoom(block, rate_func=linear)
    zoom.interpolate(1.0)
    assert list(np.ravel(patch.get_fill_opacities())) == pytest.approx([0.2, 0.8])


def test_a_zoom_moves_no_frame_but_its_cameras(opened):
    with pytest.raises(Exception, match="MovingCameraScene"):
        opened("ZoomWithAFrameOfItsOwn")


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
    assert max(max(shown[0]) for shown in block.inside_shown if shown[0]) > 0
    # It fits inside the body, below the header.
    assert inside.width <= block.body.width
    assert inside.get_top()[1] <= block.body.get_top()[1] - block.header + 1e-9
