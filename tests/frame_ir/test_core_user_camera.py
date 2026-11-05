"""A camera the scene's code changed, drawn on the core.

On ``ir_cairo`` a changed camera is described from its own capture and its
real draw stands as a backstop: a frame whose description differs from it
falls back, invisibly, since Cairo draws both. The core keeps no draw of the
camera's own, so a fallback mid-scene would show. The core path therefore
decides before the first frame, for the whole scene: a camera holding only
changes the gate has proven on ``ir_cairo`` against the camera's own draw
(``PROVEN_ON_THE_GATE``) is drawn on the core from what it produced; any
other draws on Cairo, with one finding. A frame the core refuses is
``render.unable``, never drawn on another path.

A proven change can still rebind what the camera holds as it draws, which
XF does not carry. The core watches for that in the first frame of each
play, and draws the scene's first frame as the camera draws it: a rebinding
seen there sends the whole scene to Cairo; one first seen in a later play is
``render.unable``.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir import painter
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import Unable

from ..timeline.conftest import render
from . import camera_scenes, core_scenes

arachne_core = pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


@pytest.fixture
def findings(monkeypatch):
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    return lint.raised


def _fallbacks(findings: list) -> list:
    return [f for f in findings if f.code == "render.fallback"]


#: Changed cameras whose changes the gate has proven, each on a path the core
#: draws them by: described as the scene runs (its own class) and from the
#: record (patched on the instance).
PROVEN = [
    camera_scenes.ShiftedSolids,
    camera_scenes.PatchedSolids,
    camera_scenes.CountedLive,
]


@pytest.mark.parametrize("scene_class", PROVEN, ids=lambda c: c.__name__)
def test_a_changed_camera_the_gate_proved_is_drawn_on_the_core(
    scene_class, findings, monkeypatch
):
    # As if the gate had proven every change XF carries, in every form.
    monkeypatch.setattr(
        painter,
        "PROVEN_ON_THE_GATE",
        frozenset(
            (form, member)
            for form in ("subclass", "class", "instance")
            for member in painter.CARRIED_MEMBERS
        ),
    )
    before = CorePainter.drawn_total
    _, described = render(scene_class, FPS, draw_path="ir_cairo", **MEDIUM)
    _, core = render(scene_class, FPS, draw_path="core", **MEDIUM)
    assert CorePainter.drawn_total > before, "drawn on the core"
    assert not _fallbacks(findings), findings
    assert len(described) == len(core) > 0
    # Where they differ, it is depth: the core tests it per pixel, CE paints
    # in its own order. The frames agree everywhere else.
    for index, (a, b) in enumerate(zip(described, core, strict=True)):
        differ = np.any(a != b, axis=-1).mean()
        assert differ < 0.05, f"frame {index}: {differ:.2%} of pixels differ"


def test_a_changed_camera_the_gate_has_not_proved_draws_on_cairo_for_the_whole_scene(
    findings, monkeypatch
):
    monkeypatch.setattr(painter, "PROVEN_ON_THE_GATE", frozenset())
    _, direct = render(camera_scenes.ShiftedSolids, FPS, **MEDIUM)
    before = CorePainter.drawn_total
    _, core = render(camera_scenes.ShiftedSolids, FPS, draw_path="core", **MEDIUM)
    assert CorePainter.drawn_total == before, "no frame drawn on the core"
    assert len(direct) == len(core) > 0
    for index, (a, b) in enumerate(zip(direct, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = _fallbacks(findings)
    assert "whole scene" in finding.message
    assert "project_points" in finding.message


class _Refusing:
    """A core stream that refuses every frame, as the core refuses a record
    it does not draw.
    """

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def draw(self, *args, **kwargs):
        raise arachne_core.Unsupported("path", 3, 7, "a path's clip is not drawn yet")


def test_a_frame_the_core_refuses_is_unable_not_drawn_on_another_path(monkeypatch):
    held = CorePainter.__init__

    def refusing(self, **kwargs):
        held(self, **kwargs)
        self.core = _Refusing(self.core)

    monkeypatch.setattr(CorePainter, "__init__", refusing)
    with pytest.raises(Unable, match="render.unable: the core does not draw a path"):
        render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)


def test_a_proven_change_that_rebinds_the_camera_s_state_draws_on_cairo_for_the_whole_scene(
    findings,
):
    """A change of a kind the gate proved whose code rebinds what the camera
    holds as it draws (here the line width, from the depth of what it
    projects) on every call. The core watches the first capture of each
    play, and draws the scene's very first capture as the camera draws it:
    the change is seen there, so the whole scene is drawn on Cairo, its
    first frame too, with one finding naming the code and the attribute.
    """
    scene_class = camera_scenes.ProjectingStateSolids
    _, direct = render(scene_class, FPS, **MEDIUM)
    before = CorePainter.drawn_total
    _, core = render(scene_class, FPS, draw_path="core", **MEDIUM)
    assert CorePainter.drawn_total == before, "no frame drawn on the core"
    assert len(direct) == len(core) > 0
    for index, (a, b) in enumerate(zip(direct, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = _fallbacks(findings)
    assert "whole scene" in finding.message
    assert "project_points" in finding.message
    assert "cairo_line_width_multiple" in finding.message


def test_a_rebinding_first_seen_in_a_later_play_is_unable():
    """The core has drawn the frames before it: a fallback now would show
    partway through the scene, so nothing draws it.
    """
    with pytest.raises(Unable, match="cairo_line_width_multiple"):
        render(camera_scenes.LaterPlayStateSolids, FPS, draw_path="core", **MEDIUM)


@pytest.mark.xfail(
    strict=True,
    reason="the residual of a sampled watch: a rebinding that begins after a"
    " play's first frame and ends before the next play's is not seen. The core"
    " draws it from its description: on the one frame of 15 it rebinds in, 3.9%"
    " of the pixels differ from the camera's own draw, against 2.9% for the"
    " same camera when it never rebinds",
)
def test_a_rebinding_that_begins_partway_through_a_play_is_seen(findings):
    try:
        render(camera_scenes.MidPlayStateSolids, FPS, draw_path="core", **MEDIUM)
    except Unable:
        return
    assert _fallbacks(findings), "the core drew the rebinding camera unseen"
