"""A scene the core cannot draw is drawn by Cairo, whole.

On the core a frame the core cannot draw exactly is ``render.unable``: it is
never drawn wrong, and never drawn by another painter among the core's
frames (one movie, one painter). That refusal is the core's, not the
scene's: the render executes the scene again from the start on Cairo, every
frame of it, with a ``render.fallback`` finding carrying the refusal. A scene
Cairo renders always renders.

Frames here are 192 x 108 pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import DEGREES, ThreeDCamera, tempconfig
from arachne.frame_ir.core_painter import CorePainter

from ..timeline.conftest import SMALL, render
from . import camera_scenes, core_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}
WHITE = (255, 255, 255, 255)


@pytest.fixture
def findings(monkeypatch):
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    return lint.raised


def _fallbacks(findings: list) -> list:
    return [f for f in findings if f.code == "render.fallback"]


def _as_on_cairo(scene_class, **settings):
    """``scene_class`` asked for on the core renders, every frame Cairo's."""
    _, cairo = render(scene_class, FPS, **MEDIUM, **settings)
    _, core = render(scene_class, FPS, draw_path="core", **MEDIUM, **settings)
    assert len(core) == len(cairo) > 0
    for index, (a, b) in enumerate(zip(cairo, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


class LateWritingCamera(ThreeDCamera):
    """A projection, a change the gate proved, that writes a patch of the
    camera's array once the camera has turned (the scene's second play):
    after the core has delivered frames of the scene.
    """

    def project_points(self, points):
        if self.get_theta() > -30 * DEGREES:
            self.pixel_array[0:3, 0:3] = WHITE
        return super().project_points(points)


class LateWritingSolids(camera_scenes.Solids):
    """:class:`camera_scenes.Solids` through :class:`LateWritingCamera`."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=LateWritingCamera, **kwargs)


@pytest.mark.parametrize(
    ("scene_class", "named"),
    [
        # The camera's code rebinds what it draws with, first in a later play.
        (camera_scenes.LaterPlayStateSolids, "cairo_line_width_multiple"),
        # The camera's code writes its array, first after frames were delivered.
        (LateWritingSolids, "wrote the camera's array"),
    ],
    ids=["rebinding", "writing"],
)
def test_a_scene_the_live_core_route_is_unable_to_draw_renders_on_cairo(
    scene_class, named, findings
):
    """The core drew frames of the scene before it met what it cannot draw:
    the render is not lost, the scene is executed again on Cairo, and every
    frame is Cairo's, byte for byte.
    """
    _as_on_cairo(scene_class)
    [finding] = [f for f in _fallbacks(findings) if "render.unable" in f.message]
    assert finding.subject == scene_class.__name__
    assert "for the whole scene" in finding.message
    assert named in finding.message


class _Refusing:
    """A core stream that applies deltas and refuses to draw (an
    ``Unsupported`` a frame of the record route meets).
    """

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def draw(self, *args, **kwargs):
        import arachne_core

        raise arachne_core.Unsupported("path", 3, 7, "a path's clip is not drawn yet")


def test_a_scene_whose_recorded_frame_the_core_refuses_renders_on_cairo(
    monkeypatch, findings
):
    held = CorePainter.__init__

    def refusing(self, **kwargs):
        held(self, **kwargs)
        self.core = _Refusing(self.core)

    _, cairo = render(core_scenes.FixedLabelOverSquare, FPS, **MEDIUM)
    monkeypatch.setattr(CorePainter, "__init__", refusing)
    _, core = render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    assert len(core) == len(cairo) > 0
    for index, (a, b) in enumerate(zip(cairo, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = _fallbacks(findings)
    assert "render.unable: the core does not draw a path" in finding.message


def test_the_draw_path_asked_for_is_kept_after_a_scene_drawn_on_cairo(findings):
    """Drawing the scene on Cairo is the scene's: the option is as it was."""
    from arachne import config

    with tempconfig(
        {**SMALL, "frame_rate": FPS, "dry_run": True, "disable_caching": True}
    ):
        config.draw_path = "core"
        LateWritingSolids().render()
        assert config.draw_path == "core"
    assert _fallbacks(findings)


def test_a_play_drawn_on_the_core_is_cached_apart_from_its_cairo_frames():
    """A movie segment drawn on the core is not one drawn by Cairo: a render
    on either path never reuses the other's, as the scene drawn on Cairo
    after a refusal must not reuse the core's segments of the plays before.
    """
    hashes = {}
    for path in ("cairo", "ir_cairo", "core"):
        scene, _ = render(
            core_scenes.FixedLabelOverSquare,
            FPS,
            draw_path=path,
            disable_caching=False,
            **MEDIUM,
        )
        hashes[path] = scene.renderer.animations_hashes
    assert all(hashes.values())
    assert not set(hashes["core"]) & set(hashes["cairo"])
    # Cairo draws every pixel of both of its paths.
    assert hashes["ir_cairo"] == hashes["cairo"]
