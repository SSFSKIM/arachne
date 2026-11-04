"""Which draw path a scene's frames take.

``core`` draws 3D frames on the core; ``auto`` is ``core`` for a 3D scene (its
camera a ``ThreeDCamera``) and ``cairo`` for every other, so a 2D scene never
goes through XF on ``auto``. A camera the scene's code changed draws on Cairo
on ``core`` until the core draws what such a camera's capture placed.
"""

from __future__ import annotations

import pytest

from arachne import config, tempconfig
from arachne.camera.camera import Camera
from arachne.camera.moving_camera import MovingCamera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.frame_ir.painter import (
    DRAW_PATHS,
    IrCairoPainter,
    camera_path,
    painter_class,
    scene_path,
)

from . import camera_scenes


@pytest.fixture
def findings(monkeypatch):
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    return lint.raised


@pytest.mark.parametrize("path", ["cairo", "ir_cairo", "core", "auto"])
def test_the_option_takes_each_draw_path(path):
    with tempconfig({"draw_path": path}):
        assert config.draw_path == path


def test_the_option_refuses_a_path_it_does_not_know():
    with pytest.raises(ValueError, match="draw_path"):
        config.draw_path = "gpu"


@pytest.mark.parametrize(
    ("camera_class", "path"),
    [(ThreeDCamera, "core"), (Camera, "cairo"), (MovingCamera, "cairo")],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_auto_is_the_core_for_a_3d_scene_and_cairo_for_a_2d_one(camera_class, path):
    assert scene_path(camera_class(), "auto") == path


def test_a_scene_s_own_3d_camera_class_is_a_3d_scene():
    assert scene_path(camera_scenes.ShiftedCamera(), "auto") == "core"


@pytest.mark.parametrize("path", sorted(DRAW_PATHS))
def test_a_named_path_is_itself_whatever_the_camera(path):
    assert scene_path(Camera(), path) == path
    assert scene_path(ThreeDCamera(), path) == path


def test_an_unknown_path_is_refused():
    with pytest.raises(ValueError, match="auto"):
        scene_path(Camera(), "gpu")


def test_an_engine_camera_on_auto_draws_on_its_scene_s_path(findings):
    assert camera_path(ThreeDCamera(), "auto") == "core"
    assert camera_path(Camera(), "auto") == "cairo"
    assert not findings


def test_a_changed_camera_on_the_core_draws_on_cairo_with_its_reason(findings):
    assert camera_path(camera_scenes.ShiftedCamera(), "core") == "cairo"
    [finding] = findings
    assert finding.code == "render.fallback"
    assert "the core does not draw what a camera's own capture placed" in (
        finding.message
    )


def test_a_changed_camera_on_ir_cairo_stays_described(findings):
    assert camera_path(camera_scenes.ShiftedCamera(), "ir_cairo") == "ir_cairo"
    assert not findings


def test_each_path_through_xf_has_its_painter():
    from arachne.frame_ir.core_painter import CorePainter

    assert painter_class("ir_cairo") is IrCairoPainter
    assert painter_class("core") is CorePainter
