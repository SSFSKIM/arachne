"""A camera the scene's code changed is described in XF by what it produced.

A scene's own camera class, or an engine camera with a method replaced on the
instance or the class (the scene's camera or a display's), draws the picture
its code makes. XF cannot run that code, so the frame is described from the
camera's own capture, run once as CE runs it: the points its pre-display code
placed and their depth, the colours it shaded, the order it drew in, on
records marked pre-projected. Cairo draws them back byte for byte. A change
whose effect reaches the frame other than through those values (Cairo calls
of the scene's own, pixels written by hand) is drawn on Cairo, with a finding
naming it; a camera that is no engine camera is unable.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.live import live_painter
from arachne.frame_ir.painter import IrCairoPainter
from arachne.frame_ir.xf import FEATURE_PRE_PROJECTED

from ..timeline.conftest import render
from . import camera_scenes, scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


@pytest.fixture
def findings(monkeypatch):
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    return lint.raised


@pytest.fixture
def kept(monkeypatch):
    monkeypatch.setattr(IrCairoPainter, "keep_all", True)


def _both(scene_class: type) -> tuple[list, list, object]:
    _, direct = render(scene_class, FPS, **MEDIUM)
    scene, described = render(scene_class, FPS, draw_path="ir_cairo", **MEDIUM)
    return direct, described, scene


def _same(direct: list, described: list) -> None:
    assert len(direct) == len(described) > 0
    for index, (a, b) in enumerate(zip(direct, described, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def _pre_projected(scene: object) -> int:
    """How many pre-projected records the scene's captures sent, over its
    camera and its displays' cameras.
    """
    camera = scene.renderer.camera
    painter = live_painter(camera)
    assert painter is not None, "the camera's captures were described"
    cameras = [
        camera,
        *(d.camera for d in vars(camera).get("image_mobjects_from_cameras", ())),
    ]
    return sum(
        1
        for each in cameras
        for delta in painter.painter(each).deltas
        for _, record in delta.set
        if int(record.fields.get("features", 0)) & FEATURE_PRE_PROJECTED
    )


@pytest.mark.parametrize(
    ("scene_class", "from_record"),
    [
        (camera_scenes.ShiftedSolids, False),
        (camera_scenes.PatchedSolids, True),
        (camera_scenes.TintedSolids, True),
        (camera_scenes.ZoomMirrored, False),
        (scenes.UserCamera, False),
        (scenes.PatchedCamera, True),
        (scenes.LivePatchedCamera, False),
        (scenes.PatchedOrder, True),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_changed_camera_is_drawn_from_what_it_produced(
    scene_class, from_record, findings, kept
):
    before = IrCairoPainter.drawn_total
    direct, described, scene = _both(scene_class)
    _same(direct, described)
    assert IrCairoPainter.drawn_total > before, "drawn from XF"
    assert (scene.timeline.record.refusal is None) == from_record
    assert not [f for f in findings if f.code == "render.fallback"], findings
    if scene_class is not scenes.PatchedOrder:
        # A changed order alone moves no point: its records are the plain ones.
        assert _pre_projected(scene) > 0


def test_a_class_patch_of_a_method_xf_carries_is_drawn_from_xf(monkeypatch, findings):
    from arachne.camera.camera import Camera

    monkeypatch.setattr(
        Camera,
        "get_mobjects_to_display",
        scenes._reversed(Camera.get_mobjects_to_display),
    )
    before = IrCairoPainter.drawn_total
    direct, described, _ = _both(scenes.Overlapping)
    _same(direct, described)
    assert IrCairoPainter.drawn_total > before
    assert not findings


@pytest.mark.parametrize(
    "scene_class",
    [camera_scenes.CountedLive, camera_scenes.CountedRecord],
    ids=lambda c: c.__name__,
)
def test_the_camera_s_code_runs_as_often_as_on_cairo(scene_class):
    """The frame is described from the camera's own capture, run once: so a
    projection or shading that counts its calls, or changes state, runs as
    often as when Cairo draws, on the record path and as the scene runs.
    """
    calls = camera_scenes.Counted.calls
    counts = []
    frames = []
    for draw_path in ("cairo", "ir_cairo"):
        calls.update(dict.fromkeys(calls, 0))
        before = IrCairoPainter.drawn_total
        _, drawn = render(scene_class, FPS, draw_path=draw_path, **MEDIUM)
        counts.append(dict(calls))
        frames.append(drawn)
        assert (IrCairoPainter.drawn_total > before) == (draw_path == "ir_cairo")
    assert all(counts[0][name] > 0 for name in calls), counts[0]
    assert counts[0] == counts[1]
    _same(*frames)


def test_a_class_patch_of_the_rotation_runs_as_often_as_on_cairo(monkeypatch):
    """The drawer's own cameras turn their views with the engine's rotation,
    never a patch of it: the scene's runs in the scene's capture alone.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    calls = [0]
    rotation = ThreeDCamera.generate_rotation_matrix

    def counted(self):
        calls[0] += 1
        return rotation(self)

    monkeypatch.setattr(ThreeDCamera, "generate_rotation_matrix", counted)
    counts = []
    frames = []
    for draw_path in ("cairo", "ir_cairo"):
        calls[0] = 0
        before = IrCairoPainter.drawn_total
        _, drawn = render(camera_scenes.Solids, FPS, draw_path=draw_path, **MEDIUM)
        counts.append(calls[0])
        frames.append(drawn)
        assert (IrCairoPainter.drawn_total > before) == (draw_path == "ir_cairo")
    assert counts[0] > 0
    assert counts[0] == counts[1]
    _same(*frames)


def test_a_class_patch_of_a_camera_s_constructor_draws_on_cairo(monkeypatch, findings):
    """The drawer builds engine cameras of its own, so a patched constructor
    would run there: the scene draws on Cairo, the constructor named.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    built = ThreeDCamera.__init__

    def patched(self, *args, **kwargs):
        built(self, *args, **kwargs)

    monkeypatch.setattr(ThreeDCamera, "__init__", patched)
    before = IrCairoPainter.drawn_total
    direct, described, _ = _both(camera_scenes.Solids)
    _same(direct, described)
    assert IrCairoPainter.drawn_total == before, "nothing drawn from XF"
    assert findings
    assert {f.code for f in findings} == {"render.fallback"}
    assert "__init__ reaches the frame other than through" in findings[0].message


@pytest.mark.parametrize(
    ("scene_class", "subject", "member"),
    [
        (scenes.ZoomPatched, "MovingCamera.capture_mobjects", "capture_mobjects"),
        (scenes.ZoomOwnClass, "BandedCamera", "capture_mobjects"),
        (
            camera_scenes.StrokedTwice,
            "ThreeDCamera.display_vectorized",
            "display_vectorized",
        ),
        (camera_scenes.Bordered, "BorderedCamera", "display_funcs"),
        (camera_scenes.Rewired, "ThreeDCamera.display_funcs", "display_funcs"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_a_change_xf_does_not_carry_draws_on_cairo_with_its_reason(
    scene_class, subject, member, findings
):
    before = IrCairoPainter.drawn_total
    direct, described, _ = _both(scene_class)
    _same(direct, described)
    assert IrCairoPainter.drawn_total == before, "nothing drawn from XF"
    [finding] = findings
    assert (finding.code, finding.subject) == ("render.fallback", subject)
    assert f"{member} reaches the frame other than through" in finding.message
