"""Live frames in XF: a scene drawn as it runs is described from what the
camera's own capture consumed, drawn back by Cairo byte for byte, and its
code runs exactly as often as on the direct path.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir import live as live_module
from arachne.frame_ir.extract import Unable

from ..timeline.conftest import render
from . import scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


@pytest.fixture
def painters(monkeypatch):
    """Every live painter made while the test runs."""
    made: list[live_module.LivePainter] = []
    original = live_module.LivePainter.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(live_module.LivePainter, "__init__", init)
    return made


def _both_ways(scene_class, painters):
    scenes.CountedFill.calls = 0
    scene, direct = render(scene_class, FPS, **MEDIUM)
    direct_calls = scenes.CountedFill.calls
    painters.clear()
    scenes.CountedFill.calls = 0
    described_scene, described = render(
        scene_class, FPS, draw_path="ir_cairo", **MEDIUM
    )
    return (scene, direct, direct_calls), (described_scene, described)


@pytest.mark.parametrize("scene_class", scenes.LIVE_SCENES, ids=lambda c: c.__name__)
def test_a_live_scene_drawn_from_xf_is_byte_identical(scene_class, painters):
    (scene, direct, _), (described_scene, described) = _both_ways(scene_class, painters)
    assert described_scene.timeline.record.refusal is not None, "drawn live"
    assert len(direct) == len(described) > 0
    for index, (a, b) in enumerate(zip(direct, described, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    # Each capture described (a held wait's frames are its one capture).
    assert sum(p.captures for p in painters) > 0
    assert sum(p.differing for p in painters) == 0


def test_a_live_override_runs_as_often_as_on_the_direct_path(painters):
    (_, _, direct_calls), _ = _both_ways(scenes.LiveCounted, painters)
    assert direct_calls > 0
    assert scenes.CountedFill.calls == direct_calls


def test_a_mapping_camera_records_the_mapped_points(painters, monkeypatch):
    from arachne.frame_ir.painter import IrCairoPainter

    monkeypatch.setattr(IrCairoPainter, "keep_all", True)
    _, (scene, _) = _both_ways(scenes.LiveMapped, painters)
    camera = scene.renderer.camera
    painter = live_module.live_painter(camera)
    assert painter is not None
    deltas = painter.painter(camera).deltas
    shapes = {shape.key: shape.data for d in deltas for shape in d.upload}
    clouds = [r for d in deltas for _, r in d.set if r.kind == "points"]
    # The closing wait draws the cloud where it ends, mapped.
    cloud = scene.mobjects[0].points
    mapped = np.apply_along_axis(camera.mapping_func, 1, cloud)
    assert not np.array_equal(mapped, cloud)
    assert any(np.array_equal(shapes[r.fields["shape"]], mapped) for r in clouds)


@pytest.mark.parametrize(
    "scene_class",
    [scenes.LiveCounted, *scenes.MOVED_CENTRE_LIVE],
    ids=lambda c: c.__name__,
)
def test_without_drawing_the_capture_describes_the_same_frames(
    scene_class, painters, monkeypatch
):
    """The core path's capture: Cairo's calls go nowhere, the scene's code
    still runs as often, CE's context still freezes its centre, and the
    frames drawn from XF are the same.
    """
    original = live_module.LivePainter.__init__

    def quiet(self, *, draw=True):
        original(self, draw=False)

    monkeypatch.setattr(live_module.LivePainter, "__init__", quiet)
    (_, direct, direct_calls), (_, described) = _both_ways(scene_class, painters)
    assert scenes.CountedFill.calls == direct_calls
    assert len(direct) == len(described) > 0
    for index, (a, b) in enumerate(zip(direct, described, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def test_a_read_the_record_would_answer_otherwise_is_unable():
    with pytest.raises(Unable, match="render.unable: a SplitAlone's"):
        render(scenes.LiveUnstated, FPS, draw_path="ir_cairo", **MEDIUM)
