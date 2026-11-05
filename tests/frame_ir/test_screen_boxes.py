"""The core's screen boxes against the screen projection: where the core
places each record of a frame described in XF is where the camera's own
pre-display code and frame mapping put its mobject
(:mod:`arachne.camera.projection`), within a pixel.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.camera import projection
from arachne.frame_ir.painter import IrCairoPainter

from ..timeline.conftest import render
from ..timeline.scenes import TimelineDiagram
from . import camera_scenes, scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}
#: The parity bound (a pixel).
BOUND = 1.0


def projected(camera, view, mob, kind):
    """The projection's box of a record's mobject. A record is one drawn leaf
    (its family's other members are records of their own), but a face group
    is its Surface's faces together.
    """
    if kind == "face_group":
        return projection.pixel_box(camera, view, mob)
    if not len(mob.points):
        return None
    placed = projection.pixel_points(camera, view, mob, mob.points)
    low, high = placed.min(axis=0), placed.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def boxes_against_the_projection(monkeypatch, scene_class) -> list[tuple]:
    """Render ``scene_class`` on ``ir_cairo``, and at every frame the record
    draws through XF, the core's box of each slot beside the projection's box
    of its mobject, and that box were paths mapped around the live frame
    centre, as ``(t, slot, kind, core, projected, live)``.
    """
    found: list[tuple] = []
    paint = IrCairoPainter.paint

    def watched(self, camera, groups, **kwargs):
        pixels = paint(self, camera, groups, **kwargs)
        held = self.stream._view
        view = projection.view_of(camera, mapping_centre=held["mapping_centre"])
        live = projection.view_of(camera, mapping_centre=camera.frame_center)
        mobs = {slot: mob for mob, slot in self.stream._slots.values()}
        for slot, box in self.core.screen_boxes().items():
            mob = mobs[slot]
            kind = self.stream._held[slot].kind
            found.append(
                (
                    kwargs.get("t"),
                    slot,
                    f"{kind} {type(mob).__name__}",
                    box,
                    projected(camera, view, mob, kind),
                    projected(camera, live, mob, kind),
                )
            )
        return pixels

    monkeypatch.setattr(IrCairoPainter, "paint", watched)
    render(scene_class, FPS, draw_path="ir_cairo", **MEDIUM)
    return found


@pytest.mark.parametrize(
    "scene_class",
    [
        scenes.Lit3D,  # a face group, axes, a label fixed in frame, a fixed orientation
        scenes.Orbit,  # a camera move over unchanged records
        scenes.MovedThreeDCentre,  # the frozen mapping centre against the live one
        scenes.MovedFlatCentre,
        scenes.Clouds,  # point clouds, mapped around the frame's centre
        scenes.Watched,  # a fixed orientation about a moving centre
        scenes.Reveals,  # reveal windows
        scenes.Morph,  # straight-path blends
    ],
    ids=lambda c: c.__name__,
)
def test_the_core_places_each_record_where_the_camera_does(monkeypatch, scene_class):
    found = boxes_against_the_projection(monkeypatch, scene_class)
    assert found, "boxes were compared"
    worst = 0.0
    for t, slot, kind, core, placed, _ in found:
        assert placed is not None, (t, slot, kind)
        error = float(np.max(np.abs(np.subtract(core, placed))))
        worst = max(worst, error)
        assert error <= BOUND, (t, slot, kind, core, placed)
    print(f"{scene_class.__name__}: {len(found)} boxes, worst {worst:.3g} px")


@pytest.mark.parametrize(
    ("scene_class", "kinds"),
    [
        # Drawn as the scene runs, and from the record: each kind a camera of
        # the scene's own places, boxed from the points it placed (a Surface's
        # faces are paths of their own there, captured leaf by leaf).
        (camera_scenes.ShiftedSolids, {"path"}),
        (camera_scenes.ShiftedPictures, {"points", "image"}),
        (camera_scenes.PatchedPictures, {"points", "image"}),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_changed_camera_s_records_are_boxed_where_it_placed_them(
    monkeypatch, scene_class, kinds
):
    from arachne.frame_ir import live as live_module

    found: list[tuple] = []
    describe = live_module.LivePainter._describe

    def described(self, camera, drawn, centres, background):
        pixels = describe(self, camera, drawn, centres, background)
        painter = self.painter(camera)
        view = projection.view_of(
            camera, mapping_centre=painter.stream._view["mapping_centre"]
        )
        mobs = {slot: mob for mob, slot in painter.stream._slots.values()}
        for slot, box in painter.core.screen_boxes().items():
            kind = painter.stream._held[slot].kind
            found.append((slot, kind, box, projected(camera, view, mobs[slot], kind)))
        return pixels

    monkeypatch.setattr(live_module.LivePainter, "_describe", described)
    render(scene_class, FPS, draw_path="ir_cairo", **MEDIUM)
    assert kinds <= {kind for _, kind, _, _ in found}, found
    for slot, kind, core, placed in found:
        assert placed is not None, (slot, kind)
        error = float(np.max(np.abs(np.subtract(core, placed))))
        assert error <= BOUND, (slot, kind, core, placed)


def test_a_moved_three_d_centre_maps_paths_around_the_frozen_centre(monkeypatch):
    """Under a 3D camera whose centre moved, a path lands where CE's frozen
    context puts it, a full pixel and more from where the live centre would.
    """
    found = boxes_against_the_projection(monkeypatch, scenes.MovedThreeDCentre)
    squares = [e for e in found if e[2] == "path Square"]
    apart = [e for e in squares if np.max(np.abs(np.subtract(e[3], e[5]))) > 5 * BOUND]
    assert apart, "the centre moved, so some frame maps paths around another centre"


def test_a_scene_s_own_timeline_leaves_its_record_view_the_recorder_s():
    """The view of a record takes the configured frame from the engine's
    recorder, whatever the scene's code keeps as ``self.timeline``.
    """
    scene, _ = render(TimelineDiagram, FPS, **MEDIUM)
    view = projection.record_view(scene.record)
    width, height = MEDIUM["pixel_width"], MEDIUM["pixel_height"]
    assert (view.home_width, view.home_height) == projection.configured_home(
        width, height
    )
    assert view.configured_pixels == (width, height)
