"""The core's frames do not depend on the last bits of its geometric inputs.

The geometry a scene computes with numpy's vectorised loops differs between
CPUs in its last bits; the core decides by comparison, so unsnapped, a
difference of one ulp moves whole pixels. Each test hands the core a scene's
deltas with every geometric float moved 1 to 4 ulp (seeded), and the frames
must be those drawn from the deltas as made, byte for byte.
"""

from __future__ import annotations

import copy
from itertools import pairwise

import numpy as np
import pytest

from arachne import (
    DEGREES,
    IN,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Line3D,
    Square,
    ThreeDAxes,
    ThreeDScene,
)
from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.xf import Delta, Record, Shape, StreamState

from ..timeline.conftest import render
from . import core_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}

#: The view's geometry: what a scene's camera computes.
_VIEW = (
    "frame_width",
    "frame_height",
    "frame_center",
    "mapping_centre",
    "phi",
    "theta",
    "gamma",
    "focal_distance",
    "zoom",
    "light_source",
)


class ParallelLines(ThreeDScene):
    """Two parallel lines in space over axes, under a turn of the camera:
    lines whose depth crosses the axes', tested at every sub-sample.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3), 6, 6, 6)
        line = Line3D(RIGHT * 2, UP + OUT, color=RED)
        twin = Line3D(RIGHT * 2 + IN, UP, color=YELLOW)
        self.add(axes, line, twin)
        self.move_camera(theta=-30 * DEGREES, run_time=0.2)


def _nudged(values, rng) -> np.ndarray:
    """``values`` each moved by 1 to 4 ulp, up or down."""
    found = np.array(values, dtype=np.float64)
    steps = rng.integers(1, 5, found.shape) * rng.choice([-1, 1], found.shape)
    toward = np.where(steps > 0, np.inf, -np.inf)
    for step in range(4):
        found = np.where(np.abs(steps) > step, np.nextafter(found, toward), found)
    return found


def _as_given(old, new):
    return tuple(float(v) for v in new) if isinstance(old, tuple) else float(new)


def _jittering(seed: int):
    """A ``CorePainter.draw`` handing on each delta with its geometry nudged:
    a path shape's points once, under a key of its own, kept across frames;
    a record's gradient ends and sort point; the view's geometry.
    """
    draw = CorePainter.draw
    rng = np.random.default_rng(seed)

    def jittered(self, delta):
        keys = vars(self).setdefault("_nudged_shapes", {})
        found = copy.copy(delta)
        if delta.view is not None:
            view = dict(delta.view)
            for name in _VIEW:
                if name in view:
                    view[name] = _as_given(view[name], _nudged(view[name], rng))
            found.view = view
        upload = []
        for shape in delta.upload:
            if shape.kind == "path":
                if shape.key not in keys:
                    keys[shape.key] = Shape("path", _nudged(shape.data, rng))
                upload.append(keys[shape.key])
            else:
                upload.append(shape)
        found.upload = upload
        found.release = [keys[k].key if k in keys else k for k in delta.release]
        records = []
        for slot, record in delta.set:
            fields = dict(record.fields)
            if fields.get("shape") in keys:
                fields["shape"] = keys[fields["shape"]].key
            for name in ("gradient", "sort_point"):
                if fields.get(name) is not None:
                    fields[name] = _as_given(fields[name], _nudged(fields[name], rng))
            records.append((slot, Record(record.kind, fields)))
        found.set = records
        return draw(self, found)

    return jittered


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_core_s_frames_do_not_move_with_the_last_bits_of_its_inputs(
    seed, monkeypatch
):
    _, made = render(ParallelLines, FPS, draw_path="core", **MEDIUM)
    monkeypatch.setattr(CorePainter, "draw", _jittering(seed))
    _, nudged = render(ParallelLines, FPS, draw_path="core", **MEDIUM)
    assert len(made) == len(nudged) > 0
    for index, (a, b) in enumerate(zip(made, nudged, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


class ZoomOutAndBack(ThreeDScene):
    """A square on a tilted plane and a line over it, under a zoom across two
    powers of two and back: the frame's input grid moves twice each way.
    ``line`` is the line's class.
    """

    line = Line3D

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        square = Square(side_length=3, color=RED, fill_opacity=1.0)
        square.rotate(25 * DEGREES, axis=RIGHT).rotate(10 * DEGREES, axis=UP)
        line = self.line(RIGHT * 2 + IN * 0.3, UP + OUT, color=YELLOW)
        self.add(square, line)
        self.move_camera(zoom=2.6, run_time=0.4)
        self.move_camera(zoom=1.0, run_time=0.4)


class LiveZoomOutAndBack(ZoomOutAndBack):
    """:class:`ZoomOutAndBack` drawn as the scene runs."""

    line = core_scenes.OwnLine


def _whole(state: StreamState, delta: Delta) -> Delta:
    """The frame ``state`` holds, as one delta a fresh drawer takes whole."""
    return Delta(
        t=delta.t,
        seq=delta.seq,
        view=state.view,
        objects=[(ref, *found) for ref, found in state.objects.items()],
        upload=list(state.shapes.values()),
        upload_paints=list(state.paints.values()),
        set=sorted(state.records.items()),
        order=list(state.order),
    )


@pytest.mark.parametrize(
    "scene_class", [ZoomOutAndBack, LiveZoomOutAndBack], ids=["recorded", "live"]
)
def test_a_zoom_out_and_back_snaps_each_frame_as_it_would_alone(
    scene_class, monkeypatch
):
    """Each frame of a zoom whose input grid moves and comes back is the
    frame a fresh drawer draws from that frame alone, at its own grid: the
    geometry is snapped again from the scene's values when the grid moves,
    never from what an earlier grid made of them.
    """
    draw = CorePainter.draw
    grids: list[float] = []

    def checking(self, delta):
        state = vars(self).setdefault("_scene_state", StreamState())
        state.apply(delta)
        frame = draw(self, delta)
        grids.append(self.canonical.grid)
        alone = draw(CorePainter(), _whole(state, delta))
        np.testing.assert_array_equal(frame, alone, err_msg=f"frame {len(grids)}")
        return frame

    monkeypatch.setattr(CorePainter, "draw", checking)
    render(scene_class, FPS, draw_path="core", **MEDIUM)
    moved = [grids[0], *(b for a, b in pairwise(grids) if b != a)]
    assert moved == [grids[0], grids[0] / 2, grids[0] / 4, grids[0] / 2, grids[0]]


class StronglyZoomed(core_scenes.ZoomedFrameMoves):
    """:class:`ZoomedFrameMoves` at a zoom factor of 0.02: its display's
    camera spans a fiftieth of the frame.
    """

    def __init__(self, **kwargs):
        super(core_scenes.ZoomedFrameMoves, self).__init__(
            zoom_factor=0.02, zoomed_display_height=2, zoomed_display_width=3, **kwargs
        )


@pytest.mark.parametrize(
    "scene_class",
    [ParallelLines, LiveZoomOutAndBack, StronglyZoomed],
    ids=["recorded", "live", "zoomed"],
)
def test_what_the_core_is_handed_carries_its_grid_and_planes(scene_class, monkeypatch):
    """Every delta the core is handed states its input grid, and every record
    that draws paths carries how they lie: a stream that lost either would
    have the core decide flatness from snapped points. The grid is fine
    enough for every camera that sees the geometry: a step is at most 1/128
    of the pixel of the scene's camera and of each display's.
    """
    call = Canonical.__call__
    steps: list[float] = []

    def checking(self, delta):
        found = call(self, delta)
        if self._decider is None:
            return found
        if found.view is not None:
            assert found.view["input_grid"] > 0
        views = [] if self._view is None else [self._view]
        for slot, record in found.set:
            if record.kind in ("path", "face_group") and record.fields.get("shape"):
                assert "planes" in record.fields, f"slot {slot}: {record.kind}"
            if record.fields.get("sub_frame") is not None:
                assert record.fields["sub_frame"][1]["input_grid"] == self.grid
        views += [
            record.fields["sub_frame"][1]
            for record in self._records.values()
            if record.fields.get("sub_frame")
        ]
        for view in views:
            span = view["frame_width"] / (view["pixel_width"] * view.get("zoom", 1.0))
            steps.append(self.grid / span)
        return found

    monkeypatch.setattr(Canonical, "__call__", checking)
    render(scene_class, FPS, draw_path="core", **MEDIUM)
    assert steps
    assert max(steps) <= 1 / 128


class LineAcrossTheFrame(ThreeDScene):
    """A thin line in space (a cylinder of radius 0.02, its sides between
    arcs of its cross-section), over axes.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(
            ThreeDAxes((-3, 3), (-3, 3), (-3, 3), 6, 6, 6),
            Line3D(RIGHT * 2, UP + OUT, color=RED),
        )
        self.wait(0.1)


def test_a_thin_curved_face_keeps_its_coverage_when_snapped():
    """A thin line's sides are curved by less than the snap moves a point:
    decided before the snap, they keep their curve's depth, and the line
    covers what Cairo covers, within 1% (its red, less its green, summed).
    """
    full = {"pixel_width": 854, "pixel_height": 480}
    _, cairo = render(LineAcrossTheFrame, FPS, draw_path="ir_cairo", **full)
    _, core = render(LineAcrossTheFrame, FPS, draw_path="core", **full)

    def red(frame):
        rgb = frame[..., :3].astype(np.int64)
        return int(np.clip(rgb[..., 0] - rgb[..., 1], 0, None).sum())

    assert red(core[0]) == pytest.approx(red(cairo[0]), rel=0.01)
