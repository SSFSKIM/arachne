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
    BLUE,
    DEGREES,
    IN,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Line,
    Line3D,
    Square,
    ThreeDAxes,
    ThreeDScene,
)
from arachne.frame_ir import canonical
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


def _witnessing(seen: list, draw):
    """``draw`` (a ``CorePainter.draw``) keeping in ``seen``, for each frame
    it draws: the frame; the pixels a leaf Cairo paints touches (from the
    painter's contributors); and that overlay drawn by CE's Cairo from the
    same input, the delta as handed, through XF unsnapped (``ir_cairo``),
    over the core's own frame.
    """
    from arachne.frame_ir.cairo_painter import CairoDrawer
    from arachne.frame_ir.painter import IrCairoPainter

    def witnessed(self, delta):
        frame = draw(self, delta)
        twin = vars(self).setdefault("_twin", IrCairoPainter())
        twin.draw(delta)
        width, height = self.state.view["pixel_width"], self.state.view["pixel_height"]
        ce = np.array(self.core.draw(width, height, threads=1, overlay=False))
        overlay = self.core.overlay_entries()
        if overlay:
            CairoDrawer().draw_over(twin.state, ce, overlay)
        painted = np.zeros((height, width), dtype=bool)
        for (x, y), pixel in self.contributors((0, 0, width, height)).items():
            painted[y, x] = any(leaf.painter == "cairo" for leaf in pixel.leaves)
        seen.append((frame.copy(), painted, ce))
        return frame

    return witnessed


@pytest.mark.parametrize(
    ("scene_class", "size"),
    [
        (ParallelLines, MEDIUM),
        (core_scenes.FixedLabelOverSquare, MEDIUM),
        (core_scenes.CornerLabel, {"pixel_width": 854, "pixel_height": 480}),
    ],
    ids=["lines", "overlay", "tie"],
)
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_core_s_frames_do_not_move_with_the_last_bits_of_its_inputs(
    seed, scene_class, size, monkeypatch
):
    """What the core rasterizes does not move at all. What Cairo paints over
    it, drawn from the scene's own values, moves only as CE's Cairo moves
    with the same input: its pixels are the overlay CE's Cairo draws from
    the nudged values, byte for byte, a tie in Cairo's own rounding
    (:class:`~.core_scenes.CornerLabel`) included.
    """
    draw = CorePainter.draw
    made: list = []
    monkeypatch.setattr(CorePainter, "draw", _witnessing(made, draw))
    render(scene_class, FPS, draw_path="core", **size)
    nudged: list = []
    monkeypatch.setattr(CorePainter, "draw", _witnessing(nudged, draw))
    # Nudged as the core path's painter is handed each delta, before both.
    monkeypatch.setattr(CorePainter, "draw", _jittering(seed))
    render(scene_class, FPS, draw_path="core", **size)
    assert len(made) == len(nudged) > 0
    for index, ((a, _, _), (b, painted, ce)) in enumerate(
        zip(made, nudged, strict=True)
    ):
        core = ~painted
        assert np.array_equal(a[core], b[core]), (
            f"frame {index}: {int((a[core] != b[core]).any(-1).sum())} core pixels moved"
        )
        assert np.array_equal(b[painted], ce[painted]), (
            f"frame {index}: {int((b[painted] != ce[painted]).any(-1).sum())}"
            " pixels Cairo paints differ from CE's Cairo on the same input"
        )


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


class StronglyZoomedLine(StronglyZoomed):
    """:class:`StronglyZoomed` with a line in the main frame between points
    off any grid. ``line`` is the line's class.
    """

    line = Line

    def construct(self):
        start, end = [-3.1234567, -2.2345678, 0.0], [2.9876543, -1.3456789, 0.0]
        self.add(self.line(start, end, color=YELLOW))
        super().construct()


class LiveStronglyZoomedLine(StronglyZoomedLine):
    """:class:`StronglyZoomedLine` drawn as the scene runs."""

    line = core_scenes.OwnLine


def test_a_zoomed_scene_drawn_live_is_snapped_as_drawn_from_its_record(monkeypatch):
    """Drawn as it runs, a zoomed scene's main frame is snapped at the grid
    its display's camera sets, as when it is drawn from its record: every
    frame's grid is the same on both routes, and so are its pixels.
    """
    call = Canonical.__call__
    grids: dict[str, dict] = {"recorded": {}, "live": {}}
    route = ["recorded"]

    def recording(self, delta):
        found = call(self, delta)
        grids[route[0]].setdefault(delta.t, set()).add(self.grid)
        return found

    monkeypatch.setattr(Canonical, "__call__", recording)
    _, recorded = render(StronglyZoomedLine, FPS, draw_path="core", **MEDIUM)
    route[0] = "live"
    _, live = render(LiveStronglyZoomedLine, FPS, draw_path="core", **MEDIUM)
    assert all(len(found) == 1 for found in grids["recorded"].values())
    assert grids["live"] == grids["recorded"]
    assert len(live) == len(recorded)
    for k, (a, b) in enumerate(zip(recorded, live, strict=True)):
        assert np.array_equal(a, b), k


@pytest.mark.parametrize(
    "scene_class",
    [ParallelLines, LiveZoomOutAndBack, StronglyZoomed],
    ids=["recorded", "live", "zoomed"],
)
def test_what_the_core_is_handed_carries_its_grid_and_planes(scene_class, monkeypatch):
    """Every delta the core is handed states its input grid, every record that
    draws paths carries how they lie, and every record the camera shades the
    normals it is lit by: a stream that lost any would have the core decide
    them from snapped points. The grid is fine enough for every camera that
    sees the geometry: a step is at most 1/256 of the pixel of the scene's
    camera and of each display's.
    """
    call = Canonical.__call__
    steps: list[float] = []
    shaded: list[int] = []

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
                if record.fields.get("shade_in_3d") and self._view.get(
                    "should_apply_shading"
                ):
                    assert "normals" in record.fields, f"slot {slot}: {record.kind}"
                    shaded.append(slot)
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
    assert max(steps) <= 1 / 256
    # The parallel lines are shaded: their normals were checked, not passed by.
    assert shaded or scene_class is not ParallelLines


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


def _square_closing(gap: float) -> np.ndarray:
    """A unit square at the origin as CE lays it, its last point ``gap`` off
    its first in x.
    """
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [gap, 0, 0]], float)
    return np.concatenate(
        [np.linspace(a, b, 4) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    )


@pytest.mark.parametrize(
    ("gap", "decided_otherwise"),
    [(1e-16, False), (3e-5, True), (1e-3, False)],
    ids=["noise", "in-band", "open"],
)
def test_a_closing_gap_in_the_band_is_counted(gap, decided_otherwise):
    """CE leaves a square open whose end is 3e-5 from its start at the origin
    (its tolerance there is 1e-6); snapped to 2^-14 the two meet. The counter
    records it: the band is a gap within a grid step of CE's tolerance.
    """
    made = _square_closing(gap)
    canonical.point_equality_disagreements.clear()
    canonical._count_disagreements(made, canonical.snap(made, 2.0**-14))
    found = bool(canonical.point_equality_disagreements)
    canonical.point_equality_disagreements.clear()
    assert found == decided_otherwise


#: A flat view of the configured frame at 192x108.
_FLAT_VIEW = {
    "pixel_width": 192,
    "pixel_height": 108,
    "frame_width": 14.222222222222221,
    "frame_height": 8.0,
    "frame_center": (0.0, 0.0, 0.0),
    "mapping_centre": (0.0, 0.0, 0.0),
    "cairo_line_width_multiple": 0.01,
    "use_z_index": True,
    "background": (0.0, 0.0, 0.0, 1.0),
    "camera": "flat",
}


def _revealing(shape: Shape) -> Record:
    """A path record drawing the first half of ``shape``."""
    return Record(
        "path",
        {
            "shape": shape.key,
            "reveal": (0.0, 0.5),
            "fill": np.array([[1.0, 1.0, 1.0, 0.0]]),
            "stroke": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "stroke_width": 4.0,
            "joint": "auto",
            "cap": "auto",
            "space": "screen",
            "z_index": 0.0,
            "shade_in_3d": False,
        },
    )


def _sent_to_the_core(deltas: list[Delta]) -> list[Delta]:
    """``deltas`` through Canonical, each applied to a core stream and drawn."""
    import arachne_core

    snapper = Canonical()
    core = arachne_core.Stream()
    sent = []
    for delta in deltas:
        found = snapper(delta)
        core.apply(arachne_core.FrameDelta.from_bytes(found.to_core().to_bytes()))
        core.draw(192, 108, threads=1, overlay=False)
        sent.append(found)
    return sent


def _swapping() -> list[Delta]:
    """Two reveals of two squares, then the reveals swap the squares."""
    a = Shape("path", np.array(Square(1.5).shift(2 * RIGHT).points))
    b = Shape("path", np.array(Square(1.0).shift(UP).points))
    first = Delta(seq=0, view=dict(_FLAT_VIEW), upload=[a, b])
    first.set = [(0, _revealing(a)), (1, _revealing(b))]
    return [first, Delta(seq=1, set=[(0, _revealing(b)), (1, _revealing(a))])]


def test_two_records_swapping_their_snapped_shapes_keep_both_held():
    """Two reveals swap the shapes they draw: slot 0 goes from the points of
    one to the other's, slot 1 back. Each snapped shape is held at the end of
    the delta as at its start, so it is neither uploaded again nor released,
    and the core draws the frame.
    """
    opened, swapped = _sent_to_the_core(_swapping())
    assert len(opened.upload) == 4, "two stream shapes and two reveals"
    assert not swapped.upload, [shape.key.hex()[:8] for shape in swapped.upload]
    assert not swapped.release, [key.hex()[:8] for key in swapped.release]


def test_a_snapped_shape_nothing_holds_after_a_swap_is_released():
    """After the swap, slot 1 is removed: the reveal only it held is released
    in that delta, and nothing else is.
    """
    opened, _, dropped = _sent_to_the_core([*_swapping(), Delta(seq=2, remove=[1])])
    # Slot 1 last drew the reveal slot 0 first drew.
    first_reveal = dict(opened.set)[0].fields["shape"]
    assert dropped.release == [first_reveal]


def test_a_frame_a_millionth_wide_keeps_its_view_and_a_grid_to_match():
    """Zoomed to a frame 1e-6 wide at 192 pixels, a pixel spans about 5.2e-9
    (in [2^-28, 2^-27)): the input grid is 2^-36, the frame keeps its width
    and height to 24 significant bits, and its centre, a position, lands on
    that grid rather than on a fixed one.
    """
    width, height = 1e-6, 1e-6 * 108 / 192
    centre = (3.3e-7, -1.7e-7, 0.0)
    view = {
        **_FLAT_VIEW,
        "frame_width": width,
        "frame_height": height,
        "frame_center": centre,
        "mapping_centre": centre,
    }
    assert canonical.input_grid([view]) == 2.0**-36
    found = Canonical(planes=False)(Delta(seq=0, view=view)).view
    assert found is not None
    assert found["input_grid"] == 2.0**-36
    assert found["frame_width"] == pytest.approx(width, rel=2.0**-23)
    assert found["frame_height"] == pytest.approx(height, rel=2.0**-23)
    assert found["frame_center"] == canonical.snap(centre, 2.0**-36)
    assert found["frame_center"][0] == pytest.approx(centre[0], abs=2.0**-36)


@pytest.mark.parametrize(
    ("pixel_width", "grid"), [(1920, 2.0**-16), (854, 2.0**-14), (192, 2.0**-12)]
)
def test_the_configured_frame_keeps_its_grid(pixel_width, grid):
    """The configured frame, 14.2 units wide: 2^-16 at 1080p."""
    assert canonical.input_grid([{**_FLAT_VIEW, "pixel_width": pixel_width}]) == grid


class DeepTranslucentPair(ThreeDScene):
    """Two translucent shaded squares 2e-5 wide and 5e-6 apart in depth,
    under a camera zoomed until its frame spans 1e-4: their depth keys differ
    by less than 2^-16. The nearer is added first; CE draws it last.
    """

    def construct(self):
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=14.222222 / 1e-4)
        near = Square(2e-5, color=RED, fill_opacity=0.5, stroke_width=0)
        near.shift(OUT * 5e-6 + RIGHT * 3e-6).set_shade_in_3d(True)
        far = Square(2e-5, color=BLUE, fill_opacity=0.5, stroke_width=0)
        far.set_shade_in_3d(True)
        self.add(near, far)
        self.wait(1 / FPS)


def test_a_deep_zoom_sorts_by_depth_at_the_frame_s_grid(monkeypatch):
    """At a frame 1e-4 wide the depth keys are snapped at the frame's grid,
    not at 2^-16, where the two squares' keys tie: the core is handed them
    in CE's order, far first.
    """
    draw = CorePainter.draw
    handed: list[list[str]] = []

    def watching(self, delta):
        mobs = {slot: mob for mob, slot in self.stream._slots.values()}
        if delta.order is not None:
            handed.append([mobs[slot].color.to_hex() for _, slot, _ in delta.order])
        return draw(self, delta)

    monkeypatch.setattr(CorePainter, "draw", watching)
    render(DeepTranslucentPair, FPS, draw_path="core", **MEDIUM)
    blue, red = BLUE.to_hex(), RED.to_hex()
    assert handed
    assert all(order == [blue, red] for order in handed), handed
