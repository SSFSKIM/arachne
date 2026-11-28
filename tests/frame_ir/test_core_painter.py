"""3D frames drawn on the core, with Cairo's overlay drawn over them.

The core draws a 3D scene's frame without its overlay (mobjects fixed in
frame, images, background-image fills), and Cairo draws the overlay over the
core's pixels with CE's own camera code, in CE's order. Drawn onto the core's
frame as CE draws onto its own, the overlay rounds where CE's rounds: wherever
the core and CE agree beneath it, the overlaid frame is CE's, byte for byte,
translucent members included.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.painter import IrCairoPainter

from ..timeline.conftest import render
from . import core_scenes

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


#: The frame the corner label is measured at, CE's low quality.
LOW = {"pixel_width": 854, "pixel_height": 480}


def test_the_overlay_is_ce_s_wherever_the_core_and_ce_agree_beneath_it():
    """Cairo draws the overlay over the core's frame as CE draws it over its
    own, from the scene's own values: wherever the frames beneath agree, the
    overlaid pixels are CE's, byte for byte, its edges included (its glyphs'
    and its outlines').
    """
    _, ce = render(core_scenes.FixedLabelOverSquare, FPS, **MEDIUM)
    _, ce_bare = render(core_scenes.FixedLabelOverSquareBare, FPS, **MEDIUM)
    before = CorePainter.drawn_total
    _, core = render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    _, core_bare = render(
        core_scenes.FixedLabelOverSquareBare, FPS, draw_path="core", **MEDIUM
    )
    assert CorePainter.drawn_total > before, "drawn on the core"
    assert len(ce) == len(core) == len(ce_bare) == len(core_bare) > 0
    compared = 0
    for index, frames in enumerate(zip(ce, core, ce_bare, core_bare, strict=True)):
        a, b, a_bare, b_bare = frames
        agree = np.all(a_bare == b_bare, axis=-1)
        overlaid = np.any(a != a_bare, axis=-1)
        off = np.abs(b.astype(int) - a.astype(int)).max(axis=-1)
        assert not off[agree].any(), (
            f"frame {index}: {int((off[agree] > 0).sum())} pixels off, up to"
            f" {off[agree].max()} levels"
        )
        compared += int((agree & overlaid).sum())
    # The overlay's own pixels over agreeing content, translucent ones among
    # them, not a handful at its edge.
    assert compared > 1000, compared


@pytest.mark.parametrize(
    "scene_class",
    [core_scenes.CornerLabel, core_scenes.CornerLabelOwnCamera],
    ids=["recorded", "live"],
)
def test_a_label_fixed_in_frame_is_drawn_from_the_scene_s_values(scene_class):
    """The overlay is drawn from the values the scene made, not from the
    core's snapped ones: Cairo's scan conversion is not continuous in its
    input (24.8 fixed point, whole sub-rows), so a snap of at most 1/512 of
    a pixel moved a shallow glyph edge's pixel by up to 14 levels. A label
    over the background alone is CE's frame, byte for byte, drawn from the
    record and drawn as the scene runs.
    """
    _, ce = render(scene_class, FPS, **LOW)
    before = CorePainter.drawn_total
    _, core = render(scene_class, FPS, draw_path="core", **LOW)
    assert CorePainter.drawn_total > before, "drawn on the core"
    assert len(ce) == len(core) > 0
    for index, (a, b) in enumerate(zip(ce, core, strict=True)):
        assert np.any(a != a[0, 0]), "the label is drawn"
        off = np.abs(b.astype(int) - a.astype(int)).max(axis=-1)
        assert not off.any(), (
            f"frame {index}: {int((off > 0).sum())} pixels off, up to {off.max()} levels"
        )


def test_the_overlay_is_drawn_over_every_frame_that_has_one(monkeypatch):
    painted: list[CorePainter] = []
    held = CorePainter.__init__

    def kept(self, **kwargs):
        held(self, **kwargs)
        painted.append(self)

    monkeypatch.setattr(CorePainter, "__init__", kept)
    render(core_scenes.FixedLabelOverSquare, FPS, draw_path="core", **MEDIUM)
    assert sum(p.frames for p in painted) > 0
    assert sum(p.overlaid for p in painted) == sum(p.frames for p in painted)


def test_a_painter_holds_none_of_the_frames_it_drew_its_overlay_over():
    """Cairo draws each frame's overlay through a context on that frame, and a
    context holds its frame: kept for the frames after, every frame of a render
    would stay in memory to its end (about 8 MB each at 1080p).
    """
    import gc
    import weakref
    from fractions import Fraction

    from arachne import Rectangle, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        square = Square(fill_opacity=1.0)
        bar = Rectangle(width=3, height=0.4, fill_opacity=0.5)
        camera.add_fixed_in_frame_mobjects(bar)
        painter = CorePainter()
        drawn = []
        for index in range(6):
            square.shift([0.1, 0.0, 0.0])
            frame = painter.paint(
                camera, [("moving", [square, bar])], t=Fraction(index, FPS)
            )
            drawn.append(weakref.ref(frame))
            del frame
        assert painter.overlaid == len(drawn)
        gc.collect()
        held = sum(ref() is not None for ref in drawn)
    # The last frame stays the drawing camera's picture until the next.
    assert held <= 1, f"{held} of {len(drawn)} frames held"


def _displayed():
    """A MultiCamera showing a display of a quarter of the frame, whose
    camera's centre is off any grid, of circles and lines: the camera, the
    display and the mobjects to capture.
    """
    from arachne import (
        RED,
        Circle,
        ImageMobjectFromCamera,
        Line,
        MovingCamera,
        MultiCamera,
    )

    camera = MultiCamera()
    display = ImageMobjectFromCamera(MovingCamera())
    display.stretch_to_fit_width(camera.frame_width / 2)
    display.stretch_to_fit_height(camera.frame_height / 2)
    display.move_to([camera.frame_width / 4, camera.frame_height / 4, 0.0])
    display.camera.frame.scale(0.37).move_to([0.123456789, 0.0987654321, 0.0])
    camera.add_image_mobject_from_camera(display)
    shapes = [
        Circle(radius=0.05 + 0.11 * k, color=RED).shift([0.03 * k, 0.0, 0.0])
        for k in range(12)
    ]
    shapes += [
        Line([-2.0, -1.0 + 0.17 * k, 0.0], [2.0, 0.3 * k, 0.0]) for k in range(9)
    ]
    return camera, display, [*shapes, display]


def _labelled():
    """A tilted ThreeDCamera, a square in space off any grid, which the core
    rasterizes, and a label fixed in frame, which Cairo paints: the camera,
    the label and the mobjects to capture.
    """
    from arachne import DEGREES, Square, Text
    from arachne.camera.three_d_camera import ThreeDCamera

    camera = ThreeDCamera()
    camera.set_phi(60 * DEGREES)
    camera.set_theta(-45 * DEGREES)
    square = Square(side_length=2.3456789, fill_opacity=0.5)
    square.rotate(0.123, axis=[1.0, 1.0, 0.0])
    label = Text("Overlay in a corner, W = 2", font_size=30)
    label.to_corner([1.0, 1.0, 0.0])
    camera.add_fixed_in_frame_mobjects(label)
    return camera, label, [square, label]


def test_a_display_left_after_a_frame_holds_the_sub_frame_ce_s_capture_leaves():
    """Once the frame is drawn, a display's camera is left holding the
    picture of its sub-frame as CE's capture leaves it: drawn, as the overlay
    is, from the display's record as the scene made it, its view and the
    records it draws unsnapped (a centre off the grid, here).
    """
    from fractions import Fraction

    from arachne import tempconfig

    with tempconfig(MEDIUM):
        camera, display, mobjects = _displayed()
        camera.capture_mobjects(mobjects)
        ce = display.camera.pixel_array
        camera, display, mobjects = _displayed()
        painter = CorePainter()
        painter.paint(camera, [("moving", mobjects)], t=Fraction(0))
        painter.settle()
        held = display.camera.pixel_array
    assert np.any(ce != display.camera.background), "the sub-frame draws"
    differ = np.abs(held.astype(int) - ce.astype(int)).max(-1)
    assert not differ.any(), int((differ > 0).sum())


def test_the_core_is_handed_snapped_values_and_cairo_paints_the_scene_s():
    """On one frame, every point the core rasterizes is on the frame's input
    grid, as the core read it back (``state``), and every point of the
    overlay Cairo paints is the scene's own, bit for bit (``made``).
    """
    from fractions import Fraction

    from arachne import tempconfig

    with tempconfig(LOW):
        camera, label, mobjects = _labelled()
        painter = CorePainter()
        painter.paint(camera, [("moving", mobjects)], t=Fraction(0))
    state, made = painter.state, painter.made
    grid = state.view["input_grid"]
    overlay = painter.core.overlay_entries()
    painted = {slot for _, slot, _ in overlay}
    rasterized = {slot for _, slot, _ in state.order} - painted
    assert rasterized
    for slot in rasterized:
        points = state.shapes[state.records[slot].fields["shape"]].data
        np.testing.assert_array_equal(points, np.round(points / grid) * grid)
    glyphs = label.family_members_with_points()
    assert len(overlay) == len(glyphs)
    for (_, slot, _), glyph in zip(overlay, glyphs, strict=True):
        points = made.shapes[made.records[slot].fields["shape"]].data
        np.testing.assert_array_equal(points, glyph.points)
        assert not np.array_equal(points, np.round(points / grid) * grid)


def test_the_overlay_is_reported_and_settled_from_the_state_it_is_drawn_from(
    monkeypatch,
):
    """What Cairo draws of the overlay, the contributors and outlines it
    reports of it, and the display pictures a frame leaves behind all read
    one state, the stream as the scene made it (``made``); the snapped state
    the core read (``state``) is read only for the leaves the core draws.
    """
    from fractions import Fraction

    from arachne import tempconfig
    from arachne.frame_ir.cairo_painter import CairoDrawer

    read: list[tuple[str, object, set[int]]] = []
    # Each spied method and the place of its entries among its arguments.
    for name, at in (("draw_over", 1), ("contributors", 0), ("outlines", 0)):
        kept = getattr(CairoDrawer, name)

        def spy(self, state, *args, _kept=kept, _name=name, _at=at, **kwargs):
            read.append((_name, state, {slot for _, slot, _ in args[_at]}))
            return _kept(self, state, *args, **kwargs)

        monkeypatch.setattr(CairoDrawer, name, spy)
    kept_capture = CairoDrawer.capture

    def capturing(self, state, view, order, background):
        read.append(("capture", state, {slot for _, slot, _ in order}))
        return kept_capture(self, state, view, order, background)

    monkeypatch.setattr(CairoDrawer, "capture", capturing)
    with tempconfig(LOW):
        camera, _, mobjects = _labelled()
        labelled = CorePainter()
        labelled.paint(camera, [("moving", mobjects)], t=Fraction(0))
        labelled.contributors((600, 20, 700, 60))
        labelled.outlines()
    with tempconfig(MEDIUM):
        camera, _, mobjects = _displayed()
        displayed = CorePainter()
        displayed.paint(camera, [("moving", mobjects)], t=Fraction(0))
        displayed.settle()
    assert {name for name, _, _ in read} == {
        "draw_over",
        "contributors",
        "outlines",
        "capture",
    }
    for name, state, slots in read:
        (painter,) = [p for p in (labelled, displayed) if state in (p.made, p.state)]
        if state is painter.made:
            continue
        painted = {slot for _, slot, _ in painter.core.overlay_entries()}
        assert name == "outlines", name
        assert not slots & painted, (name, slots & painted)
    assert any(name == "capture" and state is displayed.made for name, state, _ in read)


def test_a_live_frame_tests_depth_across_its_static_and_moving_captures():
    """Drawn as the scene runs, a frame is two captures, its static list and
    then its moving one, as CE draws it. The core describes the two as one
    frame and tests depth across both, as it does a frame drawn from the
    record: the square hides the stroke under it on either route, and the
    frames are the same, byte for byte.
    """
    from arachne.frame_ir.live import live_painter

    _, cairo = render(core_scenes.StrokeUnderSquare, FPS, **MEDIUM)
    _, recorded = render(core_scenes.StrokeUnderSquare, FPS, draw_path="core", **MEDIUM)
    scene, live = render(
        core_scenes.LiveStrokeUnderSquare, FPS, draw_path="core", **MEDIUM
    )
    found = live_painter(scene.renderer.camera)  # type: ignore[attr-defined]
    assert found is not None
    assert found.captures > 0, "drawn as the scene runs"
    assert len(live) == len(recorded) == len(cairo) > 0
    # Depth is at work: CE draws the stroke over the square.
    assert np.any(recorded[0] != cairo[0])
    for index, (a, b) in enumerate(zip(live, recorded, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


@pytest.mark.parametrize(
    "scene_class",
    [
        core_scenes.BlackSquareOverStroke,
        core_scenes.TranslucentSquareOverStroke,
        core_scenes.OutlineOverStroke,
        core_scenes.SquareOffFrameOverStroke,
    ],
    ids=lambda c: c.__name__,
)
def test_a_live_frame_tests_depth_across_its_captures_whatever_they_drew(
    scene_class,
):
    """Whether a live frame's captures are depth tested together is decided
    by what they are, not by what their pixels show: an occluder in the
    background's colour, a translucent one, one of stroke alone and one off
    the frame give the same frames, byte for byte, drawn as the scene runs
    and from the record.
    """
    _, recorded = render(scene_class, FPS, draw_path="core", **MEDIUM)
    live_class = type(
        f"Live{scene_class.__name__}", (scene_class,), {"line": core_scenes.OwnLine}
    )
    _, live = render(live_class, FPS, draw_path="core", **MEDIUM)
    assert len(live) == len(recorded) > 0
    for index, (a, b) in enumerate(zip(live, recorded, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def _live_line(*, cleared_square: bool) -> np.ndarray:
    """A live frame on the core of a stroke at z = -1, seen from above; with
    ``cleared_square``, captured after an opaque black square at z = 0 over
    it whose picture the scene's code then clears, setting a copy of the
    camera's background as its array.
    """
    from fractions import Fraction

    from arachne import BLACK, DEGREES, YELLOW, Line, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera
    from arachne.frame_ir.live import live

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        camera.set_phi(20 * DEGREES)
        camera.set_theta(-60 * DEGREES)
        square = Square(side_length=2, color=BLACK, fill_opacity=1.0)
        line = Line([-3.0, 0.0, -1.0], [3.0, 0.0, -1.0], color=YELLOW, stroke_width=8)
        with live(camera, Fraction(0), "core", play=0):
            camera.reset()
            if cleared_square:
                camera.capture_mobjects([square])
                camera.set_pixel_array(camera.background.copy())
            camera.capture_mobjects([line])
        return np.array(camera.pixel_array)


def test_a_picture_the_scene_s_code_clears_hides_nothing_drawn_after_it():
    """A live frame continues the description of the picture its camera
    holds only where the renderer put that picture there: a square black on
    black, its picture cleared by the scene's code with pixels just like it,
    hides nothing of the stroke captured after.
    """
    cleared = _live_line(cleared_square=True)
    alone = _live_line(cleared_square=False)
    assert np.any(alone[..., :3] > 0), "the stroke is drawn"
    differ = np.abs(cleared.astype(int) - alone.astype(int)).max(-1)
    assert not differ.any(), int((differ > 0).sum())


def test_a_resized_display_s_points_are_as_wide_drawn_as_the_scene_runs_as_from_the_record():
    """A MultiCamera's display half the frame's size shows a point cloud
    whose blocks CE widens by the ratio of the configured size to the
    display camera's: drawn as the scene runs, the display's captures are
    Cairo's, as from the record, and the frames are the same, byte for byte.
    """
    from arachne.frame_ir.live import live_painter

    _, recorded = render(
        core_scenes.ResizedDisplayPoints, FPS, draw_path="core", **MEDIUM
    )
    scene, live = render(
        core_scenes.LiveResizedDisplayPoints, FPS, draw_path="core", **MEDIUM
    )
    found = live_painter(scene.renderer.camera)  # type: ignore[attr-defined]
    assert found is not None
    assert found.captures > 0, "drawn as the scene runs"
    assert len(live) == len(recorded) > 0
    for index, (a, b) in enumerate(zip(live, recorded, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def test_auto_draws_a_3d_scene_on_the_core():
    before = CorePainter.drawn_total
    render(core_scenes.FixedLabelOverSquare, FPS, draw_path="auto", **MEDIUM)
    assert CorePainter.drawn_total > before


def test_auto_draws_a_2d_scene_as_cairo_does_without_xf():
    _, direct = render(core_scenes.FlatSquares, FPS, **MEDIUM)
    before = (CorePainter.drawn_total, IrCairoPainter.drawn_total)
    _, auto = render(core_scenes.FlatSquares, FPS, draw_path="auto", **MEDIUM)
    assert (CorePainter.drawn_total, IrCairoPainter.drawn_total) == before
    assert len(direct) == len(auto) > 0
    for index, (a, b) in enumerate(zip(direct, auto, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def _inside(frame: np.ndarray, margin: int = 2) -> np.ndarray:
    """Pixels at least ``margin`` px from any edge in ``frame``: where no
    neighbour differs from a pixel by more than a gradient's step, eroded
    by ``margin``.
    """
    rgb = frame[..., :3].astype(int)
    smooth = np.ones(rgb.shape[:2], bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        step = np.abs(rgb - np.roll(rgb, (dy, dx), axis=(0, 1))).max(axis=-1) <= 3
        smooth &= step & np.roll(step, (-dy, -dx), axis=(0, 1))
    inside = smooth.copy()
    for _ in range(margin):
        inside &= np.roll(inside, 1, 0) & np.roll(inside, -1, 0)
        inside &= np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    return inside & np.any(rgb > 0, axis=-1)


def test_a_lit_face_is_ce_s_colours_inside_it():
    """Each face of a shaded prism and surface is lit as CE lights it (two
    stops, a linear gradient), coloured as Cairo colours it: inside every
    face, away from its edges, the core's pixels are CE's, up to one level
    where the geometry the core is handed, snapped (``frame_ir.canonical``),
    rounds a colour across a half level CE's does not, at under one in a
    hundred.
    """
    size = {"pixel_width": 480, "pixel_height": 270}
    _, ce = render(core_scenes.ShadedSolids, FPS, **size)
    _, core = render(core_scenes.ShadedSolids, FPS, draw_path="core", **size)
    assert len(ce) == len(core) > 0
    compared = rounded = 0
    for index, (a, b) in enumerate(zip(ce, core, strict=True)):
        inside = _inside(a)
        off = np.abs(b[inside].astype(int) - a[inside].astype(int))
        assert off.max(initial=0) <= 1, f"frame {index}: {off.max()} levels off"
        compared += int(inside.sum())
        rounded += int(np.any(off, axis=-1).sum())
    # Hundreds of pixels inside the faces of every frame, not a handful.
    assert compared > 500 * len(ce), compared
    assert rounded <= compared // 100, (rounded, compared)
