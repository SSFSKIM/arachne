"""Each painter's contributor report: per pixel, the drawn leaves of a frame
in the order they were composited (:mod:`arachne.frame_ir.contributors`).

Frames here are 192 x 108 pixels over CE's 8-unit frame height: 13.5 pixels a
unit, the origin at pixel (96, 54).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne.frame_ir.contributors import (
    Leaf,
    LiveFrame,
    PixelContributors,
    from_core,
    over,
)
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.painter import IrCairoPainter

pytest.importorskip("arachne_core")

MEDIUM = {"pixel_width": 192, "pixel_height": 108}
CENTRE = (96, 54)


def _box(x: int, y: int) -> tuple[int, int, int, int]:
    return (x, y, x + 1, y + 1)


def _leaf(obj: int, alpha: int, shown: bool = True) -> Leaf:
    return Leaf(
        object=obj,
        part=0,
        id=f"m{obj}",
        drawn=(0, 0, alpha, alpha),
        shown=shown,
        painter="cairo",
        covers=alpha == 255,
    )


def test_cairo_reports_leaves_in_paint_order_and_hides_what_an_opaque_one_covers():
    from arachne import BLUE, RED, Camera, Square, tempconfig

    with tempconfig(MEDIUM):
        camera = Camera()
        under = Square(side_length=2, fill_color=RED, fill_opacity=1.0, stroke_width=0)
        top = Square(side_length=1, fill_color=BLUE, fill_opacity=1.0, stroke_width=0)
        painter = IrCairoPainter()
        frame = painter.paint(camera, [("moving", [under, top])], t=Fraction(0))
        found = painter.contributors(_box(*CENTRE))
    (pixel,) = found.values()
    assert [leaf.painter for leaf in pixel.leaves] == ["cairo", "cairo"]
    assert [leaf.shown for leaf in pixel.leaves] == [False, True]
    assert pixel.crossing is False
    assert pixel.orders == (((0, 255), (1, 255)),)
    # Drawn alone, the opaque top leaves the frame's pixel.
    assert pixel.leaves[1].drawn == tuple(int(v) for v in frame[CENTRE[1], CENTRE[0]])
    # Every leaf names its mobject's stable id, the same one each frame.
    assert len({leaf.id for leaf in pixel.leaves}) == 2
    assert None not in {leaf.id for leaf in pixel.leaves}


def test_the_two_painters_order_one_frame_s_leaves_by_paint_order_and_by_depth():
    """Seen straight down, a square nearer the camera and one further away,
    the far one added last: CE paints them in the order added, the far one on
    top; the core tests depth, the near one on top. The leaves are the same
    on both painters, matched by their stable ids.
    """
    from arachne import BLUE, IN, OUT, RED, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        near = Square(fill_color=RED, fill_opacity=1.0, stroke_width=0).shift(OUT)
        far = Square(fill_color=BLUE, fill_opacity=1.0, stroke_width=0).shift(IN)
        groups = [("moving", [near, far])]
        cairo = IrCairoPainter()
        cairo.paint(camera, groups, t=Fraction(0))
        core = CorePainter()
        core.paint(camera, groups, t=Fraction(0))
        painted = cairo.contributors(_box(*CENTRE))[CENTRE]
        composited = core.contributors(_box(*CENTRE))[CENTRE]
    assert painted.keys == tuple(reversed(composited.keys))
    assert [leaf.shown for leaf in painted.leaves] == [False, True]
    assert [leaf.shown for leaf in composited.leaves] == [False, True]
    assert {leaf.painter for leaf in composited.leaves} == {"core"}
    # The core knows each leaf's coverage and colour apart, and that depth ordered them.
    assert [leaf.coverage for leaf in composited.leaves] == [255, 255]
    assert composited.leaves[1].colour == composited.leaves[1].drawn
    assert composited.decided == ((0, 1, "depth"),)
    assert painted.decided == ()


def test_on_the_core_path_the_overlay_s_leaves_lie_over_the_core_s():
    from arachne import BLUE, WHITE, Rectangle, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        square = Square(side_length=2, fill_color=BLUE, fill_opacity=1.0)
        bar = Rectangle(
            width=3, height=0.4, fill_color=WHITE, fill_opacity=1.0, stroke_width=0
        )
        camera.add_fixed_in_frame_mobjects(bar)
        painter = CorePainter()
        frame = painter.paint(camera, [("moving", [square, bar])], t=Fraction(0))
        under_bar = painter.contributors(_box(*CENTRE))[CENTRE]
        beside = painter.contributors(_box(96, 45))[(96, 45)]
    assert [leaf.painter for leaf in under_bar.leaves] == ["core", "cairo"]
    assert [leaf.shown for leaf in under_bar.leaves] == [False, True]
    assert under_bar.leaves[1].drawn == tuple(
        int(v) for v in frame[CENTRE[1], CENTRE[0]]
    )
    assert under_bar.orders == (((0, 255), (1, 255)),)
    # The overlay is painted over the core's frame.
    assert under_bar.decided == ((0, 1, "paint"),)
    assert [leaf.painter for leaf in beside.leaves] == ["core"]


def test_on_the_core_path_the_overlay_s_leaves_are_reported_as_drawn_from_the_scene():
    """Each leaf of the overlay is drawn again alone, and outlined, as the
    frame drew it, from the values the scene made: at every pixel of a label
    fixed in frame, what it draws is what Cairo draws from the scene's own
    values (``ir_cairo``), its glyphs' edges included, and its outlines are
    the paths Cairo is handed there.
    """
    from arachne import DEGREES, Text, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig({"pixel_width": 854, "pixel_height": 480}):
        camera = ThreeDCamera()
        camera.set_phi(60 * DEGREES)
        camera.set_theta(-45 * DEGREES)
        label = Text("Overlay in a corner, W = 2", font_size=30)
        label.to_corner([1.0, 1.0, 0.0])
        camera.add_fixed_in_frame_mobjects(label)
        (left, top), (right, bottom) = (
            camera.points_to_pixel_coords(label, np.array([corner]))[0]
            for corner in (
                label.get_corner([-1.0, 1.0, 0.0]),
                label.get_corner([1.0, -1.0, 0.0]),
            )
        )
        box = (int(left) - 2, int(top) - 2, int(right) + 3, int(bottom) + 3)
        core, cairo = CorePainter(), IrCairoPainter()
        for painter in (core, cairo):
            painter.paint(camera, [("moving", [label])], t=Fraction(0))
        on_core, on_cairo = core.contributors(box), cairo.contributors(box)
        outlined, outlined_on_cairo = core.outlines(), cairo.outlines()
    assert on_core.keys() == on_cairo.keys()
    assert len(on_core) > 1000
    for pixel, found in on_core.items():
        assert [leaf.drawn for leaf in found.leaves if leaf.painter == "cairo"] == [
            leaf.drawn for leaf in on_cairo[pixel].leaves
        ], pixel
    assert outlined.keys() == outlined_on_cairo.keys()
    for key, ops in outlined.items():
        theirs = outlined_on_cairo[key]
        assert [len(op.subpaths) for op in ops] == [len(op.subpaths) for op in theirs]
        for op, their in zip(ops, theirs, strict=True):
            for mine, other in zip(op.subpaths, their.subpaths, strict=True):
                assert (mine.start, mine.closed) == (other.start, other.closed), key
                np.testing.assert_array_equal(mine.segments, other.segments, str(key))


def test_from_core_names_stable_ids_and_what_each_leaf_draws_alone():
    found = {
        (3, 4): (
            [
                (7, 0, 255, (0, 0, 200, 255), (0, 0, 200, 255), False, True),
                (8, 2, 128, (100, 0, 0, 128), (50, 0, 0, 64), True, False),
            ],
            [[(0, 255), (1, 255)], [(0, 255)]],
            False,
            [(0, 1, "depth")],
        )
    }
    (pixel,) = from_core(found, {7: ("Square#1", "scene.py:3")}).values()
    assert pixel.decided == ((0, 1, "depth"),)
    assert pixel.keys == (("Square#1", 0), (8, 2))
    assert pixel.leaves[1].drawn == (50, 0, 0, 64)
    assert pixel.leaves[0].drawn == (0, 0, 200, 255)
    assert pixel.orders == (((0, 255), (1, 255)), ((0, 255),))
    assert pixel.shown == (pixel.leaves[1],)
    assert [leaf.covers for leaf in pixel.leaves] == [True, False]


def test_over_hides_what_an_opaque_top_leaf_covers_and_continues_each_order():
    under = {
        (0, 0): PixelContributors(
            leaves=(_leaf(1, 255), _leaf(2, 128)),
            orders=(((0, 255), (1, 255)), ((0, 255),)),
            crossing=True,
        ),
        (1, 0): PixelContributors(leaves=(_leaf(1, 255),), orders=(((0, 255),),)),
    }
    top = {
        (0, 0): PixelContributors(leaves=(_leaf(9, 255),), orders=(((0, 255),),)),
        (1, 0): PixelContributors(leaves=(_leaf(9, 100),), orders=(((0, 100),),)),
        (2, 0): PixelContributors(leaves=(_leaf(9, 100),), orders=(((0, 100),),)),
    }
    merged = over(under, top)
    covered = merged[(0, 0)]
    assert [leaf.object for leaf in covered.leaves] == [1, 2, 9]
    assert [leaf.shown for leaf in covered.leaves] == [False, False, True]
    assert covered.orders == (
        ((0, 255), (1, 255), (2, 255)),
        ((0, 255), (2, 255)),
    )
    assert covered.crossing is True
    assert covered.decided == ((0, 2, "paint"), (1, 2, "paint"))
    # A translucent top hides nothing.
    assert [leaf.shown for leaf in merged[(1, 0)].leaves] == [True, True]
    assert merged[(2, 0)] == top[(2, 0)]


def test_a_box_off_the_frame_reports_nothing():
    from arachne import Camera, Square, tempconfig

    with tempconfig(MEDIUM):
        painter = IrCairoPainter()
        painter.paint(Camera(), [("moving", [Square(fill_opacity=1.0)])])
        assert painter.contributors((500, 500, 600, 600)) == {}
        assert painter.contributors((0, 0, 4, 4)) == {}
        found = painter.contributors((-10, -10, 1000, 1000))
    xs = np.array([x for x, _ in found])
    assert found
    assert xs.min() >= 0
    assert xs.max() < 192


def _square_under(top, *, side: float = 2.0):
    """A frame of an opaque red square under ``top``, on Cairo from XF, and
    its report at the centre: the frame and the pixel's contributors.
    """
    from arachne import RED, Camera, Square, tempconfig

    with tempconfig(MEDIUM):
        under = Square(
            side_length=side, fill_color=RED, fill_opacity=1.0, stroke_width=0
        )
        painter = IrCairoPainter()
        frame = painter.paint(Camera(), [("moving", [under, *top()])], t=Fraction(0))
        return frame, painter.contributors(_box(*CENTRE))[CENTRE]


def test_a_translucent_image_s_leaf_is_premultiplied_as_a_path_s_is():
    from arachne import ImageMobject

    def picture():
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[...] = (200, 100, 50, 128)
        return [ImageMobject(pixels).scale_to_fit_width(3.0)]

    frame, pixel = _square_under(picture)
    image = pixel.leaves[-1]
    r, g, b, a = image.drawn
    assert a == 128
    assert max(r, g, b) <= a
    assert (r, g, b) == pytest.approx((100, 50, 25), abs=1)
    # Neither leaf hides the other: the image is translucent.
    assert [leaf.shown for leaf in pixel.leaves] == [True, True]
    assert image.covers is False


def test_a_point_cloud_s_block_covers_what_it_is_written_over_whatever_its_alpha():
    from arachne.mobject.types.point_cloud_mobject import PMobject

    def points(alpha: float, rgb=(0.0, 0.0, 1.0)):
        def made():
            cloud = PMobject(stroke_width=12)
            cloud.add_points([[0.0, 0.0, 0.0]], rgbas=np.array([[*rgb, alpha]]))
            return [cloud]

        return made

    frame, pixel = _square_under(points(0.5))
    assert [leaf.shown for leaf in pixel.leaves] == [False, True]
    assert pixel.leaves[1].covers is True
    # The camera wrote the block's colour over the square: the frame is the block's.
    assert (
        tuple(int(v) for v in frame[CENTRE[1], CENTRE[0]])[3]
        == pixel.leaves[1].drawn[3]
    )
    # A transparent black block clears the pixel: reported, and hiding the square.
    frame, pixel = _square_under(points(0.0, (0.0, 0.0, 0.0)))
    assert len(pixel.leaves) == 2
    assert pixel.leaves[1].drawn == (0, 0, 0, 0)
    assert [leaf.shown for leaf in pixel.leaves] == [False, True]


def test_paths_filled_through_one_image_hide_each_other_as_one_mask_does():
    from PIL import Image

    from arachne import Square

    picture = Image.new("RGBA", (192, 108), (40, 200, 120, 128))

    def filled():
        first = Square(side_length=2, fill_opacity=1.0, stroke_width=0)
        second = Square(side_length=2, fill_opacity=1.0, stroke_width=0).shift(
            [0.5, 0, 0]
        )
        return [
            first.color_using_background_image(picture),
            second.color_using_background_image(picture),
        ]

    frame, pixel = _square_under(filled, side=4.0)
    # The red square, then the two image-filled squares: the image's alpha is 128, so the
    # red square shows through, but the second fill's opaque mask hides the first.
    assert [leaf.shown for leaf in pixel.leaves] == [True, False, True]
    assert [leaf.covers for leaf in pixel.leaves] == [True, False, False]


def test_a_box_past_the_top_left_corner_is_clipped_on_cairo_as_on_the_core():
    from arachne import Camera, Square, tempconfig

    with tempconfig(MEDIUM):
        corner = Square(side_length=2, fill_opacity=1.0).move_to([-7.1, 4.0, 0.0])
        for painter in (IrCairoPainter(), CorePainter()):
            painter.paint(Camera(), [("moving", [corner])], t=Fraction(0))
            found = painter.contributors((-2, -2, 10, 10))
            assert (0, 0) in found, type(painter).__name__
            assert all(0 <= x < 10 and 0 <= y < 10 for x, y in found)


def test_contributors_change_nothing_the_painter_draws_next():
    """A frame with a display among other leaves, drawn, reported on, and
    drawn again: the second draw is the first, byte for byte, on both
    painters.
    """
    from arachne import (
        RED,
        Circle,
        ImageMobjectFromCamera,
        MovingCamera,
        MultiCamera,
        Square,
        tempconfig,
    )

    with tempconfig(MEDIUM):
        for kind in (IrCairoPainter, CorePainter):
            camera = MultiCamera()
            display = ImageMobjectFromCamera(MovingCamera())
            display.stretch_to_fit_width(camera.frame_width / 2)
            display.stretch_to_fit_height(camera.frame_height / 2)
            camera.add_image_mobject_from_camera(display)
            shapes = [Circle(radius=1.0, color=RED), Square(fill_opacity=0.5)]
            groups = [("moving", [*shapes, display])]
            painter = kind()
            first = np.array(painter.paint(camera, groups, t=Fraction(0)))
            found = painter.contributors((0, 0, 192, 108))
            assert found
            again = painter.paint(camera, groups, t=Fraction(0))
            assert np.array_equal(first, again), kind.__name__


def test_a_live_painter_refuses_to_report_contributors():
    """Drawn as the scene runs, a capture is drawn over the captures before
    it, flattened into its background: a square drawn static, then a
    translucent one moving over it, would be reported without the first.
    The live painter refuses; a live frame is attributed from its state
    extracted whole.
    """
    from arachne.frame_ir.live import live_painter

    from ..timeline.conftest import render
    from . import core_scenes

    scene, _ = render(
        core_scenes.LiveStrokeUnderSquare, 15, draw_path="ir_cairo", **MEDIUM
    )
    camera = scene.renderer.camera  # type: ignore[attr-defined]
    found = live_painter(camera)
    assert found is not None
    assert found.captures > 0
    painter = found.painter(camera)
    with pytest.raises(LiveFrame):
        painter.contributors(_box(*CENTRE))


def test_a_box_s_report_is_the_same_cut_from_a_larger_box():
    """Cairo draws each leaf clipped to the box asked about: inside it,
    every pixel is what the leaf draws unclipped.
    """
    from arachne import BLUE, RED, Camera, Circle, Square, tempconfig

    with tempconfig(MEDIUM):
        painter = IrCairoPainter()
        under = Square(side_length=2, fill_color=RED, fill_opacity=0.7)
        ring = Circle(radius=1.2, color=BLUE, stroke_width=6)
        painter.paint(Camera(), [("moving", [under, ring])], t=Fraction(0))
        small = painter.contributors((70, 30, 90, 50))
        large = painter.contributors((0, 0, 192, 108))
    assert small
    assert small == {at: found for at, found in large.items() if at in small}
    assert set(small) == {(x, y) for x, y in large if 70 <= x < 90 and 30 <= y < 50}


def test_outlines_are_what_cairo_is_handed_in_pixels():
    from arachne import Camera, Square, tempconfig

    with tempconfig(MEDIUM):
        painter = IrCairoPainter()
        square = Square(side_length=2, fill_color="#FF0000", fill_opacity=1.0)
        square.set_stroke("#0000FF", width=4, opacity=0.5)
        painter.paint(Camera(), [("moving", [square])], t=Fraction(0))
        (ops,) = painter.outlines().values()
        frame_height = painter.state.view["frame_height"]
    fill, stroke = (op for op in ops if op.op in ("fill", "stroke"))
    assert fill.op == "fill"
    assert stroke.op == "stroke"
    # Colours in the frame's channel order, not premultiplied.
    assert fill.colours == (pytest.approx((1.0, 0.0, 0.0, 1.0)),)
    assert stroke.colours == (pytest.approx((0.0, 0.0, 1.0, 0.5)),)
    # A square of side 2 about the origin, in pixels about (96, 54).
    per_unit = 108 / frame_height
    (subpath,) = fill.subpaths
    corners = np.vstack([subpath.start, subpath.segments[:, 2]])
    assert corners[:, 0].min() == pytest.approx(96 - per_unit, abs=1e-6)
    assert corners[:, 1].max() == pytest.approx(54 + per_unit, abs=1e-6)
    assert subpath.closed is True
    # CE's stroke width 4 is 0.04 frame units.
    assert stroke.width == pytest.approx(0.04 * per_unit, rel=1e-6)
    assert stroke.join


def test_a_live_painter_refuses_outlines_too():
    from arachne.frame_ir.live import live_painter

    from ..timeline.conftest import render
    from . import core_scenes

    scene, _ = render(
        core_scenes.LiveStrokeUnderSquare, 15, draw_path="ir_cairo", **MEDIUM
    )
    camera = scene.renderer.camera  # type: ignore[attr-defined]
    found = live_painter(camera)
    assert found is not None
    with pytest.raises(LiveFrame):
        found.painter(camera).outlines()


def test_a_near_opaque_leaf_over_another_hides_nothing():
    """Source over hides what lies beneath only where the top is opaque: a
    leaf drawn (0, 0, 250, 250) over an opaque one leaves (5, 0, 250, 255),
    so the leaf beneath contributes, and is shown.
    """
    from arachne import Square

    def top():
        return [
            Square(
                side_length=1,
                fill_color="#0000FF",
                fill_opacity=250 / 255,
                stroke_width=0,
            )
        ]

    frame, pixel = _square_under_coloured("#FF0000")(top)
    assert tuple(int(v) for v in frame[CENTRE[1], CENTRE[0]]) == (5, 0, 250, 255)
    assert pixel.leaves[1].drawn == (0, 0, 250, 250)
    assert pixel.leaves[1].covers is False
    assert [leaf.shown for leaf in pixel.leaves] == [True, True]


def test_a_core_leaf_under_a_near_opaque_overlay_stays_shown():
    from arachne import BLUE, Rectangle, Square, tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        square = Square(side_length=2, fill_color=BLUE, fill_opacity=1.0)
        bar = Rectangle(
            width=3,
            height=0.4,
            fill_color="#FF0000",
            fill_opacity=250 / 255,
            stroke_width=0,
        )
        camera.add_fixed_in_frame_mobjects(bar)
        painter = CorePainter()
        painter.paint(camera, [("moving", [square, bar])], t=Fraction(0))
        under_bar = painter.contributors(_box(*CENTRE))[CENTRE]
    assert [leaf.painter for leaf in under_bar.leaves] == ["core", "cairo"]
    assert under_bar.leaves[1].drawn[3] == 250
    assert [leaf.shown for leaf in under_bar.leaves] == [True, True]


def test_faint_paths_filled_through_one_image_each_touch_the_pixel():
    """128 paths of opacity 1/255 filled through one image of alpha 128: each
    one's mask is 1, which the image multiplies to nothing alone, but the
    camera draws the batch as one mask, 128, and paints alpha 64. Each leaf
    touches the pixel, and draws what the batch draws.
    """
    from PIL import Image

    from arachne import Square

    picture = Image.new("RGBA", (192, 108), (40, 200, 120, 128))

    def faint():
        return [
            Square(side_length=2, fill_opacity=1 / 255, stroke_width=0)
            .shift([0.001 * k, 0, 0])
            .color_using_background_image(picture)
            for k in range(128)
        ]

    frame, pixel = _square_under(faint, side=4.0)
    assert len(pixel.leaves) == 129
    batch = pixel.leaves[1:]
    assert {leaf.drawn[3] for leaf in batch} == {64}
    assert all(leaf.shown for leaf in pixel.leaves)


def _square_under_coloured(colour: str):
    """:func:`_square_under` with the square beneath in ``colour``."""

    def made(top, *, side: float = 2.0):
        from arachne import Camera, Square, tempconfig

        with tempconfig(MEDIUM):
            under = Square(
                side_length=side, fill_color=colour, fill_opacity=1.0, stroke_width=0
            )
            painter = IrCairoPainter()
            frame = painter.paint(
                Camera(), [("moving", [under, *top()])], t=Fraction(0)
            )
            return frame, painter.contributors(_box(*CENTRE))[CENTRE]

    return made
