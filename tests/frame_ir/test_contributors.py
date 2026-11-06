"""Each painter's contributor report: per pixel, the drawn leaves of a frame
in the order they were composited (:mod:`arachne.frame_ir.contributors`).

Frames here are 192 x 108 pixels over CE's 8-unit frame height: 13.5 pixels a
unit, the origin at pixel (96, 54).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne.frame_ir.contributors import Leaf, PixelContributors, from_core, over
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
    # The core knows each leaf's coverage and colour apart.
    assert [leaf.coverage for leaf in composited.leaves] == [255, 255]
    assert composited.leaves[1].colour == composited.leaves[1].drawn


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
    assert [leaf.painter for leaf in beside.leaves] == ["core"]


def test_from_core_names_stable_ids_and_what_each_leaf_draws_alone():
    found = {
        (3, 4): (
            [(7, 0, 255, (0, 0, 200, 255), False), (8, 2, 128, (100, 0, 0, 128), True)],
            [[(0, 255), (1, 255)], [(0, 255)]],
            False,
        )
    }
    (pixel,) = from_core(found, {7: ("Square#1", "scene.py:3")}).values()
    assert pixel.keys == (("Square#1", 0), (8, 2))
    assert pixel.leaves[1].drawn == (50, 0, 0, 64)
    assert pixel.leaves[0].drawn == (0, 0, 200, 255)
    assert pixel.orders == (((0, 255), (1, 255)), ((0, 255),))
    assert pixel.shown == (pixel.leaves[1],)


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
