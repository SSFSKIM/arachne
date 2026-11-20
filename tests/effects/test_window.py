"""A window as a scene makes it, read with no device: the size and the
resampling the stack gives a window at its corners, what a window holds and
declares, and a frame refused on a host with no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne import Circle, Square, Window, clear_effects, routing, set_blur, set_mask
from arachne.compose.layer import layer_of
from arachne.compose.stack import _window_size
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .window_scenes import FRAME

SCENES = Path(__file__).parent / "window_scenes.py"

NEAREST, BILINEAR = 0, 2


def _rectangle(left: float, top: float, right: float, bottom: float) -> list:
    """Corners in CE's image order: upper left, upper right, lower left,
    lower right.
    """
    return [(left, top), (right, top), (left, bottom), (right, bottom)]


# The GPU build's rule, as its README words it: "per axis, `size` is the true
# rectangle's size rounded to the nearest whole pixel, ties half up, at least
# 1; the true corners pass unchanged; resampling is nearest when the
# rectangle is integral and 1:1 with `size`, and bilinear otherwise."
SIZED = [
    # Whole pixels: nearest, the size the rectangle's.
    (_rectangle(24, 13, 72, 40), ((48, 27), NEAREST)),
    (_rectangle(0, 0, 1, 1), ((1, 1), NEAREST)),
    # Off the grid: rounded to the nearest whole pixel, bilinear.
    (_rectangle(10.25, 5.5, 18.75, 13.0), ((9, 8), BILINEAR)),
    (_rectangle(0.2, 0, 10.6, 3), ((10, 3), BILINEAR)),
    # Ties go up.
    (_rectangle(0, 0, 2.5, 3.5), ((3, 4), BILINEAR)),
    (_rectangle(0.25, 0.25, 2.75, 4.75), ((3, 5), BILINEAR)),
    # At least 1.
    (_rectangle(3, 3, 3.4, 3.1), ((1, 1), BILINEAR)),
    # Whole-pixel sides at fractional corners: the size is the rectangle's,
    # but the corners are not integral, so bilinear.
    (_rectangle(0.5, 0.5, 8.5, 4.5), ((8, 4), BILINEAR)),
]


@pytest.mark.parametrize(("corners", "expected"), SIZED)
def test_a_window_at_an_axis_aligned_rectangle_is_sized_by_the_gpu_build_s_rule(
    corners, expected
):
    assert _window_size(corners) == expected


UNSIZED = {
    # The GPU build counts these rectangles; the stack keeps them at the
    # frame's size and nearest, as every window was before windows had
    # sizes.
    "mirrored left to right": [(18, 5), (10, 5), (18, 13), (10, 13)],
    "mirrored top to bottom": [(10, 13), (18, 13), (10, 5), (18, 5)],
    "turned a half": [(18, 13), (10, 13), (18, 5), (10, 5)],
    "turned a quarter": [(10, 13), (10, 5), (18, 13), (18, 5)],
    # Not rectangles at all.
    "tilted": [(15.01, 15.5), (20.01, 20.5), (10.01, 20.5), (15.01, 25.5)],
    "a trapezoid": [(10, 5), (18, 5), (8, 13), (20, 13)],
    # No area.
    "a line": _rectangle(10, 5, 18, 5),
    "a point": _rectangle(10, 5, 10, 5),
}


@pytest.mark.parametrize("placed", sorted(UNSIZED))
def test_a_window_at_any_other_corners_keeps_the_frame_s_size(placed):
    assert _window_size(UNSIZED[placed]) is None


def test_a_window_shows_its_objects_in_place_of_a_picture_of_its_own():
    dot, square = Circle(radius=0.1), Square()
    window = Window(dot, square, width=4, height=2.25)
    spec = layer_of(window)
    assert spec is not None
    assert spec.window == (dot, square)
    assert (spec.blur, spec.mask, spec.depth) == (None, None, None)
    assert (window.width, window.height) == pytest.approx((4, 2.25))
    assert window.get_fill_opacity() == 0
    assert window.get_stroke_width() == 0
    assert routing.held(window) == {"effect.window"}


def test_a_window_that_cannot_be_shown_is_refused_where_it_is_made():
    with pytest.raises(ValueError, match="at least one object"):
        Window(width=4, height=2)
    with pytest.raises(TypeError, match="shows mobjects"):
        Window("a square", width=4, height=2)
    with pytest.raises(ValueError, match="width is a positive length"):
        Window(Square(), width=0, height=2)
    with pytest.raises(ValueError, match="height is a positive length"):
        Window(Square(), width=4, height=float("nan"))


def test_clearing_a_window_s_effects_keeps_what_it_shows():
    square = Square()
    window = Window(square, width=4, height=2.25)
    set_blur(window, 0.2)
    set_mask(window, Circle(), feather=0.1)
    assert routing.held(window) == {"effect.window", "effect.blur", "effect.soft_mask"}
    clear_effects(window)
    spec = layer_of(window)
    assert spec is not None
    assert spec.window == (square,)
    assert (spec.blur, spec.mask, spec.mask_feather, spec.depth) == (
        None,
        None,
        0.0,
        None,
    )
    assert routing.held(window) == {"effect.window"}


def test_a_frame_of_a_scene_with_a_window_is_refused_with_no_gpu(tmp_path, no_gpu):
    server = Dispatcher()
    config = {**FRAME, "media_dir": str(tmp_path / "media")}
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "ShownPlainHalf", "config": config}
        )
        assert server.handle("state_at", {"t": 1})["state"]["nodes"]
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 1, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    assert refused.value.data["reason"] == "render.gpu_required"
