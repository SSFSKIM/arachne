"""A window composed at its own size, from Python: what the binding hands the compositor."""

from __future__ import annotations

import numpy as np
import pytest
from arachne_gpu import (
    Blur,
    Frame,
    Layer,
    Picture,
    SizeMismatch,
    Window,
    painted_key,
)
from gpu_helpers import CLEAR, H, W, painted_picture

NEAREST = 0
BILINEAR = 2


def upload(compositor, rgba: np.ndarray) -> bytes:
    key = painted_key(rgba)
    compositor.upload_painted(key, rgba)
    return key


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


def content(compositor, width: int, height: int) -> Frame:
    """An opaque frame of ``width`` x ``height`` with a blurred picture."""
    picture = upload(compositor, painted_picture(width, height, 11))
    blurred = upload(compositor, painted_picture(width, height, 12))
    return Frame(
        bytes((20, 40, 60, 255)),
        [Layer(Picture(picture)), Layer(Picture(blurred), blur=Blur(2.0))],
    )


@pytest.mark.parametrize("resampling", [NEAREST, BILINEAR])
def test_a_sized_window_at_whole_pixel_corners_is_its_frame_drawn_alone(
    make_compositor, resampling
):
    iw, ih = 96, 54
    small = make_compositor(iw, ih)
    alone = small.draw(content(small, iw, ih))
    compositor = make_compositor(W, H)
    x0, y0 = 40, 30
    window = Window(
        content(compositor, iw, ih),
        rectangle(x0, y0, x0 + iw, y0 + ih),
        resampling,
        size=(iw, ih),
    )
    drawn = compositor.draw(Frame(CLEAR, [Layer(window)]))
    np.testing.assert_array_equal(drawn[y0 : y0 + ih, x0 : x0 + iw], alone)
    outside = np.ones((H, W), bool)
    outside[y0 : y0 + ih, x0 : x0 + iw] = False
    assert not drawn[outside].any()


def test_a_picture_not_of_its_windows_size_raises_size_mismatch(make_compositor):
    compositor = make_compositor(W, H)
    frame_sized = upload(compositor, painted_picture(W, H, 21))
    inner = Frame(CLEAR, [Layer(Picture(frame_sized))])
    window = Window(inner, rectangle(10, 10, 106, 64), NEAREST, size=(96, 54))
    with pytest.raises(SizeMismatch) as refused:
        compositor.draw(Frame(CLEAR, [Layer(window)]))
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == (
        f"layer 0: in its window, layer 0: a picture of {W} x {H} in a frame of 96 x 54:"
        " a frame's pictures and masks are of its size, and a window's frame is of the"
        " window's size"
    )
    # With no size, the window is the frame's size, and the same picture is drawn.
    unsized = Window(inner, rectangle(0, 0, W, H), NEAREST)
    alone = compositor.draw(inner)
    np.testing.assert_array_equal(
        compositor.draw(Frame(CLEAR, [Layer(unsized)])), alone
    )


def test_outside_a_sized_window_a_size_refusal_is_the_plain_value_error(
    make_compositor,
):
    compositor = make_compositor(W, H)
    small = upload(compositor, painted_picture(96, 54, 22))
    unsized = Window(
        Frame(CLEAR, [Layer(Picture(small))]), rectangle(0, 0, W, H), NEAREST
    )
    with pytest.raises(ValueError) as refused:
        compositor.draw(Frame(CLEAR, [Layer(unsized)]))
    assert type(refused.value) is ValueError
    assert str(refused.value) == (
        f"layer 0: in its window, layer 0: a picture of 96 x 54 in a frame of {W} x {H}"
    )
