"""The effects and the object map from Python: what the binding hands the compositor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arachne_gpu import (
    Blur,
    Focus,
    Footprint,
    Frame,
    Layer,
    Picture,
    Quad,
    SoftMask,
    Unsupported,
    Window,
    painted_key,
)
from gpu_helpers import CLEAR, H, W, flat, painted_picture

NEAREST = 0
TEN_TO_NINETY = 2.563


def upload(compositor, rgba: np.ndarray) -> bytes:
    key = painted_key(rgba)
    compositor.upload_painted(key, rgba)
    return key


def step(edge: int) -> np.ndarray:
    """Opaque white left of column ``edge``, opaque black from it on."""
    picture = flat(W, H, (0, 0, 0, 255))
    picture[:, :edge] = 255
    return picture


def crossing(values: np.ndarray, level: float) -> float:
    """Where a falling row first drops below ``level``, between pixel centres."""
    x = int(np.flatnonzero((values[:-1] >= level) & (values[1:] < level))[0])
    a, b = float(values[x]), float(values[x + 1])
    return x + 0.5 + (a - level) / (a - b)


def edge_width(drawn: np.ndarray) -> float:
    values = drawn[H // 2, :, 0].astype(float)
    return crossing(values, 0.1 * 255) - crossing(values, 0.9 * 255)


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


def test_a_blurred_step_edge_has_the_declared_width(make_compositor):
    compositor = make_compositor(W, H)
    key = upload(compositor, step(240))
    drawn = compositor.draw(Frame(CLEAR, [Layer(Picture(key), blur=Blur(6.0))]))
    assert abs(edge_width(drawn) - TEN_TO_NINETY * 6.0) <= 1.0


def test_a_layer_at_the_focus_returns_its_bytes_and_one_off_it_blurs(
    make_compositor,
):
    compositor = make_compositor(W, H)
    picture = painted_picture(W, H, 9)
    key = upload(compositor, picture)
    focus = Focus(depth=3.0, aperture=1.5)
    in_focus = Frame(CLEAR, [Layer(Picture(key), depth=3.0)], focus=focus)
    np.testing.assert_array_equal(compositor.draw(in_focus), picture)
    edge = upload(compositor, step(240))
    off_focus = Frame(CLEAR, [Layer(Picture(edge), depth=7.0)], focus=focus)
    assert abs(edge_width(compositor.draw(off_focus)) - TEN_TO_NINETY * 6.0) <= 1.0


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@pytest.mark.parametrize("invert", [False, True])
def test_a_masks_feather_is_the_gaussian_it_declares(make_compositor, invert):
    # White over black: the red channel reads the feathered alpha, a Gaussian of sigma
    # feather / 3 across the mask's edge. The mask's colour differs from its alpha, since
    # only the alpha masks; with no feather the pixels below would read 255 and 0.
    compositor = make_compositor(W, H)
    feather, sigma = 10.0, 10.0 / 3
    white = upload(compositor, flat(W, H, (255, 255, 255, 255)))
    black = upload(compositor, flat(W, H, (0, 0, 0, 255)))
    across_columns = np.zeros((H, W, 4), np.uint8)
    across_columns[:, :200] = (40, 0, 90, 255)
    across_rows = np.zeros((H, W, 4), np.uint8)
    across_rows[:120] = (40, 0, 90, 255)
    for mask, edge, along_x in ((across_columns, 200, True), (across_rows, 120, False)):
        feathered = SoftMask(
            upload(compositor, mask), feather_px=feather, invert=invert
        )
        drawn = compositor.draw(
            Frame(CLEAR, [Layer(Picture(black)), Layer(Picture(white), mask=feathered)])
        )
        line = drawn[H // 2, :, 0] if along_x else drawn[:, W // 2, 0]
        for step in range(12):
            for at, sign in ((edge - 1 - step, 1), (edge + step, -1)):
                kept = phi(sign * (step + 0.5) / sigma)
                expected = 255 * (1 - kept if invert else kept)
                assert abs(float(line[at]) - expected) <= 2, (along_x, at)


def test_a_window_at_the_frames_corners_is_its_stack_composed_alone(
    make_compositor,
):
    compositor = make_compositor(W, H)
    picture = upload(compositor, painted_picture(W, H, 31))
    inner = Frame(bytes((20, 40, 60, 255)), [Layer(Picture(picture))])
    alone = compositor.draw(inner)
    window = Window(inner, rectangle(0, 0, W, H), NEAREST)
    np.testing.assert_array_equal(compositor.draw(Frame(CLEAR, [Layer(window)])), alone)


def test_the_object_map_names_each_objects_footprint(make_compositor):
    compositor = make_compositor(W, H)
    below = upload(compositor, painted_picture(W, H, 2))
    texture = painted_picture(10, 10, 3)
    texture[..., 3] = 255
    texture = upload(compositor, texture)
    frame = Frame(
        CLEAR,
        [
            Layer(Picture(below)),
            Layer(
                Quad(texture, rectangle(100, 50, 180, 120), NEAREST),
                blur=Blur(2.5),
                object_id="photo",
            ),
            Layer(
                Quad(texture, rectangle(150, 100, 200, 140), NEAREST),
                object_id="label",
            ),
        ],
    )
    photo, label = compositor.object_map(frame)
    assert isinstance(photo, Footprint)
    assert (photo.object_id, photo.layer, photo.box) == (
        "photo",
        1,
        (100, 50, 180, 120),
    )
    assert photo.mask.dtype == np.bool_
    assert photo.mask.shape == (70, 80)
    assert photo.mask.all()
    assert photo.grow_px == 7.5
    assert (label.object_id, label.layer, label.box) == (
        "label",
        2,
        (150, 100, 200, 140),
    )
    assert label.mask.all()
    assert label.grow_px == 0.0


def test_a_frame_holding_itself_is_refused_not_followed(make_compositor):
    compositor = make_compositor(W, H)
    layers: list[Layer] = []
    frame = Frame(CLEAR, layers)
    layers.append(Layer(Window(frame, rectangle(0, 0, W, H), NEAREST)))
    with pytest.raises(Unsupported):
        compositor.draw(frame)


def test_an_effect_out_of_its_range_raises(make_compositor):
    compositor = make_compositor(W, H)
    key = upload(compositor, painted_picture(W, H, 51))
    with pytest.raises(ValueError):
        compositor.draw(Frame(CLEAR, [Layer(Picture(key), blur=Blur(-1.0))]))
    with pytest.raises(ValueError):
        compositor.draw(
            Frame(CLEAR, [Layer(Picture(key))], focus=Focus(depth=0.0, aperture=-1.0))
        )
    with pytest.raises(Unsupported):
        compositor.draw(Frame(CLEAR, [Layer(Picture(key), blur=Blur(200.0))]))
