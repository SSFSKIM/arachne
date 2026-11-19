"""Stacking and read-back from Python: what a frame of layers with no effect draws."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from arachne_gpu import Frame, Layer, Picture, Quad, Unsupported, painted_key
from gpu_helpers import CLEAR, H, W, flat, over, painted_picture, xf_image_key

NEAREST = 0


def test_a_painted_picture_alone_comes_back_with_its_bytes(make_compositor):
    compositor = make_compositor(W, H)
    picture = painted_picture(W, H, 7)
    key = painted_key(picture)
    compositor.upload_painted(key, picture)
    frame = Frame(CLEAR, [Layer(Picture(key))])
    first = compositor.draw(frame)
    assert first.shape == (H, W, 4)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, picture)
    np.testing.assert_array_equal(compositor.draw(frame), first)


DRAW_IN_A_PROCESS = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
import conftest
from gpu_helpers import CLEAR, H, W, painted_picture
from arachne_gpu import Frame, Layer, Picture, painted_key
compositor = conftest.checked_compositor(W, H)
picture = painted_picture(W, H, 7)
compositor.upload_painted(painted_key(picture), picture)
drawn = compositor.draw(Frame(CLEAR, [Layer(Picture(painted_key(picture)))]))
print(hashlib.sha256(drawn.tobytes()).hexdigest())
"""


def test_a_painted_picture_comes_back_the_same_in_two_processes(make_compositor):
    here = str(Path(__file__).parent)
    digests = [
        subprocess.run(
            [sys.executable, "-c", DRAW_IN_A_PROCESS, here],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[-1]
        for _ in range(2)
    ]
    picture = painted_picture(W, H, 7)
    assert digests == [hashlib.sha256(picture.tobytes()).hexdigest()] * 2


def test_an_opaque_image_comes_back_with_its_bytes(make_compositor):
    compositor = make_compositor(W, H)
    image = painted_picture(W, H, 11)
    image[..., 3] = 255
    compositor.upload_image(xf_image_key(image), image)
    drawn = compositor.draw(
        Frame(bytes([9, 8, 7, 255]), [Layer(Picture(xf_image_key(image)))])
    )
    np.testing.assert_array_equal(drawn, image)


def test_a_straight_image_is_painted_exactly_for_every_colour_and_alpha(
    make_compositor,
):
    compositor = make_compositor(256, 256)
    colour, alpha = np.meshgrid(np.arange(256), np.arange(256))
    image = np.stack(
        [colour, 255 - colour, np.full_like(colour, 255), alpha], axis=2
    ).astype(np.uint8)
    compositor.upload_image(xf_image_key(image), image)
    drawn = compositor.draw(Frame(CLEAR, [Layer(Picture(xf_image_key(image)))]))
    a = image[..., 3:4].astype(np.uint32)
    expected = (image[..., :3].astype(np.uint32) * a + 127) // 255
    np.testing.assert_array_equal(drawn[..., :3], expected)
    np.testing.assert_array_equal(drawn[..., 3], image[..., 3])


def test_an_images_key_is_the_frame_irs_image_shape_key(make_compositor):
    compositor = make_compositor(W, H)
    image = painted_picture(W, H, 3)
    compositor.upload_image(xf_image_key(image), image)
    with pytest.raises(ValueError, match="image-shape key"):
        compositor.upload_image(painted_key(image), image)
    with pytest.raises(ValueError, match="painted key"):
        compositor.upload_painted(xf_image_key(image), image)


def test_an_image_and_a_painted_picture_of_equal_bytes_are_two_textures(
    make_compositor,
):
    compositor = make_compositor(W, H)
    pixels = flat(W, H, (200, 100, 50, 128))
    image, painted = xf_image_key(pixels), painted_key(pixels)
    assert image != painted
    compositor.upload_image(image, pixels)
    compositor.upload_painted(painted, pixels)
    from_image = compositor.draw(Frame(CLEAR, [Layer(Picture(image))]))
    from_painted = compositor.draw(Frame(CLEAR, [Layer(Picture(painted))]))
    assert from_image[0, 0].tolist() == [100, 50, 25, 128]
    assert from_painted[0, 0].tolist() == [200, 100, 50, 128]
    assert compositor.release(image) is None
    compositor.release(image)
    with pytest.raises(KeyError):
        compositor.draw(Frame(CLEAR, [Layer(Picture(image))]))
    compositor.draw(Frame(CLEAR, [Layer(Picture(painted))]))


def test_two_half_transparent_pictures_stack_over_in_their_order(make_compositor):
    compositor = make_compositor(W, H)
    a, b = painted_picture(W, H, 21), painted_picture(W, H, 22)
    compositor.upload_painted(painted_key(a), a)
    compositor.upload_painted(painted_key(b), b)
    background = (30, 60, 90, 255)
    base = flat(W, H, background)
    layers_a, layers_b = Layer(Picture(painted_key(a))), Layer(Picture(painted_key(b)))
    ab = compositor.draw(Frame(bytes(background), [layers_a, layers_b]))
    ba = compositor.draw(Frame(bytes(background), [layers_b, layers_a]))
    diff = lambda x, y: np.abs(x.astype(int) - y.astype(int)).max()  # noqa: E731
    assert diff(ab, over(b, over(a, base))) <= 1
    assert diff(ba, over(a, over(b, base))) <= 1
    assert not np.array_equal(ab, ba)


def test_a_quad_at_integer_corners_shows_its_texture_there(make_compositor):
    compositor = make_compositor(W, H)
    below = painted_picture(W, H, 31)
    texture = painted_picture(40, 30, 32)
    texture[..., 3] = 255
    compositor.upload_painted(painted_key(below), below)
    compositor.upload_painted(painted_key(texture), texture)
    x0, y0 = 100, 50
    corners = ((x0, y0), (x0 + 40, y0), (x0, y0 + 30), (x0 + 40, y0 + 30))
    drawn = compositor.draw(
        Frame(
            CLEAR,
            [
                Layer(Picture(painted_key(below))),
                Layer(Quad(painted_key(texture), corners, NEAREST)),
            ],
        )
    )
    expected = below.copy()
    expected[y0 : y0 + 30, x0 : x0 + 40] = texture
    np.testing.assert_array_equal(drawn, expected)


def test_opacity_zero_leaves_the_frame_as_without_the_layer(make_compositor):
    compositor = make_compositor(W, H)
    below, top = painted_picture(W, H, 41), painted_picture(W, H, 42)
    compositor.upload_painted(painted_key(below), below)
    compositor.upload_painted(painted_key(top), top)
    background = bytes((12, 34, 56, 255))
    without = compositor.draw(Frame(background, [Layer(Picture(painted_key(below)))]))
    zero = compositor.draw(
        Frame(
            background,
            [
                Layer(Picture(painted_key(below))),
                Layer(Picture(painted_key(top)), opacity=0.0),
            ],
        )
    )
    np.testing.assert_array_equal(zero, without)


def test_a_resampling_this_build_does_not_draw_refuses_the_frame(make_compositor):
    compositor = make_compositor(W, H)
    texture = flat(8, 8, (255, 0, 0, 255))
    compositor.upload_painted(painted_key(texture), texture)
    corners = ((0, 0), (8, 0), (0, 8), (8, 8))
    bicubic = 3
    with pytest.raises(Unsupported) as refused:
        compositor.draw(
            Frame(CLEAR, [Layer(Quad(painted_key(texture), corners, bicubic))])
        )
    assert refused.value.code == "render.unsupported"


def test_a_quad_with_no_placement_refuses_the_frame(make_compositor):
    compositor = make_compositor(W, H)
    texture = flat(8, 8, (255, 0, 0, 255))
    compositor.upload_painted(painted_key(texture), texture)
    # Upper left, upper right and lower right on one line, the lower left off it.
    collinear = ((10, 10), (50, 10), (10, 60), (90, 10))
    # Crossed (a bow-tie) and concave (a dart).
    bow_tie = ((100, 100), (200, 200), (100, 200), (200, 100))
    dart = ((100, 100), (200, 100), (100, 200), (125, 125))
    for corners in (collinear, bow_tie, dart):
        with pytest.raises(Unsupported):
            compositor.draw(
                Frame(CLEAR, [Layer(Quad(painted_key(texture), corners, NEAREST))])
            )


def test_a_frame_size_the_device_cannot_draw_raises_value_error(make_compositor):
    for width, height in ((0, 10), (10, 0), (1_000_000, 10)):
        with pytest.raises(ValueError):
            make_compositor(width, height)


def test_threads_drawing_different_frames_each_get_their_own(make_compositor):
    compositor = make_compositor(W, H)
    pictures = [painted_picture(W, H, 80 + i) for i in range(4)]
    for picture in pictures:
        compositor.upload_painted(painted_key(picture), picture)
    frames = [Frame(CLEAR, [Layer(Picture(painted_key(p)))]) for p in pictures]

    def draw_many(i: int) -> list[bool]:
        return [
            np.array_equal(compositor.draw(frames[i]), pictures[i]) for _ in range(10)
        ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(draw_many, range(4)))
    assert results == [[True] * 10] * 4
