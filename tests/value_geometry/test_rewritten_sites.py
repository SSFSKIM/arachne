"""Every place the engine wrote into an existing array now builds a new value. Each
test pins the result CE computed, and that a holder of the old value sees no change.
"""

from __future__ import annotations

import numpy as np
import pytest

import arachne.utils.hashing as hashing
from arachne import (
    BLUE,
    DOWN,
    ORIGIN,
    PI,
    RED,
    RIGHT,
    UP,
    Circle,
    DecimalNumber,
    Homotopy,
    ImageMobject,
    Mobject,
    PMobject,
    Square,
    VMobject,
    Wiggle,
)
from arachne.camera.moving_camera import MovingCamera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.mobject.types.image_mobject import ImageMobjectFromCamera
from arachne.mobject.values import FrozenPixels, FrozenPointColors
from arachne.utils.bezier import interpolate, partial_bezier_points
from arachne.utils.color import color_to_int_rgb, color_to_rgba
from arachne.utils.rate_functions import there_and_back
from arachne.utils.rate_functions import wiggle as wiggle_rate
from arachne.utils.space_ops import rotation_matrix


def _held(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The array a holder kept, and a snapshot of its content."""
    return array, np.array(array)


def test_shift_adds_the_vector():
    square = Square()
    held, before = _held(square.points)
    square.shift(RIGHT, 2 * UP)
    np.testing.assert_array_equal(square.points, before + (RIGHT + 2 * UP))
    np.testing.assert_array_equal(held, before)


def test_shift_turns_integer_points_into_floats():
    mob = Mobject()
    mob.points = np.array([[1, 2, 3]])
    mob.shift(np.array([0.5, 0, 0]))
    assert mob.points.dtype == np.float64
    np.testing.assert_array_equal(mob.points, [[1.5, 2, 3]])


@pytest.mark.parametrize(
    ("operation", "mapping"),
    [
        ("scale", lambda p: p * 3),
        ("stretch", lambda p: p * [1, 2, 1]),
        ("rotate", lambda p: np.dot(p, rotation_matrix(0.7, [0, 0, 1]).T)),
    ],
)
def test_point_functions_map_about_the_point(operation, mapping):
    about = np.array([0.5, -1.0, 0.0])
    square = Square().shift(RIGHT)
    held, before = _held(square.points)
    args = {
        "scale": (3,),
        "stretch": (2, 1),
        "rotate": (0.7,),
    }[operation]
    getattr(square, operation)(*args, about_point=about)
    np.testing.assert_allclose(square.points, mapping(before - about) + about)
    np.testing.assert_array_equal(held, before)


def test_a_point_function_may_write_into_the_array_it_is_given():
    def double_in_place(points):
        points *= 2
        return points

    square = Square()
    held, before = _held(square.points)
    square.apply_points_function_about_point(double_in_place, about_point=ORIGIN)
    np.testing.assert_array_equal(square.points, before * 2)
    np.testing.assert_array_equal(held, before)


def test_set_anchors_and_handles_interleaves_the_four_arrays():
    a1, h1, h2, a2 = (np.full((2, 3), float(k)) for k in range(4))
    mob = VMobject().set_anchors_and_handles(a1, h1, h2, a2)
    np.testing.assert_array_equal(mob.points[:, 0], [0, 1, 2, 3, 0, 1, 2, 3])
    assert not mob.points.flags.writeable


def test_append_points_extends_without_touching_the_old_value():
    square = Square()
    held, before = _held(square.points)
    extra = np.ones((4, 3))
    square.append_points(extra)
    np.testing.assert_array_equal(square.points, np.vstack([before, extra]))
    np.testing.assert_array_equal(held, before)


@pytest.mark.parametrize(("a", "b"), [(0.1, 0.9), (0.3, 0.35), (0.0, 1.0)])
def test_pointwise_become_partial_takes_the_bezier_portion(a, b):
    circle = Circle()
    part = VMobject().pointwise_become_partial(circle, a, b)
    nppc = 4
    n = circle.get_num_curves()
    curves = circle.points.reshape(n, nppc, 3)
    lo, hi = a * n, b * n
    lo_i, hi_i = min(int(lo), n - 1), min(int(hi), n - 1)
    if (a, b) == (0.0, 1.0):
        expected = circle.points
    elif lo_i == hi_i:
        expected = partial_bezier_points(curves[lo_i], lo - lo_i, hi - hi_i)
    else:
        expected = np.vstack(
            [
                partial_bezier_points(curves[lo_i], lo - lo_i, 1),
                curves[lo_i + 1 : hi_i].reshape(-1, 3),
                partial_bezier_points(curves[hi_i], 0, hi - hi_i),
            ]
        )
    np.testing.assert_allclose(part.points, expected)


def test_pointwise_become_partial_of_itself():
    circle = Circle()
    held, before = _held(circle.points)
    circle.pointwise_become_partial(circle, 0.2, 0.7)
    assert len(circle.points) < len(before)
    np.testing.assert_array_equal(held, before)


def test_decimal_number_zeroes_the_old_digits():
    number = DecimalNumber(1.5)
    old_digits = list(number.submobjects)
    held, before = _held(old_digits[0].points)
    number.set_value(23.25)
    for digit in old_digits:
        assert not digit.points.any()
    np.testing.assert_array_equal(held, before)
    assert number.get_value() == 23.25


def test_wiggle_starts_each_frame_from_the_start_points():
    square = Square().shift(RIGHT)
    wiggle = Wiggle(square, scale_value=1.5, rotation_angle=0.1)
    wiggle.begin()
    start = wiggle.starting_mobject
    held, before = _held(start.points)
    for alpha in (0.6, 0.3):  # the second frame must not build on the first
        wiggle.interpolate_submobject(square, start, alpha)
    reference = start.copy()
    reference.scale(
        interpolate(1, 1.5, there_and_back(0.3)), about_point=start.get_center()
    )
    reference.rotate(
        wiggle_rate(0.3, wiggle.n_wiggles) * 0.1, about_point=start.get_center()
    )
    np.testing.assert_allclose(square.points, reference.points)
    np.testing.assert_array_equal(held, before)


def test_homotopy_leaves_the_starting_copy_alone():
    mob = Mobject()
    mob.points = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.0]])
    homotopy = Homotopy(
        lambda x, y, z, t: (x + t, y, z),
        mob,
        apply_function_kwargs={"about_point": np.array([1.0, 1.0, 0.0])},
    )
    homotopy.begin()
    start = homotopy.starting_mobject
    held, before = _held(start.points)
    for alpha in (0.25, 0.5, 0.75):
        homotopy.interpolate_submobject(mob, start, alpha)
    np.testing.assert_array_equal(held, before)
    np.testing.assert_array_equal(start.points, before)
    np.testing.assert_allclose(mob.points, before + [0.75, 0, 0])


def test_point_cloud_colours_are_read_only_and_set_color_replaces_them():
    cloud = PMobject().add_points([[0, 0, 0], [1, 0, 0]], color=RED)
    held, before = _held(cloud.rgbas)
    assert isinstance(cloud.rgbas, FrozenPointColors)
    with pytest.raises(ValueError, match="set_color"):
        cloud.rgbas[:, 3] = 0
    cloud.set_color(BLUE)
    np.testing.assert_array_equal(cloud.rgbas, [color_to_rgba(BLUE)] * 2)
    np.testing.assert_array_equal(held, before)


def _image() -> ImageMobject:
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 0] = np.arange(16).reshape(4, 4)
    pixels[..., 3] = 200
    return ImageMobject(pixels)


def test_image_pixels_are_read_only():
    image = _image()
    assert isinstance(image.pixel_array, FrozenPixels)
    with pytest.raises(ValueError, match="set_opacity"):
        image.pixel_array[0, 0, 0] = 1
    with pytest.raises(ValueError, match="set_color"):
        image.get_pixel_array()[..., 3] ^= 1


def test_image_set_color_set_opacity_and_fade():
    image = _image()
    held, before = _held(image.pixel_array)
    image.set_color(RED, alpha=0.5)
    assert (image.pixel_array[..., :3] == color_to_int_rgb(RED)).all()
    assert (image.pixel_array[..., 3] == int(255 * 0.5)).all()
    np.testing.assert_array_equal(held, before)
    image.set_opacity(0.5)
    np.testing.assert_array_equal(
        image.pixel_array[..., 3], (before[..., 3] * 0.5).astype(np.uint8)
    )
    image.fade(0.75)
    np.testing.assert_array_equal(
        image.pixel_array[..., 3], (before[..., 3] * 0.25).astype(np.uint8)
    )


def test_an_inverted_image():
    pixels = np.full((2, 2, 4), 10, dtype=np.uint8)
    image = ImageMobject(pixels, invert=True)
    assert (image.pixel_array[..., :3] == 245).all()
    assert (image.pixel_array[..., 3] == 10).all()
    assert (pixels == 10).all()


def test_an_image_from_a_camera_shows_the_cameras_current_pixels():
    camera = MovingCamera()
    display = ImageMobjectFromCamera(camera)
    camera.set_pixel_array(np.full_like(camera.pixel_array, 7))
    np.testing.assert_array_equal(display.get_pixel_array(), camera.pixel_array)
    with pytest.raises(ValueError, match="pixel_array"):
        display.get_pixel_array()[0, 0, 0] = 1


def test_getters_that_returned_live_arrays_return_read_only_views():
    square = Square()
    views = [
        *square.get_anchors_and_handles(),
        square.get_start_anchors(),
        square.get_end_anchors(),
        ThreeDCamera().frame_center,
    ]
    for view in views:
        with pytest.raises(ValueError, match="set_points"):
            view[0] = 0


def test_hashing_reads_values_by_content():
    a, b = Square(), Square()
    assert hashing.get_json(a) == hashing.get_json(b)
    b.set_fill(RED, 0.5)
    assert hashing.get_json(a) != hashing.get_json(b)
    # Large arrays that differ only in the middle, where a printed array elides.
    points = np.zeros((4000, 3))
    c, d = VMobject(), VMobject()
    c.points = points
    points[2000] = DOWN
    d.points = points
    assert hashing.get_json(c) != hashing.get_json(d)
    e = Square().rotate(PI / 7)
    assert hashing.get_json(Square()) != hashing.get_json(e)
