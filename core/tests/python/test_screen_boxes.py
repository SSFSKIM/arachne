"""Screen boxes: where each record lands on the frame, without drawing, and the path
mapping they share with the drawer.
"""

from __future__ import annotations

import arachne_core as ac
import numpy as np
import pytest

# 16 x 9 frame units on 160 x 90 pixels: 10 px a unit, the origin at (80, 45).
FRAME = {
    "frame_width": 16.0,
    "frame_height": 9.0,
    "pixel_width": 160,
    "pixel_height": 90,
}


def segment(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])


def square(centre=(0.0, 0.0), half=1.0) -> np.ndarray:
    cx, cy = centre
    corners = [
        (cx - half, cy - half, 0),
        (cx + half, cy - half, 0),
        (cx + half, cy + half, 0),
        (cx - half, cy + half, 0),
    ]
    return np.concatenate([segment(corners[i], corners[(i + 1) % 4]) for i in range(4)])


def view_delta(**view) -> ac.FrameDelta:
    delta = ac.FrameDelta()
    delta.set_view(background=(0, 0, 0, 1), **FRAME, **view)
    return delta


def test_a_stream_without_a_view_has_no_boxes():
    with pytest.raises(ac.XfError, match="no view"):
        ac.Stream().screen_boxes()


def test_paths_map_around_the_mapping_centre_and_clouds_around_the_frame_s():
    delta = view_delta(frame_center=(0, 1, 0), mapping_centre=(1, 0, 0))
    path = delta.upload_path(square())
    delta.set_record("path", 1, 1, shape=path, fill=(1, 1, 1, 1))
    cloud = delta.upload_path(np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))
    colours = delta.upload_paint(np.ones((2, 4)))
    delta.set_record("points", 2, 2, shape=cloud, colors=colours, radius=2.0)
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    stream = ac.Stream()
    stream.apply(delta)
    boxes = stream.screen_boxes()
    assert boxes[1] == pytest.approx((60.0, 35.0, 80.0, 55.0))
    assert boxes[2] == pytest.approx((80.0, 45.0, 100.0, 55.0))


def test_the_drawer_paints_a_path_where_its_box_is():
    delta = view_delta(frame_center=(0, 0, 0), mapping_centre=(2, 0, 0))
    path = delta.upload_path(square(half=0.5))
    delta.set_record("path", 1, 1, shape=path, fill=(1, 1, 1, 1))
    delta.set_order([("moving", 1, 0)])
    stream = ac.Stream()
    stream.apply(delta)
    x0, y0, x1, y1 = stream.screen_boxes()[1]
    frame = stream.draw(160, 90)
    lit = np.argwhere(frame[..., 0] > 0)
    assert (lit[:, 1].min(), lit[:, 0].min()) == (int(x0), int(y0))
    assert (lit[:, 1].max() + 1, lit[:, 0].max() + 1) == (int(x1), int(y1))
    # The box is the square's, 2 units left of the frame's centre.
    assert (x0, x1) == pytest.approx((55.0, 65.0))


def test_a_face_group_has_a_box_and_is_drawn():
    delta = view_delta()
    faces = delta.upload_path(np.concatenate([square((-2, 0)), square((3, 1))]))
    fills = delta.upload_paint(np.ones((2, 4)))
    delta.set_record(
        "face_group",
        1,
        1,
        shape=faces,
        uv=[(0, 0), (1, 0)],
        grid=(2, 1),
        face_fill=fills,
    )
    delta.set_order([("moving", 1, 0), ("moving", 1, 1)])
    stream = ac.Stream()
    stream.apply(delta)
    assert stream.screen_boxes() == {1: pytest.approx((50.0, 25.0, 120.0, 55.0))}
    assert stream.draw(160, 90).shape == (90, 160, 4)


def test_only_slots_in_the_order_with_points_have_boxes():
    delta = view_delta()
    empty = delta.upload_path(np.zeros((0, 3)))
    path = delta.upload_path(square())
    delta.set_record("path", 1, 1, shape=empty, fill=(1, 1, 1, 1))
    delta.set_record("path", 2, 2, shape=path, fill=(1, 1, 1, 1))
    delta.set_record("path", 3, 3, shape=path, fill=(1, 1, 1, 1))
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    stream = ac.Stream()
    stream.apply(delta)
    assert set(stream.screen_boxes()) == {2}


def test_a_projected_box_moves_with_the_camera():
    delta = view_delta(
        camera="three_d",
        phi=0.0,
        theta=-np.pi / 2,
        gamma=0.0,
        focal_distance=20.0,
        zoom=1.0,
    )
    path = delta.upload_path(square())
    delta.set_record("path", 1, 1, shape=path, fill=(1, 1, 1, 1), space="world")
    delta.set_order([("moving", 1, 0)])
    stream = ac.Stream()
    stream.apply(delta)
    # Seen straight down, a square at z = 0 is its own size.
    assert stream.screen_boxes()[1] == pytest.approx((70.0, 35.0, 90.0, 55.0))
    lifted = ac.FrameDelta(seq=1)
    lifted.set_record(
        "path", 1, 1, shape=path, fill=(1, 1, 1, 1), space="world", place=_lift(10.0)
    )
    stream.apply(lifted)
    # Halfway to the eye: twice the size.
    assert stream.screen_boxes()[1] == pytest.approx((60.0, 25.0, 100.0, 65.0))


def _lift(z: float) -> np.ndarray:
    place = np.eye(3, 4)
    place[2, 3] = z
    return place
