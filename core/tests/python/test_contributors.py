"""The contributor report: per pixel, the leaves the core composites, in depth order.

Every scene here is seen straight down (a 3D view at phi 0, so a larger z is nearer) on a frame
of 16 x 9 units and 160 x 90 pixels: 10 pixels a unit, the origin at pixel (80, 45), a unit at
z = 0 its own size.
"""

from __future__ import annotations

import arachne_core as ac
import numpy as np

FRAME = {
    "frame_width": 16.0,
    "frame_height": 9.0,
    "pixel_width": 160,
    "pixel_height": 90,
}
DOWN = {
    "camera": "three_d",
    "phi": 0.0,
    "theta": -np.pi / 2,
    "gamma": 0.0,
    "focal_distance": 20.0,
    "zoom": 1.0,
    "should_apply_shading": False,
}
# Pixel (75, 40): world (-0.45, 0.45), inside every unit square centred at the origin.
INSIDE = (75, 40)


def view_delta(**view) -> ac.FrameDelta:
    delta = ac.FrameDelta()
    delta.set_view(background=(0, 0, 0, 1), **FRAME, **{**DOWN, **view})
    return delta


def plane(delta: ac.FrameDelta, z, half: float = 1.0) -> bytes:
    """A square mesh of side ``2 * half`` about the origin, its corners at heights ``z(x)``."""
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    positions = np.array([(x, y, z(x)) for x, y in corners], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return delta.upload_mesh(positions, triangles)


def flat(delta: ac.FrameDelta, z: float, half: float = 1.0) -> bytes:
    return plane(delta, lambda x: z, half)


def segment(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])


def square(z: float = 0.0, half: float = 1.0, centre=(0.0, 0.0)) -> np.ndarray:
    cx, cy = centre
    corners = [
        (cx - half, cy - half, z),
        (cx + half, cy - half, z),
        (cx + half, cy + half, z),
        (cx - half, cy + half, z),
    ]
    return np.concatenate([segment(corners[i], corners[(i + 1) % 4]) for i in range(4)])


def stream_of(delta: ac.FrameDelta) -> ac.Stream:
    stream = ac.Stream()
    stream.apply(delta)
    return stream


def at(stream: ac.Stream, x: int, y: int):
    """The report of one pixel: (leaves, orders, crossing)."""
    return stream.contributors(x, y, x + 1, y + 1)[(x, y)]


def ids(leaves) -> list[tuple[int, int]]:
    return [(obj, part) for obj, part, *_ in leaves]


def test_leaves_come_bottom_first_in_depth_order_whatever_the_draw_order():
    delta = view_delta()
    near = flat(delta, 1.0)
    middle = flat(delta, 0.0)
    far = flat(delta, -1.0)
    delta.set_record("mesh", 1, 11, shape=near, fill=(1, 0, 0, 0.5), space="world")
    delta.set_record("mesh", 2, 12, shape=middle, fill=(0, 1, 0, 0.5), space="world")
    delta.set_record("mesh", 3, 13, shape=far, fill=(0, 0, 1, 1), space="world")
    delta.set_order([("moving", 1, 0), ("moving", 2, 0), ("moving", 3, 0)])
    leaves, orders, crossing = at(stream_of(delta), *INSIDE)
    assert ids(leaves) == [(13, 0), (12, 0), (11, 0)]
    assert [shown for *_, shown in leaves] == [True, True, True]
    assert [coverage for _, _, coverage, _, _ in leaves] == [255, 255, 255]
    # One sequence: every sub-sample sees the three in one order, each covering it whole.
    assert orders == [[(0, 255), (1, 255), (2, 255)]]
    assert crossing is False


def test_behind_an_unchanged_translucent_front_two_rear_leaves_swap():
    def report(green_z: float, blue_z: float):
        delta = view_delta()
        front = flat(delta, 2.0)
        green = flat(delta, green_z)
        blue = flat(delta, blue_z)
        delta.set_record("mesh", 1, 11, shape=front, fill=(1, 0, 0, 0.5), space="world")
        delta.set_record("mesh", 2, 12, shape=green, fill=(0, 1, 0, 0.5), space="world")
        delta.set_record("mesh", 3, 13, shape=blue, fill=(0, 0, 1, 0.5), space="world")
        delta.set_order([("moving", 1, 0), ("moving", 2, 0), ("moving", 3, 0)])
        stream = stream_of(delta)
        return ids(at(stream, *INSIDE)[0]), stream.draw(160, 90)[INSIDE[1], INSIDE[0]]

    before, drawn_before = report(green_z=0.0, blue_z=-1.0)
    after, drawn_after = report(green_z=-1.0, blue_z=0.0)
    assert before == [(13, 0), (12, 0), (11, 0)]
    assert after == [(12, 0), (13, 0), (11, 0)]
    # The top leaf is the same; the pixel is not: the top alone does not explain it.
    assert before[-1] == after[-1]
    assert (drawn_before != drawn_after).any()


def test_a_leaf_behind_an_opaque_front_is_reported_and_not_shown():
    delta = view_delta()
    front = flat(delta, 1.0)
    rear = flat(delta, 0.0, half=2.0)
    delta.set_record("mesh", 1, 11, shape=front, fill=(1, 0, 0, 1), space="world")
    delta.set_record("mesh", 2, 12, shape=rear, fill=(0, 0, 1, 1), space="world")
    delta.set_order([("moving", 2, 0), ("moving", 1, 0)])
    stream = stream_of(delta)
    leaves, orders, _ = at(stream, *INSIDE)
    assert [(o, p, shown) for o, p, _, _, shown in leaves] == [
        (12, 0, False),
        (11, 0, True),
    ]
    assert leaves[1][3] == (255, 0, 0, 255)
    assert orders == [[(0, 255), (1, 255)]]
    # Outside the front, the rear is shown and alone.
    alone, _, _ = at(stream, 62, 45)
    assert [(o, shown) for o, _, _, _, shown in alone] == [(12, True)]


def test_the_report_matches_what_the_core_draws_under_an_opaque_top():
    delta = view_delta()
    top = flat(delta, 1.0)
    under = flat(delta, 0.0, half=2.0)
    delta.set_record("mesh", 1, 11, shape=top, fill=(0.2, 0.6, 0.4, 1), space="world")
    delta.set_record("mesh", 2, 12, shape=under, fill=(1, 1, 1, 1), space="world")
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    stream = stream_of(delta)
    leaves, _, _ = at(stream, *INSIDE)
    drawn = stream.draw(160, 90)[INSIDE[1], INSIDE[0]]
    assert tuple(int(v) for v in drawn) == leaves[-1][3]


def test_a_surface_overlapping_itself_reports_each_face_by_its_part():
    # Two faces of one face group, the second above the first and coloured apart.
    delta = view_delta()
    faces = delta.upload_path(np.concatenate([square(z=0.0), square(z=1.0, half=0.8)]))
    fills = delta.upload_paint(np.array([[1, 0, 0, 1], [0, 1, 0, 0.5]], dtype=float))
    delta.set_record(
        "face_group",
        1,
        21,
        shape=faces,
        uv=[(0, 0), (1, 0)],
        grid=(2, 1),
        face_fill=fills,
        space="world",
        shade_in_3d=True,
    )
    delta.set_order([("moving", 1, 1), ("moving", 1, 0)])
    leaves, _, crossing = at(stream_of(delta), *INSIDE)
    assert ids(leaves) == [(21, 0), (21, 1)]
    assert crossing is False


def test_overlapping_translucent_instances_report_their_own_objects():
    delta = view_delta()
    prototype = delta.upload_path(square())
    place = np.stack([np.eye(3, 4), np.eye(3, 4)])
    place[1, 2, 3] = 1.0  # the second instance one unit nearer
    place[1, 0, 3] = 0.5  # and half a unit right, so they overlap partly
    fills = delta.upload_paint(np.array([[1, 0, 0, 0.5], [0, 0, 1, 0.5]], dtype=float))
    strokes = delta.upload_paint(np.zeros((2, 4)))
    delta.set_record(
        "instances",
        1,
        30,
        shape=prototype,
        placements=place,
        instance_fill=fills,
        instance_stroke=strokes,
        objects=[31, 32],
        # A fill of one colour marks the members as filled, each with its own row.
        fill=(1, 1, 1, 1),
        space="world",
    )
    delta.set_order([("moving", 1, 0)])
    stream = stream_of(delta)
    # World (0, 0.45): inside both.
    assert ids(at(stream, 80, 40)[0]) == [(31, 0), (32, 0)]
    # World (-0.75, 0.45): the first alone.
    assert ids(at(stream, 72, 40)[0]) == [(31, 0)]


def test_two_surfaces_crossing_inside_a_pixel_report_each_order():
    # Planes through the line x = 0.05 (pixel column 80's centre), sloped apart: left of it the
    # falling one is nearer, right of it the rising one.
    delta = view_delta()
    rising = plane(delta, lambda x: 0.5 * (x - 0.05))
    falling = plane(delta, lambda x: -0.5 * (x - 0.05))
    delta.set_record("mesh", 1, 11, shape=rising, fill=(1, 0, 0, 1), space="world")
    delta.set_record("mesh", 2, 12, shape=falling, fill=(0, 0, 1, 1), space="world")
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    stream = stream_of(delta)
    leaves, orders, crossing = at(stream, 80, 40)
    assert crossing is True
    assert sorted(ids(leaves)) == [(11, 0), (12, 0)]
    sequences = {tuple(leaves[i][0] for i, _ in order) for order in orders}
    assert sequences == {(11, 12), (12, 11)}
    # Away from the line, one order and no crossing, the falling plane on top on the left.
    leaves, orders, crossing = at(stream, 72, 40)
    assert crossing is False
    assert ids(leaves) == [(11, 0), (12, 0)]
    assert len(orders) == 1


def test_an_edge_pixel_reports_its_share_and_each_sub_sample_s_sequence():
    delta = view_delta()
    # The square's left edge at world x = -0.95: the middle of pixel column 70.
    edge = delta.upload_path(square(half=0.95))
    delta.set_record("path", 1, 11, shape=edge, fill=(1, 1, 1, 1), space="world")
    delta.set_order([("moving", 1, 0)])
    leaves, orders, _ = at(stream_of(delta), 70, 45)
    ((_, _, coverage, _, shown),) = leaves
    assert 96 <= coverage <= 160
    assert shown is True
    assert all(index == 0 for order in orders for index, _ in order)


def test_the_box_is_clipped_to_the_frame_and_empty_pixels_are_absent():
    delta = view_delta()
    small = flat(delta, 0.0, half=0.2)
    delta.set_record("mesh", 1, 11, shape=small, fill=(1, 1, 1, 1), space="world")
    delta.set_order([("moving", 1, 0)])
    stream = stream_of(delta)
    found = stream.contributors(0, 0, 10_000, 10_000)
    assert found
    assert all(78 <= x < 82 and 43 <= y < 47 for x, y in found)
    assert stream.contributors(0, 0, 10, 10) == {}
    assert stream.contributors(50, 50, 40, 40) == {}


def test_the_overlay_is_not_reported():
    delta = view_delta()
    label = delta.upload_path(square())
    rear = flat(delta, 0.0)
    delta.set_record("mesh", 1, 11, shape=rear, fill=(0, 0, 1, 1), space="world")
    delta.set_record("path", 2, 12, shape=label, fill=(1, 1, 1, 1), space="overlay")
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    leaves, _, _ = at(stream_of(delta), *INSIDE)
    assert ids(leaves) == [(11, 0)]


def test_a_flat_frame_reports_its_leaves_in_draw_order():
    delta = ac.FrameDelta()
    delta.set_view(background=(0, 0, 0, 1), **FRAME)
    first = delta.upload_path(square())
    second = delta.upload_path(square(centre=(0.5, 0.0)))
    delta.set_record("path", 1, 11, shape=first, fill=(1, 0, 0, 1))
    delta.set_record("path", 2, 12, shape=second, fill=(0, 0, 1, 0.5))
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    leaves, _, _ = at(stream_of(delta), 80, 40)
    assert ids(leaves) == [(11, 0), (12, 0)]
