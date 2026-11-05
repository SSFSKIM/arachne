"""The Python binding: building deltas, applying them, drawing, refusing, and XF files."""

from __future__ import annotations

import hashlib
import struct
from fractions import Fraction

import arachne_core as ac
import numpy as np
import pytest

FRAME = {"frame_width": 14.222222222222221, "frame_height": 8.0}


def square(half: float = 1.0) -> np.ndarray:
    corners = np.array(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
        dtype=float,
    )
    curves = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        curves += [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
    return np.array(curves)


def background_delta(width: int, height: int, rgba) -> ac.FrameDelta:
    delta = ac.FrameDelta(t=(0, 1), seq=0)
    delta.set_view(pixel_width=width, pixel_height=height, background=rgba, **FRAME)
    return delta


def test_the_version_is_xf_1():
    assert ac.XF_VERSION == (1, 2)


def test_mapping_centre_defaults_to_the_live_frame_center():
    centre = (0.5, -0.25, 0.125)
    delta = ac.FrameDelta()
    delta.set_view(pixel_width=4, pixel_height=4, frame_center=centre, **FRAME)
    # The view item starts at 33: centre is at 24, mapping_centre at 48.
    assert struct.unpack_from("<3d", delta.to_bytes(), 81) == centre
    explicit = ac.FrameDelta()
    explicit.set_view(
        pixel_width=4,
        pixel_height=4,
        frame_center=centre,
        mapping_centre=centre,
        **FRAME,
    )
    assert explicit.to_bytes() == delta.to_bytes()


def test_background_image_fill_sets_its_feature_and_is_cairo_only():
    delta = background_delta(4, 4, (0, 0, 0, 1))
    shape = delta.upload_path(square())
    image = delta.upload_image(np.ones((4, 4, 4), dtype=np.uint8))
    delta.set_record("path", 3, 33, shape=shape, fill=(1, 0, 0, 1), fill_image=image)
    assert ac.FEATURES == {"background_image": 1, "pre_projected": 2}
    stream = ac.Stream()
    stream.apply(ac.FrameDelta.from_bytes(delta.to_bytes()))
    assert stream.draw(4, 4).shape == (4, 4, 4)  # Not in the order yet.
    order = ac.FrameDelta(seq=1)
    order.set_order([("moving", 3, 0)])
    stream.apply(order)
    with pytest.raises(ac.Unsupported) as refused:
        stream.draw(4, 4)
    err = refused.value
    assert (err.code, err.kind, err.slot, err.object) == (
        "render.unsupported",
        "path",
        3,
        33,
    )
    assert "Cairo" in err.reason


def test_background_image_feature_cannot_be_set_without_its_field():
    delta = ac.FrameDelta()
    with pytest.raises(ac.XfError, match="fill_image"):
        delta.set_record("path", 1, 1, shape=bytes(16), features=1)


@pytest.mark.parametrize("array_like", [list, np.array])
def test_image_corners_are_four_independently_stored_points(array_like):
    delta = ac.FrameDelta()
    image = delta.upload_image(np.ones((1, 1, 4), dtype=np.uint8))
    corners = array_like([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 13]])
    delta.set_record("image", 1, 1, shape=image, corners=corners)
    assert ac.FrameDelta.from_bytes(delta.to_bytes()).to_bytes() == delta.to_bytes()
    stream = ac.Stream()
    stream.apply(delta)
    for invalid in (np.zeros((3, 3)), np.zeros((4, 2)), np.zeros((2, 6))):
        with pytest.raises((TypeError, ValueError)):
            delta.set_record("image", 1, 1, shape=image, corners=invalid)


def test_a_background_frame_is_ces_bytes_as_a_numpy_array():
    stream = ac.Stream()
    stream.apply(background_delta(8, 4, (0.5, 0.25, 1.0, 1.0)))
    frame = stream.draw(8, 4)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (4, 8, 4)
    assert frame.dtype == np.uint8
    assert frame.flags.writeable
    assert (frame == np.array([127, 63, 255, 255], dtype=np.uint8)).all()
    assert (stream.draw(8, 4, threads=4) == frame).all()


def test_apply_reports_its_changes_and_a_held_delta_is_empty():
    stream = ac.Stream()
    delta = background_delta(4, 4, (0, 0, 0, 1))
    key = delta.upload_path(square())
    delta.set_record("path", 1, 11, shape=key, fill=(1, 0, 0, 1))
    stats = stream.apply(delta)
    assert (stats.uploaded_shapes, stats.changed_records) == (1, 1)
    held = ac.FrameDelta(t=Fraction(1, 15), seq=1)
    assert held.is_empty
    assert held.t == (1, 15)
    stats = stream.apply(held)
    assert (stats.uploaded_shapes, stats.changed_records) == (0, 0)


def test_a_record_this_build_cannot_draw_refuses_the_whole_frame():
    stream = ac.Stream()
    delta = background_delta(4, 4, (0, 0, 0, 1))
    key = delta.upload_path(square())
    delta.set_record(
        "path",
        3,
        33,
        shape=key,
        fill=(1, 0, 0, 1),
        stroke=[(1, 1, 1, 1), (0, 0, 0, 1)],
        stroke_width=4.0,
        joint="round",
        cap="square",
    )
    delta.set_order([("moving", 3, 0)])
    stream.apply(delta)
    with pytest.raises(ac.Unsupported) as refused:
        stream.draw(4, 4)
    err = refused.value
    assert (err.kind, err.slot, err.object) == ("path", 3, 33)
    assert err.code == "render.unsupported"
    assert "render.unsupported" in str(err)
    assert "gradient" in err.reason


def test_a_path_draws_its_fill_and_stroke():
    width, height = 64, 36
    stream = ac.Stream()
    delta = background_delta(width, height, (0, 0, 0, 1))
    key = delta.upload_path(square(1.0))
    delta.set_record(
        "path",
        1,
        11,
        shape=key,
        fill=(1, 0, 0, 1),
        stroke=(0, 0, 1, 1),
        stroke_width=8.0,
    )
    delta.set_order([("moving", 1, 0)])
    stream.apply(delta)
    rgba, resolves = stream.draw_resolves(width, height)
    assert (rgba[height // 2, width // 2] == [255, 0, 0, 255]).all()
    # The square's left side, at x = -1, under its stroke 8 x 0.01 frame units wide.
    side = int((-1.0 + FRAME["frame_width"] / 2) * width / FRAME["frame_width"])
    assert rgba[height // 2, side, 2] > 0
    assert (resolves == ac.RESOLVES.index("coverage")).sum() > 0


def test_a_reserved_kind_is_refused_and_nothing_is_drawn():
    stream = ac.Stream()
    delta = background_delta(4, 4, (0, 0, 0, 1))
    delta.set_record("volume", 5, 55, payload=b"\x01\x02")
    delta.set_order([("static", 5, 0)])
    stream.apply(delta)
    with pytest.raises(ac.Unsupported) as refused:
        stream.draw(4, 4)
    assert (refused.value.kind, refused.value.slot) == ("volume", 5)
    assert "reserved" in refused.value.reason


def test_keys_are_blake2b_of_the_documented_canonical_bytes():
    points = square()
    key = ac.FrameDelta().upload_path(points)
    canonical = (
        bytes([1, 1, 1, 2]) + struct.pack("<QQ", len(points), 3) + points.tobytes()
    )
    assert key == hashlib.blake2b(canonical, digest_size=16).digest()
    rows = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
    key = ac.FrameDelta().upload_paint(rows)
    canonical = bytes([0x10, 1, 1, 2]) + struct.pack("<QQ", 3, 4) + rows.tobytes()
    assert key == hashlib.blake2b(canonical, digest_size=16).digest()
    # A mesh: positions f64, triangles u32 (dtype code 2), normals f64.
    positions = np.array([[0, 1.5, -2], [3.25, 0, 1], [0.5, 0.25, 0.125]])
    triangles = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint32)
    normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    key = ac.FrameDelta().upload_mesh(positions, triangles, normals=normals)
    canonical = (
        bytes([2, 3])
        + bytes([1, 2])
        + struct.pack("<QQ", 3, 3)
        + positions.tobytes()
        + bytes([2, 2])
        + struct.pack("<QQ", 2, 3)
        + triangles.tobytes()
        + bytes([1, 2])
        + struct.pack("<QQ", 3, 3)
        + normals.tobytes()
    )
    assert key == hashlib.blake2b(canonical, digest_size=16).digest()
    # An image: one u8 array (dtype code 3) of dims height, width, 4.
    rgba = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    key = ac.FrameDelta().upload_image(rgba)
    canonical = bytes([3, 1, 3, 3]) + struct.pack("<QQQ", 2, 3, 4) + rgba.tobytes()
    assert key == hashlib.blake2b(canonical, digest_size=16).digest()


def test_a_delta_that_references_what_the_stream_lacks_is_refused():
    stream = ac.Stream()
    stream.apply(background_delta(4, 4, (0, 0, 0, 1)))
    delta = ac.FrameDelta(seq=1)
    delta.set_record("path", 1, 1, shape=bytes(16))
    with pytest.raises(ac.XfError):
        stream.apply(delta)


def test_unknown_record_fields_and_kinds_are_errors():
    delta = ac.FrameDelta()
    with pytest.raises(TypeError):
        delta.set_record("path", 1, 1, shape=bytes(16), colour=(1, 0, 0, 1))
    with pytest.raises(ValueError):
        delta.set_record("teapot", 1, 1)
    with pytest.raises(ValueError):
        delta.set_record("path", 1, 1, shape=bytes(16), joint="sharp")


def canonical_sample() -> list[ac.FrameDelta]:
    """Three deltas built through the binding: every record kind, camera, light model form,
    background form, layer, space, joint and cap name, every paint form, every optional field
    both set and absent, the order both set and left as it was, and fields that could be swapped
    each given a value of its own. Reserved shapes cannot be uploaded from Python; the Rust
    sample covers them.
    """
    first = ac.FrameDelta(t=(0, 1), seq=0)
    first.set_view(
        pixel_width=16,
        pixel_height=9,
        background=(0.1, 0.2, 0.3, 0.9),
        camera="three_d",
        phi=1.25,
        theta=-0.75,
        gamma=0.125,
        focal_distance=19.5,
        zoom=1.5,
        exponential_projection=True,
        light_source=(-7, -9, 10),
        shading_factor=0.25,
        should_apply_shading=False,
        frame_center=(0.5, -0.25, 0.125),
        mapping_centre=(-0.75, 0.625, -0.125),
        cairo_line_width_multiple=0.02,
        use_z_index=False,
        light_model=(7, [0.25, 0.5]),
        **FRAME,
    )
    first.add_object(11, "Square#1", "scene.py:3")
    path = first.upload_path(square())
    wide = first.upload_path(square(2.0))
    mesh = first.upload_mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.uint32),
        normals=np.array([[0, 0, 1], [0, 0.6, 0.8], [0.6, 0, 0.8]], dtype=float),
    )
    bare_mesh = first.upload_mesh(
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float),
        np.array([[0, 2, 1]], dtype=np.uint32),
    )
    picture = first.upload_image(np.zeros((2, 3, 4), dtype=np.uint8))
    per_face = first.upload_paint(np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=float))
    face_strokes = first.upload_paint(
        np.array([[0.1, 0, 0.3, 1], [0, 0.9, 0.7, 0.5]], dtype=float)
    )
    per_point = first.upload_paint(np.ones((3, 4)))
    instance_strokes = first.upload_paint(
        np.array([[0.9, 0.1, 0.1, 1], [0.1, 0.9, 0.1, 1], [0.1, 0.1, 0.9, 1]])
    )
    gradient = first.upload_paint(
        np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0.5]], dtype=float)
    )
    cloud = first.upload_path(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
    first.set_record(
        "path",
        1,
        11,
        shape=path,
        place=np.eye(3, 4),
        blend=(wide, np.eye(3, 4) * 2, 0.375),
        reveal=(0.125, 0.875),
        fill=(1, 0, 0, 0.5),
        fill_image=picture,
        stroke=[(1, 1, 1, 1), (0, 0, 0, 1)],
        background_stroke=gradient,
        stroke_width=4.0,
        background_stroke_width=1.5,
        gradient=((-1, -1, 0), (1, 1, 0)),
        joint="round",
        cap="square",
        paint_b={
            "fill": (0, 0, 1, 1),
            "stroke": per_point,
            "background_stroke": (0.5, 0.5, 0.5, 1),
            "stroke_width": 2.0,
            "background_stroke_width": 1.0,
        },
        mix=0.25,
        dash=(0.5, 0.25, 0.0625),
        space="world",
        z_index=2.0,
        sort_point=(0.25, -0.5, 1),
        shade_in_3d=True,
        fixed_orientation_centre=(1, 2, 3),
        clip=(wide, np.eye(3, 4) * 3),
        opacity=0.5,
        layer=1,
    )
    first.set_record(
        "face_group",
        2,
        12,
        shape=wide,
        uv=[(0, 0), (1, 0)],
        grid=(2, 1),
        face_fill=per_face,
        face_stroke=face_strokes,
        stroke_width=0.5,
        joint="bevel",
        cap="round",
    )
    first.set_record(
        "points", 3, 13, shape=cloud, colors=per_point, radius=2.0, reveal=(0.25, 0.75)
    )
    first.set_record(
        "image",
        4,
        14,
        shape=picture,
        corners=((-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1.125, -1.25, 0.125)),
        resampling=2,
        image_opacity=0.75,
        space="overlay",
    )
    first.set_record(
        "image",
        5,
        15,
        corners=((-2, 1, 0.25), (2, 1, 0.5), (-2, -1, 0.75), (2.25, -1.125, 1)),
        resampling=2,
        image_opacity=0.625,
        sub_frame=(
            2,
            {
                "pixel_width": 8,
                "pixel_height": 4,
                "camera": "flat",
                "background": picture,
                "light_model": "ce",
                "frame_center": (0.25, -0.5, 0.75),
                "mapping_centre": (-0.125, 0.375, 0.875),
                **FRAME,
            },
            [("static", 1, 0)],
        ),
    )
    first.set_record(
        "instances",
        6,
        16,
        shape=path,
        placements=np.stack([np.eye(3, 4) * k for k in (1, 2, 3)]),
        instance_fill=per_point,
        instance_stroke=instance_strokes,
        objects=[21, 22, 23],
        stroke_width=1.0,
        joint="miter",
        cap="butt",
    )
    first.set_record(
        "mesh",
        7,
        17,
        shape=mesh,
        fill=(0.5, 0.5, 0.5, 1),
        blend=(bare_mesh, np.eye(3, 4), 0.5),
    )
    first.set_record(
        "volume", 8, 18, payload=b"\x09\x08\x07", features=0x0102030405060708
    )
    first.set_record("field", 9, 19, payload=b"\x06\x05")
    first.set_record("emission", 10, 20, payload=b"\x04")
    first.set_order([("static", 2, 1), ("moving", 1, 0), ("overlay", 4, 0)])
    second = ac.FrameDelta(t=(2, 15), seq=1)
    second.set_view(
        pixel_width=16,
        pixel_height=9,
        camera="moving",
        background=(0.1, 0.2, 0.3, 1.0),
        light_model="ce",
        **FRAME,
    )
    second.remove(7)
    second.release(mesh)
    second.set_order([])
    third = ac.FrameDelta(t=(1, 3), seq=2)
    third.set_record(
        "face_group",
        2,
        22,
        shape=wide,
        uv=[(1, 0), (0, 0)],
        grid=(2, 1),
        face_fill=per_face,
    )
    return [first, second, third]


# The sha256 of the canonical sample's file bytes. It catches any change to how the binding builds
# the sample or how XF writes it; the field-by-field layout is pinned by core/tests/layout.rs.
# XF streams are generated, never committed. Change the pin only when such a change is meant.
SAMPLE_SHA256 = "c8eb87be5dd0ac30938cbc821802494d19462c1ad7f2045c55659993898ec1fc"


def test_every_kind_can_be_built_and_survives_a_file(tmp_path):
    deltas = canonical_sample()
    file = tmp_path / "three.xf"
    ac.write_xf(file, deltas)
    assert hashlib.sha256(file.read_bytes()).hexdigest() == SAMPLE_SHA256
    back = ac.read_xf(file)
    assert [d.to_bytes() for d in back] == [d.to_bytes() for d in deltas]
    copy = tmp_path / "copy.xf"
    ac.write_xf(copy, back)
    assert copy.read_bytes() == file.read_bytes()
    assert ac.FrameDelta.from_bytes(deltas[0].to_bytes()).to_bytes() == (
        deltas[0].to_bytes()
    )
    stream = ac.Stream()
    for delta in back:
        stream.apply(delta)
    assert (stream.draw(16, 9) == np.array([25, 51, 76, 255], dtype=np.uint8)).all()


def test_instances_take_placements_of_shape_n_3_4_and_one_object_each():
    delta = ac.FrameDelta()
    key = delta.upload_path(square())
    table = delta.upload_paint(np.ones((2, 4)))
    keys = {"shape": key, "instance_fill": table, "instance_stroke": table}
    two = np.stack([np.eye(3, 4)] * 2)
    with pytest.raises(ac.XfError):
        delta.set_record("instances", 1, 1, placements=two, objects=[1, 2, 3], **keys)
    for bad in (np.zeros((2, 12)), np.zeros((2, 4, 3)), np.zeros((1, 6, 4))):
        with pytest.raises(ValueError, match=r"\(n, 3, 4\)"):
            delta.set_record("instances", 1, 1, placements=bad, objects=[1, 2], **keys)
    delta.set_record("instances", 1, 1, placements=two, objects=[1, 2], **keys)
    assert ac.FrameDelta.from_bytes(delta.to_bytes()).to_bytes() == delta.to_bytes()


def test_ces_light_model_written_as_code_0_is_ces():
    delta = ac.FrameDelta()
    delta.set_view(
        pixel_width=4,
        pixel_height=4,
        background=(1, 0, 0, 1),
        light_model=(0, []),
        **FRAME,
    )
    built, read_back = ac.Stream(), ac.Stream()
    built.apply(delta)
    read_back.apply(ac.FrameDelta.from_bytes(delta.to_bytes()))
    assert (built.draw(4, 4) == read_back.draw(4, 4)).all()


def test_bytes_that_are_not_xf_are_an_error(tmp_path):
    bad = tmp_path / "bad.xf"
    bad.write_bytes(b"not xf at all")
    with pytest.raises(ac.XfError):
        ac.read_xf(bad)


def quad_mesh(delta: ac.FrameDelta, z: float, half: float) -> bytes:
    positions = np.array(
        [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return delta.upload_mesh(positions, triangles)


def test_meshes_draw_by_depth_and_report_how_each_pixel_resolved():
    width, height = 96, 54
    delta = ac.FrameDelta(t=(0, 1), seq=0)
    delta.set_view(
        pixel_width=width,
        pixel_height=height,
        camera="three_d",
        phi=1.2,
        theta=-0.8,
        zoom=1.5,
        background=(0, 0, 0, 1),
        **FRAME,
    )
    near = quad_mesh(delta, z=0.5, half=1.5)
    far = quad_mesh(delta, z=-0.5, half=2.5)
    delta.set_record("mesh", 1, 11, shape=near, fill=(1, 0, 0, 1), space="world")
    delta.set_record("mesh", 2, 12, shape=far, fill=(0, 0, 1, 1), space="world")
    # The far quad is drawn last: painter's order would cover the near one.
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    stream = ac.Stream()
    stream.apply(delta)
    rgba, resolves = stream.draw_resolves(width, height)
    assert rgba.shape == (height, width, 4)
    assert (resolves.shape, resolves.dtype) == ((height, width), np.uint8)
    assert (stream.draw(width, height) == rgba).all()
    red = (rgba == [255, 0, 0, 255]).all(axis=2)
    blue = (rgba == [0, 0, 255, 255]).all(axis=2)
    assert red.sum() > 100
    assert blue.sum() > 100
    assert ac.RESOLVES == ("background", "fast", "coverage", "sample")
    counts = np.bincount(resolves.ravel(), minlength=4)
    assert counts.sum() == width * height
    assert counts[ac.RESOLVES.index("fast")] <= (red | blue).sum()
    assert counts[ac.RESOLVES.index("sample")] > 0
    for threads in (2, 3, 4):
        again, again_resolves = stream.draw_resolves(width, height, threads=threads)
        assert (again == rgba).all()
        assert (again_resolves == resolves).all()


def test_soft_diffuse_light_is_light_model_1_with_ambient_and_wrap():
    delta = ac.FrameDelta()
    delta.set_view(
        pixel_width=4,
        pixel_height=4,
        camera="three_d",
        light_model=(1, [0.25, 0.5]),
        **FRAME,
    )
    stream = ac.Stream()
    stream.apply(ac.FrameDelta.from_bytes(delta.to_bytes()))
    assert stream.draw(4, 4).shape == (4, 4, 4)
    bad = ac.FrameDelta()
    bad.set_view(pixel_width=4, pixel_height=4, light_model=(1, [1.5, 0.5]), **FRAME)
    with pytest.raises(ac.XfError):
        ac.Stream().apply(bad)


def test_decode_refuses_both_background_image_bit_field_disagreements():
    for image in (None, bytes(range(16))):
        delta = ac.FrameDelta()
        delta.set_record("path", 1, 1, shape=bytes(16), fill_image=image)
        raw = bytearray(delta.to_bytes())
        # A lone record starts at 53; its feature bits start at 54.
        raw[54] ^= 1
        with pytest.raises(ac.XfError, match="fill_image"):
            ac.FrameDelta.from_bytes(bytes(raw))


def test_fill_image_preserves_other_feature_bits_and_cannot_be_released_in_use():
    delta = ac.FrameDelta()
    shape = delta.upload_path(square())
    image = delta.upload_image(np.ones((1, 1, 4), dtype=np.uint8))
    delta.set_record("path", 1, 1, shape=shape, fill_image=image, features=16)
    header = bytes([1]) + struct.pack("<QII", 17, 1, 1)
    assert header in delta.to_bytes()
    stream = ac.Stream()
    stream.apply(delta)
    for bad_key in (bytes(16), shape):
        bad = ac.FrameDelta()
        bad.set_record("path", 2, 2, shape=shape, fill_image=bad_key)
        with pytest.raises(ac.XfError):
            stream.apply(bad)
    release = ac.FrameDelta()
    release.release(image)
    with pytest.raises(ac.XfError, match="still uses"):
        stream.apply(release)
    release.remove(1)
    stream.apply(release)


def test_sub_frame_placement_rank_and_self_capture():
    corners = [[-1, 1, 0], [1, 1, 0], [-1, -1, 0], [1.125, -1.25, 0.125]]
    view = dict(pixel_width=4, pixel_height=4, frame_center=(1, 2, 3), **FRAME)
    delta = background_delta(4, 4, (0, 0, 0, 1))
    delta.set_record(
        "image",
        1,
        11,
        corners=corners,
        resampling=2,
        image_opacity=0.625,
        sub_frame=(0, view, [("moving", 1, 0)]),
    )
    delta.set_record("image", 2, 12, corners=corners, sub_frame=(0, view, []))
    stream = ac.Stream()
    stream.apply(ac.FrameDelta.from_bytes(delta.to_bytes()))
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    with pytest.raises(ac.XfError, match="rank"):
        stream.apply(delta)
    delta.set_order([("moving", 1, 0), ("overlay", 2, 0)])
    stream.apply(delta)
    with pytest.raises(ac.Unsupported):
        stream.draw(4, 4)
    # Even a valid image shape is forbidden on a sub-frame.
    bad = ac.FrameDelta()
    image = bad.upload_image(np.ones((1, 1, 4), dtype=np.uint8))
    bad.set_record(
        "image", 3, 13, shape=image, corners=corners, sub_frame=(1, view, [])
    )
    with pytest.raises(ac.XfError, match="no shape"):
        stream.apply(bad)


@pytest.mark.parametrize("kind", ["path", "points", "image"])
def test_pre_projected_sets_its_feature_and_is_drawn_from_its_tail(kind):
    delta = background_delta(4, 4, (0, 0, 0, 1))
    shape = delta.upload_path(square())
    # As many points as the shape: a points record's colours are one per point.
    points = delta.upload_path(square(0.5) + [0, 0, 1])
    body = {}
    if kind == "points":
        body = {"colors": delta.upload_paint(np.ones((16, 4))), "radius": 2}
    elif kind == "image":
        shape = delta.upload_image(np.ones((1, 1, 4), dtype=np.uint8))
        body = {"corners": np.zeros((4, 3))}
    delta.set_record(kind, 3, 33, shape=shape, pre_projected={"points": points}, **body)
    fields = delta.contents()["set"][0][3]
    assert fields["features"] == 2
    assert fields["pre_projected"] == {
        "points": points,
        "gradient": None,
        "fill": None,
        "stroke": None,
        "background_stroke": None,
    }
    assert delta.to_bytes()[:4] == struct.pack("<HH", 1, 1)
    stream = ac.Stream()
    stream.apply(ac.FrameDelta.from_bytes(delta.to_bytes()))
    stream.draw(4, 4)  # Not drawn yet.
    order = ac.FrameDelta()
    order.set_order([("moving", 3, 0)])
    stream.apply(order)
    if kind == "image":
        with pytest.raises(ac.Unsupported, match="image records"):
            stream.draw(4, 4)
    else:
        assert stream.draw(4, 4).shape == (4, 4, 4)


def test_pre_projected_bit_needs_its_field_and_preserves_other_bits():
    delta = ac.FrameDelta()
    with pytest.raises(ac.XfError, match="pre_projected"):
        delta.set_record("path", 1, 1, features=2)
    delta.set_record("path", 1, 1, features=16, pre_projected={"points": bytes(16)})
    assert delta.contents()["set"][0][3]["features"] == 18
    assert ac.FrameDelta().to_bytes()[:4] == struct.pack("<HH", 1, 0)


def pre_projected_sample():
    delta = ac.FrameDelta(t=(2, 15), seq=3)
    original = delta.upload_path(square())
    points = delta.upload_path([[1.25, -2.5, 0.125], [3.5, 4.75, -0.625]])
    table = delta.upload_paint(
        [[0.125, 0.25, 0.375, 0.5], [0.625, 0.75, 0.875, 1], [0.9, 0.8, 0.7, 0.6]]
    )
    delta.set_record(
        "path",
        11,
        111,
        shape=original,
        pre_projected={
            "points": points,
            "gradient": ((-1.25, -2.5, -3.75), (4.125, 5.25, 6.5)),
            "fill": (0.15, 0.25, 0.35, 0.45),
            "stroke": [(0.55, 0.65, 0.75, 0.85), (0.95, 0.05, 0.125, 0.375)],
            "background_stroke": table,
        },
    )
    return delta


PRE_PROJECTED_SHA256 = (
    "df6b52077f443b5db275ed07837ef7235a089298af0c410c770150ddf796983c"
)


def test_pre_projected_canonical_sample_is_pinned_separately(tmp_path):
    delta = pre_projected_sample()
    assert delta.version == (1, 1)
    file = tmp_path / "pre-projected.xf"
    ac.write_xf(file, [delta])
    assert hashlib.sha256(file.read_bytes()).hexdigest() == PRE_PROJECTED_SHA256
    assert ac.read_xf(file)[0].to_bytes() == delta.to_bytes()


def snapped_sample():
    delta = ac.FrameDelta(t=(4, 15), seq=5)
    delta.set_view(
        pixel_width=32,
        pixel_height=18,
        background=(0.0, 0.0, 0.0, 1.0),
        input_grid=1 / 32768,
        **FRAME,
    )
    shape = delta.upload_path(square())
    delta.set_record(
        "path", 12, 112, shape=shape, fill=(0.2, 0.4, 0.6, 1.0), planes=[1, 6, 0]
    )
    return delta


SNAPPED_SHA256 = "a1a03ee745bf337c03e6d01285f31ad11e70ceb20d4bc936e787222bc49d5483"


def test_snapped_canonical_sample_is_pinned_separately(tmp_path):
    delta = snapped_sample()
    assert delta.version == (1, 2)
    file = tmp_path / "snapped.xf"
    ac.write_xf(file, [delta])
    assert hashlib.sha256(file.read_bytes()).hexdigest() == SNAPPED_SHA256
    contents = ac.read_xf(file)[0].contents()
    assert contents["view"]["input_grid"] == 1 / 32768
    assert contents["set"][0][3]["planes"] == (1, 6, 0)


def test_an_unsnapped_delta_carries_neither_field_and_stays_below_1_2():
    delta = ac.FrameDelta(t=(0, 1), seq=0)
    delta.set_view(pixel_width=4, pixel_height=4, **FRAME)
    delta.set_record("path", 1, 1, shape=delta.upload_path(square()))
    contents = delta.contents()
    assert "input_grid" not in contents["view"]
    assert "planes" not in contents["set"][0][3]
    assert delta.version == (1, 0)


def test_the_planes_of_a_record_are_found_as_the_drawer_finds_them():
    delta = ac.FrameDelta(t=(0, 1), seq=0)
    delta.set_view(pixel_width=8, pixel_height=8, camera="three_d", **FRAME)
    flat = delta.upload_path(square())
    bent = square()
    bent[5, 2] = 0.25
    on_a_line = np.array([[t, 2 * t, 3 * t] for t in np.linspace(0, 1, 8)])
    delta.set_record("path", 1, 1, shape=flat, fill=(1, 1, 1, 1))
    delta.set_record("path", 2, 2, shape=delta.upload_path(bent), fill=(1, 1, 1, 1))
    delta.set_record("path", 3, 3, shape=delta.upload_path(on_a_line))
    stream = ac.Stream()
    stream.apply(delta)
    # A flat square shares z: its axis plane (z, code 6); bent: a curve (0); a line too.
    assert stream.planes([1, 2, 3, 9]) == {1: [6], 2: [0], 3: [0]}
