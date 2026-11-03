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


def test_the_version_is_xf_0():
    assert ac.XF_VERSION == (0, 0)


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
    """Two deltas built through the binding, every record kind and most fields set."""
    first = background_delta(16, 9, (0.1, 0.2, 0.3, 1.0))
    first.add_object(11, "Square#1", "scene.py:3")
    path = first.upload_path(square())
    wide = first.upload_path(square(2.0))
    mesh = first.upload_mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.uint32),
        normals=np.array([[0, 0, 1]] * 3, dtype=float),
    )
    picture = first.upload_image(np.zeros((2, 3, 4), dtype=np.uint8))
    per_face = first.upload_paint(np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=float))
    per_point = first.upload_paint(np.ones((3, 4)))
    cloud = first.upload_path(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
    first.set_record(
        "path",
        1,
        11,
        shape=path,
        place=np.eye(3, 4),
        blend=(wide, np.eye(3, 4), 0.5),
        reveal=(0.0, 0.5),
        fill=(1, 0, 0, 0.5),
        background_stroke=per_point,
        background_stroke_width=2.0,
        gradient=((-1, 0, 0), (1, 0, 0)),
        paint_b={"fill": (0, 0, 1, 1), "stroke_width": 1.0},
        mix=0.25,
        dash=(0.5, 0.5, 0.0),
        space="world",
        z_index=2.0,
        sort_point=(0, 0, 1),
        shade_in_3d=True,
        fixed_orientation_centre=(1, 2, 3),
        clip=(wide, np.eye(3, 4)),
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
        face_stroke=per_face,
    )
    first.set_record("points", 3, 13, shape=cloud, colors=per_point, radius=2.0)
    first.set_record(
        "image",
        4,
        14,
        shape=picture,
        corners=((-1, 1, 0), (1, 1, 0), (-1, -1, 0)),
        resampling=3,
        image_opacity=0.75,
    )
    first.set_record(
        "image",
        5,
        15,
        sub_frame=(
            {
                "pixel_width": 8,
                "pixel_height": 4,
                "camera": "three_d",
                "phi": 1.0,
                "theta": -0.5,
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
        placements=np.stack([np.eye(3, 4)] * 3),
        instance_fill=per_point,
        instance_stroke=per_point,
        objects=[21, 22, 23],
    )
    first.set_record("mesh", 7, 17, shape=mesh, fill=(0.5, 0.5, 0.5, 1))
    first.set_order([("static", 2, 1), ("moving", 1, 0), ("overlay", 4, 0)])
    second = ac.FrameDelta(t=(2, 15), seq=1)
    second.remove(7)
    second.release(mesh)
    second.set_order([])
    return [first, second]


# The sha256 of the canonical sample's file bytes. XF streams are generated, never committed, so
# any drift in the layout or in the builder fails this pin. Change it only when that is meant.
SAMPLE_SHA256 = "69460afc44bb4402dfe33933193789894fbef59a393478341155fb4a97c071d3"


def test_every_kind_can_be_built_and_survives_a_file(tmp_path):
    deltas = canonical_sample()
    file = tmp_path / "two.xf"
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


def test_bytes_that_are_not_xf_are_an_error(tmp_path):
    bad = tmp_path / "bad.xf"
    bad.write_bytes(b"not xf at all")
    with pytest.raises(ac.XfError):
        ac.read_xf(bad)
