"""Decoded XF values can rebuild a delta without changing a floating-point bit."""

from __future__ import annotations

import struct

import arachne_core as ac
import numpy as np
import pytest
from test_core_binding import FRAME, canonical_sample


def rebuild(contents: dict) -> ac.FrameDelta:
    delta = ac.FrameDelta(t=contents["t"], seq=contents["seq"])
    if contents["view"] is not None:
        delta.set_view(**contents["view"])
    for ref, stable_id, source in contents["objects"]:
        delta.add_object(ref, stable_id, source)
    for kind, key, data in contents["upload"]:
        if kind == "path":
            back = delta.upload_path(data)
        elif kind == "mesh":
            back = delta.upload_mesh(**data)
        elif kind == "image":
            back = delta.upload_image(data)
        else:
            raise AssertionError(f"no Python uploader for {kind}")
        assert back == key
    for key, rows in contents["upload_paints"]:
        assert delta.upload_paint(rows) == key
    for key in contents["release"]:
        delta.release(key)
    for kind, slot, obj, fields in contents["set"]:
        delta.set_record(kind, slot, obj, **fields)
    for slot in contents["remove"]:
        delta.remove(slot)
    if contents["order"] is not None:
        delta.set_order(contents["order"])
    for slot, source, facing_group in contents["source"]:
        delta.set_source(slot, source=source, facing_group=facing_group)
    return delta


def test_canonical_contents_rebuild_the_decoded_wire_byte_for_byte():
    for delta in canonical_sample() + [ac.FrameDelta(t=(-7, 9), seq=13)]:
        decoded = ac.FrameDelta.from_bytes(delta.to_bytes())
        contents = decoded.contents()
        assert set(contents) == {
            "version",
            "t",
            "seq",
            "view",
            "objects",
            "upload",
            "upload_paints",
            "release",
            "set",
            "remove",
            "order",
            "source",
        }
        assert contents["source"] == []
        assert contents["version"] == (1, 0)
        assert rebuild(contents).to_bytes() == delta.to_bytes()


def test_a_record_s_source_rebuilds_from_its_contents_byte_for_byte():
    delta = ac.FrameDelta(t=(1, 15), seq=2)
    delta.set_record(
        "path",
        1,
        1,
        source=(((9, "a.Glÿph"), (4, "a.Label")), ((9, "a.Glÿph"),)),
        facing_group=4,
    )
    delta.set_record("points", 2, 2, colors=b"\xa1" * 16, radius=2.0, facing_group=4)
    delta.set_order([("moving", 1, 0), ("moving", 2, 0)])
    delta.set_source(5, source=(((3, "a.Label"),),), facing_group=3)
    decoded = ac.FrameDelta.from_bytes(delta.to_bytes())
    contents = decoded.contents()
    assert contents["version"] == (1, 5)
    first = contents["set"][0][3]
    assert all(
        type(path) is tuple and all(type(node) is tuple for node in path)
        for path in first["source"]
    )
    assert rebuild(contents).to_bytes() == delta.to_bytes()


def test_contents_values_have_the_documented_types_and_shapes():
    first = ac.FrameDelta.from_bytes(canonical_sample()[0].to_bytes()).contents()
    view = first["view"]
    assert view["background"] == (0.1, 0.2, 0.3, 0.9)
    assert view["camera"] == "three_d"
    assert view["frame_center"] != view["mapping_centre"]
    assert view["light_model"] == (7, [0.25, 0.5])
    assert first["objects"] == [(11, "Square#1", "scene.py:3")]
    for kind, key, data in first["upload"]:
        assert isinstance(key, bytes)
        assert len(key) == 16
        if kind == "path":
            assert data.dtype == np.float64
            assert data.shape[1:] == (3,)
        elif kind == "mesh":
            assert set(data) == {"positions", "triangles", "normals"}
            assert data["positions"].dtype == np.float64
            assert data["triangles"].dtype == np.uint32
            if data["normals"] is not None:
                assert data["normals"].dtype == np.float64
        elif kind == "image":
            assert data.dtype == np.uint8
            assert data.shape == (2, 3, 4)
    for key, rows in first["upload_paints"]:
        assert isinstance(key, bytes)
        assert rows.dtype == np.float64
        assert rows.shape[1:] == (4,)
    path = first["set"][0][3]
    assert path["features"] == ac.FEATURES["background_image"]
    assert len(path["fill_image"]) == 16
    assert path["fill"].shape == (1, 4)
    assert path["stroke"].shape == (2, 4)
    assert isinstance(path["background_stroke"], bytes)
    assert isinstance(path["gradient"], tuple)
    assert len(path["gradient"]) == 6
    assert (path["joint"], path["cap"], path["space"]) == ("round", "square", "world")
    display = first["set"][4][3]
    assert display["corners"].shape == (4, 3)
    rank, nested_view, order = display["sub_frame"]
    assert rank == 2
    assert nested_view["camera"] == "flat"
    assert isinstance(nested_view["background"], bytes)
    assert order == [("static", 1, 0)]
    assert first["set"][1][3]["uv"].dtype == np.uint32
    assert first["set"][5][3]["placements"].shape == (3, 3, 4)
    assert first["set"][7][3]["payload"] == b"\x09\x08\x07"


@pytest.mark.parametrize(
    "bits",
    [
        0x8000000000000000,
        1,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0x7FF0000000000012,
        0x7FF8000000000012,
        0xFFF0000000000000,
    ],
)
def test_f64_bits_survive_contents_in_scalars_arrays_and_optional_fields(bits):
    value = struct.unpack("<d", struct.pack("<Q", bits))[0]
    delta = ac.FrameDelta()
    delta.set_view(
        pixel_width=4,
        pixel_height=4,
        frame_center=(value, value, value),
        mapping_centre=(value, value, value),
        background=(value,) * 4,
        cairo_line_width_multiple=value,
        camera="three_d",
        phi=value,
        theta=value,
        gamma=value,
        focal_distance=value,
        zoom=value,
        light_source=(value,) * 3,
        shading_factor=value,
        light_model=(7, [value]),
        frame_width=value,
        frame_height=value,
    )
    shape = delta.upload_path(np.full((4, 3), value))
    table = delta.upload_paint(np.full((3, 4), value))
    delta.set_record(
        "path",
        1,
        1,
        shape=shape,
        place=np.full((3, 4), value),
        blend=(shape, np.full((3, 4), value), value),
        reveal=(value, value),
        fill=np.full((1, 4), value),
        stroke=np.full((2, 4), value),
        background_stroke=table,
        stroke_width=value,
        background_stroke_width=value,
        gradient=((value,) * 3,) * 2,
        paint_b={
            "fill": (value,) * 4,
            "stroke_width": value,
            "background_stroke_width": value,
        },
        mix=value,
        dash=(value,) * 3,
        z_index=value,
        sort_point=(value,) * 3,
        fixed_orientation_centre=(value,) * 3,
        clip=(shape, np.full((3, 4), value)),
        opacity=value,
    )
    image = delta.upload_image(np.ones((1, 1, 4), dtype=np.uint8))
    delta.upload_mesh(
        np.full((3, 3), value),
        np.array([[0, 1, 2]], dtype=np.uint32),
        normals=np.full((3, 3), value),
    )
    delta.set_record(
        "image",
        2,
        2,
        shape=image,
        corners=np.full((4, 3), value),
        image_opacity=value,
    )
    delta.set_record("points", 3, 3, shape=shape, colors=table, radius=value)
    delta.set_record(
        "instances",
        4,
        4,
        shape=shape,
        placements=np.full((3, 3, 4), value),
        instance_fill=table,
        instance_stroke=table,
        objects=[11, 12, 13],
    )
    delta.set_record(
        "image",
        5,
        5,
        corners=np.full((4, 3), value),
        image_opacity=value,
        sub_frame=(
            7,
            {
                "pixel_width": 4,
                "pixel_height": 4,
                "frame_center": (value,) * 3,
                **FRAME,
            },
            [],
        ),
    )
    decoded = ac.FrameDelta.from_bytes(delta.to_bytes())
    assert rebuild(decoded.contents()).to_bytes() == delta.to_bytes()


def test_empty_arrays_keep_their_column_dimensions():
    delta = ac.FrameDelta()
    delta.upload_path(np.empty((0, 3)))
    delta.upload_mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32))
    delta.upload_image(np.empty((0, 2, 4), dtype=np.uint8))
    delta.upload_paint(np.empty((0, 4)))
    assert (
        rebuild(ac.FrameDelta.from_bytes(delta.to_bytes()).contents()).to_bytes()
        == delta.to_bytes()
    )


def test_contents_are_copies_not_mutable_views_of_the_delta():
    delta = canonical_sample()[0]
    before = delta.to_bytes()
    contents = delta.contents()
    contents["upload"][0][2][:] = 42
    contents["upload_paints"][0][1][:] = 42
    contents["set"][0][3]["place"][:] = 42
    assert delta.to_bytes() == before


def test_opaque_shapes_and_unknown_records_are_read_back_without_invented_fields():
    import hashlib

    payload = b"\x10\x20\x30"
    shapes = []
    expected = []
    for code, name in [(4, "volume"), (5, "field"), (99, "kind99")]:
        canonical = bytes([code, 1, 3, 1]) + struct.pack("<Q", len(payload)) + payload
        key = hashlib.blake2b(canonical, digest_size=16).digest()
        shape = bytes([code]) + key + payload
        shapes.append(struct.pack("<I", len(shape)) + shape)
        expected.append((name, key, payload))
    record = struct.pack("<BQII", 200, 5, 42, 7) + b"\xff\xfe"
    raw = (
        struct.pack("<HHqqQ", 1, 0, 0, 1, 0)
        + b"\0"  # Header, no view.
        + struct.pack("<II", 0, 3)
        + b"".join(shapes)  # Objects, shapes.
        + struct.pack("<IIII", 0, 0, 1, len(record))
        + record  # Tables, release, set.
        + struct.pack("<I", 0)
        + b"\0"  # Remove, no order.
    )
    contents = ac.FrameDelta.from_bytes(raw).contents()
    assert contents["upload"] == expected
    assert contents["set"] == [
        ("kind200", 42, 7, {"features": 5, "payload": b"\xff\xfe"})
    ]


def test_soft_diffuse_view_and_empty_instance_arrays_rebuild():
    delta = ac.FrameDelta()
    delta.set_view(pixel_width=4, pixel_height=4, light_model=(1, [0.25, 0.5]), **FRAME)
    shape = delta.upload_path(np.empty((0, 3)))
    table = delta.upload_paint(np.empty((0, 4)))
    delta.set_record(
        "instances",
        1,
        1,
        shape=shape,
        placements=np.empty((0, 3, 4)),
        instance_fill=table,
        instance_stroke=table,
        objects=[],
    )
    contents = ac.FrameDelta.from_bytes(delta.to_bytes()).contents()
    assert contents["view"]["light_model"] == (1, [0.25, 0.5])
    assert contents["set"][0][3]["placements"].shape == (0, 3, 4)
    assert rebuild(contents).to_bytes() == delta.to_bytes()


@pytest.mark.parametrize("gradient_form", ["flat", "pairs", "none"])
def test_pre_projected_read_back_preserves_all_float_bits_and_paint_forms(
    gradient_form,
):
    bits = np.array(
        [
            0x8000000000000000,
            1,
            0x7FF0000000000012,
            0x7FF8000000000034,
            0x7FEFFFFFFFFFFFFF,
            0xFFF0000000000000,
        ],
        dtype=np.uint64,
    )
    values = bits.view(np.float64)
    delta = ac.FrameDelta()
    shape = delta.upload_path(np.zeros((4, 3)))
    points = delta.upload_path(values.reshape(2, 3))
    table = delta.upload_paint(np.resize(values, (3, 4)))
    gradient = (
        None
        if gradient_form == "none"
        else tuple(values)
        if gradient_form == "flat"
        else tuple(map(tuple, values.reshape(2, 3)))
    )
    delta.set_record(
        "path",
        1,
        1,
        shape=shape,
        pre_projected={
            "points": points,
            "gradient": gradient,
            "fill": values[:4],
            "stroke": np.resize(values, (2, 4)),
            "background_stroke": table,
        },
    )
    contents = ac.FrameDelta.from_bytes(delta.to_bytes()).contents()
    tail = contents["set"][0][3]["pre_projected"]
    assert set(tail) == {"points", "gradient", "fill", "stroke", "background_stroke"}
    assert isinstance(tail["points"], bytes)
    assert len(tail["points"]) == 16
    assert tail["points"] == points
    if gradient_form == "none":
        assert tail["gradient"] is None
    else:
        assert isinstance(tail["gradient"], tuple)
        assert len(tail["gradient"]) == 6
        assert np.array(tail["gradient"]).view(np.uint64).tolist() == bits.tolist()
    assert (tail["fill"].dtype, tail["fill"].shape) == (np.float64, (1, 4))
    assert (tail["stroke"].dtype, tail["stroke"].shape) == (np.float64, (2, 4))
    assert tail["fill"].view(np.uint64).ravel().tolist() == bits[:4].tolist()
    assert (
        tail["stroke"].view(np.uint64).ravel().tolist() == np.resize(bits, 8).tolist()
    )
    assert tail["background_stroke"] == table
    assert rebuild(contents).to_bytes() == delta.to_bytes()
    tail["fill"][:] = 42
    tail["stroke"][:] = 42
    assert (
        delta.to_bytes()
        == rebuild(ac.FrameDelta.from_bytes(delta.to_bytes()).contents()).to_bytes()
    )
