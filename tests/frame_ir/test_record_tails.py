"""What follows a record's planes and normals on the wire, as the Cairo route
reads it back: the route draws a frame from the bytes the core parsed
(``IrCairoPainter.draw``), so whatever a newer writer appends there must
reach it intact and change nothing it draws.

A path record's ``joins`` (XF 1.4) are the core's: Cairo decides each join
from the points it strokes, so the route carries them and draws the frame it
draws without them. A face group's face ends and widths (XF 1.3) are its own
lists, never joins, and a frame without joins is still written at 1.3.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arachne import (
    DEGREES,
    PI,
    TAU,
    Create,
    Polygon,
    Surface,
    ThreeDCamera,
    tempconfig,
)
from arachne.frame_ir import joins
from arachne.frame_ir.extract import FrameStream
from arachne.frame_ir.painter import IrCairoPainter
from arachne.frame_ir.xf import Delta, Record

pytest.importorskip("arachne_core")

SMALL = {"pixel_width": 192, "pixel_height": 108}


def _camera():
    camera = ThreeDCamera()
    camera.set_phi(70 * DEGREES)
    camera.set_theta(-30 * DEGREES)
    return camera


def _frame() -> Delta:
    """A surface half created (its faces of different lengths: face ends)
    beside a stroked path that turns back sharply (a join Cairo decides).
    """
    surface = Surface(
        lambda u, v: np.array(
            [2 * np.cos(u) * np.cos(v), 1.2 * np.cos(u) * np.sin(v), 0.8 * np.sin(u)]
        ),
        u_range=[-PI / 2, PI / 2],
        v_range=[0, TAU],
        resolution=(8, 16),
        fill_opacity=0.32,
        stroke_width=0.5,
    )
    create = Create(surface)
    create.begin()
    create.interpolate(0.5)
    path = Polygon([-3, -1, 0], [1, 0.2, 0], [-3, 0.6, 0], stroke_width=6)
    with tempconfig(SMALL):
        return FrameStream().frame(
            _camera(), [("moving", [surface, path])], roots=[surface, path]
        )


def _with_joins(delta: Delta) -> Delta:
    """``delta`` with each path record carrying a decision for every join of
    its points, as a 1.4 writer may: each cubic joined into, and mitred.
    """
    shapes = {shape.key: shape for shape in delta.upload}
    found = []
    for slot, record in delta.set:
        fields = dict(record.fields)
        if record.kind == "path":
            cubics = len(shapes[fields["shape"]].data) // 4
            fields["joins"] = (joins.JOINED | joins.MITRED,) * cubics
        found.append((slot, Record(record.kind, fields)))
    return replace(delta, set=found)


def _read(delta: Delta) -> tuple[np.ndarray, dict[str, dict]]:
    """The frame the Cairo route draws from ``delta``, and the fields of the
    one record of each kind it read back.
    """
    with tempconfig(SMALL):
        painter = IrCairoPainter()
        pixels = painter.draw(delta)
    found: dict[str, dict] = {}
    for record in painter.state.records.values():
        assert record.kind not in found, record.kind
        found[record.kind] = record.fields
    return pixels, found


def _sent(delta: Delta, kind: str) -> dict:
    [fields] = [r.fields for _, r in delta.set if r.kind == kind]
    return fields


def test_a_path_s_joins_reach_the_cairo_route_and_draw_nothing_of_their_own():
    plain = _frame()
    joined = _with_joins(plain)
    assert joined.to_core().version == (1, 4)
    expected, _ = _read(plain)
    drawn, read = _read(joined)
    np.testing.assert_array_equal(drawn, expected)
    assert tuple(read["path"]["joins"]) == _sent(joined, "path")["joins"]
    # The face group's tail is its face lists, read as such, and no joins.
    assert "joins" not in read["face_group"]
    np.testing.assert_array_equal(
        read["face_group"]["face_ends"], _sent(plain, "face_group")["face_ends"]
    )


def test_a_frame_whose_records_carry_no_joins_is_written_at_1_3():
    """The same frame without joins: its face group's face ends need 1.3,
    and nothing read back is taken for joins.
    """
    plain = _frame()
    assert "face_ends" in _sent(plain, "face_group")
    assert plain.to_core().version == (1, 3)
    _, read = _read(plain)
    assert not any("joins" in fields for fields in read.values())
    np.testing.assert_array_equal(
        read["face_group"]["face_ends"], _sent(plain, "face_group")["face_ends"]
    )
