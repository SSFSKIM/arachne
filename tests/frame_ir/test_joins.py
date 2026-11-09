"""A stroked path's miter joins, decided as Cairo decides them on CE's points
(:mod:`arachne.frame_ir.joins`), carried to the core as the record's
``joins`` (XF 1.4).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from arachne import Polygon, Scene
from arachne.frame_ir import joins
from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.xf import Delta, Record, Shape

from ..timeline.conftest import render

#: A view of 854 x 480 pixels over CE's frame: 60.046875 pixels a unit.
VIEW = {
    "pixel_width": 854,
    "pixel_height": 480,
    "frame_width": 14.222222222222221,
    "frame_height": 7.993754879000781,
    "frame_center": (0.0, 0.0, 0.0),
    "mapping_centre": (0.0, 0.0, 0.0),
}


def _frame(view, fixed):
    """Frame-unit points whose device position is ``fixed`` (in 1/256 px)."""
    xx, yy, x0, y0 = joins.matrix(view)
    return [((x / 256.0 - x0) / xx, (y / 256.0 - y0) / yy, 0.0) for x, y in fixed]


@pytest.mark.parametrize(
    ("held", "turn"),
    [
        # Two reversals of a traced path, as Cairo held their points: arriving
        # along (2, 0) and leaving along (-6, 1) in 1/256 px (170.5 degrees),
        # and along (2, 0) then (-5, 0) (180 degrees).
        (
            [
                (208560, 64119),
                (208562, 64119),
                (208565, 64118),
                (208567, 64118),
                (208567, 64118),
                (208561, 64119),
                (208555, 64119),
                (208550, 64119),
            ],
            170.5,
        ),
        (
            [
                (209041, 64790),
                (209044, 64790),
                (209046, 64790),
                (209048, 64790),
                (209048, 64790),
                (209043, 64790),
                (209037, 64791),
                (209031, 64791),
            ],
            180.0,
        ),
    ],
    ids=["170.5", "180"],
)
def test_cairo_bevels_a_reversal_past_the_limit_on_its_rounded_slopes(held, turn):
    points = _frame(VIEW, held)
    assert [joins._fixed(joins.matrix(VIEW), p) for p in points] == held
    assert joins.joins(points, VIEW) == (0, joins.JOINED)


def test_a_device_coordinate_rounds_to_the_nearest_256th_ties_to_even():
    m = (1.0, 1.0, 0.0, 0.0)
    assert joins._fixed(m, (2.5 / 256, 3.5 / 256)) == (2, 4)
    assert joins._fixed(m, (-2.5 / 256, 2.4999 / 256)) == (-2, 2)


def test_a_join_within_the_limit_mitres_and_one_past_it_bevels():
    # In 1/256 px: 168 degrees mitres (miter 9.6 half widths), 169 bevels.
    for degrees, expected in (
        (168.0, joins.JOINED | joins.MITRED),
        (169.0, joins.JOINED),
    ):
        r = np.radians(degrees)
        out = (round(4096 * np.cos(r)), round(4096 * np.sin(r)))
        held = [
            (0, 0),
            (1000, 0),
            (3000, 0),
            (4096, 0),
            (4096, 0),
            (4096 + out[0], out[1]),
            (4096 + 2 * out[0], 2 * out[1]),
            (4096 + 3 * out[0], 3 * out[1]),
        ]
        assert joins.joins(_frame(VIEW, held), VIEW) == (0, expected), degrees


def test_a_path_whose_joins_the_snap_cannot_move_carries_none():
    # A square's corners: 90 degrees, on arms a third of a side long.
    held = [
        (0, 0),
        (3000, 0),
        (6000, 0),
        (9000, 0),
        (9000, 0),
        (9000, 3000),
        (9000, 6000),
        (9000, 9000),
    ]
    assert joins.joins(_frame(VIEW, held), VIEW) is None


def test_the_decisions_are_the_same_on_every_machine():
    """Integers and IEEE doubles through CPython (no fused multiply-add):
    the joins of many random paths, digested, are one value on every CPU.
    """
    rng = random.Random(20261109)
    digest = hashlib.sha256()
    for _ in range(3000):
        points, at = [], (rng.uniform(-7, 7), rng.uniform(-4, 4), 0.0)
        for _ in range(rng.randrange(2, 6)):
            arm = rng.choice((1.0, 1e-3, 2e-4))
            end = (at[0] + rng.uniform(-1, 1), at[1] + rng.uniform(-1, 1), 0.0)
            points += [
                at,
                (at[0] + rng.uniform(-arm, arm), at[1] + rng.uniform(-arm, arm), 0.0),
                (end[0] + rng.uniform(-arm, arm), end[1] + rng.uniform(-arm, arm), 0.0),
                end,
            ]
            at = end
        if rng.random() < 0.3:
            points[-1] = points[0]
        digest.update(repr(joins.joins(points, VIEW)).encode())
    assert digest.hexdigest() == DIGEST


#: :func:`test_the_decisions_are_the_same_on_every_machine`'s digest.
DIGEST = "43ab6e9f2a25f7f12742d4b3218c500a296133b16b1f65b425cc05956198d246"


class Spike(Scene):
    """A closed spike whose tip turns 175 degrees: Cairo bevels it."""

    def construct(self):
        self.add(Polygon([-3, -0.2, 0], [3, 0, 0], [-3, 0.2, 0], stroke_width=8))
        self.wait(0.1)


def test_the_core_is_handed_cairo_s_decision_at_a_sharp_join(monkeypatch):
    call = Canonical.__call__
    carried = []

    def checking(self, delta):
        found = call(self, delta)
        carried.extend(
            record.fields["joins"]
            for _, record in found.set
            if record.kind == "path" and "joins" in record.fields
        )
        return found

    monkeypatch.setattr(Canonical, "__call__", checking)
    render(Spike, 15, draw_path="core", pixel_width=192, pixel_height=108)
    assert carried
    # The tip is the join into the second side: joined, bevelled; the others
    # mitre (the first byte is the closing join).
    assert all(j[1::3] and set(j) <= {0, 1, 3} for j in carried)
    assert any(joins.JOINED in j for j in carried)


_FLAT = {
    "pixel_width": 192,
    "pixel_height": 108,
    "frame_width": 14.222222222222221,
    "frame_height": 8.0,
    "frame_center": (0.0, 0.0, 0.0),
    "mapping_centre": (0.0, 0.0, 0.0),
    "cairo_line_width_multiple": 0.01,
    "use_z_index": True,
    "background": (0.0, 0.0, 0.0, 1.0),
    "camera": "flat",
}


def _polyline(*corners) -> np.ndarray:
    """Straight cubics through ``corners`` in CE's layout."""
    return np.concatenate(
        [np.linspace(a, b, 4) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    )


def _stroked(shape: Shape, **fields) -> Record:
    return Record(
        "path",
        {
            "shape": shape.key,
            "fill": None,
            "stroke": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "stroke_width": 4.0,
            "joint": "auto",
            "cap": "auto",
            "space": "screen",
            "z_index": 0.0,
            "shade_in_3d": False,
            **fields,
        },
    )


def _handed(deltas: list[Delta]) -> list[dict]:
    """Each delta through Canonical: the joins each record it sets is handed."""
    snapper = Canonical()
    return [
        {slot: record.fields.get("joins") for slot, record in snapper(d).set}
        for d in deltas
    ]


def test_a_revealed_path_s_joins_are_those_of_the_part_drawn():
    """A path doubling back on itself in its second half: revealed to half,
    the part drawn has no sharp join, and none is carried; whole, the
    reversal is.
    """
    shape = Shape(
        "path", _polyline([-3.0, 0, 0], [0.0, 0, 0], [3.0, 0, 0], [0.0, 0.1, 0])
    )
    half = Delta(seq=0, view=dict(_FLAT), upload=[shape])
    half.set = [(1, _stroked(shape, reveal=(0.0, 0.5)))]
    whole = Delta(seq=0, view=dict(_FLAT), upload=[shape])
    whole.set = [(1, _stroked(shape))]
    assert _handed([half])[0][1] is None
    # The first corner runs straight on (no join); the second turns back.
    assert _handed([whole])[0][1] == (0, 0, joins.JOINED)


def test_a_view_that_moves_a_held_path_s_joins_sets_it_again():
    """A held path whose reversal turns on arms of a few 1/256 px: panned by
    a fraction of a pixel, its rounded slopes, and so Cairo's decision, can
    change; the record is set again with the joins of the new view, and a
    view that moves none sets nothing.
    """
    unit = 1 / (60.046875 * 256)  # 1/256 px in frame units at 854 px
    view = dict(
        _FLAT, pixel_width=854, pixel_height=480, frame_height=7.993754879000781
    )
    j = np.array([1.0, 0.5, 0.0])
    start, end = j - [1.0, 0, 0], j + [-1.0, 0.05, 0]
    points = np.array(
        [
            start,
            start + [0.3, 0, 0],
            j - [6.3 * unit, -1.4 * unit, 0],
            j,
            j,
            j + [-2.4 * unit, 0.7 * unit, 0],
            end + [0.3, 0, 0],
            end,
        ]
    )
    shape = Shape("path", points)
    first = Delta(seq=0, view=dict(view), upload=[shape])
    first.set = [(1, _stroked(shape))]
    seen = {joins.joins(points, view)}
    for k in range(1, 64):
        shift = (k * unit / 7, 0.0, 0.0)
        moved = dict(view, frame_center=shift, mapping_centre=shift)
        decided = joins.joins(points, moved)
        if decided not in seen:
            break
    else:
        pytest.fail("no pan moved the decision")
    handed = _handed([first, Delta(seq=1, view=dict(view)), Delta(seq=2, view=moved)])
    assert handed[0][1] is not None
    assert 1 not in handed[1], "a view that moves no join sets nothing"
    assert handed[2][1] == decided


def test_a_path_with_a_point_not_finite_carries_none():
    """The drawer omits such a path: nothing is decided, nothing raised."""
    points = _frame(
        VIEW, [(0, 0), (1, 0), (2, 0), (3, 0), (3, 0), (1, 0), (0, 1), (-1, 1)]
    )
    points[5] = (float("nan"), 0.0, 0.0)
    assert joins.joins(points, VIEW) is None


def test_a_placed_path_or_one_kept_facing_the_camera_is_left_to_the_drawer():
    """The drawer places those otherwise than the points as made: no joins
    are carried; the same path unplaced carries its reversal's.
    """
    shape = Shape(
        "path", _polyline([-3.0, 0, 0], [0.0, 0, 0], [3.0, 0, 0], [0.0, 0.1, 0])
    )
    place = np.hstack([np.eye(3), np.zeros((3, 1))])
    place[1, 1] = 2.0
    cases = {
        "plain": _stroked(shape),
        "placed": _stroked(shape, place=place),
        "facing": _stroked(shape, fixed_orientation_centre=(0.0, 0.0, 0.0)),
    }
    handed = {}
    for name, record in cases.items():
        delta = Delta(seq=0, view=dict(_FLAT), upload=[shape])
        delta.set = [(1, record)]
        handed[name] = _handed([delta])[0][1]
    assert handed == {"plain": (0, 0, joins.JOINED), "placed": None, "facing": None}
