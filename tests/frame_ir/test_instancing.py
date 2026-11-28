"""Runs of like leaves reach a core drawer as instances records.

A run of leaves of one shape batches as it is; arrows, which CE builds one by
one, batch through their canonical shapes, a shaft and a tip a group of two.
Cairo's streams keep the leaves' own records. Drawn by the core, a batch is
its leaves' frame: byte for byte where the shapes repeat exactly, within the
parity tolerance where they were factored.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import (
    BLUE,
    DOWN,
    RED,
    RIGHT,
    UP,
    ArrowVectorField,
    Circle,
    Dot,
    Square,
    Triangle,
    VGroup,
    tempconfig,
)
from arachne.camera.camera import Camera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.frame_ir import instancing
from arachne.frame_ir.affine import RECONSTRUCTION, placed
from arachne.frame_ir.extract import FrameStream, batching
from arachne.frame_ir.instancing import SLOT_BASE, Instancing
from arachne.frame_ir.xf import Delta, Record, Shape, StreamState

SMALL = {"pixel_width": 192, "pixel_height": 108}


def _deltas(camera, groups, frames=1):
    stream = FrameStream()
    batched = Instancing()
    made, sent = [], []
    for index in range(frames):
        delta = stream.frame(camera, groups, t=Fraction(index, 15))
        made.append(delta)
        sent.append(batched(delta))
    return made, sent


def _instances(delta):
    return [(slot, r) for slot, r in delta.set if r.kind == "instances"]


def _field(n: int = 4):
    return ArrowVectorField(
        lambda p: np.array([np.sin(p[1]) + 0.3, np.cos(p[0]), 0.0]),
        x_range=[-3, 3, 6 / (n - 1)],
        y_range=[-2, 2, 4 / (n - 1)],
    )


def test_an_arrow_field_reaches_the_core_as_one_run_of_shafts_and_tips():
    with tempconfig(SMALL):
        camera = ThreeDCamera()
        field = _field()
        with batching() as tally:
            (made,), (sent,) = _deltas(camera, [("moving", [field])])
    arrows = len(field.submobjects)
    assert arrows == 16
    # Cairo's stream keeps every leaf's own record.
    assert not _instances(made)
    assert len(made.set) == 2 * arrows
    ((slot, run),) = _instances(sent)
    assert slot >= SLOT_BASE
    assert sent.order == [("moving", slot, 0)]
    assert tally == {"instances": 1, "factored": 2 * arrows}
    fields = run.fields
    assert len(fields["parts"]) == 1
    # Shaft and tip, leaf by leaf in CE's order.
    leaves = [s for _, s, _ in made.order]
    assert list(fields["objects"]) == leaves
    shapes = {s.key: s.data for s in sent.upload}
    made_shapes = {s.key: s.data for s in made.upload}
    records = dict(made.set)
    prototypes = [fields["shape"], fields["parts"][0]["shape"]]
    for i, leaf in enumerate(leaves):
        place = fields["placements"][i]
        # The prototype emitted, placed as the core places it: each member
        # within the bound of its leaf's points.
        back = placed(place, shapes[prototypes[i % 2]])
        own = made_shapes[records[leaf].fields["shape"]]
        assert np.abs(back - own).max() <= RECONSTRUCTION
    # CE thins each short arrow's shaft by its length: each member strokes at
    # its own width.
    widths = [records[leaf].fields["stroke_width"] for leaf in leaves]
    assert len(set(widths[::2])) > 1
    assert list(fields["instance_widths"]) == widths


def test_leaves_of_one_shape_batch_as_they_are():
    with tempconfig(SMALL):
        camera = Camera()
        squares = VGroup(*(Square(fill_opacity=0.5, color=c) for c in (RED, BLUE) * 3))
        with batching() as tally:
            (made,), (sent,) = _deltas(camera, [("moving", [squares])])
    ((_, run),) = _instances(sent)
    assert tally == {"instances": 1, "factored": 0}
    own = {r.fields["shape"] for _, r in made.set}
    assert own == {run.fields["shape"]}
    assert "parts" not in run.fields
    assert np.array_equal(run.fields["placements"], np.tile(np.eye(3, 4), (6, 1, 1)))


def test_a_foreign_leaf_splits_a_run_and_another_stroke_width_rides_in_it():
    with tempconfig(SMALL):
        camera = Camera()
        # Dots at different places factor: eight a side, the fewest that batch.
        dots = [Dot(0.4 * i * RIGHT + side) for side in (UP, DOWN) for i in range(8)]
        thick = Dot(2 * UP).set_stroke(width=6)
        group = VGroup(*dots[:8], Triangle(), *dots[8:], thick, Dot(2 * DOWN))
        (made,), (sent,) = _deltas(camera, [("moving", [group])])
    records = dict(sent.set)
    kinds = [records[slot].kind for _, slot, _ in sent.order]
    assert kinds == ["instances", "path", "instances"]
    first, last = (records[slot].fields for _, slot, _ in sent.order[::2])
    # One width throughout: none carried; the thick dot's, member by member.
    assert "instance_widths" not in first
    assert list(last["instance_widths"]) == [
        r.fields["stroke_width"]
        for r in (dict(made.set)[s] for _, s, _ in made.order[9:])
    ]


def test_a_family_too_small_to_batch_factored_keeps_its_records():
    with tempconfig(SMALL):
        camera = Camera()
        # Two squares of different sizes factor to one shape: left as records.
        squares = VGroup(Square(side_length=4), Square(side_length=4 / 3))
        with batching() as tally:
            (made,), (sent,) = _deltas(camera, [("moving", [squares])])
    assert tally == {"instances": 0, "factored": 0}
    assert sent.set == made.set
    assert sent.order == made.order


def test_a_held_frame_sends_nothing_and_a_camera_move_only_the_view():
    with tempconfig(SMALL):
        camera = ThreeDCamera()
        field = _field(3)
        stream, batched = FrameStream(), Instancing()
        groups = [("moving", [field])]
        batched(stream.frame(camera, groups))
        assert batched(stream.frame(camera, groups, t=Fraction(1, 15))).is_empty
        camera.set_theta(camera.get_theta() + 0.2)
        moved = batched(stream.frame(camera, groups, t=Fraction(2, 15)))
    assert moved.view is not None
    assert not moved.set
    assert not moved.remove
    assert not moved.upload
    assert not moved.order


def test_an_instances_record_expands_into_its_members_as_placed():
    with tempconfig(SMALL):
        camera = ThreeDCamera()
        (made,), (sent,) = _deltas(camera, [("moving", [_field(3)])])
    from arachne.frame_ir.xf import StreamState

    state = StreamState()
    state.apply(sent)
    members = instancing.expand(state)
    assert members.order == made.order
    held = StreamState()
    held.apply(made)
    for slot in (s for _, s, _ in made.order):
        a = members.shapes[members.records[slot].fields["shape"]].data
        b = held.shapes[held.records[slot].fields["shape"]].data
        assert np.abs(a - b).max() < 1e-4
        assert (
            members.records[slot].fields["stroke_width"]
            == (held.records[slot].fields["stroke_width"])
        )


# Drawn by the core.


def _lights():
    """The light models the engine selects (``core_light``, where it has the
    option), each drawn the same way by a batch and by its leaves; ``None``,
    the one it draws, where it has not.
    """
    from arachne import config

    return ["ce", "soft"] if "core_light" in config else [None]


def _drawn(camera, groups, *, instancing_on: bool, frames: int = 3):
    """Frames of ``groups`` the core draws with batching on or off, a 3D
    camera turning a little between them.
    """
    from arachne.frame_ir.core_painter import CorePainter

    painter = CorePainter(instancing=instancing_on)
    drawn = []
    for index in range(frames):
        if isinstance(camera, ThreeDCamera):
            camera.set_theta(-np.pi / 2 + 0.15 * index)
        drawn.append(painter.paint(camera, groups, t=Fraction(index, 15)).copy())
    return drawn, painter


def _scene_light(light):
    return {} if light is None else {"core_light": light}


@pytest.mark.parametrize("light", _lights())
def test_a_run_of_one_shape_between_coplanar_translucent_records_draws_its_leaves_bytes(
    light,
):
    pytest.importorskip("arachne_core")
    with tempconfig({**SMALL, **_scene_light(light)}):
        under = Square(side_length=3, fill_opacity=0.5, color=BLUE)
        dots = VGroup(
            *(
                Circle(radius=0.6, fill_opacity=0.6, color=c).set_stroke(width=5)
                for c in (RED, BLUE, RED, BLUE)
            )
        )
        over = Square(side_length=2, color=RED).set_stroke(width=9, opacity=0.6)
        groups = [("moving", [under, dots, over])]
        with batching() as tally:
            on, painter = _drawn(ThreeDCamera(), groups, instancing_on=True)
        off, _ = _drawn(ThreeDCamera(), groups, instancing_on=False)
    assert tally["instances"] >= 1
    assert tally["factored"] == 0
    for a, b in zip(on, off, strict=True):
        assert np.array_equal(a, b)
    assert (on[0] != on[0][0, 0]).any(axis=-1).sum() > 1000


@pytest.mark.parametrize("light", _lights())
def test_a_factored_arrow_field_draws_within_the_parity_tolerance_of_its_leaves(light):
    pytest.importorskip("arachne_core")
    with tempconfig({**SMALL, **_scene_light(light)}):
        field = _field(6)
        groups = [("moving", [field])]
        with batching() as tally:
            on, painter = _drawn(ThreeDCamera(), groups, instancing_on=True)
        off, _ = _drawn(ThreeDCamera(), groups, instancing_on=False)
        outlines = painter.outlines()
    assert tally["factored"] == 2 * len(field.submobjects)
    for a, b in zip(on, off, strict=True):
        off_by = np.abs(a.astype(int) - b.astype(int)).max(axis=-1)
        # The coverage resolve's bound, two levels, at a few edge pixels.
        assert off_by.max() <= 2, off_by.max()
        assert (off_by > 0).mean() < 0.01
    # Cairo reads the members as the core placed them.
    assert len(outlines) == 2 * len(field.submobjects)


# What a batch may not lose of its leaves.


def _spike():
    """:mod:`test_joins`'s sharp join: a closed spike whose tip turns 175
    degrees, which Cairo bevels on its own points.
    """
    from arachne import Polygon

    return Polygon([-3, -0.2, 0], [3, 0, 0], [-3, 0.2, 0], stroke_width=8)


def test_paths_whose_joins_are_decided_as_made_are_drawn_unbatched():
    """Two copies of one sharp-joined path repeat exactly, but each carries
    Cairo's join decisions on its points as made (XF 1.4), which an
    instances record cannot: they stay records, and the frame is the
    leaves' bytes.
    """
    pytest.importorskip("arachne_core")
    with tempconfig(SMALL):
        spikes = VGroup(_spike(), _spike().shift(0.3 * UP))
        groups = [("moving", [spikes])]
        with batching() as tally:
            on, painter = _drawn(Camera(), groups, instancing_on=True, frames=1)
        off, _ = _drawn(Camera(), groups, instancing_on=False, frames=1)
    assert tally == {"instances": 0, "factored": 0}
    carried = [r.fields.get("joins") for _, r in painter.last.set]
    assert len(carried) == 2
    assert all(j is not None for j in carried)
    assert np.array_equal(on[0], off[0])


def _labelled_squares():
    """A label kept facing the camera, at the origin, and ten filled squares
    nearer the camera, two of them over the label: each square a sheet of
    its own, which the label goes behind where they overlap it.
    """
    from arachne import OUT

    label = VGroup(Square(0.8, fill_opacity=1.0, color=RED, stroke_width=0))
    squares = VGroup(
        *(
            Square(0.5, fill_opacity=1.0, color=BLUE, stroke_width=0).shift(
                OUT + x * RIGHT + y * UP
            )
            for x, y in [(-0.25, 0.0), (0.25, 0.0)] + [(2 + 0.6 * k, 2.0) for k in range(8)]
        )
    )
    return label, squares


def test_a_frame_with_a_camera_facing_label_keeps_its_surfaces_own_sheets():
    """A batch is drawn from its first leaf's source, but a frame that draws a
    label reads each surface's sheet from its own: leaves of other sources
    stay records there, and the frame is the leaves' bytes.
    """
    pytest.importorskip("arachne_core")
    frames = []
    for on in (True, False):
        with tempconfig({"pixel_width": 384, "pixel_height": 216}):
            camera = ThreeDCamera()
            label, squares = _labelled_squares()
            camera.add_fixed_orientation_mobjects(label)
            with batching() as tally:
                drawn, _ = _drawn(
                    camera, [("moving", [squares, label])], instancing_on=on, frames=1
                )
        frames.append(drawn[0])
        if on:
            assert tally["instances"] == 0
    assert np.array_equal(*frames)


def test_without_a_label_leaves_of_other_sources_batch():
    with tempconfig(SMALL):
        _, squares = _labelled_squares()
        with batching() as tally:
            _deltas(ThreeDCamera(), [("moving", [squares])])
    assert tally["instances"] == 1


def _member_shapes(deltas):
    state = StreamState()
    for delta in deltas:
        state.apply(delta)
    members = instancing.expand(state)
    return {
        slot: members.shapes[record.fields["shape"]].data
        for slot, record in members.records.items()
    }


def test_a_stream_regridded_finer_is_batched_as_one_extracted_at_the_finer_grid():
    with tempconfig(SMALL):
        camera = ThreeDCamera()
        field = _field()
        stream, batched = FrameStream(), Instancing()
        groups = [("moving", [field])]
        made = stream.frame(camera, groups)
        coarse, fine = 2.0**-10, 2.0**-16
        sent = [batched(made, coarse)]
        sent.append(batched(stream.frame(camera, groups, t=Fraction(1, 15)), fine))
        fresh = Instancing()(made, fine)
    regridded, extracted = _member_shapes(sent), _member_shapes([fresh])
    assert regridded.keys() == extracted.keys()
    for slot, points in extracted.items():
        assert np.array_equal(regridded[slot], points)


def test_batching_tallies_nest_and_each_closes_its_own():
    from arachne.frame_ir.extract import _count_batching

    with batching() as outer:
        with batching() as inner:
            # Both open and equal: closing the inner must not close the outer.
            assert inner == outer
        _count_batching(instances=1, factored=2)
        with batching() as again:
            _count_batching(instances=1)
    assert inner == {"instances": 0, "factored": 0}
    assert again == {"instances": 1, "factored": 0}
    assert outer == {"instances": 2, "factored": 2}
    _count_batching(instances=5)
    assert outer == {"instances": 2, "factored": 2}


_VIEW = {
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


def _placed_leaves() -> Delta:
    """Six path records of one shape, each at a placement of its own, in
    colours of their own.
    """
    shape = Shape("path", Square(0.6).points)
    delta = Delta(view=dict(_VIEW), upload=[shape])
    for i in range(6):
        place = np.hstack(
            [np.eye(3) * (1 + 0.2 * i), np.array([[-4 + 1.5 * i], [0.4 * i - 1], [0.0]])]
        )
        fields = {
            "shape": shape.key,
            "place": place,
            "fill": np.array([[0.2 + 0.1 * i, 0.5, 0.9, 0.7]]),
            "stroke": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "stroke_width": 4.0,
            "joint": "auto",
            "cap": "auto",
            "space": "screen",
            "z_index": 0.0,
            "shade_in_3d": False,
        }
        delta.set.append((i + 1, Record("path", fields)))
    delta.order = [("moving", i + 1, 0) for i in range(6)]
    return delta


def test_placed_leaves_of_one_shape_batch_at_their_placements():
    delta = _placed_leaves()
    with batching() as tally:
        sent = Instancing()(delta)
    assert tally == {"instances": 1, "factored": 0}
    ((_, run),) = sent.set
    assert run.fields["shape"] == delta.set[0][1].fields["shape"]
    places = np.stack([r.fields["place"] for _, r in delta.set])
    assert np.array_equal(run.fields["placements"], places)


def test_placed_leaves_batched_draw_their_records_bytes():
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.core_painter import CorePainter

    frames = [
        CorePainter(instancing=on).draw(_placed_leaves()).copy() for on in (True, False)
    ]
    assert np.array_equal(*frames)
    assert (frames[0] != frames[0][0, 0]).any(axis=-1).sum() > 500
