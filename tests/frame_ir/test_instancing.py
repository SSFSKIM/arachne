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
from arachne.frame_ir.extract import FrameStream, batching
from arachne.frame_ir.instancing import (
    MOST_FACTORED_POINTS,
    RECONSTRUCTION,
    SLOT_BASE,
    Instancing,
    factor,
)

SMALL = {"pixel_width": 192, "pixel_height": 108}


def _turned(points: np.ndarray, angle: float, scale: float, shift) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s, 0.2], [s, c, -0.1], [0.3, 0.1, 0.9]]) * scale
    return points @ m.T + np.asarray(shift, dtype=float)


def _snapped(values: np.ndarray, grid: float = 2.0**-16) -> np.ndarray:
    return np.round(values / grid) * grid + 0.0


def test_an_affine_image_of_a_path_factors_into_its_canonical_shape():
    triangle = Triangle().points
    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [3.0, 1.5, 0.0]]
    )
    for path in (triangle, line, Circle().points):
        found = factor(path)
        assert found is not None
        for angle, scale, shift in ((0.7, 0.4, (1, 2, 3)), (-2.1, 3.0, (-4, 0.5, 0))):
            image = _turned(path, angle, scale, shift)
            other = factor(image)
            assert other is not None
            # One canonical shape, once the snap has had its say.
            assert np.array_equal(_snapped(other.shape), _snapped(found.shape))
            back = other.shape @ other.place[:, :3].T + other.place[:, 3]
            assert np.abs(back - image).max() <= RECONSTRUCTION


def test_a_path_too_long_or_not_mapped_back_closely_enough_is_not_factored(
    monkeypatch,
):
    assert factor(np.zeros((MOST_FACTORED_POINTS + 1, 3))) is None
    assert factor(np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])) is None
    # A circle turned and scaled maps back off by its rounding, a few ulps:
    # within the bound, factored; under a bound of none, not.
    circle = _turned(Circle().points, 0.731, 0.37, (0.2, -1.3, 0.4))
    found = factor(circle)
    assert found is not None
    back = found.shape @ found.place[:, :3].T + found.place[:, 3]
    assert 0 < np.abs(back - circle).max() <= RECONSTRUCTION
    monkeypatch.setattr(instancing, "RECONSTRUCTION", 0.0)
    assert factor(circle) is None
    # A path at one point is placed there.
    monkeypatch.undo()
    dot = factor(np.full((4, 3), 2.5))
    assert dot is not None
    assert not dot.shape.any()


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
        prototype = shapes[prototypes[i % 2]]
        back = prototype @ place[:, :3].T + place[:, 3]
        own = made_shapes[records[leaf].fields["shape"]]
        # The prototype is snapped to the frame's grid already.
        assert np.abs(back - own).max() < 1e-4
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
