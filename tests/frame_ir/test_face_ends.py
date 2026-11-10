"""A surface stays one face group while it is partly drawn.

``Create`` cuts each face of a surface to the curves drawn so far
(``VMobject.pointwise_become_partial``), so mid-play the faces hold points of
different counts, a face not yet begun its four points at one spot; ``Write``
strokes each face at its own phase's width. Neither makes a face anything but a
face of its surface: the group carries each face's end in its points
(``face_ends``) and its stroke width (``face_widths``), and each part is drawn
from its own points as CE draws that face.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from arachne import (
    DEGREES,
    PI,
    TAU,
    CapStyleType,
    Create,
    Cylinder,
    Sphere,
    Surface,
    ThreeDCamera,
    Write,
    tempconfig,
)
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import FrameStream
from arachne.frame_ir.live import live, live_painter
from arachne.frame_ir.xf import Record
from arachne.mobject.three_d.three_dimensions import ThreeDVMobject

pytest.importorskip("arachne_core")

SMALL = {"pixel_width": 192, "pixel_height": 108}
LARGE = {"pixel_width": 427, "pixel_height": 240}


def _ellipsoid(resolution=(16, 32)):
    return Surface(
        lambda u, v: np.array(
            [2 * np.cos(u) * np.cos(v), 1.2 * np.cos(u) * np.sin(v), 0.8 * np.sin(u)]
        ),
        u_range=[-PI / 2, PI / 2],
        v_range=[0, TAU],
        resolution=resolution,
        fill_opacity=0.32,
        stroke_width=0.5,
    )


MAKERS = {"ellipsoid": _ellipsoid, "sphere": Sphere, "cylinder": Cylinder}


def _at(surface, animation, alpha):
    """``surface`` as ``animation`` leaves it at ``alpha``."""
    play = animation(surface)
    play.begin()
    play.interpolate(alpha)
    return surface


def _faces(surface):
    return [
        face
        for face in surface.submobjects
        if type(face) is ThreeDVMobject and hasattr(face, "u_index")
    ]


def _undrawn(face):
    points = np.asarray(face.points)
    return len(points) == 4 and np.ptp(points, axis=0).max() == 0


def _camera():
    camera = ThreeDCamera()
    camera.set_phi(70 * DEGREES)
    camera.set_theta(-30 * DEGREES)
    return camera


def _painted(surface, size=SMALL):
    """``surface`` drawn on the core from its record: the painter, its frame."""
    with tempconfig(size):
        painter = CorePainter()
        frame = painter.paint(_camera(), [("moving", [surface])], roots=[surface])
    return painter, frame


def _ce(surface, size=SMALL):
    """``surface`` as CE's own camera draws it."""
    with tempconfig(size):
        camera = _camera()
        camera.reset()
        camera.capture_mobjects([surface])
        return np.array(camera.pixel_array)


def _group(painter, surface):
    """The painter's one face group, its slot and record: ``surface``'s uv
    faces drawn as one group, and its other parts (a cylinder's caps) alone
    records of their own.
    """
    records = painter.state.records
    [(slot, record)] = [(s, r) for s, r in records.items() if r.kind == "face_group"]
    others = len(surface.submobjects) - len(_faces(surface))
    assert len(records) == 1 + others
    return slot, record


def _leaf_pixels(painter, slot, part):
    """The pixels at which the core reports leaf ``(slot, part)``."""
    view = painter.state.view
    box = (0, 0, view["pixel_width"], view["pixel_height"])
    return {
        at
        for at, pixel in painter.contributors(box).items()
        if any(leaf.object == slot and leaf.part == part for leaf in pixel.leaves)
    }


def _changed(a, b):
    """The pixels at which two frames differ, as ``(x, y)``."""
    ys, xs = np.nonzero(np.any(a != b, axis=2))
    return set(zip(xs.tolist(), ys.tolist(), strict=True))


def _near(pixels):
    """``pixels`` and their eight neighbours."""
    return {
        (x + dx, y + dy) for x, y in pixels for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    }


@pytest.mark.parametrize(
    ("make", "alpha"),
    [("ellipsoid", 0.5), ("ellipsoid", 0.998), ("sphere", 0.5), ("cylinder", 0.5)],
)
def test_a_partly_created_surface_is_one_face_group_of_its_faces_own_lengths(
    make, alpha
):
    surface = _at(MAKERS[make](), Create, alpha)
    counts = [len(face.points) for face in _faces(surface)]
    assert len(set(counts)) > 1, "faces of different lengths"
    painter, _ = _painted(surface)
    slot, record = _group(painter, surface)
    np.testing.assert_array_equal(record.fields["face_ends"], np.cumsum(counts))
    parts = sorted(part for _, s, part in painter.state.order if s == slot)
    assert parts == list(range(len(counts)))


@pytest.mark.parametrize("make", ["ellipsoid", "sphere"])
def test_a_surface_mid_write_is_one_face_group_each_face_at_its_own_width(make):
    surface = _at(MAKERS[make](), Write, 0.5)
    faces = _faces(surface)
    widths = [float(face.get_stroke_width()) for face in faces]
    assert len(set(widths)) > 1, "faces of different widths"
    painter, _ = _painted(surface)
    _, record = _group(painter, surface)
    np.testing.assert_array_equal(record.fields["face_widths"], widths)
    np.testing.assert_array_equal(
        record.fields["face_ends"], np.cumsum([len(face.points) for face in faces])
    )


#: The core's frame of the ellipsoid undrawn and whole, as it drew them
#: before a group carried its faces' ends or widths; the whole one as it
#: draws once a stroke under 0.8 px is stroked at a share of its half width
#: (its faces' strokes are about 0.07 px wide here).
FRAMES = {
    0.0: "fc779d8b138f5c8d0e3a0a2a4e901c6a4f6d3945ee4cdc7a63edee0f2ef02456",
    1.0: "8126cfcdd1d7b97f1c017c1abe26c12f419a34b94f1a1265c23cfe61a9f89835",
}


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_a_surface_undrawn_or_whole_draws_the_bytes_it_drew(alpha):
    """Its faces alike in length and width, the group carries neither list
    and draws the frame it drew.
    """
    surface = _at(_ellipsoid(), Create, alpha)
    painter, frame = _painted(surface)
    _, record = _group(painter, surface)
    assert not {"face_ends", "face_widths"} & record.fields.keys()
    assert hashlib.sha256(frame.tobytes()).hexdigest() == FRAMES[alpha]


def test_a_surface_whose_faces_differ_in_width_alone_carries_their_widths():
    """At the end of ``Write`` every face is whole while the last differ in
    width: the group carries their widths and no face ends.
    """
    surface = _at(Sphere(), Write, 0.998)
    faces = _faces(surface)
    assert len({len(face.points) for face in faces}) == 1
    painter, _ = _painted(surface)
    _, record = _group(painter, surface)
    assert "face_ends" not in record.fields
    np.testing.assert_array_equal(
        record.fields["face_widths"], [float(face.get_stroke_width()) for face in faces]
    )


def _live_record(surface):
    """The face group a live capture of ``surface`` on the core holds."""
    with tempconfig(SMALL):
        camera = _camera()
        with live(camera, Fraction(0), draw_path="core", roots=[surface]):
            camera.reset()
            camera.capture_mobjects([surface])
        painter = live_painter(camera).painter(camera)
        [record] = painter.stream._held.values()
    return record


@pytest.mark.parametrize("animation", [Create, Write])
def test_a_partly_drawn_surface_is_the_same_face_group_live_or_recorded(animation):
    surface = _at(_ellipsoid(), animation, 0.5)
    with tempconfig(SMALL):
        camera = _camera()
        delta = FrameStream().frame(camera, [("moving", [surface])], roots=[surface])
    [(_, recorded)] = delta.set
    made = _live_record(surface)
    assert recorded.kind == made.kind == "face_group"
    fields = {k: v for k, v in recorded.fields.items() if k != "sort_point"}
    assert made.fields.keys() == fields.keys()
    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(made.fields[key], value, err_msg=key)
        else:
            assert made.fields[key] == value, key


def _one_cubic(surface, index, straight):
    """``surface`` with its face ``index`` cut to its first cubic, made a
    straight segment between its ends when ``straight``.
    """
    face = _faces(surface)[index]
    points = np.asarray(face.points)[:4].copy()
    if straight:
        a, b = points[0], points[3]
        points = np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])
    face.points = points
    return face


#: A face of the coarse ellipsoid turned towards the camera.
CUT = 44


def test_a_part_of_one_straight_cubic_is_its_stroke_alone():
    """A part cut to one straight cubic has no fill area: CE fills nothing
    there, and the core reports it as its stroke alone (its frame is the
    same with the part's fill transparent).
    """
    surface = _ellipsoid((8, 16))
    face = _one_cubic(surface, CUT, straight=True)
    surface.set_stroke(width=2)
    painter, filled = _painted(surface, LARGE)
    slot, _ = _group(painter, surface)
    assert _leaf_pixels(painter, slot, CUT), "its stroke is drawn"
    ce_filled = _ce(surface, LARGE)
    face.set_fill(opacity=0)
    _, unfilled = _painted(surface, LARGE)
    assert _changed(ce_filled, _ce(surface, LARGE)) == set()
    np.testing.assert_array_equal(filled, unfilled)


def test_a_part_of_one_curved_cubic_fills_and_strokes_where_ce_does():
    """A face's edge is straight in uv but curved on the surface: cut to its
    first cubic, CE fills the sliver between the curve and its chord and
    strokes the curve. The core's part covers no pixel CE's fill leaves
    untouched, and its stroke covers CE's stroke.
    """
    surface = _ellipsoid((8, 16))
    face = _one_cubic(surface, CUT, straight=False)
    surface.set_stroke(width=2)
    face.set_stroke(opacity=0)
    ce_fill = _ce(surface, LARGE)
    painter, _ = _painted(surface, LARGE)
    slot, _ = _group(painter, surface)
    core_fill = _leaf_pixels(painter, slot, CUT)
    face.set_fill(opacity=0)
    ce_neither = _ce(surface, LARGE)
    ce_fill_px = _changed(ce_fill, ce_neither)
    assert ce_fill_px, "CE fills the sliver"
    assert core_fill, "the core fills the sliver"
    assert core_fill <= _near(ce_fill_px)
    assert ce_fill_px <= _near(core_fill)

    face.set_stroke(opacity=1)
    ce_stroke_px = _changed(_ce(surface, LARGE), ce_neither)
    painter, _ = _painted(surface, LARGE)
    core_stroke = _leaf_pixels(painter, slot, CUT)
    assert ce_stroke_px
    assert core_stroke
    assert core_stroke <= _near(ce_stroke_px)
    assert ce_stroke_px <= _near(core_stroke)


@pytest.mark.parametrize(
    "cap", [CapStyleType.AUTO, CapStyleType.BUTT, CapStyleType.SQUARE]
)
def test_an_undrawn_face_paints_nothing_and_is_no_leaf(cap):
    """A face Create has not begun is four points at one spot: CE's camera
    traces one closed subpath of no length, which no fill covers and a butt
    or square cap does not stroke. The core draws nothing of it and reports
    no leaf of it.
    """
    surface = _at(_ellipsoid(), Create, 0.5)
    for face in surface.submobjects:
        face.cap_style = cap
    undrawn = {i for i, face in enumerate(_faces(surface)) if _undrawn(face)}
    assert undrawn
    painter, frame = _painted(surface)
    slot, _ = _group(painter, surface)
    view = painter.state.view
    box = (0, 0, view["pixel_width"], view["pixel_height"])
    reported = {
        leaf.part
        for pixel in painter.contributors(box).values()
        for leaf in pixel.leaves
        if leaf.object == slot
    }
    assert reported, "the drawn faces are reported"
    assert not reported & undrawn
    # CE draws the same frame without them.
    kept = [face for face in surface.submobjects if not _undrawn(face)]
    with_them = _ce(surface)
    surface.submobjects = kept
    assert _changed(with_them, _ce(surface)) == set()


def test_an_undrawn_face_under_a_round_cap_is_ce_s_dot_with_its_reason():
    """Under a round cap Cairo strokes a closed subpath of no length as a dot
    of the stroke's width. The core paints that dot where CE does, and
    reports its leaf with the reason that keeps it out of any order read
    from the pixel; every other leaf carries none.
    """
    from arachne.frame_ir.contributors import ROUND_CAP_DOT

    surface = _at(_ellipsoid(), Create, 0.5)
    surface.set_stroke(width=4)
    for face in surface.submobjects:
        face.cap_style = CapStyleType.ROUND
    undrawn = {i for i, face in enumerate(_faces(surface)) if _undrawn(face)}
    painter, _ = _painted(surface, LARGE)
    slot, _ = _group(painter, surface)
    view = painter.state.view
    found = painter.contributors((0, 0, view["pixel_width"], view["pixel_height"]))
    dots = set()
    for at, pixel in found.items():
        for leaf in pixel.leaves:
            dot = leaf.object == slot and leaf.part in undrawn
            assert leaf.reason == (ROUND_CAP_DOT if dot else None)
            if dot:
                dots.add(at)
    with_them = _ce(surface, LARGE)
    surface.submobjects = [face for face in surface.submobjects if not _undrawn(face)]
    ce_dots = _changed(with_them, _ce(surface, LARGE))
    assert ce_dots
    assert dots
    assert ce_dots <= _near(dots)
    assert dots <= _near(ce_dots)


#: A surface whose group carries neither list, its face ends alone, its face
#: widths alone, and both: its maker, the animation and alpha, and the lists.
LISTS = {
    "neither": (_ellipsoid, Create, 1.0, set()),
    "ends alone": (_ellipsoid, Create, 0.5, {"face_ends"}),
    "widths alone": (Sphere, Write, 0.998, {"face_widths"}),
    "both": (Sphere, Write, 0.5, {"face_ends", "face_widths"}),
}


def _uniform(delta):
    """``delta`` with each face group stating both lists, an absent one in
    its uniform form: equal runs, the record's width.
    """
    from dataclasses import replace

    shapes = {shape.key: shape for shape in delta.upload}
    found = []
    for slot, record in delta.set:
        fields = dict(record.fields)
        if record.kind == "face_group":
            count = len(fields["uv"])
            size = len(shapes[fields["shape"]].data) // count
            fields.setdefault(
                "face_ends", np.arange(1, count + 1, dtype=np.uint32) * size
            )
            fields.setdefault(
                "face_widths", np.full(count, float(fields["stroke_width"]))
            )
        found.append((slot, Record(record.kind, fields)))
    return replace(delta, set=found)


@pytest.mark.parametrize("case", LISTS)
def test_each_absent_face_list_is_read_as_uniform_whatever_the_other(case):
    """The frame_ir readers read each list a face group may carry on its own:
    the Cairo route draws the group from XF as CE's camera draws the
    surface, and the core draws it as the group stating both lists.
    """
    from arachne.frame_ir.painter import IrCairoPainter

    make, animation, alpha, lists = LISTS[case]
    surface = _at(make(), animation, alpha)
    with tempconfig(SMALL):
        delta = FrameStream().frame(_camera(), [("moving", [surface])], roots=[surface])
    [record] = [r for _, r in delta.set if r.kind == "face_group"]
    assert {"face_ends", "face_widths"} & record.fields.keys() == lists
    with tempconfig(SMALL):
        drawn = IrCairoPainter().draw(delta)
    np.testing.assert_array_equal(drawn, _ce(surface))
    sent, stated = CorePainter(), CorePainter()
    with tempconfig(SMALL):
        np.testing.assert_array_equal(sent.draw(delta), stated.draw(_uniform(delta)))


def _emptied(delta):
    """``delta`` with each face group stating each list it does not carry as
    an empty one, as a Python writer may.
    """
    from dataclasses import replace

    found = []
    for slot, record in delta.set:
        fields = dict(record.fields)
        if record.kind == "face_group":
            fields.setdefault("face_ends", [])
            fields.setdefault("face_widths", [])
        found.append((slot, Record(record.kind, fields)))
    return replace(delta, set=found)


@pytest.mark.parametrize("case", LISTS)
def test_an_empty_face_list_is_the_absent_one_to_every_reader(case):
    """An empty list is a list not carried, to every reader of a Python
    delta: the record carries what it would without it, the Cairo route
    still draws CE's surface, the core the frame the record without it
    draws, and the writer's snap reads the same faces.
    """
    from arachne.frame_ir import canonical
    from arachne.frame_ir.painter import IrCairoPainter

    make, animation, alpha, _ = LISTS[case]
    surface = _at(make(), animation, alpha)
    with tempconfig(SMALL):
        delta = FrameStream().frame(_camera(), [("moving", [surface])], roots=[surface])
    emptied = _emptied(delta)
    [(record, as_sent)] = [
        (r, e)
        for (_, r), (_, e) in zip(delta.set, emptied.set, strict=True)
        if r.kind == "face_group"
    ]
    assert as_sent.fields.keys() == record.fields.keys()
    assert canonical._way(as_sent) == canonical._way(record)
    with tempconfig(SMALL):
        np.testing.assert_array_equal(IrCairoPainter().draw(emptied), _ce(surface))
        np.testing.assert_array_equal(
            CorePainter().draw(emptied), CorePainter().draw(delta)
        )


def test_a_face_the_camera_sees_end_on_is_reported_as_the_dot_the_core_draws():
    """A face of one cubic along the camera's axis is points at one spot as
    the camera shows them, though not in the world: the core strokes it as a
    round cap's dot, and reports its leaf with that reason.
    """
    from arachne.frame_ir.contributors import ROUND_CAP_DOT

    surface = _ellipsoid((8, 16))
    face = _faces(surface)[CUT]
    face.points = np.array([[0.0, 0.0, z] for z in (-0.3, -0.1, 0.1, 0.3)])
    surface.set_stroke(width=4)
    for each in surface.submobjects:
        each.cap_style = CapStyleType.ROUND
    with tempconfig(LARGE):
        camera = ThreeDCamera()
        camera.set_phi(0)
        camera.set_theta(-90 * DEGREES)
        painter = CorePainter()
        painter.paint(camera, [("moving", [surface])], roots=[surface])
    slot, _ = _group(painter, surface)
    view = painter.state.view
    found = painter.contributors((0, 0, view["pixel_width"], view["pixel_height"]))
    reasons = {
        leaf.reason
        for pixel in found.values()
        for leaf in pixel.leaves
        if leaf.object == slot and leaf.part == CUT
    }
    assert reasons == {ROUND_CAP_DOT}
