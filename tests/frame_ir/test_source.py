"""The scene lowering writes each record's source (XF 1.5): the objects it was
drawn from, by containment, each named by a token and its nearest library
class; and a camera-facing label's group. Nothing drawn reads either: frames
are the bytes they were without them.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import (
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Arrow3D,
    Circle,
    Dot,
    ImageMobject,
    MathTex,
    PMobject,
    Scene,
    Square,
    Surface,
    Text,
    ThreeDScene,
    TransformMatchingTex,
    Triangle,
    Vector,
    VGroup,
    tempconfig,
)
from arachne.camera.camera import Camera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import Blend, FrameStream, Reveal, library_class
from arachne.frame_ir.geometry import revealed
from arachne.frame_ir.painter import IrCairoPainter
from arachne.frame_ir.xf import Delta, Record
from arachne.utils.bezier import interpolate

from ..timeline import camera_scenes
from ..timeline.conftest import render
from . import scenes
from .camera_scenes import Solids

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}

SQUARE = "arachne.mobject.geometry.polygram.Square"
CIRCLE = "arachne.mobject.geometry.arc.Circle"
VGROUP = "arachne.mobject.types.vectorized_mobject.VGroup"
VECTOR = "arachne.mobject.geometry.line.Vector"
TIP = "arachne.mobject.geometry.tips.ArrowTriangleFilledTip"
SOLIDS = "arachne.mobject.three_d.three_dimensions"


class Badge(Square):
    """A scene's own square."""


class Board(VGroup):
    """A scene's own group."""


def _framed(camera, mobjects, *, stream=None, **kwargs):
    """One frame of ``mobjects`` (the scene's, in one static list), and the
    record each mobject's slot holds, by the mobject's id.
    """
    stream = FrameStream() if stream is None else stream
    delta = stream.frame(camera, [("static", mobjects)], roots=mobjects, **kwargs)
    return stream, delta, _by_mobject(stream)


def _by_mobject(stream):
    held = stream._held
    return {id(m): held[slot] for m, slot in stream._slots.values() if slot in held}


def _classes(record):
    return [[cls for _, cls in path] for path in record.fields["source"]]


def _token(record):
    """The token of the object a record draws: its source's first node's."""
    return record.fields["source"][0][0][0]


def test_a_class_is_named_by_the_nearest_library_class_it_derives_from():
    assert library_class(Square) == SQUARE
    assert library_class(Badge) == SQUARE
    assert library_class(Board) == VGROUP
    badge, inner = Badge(), Circle()
    board = Board(badge, VGroup(inner))
    _, delta, records = _framed(Camera(), [board])
    assert _classes(records[id(badge)]) == [[SQUARE, VGROUP]]
    assert _classes(records[id(inner)]) == [[CIRCLE, VGROUP, VGROUP]]
    for _, record in delta.set:
        for path in record.fields["source"]:
            for token, cls in path:
                assert isinstance(token, int)
                assert token > 0
                assert cls.startswith("arachne.")
                assert "Badge" not in cls
                assert "Board" not in cls


def test_a_vector_s_tip_names_the_vector_that_holds_it():
    vector = Vector(RIGHT * 2)
    tip = vector.tip
    _, _, records = _framed(Camera(), [vector])
    assert _classes(records[id(tip)]) == [[TIP, VECTOR]]
    assert _classes(records[id(vector)]) == [[VECTOR]]
    assert records[id(tip)].fields["source"][0][1][0] == _token(records[id(vector)])


def test_an_arrow3d_s_cone_and_shaft_name_the_arrow():
    arrow = Arrow3D(LEFT, RIGHT + UP, resolution=8)
    _, _, records = _framed(ThreeDCamera(), [arrow])
    shaft, cone = records[id(arrow)], records[id(arrow.cone)]
    assert shaft.kind == cone.kind == "face_group"
    assert _classes(shaft) == [[f"{SOLIDS}.Arrow3D"]]
    assert _classes(cone) == [[f"{SOLIDS}.Cone", f"{SOLIDS}.Arrow3D"]]
    assert cone.fields["source"][0][1][0] == _token(shaft)


def test_a_face_group_names_its_surface_however_its_faces_are_handed_over():
    for handed in ("listed", "flattened"):
        owner = Surface(
            lambda u, v: np.array([u, v, 0.0]),
            u_range=(-1, 1),
            v_range=(-1, 1),
            resolution=(3, 3),
        )
        if handed == "listed":
            groups = [("static", [owner])]
        else:
            groups = [("static", list(owner.submobjects))]
        stream = FrameStream()
        delta = stream.frame(ThreeDCamera(), groups, roots=[owner])
        [(_, record)] = delta.set
        assert record.kind == "face_group"
        assert _classes(record) == [[f"{SOLIDS}.Surface"]], handed
        assert _token(record) == stream._token(owner)


def test_moving_a_leaf_to_another_parent_sends_its_new_source_and_sets_nothing():
    """A record whose drawing is unchanged is not set again when only its
    source moves: the older minor's bytes stay what they were.
    """
    camera = Camera()
    square = Square()
    a, b = VGroup(square), VGroup()
    stream = FrameStream()
    stream.frame(camera, [("static", [a, b])], roots=[a, b])
    first = _by_mobject(stream)[id(square)]
    held = stream.frame(camera, [("static", [a, b])], roots=[a, b])
    assert not held.set, "nothing changed"
    a.remove(square)
    b.add(square)
    moved = stream.frame(camera, [("static", [a, b])], roots=[a, b])
    assert not moved.set, "its drawing is unchanged"
    assert not moved.upload
    [(slot, source, group)] = moved.source
    assert slot == stream._slots[id(square)][1]
    assert group == 0
    assert first.fields["source"][0][1][0] == stream._token(a)
    assert source[0][1][0] == stream._token(b)
    assert _by_mobject(stream)[id(square)].fields["source"] == source
    again = stream.frame(camera, [("static", [a, b])], roots=[a, b])
    assert not again.set
    assert not again.source


def test_a_shared_leaf_carries_a_path_through_each_parent():
    square = Square()
    a, b = VGroup(square), VGroup(square)
    outer = VGroup(b)
    stream, _, records = _framed(Camera(), [a, outer])
    paths = records[id(square)].fields["source"]
    tokens = [[token for token, _ in path] for path in paths]
    assert tokens == [
        [stream._token(square), stream._token(a)],
        [stream._token(square), stream._token(b), stream._token(outer)],
    ]


def test_a_blend_s_and_a_reveal_s_records_name_the_leaf_drawn():
    start, end = Square(), Square().shift(RIGHT)
    blended = start.copy()
    blended.points = interpolate(start.points, end.points, 0.5)
    whole = Square()
    part = Square()
    part.points = revealed(whole.points, 0.0, 0.5)
    terms = {
        id(blended): Blend(start, end, 0.5),
        id(part): Reveal(whole, 0.0, 0.5),
    }
    _, _, records = _framed(Camera(), [VGroup(blended, part)], terms=terms)
    assert "blend" in records[id(blended)].fields
    assert "reveal" in records[id(part)].fields
    for leaf in (blended, part):
        assert _classes(records[id(leaf)]) == [[SQUARE, VGROUP]]


def test_a_point_cloud_s_and_an_image_s_records_name_their_objects():
    cloud = PMobject().add_points([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    image = ImageMobject(np.full((2, 3, 4), 255, dtype=np.uint8))
    _, _, records = _framed(Camera(), [cloud, image])
    assert records[id(cloud)].kind == "points"
    assert _classes(records[id(cloud)]) == [
        ["arachne.mobject.types.point_cloud_mobject.PMobject"]
    ]
    assert records[id(image)].kind == "image"
    assert _classes(records[id(image)]) == [
        ["arachne.mobject.types.image_mobject.ImageMobject"]
    ]


def _centre():
    return OUT


def test_a_label_s_glyphs_share_a_facing_group_and_two_labels_never_do():
    camera = ThreeDCamera()
    one = VGroup(Square(0.2), Circle(0.1).shift(RIGHT * 0.3))
    two = VGroup(Square(0.2), Circle(0.1)).shift(DOWN)
    # Two more, registered apart with one centre function: one centre.
    three, four = VGroup(Square(0.1)), VGroup(Circle(0.1))
    plain = Square().shift(LEFT * 2)
    camera.add_fixed_orientation_mobjects(one, two)
    camera.add_fixed_orientation_mobjects(three, center_func=_centre)
    camera.add_fixed_orientation_mobjects(four, center_func=_centre)
    _, _, records = _framed(camera, [one, two, three, four, plain])

    def group(mob):
        return records[id(mob)].fields.get("facing_group")

    assert group(one[0]) == group(one[1]) != 0
    assert group(two[0]) == group(two[1]) != 0
    assert len({group(one[0]), group(two[0]), group(three[0]), group(four[0])}) == 4
    assert group(plain) is None
    for record in records.values():
        centred = record.fields.get("fixed_orientation_centre") is not None
        assert centred == ("facing_group" in record.fields)


def test_registering_a_glyph_again_splits_it_from_its_label():
    camera = ThreeDCamera()
    label = VGroup(Square(0.2), Circle(0.1).shift(RIGHT * 0.3), Square(0.1))
    camera.add_fixed_orientation_mobjects(label)
    stream, _, records = _framed(camera, [label])
    before = {records[id(glyph)].fields["facing_group"] for glyph in label}
    assert len(before) == 1
    camera.add_fixed_orientation_mobjects(label[2], use_static_center_func=True)
    delta = stream.frame(camera, [("static", [label])], roots=[label])
    after = _by_mobject(stream)
    assert [slot for slot, _ in delta.set] == [stream._slots[id(label[2])][1]]
    kept = after[id(label[0])].fields["facing_group"]
    split = after[id(label[2])].fields["facing_group"]
    assert kept == before.pop()
    assert split not in (0, kept)


def test_regrouping_a_held_label_s_glyph_sends_its_new_facing_group():
    """A glyph moved from one label into another kept facing the camera by
    the same centre function joins that label: its drawing unchanged, its
    new facing group is carried.
    """
    camera = ThreeDCamera()
    one, two = VGroup(Square(0.1)), VGroup(Circle(0.1), Square(0.1).shift(RIGHT))
    camera.add_fixed_orientation_mobjects(one, center_func=_centre)
    camera.add_fixed_orientation_mobjects(two, center_func=_centre)
    stream, _, records = _framed(camera, [one, two])
    glyph = two[0]
    slot = stream._slots[id(glyph)][1]
    before = records[id(glyph)].fields["facing_group"]
    assert before == stream._token(two)
    two.remove(glyph)
    one.add(glyph)
    delta = stream.frame(camera, [("static", [one, two])], roots=[one, two])
    carried = {s: r.fields.get("facing_group", 0) for s, r in delta.set}
    carried.update((s, group) for s, _, group in delta.source)
    assert carried[slot] == stream._token(one) != before
    assert slot in {s for s, _, _ in delta.source}, "its drawing is unchanged"
    assert _by_mobject(stream)[id(glyph)].fields["facing_group"] == carried[slot]
    assert _by_mobject(stream)[id(two[0])].fields["facing_group"] == before


def test_a_leaf_reparented_between_live_frames_sends_its_new_source_alone(
    monkeypatch,
):
    """Drawn live, too: a held record whose objects move and whose drawing
    does not is not set again.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.live import live, live_painter

    sent = []
    monkeypatch.setattr(CorePainter, "draw", _keeping(CorePainter.draw, sent))
    square = Square()
    a, b = VGroup(square), VGroup()
    with tempconfig({"pixel_width": 32, "pixel_height": 18}):
        camera = ThreeDCamera()
        for index in range(2):
            if index:
                a.remove(square)
                b.add(square)
            with live(camera, Fraction(index, FPS), draw_path="core", roots=[a, b]):
                camera.reset()
                camera.capture_mobjects([a, b])
        stream = live_painter(camera).painter(camera).stream
    [slot] = stream._held
    first, second = sent[-2:]
    assert [s for s, _ in first.set] == [slot]
    assert not second.set, "its drawing is unchanged"
    [(row_slot, source, group)] = second.source
    assert row_slot == slot
    assert group == 0
    assert source[0][1][0] == stream._token(b)
    assert stream._held[slot].fields["source"] == source


def test_a_record_drawn_live_carries_the_source_the_record_path_writes():
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.live import live, live_painter

    square = Square()
    holder = VGroup(square)
    with tempconfig({"pixel_width": 32, "pixel_height": 18}):
        camera = ThreeDCamera()
        camera.add_fixed_orientation_mobjects(holder)
        with live(camera, Fraction(0), draw_path="core", roots=[holder]):
            camera.reset()
            camera.capture_mobjects([holder])
        stream = live_painter(camera).painter(camera).stream
        [made] = stream._held.values()
        _, _, records = _framed(camera, [holder])
    assert _classes(made) == [[SQUARE, VGROUP]]
    assert made.fields["source"] == records[id(square)].fields["source"]
    assert made.fields["facing_group"] == stream._token(holder)
    assert made.fields["facing_group"] == records[id(square)].fields["facing_group"]


def _keeping(draw, sent):
    def kept(self, delta, **kwargs):
        sent.append(delta)
        return draw(self, delta, **kwargs)

    return kept


#: Scenes whose frames are drawn through XF on each creation path: a blend,
#: reveals, a point cloud and an image, displays, face groups and labels kept
#: facing the camera, and leaves drawn live.
DRAWN = [
    scenes.Morph,
    scenes.Reveals,
    scenes.Clouds,
    scenes.Lit3D,
    scenes.Watched,
    scenes.LiveOverride,
    camera_scenes.DisplayLeaves,
]


@pytest.mark.parametrize("scene_class", DRAWN, ids=lambda c: c.__name__)
@pytest.mark.parametrize("draw_path", ["ir_cairo", "core"])
def test_every_record_sent_carries_its_source(scene_class, draw_path, monkeypatch):
    pytest.importorskip("arachne_core")
    sent = []
    for painter_class in (CorePainter, IrCairoPainter):
        monkeypatch.setattr(painter_class, "draw", _keeping(painter_class.draw, sent))
    render(scene_class, FPS, draw_path=draw_path, **MEDIUM)
    records = [record for delta in sent for _, record in delta.set]
    assert records, "drawn through XF"
    for record in records:
        source = record.fields.get("source")
        assert source, record
        for path in source:
            assert path, record
            assert all(cls.startswith("arachne.") for _, cls in path), path


_STRIPPED = ("source", "facing_group")


def _older_minor(delta):
    """``delta``'s bytes without its sources: what an older minor carries."""
    bare = Delta(
        t=delta.t,
        seq=delta.seq,
        view=delta.view,
        objects=delta.objects,
        upload=delta.upload,
        upload_paints=delta.upload_paints,
        release=delta.release,
        set=[
            (
                slot,
                Record(
                    record.kind,
                    {k: v for k, v in record.fields.items() if k not in _STRIPPED},
                ),
            )
            for slot, record in delta.set
        ],
        remove=delta.remove,
        order=delta.order,
    )
    return bytes(bare.to_core().to_bytes())


def test_frames_and_the_older_minor_are_the_bytes_sent_without_a_source(monkeypatch):
    """The depth resolver reads no source, and a source moves no byte an
    older minor carries: a scene's frames on the core, and each delta it is
    handed without its sources, are the bytes of lowering that carries none,
    whatever is reparented.
    """
    pytest.importorskip("arachne_core")
    cases = [
        scenes.Lit3D,
        scenes.Watched,
        scenes.LiveOverride,
        camera_scenes.PureCentre,
        LabelledArrow,
        Regrouped,
        MatchedTex,
        Solids,
    ]

    def drawn():
        found = []
        for case in cases:
            sent = []
            with monkeypatch.context() as patched:
                patched.setattr(CorePainter, "draw", _keeping(CorePainter.draw, sent))
                frames = render(case, FPS, draw_path="core", **MEDIUM)[1]
            found.append((frames, [_older_minor(delta) for delta in sent]))
        return found

    carried = drawn()
    monkeypatch.setattr(FrameStream, "_sourced", lambda self, slot, record, *_: record)
    bare = drawn()
    for case, (ours, our_bytes), (theirs, their_bytes) in zip(
        cases, carried, bare, strict=True
    ):
        assert len(ours) == len(theirs) > 0, case.__name__
        for index, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"{case.__name__} {index}")
        assert len(our_bytes) == len(their_bytes) > 0, case.__name__
        for index, (a, b) in enumerate(zip(our_bytes, their_bytes, strict=True)):
            assert a == b, f"{case.__name__}: delta {index} moves the older minor"


class Regrouped(Scene):
    """A square moved from one group to another, its drawing unchanged."""

    def construct(self):
        square = Square()
        a, b = VGroup(square), VGroup()
        self.add(a, b)
        self.wait(0.2)
        a.remove(square)
        b.add(square)
        self.wait(0.2)


class MatchedTex(Scene):
    """Glyphs regrouped by a matching transform."""

    def construct(self):
        start = MathTex("A", "+", "B", "=", "C")
        end = MathTex("C", "=", "B", "-", "A")
        self.add(start)
        self.play(TransformMatchingTex(start, end), run_time=0.4)


class LabelledArrow(ThreeDScene):
    """Labels kept facing the camera beside an arrow, under a turn."""

    def construct(self):
        self.set_camera_orientation(phi=1.1, theta=-0.8)
        labels = [Square(0.3).shift(RIGHT * x + OUT) for x in (-1, 1)]
        self.add(Arrow3D(LEFT * 2, RIGHT * 2, resolution=6), *labels)
        self.add_fixed_orientation_mobjects(*labels)
        self.move_camera(theta=-0.5, run_time=0.3)


class FacingFamilies(ThreeDScene):
    """Kept facing the camera under a turn: a dot, a shape, and a family
    mixing text and a shape.
    """

    def construct(self):
        self.set_camera_orientation(phi=1.0, theta=-0.7)
        dot = Dot([1.0, 1.0, 0.0])
        shape = Triangle().scale(0.3).shift(LEFT * 2)
        mixed = VGroup(Text("x", font_size=24), Square(0.2)).arrange(RIGHT)
        mixed.shift(DOWN)
        self.add(dot, shape, mixed)
        self.add_fixed_orientation_mobjects(dot, shape, mixed)
        self.move_camera(theta=-0.4, run_time=0.3)


#: Scenes keeping records facing the camera: labels of shapes, a label of text,
#: a dot, a shape and a family mixing text and a shape.
FACING = [LabelledArrow, Solids, FacingFamilies]


@pytest.mark.parametrize("scene_class", FACING, ids=lambda c: c.__name__)
@pytest.mark.parametrize("draw_path", ["ir_cairo", "core"])
def test_every_record_set_facing_the_camera_has_a_facing_group(
    scene_class, draw_path, monkeypatch
):
    """Scene lowering sets no record kept facing the camera outside a label:
    each carries a facing group.
    """
    pytest.importorskip("arachne_core")
    sent = []
    for painter_class in (CorePainter, IrCairoPainter):
        monkeypatch.setattr(painter_class, "draw", _keeping(painter_class.draw, sent))
    render(scene_class, FPS, draw_path=draw_path, **MEDIUM)
    facing = [
        record
        for delta in sent
        for _, record in delta.set
        if record.fields.get("fixed_orientation_centre") is not None
    ]
    assert facing, "kept facing the camera"
    for record in facing:
        assert record.fields.get("facing_group", 0) != 0, record
