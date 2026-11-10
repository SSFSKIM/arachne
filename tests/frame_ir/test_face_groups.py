"""A surface's uv faces are one face group on every frame the core draws.

Only a face group's parts are drawn as faces of a surface (keyed on their own
plane, tied to CE's paint order inside its band); a face drawn as a path is
keyed along its curve. So a surface's faces must stay one group on a play's
frames too, where the scene's static list arrives flattened and the surface
itself is not in it, and on solids whose surface carries other parts (caps,
points, a child cone), which are drawn as records of their own. Each face
keeps its own place in CE's paint order.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    PI,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Arrow3D,
    Cone,
    Cylinder,
    Dot,
    Line3D,
    Rotate,
    Square,
    Surface,
    ThreeDCamera,
    ThreeDScene,
    VGroup,
)
from arachne.frame_ir.canonical import canonical_order, frame_grid
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import FrameStream
from arachne.frame_ir.live import live_painter
from arachne.frame_ir.painter import painter_for
from arachne.frame_ir.xf import StreamState
from arachne.mobject.three_d.three_dimensions import ThreeDVMobject
from arachne.timeline.draw import static_list

from ..timeline.conftest import render

pytest.importorskip("arachne_core")

FPS = 15
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


def _wave(resolution=(8, 8)):
    return Surface(
        lambda u, v: np.array([u, v, 0.4 * np.sin(u) * np.cos(v)]),
        u_range=(-2, 2),
        v_range=(-2, 2),
        resolution=resolution,
        checkerboard_colors=[BLUE, GREEN],
    )


class MovingSurface(ThreeDScene):
    """A surface and a cylinder (with its two caps), each turning."""

    def construct(self):
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.surface = _wave()
        self.cylinder = Cylinder(radius=0.5, height=1.2, resolution=(6, 8))
        self.cylinder.shift(RIGHT * 3)
        self.add(self.surface, self.cylinder)
        self.play(
            Rotate(self.surface, PI / 3, axis=OUT),
            Rotate(self.cylinder, PI / 3, axis=RIGHT),
            run_time=0.4,
        )


class StaticSurface(ThreeDScene):
    """The surface is added before the mobject that moves: it is in the
    play's static list, flattened to its faces.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.surface = _wave()
        dot = Dot(LEFT * 3, color=RED)
        self.add(self.surface, dot)
        self.play(dot.animate.shift(UP), run_time=0.4)


class SurfaceInGroup(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.surface = _wave()
        group = VGroup(self.surface, Square(0.5).shift(RIGHT * 3))
        dot = Dot(LEFT * 3, color=RED)
        self.add(group, dot)
        self.play(dot.animate.shift(UP), run_time=0.4)


class Solids(ThreeDScene):
    """A cone (with two points), a cylinder and a 3D line (with two caps)
    and a 3D arrow (two caps, a point and a cone), static while a dot moves.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.solids = [
            Cone(base_radius=0.6, height=1.2, resolution=8).shift(LEFT * 2.5),
            Cylinder(radius=0.5, height=1.2, resolution=(6, 8)).shift(RIGHT * 2.5),
            Line3D(LEFT + DOWN * 2, RIGHT + DOWN * 2, color=YELLOW),
            Arrow3D(LEFT + UP * 2, RIGHT + UP * 2, color=RED),
        ]
        dot = Dot(LEFT * 3 + DOWN * 3)
        self.add(*self.solids, dot)
        self.play(dot.animate.shift(UP), run_time=0.4)


class FacesAroundSquare(ThreeDScene):
    """A flat surface and an upright square through its middle, both
    static: CE's depth sort puts the square between the surface's faces.
    ``live``: below them a square whose stroke width is the scene's own
    code, so the scene is drawn as it runs.
    """

    live = False

    def construct(self):
        self.set_camera_orientation(phi=70 * PI / 180, theta=-45 * PI / 180)
        self.surface = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=(-2, 2),
            v_range=(-2, 2),
            resolution=(6, 6),
            checkerboard_colors=[BLUE, GREEN],
        )
        self.square = Square(2, fill_opacity=1, color=RED).rotate(PI / 2, RIGHT)
        self.square.set_shade_in_3d(True)
        dot = Dot(LEFT * 3 + DOWN * 3)
        if self.live:
            from .scenes import WideStroke

            self.add(WideStroke(0.5, color=YELLOW).shift(RIGHT * 3 + DOWN * 3))
        self.add(self.surface, self.square, dot)
        self.play(dot.animate.shift(UP), run_time=0.4)


class FacesAroundSquareLive(FacesAroundSquare):
    live = True


def _uv_faces(surface):
    return [
        face
        for face in surface.submobjects
        if type(face) is ThreeDVMobject and hasattr(face, "u_index")
    ]


def _drawn(scene_class, monkeypatch):
    """The scene drawn on the core, and each frame the painter drew it with:
    its instant, its records with the mobject each slot is, and its order.
    """
    monkeypatch.setattr(CorePainter, "keep_all", True)
    scene, _ = render(scene_class, FPS, draw_path="core", **MEDIUM)
    painter = painter_for(scene.timeline.record, "core")
    mobs = {slot: mob for mob, slot in painter.stream._slots.values()}
    state = StreamState()
    frames = []
    for delta in painter.deltas:
        state.apply(delta)
        records = {slot: (mobs[slot], r) for slot, r in state.records.items()}
        frames.append((delta.t, records, list(state.order)))
    assert frames
    return scene, frames


def _mid_play(scene, frames):
    found = [
        frame
        for frame in frames
        for play in scene.timeline.record.plays
        if play.start < frame[0] < play.end
    ]
    assert found, "frames inside a play"
    return found


def _assert_grouped(frame, surface):
    """``surface``'s uv faces are one face group in ``frame``, each face a
    part of the order once; its other parts are records of their own.
    """
    t, records, order = frame
    slots = [slot for slot, (mob, _) in records.items() if mob is surface]
    assert len(slots) == 1, f"t={t}: the {type(surface).__name__} has a record"
    slot = slots[0]
    assert records[slot][1].kind == "face_group", f"t={t}: {type(surface).__name__}"
    faces = _uv_faces(surface)
    parts = sorted(part for _, s, part in order if s == slot)
    assert parts == list(range(len(faces))), f"t={t}"
    held = {id(mob) for mob, _ in records.values()}
    assert not held & {id(face) for face in faces}, f"t={t}: a face drawn alone"
    for other in surface.submobjects:
        if any(other is face for face in faces):
            continue
        assert id(other) in held, f"t={t}: its {type(other).__name__} is drawn"


def test_a_moving_surface_is_one_face_group_on_a_play_s_frames(monkeypatch):
    scene, frames = _drawn(MovingSurface, monkeypatch)
    for frame in _mid_play(scene, frames):
        _assert_grouped(frame, scene.surface)
        _assert_grouped(frame, scene.cylinder)


@pytest.mark.parametrize("scene_class", [StaticSurface, SurfaceInGroup])
def test_a_static_surface_is_one_face_group_while_another_mobject_moves(
    scene_class, monkeypatch
):
    scene, frames = _drawn(scene_class, monkeypatch)
    for frame in _mid_play(scene, frames):
        _assert_grouped(frame, scene.surface)
    _assert_grouped(frames[-1], scene.surface)


@pytest.mark.parametrize("when", ["last frame", "play"])
def test_a_solid_s_faces_are_one_group_and_its_other_parts_their_own(when, monkeypatch):
    scene, frames = _drawn(Solids, monkeypatch)
    cone, cylinder, line, arrow = scene.solids
    for frame in [frames[-1]] if when == "last frame" else _mid_play(scene, frames):
        for solid in (cone, cylinder, line, arrow, arrow.cone):
            _assert_grouped(frame, solid)
        _, records, order = frame
        slots = {id(mob): slot for slot, (mob, _) in records.items()}
        for cap in (cylinder.base_top, cylinder.base_bottom):
            assert records[slots[id(cap)]][1].kind == "path"
        for part in (
            *cylinder.submobjects[-2:],
            arrow.end_point,
            *cone.submobjects[-2:],
        ):
            entries = [p for _, s, p in order if s == slots[id(part)]]
            assert entries == [0]


def test_each_face_keeps_its_place_in_ce_s_paint_order(monkeypatch):
    """On a play's frame the static list is drawn in CE's order, face by
    face: the square CE sorts between two faces of the surface is drawn
    between them.
    """
    scene, frames = _drawn(FacesAroundSquare, monkeypatch)
    camera = scene.renderer.camera
    roots = list(scene.mobjects)
    moving = [roots[-1]]
    with canonical_order(camera, frame_grid(camera)):
        camera.reset_rotation_matrix()
        expected = camera.get_mobjects_to_display(static_list(camera, roots, moving))
    faces = {id(face) for face in _uv_faces(scene.surface)}
    at = [i for i, leaf in enumerate(expected) if leaf is scene.square]
    assert len(at) == 1
    # CE sorts the square between faces of the surface.
    assert any(id(leaf) in faces for leaf in expected[: at[0]])
    assert any(id(leaf) in faces for leaf in expected[at[0] + 1 :])
    for t, records, order in _mid_play(scene, frames):
        drawn = []
        for layer, slot, part in order:
            if layer != "static":
                continue
            mob, record = records[slot]
            drawn.append(_uv_faces(mob)[part] if record.kind == "face_group" else mob)
        assert [id(m) for m in drawn] == [id(m) for m in expected], f"t={t}"
        assert any(r.kind == "face_group" for _, r in records.values()), f"t={t}"


@pytest.mark.parametrize("handed", ["listed", "moving", "flattened"])
def test_a_surface_of_uv_faces_alone_keeps_its_record_bytes(handed):
    """A surface whose submobjects are all uv faces is described as it was
    before a surface's other parts were drawn apart, however a frame hands
    it over (in a scene's list, a play's moving list, or a static list
    flattened to its faces): its face group record and what it uploads are
    these bytes. The surface is flat on dyadic coordinates, so its values
    are exact on every machine.
    """
    surface = Surface(
        lambda u, v: np.array([u, v, 0.0]),
        u_range=(-1, 1),
        v_range=(-1, 1),
        resolution=(4, 4),
        checkerboard_colors=[BLUE, GREEN],
    )
    camera = ThreeDCamera()
    if handed == "listed":
        delta = FrameStream().frame(camera, [("static", [surface])])
    elif handed == "moving":
        groups = [("static", []), ("moving", surface.get_family())]
        delta = FrameStream().frame(camera, groups)
    else:
        groups = [("static", list(surface.submobjects))]
        delta = FrameStream().frame(camera, groups, roots=[surface])
    (record,) = [r for _, r in delta.set]
    assert record.kind == "face_group"
    assert [part for _, _, part in delta.order] == list(range(16))
    # What it was drawn from (XF 1.5) is the surface, the stream's first object.
    surface_class = "arachne.mobject.three_d.three_dimensions.Surface"
    assert record.fields["source"] == (((1, surface_class),),)
    digest = hashlib.sha256(record.kind.encode())
    for name in sorted(set(record.fields) - {"source"}):
        value = record.fields[name]
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype, value.shape)).encode() + value.tobytes())
        else:
            digest.update(repr(value).encode())
    for shape in sorted(delta.upload, key=lambda s: s.key):
        digest.update(shape.key + np.ascontiguousarray(shape.data).tobytes())
    for table in sorted(delta.upload_paints, key=lambda p: p.key):
        digest.update(table.key + np.ascontiguousarray(table.rows).tobytes())
    assert (
        digest.hexdigest()
        == "92d8a1e77b6a9009e414ee425a86ba08e38c2fd45b93c30cf6ae2cc5b4b8497c"
    )


# Drawn as the scene runs: one painter per scene, so a solid drawn live is
# described as the record describes it, its uv faces one face group.


#: The solids of :class:`TurningSolids`, by attribute.
SOLIDS = ("surface", "cone", "cylinder", "bare", "shaft")


class TurningSolids(ThreeDScene):
    """A surface, a cone (with two points) and a cylinder (with two caps),
    and a surface and a cylinder stroked 0 wide (CE reads no stroke colour
    of theirs), each turning. ``live``: beside them a square whose stroke
    width is the scene's own code, so the scene is drawn as it runs.
    """

    live = True

    def construct(self):
        from .scenes import WideStroke

        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.surface = _wave()
        self.cone = Cone(base_radius=0.6, height=1.2, resolution=8)
        self.cone.shift(LEFT * 3)
        self.cylinder = Cylinder(radius=0.5, height=1.2, resolution=(6, 8))
        self.cylinder.shift(RIGHT * 3)
        self.bare = _wave((4, 4)).scale(0.5).shift(UP * 2.5).set_stroke(width=0)
        self.shaft = Cylinder(radius=0.3, height=1.0, resolution=(4, 6))
        self.shaft.shift(LEFT * 3 + UP * 2).set_stroke(width=0)
        solids = [getattr(self, name) for name in SOLIDS]
        self.add(*solids)
        if self.live:
            self.add(WideStroke(0.5, color=YELLOW).shift(DOWN * 3))
        self.play(
            *(Rotate(solid, PI / 3, axis=RIGHT) for solid in solids), run_time=0.4
        )


class TurningSolidsRecorded(TurningSolids):
    live = False


def _painted(scene_class, monkeypatch, *, live):
    """The scene drawn on the core, and each frame its painter drew, live
    or recorded: its instant, its records as made with the mobject each slot
    is, and its order.
    """
    made = []
    draw = CorePainter.draw

    def keep(self, delta, *args, **kwargs):
        slots = {slot: mob for mob, slot in self.stream._slots.values()}
        made.append((self, delta, slots))
        return draw(self, delta, *args, **kwargs)

    monkeypatch.setattr(CorePainter, "draw", keep)
    scene, _ = render(scene_class, FPS, draw_path="core", **MEDIUM)
    if live:
        painter = live_painter(scene.renderer.camera).painter(scene.renderer.camera)
    else:
        painter = painter_for(scene.timeline.record, "core")
    state = StreamState()
    mobs = {}
    frames = []
    for _, delta, slots in (entry for entry in made if entry[0] is painter):
        state.apply(delta)
        mobs.update(slots)
        records = {slot: (mobs[slot], r) for slot, r in state.records.items()}
        frames.append((delta.t, records, list(state.order)))
    assert frames
    return scene, frames


def test_a_solid_drawn_live_is_one_face_group_and_its_other_parts_their_own(
    monkeypatch,
):
    scene, frames = _painted(TurningSolids, monkeypatch, live=True)
    assert scene.timeline.record.refusal is not None, "drawn live"
    # Each frame's static capture draws nothing: every mobject moves.
    frames = [frame for frame in frames if frame[2]]
    assert len(frames) > 2
    for frame in frames:
        for name in SOLIDS:
            _assert_grouped(frame, getattr(scene, name))
        _, records, order = frame
        slots = {id(mob): slot for slot, (mob, _) in records.items()}
        for cap in (scene.cylinder.base_top, scene.cylinder.base_bottom):
            assert records[slots[id(cap)]][1].kind == "path"
        for part in (*scene.cylinder.submobjects[-2:], *scene.cone.submobjects[-2:]):
            assert [p for _, s, p in order if s == slots[id(part)]] == [0]


def test_a_solid_s_face_group_is_the_same_drawn_live_or_recorded(monkeypatch):
    """The scene's last frame drawn live and drawn from the record: each
    solid's face group is the same record, but for the sort point a live
    record never carries (its order is the capture's, and the core reads no
    sort point: :func:`test_a_surface_s_faces_keep_ce_s_order_live_or_recorded`).
    """
    live, live_frames = _painted(TurningSolids, monkeypatch, live=True)
    recorded, recorded_frames = _painted(TurningSolidsRecorded, monkeypatch, live=False)
    assert recorded.timeline.record.refusal is None, "drawn from the record"

    def groups(scene, frame):
        _, records, _ = frame
        found = {}
        for name in SOLIDS:
            solid = getattr(scene, name)
            [record] = [r for mob, r in records.values() if mob is solid]
            found[name] = record
        return found

    a = groups(live, live_frames[-1])
    b = groups(recorded, recorded_frames[-1])
    for name in a:
        assert a[name].kind == b[name].kind == "face_group", name
        assert "sort_point" not in a[name].fields
        fields = {k: v for k, v in b[name].fields.items() if k != "sort_point"}
        assert a[name].fields.keys() == fields.keys(), name
        for key, value in fields.items():
            other = a[name].fields[key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(other, value, err_msg=f"{name} {key}")
            else:
                assert other == value, (name, key)


def test_a_surface_s_faces_keep_ce_s_order_live_or_recorded(monkeypatch):
    """Both routes take a frame's order from the camera's own depth sort, at
    the frame's grid: the record from its lists, a live capture as it
    draws. On each frame of the play the square CE sorts between faces of
    the surface is drawn between the same faces, live or recorded, each face
    a part of the surface's face group.
    """

    def drawn(scene, frames):
        """Per instant, the faces (by index) and the square in the order of
        the instant's last frame.
        """
        faces = {id(face): index for index, face in enumerate(_uv_faces(scene.surface))}
        found = {}
        for t, records, order in frames:
            labels = []
            for _, slot, part in order:
                mob, record = records[slot]
                if mob is scene.surface:
                    assert record.kind == "face_group"
                    labels.append(part)
                elif mob is scene.square:
                    labels.append("square")
                else:
                    assert id(mob) not in faces, "a face drawn alone"
            if labels:
                found[t] = labels
        return found

    live, live_frames = _painted(FacesAroundSquareLive, monkeypatch, live=True)
    assert live.timeline.record.refusal is not None, "drawn live"
    recorded, recorded_frames = _painted(FacesAroundSquare, monkeypatch, live=False)
    a, b = drawn(live, live_frames), drawn(recorded, recorded_frames)
    shared = sorted(set(a) & set(b))
    assert len(shared) > 2
    for t in shared:
        assert a[t] == b[t], f"t={t}"
        at = a[t].index("square")
        assert 0 < at < len(a[t]) - 1, "CE sorts the square between faces"


class GradientFace(ThreeDScene):
    """A surface one face of which states its gradient's ends by its own
    method, beside a square whose stroke width is the scene's own code.
    """

    def construct(self):
        from .scenes import WideStroke

        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.surface = _wave((4, 4))
        face = self.surface.submobjects[0]
        start, end = face.get_gradient_start_and_end_points()
        face.get_gradient_start_and_end_points = lambda: (end, start)
        self.add(self.surface, WideStroke(0.5, color=YELLOW).shift(DOWN * 3))
        self.play(Rotate(self.surface, PI / 3, axis=RIGHT), run_time=0.4)


def test_a_face_stating_its_own_gradient_ends_is_drawn_as_a_path_live(monkeypatch):
    """A face group draws no face's gradient ends, so the record leaves a
    surface whose face states its own (``extract._one_group``) as paths, and
    a live capture does too.
    """
    scene, frames = _painted(GradientFace, monkeypatch, live=True)
    frames = [frame for frame in frames if frame[2]]
    assert frames
    faces = {id(face) for face in _uv_faces(scene.surface)}
    for t, records, _ in frames:
        kinds = {r.kind for mob, r in records.values() if id(mob) in faces}
        assert kinds == {"path"}, f"t={t}"
        assert not [r for mob, r in records.values() if mob is scene.surface]


@pytest.mark.parametrize("fill", ["steady", "changing"])
def test_a_face_drawn_twice_in_a_capture_is_drawn_one_way(fill):
    """A capture's list holds a face twice (``include_submobjects=False``
    keeps it). Each draw is a record of the reads it made: one way, the face
    is a part of its surface's group twice; two ways (its fill changed
    between the draws), it is ``render.unable``, grouped or not.
    """
    from fractions import Fraction

    from arachne import tempconfig
    from arachne.frame_ir.extract import Unable
    from arachne.frame_ir.live import live

    with tempconfig(MEDIUM):
        surface = _wave((2, 2))
        face = surface.submobjects[0]
        if fill == "changing":
            own = face.get_fill_rgbas
            calls = []

            def get_fill_rgbas():
                calls.append(None)
                rows = np.array(own(), dtype=float)
                if len(calls) > 1:
                    rows[:, 0] = 1 - rows[:, 0]
                return rows

            face.get_fill_rgbas = get_fill_rgbas
        camera = ThreeDCamera()
        leaves = [*surface.submobjects, face]

        def capture():
            with live(camera, Fraction(0), draw_path="core", roots=[surface]):
                camera.reset()
                camera.capture_mobjects(leaves, include_submobjects=False)

        if fill == "changing":
            with pytest.raises(Unable, match="drawn two ways"):
                capture()
            return
        capture()
        painter = live_painter(camera).painter(camera)
        [(_, record)] = painter.stream._held.items()
        assert record.kind == "face_group"
        assert sorted(part for _, _, part in painter.stream._order) == [0, 0, 1, 2, 3]


def _drawn_live(surface):
    """The kinds of the records a live capture of ``surface``'s faces held,
    on the core.
    """
    from fractions import Fraction

    from arachne import tempconfig
    from arachne.frame_ir.live import live

    with tempconfig(MEDIUM):
        camera = ThreeDCamera()
        with live(camera, Fraction(0), draw_path="core", roots=[surface]):
            camera.reset()
            camera.capture_mobjects(list(surface.submobjects))
        painter = live_painter(camera).painter(camera)
        return sorted(record.kind for record in painter.stream._held.values())


@pytest.mark.parametrize("owner", ["VMobject", "ThreeDVMobject"])
def test_a_stroke_accessor_replaced_on_a_class_leaves_a_live_surface_as_paths(
    owner, monkeypatch
):
    """A surface stroked 0 wide, its faces' ``get_stroke_rgbas`` replaced on
    their class (here giving two stops where their paint holds one): the
    record reads the accessor and leaves the surface as paths, and a live
    capture, which reads no stroke colour at that width, does too rather
    than give the faces their paint's.
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    cls = {"VMobject": VMobject, "ThreeDVMobject": ThreeDVMobject}[owner]
    engine = VMobject.get_stroke_rgbas

    def two_stops(self, background=False):
        return np.repeat(np.asarray(engine(self, background))[:1], 2, axis=0)

    monkeypatch.setattr(cls, "get_stroke_rgbas", two_stops)
    surface = _wave((2, 2)).set_stroke(width=0)
    recorded = FrameStream().frame(ThreeDCamera(), [("static", [surface])])
    assert {record.kind for _, record in recorded.set} == {"path"}
    assert _drawn_live(surface) == ["path"] * 4


def test_a_paint_property_on_the_face_class_never_runs_for_a_live_surface(
    monkeypatch,
):
    """CE reads no stroke colour of a face stroked 0 wide, so a live capture
    never runs a property the face's class puts over its stroke colours: it
    leaves the surface as paths.
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    engine = VMobject.stroke_rgbas
    reads = []

    def counted(self):
        reads.append(None)
        return engine.fget(self)

    surface = _wave((2, 2)).set_stroke(width=0)
    monkeypatch.setattr(ThreeDVMobject, "stroke_rgbas", property(counted, engine.fset))
    kinds = _drawn_live(surface)
    assert reads == []
    assert kinds == ["path"] * 4
