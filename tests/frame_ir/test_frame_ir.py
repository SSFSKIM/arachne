"""The Frame IR from the record: every frame described in XF and drawn back by
Cairo from that description alone is the direct path's frame, byte for
byte; and the description is a stream, so what did not change is not sent
again.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LineJointType, config, tempconfig
from arachne.frame_ir.extract import Unable
from arachne.frame_ir.painter import IrCairoPainter, painter_for
from arachne.frame_ir.xf import Record
from arachne.session import Session
from arachne.timeline.mapping import render_mapping_centre
from arachne.utils.bezier import interpolate

from ..timeline import camera_scenes
from ..timeline.conftest import render
from . import scenes

pytest.importorskip("arachne_core")

SCENES = Path(scenes.__file__)
FPS = 15
#: Large enough for glyphs and a zoomed display.
MEDIUM = {"pixel_width": 192, "pixel_height": 108}


@pytest.fixture
def kept(monkeypatch):
    """Every painter keeps the deltas it sends."""
    monkeypatch.setattr(IrCairoPainter, "keep_all", True)


def frames_both_ways(scene_class: type) -> tuple[list, list, object]:
    _, direct = render(scene_class, FPS, **MEDIUM)
    scene, described = render(scene_class, FPS, draw_path="ir_cairo", **MEDIUM)
    return direct, described, scene


@pytest.mark.parametrize("scene_class", scenes.SCENES, ids=lambda c: c.__name__)
def test_drawn_from_xf_alone_is_byte_identical(scene_class):
    direct, described, scene = frames_both_ways(scene_class)
    assert _painter(scene).frames > 0, "drawn from the record through XF"
    assert len(direct) == len(described) > 0
    for index, (a, b) in enumerate(zip(direct, described, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def test_a_display_left_by_its_camera_shows_what_the_described_frames_left():
    """The record draws the display's frames from XF; once its camera leaves,
    the frames drawn as the scene runs show the picture those left in it.
    """
    direct, described, scene = frames_both_ways(camera_scenes.DisplayLeaves)
    assert _painter(scene).frames > 0
    assert len(direct) == len(described) > 0
    for index, (a, b) in enumerate(zip(direct, described, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


def _painter(scene) -> IrCairoPainter:
    painter = painter_for(scene.timeline.record, "ir_cairo")
    assert painter is not None
    return painter


def test_a_straight_path_morph_uploads_no_shape_after_its_first_frame(kept):
    scene, _ = render(scenes.Morph, FPS, draw_path="ir_cairo", **MEDIUM)
    record = scene.timeline.record
    play = next(p for p in record.plays if p.animations == ("Transform(Square)",))
    assert play.route == "trusted"
    deltas = [d for d in _painter(scene).deltas if play.start <= d.t < play.end]
    assert len(deltas) >= 5
    blended = [r for d in deltas for _, r in d.set if "blend" in r.fields]
    assert len(blended) >= len(deltas) - 2, "the morph's frames are blends"
    assert all(not d.upload for d in deltas[2:]), [len(d.upload) for d in deltas]


def test_a_lagged_morph_blends_each_leaf_at_its_own_alpha(kept):
    scene, _ = render(scenes.LaggedMorph, FPS, draw_path="ir_cairo", **MEDIUM)
    deltas = _painter(scene).deltas
    mixes = {
        round(r.fields["mix"], 12)
        for d in deltas[2:4]
        for _, r in d.set
        if "blend" in r.fields
    }
    assert len(mixes) > 1
    assert all(not d.upload for d in deltas[2:])


def test_a_create_play_uploads_no_shape_after_its_first_frame(kept):
    scene, _ = render(scenes.Reveals, FPS, draw_path="ir_cairo", **MEDIUM)
    record = scene.timeline.record
    for play in record.plays:
        assert play.route == "trusted"
        deltas = [d for d in _painter(scene).deltas if play.start <= d.t < play.end]
        assert len(deltas) >= 5
        revealed = [r for d in deltas for _, r in d.set if "reveal" in r.fields]
        assert len(revealed) >= len(deltas) - 2, "the frames are reveal windows"
        assert all(not d.upload for d in deltas[2:]), [len(d.upload) for d in deltas]


def _halfway():
    from arachne import RIGHT, Square

    start, end = Square(), Square().shift(RIGHT)
    leaf = start.copy()
    leaf.points = interpolate(start.points, end.points, 0.5)
    return leaf, start, end


@pytest.mark.parametrize(
    "spoil",
    [
        None,
        lambda leaf, start, end: setattr(leaf, "tolerance_for_point_equality", 1e-3),
        lambda leaf, start, end: setattr(leaf, "joint_type", LineJointType.ROUND),
        # The leaf holds the interpolated width, so only CE's clamp, which
        # comes after the interpolation, tells the blend from the leaf.
        lambda leaf, start, end: (
            setattr(start, "stroke_width", -1.0),
            setattr(leaf, "stroke_width", interpolate(-1.0, end.stroke_width, 0.5)),
        ),
    ],
    ids=["plain", "tolerance", "own joint", "negative width"],
)
def test_a_blend_is_stated_only_where_it_reads_as_the_plain_path(spoil):
    from arachne.camera.camera import Camera
    from arachne.frame_ir.extract import Blend, FrameStream

    leaf, start, end = _halfway()
    if spoil is not None:
        spoil(leaf, start, end)
    record = FrameStream()._blend_record(Camera(), leaf, Blend(start, end, 0.5), {})
    assert (record is not None) == (spoil is None)


def test_a_held_frame_sends_an_empty_delta():
    # Drawn again, not reused, so that the stream describes it.
    settings = {**MEDIUM, "draw_path": "ir_cairo", "draw_reuse": False}
    session = Session.open(SCENES, "Morph", config=settings)
    try:
        first = np.asarray(session.frame(Fraction(1, 30)))
        painter = _painter(session.scene)
        second = np.asarray(session.frame(Fraction(2, 30)))
        assert painter.last is not None
        assert painter.last.is_empty
        np.testing.assert_array_equal(first, second)
    finally:
        session.close()


def test_a_camera_only_change_sends_the_view_and_the_order_alone(kept):
    scene, _ = render(scenes.Orbit, FPS, draw_path="ir_cairo", **MEDIUM)
    deltas = _painter(scene).deltas
    assert len(deltas) > 3
    for delta in deltas[1:]:
        assert delta.view is not None
        assert not delta.upload
        assert not delta.upload_paints
        assert not delta.set
        assert not delta.remove
        assert not delta.release


def test_an_unchanged_node_state_yields_the_same_record_object(kept):
    scene, _ = render(scenes.Shapes2D, FPS, draw_path="ir_cairo", **MEDIUM)
    painter = _painter(scene)
    stream = painter.stream
    deltas = painter.deltas
    first = dict(deltas[0].set)
    later = {slot for d in deltas[1:] for slot, _ in d.set}
    # The text, the gradient and the dashes never change: never sent again,
    # and the extractor still holds the very record it first sent.
    unchanged = sorted(set(first) - later)
    assert len(unchanged) >= 3
    assert all(stream._held[slot] is first[slot] for slot in unchanged)


def test_extraction_runs_no_scene_code_on_the_record_path():
    scenes.ARMED["on"] = False
    session = Session.open(
        SCENES, "Watched", config={**MEDIUM, "draw_path": "ir_cairo"}
    )
    try:
        scenes.ARMED["on"] = True
        grid = [Fraction(i, FPS) for i in range(int(session.record.duration * FPS))]
        for t in grid:
            session.frame(t)
    finally:
        scenes.ARMED["on"] = False
        session.close()


def test_the_view_records_the_centre_the_cached_context_froze():
    scene, _ = render(scenes.MovedFlatCentre, FPS, draw_path="ir_cairo", **MEDIUM)
    stream = _painter(scene).stream
    assert stream.mapping_offset_frames > 0
    view = stream._view
    assert view["mapping_centre"] == (0.0, 0.0, 0.0)
    assert view["frame_center"] == (1.0, 0.5, 0.0)


def test_a_moved_three_d_centre_keeps_the_first_mapping_centre():
    scene, _ = render(scenes.MovedThreeDCentre, FPS, draw_path="ir_cairo", **MEDIUM)
    stream = _painter(scene).stream
    assert stream.mapping_offset_frames > 0
    assert stream._view["mapping_centre"] == (0.0, 0.0, 0.0)


def test_a_late_frame_asked_first_keeps_the_render_s_mapping_centre():
    """The render freezes the mapping centre at its own first path draw; a
    session asked for a late frame first must draw it where the render does,
    not freeze the centre the camera has moved to by then. The session
    freezes the centre the record's render froze (``render_mapping_centre``)
    before it draws, and the frame's description maps around it.
    """
    _, direct = render(scenes.MovedThreeDCentre, FPS, **MEDIUM)
    late = len(direct) - 1
    session = Session.open(
        SCENES,
        "MovedThreeDCentre",
        config={**MEDIUM, "frame_rate": FPS, "draw_path": "ir_cairo"},
    )
    try:
        frame = np.asarray(session.frame(Fraction(late, FPS)))
        painter = _painter(session.scene)
        assert painter.frames > 0, "drawn from XF"
        centre = render_mapping_centre(session.record)
        assert centre is not None
        assert painter.stream._view["mapping_centre"] == centre
    finally:
        session.close()
    np.testing.assert_array_equal(frame, direct[late])


def test_a_surface_is_one_face_group_record(kept):
    scene, _ = render(scenes.Lit3D, FPS, draw_path="ir_cairo", **MEDIUM)
    painter = _painter(scene)
    groups = {
        slot: r for slot, r in painter.state.records.items() if r.kind == "face_group"
    }
    # The surface and the ball (a Dot3D is an 8 by 8 sphere): one record each,
    # each face of it a part of the order.
    assert len(groups) == 2
    for slot, group in groups.items():
        assert tuple(group.fields["grid"]) == (8, 8)
        parts = [part for _, s, part in painter.state.order if s == slot]
        assert sorted(parts) == list(range(64))


def test_the_default_path_imports_no_core():
    import subprocess
    import sys

    code = (
        "import sys, arachne\n"
        "from arachne.timeline.draw import Drawing\n"
        "import arachne.frame_ir.painter\n"
        "assert 'arachne_core' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_draw_path_is_an_option():
    assert config.draw_path == "cairo"
    with tempconfig({"draw_path": "ir_cairo"}):
        assert config.draw_path == "ir_cairo"
    with pytest.raises(ValueError):
        config.draw_path = "gpu"


#: A placement one unit to the right.
_SHIFTED = np.array([[1.0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])


def _blank():
    from arachne.frame_ir.xf import Shape

    return Shape("image", np.zeros((9, 16, 4), dtype=np.uint8))


_BLANK = _blank()


def _square_and(record: object) -> object:
    """A described frame: a square path, then ``record`` in the order."""
    from arachne.frame_ir.xf import Delta, Shape

    background = Shape("image", np.zeros((9, 16, 4), dtype=np.uint8))
    square = Shape(
        "path", np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=float)
    )
    return Delta(
        view={
            "pixel_width": 16,
            "pixel_height": 9,
            "frame_width": 16.0,
            "frame_height": 9.0,
            "frame_center": (0.0, 0.0, 0.0),
            "mapping_centre": (0.0, 0.0, 0.0),
            "background": background.key,
            "camera": "flat",
            "cairo_line_width_multiple": 0.01,
            "use_z_index": True,
        },
        upload=[background, square],
        set=[
            (0, Record("path", {"shape": square.key, "stroke_width": 4.0})),
            (1, record(square.key) if callable(record) else record),
        ],
        order=[("static", 0, 0), ("static", 1, 0)],
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda key: Record("volume", {"payload": b"\x01\x02"}),
        lambda key: Record("path", {"shape": key, "features": 4}),
        lambda key: Record("path", {"shape": key, "place": _SHIFTED}),
        lambda key: Record("path", {"shape": key, "blend": (key, _SHIFTED, 0.5)}),
        lambda key: Record("path", {"shape": key, "dash": (1.0, 0.5, 0.0)}),
        lambda key: Record("path", {"shape": key, "clip": (key, np.eye(3, 4))}),
        lambda key: Record("path", {"shape": key, "opacity": 0.5}),
        lambda key: Record(
            "image",
            {
                "shape": _BLANK.key,
                "corners": tuple(range(12)),
                "image_opacity": 0.5,
            },
        ),
    ],
    ids=[
        "reserved kind",
        "unknown feature bit",
        "placement",
        "blend placement",
        "dash",
        "clip",
        "opacity",
        "image opacity",
    ],
)
def test_a_frame_with_what_cairo_does_not_draw_is_refused_whole(make):
    import arachne_core

    from arachne.frame_ir.cairo_painter import CairoDrawer, Unsupported
    from arachne.frame_ir.xf import Delta, StreamState

    delta = _square_and(make)
    read = arachne_core.FrameDelta.from_bytes(delta.to_core().to_bytes())
    state = StreamState()
    state.apply(Delta.from_contents(read.contents()))
    drawer = CairoDrawer()
    with pytest.raises(Unsupported, match="render.unsupported"):
        drawer.draw(state)
    assert not drawer._arrays, "nothing is drawn"


@pytest.mark.parametrize(
    "camera_class",
    ["Camera", "MovingCamera", "MultiCamera", "ThreeDCamera", "MappingCamera"],
)
def test_extraction_reads_no_mobject_name_the_camera_does_not(camera_class):
    """What extraction reads off a mobject that a class can override, the
    camera's capture reads too: the refused-override names are the camera's
    own, none added.
    """
    import arachne
    from arachne.frame_ir.extract import MOBJECT_READS
    from arachne.timeline import dispatch

    cls = getattr(arachne, camera_class)
    own = dispatch.camera_draw_methods(cls)
    overridable = set(MOBJECT_READS) & set(dispatch._mobject_definitions())
    assert overridable
    assert overridable <= own
    assert dispatch.draw_methods(cls) == own


def test_a_frame_with_what_cairo_does_not_draw_inside_a_sub_frame_is_refused():
    """A record drawn only inside a display's sub-frame is checked too."""
    import arachne_core

    from arachne.frame_ir.cairo_painter import CairoDrawer, Unsupported
    from arachne.frame_ir.xf import Delta, StreamState

    delta = _square_and(lambda key: Record("path", {"shape": key, "opacity": 0.5}))
    display = Record(
        "image",
        {"corners": tuple(range(12)), "sub_frame": (0, delta.view, [("static", 1, 0)])},
    )
    delta.set.append((2, display))
    delta.order = [("static", 0, 0), ("static", 2, 0)]
    read = arachne_core.FrameDelta.from_bytes(delta.to_core().to_bytes())
    state = StreamState()
    state.apply(Delta.from_contents(read.contents()))
    with pytest.raises(Unsupported, match="opacity other than 1"):
        CairoDrawer().draw(state)


def test_drawing_restores_the_shown_states_when_settling_fails():
    from arachne.timeline.draw import Drawing

    session = Session.open(SCENES, "Morph", config={**MEDIUM, "draw_path": "ir_cairo"})
    try:
        record = session.record
        square = session.scene.mobjects[0]
        before = np.array(square.points)
        play = next(p for p in record.plays if p.animations == ("Transform(Square)",))
        capture = play.capture_at(play.start + (play.end - play.start) / 2)

        class Failing:
            def settle(self) -> None:
                raise RuntimeError("settling failed")

        drawing = Drawing(record)
        drawing.show(capture, capture.draw_list())
        assert not np.array_equal(square.points, before)
        drawing._painted_by = Failing()
        with pytest.raises(RuntimeError, match="settling failed"):
            drawing.__exit__(None, None, None)
        np.testing.assert_array_equal(square.points, before)
    finally:
        session.close()


#: Names the read scan finds in extraction's source that are no mobject's:
#: dictionaries' and deltas' methods, and locals that are arrays or fields.
_NOT_MOBJECT_READS = frozenset({"set", "update", "remove", "base", "pixel_array"})


def test_every_mobject_read_in_extraction_is_declared():
    """``extract.MOBJECT_READS`` lists what extraction reads off a mobject:
    the read scan of its source finds nothing else a mobject class defines
    (but a sort point, which only a ThreeDCamera's depth asks for, and its
    own sort reads it already).
    """
    from arachne.camera.three_d_camera import ThreeDCamera
    from arachne.frame_ir import extract
    from arachne.timeline import dispatch

    functions = []
    for value in vars(extract).values():
        if dispatch._engine_function(value) and value.__module__ == extract.__name__:
            functions.append(value)
        elif isinstance(value, type) and value.__module__ == extract.__name__:
            functions.extend(
                f for f in map(dispatch._function_of, vars(value).values()) if f
            )
    reads = set().union(*(dispatch._reads(f)[0] for f in functions))
    found = reads & set(dispatch._mobject_definitions())
    undeclared = found - set(extract.MOBJECT_READS) - _NOT_MOBJECT_READS
    assert undeclared <= {"get_z_index_reference_point"}, undeclared
    assert undeclared <= dispatch.camera_draw_methods(ThreeDCamera)


def test_the_drawing_lint_codes_are_declared():
    from arachne.frame_ir.lint import CODES

    assert set(CODES) == {"render.unsupported", "render.fallback"}


def test_a_camera_that_is_no_engine_camera_is_unable():
    from arachne.frame_ir.extract import FrameStream
    from arachne.frame_ir.painter import camera_path

    class Duck:
        """Draws like a camera, and is none of the engine's."""

    assert camera_path(Duck(), "ir_cairo") == "ir_cairo"
    with pytest.raises(Unable, match="a Duck draws what XF 1 does not describe"):
        FrameStream().frame(Duck(), [])
