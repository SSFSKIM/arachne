"""Rigs whose parts are stills in a scene, with no GPU: a layered set viewed
through a moving camera slides by its depths at every tick, through a fade
of its own, and an anchor on it is right before the instant's settle;
``view_through`` checks what it is given at every call; a manifest that
would draw a still wrong is refused at the server by member and reason;
and a scene with a raster member is on the ``gpu`` route, refused its
frames on a host with no GPU, while the vector rigs stay CPU frames. The
lint reads a raster member as it reads the same still: a text over its
clear part is in contact with nothing, and every part is read alike.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Rig
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, SCENE_ERROR, RpcError
from arachne.session import Session

from .conftest import SMALL, copied, edited
from .raster_scenes import PAN, depth_of, layer_pictures

SCENES = Path(__file__).parent / "raster_scenes.py"

#: The frame height the scenes are configured at: the default viewing
#: distance.
DISTANCE = 8.0


def _box(corners: np.ndarray) -> tuple[float, float, float, float]:
    low, high = corners.min(axis=0), corners.max(axis=0)
    return (low[0], low[1], high[0], high[1])


def _nodes(state: dict) -> dict[str, dict]:
    return {node["name"]: node for node in state["nodes"] if node.get("name")}


def _opened(tmp_path: Path, scene: str, file: Path = SCENES, **settings) -> Dispatcher:
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media"), **settings}
    server.handle("open", {"file": str(file), "scene": scene, "config": config})
    return server


def _refused(tmp_path: Path, file: Path, scene: str) -> RpcError:
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as raised:
        server.handle("open", {"file": str(file), "scene": scene, "config": config})
    return raised.value


def _scene_boxes(session, t: Fraction) -> tuple[Fraction, dict[str, tuple]]:
    """The instant ``state_at(t)`` answers for, and each named node's box
    in scene units then.
    """
    state = session.state_at(t)
    return state.as_of, {n.name: n.scene_box for n in state.nodes if n.name}


def test_a_layered_set_slides_by_its_depths_at_every_tick_while_it_fades_in(
    session_of,
):
    session = session_of("LayeredPan", SCENES, frame_rate=60)
    home = layer_pictures()
    for k in range(61):
        as_of, boxes = _scene_boxes(session, Fraction(k, 60))
        assert as_of == Fraction(k, 60)
        moved = PAN * float(as_of)
        for name, corners in home.items():
            slide = (1 - DISTANCE / (DISTANCE + depth_of(name))) * moved
            found = boxes[f"rig[{_index(name)}]"]
            assert np.allclose(found, _box(corners + slide), rtol=0, atol=1e-9), (
                name,
                k,
            )


def _index(name: str) -> int:
    """Where the layer ``name`` stands among the rig's members (after its
    carrier).
    """
    return ["far", "middle", "near"].index(name) + 1


def test_an_anchor_on_a_layer_is_right_before_the_instant_s_settle(session_of):
    """A dot an updater keeps on the middle layer's anchor stands where the
    layer is at that instant: the anchor reads the camera as it is, not as
    the rig's last settle left it.
    """
    from .raster_scenes import GROUND, canvas_to_scene

    session = session_of("LayeredPan", SCENES, frame_rate=60)
    peak = canvas_to_scene(GROUND, np.array([[120.0, 90.0]]))[0]
    slide = 1 - DISTANCE / (DISTANCE + depth_of("middle"))
    for k in range(1, 61):
        as_of, boxes = _scene_boxes(session, Fraction(k, 60))
        expected = peak + slide * PAN * float(as_of)
        x0, y0, x1, y1 = boxes["dot"]
        assert np.allclose(
            [(x0 + x1) / 2, (y0 + y1) / 2], expected[:2], rtol=0, atol=1e-9
        ), k


VIEWED = """\
from arachne import *


class Viewed(MovingCameraScene):
    def construct(self):
        rig = Rig({asset!r})
        self.add(rig)
        rig.view_through(self.camera, distance={first})
        self.wait(0.25)
        rig.view_through(self.camera, distance={second})  # @second
        self.wait(0.25)
"""


def _viewed(tmp_path: Path, asset: str, first: str, second: str) -> tuple[Path, int]:
    file = tmp_path / "viewed.py"
    file.write_text(
        textwrap.dedent(VIEWED.format(asset=asset, first=first, second=second))
    )
    line = next(
        n
        for n, text in enumerate(file.read_text().splitlines(), start=1)
        if text.endswith("# @second")
    )
    return file, line


@pytest.mark.parametrize(
    ("distance", "says"),
    [
        ("0", "above 0"),
        ("-3", "above 0"),
        ("float('nan')", "above 0"),
        ("float('inf')", "above 0"),
        ("2", "D + d must be above 0"),
    ],
)
@pytest.mark.parametrize("call", ["first", "second"])
def test_view_through_checks_its_distance_against_every_depth_at_each_call(
    tmp_path, distance, says, call
):
    asset = edited(
        copied(tmp_path, "layered-ground"),
        lambda d: d["parts"][2].update(depth=-3),
    )
    from tests.assets.helpers import admitted

    admitted(asset)
    first, second = (distance, "10") if call == "first" else ("10", distance)
    file, line = _viewed(tmp_path, str(asset), first, second)
    error = _refused(tmp_path, file, "Viewed")
    assert error.code == SCENE_ERROR, error.data
    assert error.data["cls"] == "ValueError"
    assert says in str(error.data)
    if call == "second":
        assert error.data["line"] == line


def test_view_through_needs_a_camera_whose_frame_moves(tmp_path):
    file = tmp_path / "flat.py"
    file.write_text(
        textwrap.dedent(
            """\
            from arachne import *


            class Flat(Scene):
                def construct(self):
                    Rig("fixture:layered-ground").view_through(self.camera)
            """
        )
    )
    error = _refused(tmp_path, file, "Flat")
    assert error.data["cls"] == "ValueError"
    assert "MovingCameraScene" in str(error.data)


def test_view_through_needs_a_running_scene():
    from arachne import MovingCamera

    with pytest.raises(RuntimeError, match="inside the scene's construct"):
        Rig("fixture:layered-ground").view_through(MovingCamera())


SHOWN = """\
from arachne import *


class Shown(Scene):
    def construct(self):
        rig = Rig({asset!r})  # @rig
        self.add(rig)
        self.wait(0.25)
"""


def _flap_driver(document: dict, kind: str) -> None:
    drive = document["parameters"][0]["drives"][0]
    drive.pop("unit")
    drive["kind"] = kind


@pytest.mark.parametrize(
    ("asset", "edit", "member", "says"),
    [
        ("hinged-flap", lambda d: d.update(requires=[]), "parts[0].box", "raster"),
        (
            "layered-ground",
            lambda d: d.update(requires=["raster"]),
            "parts[0].depth",
            "depth",
        ),
        (
            "hinged-flap",
            lambda d: _flap_driver(d, "warp"),
            "parameters[0].drives[0].kind",
            "a warp",
        ),
        (
            "hinged-flap",
            lambda d: _flap_driver(d, "shape"),
            "parameters[0].drives[0].kind",
            "a shape",
        ),
        (
            "hinged-flap",
            lambda d: d["parts"][1].update(
                lattice={"cols": 1, "rows": 1, "bounds": [0, 20, 160, 44]}
            ),
            "parts[1].lattice",
            "no lattice",
        ),
    ],
    ids=["no-raster", "no-depth", "warp", "shape", "lattice"],
)
def test_a_manifest_that_would_draw_a_still_wrong_is_refused_at_the_server(
    tmp_path, asset, edit, member, says
):
    directory = edited(copied(tmp_path, asset), edit)
    file = tmp_path / "shown.py"
    file.write_text(textwrap.dedent(SHOWN.format(asset=str(directory))))
    error = _refused(tmp_path, file, "Shown")
    assert error.code == REFUSED, error.data
    finding = error.data["finding"]
    assert finding["code"] == "asset.manifest_invalid"
    assert finding["detail"]["member"] == member
    assert says in finding["detail"]["reason"]
    assert finding["line"] == 6


def test_a_raster_rig_is_on_the_gpu_route_and_the_vector_rigs_on_the_cpu(session_of):
    assert session_of("PosedFlap", SCENES).route == "gpu"
    assert session_of("LayeredFocus", SCENES).route == "gpu"
    assert session_of("VectorRig", SCENES).route == "cpu"


def test_with_no_gpu_a_raster_rig_answers_its_states_and_lint_and_refuses_frames(
    tmp_path, no_gpu
):
    server = _opened(tmp_path, "PosedFlap")
    try:
        state = server.handle("state_at", {"t": 0.25})["state"]
        assert "RasterPart" in {node["kind"] for node in state["nodes"]}
        assert "lint_result" in server.handle("lint", {})
        with pytest.raises(RpcError) as answer:
            server.handle("frame", {"t": 0.25, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert answer.value.code == REFUSED
    assert answer.value.data["reason"] == "render.gpu_required"
    assert not (tmp_path / "f.png").exists()


def test_the_parallax_is_d_over_d_plus_depth_of_the_camera_s_move():
    from arachne.assets.pose import parallax

    moved = parallax(6, 8, np.array([2.0, -1.0, 0]), np.array([0.5, 0.5, 0]))
    assert np.allclose(moved, (1 - 8 / 14) * np.array([1.5, -1.5, 0]))
    assert np.allclose(parallax(0, 8, np.ones(3), np.zeros(3)), 0)
    assert math.isclose(parallax(-4, 8, np.array([1.0, 0, 0]), np.zeros(3))[0], -1.0)
    assert Rig("fixture:layered-ground")._parallax("far").tolist() == [0, 0, 0]


LINTED = """\
import numpy as np

from arachne import *


def image(file, corners):
    picture = ImageMobject(file)
    picture.points = np.array(corners)
    return picture


class Raster(Scene):
    def construct(self):
        label = Text("over", font_size=20).move_to([0, 1.0, 0])
        art = Rig({asset!r}).shift({shift!r})
        {extra}
        self.add({order})
        self.wait(0.5)


class Images(Scene):
    def construct(self):
        label = Text("over", font_size=20).move_to([0, 1.0, 0])
        art = Group(*[image(f, c) for f, c in {pictures!r}]).shift({shift!r})
        {extra}
        self.add({order})
        self.wait(0.5)
"""


def _clear_on_top(tmp_path: Path) -> Path:
    """The layered set with its far layer clear over its upper part: the
    canvas rows 0 to 60, the scene's y from 1.6 down to 0.4.
    """
    from PIL import Image

    from arachne.mobject.media.frames import frame_digest
    from tests.assets.helpers import admitted

    asset = copied(tmp_path, "layered-ground", "clear-on-top")
    pixels = np.zeros((20, 48, 4), dtype=np.uint8)
    pixels[12:] = (110, 130, 190, 255)
    Image.fromarray(pixels).save(asset / "layers" / "far.png", format="PNG")
    edited(asset, lambda d: d["parts"][0].update(digest=frame_digest(pixels)))
    return admitted(asset)


def _linted(
    tmp_path: Path, asset: Path, scene: str, order: str, shift, extra: str = "pass"
) -> Session:
    """A session of ``scene`` of :data:`LINTED` on ``asset``."""
    from .raster_scenes import files_of

    pictures = [
        (str(asset / files_of(asset)[name]), corners.tolist())
        for name, corners in layer_pictures().items()
    ]
    file = tmp_path / f"linted_{scene}.py"
    file.write_text(
        textwrap.dedent(
            LINTED.format(
                asset=str(asset),
                pictures=pictures,
                order=order,
                shift=list(shift),
                extra=extra,
            )
        )
    )
    return Session.open(file, scene, config=SMALL)


def _codes(session: Session) -> set[str]:
    """The codes the sweep finds in ``session``."""
    from arachne.lint import lint
    from arachne.lint.rules import sweep

    return {finding.code for finding in lint(session, rules=(sweep,)).findings}


def _read(session: Session, *, memo: bool) -> list[tuple]:
    """What the lint reads of each drawn thing but the label at 0.25 s: its
    kind, and each leaf as drawn, laid and measured, boxes to 1e-6 pixels.
    """
    from arachne.lint.parts import PartsCache
    from arachne.timeline.routes import _resolve

    record, t = session.record, Fraction(1, 4)
    resolved = _resolve(record, t)
    parts = PartsCache(memo=memo).build(
        record, t, resolved.play, resolved.capture, answer="recorded", target=(96, 54)
    )

    def box(found) -> tuple:
        return tuple(round(float(v), 6) for v in found)

    return [
        (
            part.kind,
            box(part.screen_box),
            box(part.ink_box),
            [
                (
                    leaf.draw,
                    leaf.unit is None,
                    box(leaf.screen_box),
                    box(leaf.ink_box),
                    leaf.fill_opacity,
                    leaf.stroke_opacity,
                    leaf.stroke_width,
                    leaf.first_fill_opacity,
                    leaf.depth,
                )
                for leaf in part.leaves
            ],
        )
        for part in parts.parts
        if part.kind != "text"
    ]


#: The families of the codes a text near or under something draws: drawn
#: over (``occlusion``), in contact, overlapping or too near (``gap``).
CLEARANCE = ("occlusion", "contact", "overlap", "gap")


@pytest.mark.parametrize("order", ["label, art", "art, label"], ids=["under", "over"])
def test_text_over_a_clear_area_of_a_still_or_a_raster_member_is_not_in_contact(
    tmp_path, order
):
    asset = _clear_on_top(tmp_path)
    for scene in ("Raster", "Images"):
        session = _linted(tmp_path, asset, scene, order, (0, 0, 0))
        try:
            codes = _codes(session)
        finally:
            session.close()
        assert not {c for c in codes if c.split(".")[0] in CLEARANCE}, (scene, codes)


@pytest.mark.parametrize("shift", [(0, 0, 0), (6.5, 0, 0), (0, 3.5, 0)])
def test_the_lint_reads_a_raster_member_as_it_reads_the_same_still(tmp_path, shift):
    asset = _clear_on_top(tmp_path)
    read, codes = {}, {}
    for scene in ("Raster", "Images"):
        session = _linted(tmp_path, asset, scene, "art, label", shift)
        try:
            read[scene] = [_read(session, memo=memo) for memo in (False, True)]
            codes[scene] = _codes(session)
        finally:
            session.close()
    fresh, kept = read["Raster"]
    assert fresh == kept
    assert [kind for kind, *_ in fresh] == ["image"] * 3
    assert fresh == read["Images"][0]
    assert codes["Raster"] == codes["Images"]


def test_view_through_needs_the_running_scene_s_own_camera(tmp_path):
    file = tmp_path / "other.py"
    file.write_text(
        textwrap.dedent(
            """\
            from arachne import *


            class Other(MovingCameraScene):
                def construct(self):
                    Rig("fixture:layered-ground").view_through(MovingCamera())
            """
        )
    )
    error = _refused(tmp_path, file, "Other")
    assert error.data["cls"] == "ValueError"
    assert "self.camera" in str(error.data)
    assert "_view_frame" in Rig._content_key_omits


def _clear(tmp_path: Path) -> Path:
    """The layered set with its far layer clear everywhere."""
    from PIL import Image

    from arachne.mobject.media.frames import frame_digest
    from tests.assets.helpers import admitted

    asset = copied(tmp_path, "layered-ground", "clear")
    pixels = np.zeros((20, 48, 4), dtype=np.uint8)
    Image.fromarray(pixels).save(asset / "layers" / "far.png", format="PNG")
    edited(asset, lambda d: d["parts"][0].update(digest=frame_digest(pixels)))
    return admitted(asset)


@pytest.mark.parametrize(
    ("variant", "extra", "left"),
    [
        # Under half a pixel high: drawn by no camera.
        ("on-top", "art.stretch(0.003, 1)", 0),
        # Every pixel transparent, at the rig's full opacity or at none.
        ("clear", "pass", 2),
        (
            "on-top",
            "for m in art.family_members_with_points(): m.set_opacity(0)",
            0,
        ),
    ],
    ids=["thin", "clear", "faded"],
)
def test_the_lint_leaves_out_a_picture_that_paints_nothing_as_it_leaves_the_still(
    tmp_path, variant, extra, left
):
    asset = _clear_on_top(tmp_path) if variant == "on-top" else _clear(tmp_path)
    read = {}
    for scene in ("Raster", "Images"):
        session = _linted(tmp_path, asset, scene, "art, label", (0, 0, 0), extra=extra)
        try:
            read[scene] = [_read(session, memo=memo) for memo in (False, True)]
        finally:
            session.close()
    fresh, kept = read["Raster"]
    assert fresh == kept == read["Images"][0]
    assert len(fresh) == left


STANDING = """\
from arachne import *
from arachne.assets import RasterPart, load_asset


class Standing(Scene):
    def construct(self):
        asset = load_asset("fixture:hinged-flap")
        part = asset.manifest.parts[0]
        quad = RasterPart(
            part.box,
            asset_digest=asset.digest,
            file=part.file,
            digest=part.digest,
            asset=asset.name,
            directory=asset.directory,
        )
        self.add(quad.scale(0.02).move_to(ORIGIN))
        self.wait(0.5)
"""


def test_a_standing_picture_s_place_is_kept_and_read_again_from_the_memo(
    tmp_path, monkeypatch
):
    from arachne.lint import parts as parts_module
    from arachne.lint.leaf_memo import LeafMemo
    from arachne.lint.parts import build_parts
    from arachne.timeline.routes import _resolve

    file = tmp_path / "standing.py"
    file.write_text(STANDING)
    session = Session.open(file, "Standing", config=SMALL)
    placed: list[object] = []
    real = parts_module._Reader._place

    def counted(self, mob, picture=False):
        placed.append(picture)
        return real(self, mob, picture)

    monkeypatch.setattr(parts_module._Reader, "_place", counted)
    memo = LeafMemo()
    try:
        record = session.record
        read = []
        for t in (Fraction(1, 10), Fraction(2, 5)):
            resolved = _resolve(record, t)
            read.append(
                build_parts(
                    record,
                    t,
                    resolved.play,
                    resolved.capture,
                    answer="recorded",
                    target=(96, 54),
                    memo=memo,
                )
            )
    finally:
        session.close()
    (first,), (second,) = (r.parts for r in read)
    assert first.kind == second.kind == "image"
    (one,), (two,) = first.leaves, second.leaves
    for field in fields(one):
        a, b = getattr(one, field.name), getattr(two, field.name)
        assert np.array_equal(a, b) if field.name == "points" else a == b, field
    # Placed once, as a picture: the second read is the memo's.
    assert placed == [True]
