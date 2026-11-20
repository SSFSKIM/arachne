"""Rigs whose parts are stills in a scene, with no GPU: a layered set viewed
through a moving camera slides by its depths at every tick, through a fade
of its own, and an anchor on it is right before the instant's settle;
``view_through`` checks what it is given at every call; a manifest that
would draw a still wrong is refused at the server by member and reason;
and a scene with a raster member is on the ``gpu`` route, refused its
frames on a host with no GPU, while the vector rigs stay CPU frames.
"""

from __future__ import annotations

import math
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Rig
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, SCENE_ERROR, RpcError

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
            assert np.allclose(found, _box(corners + slide), atol=1e-9), (name, k)


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
        assert np.allclose([(x0 + x1) / 2, (y0 + y1) / 2], expected[:2], atol=1e-9), k


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
