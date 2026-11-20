"""Rigs whose parts are stills, drawn on a device, the launch mode's: each
picture stands where the rig's placement and its part's transform put it,
as its still opened by path stands there; a fade of the rig fades its
pictures at every tick; two stills of one size show their own pixels in
either load order and after a slot switch; a still that decodes otherwise
draws nothing; a layered set slides by its depths in the drawn frame and
its focus blurs each layer by its depth; and old scenes keep their bytes
around a scene that drew a rig of stills. Collected only where the GPU
build is installed.
"""

from __future__ import annotations

import json
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache, decode, frame_digest
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from arachne.timeline.footprint import object_map
from tests.assets.helpers import admitted
from tests.helpers.gpu_door import run_child

from .conftest import SMALL, copied, line_of
from .raster_scenes import (
    FLAP,
    PAN,
    depth_of,
    files_of,
    flap_pictures,
    layer_pictures,
)

SCENES = Path(__file__).parent / "raster_scenes.py"
#: The frame height the scenes are configured at: the default viewing
#: distance; and the frame's width.
DISTANCE, WIDTH = 8.0, 8.0 * 16 / 9


def _pixels(session, t: Fraction | float) -> np.ndarray:
    return np.asarray(session.frame(t)).astype(int)


def _close(a: np.ndarray, b: np.ndarray, within: int = 1) -> bool:
    return a.shape == b.shape and int(np.abs(a - b).max()) <= within


def test_a_posed_rig_s_pictures_stand_where_its_placement_puts_them(session_of, door):
    rig = session_of("PosedFlap", SCENES)
    images = session_of("PosedFlapAsImages", SCENES)
    for t in (0, Fraction(1, 4)):
        drawn = _pixels(rig, t)
        assert _close(drawn, _pixels(images, t))
    # Every picture is drawn, the badge's switched drawing included.
    assert len(np.unique(drawn.reshape(-1, 4), axis=0)) > 8
    assert door.draws == 2


def test_a_fade_of_the_rig_fades_its_pictures_at_every_tick(session_of, door):
    session = session_of("FadingFlap", SCENES, frame_rate=60)
    body = np.array([150, 110, 70])
    # The body's middle: the scene point (0, -0.2) on a frame 96 x 54.
    column, row = 48, int(27 + 0.2 * 96 / WIDTH)
    for k in range(31):
        opacity = k / 30
        alpha = int(255 * opacity) / 255
        drawn = _pixels(session, Fraction(k, 60))[row, column, :3]
        assert np.abs(drawn - body * alpha).max() <= 2, k


TWO = """\
import numpy as np

from arachne import *


def image(file, corners):
    picture = ImageMobject(file)
    picture.points = np.array(corners)
    return picture


class Two(Scene):
    def construct(self):
        if {a_first}:
            a, b = Rig({a!r}), Rig({b!r})
        else:
            b, a = Rig({b!r}), Rig({a!r})
        self.add(a.shift(2.5 * LEFT), b.shift(2.5 * RIGHT))
        self.wait(0.5)
        a.show(badge="off")
        b.show(badge="off")
        self.wait(0.5)


class TwoAsImages(Scene):
    def construct(self):
        self.add(*{before})
        self.wait(0.5)
        self.clear()
        self.add(*{after})
        self.wait(0.5)
"""


def _images(directory: Path, shown: str, shift: float) -> str:
    """Source of the flap rig's pictures from ``directory`` as images, at
    rest with ``shown`` on its badge, moved ``shift`` across: the items of a
    list.
    """
    files = files_of(directory)
    made = []
    for name, corners in flap_pictures(0, 1, shown).items():
        at = (corners + [shift, 0, 0]).tolist()
        made.append(f"image({str(directory / files[name])!r}, {at!r})")
    return ", ".join(made)


@pytest.mark.parametrize("a_first", [True, False])
def test_two_stills_of_one_size_show_their_own_pixels_in_either_load_order(
    tmp_path, session_of, door, a_first
):
    other = copied(tmp_path, "hinged-flap", "recoloured")
    on = other / "pieces" / "on.png"
    pixels = decode(on.read_bytes())
    pixels[..., :3] = 255 - pixels[..., :3]
    from PIL import Image

    Image.fromarray(pixels).save(on, format="PNG")
    document = json.loads((other / "manifest.json").read_text())
    document["slots"][0]["drawings"][0]["digest"] = frame_digest(pixels)
    (other / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    admitted(other)
    file = tmp_path / "two.py"
    file.write_text(
        textwrap.dedent(
            TWO.format(
                a=str(FLAP),
                b=str(other),
                a_first=a_first,
                before=f"[{_images(FLAP, 'on', -2.5)}, {_images(other, 'on', 2.5)}]",
                after=f"[{_images(FLAP, 'off', -2.5)}, {_images(other, 'off', 2.5)}]",
            )
        )
    )
    rigs, images = session_of("Two", file), session_of("TwoAsImages", file)
    for t in (Fraction(1, 4), Fraction(3, 4)):
        drawn, expected = _pixels(rigs, t), _pixels(images, t)
        # Within 2: a picture's corners reached by a rig's moves and those
        # written out here differ in their last bits, which moves a
        # resampled edge pixel of the small badge by 2 at most.
        assert _close(drawn, expected, within=2), (t, _where(drawn, expected))


def _where(drawn: np.ndarray, expected: np.ndarray) -> dict:
    """Where two frames differ by more than 1: how much, the box of those
    pixels, and a few of them with both values.
    """
    off = np.abs(drawn - expected).max(axis=2) > 1
    rows, columns = np.nonzero(off)
    if not len(rows):
        return {}
    picked = list(zip(rows[:4].tolist(), columns[:4].tolist(), strict=True))
    return {
        "most": int(np.abs(drawn - expected).max()),
        "box": (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        ),
        "count": int(off.sum()),
        "at": [
            (r, c, drawn[r, c].tolist(), expected[r, c].tolist()) for r, c in picked
        ],
    }


def test_a_still_that_decodes_otherwise_draws_nothing_and_names_its_file(
    tmp_path, monkeypatch, door
):
    monkeypatch.setattr(frames, "cache", FrameCache())
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        if pixels.shape[:2] == (6, 40):  # the flap's still alone
            pixels[2, 2, 0] ^= 8
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    server = Dispatcher()
    picture = tmp_path / "frame.png"
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "PosedFlap", "config": SMALL}
        )
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.25, "path": str(picture)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.inexact"
    assert finding["detail"]["reason"] == "digest"
    assert finding["detail"]["file"] == "pieces/flap.png"
    assert finding["line"] == line_of("posed-flap", SCENES).line
    assert door.draws == 0
    assert not picture.exists()


def _columns(x0: float, x1: float, centre: float) -> tuple[float, float]:
    """The pixel columns the scene's ``x0`` to ``x1`` span on a frame 96
    pixels wide centred on ``centre``.
    """
    scale = 96 / WIDTH
    return (x0 - centre) * scale + 48, (x1 - centre) * scale + 48


def test_a_layered_set_slides_by_its_depths_in_the_drawn_frame(session_of, door):
    session = session_of("LayeredPan", SCENES, frame_rate=60)
    home = layer_pictures()
    names = {"far": "rig[1]", "middle": "rig[2]", "near": "rig[3]"}
    # From the first tick the fade shows the set at.
    for k in range(1, 61):
        t = Fraction(k, 60)
        state = session.state_at(t)
        by_id = {node.id: node for node in state.nodes}
        named = {node.name: node for node in state.nodes if node.name}
        # Each layer's quad, the one drawable its group holds.
        quads = {
            name: next(
                by_id[child]
                for child in named[names[name]].children
                if by_id[child].type == "RasterPart"
            )
            for name in home
        }
        drawn = object_map(session, t, [quad.id for quad in quads.values()])
        moved = PAN * k / 60
        for name, corners in home.items():
            node = quads[name]
            slide = (1 - DISTANCE / (DISTANCE + depth_of(name))) * moved
            x0, x1 = (corners[:, 0] + slide[0]).min(), (corners[:, 0] + slide[0]).max()
            left, right = _columns(x0, x1, moved[0])
            box = drawn.objects[node.id].box
            assert abs(box[0] - left) <= 1, (name, k)
            assert abs(box[2] - right) <= 1, (name, k)
            assert abs(node.screen_box[0] - box[0]) <= 1, (name, k)
            assert abs(node.screen_box[2] - box[2]) <= 1, (name, k)
            if name == "near":
                # Opaque and drawn last: its footprint is its whole box.
                assert np.allclose(node.screen_box, box, atol=1), k


def test_the_focus_blurs_each_layer_by_its_depth(session_of, door):
    layered = session_of("LayeredFocus", SCENES)
    images = session_of("LayeredFocusAsImages", SCENES)
    assert images.route == layered.route == "gpu"
    assert _close(_pixels(layered, 0), _pixels(images, 0))
    assert door.draws == 2


#: Opens each old scene, then draws a scene of a rig of stills on the
#: launch mode's device, then each old scene again, in one fresh
#: interpreter.
PROGRAM = """\
import hashlib, json, sys

from arachne.compose import device
from arachne.session import Session
from tests.helpers.gpu_door import LaunchDoor

door = device.door = LaunchDoor()

directory, media, scenes = sys.argv[1:4]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4, "media_dir": media}


def opened(file, scene):
    session = Session.open(file, scene, config=config)
    try:
        return {
            "route": session.route,
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.25, 0.5, 0.7)
            ],
        }
    finally:
        session.close()


old = f"{directory}/old.py"
fresh = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
raster = opened(scenes, "LayeredPan")
drawn = door.draws
after = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
print(json.dumps({"fresh": fresh, "raster": raster, "drawn": drawn, "after": after}))
"""


def test_old_scenes_draw_the_same_bytes_after_a_scene_drew_a_rig_of_stills(tmp_path):
    from tests.assets.test_old_constructors import _scenes

    directory = _scenes(tmp_path / "scenes")
    printed = run_child(PROGRAM, str(directory), str(tmp_path / "media"), str(SCENES))
    found = json.loads(printed.strip().splitlines()[-1])
    assert found["raster"]["route"] == "gpu"
    assert len(set(found["raster"]["frames"])) > 1
    assert found["drawn"] == 4
    assert found["fresh"] == found["after"]
    assert all(scene["route"] == "cpu" for scene in found["fresh"].values())
